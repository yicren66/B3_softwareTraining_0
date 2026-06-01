#!/usr/bin/env python3
"""
Main training script for JujubeClassifier.

Supports:
- YAML config loading via argparse
- JujubeDataset with augmentations
- Cosine annealing warm restarts scheduler
- AdamW optimizer with weight decay
- Gradient clipping (max_norm=1.0)
- Early stopping on val_f1_macro
- TensorBoard logging
- Checkpoint saving (top-k)
- Mixed precision training (torch.cuda.amp)
- Multi-GPU via DistributedDataParallel (DDP)
- Per-class metric computation
- Final test-set evaluation
- ONNX export

Usage:
    # Single GPU
    python train.py --config config.yaml

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 train.py --config config.yaml --distributed
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import yaml

# Project imports
from datasets.jujube_dataset import JujubeDataset, split_dataset
from datasets.augmentations import (
    get_train_transforms,
    get_val_transforms,
    get_small_target_transforms,
)
from losses.focal_loss import MultiClassFocalLoss, build_focal_loss

try:
    from models.jujube_classifier import JujubeClassifier
except ImportError:
    JujubeClassifier = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility: metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int,
) -> Dict[str, float]:
    """
    Compute macro-averaged metrics from targets and predictions.

    Args:
        targets: (N,) ground-truth class indices.
        predictions: (N,) predicted class indices.
        num_classes: Total number of classes.

    Returns:
        Dict with keys: accuracy, precision_macro, recall_macro, f1_macro.
    """
    t = targets.cpu().numpy()
    p = predictions.cpu().numpy()

    # Per-class counts
    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)

    for c in range(num_classes):
        tp[c] = np.sum((t == c) & (p == c))
        fp[c] = np.sum((t != c) & (p == c))
        fn[c] = np.sum((t == c) & (p != c))

    # Precision, recall, F1 per class
    eps = 1e-8
    prec_per_class = tp / (tp + fp + eps)
    rec_per_class = tp / (tp + fn + eps)
    f1_per_class = 2 * prec_per_class * rec_per_class / (prec_per_class + rec_per_class + eps)

    accuracy = np.mean(t == p)

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(np.mean(prec_per_class)),
        "recall_macro": float(np.mean(rec_per_class)),
        "f1_macro": float(np.mean(f1_per_class)),
    }


def compute_per_class_metrics(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int,
) -> Dict[str, Any]:
    """
    Compute per-class precision, recall, F1, and support.

    Returns:
        Dict with keys 'per_class' (list of dicts) and 'macro' averages.
    """
    t = targets.cpu().numpy()
    p = predictions.cpu().numpy()
    eps = 1e-8

    per_class: List[Dict] = []
    for c in range(num_classes):
        tp_c = int(np.sum((t == c) & (p == c)))
        fp_c = int(np.sum((t != c) & (p == c)))
        fn_c = int(np.sum((t == c) & (p != c)))
        support_c = int(np.sum(t == c))

        prec = tp_c / (tp_c + fp_c + eps)
        rec = tp_c / (tp_c + fn_c + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)

        per_class.append({
            "class_id": c,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": support_c,
        })

    macro_prec = float(np.mean([x["precision"] for x in per_class]))
    macro_rec = float(np.mean([x["recall"] for x in per_class]))
    macro_f1 = float(np.mean([x["f1"] for x in per_class]))

    return {
        "per_class": per_class,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
    }


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Early stopping tracker."""

    def __init__(self, patience: int = 15, mode: str = "max", min_delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def step(self, score: float) -> bool:
        """
        Returns True if this score is a new best (should checkpoint).
        """
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
            return True

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Keeps the top-k checkpoints based on a monitored metric."""

    def __init__(
        self,
        save_dir: Path,
        save_top_k: int = 3,
        mode: str = "max",
    ):
        self.save_dir = Path(save_dir)
        self.save_top_k = save_top_k
        self.mode = mode
        self.best_scores: List[Tuple[float, Path]] = []
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def update(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        score: float,
        scaler: Optional[GradScaler] = None,
        extra: Optional[Dict] = None,
    ) -> Optional[Path]:
        """
        Save checkpoint if it is among the top-k.

        Returns the path of the saved checkpoint, or None if not saved.
        """
        path = self.save_dir / f"epoch_{epoch:03d}_score_{score:.4f}.pt"

        self.best_scores.append((score, path))
        self.best_scores.sort(
            key=lambda x: x[0],
            reverse=(self.mode == "max"),
        )

        # Prune excess
        while len(self.best_scores) > self.save_top_k:
            _, stale_path = self.best_scores.pop()
            if stale_path.exists():
                stale_path.unlink()

        # Only save if this checkpoint is still in the top-k
        if path in {p for _, p in self.best_scores}:
            state = {
                "epoch": epoch,
                "model_state_dict": (
                    model.module.state_dict()
                    if isinstance(model, DDP)
                    else model.state_dict()
                ),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "score": score,
            }
            if scaler is not None:
                state["scaler_state_dict"] = scaler.state_dict()
            if extra:
                state.update(extra)
            torch.save(state, path)
            return path
        return None

    def load_best(self, model: nn.Module) -> Optional[Path]:
        """Load the best checkpoint into the model. Returns its path or None."""
        if not self.best_scores:
            return None
        best_path = self.best_scores[0][1]
        if best_path.exists():
            ckpt = torch.load(best_path, map_location="cpu")
            if isinstance(model, DDP):
                model.module.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt["model_state_dict"])
            return best_path
        return None


# ---------------------------------------------------------------------------
# Epoch loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    max_grad_norm: float = 1.0,
    is_distributed: bool = False,
    world_size: int = 1,
) -> Dict[str, float]:
    """Run a single training epoch. Returns average loss and metrics."""
    model.train()
    total_loss = 0.0
    all_targets = []
    all_preds = []
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["class_label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            # Handle models that return dicts
            output = model(images)
            if isinstance(output, dict):
                logits = output["logits"]
            else:
                logits = output

            loss = criterion(logits, targets)

        scaler.scale(loss).backward()

        # Unscale before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step(epoch + batch_idx / num_batches)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_targets.append(targets.detach())
        all_preds.append(preds.detach())

        # Log batch-level loss
        if writer and batch_idx % 20 == 0:
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar("train/batch_loss", loss.item(), global_step)
            writer.add_scalar(
                "train/lr",
                scheduler.get_last_lr()[0],
                global_step,
            )

    avg_loss = total_loss / num_batches
    metrics = compute_metrics(
        torch.cat(all_targets), torch.cat(all_preds),
        num_classes=getattr(criterion, "num_classes", 15) or 15,
    )

    if writer:
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/accuracy", metrics["accuracy"], epoch)
        writer.add_scalar("train/f1_macro", metrics["f1_macro"], epoch)

    # Reduce across processes in DDP
    if is_distributed and world_size > 1:
        metrics_tensor = torch.tensor(
            [avg_loss, metrics["accuracy"], metrics["precision_macro"],
             metrics["recall_macro"], metrics["f1_macro"]],
            device=device,
        )
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)
        avg_loss, accuracy, prec, rec, f1 = metrics_tensor.tolist()
        metrics = {
            "accuracy": accuracy, "precision_macro": prec,
            "recall_macro": rec, "f1_macro": f1,
        }

    return {"loss": avg_loss, **metrics}


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    is_distributed: bool = False,
    world_size: int = 1,
) -> Dict[str, float]:
    """Run validation. Returns average loss and metrics."""
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []
    all_logits = []

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["class_label"].to(device, non_blocking=True)

        output = model(images)
        if isinstance(output, dict):
            logits = output["logits"]
        else:
            logits = output

        loss = criterion(logits, targets)
        total_loss += loss.item()

        all_targets.append(targets)
        all_preds.append(logits.argmax(dim=1))
        all_logits.append(logits)

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(
        torch.cat(all_targets), torch.cat(all_preds),
        num_classes=getattr(criterion, "num_classes", 15) or 15,
    )

    if writer:
        writer.add_scalar("val/loss", avg_loss, epoch)
        writer.add_scalar("val/accuracy", metrics["accuracy"], epoch)
        writer.add_scalar("val/f1_macro", metrics["f1_macro"], epoch)

    if is_distributed and world_size > 1:
        metrics_tensor = torch.tensor(
            [avg_loss, metrics["accuracy"], metrics["precision_macro"],
             metrics["recall_macro"], metrics["f1_macro"]],
            device=device,
        )
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)
        avg_loss, accuracy, prec, rec, f1 = metrics_tensor.tolist()
        metrics = {
            "accuracy": accuracy, "precision_macro": prec,
            "recall_macro": rec, "f1_macro": f1,
        }

    return {"loss": avg_loss, **metrics}


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

@torch.no_grad()
def export_onnx(
    model: nn.Module,
    output_path: Path,
    input_shape: Tuple[int, int, int, int] = (1, 3, 380, 380),
    opset_version: int = 14,
    dynamic_batch: bool = True,
) -> None:
    """Export a trained model to ONNX format."""
    model.eval()
    device = next(model.parameters()).device

    dummy = torch.randn(*input_shape, device=device)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )
    logger.info(f"ONNX model exported to {output_path}")


# ---------------------------------------------------------------------------
# DDP setup / cleanup
# ---------------------------------------------------------------------------

def setup_ddp(rank: int, world_size: int, backend: str = "nccl") -> None:
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def build_model(cfg: Dict) -> nn.Module:
    """Instantiate the JujubeClassifier or a fallback."""
    model_cfg = cfg.get("model", {})
    backbone = model_cfg.get("backbone", "efficientnet_b4")
    pretrained = model_cfg.get("pretrained", True)
    num_classes = model_cfg.get("num_classes", 15)

    if JujubeClassifier is not None:
        model = JujubeClassifier(
            backbone_name=backbone,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        logger.info(f"Created JujubeClassifier (backbone={backbone}, classes={num_classes})")
    else:
        # Fallback: use timm or torchvision directly
        logger.warning("JujubeClassifier not found; using timm fallback.")
        import timm
        model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes,
        )

    return model


def build_criterion(cfg: Dict, dataset: JujubeDataset) -> nn.Module:
    """Build loss function from config."""
    loss_cfg = cfg.get("loss", {})
    loss_type = loss_cfg.get("type", "focal")

    if loss_type == "focal":
        class_counts = torch.tensor(
            list(dataset.class_distribution.values()), dtype=torch.float32
        )
        return build_focal_loss(
            num_classes=dataset.num_classes,
            class_counts=class_counts,
            gamma=loss_cfg.get("focal_gamma", 2.0),
            alpha=loss_cfg.get("focal_alpha", 0.25),
            reduction="mean",
            label_smoothing=loss_cfg.get("label_smoothing", 0.0),
        )
    elif loss_type == "ce":
        class_weights = dataset.get_class_weights(norm="balanced")
        return nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def build_scheduler(
    cfg: Dict,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
) -> CosineAnnealingWarmRestarts:
    """Build cosine annealing with warm restarts."""
    train_cfg = cfg.get("training", {})
    epochs = train_cfg.get("epochs", 80)
    t_0 = train_cfg.get("cosine_t0", 10)  # epochs per restart cycle

    return CosineAnnealingWarmRestarts(
        optimizer,
        T_0=t_0 * steps_per_epoch,
        T_mult=train_cfg.get("cosine_t_mult", 1),
        eta_min=train_cfg.get("min_lr", 1e-6),
    )


def train(cfg: Dict, rank: int = 0, world_size: int = 1) -> None:
    """
    Run the full training pipeline.

    Args:
        cfg: Configuration dictionary (loaded from YAML).
        rank: Process rank for DDP (0 for single-GPU).
        world_size: Number of processes for DDP.
    """
    # ------------------------------------------------------------------
    # DDP setup
    # ------------------------------------------------------------------
    is_distributed = world_size > 1
    if is_distributed:
        setup_ddp(rank, world_size)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main_process = rank == 0

    # ------------------------------------------------------------------
    # Config extraction
    # ------------------------------------------------------------------
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    aug_cfg = cfg.get("augmentation", {})
    es_cfg = cfg.get("early_stopping", {})
    ckpt_cfg = cfg.get("checkpoint", {})
    data_cfg = cfg.get("data", {})

    image_size = tuple(model_cfg.get("image_size", [3, 380, 380]))
    h, w = image_size[1], image_size[2]  # channels, height, width
    num_classes = model_cfg.get("num_classes", 15)
    batch_size = train_cfg.get("batch_size", 32)
    epochs = train_cfg.get("epochs", 80)
    lr = train_cfg.get("lr", 0.001)
    weight_decay = train_cfg.get("weight_decay", 0.0001)
    num_workers = train_cfg.get("num_workers", 4)

    output_dir = Path(train_cfg.get("output_dir", "./outputs"))
    log_dir = output_dir / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = output_dir / "checkpoints"

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------
    small_target_classes = aug_cfg.get("small_target_classes", [])

    train_transform = get_train_transforms(
        image_size=(h, w),
        crop_scale=tuple(aug_cfg.get("crop_scale", [0.6, 1.0])),
        rotation_limit=aug_cfg.get("rotation_limit", 45),
        brightness_limit=aug_cfg.get("brightness_limit", 0.3),
        contrast_limit=aug_cfg.get("contrast_limit", 0.3),
    )

    val_transform = get_val_transforms(image_size=(h, w))

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    annotation_path = data_cfg.get("annotation_path", "annotations.csv")
    image_dir = data_cfg.get("image_dir", "")

    train_ds, val_ds, test_ds = split_dataset(
        annotations=annotation_path,
        image_dir=image_dir,
        train_ratio=data_cfg.get("train_ratio", 0.7),
        val_ratio=data_cfg.get("val_ratio", 0.15),
        test_ratio=data_cfg.get("test_ratio", 0.15),
        stratify=True,
        transform_train=train_transform,
        transform_val=val_transform,
        transform_test=val_transform,
        target_size=(h, w),
    )

    if is_main_process:
        logger.info(f"Train samples: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
        logger.info(f"Class distribution: {train_ds.class_distribution}")

    # Optionally apply small-target transforms to specific classes
    if small_target_classes:
        small_transform = get_small_target_transforms(image_size=(h, w))
        # Apply by overriding transform for samples of those classes
        for i, sample in enumerate(train_ds._samples):
            if sample["class_id"] in small_target_classes:
                # Store flag for custom collation
                sample["_use_small_transform"] = True
        # Note: a custom collate_fn would be needed for full per-sample transform
        # switching. For now we log the advice.
        logger.info(
            f"Small-target classes {small_target_classes} flagged. "
            "Consider oversampling or custom DataLoader for best results."
        )

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    sampler_kwargs: Dict[str, Any] = {}
    if is_distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
        sampler_kwargs = {"sampler": train_sampler}
        val_sampler_kwargs = {"sampler": val_sampler}
    else:
        train_sampler = None
        val_sampler = None
        sampler_kwargs = {"shuffle": True}
        val_sampler_kwargs = {"shuffle": False}

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        **sampler_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        **val_sampler_kwargs,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(cfg)
    model = model.to(device)

    if is_distributed:
        model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False,
        )

    # ------------------------------------------------------------------
    # Loss, Optimizer, Scheduler
    # ------------------------------------------------------------------
    criterion = build_criterion(cfg, train_ds)
    criterion = criterion.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch)

    # ------------------------------------------------------------------
    # Mixed precision, early stopping, checkpointing
    # ------------------------------------------------------------------
    scaler = GradScaler()
    early_stopping = EarlyStopping(
        patience=es_cfg.get("patience", 15),
        mode="max",
        min_delta=es_cfg.get("min_delta", 0.0),
    )
    checkpoint_manager = CheckpointManager(
        save_dir=checkpoint_dir,
        save_top_k=ckpt_cfg.get("save_top_k", 3),
        mode="max",
    )

    # ------------------------------------------------------------------
    # TensorBoard
    # ------------------------------------------------------------------
    writer: Optional[SummaryWriter] = None
    if is_main_process and train_cfg.get("tensorboard", True):
        writer = SummaryWriter(log_dir=str(log_dir))
        logger.info(f"TensorBoard logging to {log_dir}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_f1 = 0.0
    start_epoch = 0

    for epoch in range(start_epoch, epochs):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # -- Train -------------------------------------------------------
        t0 = time.time()
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            writer=writer,
            max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
            is_distributed=is_distributed,
            world_size=world_size,
        )
        train_time = time.time() - t0

        # -- Validate ----------------------------------------------------
        val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            writer=writer,
            is_distributed=is_distributed,
            world_size=world_size,
        )

        # -- Log ---------------------------------------------------------
        if is_main_process:
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"T-loss: {train_metrics['loss']:.4f} | "
                f"T-f1: {train_metrics['f1_macro']:.4f} | "
                f"V-loss: {val_metrics['loss']:.4f} | "
                f"V-f1: {val_metrics['f1_macro']:.4f} | "
                f"Time: {train_time:.1f}s"
            )

            if writer:
                writer.add_scalar("epoch/train_time", train_time, epoch)

            # -- Checkpoint ----------------------------------------------
            monitor_key = es_cfg.get("metric", "f1_macro")
            monitor_score = val_metrics.get(monitor_key, val_metrics["f1_macro"])

            is_new_best = early_stopping.step(monitor_score)
            if is_new_best:
                best_val_f1 = val_metrics["f1_macro"]
                logger.info(f"  >> New best {monitor_key}: {monitor_score:.4f}")

            checkpoint_manager.update(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                score=monitor_score,
                scaler=scaler,
            )

            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    # ------------------------------------------------------------------
    # Post-training: load best model
    # ------------------------------------------------------------------
    best_path = checkpoint_manager.load_best(model)
    if best_path and is_main_process:
        logger.info(f"Loaded best checkpoint from {best_path}")

    # ------------------------------------------------------------------
    # Final evaluation on test set
    # ------------------------------------------------------------------
    if is_main_process:
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
        )

        test_metrics = validate_one_epoch(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            epoch=epochs,  # use sentinel epoch for logging
            writer=writer,
        )
        logger.info(
            f"Test set -- Loss: {test_metrics['loss']:.4f}, "
            f"Acc: {test_metrics['accuracy']:.4f}, "
            f"F1: {test_metrics['f1_macro']:.4f}"
        )

        # Per-class metrics
        all_targets = []
        all_preds = []
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                targets = batch["class_label"]
                output = model(images)
                logits = output["logits"] if isinstance(output, dict) else output
                all_targets.append(targets)
                all_preds.append(logits.argmax(dim=1).cpu())

        per_class = compute_per_class_metrics(
            torch.cat(all_targets), torch.cat(all_preds), num_classes=num_classes,
        )

        # Save results
        results_dir = Path(output_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "test_metrics.json", "w") as f:
            json.dump({
                "overall": {k: v for k, v in test_metrics.items()},
                "per_class": per_class["per_class"],
                "macro": {
                    "precision": per_class["macro_precision"],
                    "recall": per_class["macro_recall"],
                    "f1": per_class["macro_f1"],
                },
            }, f, indent=2)
        logger.info(f"Test metrics saved to {results_dir / 'test_metrics.json'}")

        # ------------------------------------------------------------------
        # ONNX export
        # ------------------------------------------------------------------
        if train_cfg.get("export_onnx", True):
            onnx_path = output_dir / "jujube_classifier.onnx"
            # Use the underlying model (unwrap DDP)
            model_for_export = model.module if isinstance(model, DDP) else model
            input_shape = (1, image_size[0], image_size[1], image_size[2])
            export_onnx(model_for_export, onnx_path, input_shape=input_shape)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    if writer:
        writer.close()
    if is_distributed:
        cleanup_ddp()


# ---------------------------------------------------------------------------
# Argparse CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train JujubeClassifier for jujube fruit classification."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="Enable DistributedDataParallel (use with torchrun).",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank for DDP (set by torchrun).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config keys, e.g. training.epochs=100 training.lr=0.0005",
    )
    return parser.parse_args()


def load_config(config_path: str, overrides: List[str]) -> Dict:
    """Load YAML config and apply CLI overrides."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for override in overrides:
        if "=" not in override:
            continue
        key_path, value_str = override.split("=", 1)
        # Attempt type inference
        try:
            value = eval(value_str)
        except Exception:
            value = value_str

        # Navigate nested keys
        keys = key_path.split(".")
        target = cfg
        for k in keys[:-1]:
            target = target.setdefault(k, {})
        target[keys[-1]] = value
        logger.info(f"Override: {key_path} = {value}")

    return cfg


def main() -> None:
    args = parse_args()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Config
    cfg = load_config(args.config, args.override)

    # DDP world size
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank)))

    if args.distributed:
        world_size = max(world_size, 1)

    if world_size > 1:
        # torchrun / multi-GPU: spawn or direct
        train(cfg, rank=local_rank, world_size=world_size)
    else:
        train(cfg, rank=0, world_size=1)


if __name__ == "__main__":
    main()
