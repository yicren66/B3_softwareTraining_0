#!/usr/bin/env python3
"""
Evaluation script for JujubeClassifier.

Computes comprehensive metrics, generates visualisations, and exports a
detailed JSON evaluation report.

Usage:
    python evaluate.py \
        --checkpoint outputs/checkpoints/epoch_050_score_0.9234.pt \
        --config config.yaml \
        --output-dir outputs/evaluation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

import yaml

from datasets.jujube_dataset import JujubeDataset
from datasets.augmentations import get_val_transforms

try:
    from models.jujube_classifier import JujubeClassifier
except ImportError:
    JujubeClassifier = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Model evaluator that computes metrics, generates plots, and exports
    a JSON evaluation report.

    Args:
        model: Trained PyTorch model.
        dataset: JujubeDataset for evaluation.
        device: torch device.
        num_classes: Number of classes.
        class_names: Optional list of class names.
        output_dir: Directory for saving output files.
        batch_size: Batch size for DataLoader.
        num_workers: DataLoader workers.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: JujubeDataset,
        device: torch.device,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        output_dir: Path = Path("./evaluation"),
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Cached results
        self._targets: Optional[np.ndarray] = None
        self._predictions: Optional[np.ndarray] = None
        self._probabilities: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_inference(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run model on the full dataset. Returns (targets, predictions, probs)."""
        if self._targets is not None:
            return self._targets, self._predictions, self._probabilities

        self.model.eval()
        all_targets: List[torch.Tensor] = []
        all_preds: List[torch.Tensor] = []
        all_probs: List[torch.Tensor] = []

        for batch in self.loader:
            images = batch["image"].to(self.device, non_blocking=True)
            targets = batch["class_label"]

            output = self.model(images)
            if isinstance(output, dict):
                logits = output["logits"]
            else:
                logits = output

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_targets.append(targets)
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

        self._targets = torch.cat(all_targets).numpy()
        self._predictions = torch.cat(all_preds).numpy()
        self._probabilities = torch.cat(all_probs).numpy()

        return self._targets, self._predictions, self._probabilities

    # ------------------------------------------------------------------
    # Overall metrics
    # ------------------------------------------------------------------

    def compute_overall_metrics(self) -> Dict[str, float]:
        """Compute macro-averaged metrics."""
        t, p, _ = self._run_inference()

        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            t, p, average="macro", zero_division=0
        )
        accuracy = accuracy_score(t, p)

        return {
            "accuracy": float(accuracy),
            "precision_macro": float(prec_macro),
            "recall_macro": float(rec_macro),
            "f1_macro": float(f1_macro),
        }

    # ------------------------------------------------------------------
    # Per-class metrics
    # ------------------------------------------------------------------

    def compute_per_class_metrics(self) -> Dict[str, Any]:
        """Compute per-class precision, recall, F1, and support."""
        t, p, _ = self._run_inference()

        prec, rec, f1, support = precision_recall_fscore_support(
            t, p, labels=list(range(self.num_classes)), zero_division=0,
        )

        per_class: List[Dict] = []
        for c in range(self.num_classes):
            per_class.append({
                "class_id": c,
                "class_name": self.class_names[c] if c < len(self.class_names) else f"class_{c}",
                "accuracy": float(np.mean((t == c) == (p == c))) if support[c] > 0 else 0.0,
                "precision": float(prec[c]),
                "recall": float(rec[c]),
                "f1": float(f1[c]),
                "support": int(support[c]),
            })

        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            t, p, average="macro", zero_division=0,
        )

        return {
            "per_class": per_class,
            "macro_precision": float(prec_macro),
            "macro_recall": float(rec_macro),
            "macro_f1": float(f1_macro),
        }

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        normalize: bool = True,
        figsize: Tuple[int, int] = (14, 12),
        save: bool = True,
    ) -> np.ndarray:
        """
        Generate and optionally save a confusion matrix plot.

        Args:
            normalize: If True, normalise rows to sum to 1.
            figsize: Figure size in inches.
            save: Whether to save the figure to disk.

        Returns:
            Confusion matrix as numpy array.
        """
        t, p, _ = self._run_inference()
        cm = confusion_matrix(t, p, labels=list(range(self.num_classes)))

        if normalize:
            cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1)
        else:
            cm_display = cm

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set(
            xticks=np.arange(self.num_classes),
            yticks=np.arange(self.num_classes),
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ylabel="True label",
            xlabel="Predicted label",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Annotate cells
        threshold = cm_display.max() / 2.0
        for i in range(cm_display.shape[0]):
            for j in range(cm_display.shape[1]):
                val = cm_display[i, j]
                if normalize:
                    text = f"{val:.2f}"
                else:
                    text = f"{cm[i, j]}"
                ax.text(
                    j, i, text,
                    ha="center", va="center",
                    color="white" if val > threshold else "black",
                    fontsize=6,
                )

        ax.set_title("Confusion Matrix")
        fig.tight_layout()

        if save:
            path = self.output_dir / "confusion_matrix.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {path}")
            plt.close(fig)

        return cm

    # ------------------------------------------------------------------
    # ROC curves (top-5 classes)
    # ------------------------------------------------------------------

    def plot_roc_curves(
        self,
        top_k: int = 5,
        figsize: Tuple[int, int] = (10, 8),
        save: bool = True,
    ) -> Dict[int, float]:
        """
        Plot ROC curves for the top-k classes by support (One-vs-Rest).

        Args:
            top_k: Number of top classes to include.
            figsize: Figure size.
            save: Whether to save.

        Returns:
            {class_id: auc} mapping for plotted classes.
        """
        t, p, probs = self._run_inference()

        # Select top-k classes by number of samples
        class_support = [(c, int((t == c).sum())) for c in range(self.num_classes)]
        class_support.sort(key=lambda x: x[1], reverse=True)
        top_classes = [c for c, _ in class_support[:top_k]]

        fig, ax = plt.subplots(figsize=figsize)
        aucs: Dict[int, float] = {}

        for class_id in top_classes:
            y_true = (t == class_id).astype(int)
            y_score = probs[:, class_id]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            try:
                auc = roc_auc_score(y_true, y_score)
            except ValueError:
                auc = float("nan")
            aucs[class_id] = auc
            ax.plot(fpr, tpr, label=f"{self.class_names[class_id]} (AUC={auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves (Top-{top_k} Classes, One-vs-Rest)")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()

        if save:
            path = self.output_dir / "roc_curves.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"ROC curves saved to {path}")
            plt.close(fig)

        return aucs

    # ------------------------------------------------------------------
    # Small target miss rate
    # ------------------------------------------------------------------

    def compute_small_target_miss_rate(
        self,
        area_threshold: float = 32 * 32,
    ) -> Dict[str, Any]:
        """
        Compute miss rate for small targets.

        A "small target" is defined as a sample whose bbox area is below
        the given threshold. Miss rate = fraction of small targets that
        were incorrectly classified.

        Args:
            area_threshold: Bounding-box area in pixels below which a
                target is considered small.

        Returns:
            Dict with overall and per-class miss rates.
        """
        small_indices = self.dataset.identify_small_targets(area_threshold)
        if not small_indices:
            return {"overall_miss_rate": 0.0, "per_class_miss_rate": {}, "num_small_targets": 0}

        t, p, _ = self._run_inference()

        missed = 0
        per_class_missed: Dict[int, int] = defaultdict(int)
        per_class_total: Dict[int, int] = defaultdict(int)

        for idx in small_indices:
            true_cls = t[idx]
            pred_cls = p[idx]
            per_class_total[true_cls] += 1
            if true_cls != pred_cls:
                missed += 1
                per_class_missed[true_cls] += 1

        overall_miss_rate = missed / len(small_indices)

        per_class_rates = {}
        for cls_id in per_class_total:
            per_class_rates[self.class_names[cls_id]] = (
                per_class_missed.get(cls_id, 0) / per_class_total[cls_id]
            )

        return {
            "overall_miss_rate": overall_miss_rate,
            "num_small_targets": len(small_indices),
            "num_missed": missed,
            "per_class_miss_rate": per_class_rates,
        }

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def generate_report(self) -> Dict[str, Any]:
        """Generate full evaluation report as a dict, and save plots."""
        overall = self.compute_overall_metrics()
        per_class = self.compute_per_class_metrics()
        cm = self.plot_confusion_matrix(normalize=True)
        roc_aucs = self.plot_roc_curves(top_k=5)
        small_target = self.compute_small_target_miss_rate()

        report = {
            "overall": overall,
            "per_class": per_class["per_class"],
            "macro": {
                "precision": per_class["macro_precision"],
                "recall": per_class["macro_recall"],
                "f1": per_class["macro_f1"],
            },
            "confusion_matrix": cm.tolist(),
            "roc_auc_top5": {self.class_names[k]: v for k, v in roc_aucs.items()},
            "small_target_miss_rate": small_target,
        }

        # Save JSON
        report_path = self.output_dir / "evaluation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Evaluation report saved to {report_path}")

        return report


# ---------------------------------------------------------------------------
# Standalone convenience function
# ---------------------------------------------------------------------------

def evaluate_model(
    checkpoint_path: str,
    config_path: str,
    output_dir: str = "./evaluation",
) -> Dict[str, Any]:
    """
    High-level function: load model, dataset, and run evaluation.

    Args:
        checkpoint_path: Path to the model checkpoint (.pt).
        config_path: Path to the training YAML config.
        output_dir: Directory for output files.

    Returns:
        Evaluation report dict.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    image_size = tuple(model_cfg.get("image_size", [3, 380, 380]))
    h, w = image_size[1], image_size[2]
    num_classes = model_cfg.get("num_classes", 15)

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if JujubeClassifier is not None:
        model = JujubeClassifier(
            backbone_name=model_cfg.get("backbone", "efficientnet_b4"),
            pretrained=False,
            num_classes=num_classes,
        )
    else:
        import timm
        model = timm.create_model(
            model_cfg.get("backbone", "efficientnet_b4"),
            pretrained=False,
            num_classes=num_classes,
        )

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt, strict=False)

    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    # Build dataset (test split only)
    val_transform = get_val_transforms(image_size=(h, w))
    _, _, test_ds = _build_datasets(cfg, val_transform, (h, w))

    # Evaluate
    evaluator = Evaluator(
        model=model,
        dataset=test_ds,
        device=device,
        num_classes=num_classes,
        output_dir=Path(output_dir),
    )

    return evaluator.generate_report()


def _build_datasets(
    cfg: Dict,
    val_transform: Any,
    target_size: Tuple[int, int],
) -> Tuple[JujubeDataset, JujubeDataset, JujubeDataset]:
    """Helper: build train/val/test splits from config."""
    from datasets.jujube_dataset import split_dataset

    data_cfg = cfg.get("data", {})
    return split_dataset(
        annotations=data_cfg.get("annotation_path", "annotations.csv"),
        image_dir=data_cfg.get("image_dir", ""),
        train_ratio=data_cfg.get("train_ratio", 0.7),
        val_ratio=data_cfg.get("val_ratio", 0.15),
        test_ratio=data_cfg.get("test_ratio", 0.15),
        stratify=True,
        transform_train=val_transform,
        transform_val=val_transform,
        transform_test=val_transform,
        target_size=target_size,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained JujubeClassifier."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation",
        help="Directory for evaluation outputs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation DataLoader.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    report = evaluate_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    overall = report["overall"]
    print(f"  Accuracy:        {overall['accuracy']:.4f}")
    print(f"  Precision (macro): {overall['precision_macro']:.4f}")
    print(f"  Recall (macro):    {overall['recall_macro']:.4f}")
    print(f"  F1 (macro):        {overall['f1_macro']:.4f}")
    print(f"\n  Small target miss rate: {report['small_target_miss_rate']['overall_miss_rate']:.4f}")
    print(f"  Report saved to: {args.output_dir}/evaluation_report.json")


if __name__ == "__main__":
    main()
