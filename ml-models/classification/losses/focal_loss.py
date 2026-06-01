"""
Focal Loss for multi-class classification.

Implements the focal loss from "Focal Loss for Dense Object Detection" (Lin et al.,
ICCV 2017), adapted for multi-class problems with optional per-class alpha weighting.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassFocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)

    where pt is the model's estimated probability for the ground-truth class.

    Args:
        gamma: Focusing parameter. Higher gamma down-weights easy examples more.
            Default 2.0 (as per the original paper).
        alpha: Per-class weighting factors. Can be:
            - None: no class weighting
            - float: single scalar applied uniformly
            - Tensor / list of shape (num_classes,): per-class weights
        reduction: 'mean', 'sum', or 'none'.
        label_smoothing: Optional label smoothing factor in [0, 1].
        ignore_index: Target value to ignore (useful for padding/unlabeled).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            if isinstance(alpha, torch.Tensor):
                self.register_buffer("alpha", alpha)
            else:
                # scalar
                self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    @property
    def num_classes(self) -> Optional[int]:
        if self.alpha is not None and self.alpha.ndim > 0:
            return self.alpha.shape[0]
        return None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Logits of shape (N, C) where C is the number of classes.
            targets: Ground-truth class indices of shape (N,).

        Returns:
            Scalar loss if reduction is 'mean' or 'sum', else per-sample loss
            of shape (N,).
        """
        # -- Validate ---------------------------------------------------
        if inputs.ndim != 2:
            raise ValueError(
                f"inputs must be 2D (N, C), got shape {inputs.shape}"
            )
        if targets.ndim != 1 or targets.shape[0] != inputs.shape[0]:
            raise ValueError(
                f"targets shape {targets.shape} incompatible with inputs shape {inputs.shape}"
            )

        num_classes = inputs.shape[1]

        # -- Build alpha per sample ------------------------------------
        if self.alpha is not None:
            if self.alpha.ndim == 0:
                # Scalar alpha
                alpha_per_sample = self.alpha.expand(targets.shape[0])
            else:
                # Per-class alpha
                assert self.alpha.shape[0] == num_classes, (
                    f"alpha shape {self.alpha.shape} does not match num_classes={num_classes}"
                )
                alpha_per_sample = self.alpha[targets]
        else:
            alpha_per_sample = torch.ones(targets.shape[0], device=inputs.device)

        # -- Label smoothing --------------------------------------------
        if self.label_smoothing > 0:
            smooth_targets = torch.full_like(
                inputs, self.label_smoothing / (num_classes - 1)
            )
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            log_probs = F.log_softmax(inputs, dim=1)
            ce_loss = -(smooth_targets * log_probs).sum(dim=1)
        else:
            ce_loss = F.cross_entropy(
                inputs, targets, reduction="none", ignore_index=self.ignore_index
            )

        # -- Focal term -------------------------------------------------
        probs = torch.exp(-ce_loss)  # pt for the true class
        focal_weight = (1.0 - probs) ** self.gamma

        # -- Apply alpha & focal weight ---------------------------------
        loss = alpha_per_sample * focal_weight * ce_loss

        # -- Mask ignored indices ---------------------------------------
        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).float()
            loss = loss * mask

        # -- Reduce -----------------------------------------------------
        if self.reduction == "mean":
            if self.ignore_index >= 0:
                return loss.sum() / mask.sum().clamp(min=1)
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def build_focal_loss(
    num_classes: int,
    class_counts: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
) -> MultiClassFocalLoss:
    """
    Convenience builder for MultiClassFocalLoss with balanced alpha.

    If class_counts is provided, alpha is computed as inverse frequency
    weighted by ``alpha`` (which then acts as an overall scale factor).

    Args:
        num_classes: Number of classes.
        class_counts: Tensor of per-class sample counts (N,).
        gamma: Focusing parameter.
        alpha: Overall alpha scalar (multiplied with normalised inverse frequency
            if class_counts is provided).
        reduction: Loss reduction mode.
        label_smoothing: Label smoothing factor.
        ignore_index: Index to ignore.

    Returns:
        Configured MultiClassFocalLoss instance.
    """
    if class_counts is not None:
        # Balanced alpha: inverse frequency, scaled by alpha
        counts = class_counts.float()
        balanced = 1.0 / (counts / counts.sum()).clamp(min=1e-8)
        balanced = balanced / balanced.max()  # normalise so max = 1.0
        alpha_tensor = balanced * alpha
    else:
        alpha_tensor = torch.full((num_classes,), alpha, dtype=torch.float32)

    return MultiClassFocalLoss(
        gamma=gamma,
        alpha=alpha_tensor,
        reduction=reduction,
        label_smoothing=label_smoothing,
        ignore_index=ignore_index,
    )
