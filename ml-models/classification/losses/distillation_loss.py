"""
Knowledge distillation losses for model compression.

Provides:
- SoftTargetLoss: KL divergence with temperature scaling (Hinton et al., 2015)
- FeatureDistillationLoss: L2 on intermediate features with adaptation layer
- AttentionTransferLoss: L2 on spatial attention maps (Zagoruyko & Komodakis, 2017)
- CombinedDistillationLoss: weighted sum with configurable alphas
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Soft Target Loss
# ---------------------------------------------------------------------------

class SoftTargetLoss(nn.Module):
    """
    KL-divergence loss between teacher and student soft targets with
    temperature scaling.

    L_soft = T^2 * KL(softmax(z_t / T) || softmax(z_s / T))

    The T^2 factor keeps gradient magnitudes roughly invariant to temperature
    when using hard-target cross-entropy alongside the soft loss.

    Args:
        temperature: Softening temperature. Higher values produce softer
            probability distributions. Default 4.0.
    """

    def __init__(self, temperature: float = 4.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute soft-target distillation loss.

        Args:
            student_logits: (N, C) logits from the student model.
            teacher_logits: (N, C) logits from the teacher model.

        Returns:
            Scalar loss.
        """
        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)
        loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean")
        return loss * (T * T)


# ---------------------------------------------------------------------------
# Feature Distillation Loss
# ---------------------------------------------------------------------------

class FeatureDistillationLoss(nn.Module):
    """
    L2 loss between intermediate feature maps after passing student features
    through a learnable adaptation layer.

    This allows matching features when student and teacher have different
    channel dimensions.

    Args:
        student_channels: Number of channels in the student feature map.
        teacher_channels: Number of channels in the teacher feature map.
        adaptation_type: 'conv1x1' or 'linear'. Default 'conv1x1'.
    """

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        adaptation_type: str = "conv1x1",
    ) -> None:
        super().__init__()
        self.adaptation_type = adaptation_type

        if adaptation_type == "conv1x1":
            self.adaptation = nn.Conv2d(
                student_channels, teacher_channels, kernel_size=1, bias=False
            )
        elif adaptation_type == "linear":
            self.adaptation = nn.Sequential(
                nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(teacher_channels),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"Unknown adaptation_type: {adaptation_type}")

    def forward(
        self,
        student_feature: torch.Tensor,
        teacher_feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute feature distillation loss.

        Args:
            student_feature: (N, C_s, H, W) intermediate feature from student.
            teacher_feature: (N, C_t, H, W) intermediate feature from teacher.

        Returns:
            Scalar loss.
        """
        # Adapt student channels to match teacher
        adapted = self.adaptation(student_feature)

        # Interpolate if spatial dimensions differ
        if adapted.shape[2:] != teacher_feature.shape[2:]:
            adapted = F.interpolate(
                adapted,
                size=teacher_feature.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        return F.mse_loss(adapted, teacher_feature.detach())


# ---------------------------------------------------------------------------
# Attention Transfer Loss
# ---------------------------------------------------------------------------

class AttentionTransferLoss(nn.Module):
    """
    L2 loss on spatial attention maps.

    Attention map is computed as the channel-wise sum of squared activations
    (following Zagoruyko & Komodakis, 2017), then L2-normalised.

    Args:
        p: Power for activation pooling (p=2 yields activation-based attention).
        aggregate: 'sum' (default) or 'mean' for channel aggregation.
    """

    def __init__(self, p: int = 2, aggregate: str = "sum") -> None:
        super().__init__()
        self.p = p
        self.aggregate = aggregate

    def _attention_map(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial attention map from feature tensor.

        Args:
            feature: (N, C, H, W).

        Returns:
            (N, 1, H, W) attention map.
        """
        if self.aggregate == "sum":
            attn = feature.abs().pow(self.p).sum(dim=1, keepdim=True)
        else:
            attn = feature.abs().pow(self.p).mean(dim=1, keepdim=True)

        # L2 normalise each spatial map
        norm = attn.view(attn.shape[0], -1).norm(p=2, dim=1, keepdim=True)
        norm = norm.view(-1, 1, 1, 1).clamp(min=1e-8)
        return attn / norm

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_features: (N, C, H, W).
            teacher_features: (N, C, H, W).

        Returns:
            Scalar loss.
        """
        student_attn = self._attention_map(student_features)
        teacher_attn = self._attention_map(teacher_features)

        # Interpolate if needed
        if student_attn.shape[2:] != teacher_attn.shape[2:]:
            student_attn = F.interpolate(
                student_attn,
                size=teacher_attn.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        return F.mse_loss(student_attn, teacher_attn.detach())


# ---------------------------------------------------------------------------
# Combined Distillation Loss
# ---------------------------------------------------------------------------

class CombinedDistillationLoss(nn.Module):
    """
    Weighted combination of multiple distillation losses.

    L_total = alpha_soft * L_soft + alpha_feat * L_feat + alpha_attn * L_attn

    Args:
        alpha_soft: Weight for SoftTargetLoss (or None to disable).
        alpha_feat: Weight for FeatureDistillationLoss (or None to disable).
        alpha_attn: Weight for AttentionTransferLoss (or None to disable).
        soft_temperature: Temperature for soft targets.
        student_channels: Student feature channels (for FeatureDistillationLoss).
        teacher_channels: Teacher feature channels (for FeatureDistillationLoss).
        adaptation_type: Adaptation layer type for feature distillation.
        attn_p: Power for attention pooling.
    """

    def __init__(
        self,
        alpha_soft: Optional[float] = 1.0,
        alpha_feat: Optional[float] = 0.1,
        alpha_attn: Optional[float] = 0.01,
        soft_temperature: float = 4.0,
        student_channels: int = 256,
        teacher_channels: int = 512,
        adaptation_type: str = "conv1x1",
        attn_p: int = 2,
    ) -> None:
        super().__init__()

        self.alpha_soft = alpha_soft
        self.alpha_feat = alpha_feat
        self.alpha_attn = alpha_attn

        self.soft_loss: Optional[SoftTargetLoss] = None
        self.feat_loss: Optional[FeatureDistillationLoss] = None
        self.attn_loss: Optional[AttentionTransferLoss] = None

        if alpha_soft is not None and alpha_soft > 0:
            self.soft_loss = SoftTargetLoss(temperature=soft_temperature)

        if alpha_feat is not None and alpha_feat > 0:
            self.feat_loss = FeatureDistillationLoss(
                student_channels=student_channels,
                teacher_channels=teacher_channels,
                adaptation_type=adaptation_type,
            )

        if alpha_attn is not None and alpha_attn > 0:
            self.attn_loss = AttentionTransferLoss(p=attn_p, aggregate="sum")

    def forward(
        self,
        student_logits: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        student_features: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined distillation loss.

        At least one pair of (student_logits, teacher_logits) or
        (student_features, teacher_features) must be provided.

        Args:
            student_logits: (N, C) student logits.
            teacher_logits: (N, C) teacher logits.
            student_features: (N, C_s, H, W) student features.
            teacher_features: (N, C_t, H, W) teacher features.

        Returns:
            (total_loss, loss_components_dict) where loss_components_dict
            contains individual loss values for logging.
        """
        total = torch.tensor(0.0, device=self._device())
        components: Dict[str, float] = {}

        if self.soft_loss is not None:
            assert student_logits is not None, "student_logits required for soft loss"
            assert teacher_logits is not None, "teacher_logits required for soft loss"
            l_soft = self.soft_loss(student_logits, teacher_logits)
            total = total + self.alpha_soft * l_soft  # type: ignore[operator]
            components["distill/soft_loss"] = l_soft.item()

        if self.feat_loss is not None:
            assert student_features is not None, "student_features required for feat loss"
            assert teacher_features is not None, "teacher_features required for feat loss"
            l_feat = self.feat_loss(student_features, teacher_features)
            total = total + self.alpha_feat * l_feat  # type: ignore[operator]
            components["distill/feat_loss"] = l_feat.item()

        if self.attn_loss is not None:
            assert student_features is not None, "student_features required for attn loss"
            assert teacher_features is not None, "teacher_features required for attn loss"
            l_attn = self.attn_loss(student_features, teacher_features)
            total = total + self.alpha_attn * l_attn  # type: ignore[operator]
            components["distill/attn_loss"] = l_attn.item()

        components["distill/total"] = total.item()
        return total, components

    def _device(self) -> torch.device:
        """Infer device from module parameters."""
        for p in self.parameters():
            return p.device
        return torch.device("cpu")
