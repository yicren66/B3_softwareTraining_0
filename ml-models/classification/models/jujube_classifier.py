"""
JujubeClassifier — 枣树「7病8虫」病虫害图像分类模型

Wraps a timm (PyTorch Image Models) backbone with a classification head for
the 15-class jujube pest/disease taxonomy defined in SRS-SYS02 §4.1.1 IR-02.

Classes (15):
  病害 (7): 枣炭疽病, 枣疯病, 枣树锈病, 枣缩果病, 枣果腐病, 枣褐斑病, 枣叶黑斑病
  虫害 (8): 枣芽象甲, 枣瘿蚊, 桃小食心虫, 绿盲蝽, 枣尺蠖, 枣镰翅小卷蛾, 枣红蜘蛛, 枣龟蜡蚧
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class JujubeClassifier(nn.Module):
    """
    Jujube pest & disease classifier — 「7病8虫」15-class taxonomy.

    Wraps a timm backbone (default: EfficientNet-B4) with a configurable
    classifier head. Supports feature extraction and knowledge distillation.

    Args:
        backbone_name: timm model name (e.g. 'efficientnet_b4').
        pretrained: Load ImageNet pretrained weights.
        num_classes: Number of output classes (default 15 for 7病8虫).
        dropout: Dropout rate for the classifier head.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b4",
        pretrained: bool = True,
        num_classes: int = 15,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        import timm

        self.backbone_name = backbone_name
        self.num_classes = num_classes

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # remove classifier head
        )

        # Determine feature dimension
        feat_dim = self._get_feature_dim()

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def _get_feature_dim(self) -> int:
        """Infer feature dimension from backbone."""
        dummy = torch.randn(2, 3, 380, 380)
        with torch.no_grad():
            feats = self.backbone(dummy)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        return feats.shape[1]

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (N, 3, H, W).
            return_features: If True, also return intermediate features.

        Returns:
            Dict with keys 'logits' and optionally 'features'.
        """
        features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            features = features[-1]

        logits = self.head(features)

        output: Dict[str, torch.Tensor] = {"logits": logits}
        if return_features:
            output["features"] = features

        return output

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features only (no classification head)."""
        features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        return features
