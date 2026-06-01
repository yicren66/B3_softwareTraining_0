"""
Image preprocessing pipeline.

Steps:
  1. Validate bytes, open as PIL Image, ensure RGB.
  2. Resize preserving aspect ratio (pad to square).
  3. Convert to tensor, apply ImageNet normalisation.
  4. Batch collation.
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF  # type: ignore[import-untyped]
from PIL import Image, UnidentifiedImageError

from config import settings

logger = logging.getLogger(__name__)

# ImageNet statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Maximum image size to guard against OOM
MAX_SOURCE_DIM = 4096


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_and_open(image_bytes: bytes) -> Image.Image:
    """Validate raw bytes and open as an RGB PIL Image.

    Raises:
        ValueError: If bytes are empty, corrupt, or exceed max dimensions.
    """
    if not image_bytes:
        raise ValueError("Empty image bytes.")

    try:
        img = Image.open(BytesIO(image_bytes))
    except UnidentifiedImageError:
        raise ValueError("Unrecognised image format.") from None

    # Reject massive images early
    if img.width > MAX_SOURCE_DIM or img.height > MAX_SOURCE_DIM:
        raise ValueError(
            f"Image dimensions ({img.width}x{img.height}) exceed "
            f"maximum allowed ({MAX_SOURCE_DIM}x{MAX_SOURCE_DIM})."
        )

    # Ensure RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    return img


# ---------------------------------------------------------------------------
# Aspect-ratio-preserving resize → pad to square
# ---------------------------------------------------------------------------


def resize_pad_to_square(
    img: Image.Image, target_size: int = 380, fill: Tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    """Resize *img* so its longest side equals *target_size*, then pad the
    shorter side with *fill* to produce a square image."""
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = img.resize((new_w, new_h), Image.BILINEAR)

    square = Image.new("RGB", (target_size, target_size), fill)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    square.paste(resized, (paste_x, paste_y))

    return square


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def to_normalized_tensor(img: Image.Image) -> torch.Tensor:
    """PIL Image → (C, H, W) float tensor, normalised with ImageNet stats."""
    tensor = TF.to_tensor(img)  # scales [0,255] uint8 → [0,1] float
    tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return tensor


# ---------------------------------------------------------------------------
# Full preprocessing for each model
# ---------------------------------------------------------------------------


def preprocess_for_classifier(image: Image.Image) -> torch.Tensor:
    """Prepare image for the jujube-disease classifier.

    Returns:
        torch.Tensor of shape (1, 3, 380, 380).
    """
    h, w = settings.input_height, settings.input_width
    square = resize_pad_to_square(image, target_size=h)
    tensor = to_normalized_tensor(square)
    return tensor.unsqueeze(0)  # (1, C, H, W)


def preprocess_for_detector(image: Image.Image) -> torch.Tensor:
    """Prepare image for the small-target FPN detector.

    Returns:
        torch.Tensor of shape (1, 3, 380, 380).
    """
    h, w = settings.input_height, settings.input_width
    square = resize_pad_to_square(image, target_size=h)
    tensor = to_normalized_tensor(square)
    return tensor.unsqueeze(0)


def preprocess_for_severity(image: Image.Image) -> torch.Tensor:
    """Prepare image for the severity classifier.

    Returns:
        torch.Tensor of shape (1, 3, 380, 380).
    """
    # Same pipeline as classifier — shared input shape
    return preprocess_for_classifier(image)


# ---------------------------------------------------------------------------
# Batch collation
# ---------------------------------------------------------------------------


def collate_batch(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Concatenate a list of (1, C, H, W) tensors into (B, C, H, W)."""
    if not tensors:
        raise ValueError("Cannot collate empty tensor list.")
    # Validate shapes
    ref_shape = tensors[0].shape
    for i, t in enumerate(tensors):
        if t.shape != ref_shape:
            raise ValueError(
                f"Shape mismatch at index {i}: got {t.shape}, expected {ref_shape}"
            )
    return torch.cat(tensors, dim=0)


def validate_batch_size(count: int) -> int:
    """Clamp batch size to configured maximum."""
    if count > settings.MAX_BATCH_SIZE:
        logger.warning(
            "Requested batch size %d exceeds MAX_BATCH_SIZE %d — clamping.",
            count, settings.MAX_BATCH_SIZE,
        )
        return settings.MAX_BATCH_SIZE
    return count


# ---------------------------------------------------------------------------
# Convenience: end-to-end bytes → tensor
# ---------------------------------------------------------------------------


def bytes_to_classifier_tensor(image_bytes: bytes) -> torch.Tensor:
    img = validate_and_open(image_bytes)
    return preprocess_for_classifier(img)


def bytes_to_detector_tensor(image_bytes: bytes) -> torch.Tensor:
    img = validate_and_open(image_bytes)
    return preprocess_for_detector(img)


def batch_bytes_to_tensors(
    image_batches: List[bytes],
) -> Tuple[torch.Tensor, List[Image.Image]]:
    """Process a list of raw image bytes into a single batched tensor.

    Returns:
        (batched_tensor, list_of_pil_images) where batched_tensor shape
        is (N, 3, 380, 380) clipped to MAX_BATCH_SIZE.
    """
    image_batches = image_batches[: settings.MAX_BATCH_SIZE]
    tensors: List[torch.Tensor] = []
    pil_images: List[Image.Image] = []
    for raw in image_batches:
        img = validate_and_open(raw)
        pil_images.append(img)
        square = resize_pad_to_square(img, target_size=settings.input_height)
        tensors.append(to_normalized_tensor(square).unsqueeze(0))
    return collate_batch(tensors), pil_images
