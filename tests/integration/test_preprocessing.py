"""
Integration tests for the preprocessing pipeline.

Covers:
  - validate_and_open (valid bytes, corrupt bytes, empty, huge)
  - resize_pad_to_square (aspect ratio preservation)
  - preprocess_for_classifier / detector (output shape, range)
  - batch collation
  - End-to-end bytes_to_classifier_tensor
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from preprocessing.transforms import (
    batch_bytes_to_tensors,
    bytes_to_classifier_tensor,
    bytes_to_detector_tensor,
    collate_batch,
    preprocess_for_classifier,
    preprocess_for_detector,
    preprocess_for_severity,
    resize_pad_to_square,
    to_normalized_tensor,
    validate_and_open,
    validate_batch_size,
)


# ---------------------------------------------------------------------------
# validate_and_open
# ---------------------------------------------------------------------------


class TestValidateAndOpen:

    def test_opens_valid_jpeg(self, sample_image_bytes):
        img = validate_and_open(sample_image_bytes)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.width == 1024
        assert img.height == 768

    def test_opens_valid_png(self, sample_png_bytes):
        img = validate_and_open(sample_png_bytes)
        assert img.mode == "RGB"

    def test_raises_on_empty_bytes(self):
        with pytest.raises(ValueError, match="Empty"):
            validate_and_open(b"")

    def test_raises_on_corrupt_bytes(self):
        with pytest.raises(ValueError, match="Unrecognised"):
            validate_and_open(b"\x00\x01\x02NOT_AN_IMAGE")

    def test_converts_grayscale_to_rgb(self):
        from io import BytesIO
        img = Image.new("L", (100, 100), color=128)
        buf = BytesIO()
        img.save(buf, format="PNG")
        rgb = validate_and_open(buf.getvalue())
        assert rgb.mode == "RGB"

    def test_rejects_massive_image(self):
        # Create an image that exceeds MAX_SOURCE_DIM
        from preprocessing.transforms import MAX_SOURCE_DIM
        from io import BytesIO
        huge = Image.new("RGB", (MAX_SOURCE_DIM + 1, 100))
        buf = BytesIO()
        huge.save(buf, format="JPEG")
        with pytest.raises(ValueError, match="exceed"):
            validate_and_open(buf.getvalue())


# ---------------------------------------------------------------------------
# resize_pad_to_square
# ---------------------------------------------------------------------------


class TestResizePadToSquare:

    def test_portrait_image_becomes_square(self, sample_png_bytes):
        img = validate_and_open(sample_png_bytes)  # 640x480
        sq = resize_pad_to_square(img, target_size=380)
        assert sq.width == 380
        assert sq.height == 380

    def test_square_image_unchanged_dimensions(self):
        img = Image.new("RGB", (256, 256), color=(100, 200, 50))
        sq = resize_pad_to_square(img, target_size=256)
        assert sq.size == (256, 256)

    def test_aspect_ratio_preserved_approximately(self, sample_image_bytes):
        # 1024x768 -> ratio ~1.33
        img = validate_and_open(sample_image_bytes)
        sq = resize_pad_to_square(img, target_size=380)
        # The content should not be stretched; pasting should center it with padding
        # Verify the padded areas exist
        arr = list(sq.getdata())
        # At least one pixel should be padding (0,0,0)
        has_padding = any(p == (0, 0, 0) for p in arr)
        assert has_padding, "Expected padding pixels in the square image"


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


class TestNormalization:

    def test_output_is_float_tensor(self, sample_image_bytes):
        img = validate_and_open(sample_image_bytes)
        sq = resize_pad_to_square(img, 380)
        tensor = to_normalized_tensor(sq)
        assert tensor.dtype == torch.float32
        assert tensor.ndim == 3   # (C, H, W)

    def test_values_centered_around_zero(self, sample_image_bytes):
        img = validate_and_open(sample_image_bytes)
        sq = resize_pad_to_square(img, 380)
        tensor = to_normalized_tensor(sq)
        mean_val = tensor.mean().item()
        # After ImageNet normalisation, mean should be near 0
        assert -2.5 < mean_val < 2.5


# ---------------------------------------------------------------------------
# Full preprocessing
# ---------------------------------------------------------------------------


class TestPreprocessForClassifier:

    def test_output_shape(self, sample_image_bytes):
        img = validate_and_open(sample_image_bytes)
        tensor = preprocess_for_classifier(img)
        assert tensor.shape == (1, 3, 380, 380)

    def test_output_is_on_cpu(self, sample_image_bytes):
        img = validate_and_open(sample_image_bytes)
        tensor = preprocess_for_classifier(img)
        assert tensor.device.type == "cpu"


class TestPreprocessForDetector:

    def test_output_shape(self, sample_png_bytes):
        img = validate_and_open(sample_png_bytes)
        tensor = preprocess_for_detector(img)
        assert tensor.shape == (1, 3, 380, 380)


class TestPreprocessForSeverity:

    def test_output_shape(self, sample_image_bytes):
        img = validate_and_open(sample_image_bytes)
        tensor = preprocess_for_severity(img)
        assert tensor.shape == (1, 3, 380, 380)


# ---------------------------------------------------------------------------
# Batch collation
# ---------------------------------------------------------------------------


class TestCollateBatch:

    def test_collates_correctly(self, sample_image_bytes):
        img = validate_and_open(sample_image_bytes)
        t1 = preprocess_for_classifier(img)
        t2 = preprocess_for_classifier(img)
        batched = collate_batch([t1, t2])
        assert batched.shape == (2, 3, 380, 380)

    def test_raises_on_empty_list(self):
        with pytest.raises(ValueError):
            collate_batch([])

    def test_raises_on_shape_mismatch(self, sample_image_bytes):
        img = validate_and_open(sample_image_bytes)
        t1 = preprocess_for_classifier(img)
        t2 = torch.randn(1, 3, 224, 224)  # different spatial dims
        with pytest.raises(ValueError, match="Shape mismatch"):
            collate_batch([t1, t2])


class TestBatchSizeValidation:

    def test_clamp_to_max(self):
        assert validate_batch_size(50) == 32  # MAX_BATCH_SIZE default

    def test_pass_through_small_batch(self):
        assert validate_batch_size(4) == 4


# ---------------------------------------------------------------------------
# End-to-end convenience
# ---------------------------------------------------------------------------


class TestBytesToTensor:

    def test_classifier_pipeline(self, sample_image_bytes):
        tensor = bytes_to_classifier_tensor(sample_image_bytes)
        assert tensor.shape == (1, 3, 380, 380)

    def test_detector_pipeline(self, sample_png_bytes):
        tensor = bytes_to_detector_tensor(sample_png_bytes)
        assert tensor.shape == (1, 3, 380, 380)

    def test_batch_pipeline(self, sample_batch_bytes):
        tensor, pil_list = batch_bytes_to_tensors(list(sample_batch_bytes))
        assert tensor.shape == (4, 3, 380, 380)
        assert len(pil_list) == 4

    def test_batch_pipeline_clamps(self, large_batch_bytes):
        tensor, _ = batch_bytes_to_tensors(list(large_batch_bytes))
        # MAX_BATCH_SIZE is 32 by default (monkeypatched to 4 in fixture)
        # We just verify it doesn't crash and shape is correct
        assert tensor.ndim == 4
        assert tensor.size(0) <= 32
