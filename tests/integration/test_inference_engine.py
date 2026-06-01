"""
Integration tests for the InferenceEngine.

Covers:
  - Engine lifecycle (load, ready flag)
  - classify() output range and shape
  - detect_small_targets()
  - assess_severity()
  - run_full_pipeline() end-to-end
  - batch_classify() with multiple images
  - Latency tracking in results
  - CPU fallback
"""

from __future__ import annotations

import pytest
import torch

from inference.engine import (
    InferenceEngine,
    RecognitionResult,
    SmallTarget,
)


# ---------------------------------------------------------------------------
# Engine fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    """A pre-loaded engine using PyTorch stub models on CPU."""
    eng = InferenceEngine()
    eng.load_models()
    return eng


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestEngineLifecycle:

    def test_engine_ready_after_load(self, engine):
        assert engine.ready is True

    def test_can_load_multiple_times(self, engine):
        # Should be idempotent
        engine.load_models()
        assert engine.ready is True

    def test_fallback_to_cpu(self, engine):
        # CPU fallback should not crash when already on CPU
        engine.fallback_to_cpu()
        assert engine.ready is True


# ---------------------------------------------------------------------------
# Classify
# ---------------------------------------------------------------------------


class TestClassify:

    def test_returns_valid_index_and_confidence(self, engine):
        dummy = torch.randn(1, 3, 380, 380)
        idx, conf, probs = engine.classify(dummy)
        assert 0 <= idx < 20
        assert 0.0 <= conf <= 1.0
        assert len(probs) == 20

    def test_probabilities_sum_to_one(self, engine):
        dummy = torch.randn(1, 3, 380, 380)
        _, _, probs = engine.classify(dummy)
        assert abs(sum(probs) - 1.0) < 0.001

    def test_different_input_gives_same_output_shape(self, engine):
        for _ in range(3):
            x = torch.randn(1, 3, 380, 380)
            idx, conf, probs = engine.classify(x)
            assert len(probs) == 20
            assert 0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# Detect small targets
# ---------------------------------------------------------------------------


class TestDetectSmallTargets:

    def test_returns_list(self, engine):
        dummy = torch.randn(1, 3, 380, 380)
        targets = engine.detect_small_targets(dummy)
        assert isinstance(targets, list)

    def test_each_target_has_required_fields(self, engine):
        dummy = torch.randn(1, 3, 380, 380)
        targets = engine.detect_small_targets(dummy)
        for t in targets:
            assert isinstance(t, SmallTarget)
            assert len(t.bbox) == 4
            assert 0.0 <= t.confidence <= 1.0


# ---------------------------------------------------------------------------
# Assess severity
# ---------------------------------------------------------------------------


class TestAssessSeverity:

    def test_returns_valid_index(self, engine):
        dummy = torch.randn(1, 3, 380, 380)
        idx, probs = engine.assess_severity(dummy)
        assert idx in (0, 1, 2, 3)
        assert len(probs) == 4

    def test_probabilities_sum_to_one(self, engine):
        dummy = torch.randn(1, 3, 380, 380)
        _, probs = engine.assess_severity(dummy)
        assert abs(sum(probs) - 1.0) < 0.001


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:

    def test_returns_recognition_result(self, engine, sample_image_bytes):
        result = engine.run_full_pipeline(sample_image_bytes)
        assert isinstance(result, RecognitionResult)

    def test_latency_is_positive(self, engine, sample_image_bytes):
        result = engine.run_full_pipeline(sample_image_bytes)
        assert result.latency_ms > 0
        assert result.preprocess_ms > 0

    def test_confidence_is_valid(self, engine, sample_image_bytes):
        result = engine.run_full_pipeline(sample_image_bytes)
        assert 0.0 <= result.confidence <= 1.0

    def test_class_idx_in_range(self, engine, sample_image_bytes):
        result = engine.run_full_pipeline(sample_image_bytes)
        assert 0 <= result.class_idx < 20

    def test_severity_idx_in_range(self, engine, sample_image_bytes):
        result = engine.run_full_pipeline(sample_image_bytes)
        assert result.severity_idx in (0, 1, 2, 3)

    def test_probs_are_non_empty(self, engine, sample_image_bytes):
        result = engine.run_full_pipeline(sample_image_bytes)
        assert len(result.class_probs) == 20
        assert len(result.severity_probs) == 4

    def test_png_format_works(self, engine, sample_png_bytes):
        result = engine.run_full_pipeline(sample_png_bytes)
        assert isinstance(result, RecognitionResult)
        assert result.confidence >= 0

    def test_tiny_image_works(self, engine, tiny_image_bytes):
        """Minimal 8x8 image should still go through the pipeline."""
        result = engine.run_full_pipeline(tiny_image_bytes)
        assert isinstance(result, RecognitionResult)

    def test_invalid_bytes_raises(self, engine):
        with pytest.raises(ValueError):
            engine.run_full_pipeline(b"not an image at all")


# ---------------------------------------------------------------------------
# Batch classify
# ---------------------------------------------------------------------------


class TestBatchClassify:

    def test_batch_of_three(self, engine, sample_image_bytes):
        results = engine.batch_classify(
            [sample_image_bytes, sample_image_bytes, sample_image_bytes]
        )
        assert len(results) == 3
        for idx, conf, probs in results:
            assert 0 <= idx < 20
            assert 0.0 <= conf <= 1.0
            assert len(probs) == 20

    def test_batch_exceeding_max(self, engine, sample_image_bytes):
        # MAX_BATCH_SIZE=32 by default; send 40
        many = [sample_image_bytes] * 40
        results = engine.batch_classify(many)
        assert len(results) == 40

    def test_empty_batch(self, engine):
        results = engine.batch_classify([])
        assert results == []
