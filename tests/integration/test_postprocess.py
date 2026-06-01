"""
Integration tests for the postprocessing module.

Covers:
  - Label mapping (known index, unknown index)
  - Severity mapping
  - Confidence calibration (temperature scaling)
  - Severity threshold logic
  - Full format_recognition_result pipeline
  - Result summary string
"""

from __future__ import annotations

import math

import pytest

from inference.engine import RecognitionResult, SmallTarget
from inference.postprocess import (
    DISEASE_MAP,
    SEVERITY_MAP,
    apply_severity_threshold,
    calibrate_confidence,
    format_recognition_result,
    map_class_idx_to_name,
    map_severity,
    result_summary,
)


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------


class TestClassIdxToName:

    def test_known_index_0(self):
        name_cn, category, affected, symptoms, kg_id = map_class_idx_to_name(0)
        assert "枣炭疽病" in name_cn
        assert category == "病害"
        assert "果实" in affected
        assert len(symptoms) > 0
        assert kg_id == "KG-ENT-001"

    def test_known_index_1(self):
        name_cn, category, affected, symptoms, kg_id = map_class_idx_to_name(1)
        assert "枣疯病" in name_cn
        assert category == "病害"

    def test_unknown_index(self):
        name_cn, category, affected, symptoms, kg_id = map_class_idx_to_name(999)
        assert name_cn == "未知"
        assert category == "unknown"
        assert kg_id.endswith("unknown")

    def test_every_registered_index_has_all_fields(self):
        for idx in DISEASE_MAP:
            name_cn, category, affected, symptoms, kg_id = map_class_idx_to_name(idx)
            assert name_cn
            assert category
            assert kg_id


class TestSeverityMap:

    def test_level_0_is_healthy(self):
        s = map_severity(0)
        assert s["label_cn"] == "健康"

    def test_level_3_is_severe(self):
        s = map_severity(3)
        assert s["label_cn"] == "重度"

    def test_out_of_range_falls_back_to_healthy(self):
        s = map_severity(99)
        assert s["level"] == 0


# ---------------------------------------------------------------------------
# Confidence calibration
# ---------------------------------------------------------------------------


class TestCalibrateConfidence:

    def test_default_temperature_1_unchanged(self):
        probs = [0.1, 0.2, 0.3, 0.4]
        result = calibrate_confidence(probs, temperature=1.0)
        assert len(result) == 4
        assert math.isclose(sum(result), 1.0, rel_tol=1e-6)
        # With T=1 should be close to original
        for a, b in zip(result, probs):
            assert math.isclose(a, b, rel_tol=1e-6)

    def test_high_temperature_flattens(self):
        probs = [0.01, 0.99]
        result = calibrate_confidence(probs, temperature=10.0)
        assert len(result) == 2
        assert math.isclose(sum(result), 1.0, rel_tol=1e-6)
        # T >> 1 pulls extremes towards uniform
        assert result[0] > probs[0]  # low prob increased
        assert result[1] < probs[1]  # high prob decreased

    def test_low_temperature_sharpens(self):
        probs = [0.4, 0.6]
        result = calibrate_confidence(probs, temperature=0.2)
        assert math.isclose(sum(result), 1.0, rel_tol=1e-6)
        # T << 1 sharpens — max gets larger
        assert result[1] > 0.6

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            calibrate_confidence([0.5, 0.5], temperature=-1)

    def test_zero_temperature_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            calibrate_confidence([0.5, 0.5], temperature=0)


# ---------------------------------------------------------------------------
# Severity threshold
# ---------------------------------------------------------------------------


class TestSeverityThreshold:

    def test_high_confidence_unchanged(self):
        idx, conf = apply_severity_threshold(
            2, [0.1, 0.1, 0.7, 0.1], confidence_threshold=0.5
        )
        assert idx == 2
        assert math.isclose(conf, 0.7)

    def test_low_confidence_defaults_to_mild(self):
        idx, conf = apply_severity_threshold(
            3, [0.3, 0.3, 0.2, 0.2], confidence_threshold=0.5
        )
        assert idx == 1   # Mild
        assert conf == 0.3

    def test_no_threshold_argument_uses_default(self):
        idx, conf = apply_severity_threshold(0, [0.9, 0.05, 0.03, 0.02])
        assert idx == 0
        assert math.isclose(conf, 0.9)


# ---------------------------------------------------------------------------
# Format result
# ---------------------------------------------------------------------------


def _make_result(
    class_idx: int = 0,
    confidence: float = 0.92,
    severity_idx: int = 1,
    targets: list | None = None,
) -> RecognitionResult:
    return RecognitionResult(
        class_idx=class_idx,
        confidence=confidence,
        class_probs=[0.92 if i == class_idx else 0.08 / 19 for i in range(20)],
        severity_idx=severity_idx,
        severity_probs=[0.1, 0.7, 0.15, 0.05],
        small_targets=targets or [],
        latency_ms=42.5,
        preprocess_ms=10.0,
        classify_ms=20.0,
        detect_ms=8.0,
        severity_ms=4.5,
    )


class TestFormatRecognitionResult:

    def test_output_has_expected_top_level_keys(self):
        data = format_recognition_result(_make_result())
        for key in ("prediction", "severity", "small_targets", "target_count",
                     "knowledge_graph", "latency"):
            assert key in data, f"Missing key: {key}"

    def test_prediction_structure(self):
        data = format_recognition_result(_make_result(0))
        p = data["prediction"]
        assert p["class_idx"] == 0
        assert "name_cn" in p
        assert "name_en" in p
        assert "confidence" in p
        assert "class_probabilities" in p
        assert len(p["class_probabilities"]) == 20

    def test_severity_structure(self):
        data = format_recognition_result(_make_result(severity_idx=2))
        s = data["severity"]
        assert s["level"] in (0, 1, 2, 3)
        assert "label_cn" in s
        assert "confidence" in s

    def test_small_targets_included(self):
        target = SmallTarget(bbox=(0.1, 0.2, 0.3, 0.4), class_id=1, confidence=0.81)
        data = format_recognition_result(_make_result(targets=[target]))
        assert data["target_count"] == 1
        assert len(data["small_targets"]) == 1
        t_out = data["small_targets"][0]
        assert t_out["bbox"] == [0.1, 0.2, 0.3, 0.4]
        assert t_out["confidence"] == 0.81

    def test_latency_breakdown(self):
        data = format_recognition_result(_make_result())
        lt = data["latency"]
        assert lt["total_ms"] == 42.5
        assert lt["preprocess_ms"] == 10.0
        assert lt["classify_ms"] == 20.0

    def test_kg_entity_included(self):
        data = format_recognition_result(_make_result(1))
        assert data["knowledge_graph"]["entity_id"].startswith("KG-ENT-")

    def test_probabilities_sum_to_one(self):
        data = format_recognition_result(_make_result())
        for probs in (data["prediction"]["class_probabilities"],
                       data["severity"]["probabilities"]):
            assert abs(sum(probs) - 1.0) < 0.001


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestResultSummary:

    def test_summary_is_non_empty_string(self):
        data = format_recognition_result(_make_result())
        summary = result_summary(data)
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "[" in summary   # disease name bracket
        assert "latency" in summary.lower()
