"""
Shared pytest fixtures for the Jujube Platform integration test suite.

Fixtures provide:
  - Synthetic RGB test images (JPEG, PNG)
  - Sample batches for batch inference testing
  - Environment overrides for testing
  - A seeded InferenceEngine
  - Test clients for KG, Reasoning, and Model Management services
"""

from __future__ import annotations

import os
from io import BytesIO
from typing import List

import pytest
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Synthetic test images
# ---------------------------------------------------------------------------


def _make_test_image(width: int = 1024, height: int = 768, label: str = "test") -> bytes:
    """Generate a synthetic RGB image with a text label as raw bytes."""
    img = Image.new("RGB", (width, height), color=(60, 80, 40))
    draw = ImageDraw.Draw(img)
    # Draw a few rectangles to simulate leaf texture
    for i in range(10):
        x0, y0 = i * 100, (i * 77) % height
        draw.rectangle([x0, y0, x0 + 90, y0 + 60], outline=(120, 160, 80), width=2)
    # Add text label
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except OSError:
        font = ImageFont.load_default()
    draw.text((20, 20), label, fill=(255, 255, 255), font=font)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _make_test_png(width: int = 640, height: int = 480) -> bytes:
    """Generate a synthetic RGB PNG image."""
    img = Image.new("RGB", (width, height), color=(180, 120, 60))
    draw = ImageDraw.Draw(img)
    draw.ellipse([200, 150, 440, 330], fill=(220, 60, 40), outline=(255, 255, 255), width=3)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_tiny_image() -> bytes:
    """A minimal 8x8 RGB PNG for edge-case testing."""
    img = Image.new("RGB", (8, 8), color=(0, 128, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_image_bytes() -> bytes:
    """A single 1024x768 JPEG image as bytes."""
    return _make_test_image(1024, 768, "sample")


@pytest.fixture(scope="session")
def sample_png_bytes() -> bytes:
    """A single 640x480 PNG image as bytes."""
    return _make_test_png()


@pytest.fixture(scope="session")
def tiny_image_bytes() -> bytes:
    """A minimal 8x8 PNG."""
    return _make_tiny_image()


@pytest.fixture(scope="session")
def sample_batch_bytes() -> List[bytes]:
    """A list of 4 synthetic JPEG image bytes."""
    return [_make_test_image(800, 600, f"batch-{i}") for i in range(4)]


@pytest.fixture(scope="session")
def large_batch_bytes() -> List[bytes]:
    """A list of 40 synthetic image bytes (exceeds MAX_BATCH_SIZE of 32)."""
    return [_make_test_image(512, 512, f"big-{i}") for i in range(40)]


@pytest.fixture(autouse=True)
def mock_model_paths(monkeypatch):
    """Point config paths to non-existent locations so it uses stub models."""
    monkeypatch.setenv("IR_MODEL_PATH", "/tmp/_test_classifier.pth")
    monkeypatch.setenv("IR_ONNX_MODEL_PATH", "/tmp/_test_classifier.onnx")
    monkeypatch.setenv("IR_TENSORRT_ENGINE_PATH", "/tmp/_test_classifier.trt")
    monkeypatch.setenv("IR_DETECTOR_MODEL_PATH", "/tmp/_test_detector.pth")
    monkeypatch.setenv("IR_DETECTOR_ONNX_PATH", "/tmp/_test_detector.onnx")
    monkeypatch.setenv("IR_DETECTOR_TRT_PATH", "/tmp/_test_detector.trt")
    monkeypatch.setenv("IR_SEVERITY_MODEL_PATH", "/tmp/_test_severity.pth")
    monkeypatch.setenv("IR_SEVERITY_ONNX_PATH", "/tmp/_test_severity.onnx")
    monkeypatch.setenv("IR_SEVERITY_TRT_PATH", "/tmp/_test_severity.trt")
    monkeypatch.setenv("IR_USE_GPU", "false")
    monkeypatch.setenv("IR_INFERENCE_BACKEND", "pytorch")
    monkeypatch.setenv("IR_BATCH_SIZE", "4")
    monkeypatch.setenv("IR_LOG_LEVEL", "WARNING")


@pytest.fixture(scope="session")
def engine_with_stubs():
    """An InferenceEngine loaded with PyTorch stub models on CPU."""
    # Re-import settings after monkeypatch so env vars take effect
    import importlib
    import config
    importlib.reload(config)
    from inference.engine import InferenceEngine
    engine = InferenceEngine()
    engine.load_models()
    return engine


# ---------------------------------------------------------------------------
# Service test clients (for integration tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def kg_client():
    """FastAPI TestClient for the Knowledge Graph Service."""
    import sys
    sys.path.insert(0, str(__file__).rsplit("tests", 1)[0] + "services/knowledge-graph/src")
    try:
        from main import app
        from fastapi.testclient import TestClient
        return TestClient(app)
    except ImportError:
        pytest.skip("Knowledge Graph service not importable")


@pytest.fixture(scope="session")
def reasoning_client():
    """FastAPI TestClient for the Reasoning Engine."""
    import sys
    sys.path.insert(0, str(__file__).rsplit("tests", 1)[0] + "services/reasoning-engine/src")
    try:
        from main import app
        from fastapi.testclient import TestClient
        return TestClient(app)
    except ImportError:
        pytest.skip("Reasoning Engine not importable")


@pytest.fixture(scope="session")
def model_mgmt_client():
    """FastAPI TestClient for the Model Management Service."""
    import sys
    sys.path.insert(0, str(__file__).rsplit("tests", 1)[0] + "services/model-management/src")
    try:
        from main import app
        from fastapi.testclient import TestClient
        return TestClient(app)
    except ImportError:
        pytest.skip("Model Management service not importable")
