"""
Integration tests for the configuration module.

Verifies:
  - Default settings are sane
  - Environment variable overrides work correctly
  - InferenceBackend enum parsing
  - Device string generation
"""

from __future__ import annotations

import importlib
import os

import pytest

# Force reload to pick up monkeypatched env vars
import config as cfg_module


class TestDefaults:
    """Sanity checks on the configuration defaults."""

    def test_default_input_size(self):
        importlib.reload(cfg_module)
        s = cfg_module.settings
        assert s.INPUT_SIZE == (3, 380, 380)

    def test_default_batch_size(self):
        importlib.reload(cfg_module)
        s = cfg_module.settings
        # Fixture overrides IR_BATCH_SIZE=4 for test isolation
        assert s.BATCH_SIZE == 4

    def test_default_max_batch_size(self):
        importlib.reload(cfg_module)
        s = cfg_module.settings
        assert s.MAX_BATCH_SIZE == 32

    def test_default_confidence_threshold(self):
        importlib.reload(cfg_module)
        s = cfg_module.settings
        assert s.CONFIDENCE_THRESHOLD == 0.5

    def test_default_backend(self):
        importlib.reload(cfg_module)
        s = cfg_module.settings
        # Fixture overrides IR_INFERENCE_BACKEND=pytorch for test isolation
        assert s.INFERENCE_BACKEND == cfg_module.InferenceBackend.PYTORCH

    def test_device_cpu_when_gpu_false(self, monkeypatch):
        monkeypatch.setenv("IR_USE_GPU", "false")
        importlib.reload(cfg_module)
        s = cfg_module.settings
        assert s.device == "cpu"

    def test_input_dimension_properties(self):
        importlib.reload(cfg_module)
        s = cfg_module.settings
        assert s.input_height == 380
        assert s.input_width == 380


class TestEnvironmentOverrides:
    """Environment variables correctly change settings."""

    def test_batch_size_from_env(self, monkeypatch):
        monkeypatch.setenv("IR_BATCH_SIZE", "16")
        importlib.reload(cfg_module)
        s = cfg_module.settings
        assert s.BATCH_SIZE == 16

    def test_confidence_from_env(self, monkeypatch):
        monkeypatch.setenv("IR_CONFIDENCE_THRESHOLD", "0.75")
        importlib.reload(cfg_module)
        s = cfg_module.settings
        assert s.CONFIDENCE_THRESHOLD == 0.75

    def test_backend_from_env(self, monkeypatch):
        monkeypatch.setenv("IR_INFERENCE_BACKEND", "tensorrt")
        importlib.reload(cfg_module)
        s = cfg_module.settings
        assert s.INFERENCE_BACKEND == cfg_module.InferenceBackend.TENSORRT

    def test_invalid_backend_raises(self, monkeypatch):
        monkeypatch.setenv("IR_INFERENCE_BACKEND", "unknown_engine")
        with pytest.raises(ValueError):
            importlib.reload(cfg_module)
            _ = cfg_module.settings.INFERENCE_BACKEND

    def test_gpu_id_from_env(self, monkeypatch):
        monkeypatch.setenv("IR_GPU_ID", "2")
        importlib.reload(cfg_module)
        s = cfg_module.settings
        assert s.GPU_ID == 2

    def test_model_path_override(self, monkeypatch):
        monkeypatch.setenv("IR_MODEL_PATH", "/custom/path/model.pth")
        importlib.reload(cfg_module)
        s = cfg_module.settings
        assert s.MODEL_PATH == "/custom/path/model.pth"


class TestSettingsImmutability:
    """Settings dataclass is frozen."""

    def test_settings_are_frozen(self):
        importlib.reload(cfg_module)
        s = cfg_module.settings
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            s.BATCH_SIZE = 999  # type: ignore[misc]
