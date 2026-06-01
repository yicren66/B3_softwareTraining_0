"""
Image Recognition Service — Configuration

All paths, model parameters, and runtime settings in one place.
Values can be overridden via environment variables (prefix IR_).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env(key: str, default: str) -> str:
    return os.environ.get(f"IR_{key}", default)


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(f"IR_{key}")
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(f"IR_{key}")
    return int(val) if val is not None else default


def _env_float(key: str, default: float) -> float:
    val = os.environ.get(f"IR_{key}")
    return float(val) if val is not None else default


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


class InferenceBackend(str, Enum):
    ONNX = "onnxruntime"
    TENSORRT = "tensorrt"
    PYTORCH = "pytorch"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    # ---- Model artefacts ----
    MODEL_PATH: str = _env("MODEL_PATH", "/models/classifier/jujube_classifier.pth")
    ONNX_MODEL_PATH: str = _env("ONNX_MODEL_PATH", "/models/classifier/jujube_classifier.onnx")
    TENSORRT_ENGINE_PATH: str = _env("TENSORRT_ENGINE_PATH", "/models/classifier/jujube_classifier.trt")

    DETECTOR_MODEL_PATH: str = _env("DETECTOR_MODEL_PATH", "/models/detector/small_target_fpn.pth")
    DETECTOR_ONNX_PATH: str = _env("DETECTOR_ONNX_PATH", "/models/detector/small_target_fpn.onnx")
    DETECTOR_TRT_PATH: str = _env("DETECTOR_TRT_PATH", "/models/detector/small_target_fpn.trt")

    SEVERITY_MODEL_PATH: str = _env("SEVERITY_MODEL_PATH", "/models/severity/severity_classifier.pth")
    SEVERITY_ONNX_PATH: str = _env("SEVERITY_ONNX_PATH", "/models/severity/severity_classifier.onnx")
    SEVERITY_TRT_PATH: str = _env("SEVERITY_TRT_PATH", "/models/severity/severity_classifier.trt")

    # ---- Input shape ----
    INPUT_SIZE: Tuple[int, int, int] = (3, 380, 380)

    # ---- Inference ----
    BATCH_SIZE: int = _env_int("BATCH_SIZE", 8)
    MAX_BATCH_SIZE: int = _env_int("MAX_BATCH_SIZE", 32)
    CONFIDENCE_THRESHOLD: float = _env_float("CONFIDENCE_THRESHOLD", 0.5)
    INFERENCE_BACKEND: InferenceBackend = InferenceBackend(
        _env("INFERENCE_BACKEND", "onnxruntime")
    )

    # ---- GPU ----
    USE_GPU: bool = _env_bool("USE_GPU", True)
    GPU_ID: int = _env_int("GPU_ID", 0)

    # ---- Post-processing ----
    TEMPERATURE: float = _env_float("TEMPERATURE", 1.0)  # temperature scaling for calibration

    # ---- Serving ----
    HTTP_PORT: int = _env_int("HTTP_PORT", 8001)
    GRPC_PORT: int = _env_int("GRPC_PORT", 9001)
    METRICS_ENABLED: bool = _env_bool("METRICS_ENABLED", True)

    # ---- Hot-reload ----
    MODEL_WATCH_INTERVAL: int = _env_int("MODEL_WATCH_INTERVAL", 30)  # seconds
    MODEL_SHARED_VOLUME: str = _env("MODEL_SHARED_VOLUME", "/models")

    # ---- Logging ----
    LOG_LEVEL: str = _env("LOG_LEVEL", "INFO")

    @property
    def device(self) -> str:
        if self.USE_GPU:
            return f"cuda:{self.GPU_ID}"
        return "cpu"

    @property
    def input_height(self) -> int:
        return self.INPUT_SIZE[1]

    @property
    def input_width(self) -> int:
        return self.INPUT_SIZE[2]


# Singleton
settings = Settings()
