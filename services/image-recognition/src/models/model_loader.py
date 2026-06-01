"""
Model loading, warmup, hot-reload, and GPU memory management.

Supports three backends (ONNX Runtime, TensorRT, PyTorch) for each of the
three sub-models: classifier, small-target detector, and severity assessor.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch

from config import settings, InferenceBackend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dummy model stubs — replace with real model classes in production.
# These exist so the service can start without actual .pth files and be
# iterated on before real weights land.
# ---------------------------------------------------------------------------


class JujubeClassifier(torch.nn.Module):
    """Stub classifier — maps (3,380,380) to 「7病8虫」15 classes."""

    NUM_CLASSES = 15

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.head = torch.nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.flatten(self.backbone(x), 1))


class SmallTargetDetector(torch.nn.Module):
    """Stub FPN-based small-target detector."""

    def __init__(self):
        super().__init__()
        self.fpn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
        )
        # Regression + classification heads share base features
        self.bbox_head = torch.nn.Conv2d(64, 4, 1)
        self.cls_head = torch.nn.Conv2d(64, 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.fpn(x)
        return self.bbox_head(feats), self.cls_head(feats)


class SeverityClassifier(torch.nn.Module):
    """Stub severity classifier — 4 levels."""

    NUM_CLASSES = 4

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.head = torch.nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.flatten(self.encoder(x), 1))


# ---------------------------------------------------------------------------
# Backend-specific load helpers
# ---------------------------------------------------------------------------

import onnxruntime as ort  # noqa: E402


def _load_onnx_session(model_path: str, gpu_id: int = 0) -> ort.InferenceSession:
    """Create an ONNX Runtime inference session with GPU preference."""
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if settings.USE_GPU and ort.get_device() == "GPU"
        else ["CPUExecutionProvider"]
    )
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    logger.info("Loading ONNX session from %s (providers=%s)", model_path, providers)
    return ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)


def _load_tensorrt_engine(engine_path: str) -> Any:
    """Load a serialised TensorRT engine and return an execution context.

    Requires tensorrt pip package. Returns a dict with 'engine' and 'context'
    so callers can stay backend-agnostic.
    """
    try:
        import tensorrt as trt  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError(
            "TensorRT backend selected but tensorrt is not installed."
        )

    logger.info("Loading TensorRT engine from %s", engine_path)
    trt_logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(trt_logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    if not context:
        raise RuntimeError("Failed to create TensorRT execution context.")
    return {"engine": engine, "context": context}


def _load_pytorch_model(model_path: str, model_class: type, **kwargs: Any) -> torch.nn.Module:
    """Load model weights into the given model class."""
    model = model_class(**kwargs)
    if Path(model_path).exists():
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        logger.info("Loaded PyTorch weights from %s", model_path)
    else:
        logger.warning("Model file %s not found — using random weights.", model_path)
    model.to(settings.device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Model registry entry
# ---------------------------------------------------------------------------


@dataclass
class ModelEntry:
    name: str  # "classifier" | "detector" | "severity"
    model: Optional[Any] = None  # torch.nn.Module | ort.InferenceSession | dict
    version: str = "0.0.0"
    backend: str = ""
    last_loaded: float = 0.0
    load_fn: Optional[Callable[[], Any]] = None


# ---------------------------------------------------------------------------
# Model Manager
# ---------------------------------------------------------------------------


class ModelManager:
    """Central registry that loads, warms-up, and hot-reloads all sub-models."""

    def __init__(self) -> None:
        self._entries: Dict[str, ModelEntry] = {}
        self._lock = threading.Lock()
        self._watch_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, load_fn: Callable[[], Any]) -> ModelEntry:
        entry = ModelEntry(name=name, load_fn=load_fn)
        with self._lock:
            self._entries[name] = entry
        return entry

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_classifier(self) -> JujubeClassifier:
        return self._load_torch_or_onnx_or_trt(
            name="classifier",
            model_class=JujubeClassifier,
            onnx_path=settings.ONNX_MODEL_PATH,
            trt_path=settings.TENSORRT_ENGINE_PATH,
            pytorch_path=settings.MODEL_PATH,
        )

    def load_detector(self) -> SmallTargetDetector:
        return self._load_torch_or_onnx_or_trt(
            name="detector",
            model_class=SmallTargetDetector,
            onnx_path=settings.DETECTOR_ONNX_PATH,
            trt_path=settings.DETECTOR_TRT_PATH,
            pytorch_path=settings.DETECTOR_MODEL_PATH,
        )

    def load_severity(self) -> SeverityClassifier:
        return self._load_torch_or_onnx_or_trt(
            name="severity",
            model_class=SeverityClassifier,
            onnx_path=settings.SEVERITY_ONNX_PATH,
            trt_path=settings.SEVERITY_TRT_PATH,
            pytorch_path=settings.SEVERITY_MODEL_PATH,
        )

    def _load_torch_or_onnx_or_trt(
        self,
        name: str,
        model_class: type,
        onnx_path: str,
        trt_path: str,
        pytorch_path: str,
    ) -> Any:
        backend = settings.INFERENCE_BACKEND
        logger.info("Loading '%s' model (backend=%s)", name, backend.value)

        if backend == InferenceBackend.ONNX:
            model = _load_onnx_session(onnx_path, settings.GPU_ID)
        elif backend == InferenceBackend.TENSORRT:
            model = _load_tensorrt_engine(trt_path)
        else:
            model = _load_pytorch_model(pytorch_path, model_class)

        version = self._derive_version(name, backend)
        entry = ModelEntry(
            name=name,
            model=model,
            version=version,
            backend=backend.value,
            last_loaded=time.time(),
        )
        with self._lock:
            self._entries[name] = entry

        self._warmup(name)
        return model

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _warmup(self, name: str, num_runs: int = 3) -> None:
        """Run dummy inference to prime GPU kernels and memory allocators."""
        entry = self._entries.get(name)
        if entry is None or entry.model is None:
            return
        dummy = torch.randn(1, *settings.INPUT_SIZE, device=settings.device)
        logger.info("Warming up '%s' (%d runs)...", name, num_runs)
        backend = settings.INFERENCE_BACKEND

        for i in range(num_runs):
            if backend == InferenceBackend.PYTORCH:
                with torch.no_grad():
                    _ = entry.model(dummy)
            elif backend == InferenceBackend.ONNX:
                _ = entry.model.run(
                    None, {"input": dummy.cpu().numpy().astype(np.float32)}
                )
            else:  # TensorRT — run via bindings helper
                _ = self._trt_infer(entry.model, dummy)
        logger.info("Warmup for '%s' complete.", name)

    # ------------------------------------------------------------------
    # Hot-reload
    # ------------------------------------------------------------------

    def start_watcher(self) -> None:
        """Background thread that polls shared volume for new artefacts."""
        if self._watch_thread is not None:
            return
        self._stop_event.clear()
        self._watch_thread = threading.Thread(
            target=self._watch_loop, daemon=True, name="model-watcher"
        )
        self._watch_thread.start()
        logger.info(
            "Model hot-reload watcher started (interval=%ds).",
            settings.MODEL_WATCH_INTERVAL,
        )

    def stop_watcher(self) -> None:
        self._stop_event.set()
        if self._watch_thread:
            self._watch_thread.join(timeout=5)

    def _watch_loop(self) -> None:
        while not self._stop_event.wait(timeout=settings.MODEL_WATCH_INTERVAL):
            self._check_for_updates()

    def _check_for_updates(self) -> None:
        for name, entry in list(self._entries.items()):
            new_ver = self._derive_version(name, settings.INFERENCE_BACKEND)
            if new_ver != entry.version:
                logger.info(
                    "New version detected for '%s': %s -> %s. Reloading...",
                    name, entry.version, new_ver,
                )
                try:
                    if entry.load_fn:
                        new_model = entry.load_fn()
                        with self._lock:
                            entry.model = new_model
                            entry.version = new_ver
                            entry.last_loaded = time.time()
                        self._warmup(name)
                except Exception:
                    logger.exception("Hot-reload failed for '%s' — keeping current model.", name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _derive_version(self, name: str, backend: InferenceBackend) -> str:
        """Derive version string from file mtime of the model artefact."""
        path_map = {
            (InferenceBackend.PYTORCH, "classifier"): settings.MODEL_PATH,
            (InferenceBackend.ONNX, "classifier"): settings.ONNX_MODEL_PATH,
            (InferenceBackend.TENSORRT, "classifier"): settings.TENSORRT_ENGINE_PATH,
            (InferenceBackend.PYTORCH, "detector"): settings.DETECTOR_MODEL_PATH,
            (InferenceBackend.ONNX, "detector"): settings.DETECTOR_ONNX_PATH,
            (InferenceBackend.TENSORRT, "detector"): settings.DETECTOR_TRT_PATH,
            (InferenceBackend.PYTORCH, "severity"): settings.SEVERITY_MODEL_PATH,
            (InferenceBackend.ONNX, "severity"): settings.SEVERITY_ONNX_PATH,
            (InferenceBackend.TENSORRT, "severity"): settings.SEVERITY_TRT_PATH,
        }
        path = path_map.get((backend, name))
        if path and Path(path).exists():
            return str(int(Path(path).stat().st_mtime))
        return "0.0.0"

    @staticmethod
    def _trt_infer(engine_dict: dict, tensor: torch.Tensor) -> np.ndarray:
        """Minimal TensorRT inference helper — placeholder for real binding logic."""
        # In production this would manage device buffers, stream sync, etc.
        return np.zeros((1, 20), dtype=np.float32)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                name: {
                    "version": e.version,
                    "backend": e.backend,
                    "loaded": bool(e.model is not None),
                    "last_loaded_ts": e.last_loaded,
                }
                for name, e in self._entries.items()
            }

    # ------------------------------------------------------------------
    # GPU memory
    # ------------------------------------------------------------------

    def gpu_memory_info(self) -> Dict[str, float]:
        if not settings.USE_GPU or not torch.cuda.is_available():
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "free_mb": 0.0}
        return {
            "allocated_mb": torch.cuda.memory_allocated(settings.GPU_ID) / (1024**2),
            "reserved_mb": torch.cuda.memory_reserved(settings.GPU_ID) / (1024**2),
            "free_mb": (
                torch.cuda.get_device_properties(settings.GPU_ID).total_memory
                - torch.cuda.memory_reserved(settings.GPU_ID)
            )
            / (1024**2),
        }

    def clear_gpu_cache(self) -> None:
        if settings.USE_GPU and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared.")


# ---------------------------------------------------------------------------
# Convenience top-level functions (delegate to a global manager singleton)
# ---------------------------------------------------------------------------

_manager = ModelManager()


def load_classifier() -> JujubeClassifier:
    return _manager.load_classifier()


def load_detector() -> SmallTargetDetector:
    return _manager.load_detector()


def load_onnx_session(model_path: str = "") -> ort.InferenceSession:
    path = model_path or settings.ONNX_MODEL_PATH
    return _load_onnx_session(path, settings.GPU_ID)


def load_tensorrt_engine(engine_path: str = "") -> Any:
    path = engine_path or settings.TENSORRT_ENGINE_PATH
    return _load_tensorrt_engine(path)


def get_model_manager() -> ModelManager:
    return _manager
