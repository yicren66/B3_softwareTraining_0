"""
Inference engine wrapping ONNX / TensorRT / PyTorch backends.

Orchestrates the full pipeline:
  preprocess → classify → detect small targets → assess severity → format.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from config import settings, InferenceBackend
from models.model_loader import (
    get_model_manager,
    load_classifier,
    load_detector,
    JujubeClassifier,
    SeverityClassifier,
    SmallTargetDetector,
)
from preprocessing.transforms import (
    batch_bytes_to_tensors,
    bytes_to_classifier_tensor,
    preprocess_for_classifier,
    preprocess_for_detector,
    preprocess_for_severity,
    validate_and_open,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


@dataclass
class SmallTarget:
    """A detected small target (pest / spot) in the image."""

    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalised [0,1]
    class_id: int
    confidence: float
    class_name: str = ""


@dataclass
class RecognitionResult:
    """Aggregated result returned by the full pipeline."""

    class_idx: int
    confidence: float
    class_probs: List[float]
    severity_idx: int
    severity_probs: List[float]
    small_targets: List[SmallTarget] = field(default_factory=list)
    latency_ms: float = 0.0
    preprocess_ms: float = 0.0
    classify_ms: float = 0.0
    detect_ms: float = 0.0
    severity_ms: float = 0.0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class InferenceEngine:
    """Unified inference across backends with latency tracking and fallback."""

    def __init__(self) -> None:
        self._backend = settings.INFERENCE_BACKEND
        self._model_mgr = get_model_manager()
        self._classifier: Optional[Any] = None
        self._detector: Optional[Any] = None
        self._severity: Optional[Any] = None
        self._ready = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Load (or reload) all sub-models."""
        try:
            self._classifier = load_classifier()
            self._detector = load_detector()
            self._severity = self._model_mgr.load_severity()
            self._ready = True
            logger.info("All models loaded successfully.")
        except Exception:
            logger.exception("Model loading failed.")
            self._ready = False
            raise

    @property
    def ready(self) -> bool:
        return self._ready

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(
        self, image_tensor: torch.Tensor
    ) -> Tuple[int, float, List[float]]:
        """Run classification on a preprocessed tensor.

        Args:
            image_tensor: (1, 3, 380, 380) on the correct device.

        Returns:
            (class_idx, confidence, class_probs) — class_probs is the full
            softmax vector (list of floats).
        """
        t0 = time.perf_counter()
        probs = self._forward_classifier(image_tensor)
        idx = int(probs.argmax())
        conf = float(probs.max())
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug("Classification: idx=%d conf=%.3f (%.2f ms)", idx, conf, elapsed)
        return idx, conf, probs.tolist()

    # ------------------------------------------------------------------
    # Small-target detection
    # ------------------------------------------------------------------

    def detect_small_targets(
        self, image_tensor: torch.Tensor
    ) -> List[SmallTarget]:
        """Run the FPN small-target detector.

        Args:
            image_tensor: (1, 3, 380, 380) on the correct device.

        Returns:
            List of SmallTarget with confidences above CONFIDENCE_THRESHOLD.
        """
        t0 = time.perf_counter()
        boxes, scores = self._forward_detector(image_tensor)
        # boxes: (1, 4, H', W'), scores: (1, 2, H', W')
        # Simple thresholded extraction (replace with NMS in production)
        targets: List[SmallTarget] = []
        score_map = torch.softmax(scores, dim=1)[0, 1]  # foreground channel
        above = score_map > settings.CONFIDENCE_THRESHOLD
        if above.any():
            ys, xs = above.nonzero(as_tuple=True)
            for y, x in zip(ys.tolist(), xs.tolist()):
                bx = boxes[0, :, y, x].tolist()
                targets.append(
                    SmallTarget(
                        bbox=tuple(float(v) for v in bx),
                        class_id=1,
                        confidence=float(score_map[y, x]),
                    )
                )
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug("Detection: %d targets found (%.2f ms)", len(targets), elapsed)
        return targets

    # ------------------------------------------------------------------
    # Severity assessment
    # ------------------------------------------------------------------

    def assess_severity(
        self, image_tensor: torch.Tensor
    ) -> Tuple[int, List[float]]:
        """Assess disease severity level.

        Returns:
            (severity_idx, severity_probs) — 4 classes.
        """
        t0 = time.perf_counter()
        probs = self._forward_severity(image_tensor)
        idx = int(probs.argmax())
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug("Severity: idx=%d (%.2f ms)", idx, elapsed)
        return idx, probs.tolist()

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(self, image_bytes: bytes) -> RecognitionResult:
        """End-to-end: bytes → RecognitionResult.

        Steps:
          1. Validate & preprocess
          2. Classify
          3. Detect small targets
          4. Assess severity
          5. Package result
        """
        t_total = time.perf_counter()

        # Step 1 — Preprocess
        t_pre = time.perf_counter()
        img = validate_and_open(image_bytes)
        cls_tensor = preprocess_for_classifier(img).to(settings.device)
        det_tensor = preprocess_for_detector(img).to(settings.device)
        sev_tensor = preprocess_for_severity(img).to(settings.device)
        preprocess_ms = (time.perf_counter() - t_pre) * 1000

        # Step 2 — Classify
        class_idx, confidence, class_probs = self.classify(cls_tensor)
        classify_ms = (time.perf_counter() - t_pre - preprocess_ms / 1000) * 1000

        # Step 3 — Detect
        t_det = time.perf_counter()
        targets = self.detect_small_targets(det_tensor)
        detect_ms = (time.perf_counter() - t_det) * 1000

        # Step 4 — Severity
        sev_idx, sev_probs = self.assess_severity(sev_tensor)
        severity_ms = (time.perf_counter() - t_det - detect_ms / 1000) * 1000

        total_ms = (time.perf_counter() - t_total) * 1000

        return RecognitionResult(
            class_idx=class_idx,
            confidence=confidence,
            class_probs=class_probs,
            severity_idx=sev_idx,
            severity_probs=sev_probs,
            small_targets=targets,
            latency_ms=total_ms,
            preprocess_ms=preprocess_ms,
            classify_ms=classify_ms,
            detect_ms=detect_ms,
            severity_ms=severity_ms,
        )

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def batch_classify(
        self, image_batches: List[bytes]
    ) -> List[Tuple[int, float, List[float]]]:
        """Classify multiple images in a single forward pass.

        Dynamically batches up to MAX_BATCH_SIZE; if more images are
        provided they are processed in chunks.
        """
        results: List[Tuple[int, float, List[float]]] = []
        for i in range(0, len(image_batches), settings.MAX_BATCH_SIZE):
            chunk = image_batches[i : i + settings.MAX_BATCH_SIZE]
            batch_tensor, _ = batch_bytes_to_tensors(chunk)
            batch_tensor = batch_tensor.to(settings.device)
            probs = self._forward_classifier(batch_tensor)  # (B, C)
            for b in range(probs.size(0)):
                p = probs[b]
                idx = int(p.argmax())
                conf = float(p.max())
                results.append((idx, conf, p.tolist()))
        return results

    # ------------------------------------------------------------------
    # Backend-agnostic forward helpers
    # ------------------------------------------------------------------

    def _forward_classifier(self, x: torch.Tensor) -> torch.Tensor:
        if self._backend == InferenceBackend.PYTORCH:
            with torch.no_grad():
                logits = self._classifier(x)
        elif self._backend == InferenceBackend.ONNX:
            ort_inputs = {"input": x.cpu().numpy().astype(np.float32)}
            logits_np = self._classifier.run(None, ort_inputs)[0]
            logits = torch.from_numpy(logits_np)
        else:  # TensorRT
            logits = self._trt_forward(self._classifier, x)
        return torch.softmax(logits, dim=-1)

    def _forward_detector(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._backend == InferenceBackend.PYTORCH:
            with torch.no_grad():
                return self._detector(x)
        elif self._backend == InferenceBackend.ONNX:
            ort_inputs = {"input": x.cpu().numpy().astype(np.float32)}
            out = self._detector.run(None, ort_inputs)
            return torch.from_numpy(out[0]), torch.from_numpy(out[1])
        else:
            return self._trt_forward(self._detector, x), self._trt_forward(
                self._detector, x
            )

    def _forward_severity(self, x: torch.Tensor) -> torch.Tensor:
        if self._backend == InferenceBackend.PYTORCH:
            with torch.no_grad():
                logits = self._severity(x)
        elif self._backend == InferenceBackend.ONNX:
            ort_inputs = {"input": x.cpu().numpy().astype(np.float32)}
            logits_np = self._severity.run(None, ort_inputs)[0]
            logits = torch.from_numpy(logits_np)
        else:
            logits = self._trt_forward(self._severity, x)
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def _trt_forward(engine_dict: dict, x: torch.Tensor) -> torch.Tensor:
        """Placeholder TensorRT forward — real impl binds to device pointers."""
        # In production: populate binding buffers, execute_async_v2, sync stream
        B = x.size(0)
        return torch.randn(B, 20)  # stub

    # ------------------------------------------------------------------
    # GPU / CPU fallback
    # ------------------------------------------------------------------

    def fallback_to_cpu(self) -> None:
        """Attempt to move inference to CPU after a GPU error."""
        logger.warning("Attempting CPU fallback for inference.")
        try:
            if self._classifier is not None and hasattr(self._classifier, "cpu"):
                self._classifier.cpu()
            if self._detector is not None and hasattr(self._detector, "cpu"):
                self._detector.cpu()
            if self._severity is not None and hasattr(self._severity, "cpu"):
                self._severity.cpu()
            self._ready = True
        except Exception:
            logger.exception("CPU fallback failed.")
            self._ready = False
