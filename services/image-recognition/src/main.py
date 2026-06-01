"""
Image Recognition Service — Entry Point

- FastAPI server for health checks and Prometheus metrics on HTTP port 8001.
- gRPC server for ImageRecognition service on port 9001.
- Loads all models with warmup on startup.
- Implements ClassifyImage and BatchClassify RPCs.
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from concurrent import futures
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import grpc
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from config import settings
from inference.engine import InferenceEngine
from inference.postprocess import format_recognition_result, result_summary
from models.model_loader import get_model_manager
from preprocessing.transforms import validate_and_open

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("image-recognition")

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

METRIC_PREFIX = "ir"

request_total = Counter(
    f"{METRIC_PREFIX}_requests_total",
    "Total inference requests",
    ["method", "status"],
)
request_latency = Histogram(
    f"{METRIC_PREFIX}_request_latency_seconds",
    "End-to-end request latency",
    ["method"],
)
model_status_gauge = Gauge(
    f"{METRIC_PREFIX}_model_loaded",
    "Whether each sub-model is currently loaded (1=yes)",
    ["model_name"],
)
gpu_memory_allocated = Gauge(
    f"{METRIC_PREFIX}_gpu_memory_allocated_mb",
    "GPU memory currently allocated (MB)",
)
gpu_memory_reserved = Gauge(
    f"{METRIC_PREFIX}_gpu_memory_reserved_mb",
    "GPU memory reserved by PyTorch caching allocator (MB)",
)
inference_errors = Counter(
    f"{METRIC_PREFIX}_inference_errors_total",
    "Total inference errors",
    ["error_type"],
)

# ---------------------------------------------------------------------------
# Inference engine (singleton)
# ---------------------------------------------------------------------------

engine = InferenceEngine()

# ---------------------------------------------------------------------------
# gRPC stub — in production import the generated *_pb2* modules.
# Here we use a lightweight stand-in so the service structure is complete
# without needing the protobuf compilation step.
# ---------------------------------------------------------------------------


class ImageRecognitionServicer:
    """gRPC servicer implementing the ImageRecognition service.

    Real implementation would subclass the generated
    image_recognition_pb2_grpc.ImageRecognitionServicer.
    """

    def ClassifyImage(self, request: Any, context: grpc.ServicerContext) -> Dict[str, Any]:
        """Single-image classification RPC."""
        start = time.time()
        try:
            image_bytes = getattr(request, "image", None)
            if image_bytes is None:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Missing 'image' field.")
                return {}  # unreachable

            result = engine.run_full_pipeline(image_bytes)
            data = format_recognition_result(result)

            elapsed = time.time() - start
            request_latency.labels(method="ClassifyImage").observe(elapsed)
            request_total.labels(method="ClassifyImage", status="success").inc()
            logger.info("ClassifyImage: %s", result_summary(data))
            return data

        except Exception as exc:
            request_total.labels(method="ClassifyImage", status="error").inc()
            inference_errors.labels(error_type=type(exc).__name__).inc()
            logger.exception("ClassifyImage failed.")
            context.abort(grpc.StatusCode.INTERNAL, str(exc))
            return {}  # unreachable

    def BatchClassify(self, request: Any, context: grpc.ServicerContext) -> Dict[str, Any]:
        """Batch classification RPC."""
        start = time.time()
        try:
            images: List[bytes] = list(getattr(request, "images", []))
            if not images:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Empty 'images' list.")
                return {}

            results: List[Dict[str, Any]] = []
            for img_bytes in images:
                result = engine.run_full_pipeline(img_bytes)
                results.append(format_recognition_result(result))

            elapsed = time.time() - start
            request_latency.labels(method="BatchClassify").observe(elapsed)
            request_total.labels(method="BatchClassify", status="success").inc()
            logger.info("BatchClassify: %d images processed in %.2f s", len(images), elapsed)
            return {"results": results, "count": len(results)}

        except Exception as exc:
            request_total.labels(method="BatchClassify", status="error").inc()
            inference_errors.labels(error_type=type(exc).__name__).inc()
            logger.exception("BatchClassify failed.")
            context.abort(grpc.StatusCode.INTERNAL, str(exc))
            return {}  # unreachable

    def HealthCheck(self, request: Any, context: grpc.ServicerContext) -> Dict[str, Any]:
        """gRPC health check with model status."""
        return get_health_status()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    # Startup
    logger.info("=" * 60)
    logger.info("Starting Image Recognition Service")
    logger.info("  Backend: %s", settings.INFERENCE_BACKEND.value)
    logger.info("  Device:  %s", settings.device)
    logger.info("  HTTP:    port %d", settings.HTTP_PORT)
    logger.info("  gRPC:    port %d", settings.GRPC_PORT)
    logger.info("=" * 60)

    # Load models + warmup
    try:
        engine.load_models()
        logger.info("Models loaded and warmed up successfully.")
    except Exception:
        logger.critical("Startup model loading failed — service will start degraded.")
        model_status_gauge.labels(model_name="classifier").set(0)
        model_status_gauge.labels(model_name="detector").set(0)
        model_status_gauge.labels(model_name="severity").set(0)

    # Start hot-reload watcher
    get_model_manager().start_watcher()

    # Start gRPC server in background thread
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # In production: image_recognition_pb2_grpc.add_ImageRecognitionServicer_to_server(
    #     ImageRecognitionServicer(), grpc_server
    # )
    grpc_server.add_insecure_port(f"[::]:{settings.GRPC_PORT}")
    grpc_server.start()
    logger.info("gRPC server listening on port %d", settings.GRPC_PORT)

    # Update model status metrics
    _update_model_metrics()

    yield  # --- app running ---

    # Shutdown
    logger.info("Shutting down...")
    get_model_manager().stop_watcher()
    grpc_server.stop(grace=5).wait()
    logger.info("Service stopped.")


app = FastAPI(
    title="Image Recognition Service",
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Health / Metrics endpoints
# ---------------------------------------------------------------------------


def get_health_status() -> Dict[str, Any]:
    """Build health-check payload."""
    model_status = get_model_manager().status()
    all_loaded = all(v["loaded"] for v in model_status.values())
    gpu = get_model_manager().gpu_memory_info()
    return {
        "status": "healthy" if all_loaded else "degraded",
        "models": model_status,
        "backend": settings.INFERENCE_BACKEND.value,
        "device": settings.device,
        "gpu_memory_mb": gpu,
        "uptime_seconds": time.time() - _start_ts,
    }


_start_ts = time.time()


@app.get("/health")
async def health():
    """Health check with model status."""
    status = get_health_status()
    if status["status"] == "healthy":
        return JSONResponse(content=status)
    return JSONResponse(content=status, status_code=503)


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    if not settings.METRICS_ENABLED:
        raise HTTPException(status_code=404, detail="Metrics disabled.")
    _update_model_metrics()
    _update_gpu_metrics()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/ready")
async def ready():
    """Kubernetes-style readiness probe."""
    if engine.ready:
        return JSONResponse(content={"ready": True})
    return JSONResponse(content={"ready": False}, status_code=503)


@app.get("/live")
async def live():
    """Kubernetes-style liveness probe."""
    return JSONResponse(content={"live": True})


def _update_model_metrics() -> None:
    for name, info in get_model_manager().status().items():
        model_status_gauge.labels(model_name=name).set(1 if info["loaded"] else 0)


def _update_gpu_metrics() -> None:
    mem = get_model_manager().gpu_memory_info()
    gpu_memory_allocated.set(mem["allocated_mb"])
    gpu_memory_reserved.set(mem["reserved_mb"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logger.info("Launching FastAPI on port %d", settings.HTTP_PORT)
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.HTTP_PORT,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=False,
    )
