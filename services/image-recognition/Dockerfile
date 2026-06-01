# =============================================================================
# Image Recognition Service — Multi-stage Dockerfile
# =============================================================================
#
# Stages:
#   1. builder      — compile ONNX Runtime with CUDA, install Python deps
#   2. runtime      — slim production image with GPU support
#
# Build:
#   docker build -t jujube/image-recognition:latest \
#     -f services/image-recognition/Dockerfile .
#
# Run:
#   docker run --gpus all -p 9001:9001 -p 8001:8001 \
#     -v /path/to/models:/app/models_registry:ro \
#     jujube/image-recognition:latest
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — Builder
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Virtual env
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies in a cache-friendly layer
COPY services/image-recognition/requirements.txt /build/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /build/requirements.txt

# Build ONNX Runtime with CUDA (if needed — otherwise use prebuilt wheel)
RUN pip install --no-cache-dir \
    onnxruntime-gpu \
    onnx \
    pycuda

# Copy application source
COPY services/image-recognition/src /app/services/image-recognition/src
COPY libs/common-py /app/libs/common-py

# Install the common library in editable mode
RUN pip install --no-cache-dir -e /app/libs/common-py

# ---------------------------------------------------------------------------
# Stage 2 — Runtime
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

LABEL org.opencontainers.image.title="Jujube Image Recognition Service"
LABEL org.opencontainers.image.description="Disease/pest classification and small-target detection for jujube orchards"
LABEL org.opencontainers.image.version="0.1.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN groupadd --system app && useradd --system --gid app --no-create-home app

# Application directory
WORKDIR /app

# Copy application code
COPY --from=builder --chown=app:app /app/services/image-recognition/src /app/services/image-recognition/src
COPY --from=builder --chown=app:app /app/libs/common-py /app/libs/common-py

# Model registry mount point
RUN mkdir -p /app/models_registry && chown -R app:app /app

# Health check port (HTTP) and gRPC port
EXPOSE 8001 9001

# Drop to non-root
USER app

# Environment defaults (overridable at runtime)
ENV IR_MODEL_PATH=/app/models_registry/classifier/jujube_classifier.pth \
    IR_ONNX_MODEL_PATH=/app/models_registry/classifier/jujube_classifier.onnx \
    IR_TENSORRT_ENGINE_PATH=/app/models_registry/classifier/jujube_classifier.trt \
    IR_DETECTOR_MODEL_PATH=/app/models_registry/detector/small_target_fpn.pth \
    IR_DETECTOR_ONNX_PATH=/app/models_registry/detector/small_target_fpn.onnx \
    IR_DETECTOR_TRT_PATH=/app/models_registry/detector/small_target_fpn.trt \
    IR_SEVERITY_MODEL_PATH=/app/models_registry/severity/severity_classifier.pth \
    IR_SEVERITY_ONNX_PATH=/app/models_registry/severity/severity_classifier.onnx \
    IR_SEVERITY_TRT_PATH=/app/models_registry/severity/severity_classifier.trt \
    IR_USE_GPU=true \
    IR_GPU_ID=0 \
    IR_INFERENCE_BACKEND=onnxruntime \
    IR_HTTP_PORT=8001 \
    IR_GRPC_PORT=9001 \
    IR_CONFIDENCE_THRESHOLD=0.5 \
    IR_MAX_BATCH_SIZE=32 \
    IR_LOG_LEVEL=INFO

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
    CMD curl -sf http://localhost:${IR_HTTP_PORT}/health || exit 1

ENTRYPOINT ["python3", "-m", "services.image-recognition.src.main"]
