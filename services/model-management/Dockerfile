FROM python:3.10-slim AS builder
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY services/model-management/requirements.txt /build/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r /build/requirements.txt

FROM python:3.10-slim
LABEL org.opencontainers.image.title="Jujube Model Management Service"
LABEL org.opencontainers.image.description="ML model registry, version management, deployment & A/B testing"
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN groupadd --system app && useradd --system --gid app --no-create-home app
WORKDIR /app
RUN mkdir -p /models && chown -R app:app /app /models
COPY services/model-management/src /app/services/model-management/src
COPY libs/common-py /app/libs/common-py
RUN pip install --no-cache-dir -e /app/libs/common-py && chown -R app:app /app
USER app
EXPOSE 8004 9004
ENV MM_MODEL_REGISTRY_ROOT=/models \
    MM_DB_HOST=postgres \
    MM_IMAGE_RECOGNITION_URL=http://image-recognition:8001 \
    MM_HTTP_PORT=8004 MM_GRPC_PORT=9004 MM_LOG_LEVEL=INFO
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=15s \
    CMD curl -sf http://localhost:${MM_HTTP_PORT}/health || exit 1
ENTRYPOINT ["python3", "-m", "services.model-management.src.main"]
