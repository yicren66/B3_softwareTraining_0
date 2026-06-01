FROM python:3.10-slim AS builder
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY services/reasoning-engine/requirements.txt /build/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r /build/requirements.txt

FROM python:3.10-slim
LABEL org.opencontainers.image.title="Jujube Reasoning Engine"
LABEL org.opencontainers.image.description="Prevention planning, QA, and role-differentiated pest management recommendations"
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN groupadd --system app && useradd --system --gid app --no-create-home app
WORKDIR /app
COPY services/reasoning-engine/src /app/services/reasoning-engine/src
COPY libs/common-py /app/libs/common-py
RUN pip install --no-cache-dir -e /app/libs/common-py && chown -R app:app /app
USER app
EXPOSE 8003 9003
ENV RE_KG_SERVICE_URL=http://knowledge-graph:8002 \
    RE_RECOGNITION_SERVICE_URL=http://image-recognition:8001 \
    RE_HTTP_PORT=8003 RE_GRPC_PORT=9003 RE_LOG_LEVEL=INFO
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=30s \
    CMD curl -sf http://localhost:${RE_HTTP_PORT}/health || exit 1
ENTRYPOINT ["python3", "-m", "services.reasoning-engine.src.main"]
