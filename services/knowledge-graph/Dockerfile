# =============================================================================
# Knowledge Graph Service — Multi-stage Dockerfile
# =============================================================================
#
# Build:
#   docker build -t jujube/knowledge-graph:latest \
#     -f services/knowledge-graph/Dockerfile .
#
# Run:
#   docker run -p 9002:9002 -p 8002:8002 \
#     -e KG_NEO4J_URI=bolt://neo4j:7687 \
#     -e KG_NEO4J_PASSWORD=changeme \
#     jujube/knowledge-graph:latest
# =============================================================================

FROM python:3.10-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Virtual env
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependencies in cache-friendly layer
COPY services/knowledge-graph/requirements.txt /build/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /build/requirements.txt

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM python:3.10-slim

LABEL org.opencontainers.image.title="Jujube Knowledge Graph Service"
LABEL org.opencontainers.image.description="Neo4j-backed knowledge graph search, recommendation & risk prediction for jujube pests/diseases"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy venv
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN groupadd --system app && useradd --system --gid app --no-create-home app

WORKDIR /app

# Application code
COPY services/knowledge-graph/src /app/services/knowledge-graph/src
COPY libs/common-py /app/libs/common-py

RUN pip install --no-cache-dir -e /app/libs/common-py

RUN chown -R app:app /app
USER app

EXPOSE 8002 9002

ENV KG_NEO4J_URI=bolt://neo4j:7687 \
    KG_NEO4J_USER=neo4j \
    KG_NEO4J_DATABASE=neo4j \
    KG_HTTP_PORT=8002 \
    KG_GRPC_PORT=9002 \
    KG_LOG_LEVEL=INFO

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
    CMD curl -sf http://localhost:${KG_HTTP_PORT}/health || exit 1

ENTRYPOINT ["python3", "-m", "services.knowledge-graph.src.main"]
