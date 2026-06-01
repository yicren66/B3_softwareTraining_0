# =============================================================================
# API Gateway — Multi-stage Go build
# =============================================================================
FROM golang:1.22-alpine AS builder

RUN apk add --no-cache git ca-certificates

WORKDIR /build

# Cache dependencies
COPY services/api-gateway/go.mod services/api-gateway/go.sum* ./
RUN go mod download 2>/dev/null || true

# Build
COPY services/api-gateway/ ./
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /build/api-gateway ./cmd/server/

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM alpine:3.19

RUN apk add --no-cache ca-certificates curl tzdata

RUN addgroup -S app && adduser -S app -G app

WORKDIR /app
COPY --from=builder /build/api-gateway .

USER app
EXPOSE 8000 9000

ENV ENVIRONMENT=production \
    API_GATEWAY_PORT=8000 \
    LOG_LEVEL=info \
    JWT_SECRET_KEY=change-me-in-production \
    JWT_EXPIRE_HOURS=24

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=5s \
    CMD curl -sf http://localhost:${API_GATEWAY_PORT}/health || exit 1

ENTRYPOINT ["./api-gateway"]
