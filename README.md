# Jujube Platform

Multi-service AI platform for jujube (Chinese date) crop intelligence — disease/pest recognition, knowledge graph, reasoning engine, model management, and operational analytics.

## Architecture

```
                              ┌─────────────────────────────┐
                              │      Nginx / Ingress         │
                              │  (TLS termination, proxy)    │
                              └─────────────┬───────────────┘
                                            │
                              ┌─────────────▼───────────────┐
                              │        API Gateway           │
                              │     (Port 8000, gRPC)        │
                              └──┬──────┬──────┬──────┬─────┘
                                 │      │      │      │
        ┌────────────────────────┤      │      │      ├──────────────────┐
        │                        │      │      │                        │
  ┌─────▼──────┐  ┌──────▼─────┐ │ ┌────▼────┐ │ ┌────────▼──────┐ ┌───▼──────────┐
  │  Image     │  │ Knowledge  │ │ │Reasoning│ │ │   Model       │ │ Log & Stats  │
  │Recognition │  │  Graph     │ │ │ Engine  │ │ │ Management    │ │   Service    │
  │  :9001     │  │  :9002     │ │ │  :9003  │ │ │   :9004       │ │   :9005      │
  └─────┬──────┘  └──────┬─────┘ │ └────┬────┘ │ └───────┬───────┘ └───┬──────────┘
        │                │       │      │      │         │              │
        │         ┌──────▼───┐   │      │      │         │    ┌─────────▼──────┐
        │         │   Neo4j  │   │      │      │         │    │   PostgreSQL   │
        │         │   :7687  │   │      │      │         │    │     :5432      │
        │         └──────────┘   │      │      │         │    └────────────────┘
        │                        │      │      │         │
        └────────────────────────┴──────┴──────┴─────────┘
                           ┌───────▼──────┐
                           │    Redis 7   │
                           │    :6379     │
                           └──────────────┘
```

## Quick Start

### Prerequisites

- Docker & Docker Compose 2.x
- Python 3.11+
- Go 1.22+ (for gRPC services)
- buf (protobuf toolchain)
- (Optional) NVIDIA Container Toolkit for GPU acceleration

### 1. Clone and configure

```bash
git clone <repo-url> jujube-platform
cd jujube-platform

# Copy and edit environment
cp .env.example .env
# Edit .env — set strong passwords and a random JWT_SECRET_KEY
```

### 2. Start development stack

```bash
# Start all services
make dev-up

# Check health
curl http://localhost:8000/health

# Tail logs
make dev-logs
```

### 3. Run tests

```bash
# All tests
make test

# Unit tests only
make test-unit

# Integration tests (requires running services)
make test-integration
```

### 4. Generate proto code

```bash
make proto-gen
```

## Services

| Service             | Port  | Tech                 | Description                            |
|---------------------|-------|----------------------|----------------------------------------|
| api-gateway         | 8000  | Go / Gin            | Central API gateway, auth, routing     |
| image-recognition   | 9001  | Python / PyTorch     | Disease & pest image classification    |
| knowledge-graph     | 9002  | Python / Neo4j       | Knowledge graph queries & management   |
| reasoning-engine    | 9003  | Python / LLM         | Diagnostic reasoning & recommendations |
| model-management    | 9004  | Python               | ML model versioning & registry         |
| log-statistics      | 9005  | Go                    | Operational logging & statistics       |
| postgres            | 5432  | PostgreSQL 14         | Relational storage                     |
| neo4j               | 7687  | Neo4j 5 Enterprise    | Graph knowledge base                   |
| redis               | 6379  | Redis 7               | Cache & message broker                 |

## API Documentation

API documentation is served by the API Gateway at `/docs` (Swagger UI) and `/redoc` when running.

Key endpoints:

- `POST /api/v1/recognition/detect` — Run disease/pest detection on an image
- `GET  /api/v1/kg/search` — Search the knowledge graph
- `GET  /api/v1/kg/entity/:id` — Get entity details from the graph
- `POST /api/v1/model/upload` — Upload a new model artifact
- `GET  /api/v1/model/list` — List registered models
- `GET  /api/v1/stats/dashboard` — Get operational statistics

## Project Structure

```
.
├── services/
│   ├── api-gateway/          # Go — API Gateway
│   ├── image-recognition/    # Python — Image Recognition Service
│   ├── knowledge-graph/      # Python — Knowledge Graph Service
│   ├── reasoning-engine/     # Python — Reasoning Engine
│   ├── model-management/    # Python — Model Management
│   └── log-statistics/       # Go — Log & Statistics Service
├── pkg/                       # Shared Go packages
├── proto/                     # Protobuf definitions
├── deploy/
│   ├── kubernetes/            # K8s manifests
│   ├── docker/nginx/          # Nginx config
│   └── scripts/               # Init scripts
├── models_registry/           # Model artifacts (gitignored)
├── datasets/                  # Training datasets (gitignored)
├── tests/                     # Integration & E2E tests
├── docker-compose.yml         # Dev stack
├── Makefile                   # Build / deploy targets
└── pyproject.toml             # Python tooling config
```

## Deployment

### Kubernetes

```bash
# Deploy to staging
make deploy-staging

# Deploy to production
make deploy-prod
```

See `deploy/kubernetes/` for all manifests:
- `namespace.yaml` — jujube-platform namespace
- `configmap.yaml` — Non-secret configuration
- `secrets.yaml.example` — Secret template (copy and fill in)
- `ingress.yaml` — Nginx Ingress with TLS
- `api-gateway/deployment.yaml` — API Gateway (3 replicas, HPA)
- `image-recognition/deployment.yaml` — Image Recognition (2 replicas, GPU, HPA)
- `neo4j/statefulset.yaml` — Neo4j StatefulSet
- `postgresql/statefulset.yaml` — PostgreSQL StatefulSet

### Docker Compose (dev only)

```bash
docker compose up -d --build
docker compose --profile proxy up -d   # includes nginx reverse proxy
```

## Configuration

All configuration is managed through environment variables. See `.env.example` for the full list. Key variables:

| Variable              | Default                      | Description                |
|-----------------------|------------------------------|----------------------------|
| `DB_HOST`             | postgres                     | PostgreSQL host            |
| `NEO4J_URI`           | bolt://neo4j:7687           | Neo4j bolt endpoint        |
| `REDIS_HOST`           | redis                        | Redis host                 |
| `JWT_SECRET_KEY`      | (required)                   | JWT signing secret         |
| `API_GATEWAY_PORT`    | 8000                         | API Gateway port           |
| `LOG_LEVEL`           | INFO                         | Logging level              |
| `ENVIRONMENT`         | development                  | Runtime environment        |

## License

Proprietary — all rights reserved.
