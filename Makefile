# =============================================================================
# Jujube Platform — Makefile
# =============================================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help

# --- Docker image registry ---
REGISTRY ?= docker.io/jujube
IMAGE_TAG ?= latest
DOCKER_COMPOSE := docker compose

# --- Proto ---
PROTO_DIR := proto
PROTO_OUT := pkg/pb

# --- Test flags ---
TEST_FLAGS ?= -v -race -count=1
COVER_FLAGS ?= -coverprofile=coverage.out -covermode=atomic

# =============================================================================
# Help
# =============================================================================
.PHONY: help
help: ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} \
	     /^[a-zA-Z_-]+:.*##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# =============================================================================
# Protocol Buffers
# =============================================================================
.PHONY: proto-gen
proto-gen: ## Generate gRPC / protobuf code
	@echo "Generating protobuf code..."
	buf generate $(PROTO_DIR)
	@echo "Proto generation complete."

# =============================================================================
# Development (Docker Compose)
# =============================================================================
.PHONY: dev-up
dev-up: ## Start all services with docker compose
	$(DOCKER_COMPOSE) up -d --build

.PHONY: dev-down
dev-down: ## Stop all services and remove containers/networks
	$(DOCKER_COMPOSE) down -v --remove-orphans

.PHONY: dev-logs
dev-logs: ## Tail logs from all services
	$(DOCKER_COMPOSE) logs -f

.PHONY: dev-restart
dev-restart: dev-down dev-up ## Restart all services

# =============================================================================
# Testing
# =============================================================================
.PHONY: test
test: test-unit test-integration ## Run all tests (unit + integration)

.PHONY: test-unit
test-unit: ## Run unit tests
	@echo "Running unit tests..."
	go test $(TEST_FLAGS) -short ./...

.PHONY: test-integration
test-integration: ## Run integration tests (requires docker)
	@echo "Running integration tests..."
	go test $(TEST_FLAGS) -tags=integration ./test/integration/...

.PHONY: test-coverage
test-coverage: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	go test $(TEST_FLAGS) $(COVER_FLAGS) ./...
	go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

# =============================================================================
# Code Quality
# =============================================================================
.PHONY: lint
lint: ## Run all linters
	@echo "Running linters..."
	golangci-lint run ./...
	python -m flake8 services/ --count --statistics

.PHONY: format
format: ## Auto-format code
	@echo "Formatting Go code..."
	gofmt -s -w .
	goimports -w .
	@echo "Formatting Python code..."
	python -m black services/ tests/
	python -m isort services/ tests/

# =============================================================================
# Docker Images
# =============================================================================
.PHONY: build-images
build-images: ## Build all Docker images
	$(DOCKER_COMPOSE) build

.PHONY: push-images
push-images: ## Build, tag, and push images to registry
	$(DOCKER_COMPOSE) build
	for svc in api-gateway image-recognition knowledge-graph reasoning-engine model-management log-statistics; do \
		docker tag jujube/$$svc:$(IMAGE_TAG) $(REGISTRY)/$$svc:$(IMAGE_TAG); \
		docker push $(REGISTRY)/$$svc:$(IMAGE_TAG); \
	done

# =============================================================================
# Deploy
# =============================================================================
.PHONY: deploy-staging
deploy-staging: ## Deploy to Kubernetes staging
	@echo "Deploying to staging..."
	kubectl apply -f deploy/kubernetes/namespace.yaml
	kubectl apply -f deploy/kubernetes/configmap.yaml
	kubectl apply -f deploy/kubernetes/secrets.yaml
	kubectl apply -f deploy/kubernetes/postgresql/
	kubectl apply -f deploy/kubernetes/neo4j/
	kubectl apply -f deploy/kubernetes/api-gateway/
	kubectl apply -f deploy/kubernetes/image-recognition/
	kubectl apply -f deploy/kubernetes/knowledge-graph/
	kubectl apply -f deploy/kubernetes/reasoning-engine/
	kubectl apply -f deploy/kubernetes/model-management/
	kubectl apply -f deploy/kubernetes/log-statistics/
	kubectl apply -f deploy/kubernetes/ingress.yaml
	@echo "Staging deployment complete."

.PHONY: deploy-prod
deploy-prod: ## Deploy to Kubernetes production
	@echo "Deploying to production..."
	kubectl apply -f deploy/kubernetes/namespace.yaml
	kubectl apply -f deploy/kubernetes/configmap.yaml
	kubectl apply -f deploy/kubernetes/secrets.prod.yaml
	kubectl apply -f deploy/kubernetes/postgresql/
	kubectl apply -f deploy/kubernetes/neo4j/
	kubectl apply -f deploy/kubernetes/api-gateway/
	kubectl apply -f deploy/kubernetes/image-recognition/
	kubectl apply -f deploy/kubernetes/knowledge-graph/
	kubectl apply -f deploy/kubernetes/reasoning-engine/
	kubectl apply -f deploy/kubernetes/model-management/
	kubectl apply -f deploy/kubernetes/log-statistics/
	kubectl apply -f deploy/kubernetes/ingress.yaml
	@echo "Production deployment complete."

# =============================================================================
# Housekeeping
# =============================================================================
.PHONY: clean
clean: ## Remove build artifacts, caches, and temporary files
	@echo "Cleaning up..."
	rm -rf dist/ build/ __pycache__/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	find . -type f -name '*.log' -delete 2>/dev/null || true
	@echo "Clean complete."

.PHONY: clean-docker
clean-docker: ## Remove all docker resources for this project
	$(DOCKER_COMPOSE) down -v --remove-orphans --rmi all
