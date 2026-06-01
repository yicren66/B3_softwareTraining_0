#!/bin/bash
# =============================================================================
# PostgreSQL initialisation script for Jujube Platform
# Creates the log_statistics schema and required tables.
# Mounted at /docker-entrypoint-initdb.d/init.sh in the postgres container.
# =============================================================================
set -e

echo "[init_postgres] Initialising Jujube Platform database..."

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<'SQL'
BEGIN;

-- ------------------------------------------------------------------
-- Extensions
-- ------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ------------------------------------------------------------------
-- Schema: recognition_logs
-- ------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS log_statistics;

CREATE TABLE IF NOT EXISTS log_statistics.recognition_logs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_hash      VARCHAR(64)  NOT NULL,
    request_id      VARCHAR(36),
    user_id         VARCHAR(64),
    class_idx       INTEGER      NOT NULL,
    disease_name    VARCHAR(128) NOT NULL,
    category        VARCHAR(32)  NOT NULL,
    confidence      REAL         NOT NULL DEFAULT 0,
    severity        VARCHAR(32)  NOT NULL DEFAULT 'unknown',
    severity_conf   REAL         NOT NULL DEFAULT 0,
    affected_part   VARCHAR(64),
    symptoms        TEXT,
    kg_entity_id    VARCHAR(256),
    latency_ms      INTEGER      NOT NULL DEFAULT 0,
    backend         VARCHAR(32)  NOT NULL,
    model_version   VARCHAR(32),
    location_lat    DOUBLE PRECISION,
    location_lng    DOUBLE PRECISION,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_recognition_logs_created_at
    ON log_statistics.recognition_logs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_recognition_logs_disease
    ON log_statistics.recognition_logs (disease_name);
CREATE INDEX IF NOT EXISTS idx_recognition_logs_user_id
    ON log_statistics.recognition_logs (user_id);
CREATE INDEX IF NOT EXISTS idx_recognition_logs_severity
    ON log_statistics.recognition_logs (severity);

-- ------------------------------------------------------------------
-- Schema: model_registry
-- ------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS model_registry;

CREATE TABLE IF NOT EXISTS model_registry.models (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(128) NOT NULL,
    version         VARCHAR(64)  NOT NULL,
    backend         VARCHAR(32)  NOT NULL DEFAULT 'pytorch',
    artifact_path   VARCHAR(512) NOT NULL,
    metrics_json    JSONB,
    is_active       BOOLEAN      NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE(name, version)
);

COMMIT;

echo "[init_postgres] Database initialisation complete."
SQL
