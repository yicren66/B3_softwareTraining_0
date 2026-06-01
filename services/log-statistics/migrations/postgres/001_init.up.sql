-- =============================================================================
-- 001_init.up.sql
-- Log & Statistics Service — Initial Schema
-- =============================================================================
-- Creates all tables, indexes, constraints, and seed data for the
-- log-statistics service per the Software Requirements Specification.
-- Uses: gen_random_uuid() for PKs, TIMESTAMPTZ for all timestamps,
--       ON DELETE SET NULL for user-reference FKs.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. users
-- ---------------------------------------------------------------------------
CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username        VARCHAR(64)  NOT NULL UNIQUE,
    password_hash   VARCHAR(256) NOT NULL,
    role            VARCHAR(32)  NOT NULL CHECK (role IN (
                        'super_admin','work_admin','plant_expert','farmer','guest'
                    )),
    display_name    VARCHAR(128),
    county          VARCHAR(64),
    township        VARCHAR(64),
    phone           VARCHAR(20),
    email           VARCHAR(128),
    is_active       BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX idx_users_role   ON users (role);
CREATE INDEX idx_users_county ON users (county);

-- ---------------------------------------------------------------------------
-- 2. image_datasets
-- ---------------------------------------------------------------------------
CREATE TABLE image_datasets (
    id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version                  VARCHAR(32)  NOT NULL,
    name                     VARCHAR(256) NOT NULL,
    description              TEXT,
    total_images             INTEGER      NOT NULL DEFAULT 0,
    class_distribution       JSONB,
    annotation_schema_version VARCHAR(16),
    status                   VARCHAR(32)  NOT NULL DEFAULT 'active',
    created_at               TIMESTAMPTZ  NOT NULL DEFAULT now(),
    created_by               UUID         REFERENCES users(id)
                                ON DELETE SET NULL
);

-- ---------------------------------------------------------------------------
-- 3. image_annotations
-- ---------------------------------------------------------------------------
CREATE TABLE image_annotations (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id       UUID         NOT NULL REFERENCES image_datasets(id)
                                    ON DELETE CASCADE,
    image_path       VARCHAR(512) NOT NULL,
    image_hash       VARCHAR(64),
    pest_type        VARCHAR(128),
    category         VARCHAR(16)  CHECK (category IN ('病害','虫害')),
    severity         VARCHAR(8)   CHECK (severity IN ('轻','中','重')),
    affected_part    VARCHAR(64),
    bbox             JSONB,
    has_small_target BOOLEAN,
    source_county    VARCHAR(64),
    capture_date     DATE,
    lighting_condition VARCHAR(32),
    annotation_by    UUID         REFERENCES users(id) ON DELETE SET NULL,
    annotation_at    TIMESTAMPTZ,
    is_verified      BOOLEAN      NOT NULL DEFAULT FALSE,
    verified_by      UUID         REFERENCES users(id) ON DELETE SET NULL,
    verified_at      TIMESTAMPTZ
);

CREATE INDEX idx_image_annotations_dataset_id ON image_annotations (dataset_id);
CREATE INDEX idx_image_annotations_pest_type  ON image_annotations (pest_type);
CREATE INDEX idx_image_annotations_image_hash ON image_annotations (image_hash);

-- ---------------------------------------------------------------------------
-- 4. text_corpus
-- ---------------------------------------------------------------------------
CREATE TABLE text_corpus (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title              VARCHAR(256),
    source_type        VARCHAR(32),
    source_path        VARCHAR(512),
    content_text       TEXT,
    content_tokens     TEXT[],
    entities_extracted JSONB,
    relations_extracted JSONB,
    is_processed       BOOLEAN     NOT NULL DEFAULT FALSE,
    version            VARCHAR(32),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- 5. model_versions
-- ---------------------------------------------------------------------------
CREATE TABLE model_versions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name      VARCHAR(128) NOT NULL,
    version         VARCHAR(32)  NOT NULL,
    model_type      VARCHAR(32)  NOT NULL CHECK (model_type IN (
                        'classification','small_target','severity','distilled'
                    )),
    backbone        VARCHAR(64),
    framework       VARCHAR(32),
    artifact_path   VARCHAR(512),
    onnx_path       VARCHAR(512),
    tensorrt_path   VARCHAR(512),
    tflite_path      VARCHAR(512),
    dataset_version VARCHAR(32),
    input_size      JSONB,
    num_classes     INTEGER      NOT NULL DEFAULT 15,
    class_labels    TEXT[],
    metrics         JSONB,
    training_config JSONB,
    training_date   TIMESTAMPTZ,
    trained_by      UUID         REFERENCES users(id) ON DELETE SET NULL,
    status          VARCHAR(32)  NOT NULL DEFAULT 'draft',
    deployed_at     TIMESTAMPTZ,
    is_production   BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),

    CONSTRAINT uq_model_versions_name_ver UNIQUE (model_name, version)
);

CREATE INDEX idx_model_versions_name_ver ON model_versions (model_name, version);
CREATE INDEX idx_model_versions_status   ON model_versions (status);

-- ---------------------------------------------------------------------------
-- 6. training_jobs
-- ---------------------------------------------------------------------------
CREATE TABLE training_jobs (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_type          VARCHAR(32)  NOT NULL CHECK (job_type IN (
                          'classification','small_target','severity','distilled'
                      )),
    model_name        VARCHAR(128) NOT NULL,
    dataset_version   VARCHAR(32),
    base_model_version UUID        REFERENCES model_versions(id)
                                     ON DELETE SET NULL,
    status            VARCHAR(32)  NOT NULL DEFAULT 'queued',
    progress          REAL         NOT NULL DEFAULT 0.0,
    current_epoch     INTEGER,
    total_epochs      INTEGER,
    config            JSONB,
    gpu_count         INTEGER      NOT NULL DEFAULT 1,
    started_at        TIMESTAMPTZ,
    completed_at      TIMESTAMPTZ,
    error_message     TEXT,
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
    created_by        UUID         REFERENCES users(id) ON DELETE SET NULL
);

-- ---------------------------------------------------------------------------
-- 7. recognition_logs
-- ---------------------------------------------------------------------------
CREATE TABLE recognition_logs (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id            VARCHAR(36)  NOT NULL UNIQUE,
    user_id               UUID         REFERENCES users(id) ON DELETE SET NULL,
    timestamp             TIMESTAMPTZ  NOT NULL DEFAULT now(),
    geo_location          JSONB,
    image_hash            VARCHAR(64),
    image_count           INTEGER      NOT NULL DEFAULT 1,
    is_batch              BOOLEAN      NOT NULL DEFAULT FALSE,
    model_name            VARCHAR(128),
    model_version         VARCHAR(32),
    recognition_result    JSONB,
    small_target_detected BOOLEAN,
    confidence            REAL,
    severity              VARCHAR(8),
    latency_ms            INTEGER,
    expert_feedback       JSONB,
    feedback_by           UUID         REFERENCES users(id) ON DELETE SET NULL,
    feedback_at           TIMESTAMPTZ,
    client_ip             INET,
    client_platform       VARCHAR(32)
);

CREATE INDEX idx_recognition_logs_user_id          ON recognition_logs (user_id);
CREATE INDEX idx_recognition_logs_timestamp        ON recognition_logs (timestamp DESC);
CREATE INDEX idx_recognition_logs_result_disease   ON recognition_logs ((recognition_result->>'disease_name'));
CREATE INDEX idx_recognition_logs_severity         ON recognition_logs (severity);
CREATE INDEX idx_recognition_logs_geo_county       ON recognition_logs ((geo_location->>'county'));

-- ---------------------------------------------------------------------------
-- 8. feedback_queue
-- ---------------------------------------------------------------------------
CREATE TABLE feedback_queue (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    log_id                  UUID         NOT NULL REFERENCES recognition_logs(id)
                                           ON DELETE CASCADE,
    original_result         JSONB,
    corrected_pest          VARCHAR(128),
    corrected_severity      VARCHAR(8),
    correction_note         TEXT,
    submitted_by            UUID         REFERENCES users(id) ON DELETE SET NULL,
    status                  VARCHAR(32)  NOT NULL DEFAULT 'pending',
    incorporated_in_version VARCHAR(32),
    created_at              TIMESTAMPTZ  NOT NULL DEFAULT now(),
    processed_at            TIMESTAMPTZ
);

-- ---------------------------------------------------------------------------
-- 9. stats_daily
-- ---------------------------------------------------------------------------
CREATE TABLE stats_daily (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    stat_date         DATE         NOT NULL,
    county            VARCHAR(64)  NOT NULL,
    pest_type         VARCHAR(128) NOT NULL,
    recognition_count INTEGER      NOT NULL DEFAULT 0,
    avg_confidence    REAL,
    severity_dist     JSONB,
    accuracy_rate     REAL,
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),

    CONSTRAINT uq_stats_daily_date_county_pest UNIQUE (stat_date, county, pest_type)
);

-- ---------------------------------------------------------------------------
-- 10. stats_hourly
-- ---------------------------------------------------------------------------
CREATE TABLE stats_hourly (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    stat_hour       TIMESTAMPTZ NOT NULL,
    total_requests  INTEGER     NOT NULL DEFAULT 0,
    avg_latency_ms  REAL,
    tp99_latency_ms REAL,
    error_count     INTEGER     NOT NULL DEFAULT 0,
    service_name    VARCHAR(64),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- 11. audit_logs
-- ---------------------------------------------------------------------------
CREATE TABLE audit_logs (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID         REFERENCES users(id) ON DELETE SET NULL,
    action        VARCHAR(64)  NOT NULL,
    resource_type VARCHAR(64)  NOT NULL,
    resource_id   VARCHAR(64),
    detail        JSONB,
    ip_address    INET,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX idx_audit_logs_user_id   ON audit_logs (user_id);
CREATE INDEX idx_audit_logs_action    ON audit_logs (action);
CREATE INDEX idx_audit_logs_created   ON audit_logs (created_at DESC);

-- ---------------------------------------------------------------------------
-- 12. weather_cache
-- ---------------------------------------------------------------------------
CREATE TABLE weather_cache (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    county        VARCHAR(64)  NOT NULL,
    record_time   TIMESTAMPTZ  NOT NULL,
    temperature   REAL,
    humidity      REAL,
    precipitation REAL,
    wind_speed    REAL,
    weather_code  VARCHAR(16),
    source_api    VARCHAR(128),
    fetched_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),

    CONSTRAINT uq_weather_cache_county_time UNIQUE (county, record_time)
);

-- =============================================================================
-- Seed Data — Default Test Users
-- =============================================================================
-- NOTE: The password hashes below are PLACEHOLDERS.
-- Replace them with real bcrypt hashes before deploying to any environment.
-- Generate with:  $2a$10$...  (or your preferred bcrypt cost factor).
--
--   admin  / admin123   -> super_admin
--   expert / expert123  -> plant_expert
--   farmer / farmer123  -> farmer
-- =============================================================================

INSERT INTO users (username, password_hash, role, display_name, is_active)
VALUES
(
    'admin',
    -- PLACEHOLDER -- bcrypt hash of 'admin123' (cost 10)
    '$2a$10$PLACEHOLDER_admin_hash_replace_me',
    'super_admin',
    'System Administrator',
    TRUE
),
(
    'expert',
    -- PLACEHOLDER -- bcrypt hash of 'expert123' (cost 10)
    '$2a$10$PLACEHOLDER_expert_hash_replace_me',
    'plant_expert',
    'Plant Expert',
    TRUE
),
(
    'farmer',
    -- PLACEHOLDER -- bcrypt hash of 'farmer123' (cost 10)
    '$2a$10$PLACEHOLDER_farmer_hash_replace_me',
    'farmer',
    'Demo Farmer',
    TRUE
);
