-- =============================================================================
-- 001_init.down.sql
-- Log & Statistics Service — Rollback Initial Schema
-- =============================================================================
-- Drops all tables created by 001_init.up.sql in reverse dependency order
-- so that foreign-key constraints are honoured during teardown.
-- =============================================================================

DROP TABLE IF EXISTS weather_cache      CASCADE;
DROP TABLE IF EXISTS audit_logs         CASCADE;
DROP TABLE IF EXISTS stats_hourly       CASCADE;
DROP TABLE IF EXISTS stats_daily        CASCADE;
DROP TABLE IF EXISTS feedback_queue     CASCADE;
DROP TABLE IF EXISTS recognition_logs   CASCADE;
DROP TABLE IF EXISTS training_jobs      CASCADE;
DROP TABLE IF EXISTS model_versions     CASCADE;
DROP TABLE IF EXISTS text_corpus        CASCADE;
DROP TABLE IF EXISTS image_annotations  CASCADE;
DROP TABLE IF EXISTS image_datasets     CASCADE;
DROP TABLE IF EXISTS users              CASCADE;
