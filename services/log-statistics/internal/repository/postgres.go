// Package repository implements the PostgreSQL data access layer for the
// Log & Statistics service. It uses pgx v5 for connection pooling and
// leverages pgx.Batch for high-throughput bulk inserts.
package repository

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rs/zerolog"

	"github.com/jujube-platform/log-statistics/internal/model"
)

// =============================================================================
// PostgresRepo
// =============================================================================

// PostgresRepo wraps a pgxpool.Pool and exposes all data-access methods needed
// by the log-statistics service.
type PostgresRepo struct {
	pool *pgxpool.Pool
	log  zerolog.Logger
}

// NewPostgresRepo creates a new PostgresRepo backed by the given connection pool.
func NewPostgresRepo(pool *pgxpool.Pool, log zerolog.Logger) *PostgresRepo {
	return &PostgresRepo{pool: pool, log: log}
}

// Pool returns the underlying pool for health checks.
func (r *PostgresRepo) Pool() *pgxpool.Pool {
	return r.pool
}

// Ping verifies the database connection is alive.
func (r *PostgresRepo) Ping(ctx context.Context) error {
	return r.pool.Ping(ctx)
}

// =============================================================================
// Recognition Logs
// =============================================================================

const insertRecognitionLogSQL = `
INSERT INTO recognition_logs (
    request_id, user_id, timestamp, geo_location, image_hash, image_count,
    is_batch, model_name, model_version, recognition_result,
    small_target_detected, confidence, severity, latency_ms,
    client_ip, client_platform
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16
) RETURNING id
`

// InsertRecognitionLog inserts a single recognition log row and returns the
// generated UUID.
func (r *PostgresRepo) InsertRecognitionLog(ctx context.Context, l *model.RecognitionLog) (uuid.UUID, error) {
	var id uuid.UUID

	userID := pgxTypeFromUUIDPtr(l.UserID)
	imageHash := pgxTypeFromStringPtr(l.ImageHash)
	modelName := pgxTypeFromStringPtr(l.ModelName)
	modelVersion := pgxTypeFromStringPtr(l.ModelVersion)
	smallTarget := pgxTypeFromBoolPtr(l.SmallTargetDetected)
	confidence := pgxTypeFromFloat32Ptr(l.Confidence)
	severity := pgxTypeFromStringPtr(l.Severity)
	latency := pgxTypeFromInt32Ptr(l.LatencyMs)
	clientIP := pgxTypeFromStringPtr(l.ClientIP)
	clientPlatform := pgxTypeFromStringPtr(l.ClientPlatform)

	err := r.pool.QueryRow(ctx, insertRecognitionLogSQL,
		l.RequestID, userID, l.Timestamp, l.GeoLocation, imageHash, l.ImageCount,
		l.IsBatch, modelName, modelVersion, l.RecognitionResult,
		smallTarget, confidence, severity, latency,
		clientIP, clientPlatform,
	).Scan(&id)

	if err != nil {
		r.log.Error().Err(err).Str("request_id", l.RequestID).Msg("InsertRecognitionLog failed")
		return uuid.Nil, fmt.Errorf("insert recognition log: %w", err)
	}
	return id, nil
}

// InsertRecognitionLogsBatch inserts multiple recognition logs using a pgx
// batch, which sends all statements in a single round-trip for high throughput.
func (r *PostgresRepo) InsertRecognitionLogsBatch(ctx context.Context, logs []*model.RecognitionLog) (int, error) {
	batch := &pgx.Batch{}
	for _, l := range logs {
		batch.Queue(insertRecognitionLogSQL,
			l.RequestID,
			pgxTypeFromUUIDPtr(l.UserID),
			l.Timestamp,
			l.GeoLocation,
			pgxTypeFromStringPtr(l.ImageHash),
			l.ImageCount,
			l.IsBatch,
			pgxTypeFromStringPtr(l.ModelName),
			pgxTypeFromStringPtr(l.ModelVersion),
			l.RecognitionResult,
			pgxTypeFromBoolPtr(l.SmallTargetDetected),
			pgxTypeFromFloat32Ptr(l.Confidence),
			pgxTypeFromStringPtr(l.Severity),
			pgxTypeFromInt32Ptr(l.LatencyMs),
			pgxTypeFromStringPtr(l.ClientIP),
			pgxTypeFromStringPtr(l.ClientPlatform),
		)
	}

	br := r.pool.SendBatch(ctx, batch)
	defer br.Close()

	successCount := 0
	for range logs {
		var id uuid.UUID
		if err := br.QueryRow().Scan(&id); err != nil {
			r.log.Warn().Err(err).Int("index", successCount).Msg("batch insert row failed")
		} else {
			successCount++
		}
	}

	return successCount, nil
}

// GetRecognitionLog retrieves a single recognition log by request_id.
func (r *PostgresRepo) GetRecognitionLog(ctx context.Context, requestID string) (*model.RecognitionLog, error) {
	const query = `
		SELECT id, request_id, user_id, timestamp, geo_location, image_hash,
		       image_count, is_batch, model_name, model_version,
		       recognition_result, small_target_detected, confidence, severity,
		       latency_ms, expert_feedback, feedback_by, feedback_at,
		       client_ip, client_platform
		FROM recognition_logs
		WHERE request_id = $1
	`

	var l model.RecognitionLog
	err := r.pool.QueryRow(ctx, query, requestID).Scan(
		&l.ID, &l.RequestID, &l.UserID, &l.Timestamp, &l.GeoLocation,
		&l.ImageHash, &l.ImageCount, &l.IsBatch, &l.ModelName, &l.ModelVersion,
		&l.RecognitionResult, &l.SmallTargetDetected, &l.Confidence, &l.Severity,
		&l.LatencyMs, &l.ExpertFeedback, &l.FeedbackBy, &l.FeedbackAt,
		&l.ClientIP, &l.ClientPlatform,
	)
	if err != nil {
		if err == pgx.ErrNoRows {
			return nil, nil
		}
		return nil, fmt.Errorf("get recognition log: %w", err)
	}
	return &l, nil
}

// QueryLogsByTimeRange returns recognition logs within [start, end).
func (r *PostgresRepo) QueryLogsByTimeRange(ctx context.Context, start, end time.Time) ([]model.RecognitionLog, error) {
	const query = `
		SELECT id, request_id, user_id, timestamp, geo_location, image_hash,
		       image_count, is_batch, model_name, model_version,
		       recognition_result, small_target_detected, confidence, severity,
		       latency_ms, expert_feedback, feedback_by, feedback_at,
		       client_ip, client_platform
		FROM recognition_logs
		WHERE timestamp >= $1 AND timestamp < $2
		ORDER BY timestamp DESC
		LIMIT 10000
	`
	return r.queryLogs(ctx, query, start, end)
}

// QueryLogsByCounty returns recognition logs filtered by county and time range.
func (r *PostgresRepo) QueryLogsByCounty(ctx context.Context, county string, start, end time.Time) ([]model.RecognitionLog, error) {
	const query = `
		SELECT id, request_id, user_id, timestamp, geo_location, image_hash,
		       image_count, is_batch, model_name, model_version,
		       recognition_result, small_target_detected, confidence, severity,
		       latency_ms, expert_feedback, feedback_by, feedback_at,
		       client_ip, client_platform
		FROM recognition_logs
		WHERE geo_location->>'county' = $1
		  AND timestamp >= $2 AND timestamp < $3
		ORDER BY timestamp DESC
		LIMIT 10000
	`
	return r.queryLogs(ctx, query, county, start, end)
}

func (r *PostgresRepo) queryLogs(ctx context.Context, query string, args ...interface{}) ([]model.RecognitionLog, error) {
	rows, err := r.pool.Query(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("query logs: %w", err)
	}
	defer rows.Close()

	var logs []model.RecognitionLog
	for rows.Next() {
		var l model.RecognitionLog
		if err := rows.Scan(
			&l.ID, &l.RequestID, &l.UserID, &l.Timestamp, &l.GeoLocation,
			&l.ImageHash, &l.ImageCount, &l.IsBatch, &l.ModelName, &l.ModelVersion,
			&l.RecognitionResult, &l.SmallTargetDetected, &l.Confidence, &l.Severity,
			&l.LatencyMs, &l.ExpertFeedback, &l.FeedbackBy, &l.FeedbackAt,
			&l.ClientIP, &l.ClientPlatform,
		); err != nil {
			return nil, fmt.Errorf("scan log row: %w", err)
		}
		logs = append(logs, l)
	}
	return logs, rows.Err()
}

// =============================================================================
// Feedback
// =============================================================================

const insertFeedbackSQL = `
INSERT INTO feedback_queue (
    log_id, original_result, corrected_pest, corrected_severity,
    correction_note, submitted_by, status
) VALUES ($1, $2, $3, $4, $5, $6, 'pending')
RETURNING id
`

// InsertFeedback inserts a new feedback record and returns the generated UUID.
func (r *PostgresRepo) InsertFeedback(ctx context.Context, fb *model.FeedbackRecord) (uuid.UUID, error) {
	var id uuid.UUID
	err := r.pool.QueryRow(ctx, insertFeedbackSQL,
		fb.LogID, fb.OriginalResult,
		pgxTypeFromStringPtr(fb.CorrectedPest),
		pgxTypeFromStringPtr(fb.CorrectedSeverity),
		pgxTypeFromStringPtr(fb.CorrectionNote),
		pgxTypeFromUUIDPtr(fb.SubmittedBy),
	).Scan(&id)
	if err != nil {
		r.log.Error().Err(err).Str("log_id", fb.LogID.String()).Msg("InsertFeedback failed")
		return uuid.Nil, fmt.Errorf("insert feedback: %w", err)
	}

	// Also update the recognition_logs table with the feedback reference.
	const updateLogSQL = `
		UPDATE recognition_logs
		SET expert_feedback = $1, feedback_by = $2, feedback_at = now()
		WHERE id = $3
	`
	_, err = r.pool.Exec(ctx, updateLogSQL, fb.OriginalResult,
		pgxTypeFromUUIDPtr(fb.SubmittedBy), fb.LogID)
	if err != nil {
		r.log.Warn().Err(err).Str("log_id", fb.LogID.String()).Msg("failed to update recognition_log with feedback")
	}

	return id, nil
}

// UpdateFeedbackStatus transitions a feedback record to a new status.
func (r *PostgresRepo) UpdateFeedbackStatus(ctx context.Context, id uuid.UUID, status string) error {
	const query = `
		UPDATE feedback_queue
		SET status = $1, processed_at = CASE WHEN $1 IN ('reviewed','incorporated','rejected') THEN now() ELSE processed_at END
		WHERE id = $2
	`
	ct, err := r.pool.Exec(ctx, query, status, id)
	if err != nil {
		return fmt.Errorf("update feedback status: %w", err)
	}
	if ct.RowsAffected() == 0 {
		return fmt.Errorf("feedback record %s not found", id)
	}
	return nil
}

// =============================================================================
// Daily Stats
// =============================================================================

const upsertDailyStatsSQL = `
INSERT INTO stats_daily (stat_date, county, pest_type, recognition_count, avg_confidence, severity_dist, accuracy_rate)
VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT (stat_date, county, pest_type)
DO UPDATE SET
    recognition_count = stats_daily.recognition_count + EXCLUDED.recognition_count,
    avg_confidence    = (stats_daily.avg_confidence + EXCLUDED.avg_confidence) / 2.0,
    severity_dist     = EXCLUDED.severity_dist,
    accuracy_rate     = COALESCE(EXCLUDED.accuracy_rate, stats_daily.accuracy_rate)
`

// UpsertDailyStats inserts or updates a daily stats row atomically.
func (r *PostgresRepo) UpsertDailyStats(ctx context.Context, s *model.StatsDaily) error {
	_, err := r.pool.Exec(ctx, upsertDailyStatsSQL,
		s.StatDate, s.County, s.PestType,
		s.RecognitionCount, s.AvgConfidence, s.SeverityDist, s.AccuracyRate,
	)
	if err != nil {
		return fmt.Errorf("upsert daily stats: %w", err)
	}
	return nil
}

// UpsertDailyStatsBatch inserts or updates multiple daily stats rows using a
// pgx batch for high throughput.
func (r *PostgresRepo) UpsertDailyStatsBatch(ctx context.Context, stats []*model.StatsDaily) error {
	batch := &pgx.Batch{}
	for _, s := range stats {
		batch.Queue(upsertDailyStatsSQL,
			s.StatDate, s.County, s.PestType,
			s.RecognitionCount, s.AvgConfidence, s.SeverityDist, s.AccuracyRate,
		)
	}
	br := r.pool.SendBatch(ctx, batch)
	defer br.Close()

	for range stats {
		if _, err := br.Exec(); err != nil {
			return fmt.Errorf("batch upsert daily stats: %w", err)
		}
	}
	return nil
}

// GetDailyStats retrieves daily stats matching the given filters.
func (r *PostgresRepo) GetDailyStats(ctx context.Context, county string, date string) ([]model.StatsDaily, error) {
	query := `
		SELECT id, stat_date, county, pest_type, recognition_count,
		       avg_confidence, severity_dist, accuracy_rate, created_at
		FROM stats_daily
		WHERE 1=1
	`
	var args []interface{}
	argIdx := 1

	if county != "" {
		query += fmt.Sprintf(" AND county = $%d", argIdx)
		args = append(args, county)
		argIdx++
	}
	if date != "" {
		query += fmt.Sprintf(" AND stat_date = $%d", argIdx)
		args = append(args, date)
		argIdx++
	}
	query += " ORDER BY recognition_count DESC LIMIT 200"

	rows, err := r.pool.Query(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("get daily stats: %w", err)
	}
	defer rows.Close()

	var stats []model.StatsDaily
	for rows.Next() {
		var s model.StatsDaily
		if err := rows.Scan(&s.ID, &s.StatDate, &s.County, &s.PestType,
			&s.RecognitionCount, &s.AvgConfidence, &s.SeverityDist,
			&s.AccuracyRate, &s.CreatedAt); err != nil {
			return nil, fmt.Errorf("scan stats row: %w", err)
		}
		stats = append(stats, s)
	}
	return stats, rows.Err()
}

// =============================================================================
// Aggregation Queries
// =============================================================================

// CountRecognitionsByCountyPest returns recognition counts grouped by county
// and pest type for a given date range. Used by AggregateDailyStats.
type CountyPestCount struct {
	County string
	Pest   string
	Count  int
	AvgConf *float32
	Severities json.RawMessage
}

// QueryRecognitionAggregation executes an aggregation query over recognition_logs.
func (r *PostgresRepo) QueryRecognitionAggregation(ctx context.Context, start, end time.Time) ([]CountyPestCount, error) {
	const query = `
		SELECT
			COALESCE(geo_location->>'county', 'unknown') AS county,
			COALESCE(recognition_result->>'disease_name', 'unknown') AS pest_type,
			COUNT(*) AS cnt,
			AVG(confidence) AS avg_conf,
			jsonb_build_object(
				'mild',     COUNT(*) FILTER (WHERE severity = '轻度'),
				'moderate', COUNT(*) FILTER (WHERE severity = '中度'),
				'severe',   COUNT(*) FILTER (WHERE severity = '重度')
			) AS sev_dist
		FROM recognition_logs
		WHERE timestamp >= $1 AND timestamp < $2
		  AND recognition_result IS NOT NULL
		GROUP BY county, pest_type
		ORDER BY cnt DESC
	`
	rows, err := r.pool.Query(ctx, query, start, end)
	if err != nil {
		return nil, fmt.Errorf("query recognition aggregation: %w", err)
	}
	defer rows.Close()

	var results []CountyPestCount
	for rows.Next() {
		var cpc CountyPestCount
		if err := rows.Scan(&cpc.County, &cpc.Pest, &cpc.Count, &cpc.AvgConf, &cpc.Severities); err != nil {
			return nil, fmt.Errorf("scan aggregation row: %w", err)
		}
		results = append(results, cpc)
	}
	return results, rows.Err()
}

// CountFeedback returns the total number of feedback records in a given status.
func (r *PostgresRepo) CountFeedback(ctx context.Context, status string) (int64, error) {
	const query = `SELECT COUNT(*) FROM feedback_queue WHERE status = $1`
	var cnt int64
	err := r.pool.QueryRow(ctx, query, status).Scan(&cnt)
	return cnt, err
}

// CountRecognitionLogs returns the total count of recognition logs optionally
// filtered by time range.
func (r *PostgresRepo) CountRecognitionLogs(ctx context.Context, start, end time.Time) (int64, error) {
	query := `SELECT COUNT(*) FROM recognition_logs`
	var args []interface{}
	if !start.IsZero() && !end.IsZero() {
		query += ` WHERE timestamp >= $1 AND timestamp < $2`
		args = append(args, start, end)
	}
	var cnt int64
	err := r.pool.QueryRow(ctx, query, args...).Scan(&cnt)
	return cnt, err
}

// =============================================================================
// nil-safe helpers for pgx
// =============================================================================

func pgxTypeFromUUIDPtr(p *uuid.UUID) interface{} {
	if p == nil || *p == uuid.Nil {
		return nil
	}
	return *p
}

func pgxTypeFromStringPtr(s *string) interface{} {
	if s == nil {
		return nil
	}
	return *s
}

func pgxTypeFromBoolPtr(b *bool) interface{} {
	if b == nil {
		return nil
	}
	return *b
}

func pgxTypeFromFloat32Ptr(f *float32) interface{} {
	if f == nil {
		return nil
	}
	return *f
}

func pgxTypeFromInt32Ptr(i *int32) interface{} {
	if i == nil {
		return nil
	}
	return *i
}
