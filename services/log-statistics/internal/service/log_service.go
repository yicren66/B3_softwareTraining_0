// Package service implements the business logic layer for the Log &
// Statistics service. It sits between the gRPC handlers and the PostgreSQL
// repository, adding retry logic, validation, and aggregation orchestration.
package service

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"time"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
	"github.com/rs/zerolog"

	"github.com/jujube-platform/log-statistics/internal/model"
	pb "github.com/jujube-platform/log-statistics/internal/grpc/proto"
	"github.com/jujube-platform/log-statistics/internal/repository"
)

// MaxRetries for database operations before giving up.
const maxRetries = 3

// retryBackoff is the base duration for exponential backoff between retries.
const retryBackoff = 100 * time.Millisecond

// =============================================================================
// LogService
// =============================================================================

// LogService provides all business-logic operations for the log-statistics
// service. It depends on PostgresRepo and Redis and exposes methods that
// map 1:1 to the gRPC service definition.
type LogService struct {
	repo   *repository.PostgresRepo
	redis  *redis.Client
	log    zerolog.Logger
	start  time.Time // used for uptime calculation
}

// NewLogService constructs a LogService.
func NewLogService(repo *repository.PostgresRepo, rdb *redis.Client, log zerolog.Logger) *LogService {
	return &LogService{
		repo:  repo,
		redis: rdb,
		log:   log,
		start: time.Now(),
	}
}

// Repo returns the underlying repository for advanced queries by handlers.
func (s *LogService) Repo() *repository.PostgresRepo { return s.repo }

// =============================================================================
// RecordRecognition
// =============================================================================

// RecordRecognition validates and persists a batch of recognition log entries.
// It uses the batch insert path for high-throughput and retries transient
// failures with exponential backoff.
func (s *LogService) RecordRecognition(ctx context.Context, req *pb.RecordRecognitionRequest) (*pb.RecordRecognitionResponse, error) {
	if len(req.Logs) == 0 {
		return &pb.RecordRecognitionResponse{
			Success:       false,
			RecordedCount: 0,
			ErrorMessage:  "no log entries provided",
		}, nil
	}

	logs := make([]*model.RecognitionLog, 0, len(req.Logs))
	for _, entry := range req.Logs {
		l, err := s.convertLogEntry(&entry)
		if err != nil {
			s.log.Warn().Err(err).Str("request_id", entry.RequestId).Msg("skipping invalid log entry")
			continue
		}
		logs = append(logs, l)
	}

	if len(logs) == 0 {
		return &pb.RecordRecognitionResponse{
			Success:       false,
			RecordedCount: 0,
			ErrorMessage:  "all log entries were invalid",
		}, nil
	}

	var recordedCount int
	var lastErr error

	for attempt := 0; attempt < maxRetries; attempt++ {
		count, err := s.repo.InsertRecognitionLogsBatch(ctx, logs)
		if err == nil {
			recordedCount = count
			lastErr = nil
			break
		}
		lastErr = err
		s.log.Warn().Err(err).Int("attempt", attempt+1).Msg("batch insert failed, retrying")

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(retryBackoff * time.Duration(1<<attempt)):
		}
	}

	if lastErr != nil {
		return &pb.RecordRecognitionResponse{
			Success:       false,
			RecordedCount: int32(recordedCount),
			ErrorMessage:  fmt.Sprintf("insert failed after %d attempts: %v", maxRetries, lastErr),
		}, nil
	}

	s.log.Info().Int("total", len(logs)).Int("recorded", recordedCount).Msg("recognition logs recorded")

	return &pb.RecordRecognitionResponse{
		Success:       true,
		RecordedCount: int32(recordedCount),
	}, nil
}

func (s *LogService) convertLogEntry(entry *pb.RecognitionLogEntry) (*model.RecognitionLog, error) {
	if entry.RequestId == "" {
		return nil, fmt.Errorf("request_id is required")
	}

	l := &model.RecognitionLog{
		RequestID:  entry.RequestId,
		ImageCount: int(entry.ImageCount),
		IsBatch:    entry.IsBatch,
	}

	if entry.Timestamp > 0 {
		t := time.UnixMilli(entry.Timestamp)
		l.Timestamp = t
	} else {
		l.Timestamp = time.Now()
	}

	if entry.UserId != "" {
		uid, err := uuid.Parse(entry.UserId)
		if err == nil {
			l.UserID = &uid
		}
	}

	if entry.GeoLocation != nil {
		b, err := json.Marshal(entry.GeoLocation)
		if err == nil {
			l.GeoLocation = b
		}
	}

	if entry.ImageHash != "" {
		l.ImageHash = &entry.ImageHash
	}
	if entry.ModelName != "" {
		l.ModelName = &entry.ModelName
	}
	if entry.ModelVersion != "" {
		l.ModelVersion = &entry.ModelVersion
	}

	if entry.RecognitionResult != nil {
		b, err := json.Marshal(entry.RecognitionResult)
		if err == nil {
			l.RecognitionResult = b
		}
		if entry.RecognitionResult.DiseaseName != "" && l.ImageHash == nil {
			hash := entry.RequestId
			l.ImageHash = &hash
		}
	}

	l.SmallTargetDetected = &entry.SmallTargetDetected

	if entry.Confidence > 0 {
		l.Confidence = &entry.Confidence
	}
	if entry.Severity != "" {
		sev := model.NormalizeSeverity(entry.Severity)
		l.Severity = &sev
	}

	latency := entry.LatencyMs
	l.LatencyMs = &latency

	if entry.ClientIp != "" {
		l.ClientIP = &entry.ClientIp
	}
	if entry.ClientPlatform != "" {
		l.ClientPlatform = &entry.ClientPlatform
	}

	return l, nil
}

// =============================================================================
// RecordFeedback
// =============================================================================

// RecordFeedback validates and persists an expert feedback record. It first
// looks up the associated recognition log, then inserts the feedback and
// updates the log in a transactional manner.
func (s *LogService) RecordFeedback(ctx context.Context, req *pb.RecordFeedbackRequest) (*pb.RecordFeedbackResponse, error) {
	if req.LogId == "" && req.RequestId == "" {
		return &pb.RecordFeedbackResponse{
			Success:      false,
			ErrorMessage: "either log_id or request_id is required",
		}, nil
	}

	// Resolve the log entry.
	var logEntry *model.RecognitionLog
	var logID uuid.UUID
	var err error

	if req.LogId != "" {
		logID, err = uuid.Parse(req.LogId)
		if err != nil {
			return &pb.RecordFeedbackResponse{
				Success:      false,
				ErrorMessage: fmt.Sprintf("invalid log_id: %v", err),
			}, nil
		}
		// Direct ID lookup — we need the request_id to find the log.
		// Since we don't have a GetByID method, we use RequestId if available.
	}

	if req.RequestId != "" {
		logEntry, err = s.repo.GetRecognitionLog(ctx, req.RequestId)
		if err != nil {
			return &pb.RecordFeedbackResponse{
				Success:      false,
				ErrorMessage: fmt.Sprintf("lookup failed: %v", err),
			}, nil
		}
		if logEntry == nil {
			return &pb.RecordFeedbackResponse{
				Success:      false,
				ErrorMessage: "recognition log not found",
			}, nil
		}
		logID = logEntry.ID
	}

	// Build feedback record.
	originalResult := json.RawMessage("{}")
	if logEntry != nil && len(logEntry.RecognitionResult) > 0 {
		originalResult = logEntry.RecognitionResult
	}

	fb := &model.FeedbackRecord{
		LogID:          logID,
		OriginalResult: originalResult,
		Status:         model.FeedbackStatusPending,
	}

	if req.CorrectedPest != "" {
		fb.CorrectedPest = &req.CorrectedPest
	}
	if req.CorrectedSeverity != "" {
		sev := model.NormalizeSeverity(req.CorrectedSeverity)
		fb.CorrectedSeverity = &sev
	}
	if req.CorrectionNote != "" {
		fb.CorrectionNote = &req.CorrectionNote
	}
	if req.SubmittedBy != "" {
		uid, err := uuid.Parse(req.SubmittedBy)
		if err == nil {
			fb.SubmittedBy = &uid
		}
	}

	var feedbackID uuid.UUID
	for attempt := 0; attempt < maxRetries; attempt++ {
		feedbackID, err = s.repo.InsertFeedback(ctx, fb)
		if err == nil {
			break
		}
		s.log.Warn().Err(err).Int("attempt", attempt+1).Msg("insert feedback failed, retrying")
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(retryBackoff * time.Duration(1<<attempt)):
		}
	}

	if err != nil {
		return &pb.RecordFeedbackResponse{
			Success:      false,
			ErrorMessage: fmt.Sprintf("insert feedback failed: %v", err),
		}, nil
	}

	s.log.Info().Str("feedback_id", feedbackID.String()).Str("request_id", req.RequestId).Msg("feedback recorded")

	return &pb.RecordFeedbackResponse{
		Success:    true,
		FeedbackId: feedbackID.String(),
	}, nil
}

// =============================================================================
// GetRecognitionStats
// =============================================================================

// GetRecognitionStats compiles aggregated statistics based on the request
// filters. It queries the database and assembles a StatsData struct.
func (s *LogService) GetRecognitionStats(ctx context.Context, req *pb.GetStatsRequest) (*pb.StatsData, error) {
	startDate, endDate := parseDateRange(req.StartDate, req.EndDate)
	county := req.County

	stats := &pb.StatsData{}

	// Total counts.
	total, err := s.repo.CountRecognitionLogs(ctx, startDate, endDate)
	if err != nil {
		return nil, fmt.Errorf("count recognition logs: %w", err)
	}
	stats.TotalRecognitions = total

	// Feedback count.
	fbCount, err := s.repo.CountFeedback(ctx, "")
	if err != nil {
		s.log.Warn().Err(err).Msg("count feedback failed")
	}
	stats.TotalFeedback = fbCount

	// Daily stats from the pre-aggregated table.
	dailyStats, err := s.repo.GetDailyStats(ctx, county, "")
	if err != nil {
		s.log.Warn().Err(err).Msg("get daily stats failed")
	}
	for _, ds := range dailyStats {
		stats.DailyStats = append(stats.DailyStats, pb.DailyStatEntry{
			Date:   ds.StatDate,
			Count:  int64(ds.RecognitionCount),
			AvgLat: 0, // latency not stored in daily stats table
		})
		if ds.AvgConfidence != nil {
			stats.AvgConfidence += *ds.AvgConfidence
		}
	}
	if len(dailyStats) > 0 {
		stats.AvgConfidence /= float32(len(dailyStats))
	}

	// Per-county breakdown.
	countyMap := make(map[string]*pb.CountyStatEntry)
	for _, ds := range dailyStats {
		if _, ok := countyMap[ds.County]; !ok {
			countyMap[ds.County] = &pb.CountyStatEntry{County: ds.County}
		}
		entry := countyMap[ds.County]
		entry.Count += int64(ds.RecognitionCount)
		if ds.AvgConfidence != nil {
			entry.AvgConf += *ds.AvgConfidence
		}
		if entry.TopPest == "" || ds.RecognitionCount > 0 {
			entry.TopPest = ds.PestType
		}
	}
	for _, v := range countyMap {
		if v.Count > 0 {
			v.AvgConf /= float32(v.Count)
		}
		stats.CountyBreakdown = append(stats.CountyBreakdown, *v)
	}

	// Severity distribution.
	stats.SeverityDist = s.computeSeverityDist(ctx, startDate, endDate, county)

	// Accuracy rate from feedback.
	if stats.TotalFeedback > 0 {
		incorporated, _ := s.repo.CountFeedback(ctx, model.FeedbackStatusIncorporated)
		stats.AccuracyRate = float32(incorporated) / float32(stats.TotalFeedback)
	}

	return stats, nil
}

func (s *LogService) computeSeverityDist(ctx context.Context, start, end time.Time, county string) *pb.SeverityDist {
	// Aggregate severity counts from recognition_logs.
	query := `
		SELECT
			COUNT(*) FILTER (WHERE severity = '轻度') AS mild,
			COUNT(*) FILTER (WHERE severity = '中度') AS moderate,
			COUNT(*) FILTER (WHERE severity = '重度') AS severe
		FROM recognition_logs
		WHERE 1=1
	`
	var args []interface{}
	argIdx := 1

	if !start.IsZero() && !end.IsZero() {
		query += fmt.Sprintf(" AND timestamp >= $%d AND timestamp < $%d", argIdx, argIdx+1)
		args = append(args, start, end)
		argIdx += 2
	}
	if county != "" {
		query += fmt.Sprintf(" AND geo_location->>'county' = $%d", argIdx)
		args = append(args, county)
		argIdx++
	}

	row := s.repo.Pool().QueryRow(ctx, query, args...)
	dist := &pb.SeverityDist{}
	if err := row.Scan(&dist.Mild, &dist.Moderate, &dist.Severe); err != nil {
		s.log.Warn().Err(err).Msg("severity dist query failed")
	}
	return dist
}

// =============================================================================
// ComputeHeatmapData
// =============================================================================

// ComputeHeatmapData generates per-county pest occurrence data formatted for
// WebGIS heatmap rendering. Each county is represented as a point with
// normalized intensity.
func (s *LogService) ComputeHeatmapData(ctx context.Context, start, end time.Time) (*pb.HeatmapData, error) {
	// Use raw aggregation from recognition_logs for the most accurate heatmap.
	aggregations, err := s.repo.QueryRecognitionAggregation(ctx, start, end)
	if err != nil {
		return nil, fmt.Errorf("aggregate for heatmap: %w", err)
	}

	// Group by county to roll up per-pest counts.
	type countyAgg struct {
		count        int32
		pestCounts   map[string]int32
		topPest      string
		topPestCount int32
	}

	countyMap := make(map[string]*countyAgg)
	for _, ag := range aggregations {
		c, ok := countyMap[ag.County]
		if !ok {
			c = &countyAgg{pestCounts: make(map[string]int32)}
			countyMap[ag.County] = c
		}
		c.count += int32(ag.Count)
		c.pestCounts[ag.Pest] = int32(ag.Count)
		if ag.Count > int(c.topPestCount) {
			c.topPest = ag.Pest
			c.topPestCount = int32(ag.Count)
		}
	}

	var maxCount int32
	for _, c := range countyMap {
		if c.count > maxCount {
			maxCount = c.count
		}
	}

	// Build heatmap points. We use representative coordinates per county;
	// in production these would come from a county geocoding table or from
	// the actual recognition_logs.geo_location field.
	heatmap := &pb.HeatmapData{
		Points:   make([]pb.HeatmapPoint, 0, len(countyMap)),
		MaxCount: maxCount,
	}

	for county, c := range countyMap {
		intensity := float32(0)
		if maxCount > 0 {
			intensity = float32(c.count) / float32(maxCount)
		}
		heatmap.Points = append(heatmap.Points, pb.HeatmapPoint{
			County:   county,
			Count:    c.count,
			Intensity: intensity,
			TopPest:  c.topPest,
		})
	}

	s.log.Info().Int("counties", len(heatmap.Points)).Int32("max_count", maxCount).Msg("heatmap computed")

	return heatmap, nil
}

// =============================================================================
// ComputeTrendData
// =============================================================================

// ComputeTrendData builds a time-series of recognition counts, average
// confidence, and accuracy trend for the specified pest type. An empty
// pestType aggregates across all pests.
func (s *LogService) ComputeTrendData(ctx context.Context, pestType string, start, end time.Time) (*pb.TrendData, error) {
	query := `
		SELECT
			DATE(timestamp) AS d,
			COUNT(*) AS cnt,
			AVG(confidence) AS avg_conf,
			COUNT(*) FILTER (WHERE expert_feedback IS NOT NULL) AS feedback_count
		FROM recognition_logs
		WHERE timestamp >= $1 AND timestamp < $2
	`
	args := []interface{}{start, end}
	argIdx := 3

	if pestType != "" {
		query += fmt.Sprintf(" AND recognition_result->>'disease_name' = $%d", argIdx)
		args = append(args, pestType)
		argIdx++
	}

	query += " GROUP BY d ORDER BY d ASC"

	rows, err := s.repo.Pool().Query(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("compute trend: %w", err)
	}
	defer rows.Close()

	trend := &pb.TrendData{PestType: pestType}
	for rows.Next() {
		var dp pb.TrendDataPoint
		var d time.Time
		var avgConf *float32
		var fbCount int32
		if err := rows.Scan(&d, &dp.Count, &avgConf, &fbCount); err != nil {
			return nil, fmt.Errorf("scan trend row: %w", err)
		}
		dp.Date = d.Format("2006-01-02")
		if avgConf != nil {
			dp.AvgConfidence = *avgConf
		}
		// Accuracy is inferred: if we have feedback, we use it as a proxy.
		// In production, compare original vs corrected labels.
		if dp.Count > 0 {
			dp.Accuracy = 1.0 - (float32(fbCount) / float32(dp.Count))
			dp.Accuracy = float32(math.Max(0, float64(dp.Accuracy)))
		}
		trend.DataPoints = append(trend.DataPoints, dp)
	}

	return trend, rows.Err()
}

// =============================================================================
// AggregateDailyStats
// =============================================================================

// AggregateDailyStats is a batch job that computes daily statistics from the
// raw recognition_logs table and upserts them into stats_daily. It should be
// called periodically (e.g., via a cron job) to keep the pre-aggregated data
// fresh.
func (s *LogService) AggregateDailyStats(ctx context.Context, date string) error {
	targetDate, err := time.Parse("2006-01-02", date)
	if err != nil {
		return fmt.Errorf("invalid date %q: %w", date, err)
	}

	start := targetDate
	end := targetDate.Add(24 * time.Hour)

	s.log.Info().Str("date", date).Msg("starting daily stats aggregation")

	aggregations, err := s.repo.QueryRecognitionAggregation(ctx, start, end)
	if err != nil {
		return fmt.Errorf("query aggregation: %w", err)
	}

	stats := make([]*model.StatsDaily, 0, len(aggregations))
	for _, ag := range aggregations {
		sd := &model.StatsDaily{
			StatDate:         date,
			County:           ag.County,
			PestType:         ag.Pest,
			RecognitionCount: ag.Count,
			SeverityDist:     ag.Severities,
		}
		if ag.AvgConf != nil {
			sd.AvgConfidence = ag.AvgConf
		}
		stats = append(stats, sd)
	}

	if len(stats) == 0 {
		s.log.Info().Str("date", date).Msg("no recognition data to aggregate")
		return nil
	}

	if err := s.repo.UpsertDailyStatsBatch(ctx, stats); err != nil {
		return fmt.Errorf("upsert batch: %w", err)
	}

	s.log.Info().Str("date", date).Int("records", len(stats)).Msg("daily stats aggregation complete")
	return nil
}

// =============================================================================
// Health
// =============================================================================

// HealthCheck returns the current service health status including DB/Redis
// connectivity and uptime.
func (s *LogService) HealthCheck(ctx context.Context) (*pb.HealthCheckResponse, error) {
	resp := &pb.HealthCheckResponse{Healthy: true, Version: "1.0.0"}

	if err := s.repo.Ping(ctx); err != nil {
		resp.DbConnected = false
		resp.Healthy = false
	} else {
		resp.DbConnected = true
	}

	if err := s.redis.Ping(ctx).Err(); err != nil {
		resp.RedisConnected = false
		resp.Healthy = false
	} else {
		resp.RedisConnected = true
	}

	resp.UptimeSeconds = int64(time.Since(s.start).Seconds())
	return resp, nil
}

// =============================================================================
// Helpers
// =============================================================================

// parseDateRange converts date strings to a time range. If dates are empty,
// it defaults to the last 30 days.
func parseDateRange(startStr, endStr string) (time.Time, time.Time) {
	end := time.Now().UTC()
	start := end.Add(-30 * 24 * time.Hour)

	if startStr != "" {
		if t, err := time.Parse("2006-01-02", startStr); err == nil {
			start = t
		}
	}
	if endStr != "" {
		if t, err := time.Parse("2006-01-02", endStr); err == nil {
			end = t.Add(24 * time.Hour) // inclusive of the end date
		}
	}
	return start, end
}
