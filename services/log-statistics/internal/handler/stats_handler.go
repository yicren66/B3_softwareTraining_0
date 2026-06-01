// Package handler — statistics handler implementation.
package handler

import (
	"context"
	"fmt"
	"sort"
	"time"

	"github.com/rs/zerolog"

	pb "github.com/jujube-platform/log-statistics/internal/grpc/proto"
	"github.com/jujube-platform/log-statistics/internal/service"
)

// =============================================================================
// StatsHandler
// =============================================================================

// StatsHandler implements the GetStats RPC with heatmap computation,
// time-series trend analysis, and severity distribution pie-chart data.
type StatsHandler struct {
	svc *service.LogService
	log zerolog.Logger
}

// NewStatsHandler creates a StatsHandler backed by the given service.
func NewStatsHandler(svc *service.LogService, log zerolog.Logger) *StatsHandler {
	return &StatsHandler{svc: svc, log: log}
}

// =============================================================================
// GetStats — primary statistics RPC
// =============================================================================

// GetStats returns a comprehensive statistics response including:
//   - Aggregated stats data (counts, confidence, accuracy, severity dist)
//   - Heatmap data for WebGIS (per-county pest occurrence)
//   - Trend data (time-series)
//
// The caller may pass zero or more filters (county, date range, pest type).
func (h *StatsHandler) GetStats(ctx context.Context, req *pb.GetStatsRequest) (*pb.GetStatsResponse, error) {
	h.log.Debug().
		Str("county", req.County).
		Str("start", req.StartDate).
		Str("end", req.EndDate).
		Str("pest", req.PestType).
		Msg("GetStats called")

	resp := &pb.GetStatsResponse{}

	// ------------------------------------------------------------------
	// 1. Aggregated statistics
	// ------------------------------------------------------------------
	stats, err := h.svc.GetRecognitionStats(ctx, req)
	if err != nil {
		h.log.Error().Err(err).Msg("GetRecognitionStats failed")
		resp.ErrorMessage = fmt.Sprintf("stats query failed: %v", err)
		return resp, nil
	}
	resp.Stats = stats

	// ------------------------------------------------------------------
	// 2. Heatmap data — per-county pest occurrence for WebGIS
	// ------------------------------------------------------------------
	startDate, endDate := parseDateRange(req.StartDate, req.EndDate)
	heatmap, err := h.svc.ComputeHeatmapData(ctx, startDate, endDate)
	if err != nil {
		h.log.Warn().Err(err).Msg("ComputeHeatmapData failed")
	} else {
		// Enrich heatmap with geocoding (county centroid lookup).
		// In production this would query a geocoding table. Here we use
		// representative coordinates from the recognition logs.
		h.enrichHeatmapGeo(ctx, heatmap, startDate, endDate)
		resp.Heatmap = heatmap
	}

	// ------------------------------------------------------------------
	// 3. Trend data — time-series recognition frequency and accuracy
	// ------------------------------------------------------------------
	trend, err := h.svc.ComputeTrendData(ctx, req.PestType, startDate, endDate)
	if err != nil {
		h.log.Warn().Err(err).Msg("ComputeTrendData failed")
	} else {
		resp.Trend = trend
	}

	h.log.Debug().
		Int64("recognitions", stats.TotalRecognitions).
		Int("heatmap_points", len(heatmap.GetPoints())).
		Int("trend_points", len(trend.GetDataPoints())).
		Msg("GetStats completed")

	return resp, nil
}

// =============================================================================
// HealthCheck
// =============================================================================

// HealthCheck delegates to the service layer.
func (h *StatsHandler) HealthCheck(ctx context.Context, req *pb.HealthCheckRequest) (*pb.HealthCheckResponse, error) {
	return h.svc.HealthCheck(ctx)
}

// =============================================================================
// Heatmap enrichment
// =============================================================================

// enrichHeatmapGeo attempts to fill in lat/lng coordinates for each heatmap
// point by querying the centroid of recognition_logs per county. If the
// query fails, points retain zero coordinates (caller should handle this).
func (h *StatsHandler) enrichHeatmapGeo(ctx context.Context, heatmap *pb.HeatmapData, start, end time.Time) {
	if heatmap == nil || len(heatmap.Points) == 0 {
		return
	}

	// Build county list.
	counties := make([]string, 0, len(heatmap.Points))
	for _, p := range heatmap.Points {
		counties = append(counties, p.County)
	}

	// Query representative coordinates per county from recognition_logs.
	// We average lat/lng of all logs in that county within the time window.
	query := `
		SELECT
			geo_location->>'county' AS county,
			AVG((geo_location->>'lat')::float) AS avg_lat,
			AVG((geo_location->>'lng')::float) AS avg_lng,
			mode() WITHIN GROUP (ORDER BY severity) AS top_severity
		FROM recognition_logs
		WHERE geo_location IS NOT NULL
		  AND geo_location->>'county' = ANY($1)
		  AND timestamp >= $2 AND timestamp < $3
		GROUP BY county
	`

	rows, err := h.svc.Repo().Pool().Query(ctx, query, counties, start, end)
	if err != nil {
		h.log.Warn().Err(err).Msg("heatmap geocoding query failed")
		return
	}
	defer rows.Close()

	geoMap := make(map[string]struct {
		lat         float64
		lng         float64
		topSeverity string
	})

	for rows.Next() {
		var county string
		var lat, lng float64
		var topSev *string
		if err := rows.Scan(&county, &lat, &lng, &topSev); err != nil {
			continue
		}
		ts := ""
		if topSev != nil {
			ts = *topSev
		}
		geoMap[county] = struct {
			lat         float64
			lng         float64
			topSeverity string
		}{lat, lng, ts}
	}

	// Apply coordinates to heatmap points.
	for i := range heatmap.Points {
		if g, ok := geoMap[heatmap.Points[i].County]; ok {
			heatmap.Points[i].Lat = g.lat
			heatmap.Points[i].Lng = g.lng
			if g.topSeverity != "" && heatmap.Points[i].TopSeverity == "" {
				heatmap.Points[i].TopSeverity = g.topSeverity
			}
		}
	}
}

// =============================================================================
// Additional statistics helpers
// =============================================================================

// ComputePestDistribution returns the frequency distribution of pest types.
// Useful for pie charts and bar charts in the admin dashboard.
func (h *StatsHandler) ComputePestDistribution(ctx context.Context, start, end time.Time) (map[string]int32, error) {
	query := `
		SELECT
			COALESCE(recognition_result->>'disease_name', 'unknown') AS pest,
			COUNT(*) AS cnt
		FROM recognition_logs
		WHERE timestamp >= $1 AND timestamp < $2
		GROUP BY pest
		ORDER BY cnt DESC
		LIMIT 20
	`
	rows, err := h.svc.Repo().Pool().Query(ctx, query, start, end)
	if err != nil {
		return nil, fmt.Errorf("pest distribution query: %w", err)
	}
	defer rows.Close()

	dist := make(map[string]int32)
	for rows.Next() {
		var pest string
		var cnt int32
		if err := rows.Scan(&pest, &cnt); err != nil {
			continue
		}
		dist[pest] = cnt
	}
	return dist, rows.Err()
}

// ComputeAccuracyTrend compares original recognition confidence against
// expert-corrected labels over time, giving an accuracy trend suitable
// for before/after expert correction analysis.
func (h *StatsHandler) ComputeAccuracyTrend(ctx context.Context, pestType string, start, end time.Time) (*pb.TrendData, error) {
	query := `
		SELECT
			DATE(r.timestamp) AS d,
			COUNT(*) AS total,
			COUNT(*) FILTER (WHERE f.status = 'incorporated') AS corrected,
			AVG(r.confidence) AS avg_conf
		FROM recognition_logs r
		LEFT JOIN feedback_queue f ON f.log_id = r.id
		WHERE r.timestamp >= $1 AND r.timestamp < $2
	`
	args := []interface{}{start, end}
	argIdx := 3

	if pestType != "" {
		query += fmt.Sprintf(" AND r.recognition_result->>'disease_name' = $%d", argIdx)
		args = append(args, pestType)
		argIdx++
	}

	query += " GROUP BY d ORDER BY d ASC"

	rows, err := h.svc.Repo().Pool().Query(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("accuracy trend query: %w", err)
	}
	defer rows.Close()

	trend := &pb.TrendData{PestType: pestType}
	for rows.Next() {
		var dp pb.TrendDataPoint
		var d time.Time
		var total, corrected int32
		var avgConf *float32
		if err := rows.Scan(&d, &total, &corrected, &avgConf); err != nil {
			return nil, fmt.Errorf("scan accuracy row: %w", err)
		}
		dp.Date = d.Format("2006-01-02")
		dp.Count = total
		if avgConf != nil {
			dp.AvgConfidence = *avgConf
		}
		if total > 0 {
			// Accuracy = 1 - correction_rate; before expert correction it's
			// the raw model confidence, after correction the ratio improves.
			dp.Accuracy = 1.0 - float32(corrected)/float32(total)
			if dp.Accuracy < 0 {
				dp.Accuracy = 0
			}
		}
		trend.DataPoints = append(trend.DataPoints, dp)
	}
	return trend, rows.Err()
}

// ComputeSeverityPieChartData returns severity distribution suitable for
// consumption by front-end pie chart components (e.g., ECharts).
func (h *StatsHandler) ComputeSeverityPieChartData(ctx context.Context, start, end time.Time, county string) ([]PieChartEntry, error) {
	query := `
		SELECT
			COALESCE(severity, 'unknown') AS sev,
			COUNT(*) AS cnt
		FROM recognition_logs
		WHERE timestamp >= $1 AND timestamp < $2
	`
	args := []interface{}{start, end}
	argIdx := 3

	if county != "" {
		query += fmt.Sprintf(" AND geo_location->>'county' = $%d", argIdx)
		args = append(args, county)
		argIdx++
	}

	query += " GROUP BY sev ORDER BY cnt DESC"

	rows, err := h.svc.Repo().Pool().Query(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("severity pie query: %w", err)
	}
	defer rows.Close()

	var entries []PieChartEntry
	for rows.Next() {
		var e PieChartEntry
		if err := rows.Scan(&e.Name, &e.Value); err != nil {
			continue
		}
		entries = append(entries, e)
	}

	// Sort by value descending for consistent chart rendering.
	sort.Slice(entries, func(i, j int) bool { return entries[i].Value > entries[j].Value })
	return entries, rows.Err()
}

// PieChartEntry is a name/value pair for chart rendering.
type PieChartEntry struct {
	Name  string `json:"name"`
	Value int32  `json:"value"`
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

