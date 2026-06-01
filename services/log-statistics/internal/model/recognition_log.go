// Package model defines the domain data models for the Log & Statistics
// service. Structs map 1:1 to the PostgreSQL tables defined in
// migrations/postgres/001_init.up.sql.
package model

import (
	"encoding/json"
	"time"

	"github.com/google/uuid"
)

// =============================================================================
// RecognitionLog — maps to table recognition_logs
// =============================================================================

// RecognitionLog represents a single image recognition event recorded in the
// recognition_logs table. Every field carries a json tag for serialization
// and a db tag hint for manual query construction.
type RecognitionLog struct {
	ID                   uuid.UUID       `json:"id"`
	RequestID            string          `json:"request_id"`
	UserID               *uuid.UUID      `json:"user_id,omitempty"`
	Timestamp            time.Time       `json:"timestamp"`
	GeoLocation          json.RawMessage `json:"geo_location,omitempty"`
	ImageHash            *string         `json:"image_hash,omitempty"`
	ImageCount           int             `json:"image_count"`
	IsBatch              bool            `json:"is_batch"`
	ModelName            *string         `json:"model_name,omitempty"`
	ModelVersion         *string         `json:"model_version,omitempty"`
	RecognitionResult    json.RawMessage `json:"recognition_result,omitempty"`
	SmallTargetDetected  *bool           `json:"small_target_detected,omitempty"`
	Confidence           *float32        `json:"confidence,omitempty"`
	Severity             *string         `json:"severity,omitempty"`
	LatencyMs            *int32          `json:"latency_ms,omitempty"`
	ExpertFeedback       json.RawMessage `json:"expert_feedback,omitempty"`
	FeedbackBy           *uuid.UUID      `json:"feedback_by,omitempty"`
	FeedbackAt           *time.Time      `json:"feedback_at,omitempty"`
	ClientIP             *string         `json:"client_ip,omitempty"`
	ClientPlatform       *string         `json:"client_platform,omitempty"`
}

// =============================================================================
// FeedbackRecord — maps to table feedback_queue
// =============================================================================

// FeedbackRecord represents an expert correction submitted for a recognition
// log entry. It is stored in the feedback_queue table and can transition
// through statuses: pending → reviewed → incorporated / rejected.
type FeedbackRecord struct {
	ID                     uuid.UUID       `json:"id"`
	LogID                  uuid.UUID       `json:"log_id"`
	OriginalResult         json.RawMessage `json:"original_result,omitempty"`
	CorrectedPest          *string         `json:"corrected_pest,omitempty"`
	CorrectedSeverity      *string         `json:"corrected_severity,omitempty"`
	CorrectionNote         *string         `json:"correction_note,omitempty"`
	SubmittedBy            *uuid.UUID      `json:"submitted_by,omitempty"`
	Status                 string          `json:"status"` // pending|reviewed|incorporated|rejected
	IncorporatedInVersion  *string         `json:"incorporated_in_version,omitempty"`
	CreatedAt              time.Time       `json:"created_at"`
	ProcessedAt            *time.Time      `json:"processed_at,omitempty"`
}

// =============================================================================
// StatsDaily — maps to table stats_daily
// =============================================================================

// StatsDaily holds per-county, per-pest, per-day aggregated recognition
// statistics. The unique constraint is (stat_date, county, pest_type).
type StatsDaily struct {
	ID               uuid.UUID       `json:"id"`
	StatDate         string          `json:"stat_date"` // "2006-01-02"
	County           string          `json:"county"`
	PestType         string          `json:"pest_type"`
	RecognitionCount int             `json:"recognition_count"`
	AvgConfidence    *float32        `json:"avg_confidence,omitempty"`
	SeverityDist     json.RawMessage `json:"severity_dist,omitempty"`
	AccuracyRate     *float32        `json:"accuracy_rate,omitempty"`
	CreatedAt        time.Time       `json:"created_at"`
}

// SeverityDistJSON is a helper struct for marshalling severity distribution.
type SeverityDistJSON struct {
	Mild     int `json:"mild"`
	Moderate int `json:"moderate"`
	Severe   int `json:"severe"`
}

// =============================================================================
// StatsHourly — maps to table stats_hourly
// =============================================================================

// StatsHourly holds per-hour aggregated performance metrics (latency, error
// counts, throughput). Used for operational monitoring dashboards.
type StatsHourly struct {
	ID             uuid.UUID  `json:"id"`
	StatHour       time.Time  `json:"stat_hour"`
	TotalRequests  int        `json:"total_requests"`
	AvgLatencyMs   *float32   `json:"avg_latency_ms,omitempty"`
	Tp99LatencyMs  *float32   `json:"tp99_latency_ms,omitempty"`
	ErrorCount     int        `json:"error_count"`
	ServiceName    *string    `json:"service_name,omitempty"`
	CreatedAt      time.Time  `json:"created_at"`
}

// =============================================================================
// AuditLog — maps to table audit_logs
// =============================================================================

// AuditLog records an auditable action for compliance and diagnostics.
type AuditLog struct {
	ID           uuid.UUID       `json:"id"`
	UserID       *uuid.UUID      `json:"user_id,omitempty"`
	Action       string          `json:"action"`
	ResourceType string          `json:"resource_type"`
	ResourceID   *string         `json:"resource_id,omitempty"`
	Detail       json.RawMessage `json:"detail,omitempty"`
	IPAddress    *string         `json:"ip_address,omitempty"`
	CreatedAt    time.Time       `json:"created_at"`
}

// =============================================================================
// Canonical severity labels (matches the CHECK constraint in the DDL).
// =============================================================================

const (
	SeverityLabelMild     = "轻度"
	SeverityLabelModerate = "中度"
	SeverityLabelSevere   = "重度"

	// Short forms used in some tables (image_annotations).
	SeverityShortMild     = "轻"
	SeverityShortModerate = "中"
	SeverityShortSevere   = "重"
)

// Category values.
const (
	CategoryDisease = "病害"
	CategoryPest    = "虫害"
)

// Feedback status values.
const (
	FeedbackStatusPending      = "pending"
	FeedbackStatusReviewed     = "reviewed"
	FeedbackStatusIncorporated = "incorporated"
	FeedbackStatusRejected     = "rejected"
)

// =============================================================================
// Helper methods
// =============================================================================

// NormalizeSeverity maps short severity forms to full labels and vice-versa.
func NormalizeSeverity(s string) string {
	switch s {
	case SeverityShortMild:
		return SeverityLabelMild
	case SeverityShortModerate:
		return SeverityLabelModerate
	case SeverityShortSevere:
		return SeverityLabelSevere
	default:
		return s
	}
}

// SeverityCounts is a convenience type for aggregating severity distributions.
type SeverityCounts map[string]int

// NewSeverityCounts returns a zero-initialised severity count map.
func NewSeverityCounts() SeverityCounts {
	return SeverityCounts{
		SeverityLabelMild:     0,
		SeverityLabelModerate: 0,
		SeverityLabelSevere:   0,
	}
}

// ToSeverityDistJSON converts a SeverityCounts map to the JSON representation
// expected by the stats_daily.severity_dist column.
func (sc SeverityCounts) ToSeverityDistJSON() (json.RawMessage, error) {
	d := SeverityDistJSON{
		Mild:     sc[SeverityLabelMild],
		Moderate: sc[SeverityLabelModerate],
		Severe:   sc[SeverityLabelSevere],
	}
	b, err := json.Marshal(d)
	if err != nil {
		return nil, err
	}
	return json.RawMessage(b), nil
}
