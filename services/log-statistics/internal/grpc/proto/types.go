// Package proto defines the protobuf message types and service interface for
// the Log & Statistics gRPC service. These types mirror the generated code
// that would be produced from libs/proto/log_statistics.proto.
package proto

import (
	"context"
	"fmt"

	"google.golang.org/grpc"
)

// =============================================================================
// Enums
// =============================================================================

// Severity represents the severity level of a pest/disease occurrence.
type Severity int32

const (
	SeverityUnspecified Severity = 0
	SeverityMild        Severity = 1 // 轻度
	SeverityModerate    Severity = 2 // 中度
	SeveritySevere      Severity = 3 // 重度
)

// FeedbackStatus represents the processing status of a feedback record.
type FeedbackStatus int32

const (
	FeedbackStatusPending      FeedbackStatus = 0
	FeedbackStatusReviewed     FeedbackStatus = 1
	FeedbackStatusIncorporated FeedbackStatus = 2
	FeedbackStatusRejected     FeedbackStatus = 3
)

// =============================================================================
// Shared Types
// =============================================================================

// GeoLocation represents a geographic point with administrative hierarchy.
type GeoLocation struct {
	Lat      float64 `json:"lat"`
	Lng      float64 `json:"lng"`
	County   string  `json:"county"`
	Township string  `json:"township"`
}

// BoundingBox represents a detection bounding box.
type BoundingBox struct {
	X int32 `json:"x"`
	Y int32 `json:"y"`
	W int32 `json:"w"`
	H int32 `json:"h"`
}

// SmallTarget represents a detected small target (tiny pest, early lesion).
type SmallTarget struct {
	Type       string      `json:"type"`
	Bbox       BoundingBox `json:"bbox"`
	Confidence float32     `json:"confidence"`
}

// RecognitionResult contains the full AI recognition output for a single image.
type RecognitionResult struct {
	DiseaseName            string        `json:"disease_name"`
	Category               string        `json:"category"` // "病害" or "虫害"
	Confidence             float32       `json:"confidence"`
	Severity               string        `json:"severity"` // "轻度", "中度", "重度"
	SeverityConfidence     float32       `json:"severity_confidence"`
	AffectedPart           string        `json:"affected_part"`
	Symptoms               string        `json:"symptoms"`
	SmallTargetDetected    bool          `json:"small_target_detected"`
	SmallTargets           []SmallTarget `json:"small_targets,omitempty"`
	RecommendationsSummary string        `json:"recommendations_summary,omitempty"`
	KgEntityId             string        `json:"kg_entity_id,omitempty"`
	KgPreventionId         string        `json:"kg_prevention_id,omitempty"`
}

// ExpertFeedback represents correction feedback from a plant expert.
type ExpertFeedback struct {
	OriginalResult    RecognitionResult `json:"original_result"`
	CorrectedPest     string            `json:"corrected_pest"`
	CorrectedSeverity string            `json:"corrected_severity"`
	CorrectionNote    string            `json:"correction_note"`
}

// =============================================================================
// RecordRecognition
// =============================================================================

// RecordRecognitionRequest contains one or more recognition log entries to
// record. Supports high-throughput batch insertion.
type RecordRecognitionRequest struct {
	Logs []RecognitionLogEntry `json:"logs"`
}

// RecognitionLogEntry is a single recognition event to be recorded.
type RecognitionLogEntry struct {
	RequestId           string             `json:"request_id"`
	UserId              string             `json:"user_id,omitempty"`
	Timestamp           int64              `json:"timestamp"` // unix milliseconds
	GeoLocation         *GeoLocation       `json:"geo_location,omitempty"`
	ImageHash           string             `json:"image_hash,omitempty"`
	ImageCount          int32              `json:"image_count"`
	IsBatch             bool               `json:"is_batch"`
	ModelName           string             `json:"model_name,omitempty"`
	ModelVersion        string             `json:"model_version,omitempty"`
	RecognitionResult   *RecognitionResult `json:"recognition_result,omitempty"`
	SmallTargetDetected bool               `json:"small_target_detected"`
	Confidence          float32            `json:"confidence"`
	Severity            string             `json:"severity,omitempty"`
	LatencyMs           int32              `json:"latency_ms"`
	ClientIp            string             `json:"client_ip,omitempty"`
	ClientPlatform      string             `json:"client_platform,omitempty"`
}

// RecordRecognitionResponse returns the status of a batch recording operation.
type RecordRecognitionResponse struct {
	Success       bool   `json:"success"`
	RecordedCount int32  `json:"recorded_count"`
	ErrorMessage  string `json:"error_message,omitempty"`
}

// =============================================================================
// RecordFeedback
// =============================================================================

// RecordFeedbackRequest contains expert feedback for a recognition result.
type RecordFeedbackRequest struct {
	LogId             string `json:"log_id"`
	RequestId         string `json:"request_id"`
	CorrectedPest     string `json:"corrected_pest"`
	CorrectedSeverity string `json:"corrected_severity"`
	CorrectionNote    string `json:"correction_note,omitempty"`
	SubmittedBy       string `json:"submitted_by,omitempty"`
}

// RecordFeedbackResponse returns the status of a feedback recording.
type RecordFeedbackResponse struct {
	Success      bool   `json:"success"`
	FeedbackId   string `json:"feedback_id,omitempty"`
	ErrorMessage string `json:"error_message,omitempty"`
}

// =============================================================================
// GetStats
// =============================================================================

// GetStatsRequest queries statistics with optional filters.
type GetStatsRequest struct {
	County    string `json:"county,omitempty"`
	StartDate string `json:"start_date,omitempty"` // "2006-01-02"
	EndDate   string `json:"end_date,omitempty"`   // "2006-01-02"
	PestType  string `json:"pest_type,omitempty"`
}

// GetStatsResponse returns aggregated statistics.
type GetStatsResponse struct {
	Stats        *StatsData   `json:"stats,omitempty"`
	Heatmap      *HeatmapData `json:"heatmap,omitempty"`
	Trend        *TrendData   `json:"trend,omitempty"`
	ErrorMessage string       `json:"error_message,omitempty"`
}

// StatsData contains aggregated recognition statistics.
type StatsData struct {
	TotalRecognitions int64             `json:"total_recognitions"`
	TotalFeedback     int64             `json:"total_feedback"`
	AvgConfidence     float32           `json:"avg_confidence"`
	AvgLatencyMs      float32           `json:"avg_latency_ms"`
	AccuracyRate      float32           `json:"accuracy_rate"`
	SeverityDist      *SeverityDist     `json:"severity_dist,omitempty"`
	CountyBreakdown   []CountyStatEntry `json:"county_breakdown,omitempty"`
	DailyStats        []DailyStatEntry  `json:"daily_stats,omitempty"`
}

// SeverityDist contains the distribution of severity levels.
type SeverityDist struct {
	Mild     int64 `json:"mild"`
	Moderate int64 `json:"moderate"`
	Severe   int64 `json:"severe"`
}

// CountyStatEntry is per-county aggregated statistics.
type CountyStatEntry struct {
	County  string  `json:"county"`
	Count   int64   `json:"count"`
	AvgConf float32 `json:"avg_conf"`
	TopPest string  `json:"top_pest,omitempty"`
}

// DailyStatEntry is per-day aggregated statistics.
type DailyStatEntry struct {
	Date   string  `json:"date"`
	Count  int64   `json:"count"`
	AvgLat float32 `json:"avg_latency_ms"`
}

// HeatmapData contains per-county pest occurrence data for WebGIS heatmaps.
type HeatmapData struct {
	Points   []HeatmapPoint `json:"points"`
	MaxCount int32          `json:"max_count"`
}

// HeatmapPoint is a single geographic point in the heatmap.
type HeatmapPoint struct {
	County      string  `json:"county"`
	Lat         float64 `json:"lat"`
	Lng         float64 `json:"lng"`
	Count       int32   `json:"count"`
	Intensity   float32 `json:"intensity"` // 0.0-1.0 normalized
	TopPest     string  `json:"top_pest,omitempty"`
	TopSeverity string  `json:"top_severity,omitempty"`
}

// TrendData contains time-series trend data for a pest or overall metrics.
type TrendData struct {
	PestType   string           `json:"pest_type"`
	DataPoints []TrendDataPoint `json:"data_points"`
}

// TrendDataPoint is a single point in a time-series trend.
type TrendDataPoint struct {
	Date          string  `json:"date"` // "2006-01-02"
	Count         int32   `json:"count"`
	AvgConfidence float32 `json:"avg_confidence"`
	Accuracy      float32 `json:"accuracy"`
}

// =============================================================================
// HealthCheck
// =============================================================================

// HealthCheckRequest is an empty request for health checking.
type HealthCheckRequest struct{}

// HealthCheckResponse returns the service health status.
type HealthCheckResponse struct {
	Healthy        bool   `json:"healthy"`
	DbConnected    bool   `json:"db_connected"`
	RedisConnected bool   `json:"redis_connected"`
	UptimeSeconds  int64  `json:"uptime_seconds"`
	Version        string `json:"version"`
}

// =============================================================================
// Service Interface
// =============================================================================

// LogStatisticsServer defines the gRPC server interface that handlers must
// implement.
type LogStatisticsServer interface {
	RecordRecognition(ctx context.Context, req *RecordRecognitionRequest) (*RecordRecognitionResponse, error)
	RecordFeedback(ctx context.Context, req *RecordFeedbackRequest) (*RecordFeedbackResponse, error)
	GetStats(ctx context.Context, req *GetStatsRequest) (*GetStatsResponse, error)
	HealthCheck(ctx context.Context, req *HealthCheckRequest) (*HealthCheckResponse, error)
}

// UnimplementedLogStatisticsServer provides default "not implemented" stubs
// so that a handler struct that embeds this type satisfies the interface
// even when only a subset of RPCs are implemented.
type UnimplementedLogStatisticsServer struct{}

func (UnimplementedLogStatisticsServer) RecordRecognition(context.Context, *RecordRecognitionRequest) (*RecordRecognitionResponse, error) {
	return nil, fmt.Errorf("RecordRecognition not implemented")
}
func (UnimplementedLogStatisticsServer) RecordFeedback(context.Context, *RecordFeedbackRequest) (*RecordFeedbackResponse, error) {
	return nil, fmt.Errorf("RecordFeedback not implemented")
}
func (UnimplementedLogStatisticsServer) GetStats(context.Context, *GetStatsRequest) (*GetStatsResponse, error) {
	return nil, fmt.Errorf("GetStats not implemented")
}
func (UnimplementedLogStatisticsServer) HealthCheck(context.Context, *HealthCheckRequest) (*HealthCheckResponse, error) {
	return nil, fmt.Errorf("HealthCheck not implemented")
}

// Compile-time interface satisfaction guard.
var _ LogStatisticsServer = (*UnimplementedLogStatisticsServer)(nil)

// =============================================================================
// gRPC Registration — Adapter
// =============================================================================

// ServiceDesc describes the gRPC service for reflection-based registration.
var ServiceDesc = grpc.ServiceDesc{
	ServiceName: "jujube.logstatistics.LogStatistics",
	HandlerType: (*LogStatisticsServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "RecordRecognition",
			Handler:    recordRecognitionHandler,
		},
		{
			MethodName: "RecordFeedback",
			Handler:    recordFeedbackHandler,
		},
		{
			MethodName: "GetStats",
			Handler:    getStatsHandler,
		},
		{
			MethodName: "HealthCheck",
			Handler:    healthCheckHandler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "log_statistics.proto",
}

// typedHandler adapts a LogStatisticsServer method to grpc.MethodHandler.
func recordRecognitionHandler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(RecordRecognitionRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(LogStatisticsServer).RecordRecognition(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/jujube.logstatistics.LogStatistics/RecordRecognition",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(LogStatisticsServer).RecordRecognition(ctx, req.(*RecordRecognitionRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func recordFeedbackHandler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(RecordFeedbackRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(LogStatisticsServer).RecordFeedback(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/jujube.logstatistics.LogStatistics/RecordFeedback",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(LogStatisticsServer).RecordFeedback(ctx, req.(*RecordFeedbackRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func getStatsHandler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(GetStatsRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(LogStatisticsServer).GetStats(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/jujube.logstatistics.LogStatistics/GetStats",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(LogStatisticsServer).GetStats(ctx, req.(*GetStatsRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func healthCheckHandler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(HealthCheckRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(LogStatisticsServer).HealthCheck(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/jujube.logstatistics.LogStatistics/HealthCheck",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(LogStatisticsServer).HealthCheck(ctx, req.(*HealthCheckRequest))
	}
	return interceptor(ctx, in, info, handler)
}

