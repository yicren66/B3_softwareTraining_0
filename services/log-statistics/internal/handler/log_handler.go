// Package handler contains the gRPC handler implementations that bridge
// protobuf request/response messages to the service business-logic layer.
package handler

import (
	"context"

	"github.com/rs/zerolog"

	pb "github.com/jujube-platform/log-statistics/internal/grpc/proto"
	"github.com/jujube-platform/log-statistics/internal/service"
)

// =============================================================================
// LogHandler
// =============================================================================

// LogHandler implements the RecordRecognition, RecordFeedback, and
// HealthCheck RPCs for the Log & Statistics service.
type LogHandler struct {
	svc *service.LogService
	log zerolog.Logger
}

// NewLogHandler creates a LogHandler backed by the given service.
func NewLogHandler(svc *service.LogService, log zerolog.Logger) *LogHandler {
	return &LogHandler{svc: svc, log: log}
}

// =============================================================================
// RecordRecognition
// =============================================================================

// RecordRecognition handles high-throughput recording of one or more
// recognition log entries. It delegates to LogService.RecordRecognition
// which in turn uses pgx batch insertion with retry.
func (h *LogHandler) RecordRecognition(ctx context.Context, req *pb.RecordRecognitionRequest) (*pb.RecordRecognitionResponse, error) {
	h.log.Debug().Int("batch_size", len(req.Logs)).Msg("RecordRecognition called")

	resp, err := h.svc.RecordRecognition(ctx, req)
	if err != nil {
		h.log.Error().Err(err).Msg("RecordRecognition service error")
		return &pb.RecordRecognitionResponse{
			Success:       false,
			RecordedCount: 0,
			ErrorMessage:  err.Error(),
		}, nil // return structured error, not gRPC error
	}

	h.log.Debug().Int32("recorded", resp.RecordedCount).Msg("RecordRecognition completed")
	return resp, nil
}

// =============================================================================
// RecordFeedback
// =============================================================================

// RecordFeedback handles expert feedback submission for a recognition result.
// It looks up the associated recognition log and persists the correction.
func (h *LogHandler) RecordFeedback(ctx context.Context, req *pb.RecordFeedbackRequest) (*pb.RecordFeedbackResponse, error) {
	h.log.Debug().Str("request_id", req.RequestId).Str("log_id", req.LogId).Msg("RecordFeedback called")

	resp, err := h.svc.RecordFeedback(ctx, req)
	if err != nil {
		h.log.Error().Err(err).Msg("RecordFeedback service error")
		return &pb.RecordFeedbackResponse{
			Success:      false,
			ErrorMessage: err.Error(),
		}, nil
	}

	h.log.Debug().Str("feedback_id", resp.FeedbackId).Msg("RecordFeedback completed")
	return resp, nil
}

// =============================================================================
// HealthCheck
// =============================================================================

// HealthCheck returns the service's current health and connectivity status.
func (h *LogHandler) HealthCheck(ctx context.Context, req *pb.HealthCheckRequest) (*pb.HealthCheckResponse, error) {
	return h.svc.HealthCheck(ctx)
}
