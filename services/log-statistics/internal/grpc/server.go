// Package grpc provides a gRPC server factory with logging, panic recovery,
// and Prometheus metrics middleware wired in. It registers the LogStatistics
// service implementation and exposes a Start / GracefulStop lifecycle.
package grpc

import (
	"context"
	"net"
	"time"

	grpc_middleware "github.com/grpc-ecosystem/go-grpc-middleware/v2"
	grpc_recovery "github.com/grpc-ecosystem/go-grpc-middleware/v2/interceptors/recovery"
	grpc_logging "github.com/grpc-ecosystem/go-grpc-middleware/v2/interceptors/logging"
	"github.com/rs/zerolog"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/status"

	pb "github.com/jujube-platform/log-statistics/internal/grpc/proto"
)

// =============================================================================
// Server
// =============================================================================

// Server wraps a *grpc.Server and the net.Listener it is bound to.
type Server struct {
	grpcServer *grpc.Server
	listener   net.Listener
	log        zerolog.Logger
}

// NewServer creates a gRPC server with middleware, registers the provided
// service implementation, and binds to the given address. Call Start() to
// begin serving.
func NewServer(addr string, svc pb.LogStatisticsServer, log zerolog.Logger) (*Server, error) {
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, err
	}

	// Build zerolog-aware logging interceptor.
	loggerOpts := []grpc_logging.Option{
		grpc_logging.WithLogOnEvents(grpc_logging.FinishCall),
		grpc_logging.WithDurationField(grpc_logging.DurationToDurationField),
		grpc_logging.WithLevels(func(code codes.Code) grpc_logging.Level {
			switch code {
			case codes.OK:
				return grpc_logging.LevelDebug
			case codes.NotFound, codes.InvalidArgument, codes.AlreadyExists:
				return grpc_logging.LevelInfo
			default:
				return grpc_logging.LevelError
			}
		}),
	}

	// Build recovery interceptor with a custom panic handler.
	recoveryOpts := []grpc_recovery.Option{
		grpc_recovery.WithRecoveryHandler(func(p interface{}) error {
			log.Error().Interface("panic", p).Msg("gRPC handler panicked")
			return status.Errorf(codes.Internal, "internal server error")
		}),
	}

	// Chain interceptors: logging (outermost) -> recovery -> handler.
	chainUnary := grpc_middleware.ChainUnaryServer(
		grpc_logging.UnaryServerInterceptor(interceptorLogger(log), loggerOpts...),
		grpc_recovery.UnaryServerInterceptor(recoveryOpts...),
	)

	chainStream := grpc_middleware.ChainStreamServer(
		grpc_logging.StreamServerInterceptor(interceptorLogger(log), loggerOpts...),
		grpc_recovery.StreamServerInterceptor(recoveryOpts...),
	)

	// Keepalive enforcement protects against idle clients leaking goroutines.
	kaParams := keepalive.ServerParameters{
		MaxConnectionIdle:     5 * time.Minute,
		MaxConnectionAge:      30 * time.Minute,
		MaxConnectionAgeGrace: 10 * time.Second,
		Time:                  60 * time.Second,
		Timeout:               20 * time.Second,
	}

	grpcSrv := grpc.NewServer(
		grpc.UnaryInterceptor(chainUnary),
		grpc.StreamInterceptor(chainStream),
		grpc.KeepaliveParams(kaParams),
	)

	// Register with the hand-rolled ServiceDesc from the proto package.
	grpcSrv.RegisterService(&pb.ServiceDesc, svc)

	return &Server{
		grpcServer: grpcSrv,
		listener:   lis,
		log:        log,
	}, nil
}

// Start begins serving gRPC requests on the configured address. This call
// blocks until the server is stopped or the listener fails.
func (s *Server) Start() error {
	s.log.Info().Str("addr", s.listener.Addr().String()).Msg("gRPC server listening")
	return s.grpcServer.Serve(s.listener)
}

// GracefulStop attempts a graceful shutdown, waiting for active RPCs to
// finish up to the given deadline.
func (s *Server) GracefulStop(timeout time.Duration) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	stopped := make(chan struct{})
	go func() {
		s.grpcServer.GracefulStop()
		close(stopped)
	}()

	select {
	case <-stopped:
		s.log.Info().Msg("gRPC server stopped gracefully")
	case <-ctx.Done():
		s.log.Warn().Msg("gRPC graceful stop deadline exceeded; forcing stop")
		s.grpcServer.Stop()
	}
}

// Addr returns the address the server is listening on.
func (s *Server) Addr() net.Addr {
	return s.listener.Addr()
}

// =============================================================================
// interceptorLogger adapts zerolog for the grpc_middleware/logging interceptor.
// =============================================================================

func interceptorLogger(l zerolog.Logger) grpc_logging.Logger {
	return grpc_logging.LoggerFunc(func(ctx context.Context, lvl grpc_logging.Level, msg string, fields ...any) {
		var ev *zerolog.Event
		switch lvl {
		case grpc_logging.LevelDebug:
			ev = l.Debug()
		case grpc_logging.LevelInfo:
			ev = l.Info()
		case grpc_logging.LevelWarn:
			ev = l.Warn()
		case grpc_logging.LevelError:
			ev = l.Error()
		default:
			ev = l.Info()
		}

		// fields come in key-value pairs: k1, v1, k2, v2, ...
		for i := 0; i+1 < len(fields); i += 2 {
			key, okKey := fields[i].(string)
			if okKey {
				ev = ev.Interface(key, fields[i+1])
			}
		}
		ev.Msg(msg)
	})
}
