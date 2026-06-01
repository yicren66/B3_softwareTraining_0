// Package main is the entry point for the Log & Statistics service.
// It initializes all infrastructure (logger, config, PostgreSQL, Redis),
// starts the gRPC server on port 9005, the HTTP health/metrics server on
// port 8005, and handles OS-signal-based graceful shutdown.
package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/redis/go-redis/v9"
	"github.com/rs/zerolog"

	grpcserver "github.com/jujube-platform/log-statistics/internal/grpc"
	pb "github.com/jujube-platform/log-statistics/internal/grpc/proto"
	"github.com/jujube-platform/log-statistics/internal/handler"
	"github.com/jujube-platform/log-statistics/internal/repository"
	"github.com/jujube-platform/log-statistics/internal/service"
)

// =============================================================================
// Configuration
// =============================================================================

// config holds all environment-derived configuration values.
type config struct {
	Environment string
	LogLevel    string

	// gRPC
	GRPCPort int

	// HTTP health / metrics
	HTTPPort int

	// PostgreSQL
	DBHost     string
	DBPort     int
	DBName     string
	DBUser     string
	DBPassword string

	// Redis
	RedisHost string
	RedisPort int
	RedisDB   int
	RedisPass string
}

// loadConfig reads configuration from environment variables with sensible
// defaults for local development.
func loadConfig() *config {
	return &config{
		Environment: envStr("ENVIRONMENT", "development"),
		LogLevel:    envStr("LOG_LEVEL", "debug"),

		GRPCPort: envInt("GRPC_PORT", 9005),
		HTTPPort: envInt("HTTP_PORT", 8005),

		DBHost:     envStr("DB_HOST", "localhost"),
		DBPort:     envInt("DB_PORT", 5432),
		DBName:     envStr("DB_NAME", "jujube_platform"),
		DBUser:     envStr("DB_USER", "jujube"),
		DBPassword: envStr("DB_PASSWORD", "changeme"),

		RedisHost: envStr("REDIS_HOST", "localhost"),
		RedisPort: envInt("REDIS_PORT", 6379),
		RedisDB:   envInt("REDIS_DB", 0),
		RedisPass: envStr("REDIS_PASSWORD", ""),
	}
}

// =============================================================================
// Main
// =============================================================================

func main() {
	// ------------------------------------------------------------------
	// 1. Logger
	// ------------------------------------------------------------------
	cfg := loadConfig()

	level, err := zerolog.ParseLevel(cfg.LogLevel)
	if err != nil {
		level = zerolog.InfoLevel
	}

	log := zerolog.New(zerolog.ConsoleWriter{
		Out:        os.Stderr,
		TimeFormat: time.RFC3339,
	}).Level(level).With().Timestamp().Caller().Logger()

	log.Info().
		Str("env", cfg.Environment).
		Str("log_level", level.String()).
		Int("grpc_port", cfg.GRPCPort).
		Int("http_port", cfg.HTTPPort).
		Msg("Log & Statistics Service starting")

	// ------------------------------------------------------------------
	// 2. PostgreSQL connection pool
	// ------------------------------------------------------------------
	dbURL := fmt.Sprintf(
		"postgres://%s:%s@%s:%d/%s?sslmode=disable&pool_max_conns=20&pool_min_conns=4",
		cfg.DBUser, cfg.DBPassword, cfg.DBHost, cfg.DBPort, cfg.DBName,
	)

	poolCfg, err := pgxpool.ParseConfig(dbURL)
	if err != nil {
		log.Fatal().Err(err).Msg("failed to parse database URL")
	}
	poolCfg.MaxConns = 20
	poolCfg.MinConns = 4
	poolCfg.MaxConnLifetime = 30 * time.Minute
	poolCfg.MaxConnIdleTime = 5 * time.Minute
	poolCfg.HealthCheckPeriod = 30 * time.Second

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	pool, err := pgxpool.NewWithConfig(ctx, poolCfg)
	if err != nil {
		log.Fatal().Err(err).Msg("failed to create PostgreSQL connection pool")
	}
	defer pool.Close()

	if err := pool.Ping(ctx); err != nil {
		log.Fatal().Err(err).Msg("failed to ping PostgreSQL")
	}
	log.Info().Msg("PostgreSQL connection pool established")

	// ------------------------------------------------------------------
	// 3. Redis client
	// ------------------------------------------------------------------
	rdb := redis.NewClient(&redis.Options{
		Addr:     fmt.Sprintf("%s:%d", cfg.RedisHost, cfg.RedisPort),
		Password: cfg.RedisPass,
		DB:       cfg.RedisDB,
		PoolSize: 10,
	})

	if err := rdb.Ping(ctx).Err(); err != nil {
		log.Warn().Err(err).Msg("Redis connection failed — running without cache")
	} else {
		log.Info().Msg("Redis connection established")
	}

	// ------------------------------------------------------------------
	// 4. Repository, Service, Handlers
	// ------------------------------------------------------------------
	repo := repository.NewPostgresRepo(pool, log)
	svc := service.NewLogService(repo, rdb, log)

	// The StatsHandler implements all four RPCs (RecordRecognition,
	// RecordFeedback, GetStats, HealthCheck) by composing LogHandler.
	logHandler := handler.NewLogHandler(svc, log)
	statsHandler := handler.NewStatsHandler(svc, log)

	// Build a combined service implementation: StatsHandler covers
	// GetStats; LogHandler covers RecordRecognition, RecordFeedback;
	// both have HealthCheck — we choose the StatsHandler version.
	_ = logHandler // unused in multiplexing; we embed into the multiplexer.

	grpcSvc := &compositeServer{
		LogHandler:   logHandler,
		StatsHandler: statsHandler,
	}

	// ------------------------------------------------------------------
	// 5. gRPC server
	// ------------------------------------------------------------------
	grpcAddr := fmt.Sprintf("0.0.0.0:%d", cfg.GRPCPort)
	grpcSrv, err := grpcserver.NewServer(grpcAddr, grpcSvc, log)
	if err != nil {
		log.Fatal().Err(err).Msg("failed to create gRPC server")
	}

	// ------------------------------------------------------------------
	// 6. HTTP health / metrics server
	// ------------------------------------------------------------------
	httpMux := http.NewServeMux()

	// Health endpoint
	httpMux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		resp, err := svc.HealthCheck(r.Context())
		if err != nil || !resp.Healthy {
			w.WriteHeader(http.StatusServiceUnavailable)
			fmt.Fprintf(w, `{"status":"unhealthy"}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `{"status":"healthy","version":"%s","uptime_seconds":%d}`,
			resp.Version, resp.UptimeSeconds)
	})

	// Prometheus metrics endpoint
	httpMux.Handle("/metrics", promhttp.Handler())

	httpAddr := fmt.Sprintf("0.0.0.0:%d", cfg.HTTPPort)
	httpSrv := &http.Server{
		Addr:         httpAddr,
		Handler:      httpMux,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Start HTTP server in background.
	go func() {
		log.Info().Str("addr", httpAddr).Msg("HTTP health/metrics server listening")
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal().Err(err).Msg("HTTP server error")
		}
	}()

	// Start gRPC server in background.
	go func() {
		if err := grpcSrv.Start(); err != nil {
			log.Fatal().Err(err).Msg("gRPC server error")
		}
	}()

	log.Info().Msg("Log & Statistics Service is ready")

	// ------------------------------------------------------------------
	// 7. Graceful shutdown
	// ------------------------------------------------------------------
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	sig := <-quit

	log.Info().Str("signal", sig.String()).Msg("shutting down...")

	// Shut down HTTP server first (stop accepting new health checks).
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer shutdownCancel()

	if err := httpSrv.Shutdown(shutdownCtx); err != nil {
		log.Error().Err(err).Msg("HTTP server shutdown error")
	}

	// Shut down gRPC server (wait for in-flight RPCs).
	grpcSrv.GracefulStop(10 * time.Second)

	// Close Redis.
	if err := rdb.Close(); err != nil {
		log.Error().Err(err).Msg("Redis close error")
	}

	// PostgreSQL pool is closed via the deferred pool.Close() above (runs after
	// the function scope ends, but we also trigger it explicitly for clarity).
	pool.Close()

	log.Info().Msg("Log & Statistics Service stopped")
}

// =============================================================================
// compositeServer multiplexes across LogHandler and StatsHandler
// =============================================================================

// compositeServer implements pb.LogStatisticsServer by delegating each RPC
// to the appropriate handler. RecordRecognition and RecordFeedback go to
// LogHandler; GetStats goes to StatsHandler; HealthCheck is served by both
// but we prefer StatsHandler (which in turn delegates to the service, same
// as LogHandler).
type compositeServer struct {
	pb.UnimplementedLogStatisticsServer
	*handler.LogHandler
	*handler.StatsHandler
}

// RecordRecognition delegates to LogHandler.
func (c *compositeServer) RecordRecognition(ctx context.Context, req *pb.RecordRecognitionRequest) (*pb.RecordRecognitionResponse, error) {
	return c.LogHandler.RecordRecognition(ctx, req)
}

// RecordFeedback delegates to LogHandler.
func (c *compositeServer) RecordFeedback(ctx context.Context, req *pb.RecordFeedbackRequest) (*pb.RecordFeedbackResponse, error) {
	return c.LogHandler.RecordFeedback(ctx, req)
}

// GetStats delegates to StatsHandler.
func (c *compositeServer) GetStats(ctx context.Context, req *pb.GetStatsRequest) (*pb.GetStatsResponse, error) {
	return c.StatsHandler.GetStats(ctx, req)
}

// HealthCheck delegates to StatsHandler (which uses service.HealthCheck).
func (c *compositeServer) HealthCheck(ctx context.Context, req *pb.HealthCheckRequest) (*pb.HealthCheckResponse, error) {
	return c.StatsHandler.HealthCheck(ctx, req)
}

// =============================================================================
// Environment helpers
// =============================================================================

func envStr(key, defaultVal string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return defaultVal
}

func envInt(key string, defaultVal int) int {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		n, err := strconv.Atoi(v)
		if err == nil {
			return n
		}
	}
	return defaultVal
}
