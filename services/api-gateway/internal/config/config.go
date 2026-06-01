// Package config holds all environment-derived configuration for the API Gateway.
package config

import (
	"os"
	"strconv"
	"time"
)

// Config holds all configuration values with sensible defaults.
type Config struct {
	Environment string
	LogLevel    string

	// Server
	HTTPPort    int
	GRPCPort    int
	ReadTimeout time.Duration

	// JWT
	JWTSecretKey     string
	JWTExpireHours   int
	JWTRefreshHours  int

	// Rate limiting
	RateLimitEnabled  bool
	RateLimitPerMinute int

	// Microservice endpoints
	ImageRecognitionURL string
	KnowledgeGraphURL   string
	ReasoningEngineURL  string
	ModelManagementURL  string
	LogStatisticsURL    string
}

// Load reads configuration from environment variables.
func Load() *Config {
	return &Config{
		Environment:  envStr("ENVIRONMENT", "development"),
		LogLevel:     envStr("LOG_LEVEL", "info"),

		HTTPPort:     envInt("API_GATEWAY_PORT", 8000),
		GRPCPort:     envInt("GRPC_PORT", 9000),
		ReadTimeout:  time.Duration(envInt("READ_TIMEOUT", 30)) * time.Second,

		JWTSecretKey:    envStr("JWT_SECRET_KEY", "change-me-in-production"),
		JWTExpireHours:  envInt("JWT_EXPIRE_HOURS", 24),
		JWTRefreshHours: envInt("JWT_REFRESH_HOURS", 168),

		RateLimitEnabled:   envBool("RATE_LIMIT_ENABLED", true),
		RateLimitPerMinute: envInt("RATE_LIMIT_PER_MINUTE", 100),

		ImageRecognitionURL: envStr("IMAGE_RECOGNITION_URL", "http://image-recognition:8001"),
		KnowledgeGraphURL:   envStr("KNOWLEDGE_GRAPH_URL", "http://knowledge-graph:8002"),
		ReasoningEngineURL:  envStr("REASONING_ENGINE_URL", "http://reasoning-engine:8003"),
		ModelManagementURL:  envStr("MODEL_MANAGEMENT_URL", "http://model-management:8004"),
		LogStatisticsURL:    envStr("LOG_STATISTICS_URL", "http://log-statistics:8005"),
	}
}

func envStr(key, defaultVal string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return defaultVal
}

func envInt(key string, defaultVal int) int {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return defaultVal
}

func envBool(key string, defaultVal bool) bool {
	if v, ok := os.LookupEnv(key); ok {
		switch v {
		case "1", "true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON":
			return true
		case "0", "false", "False", "FALSE", "no", "No", "NO", "off", "Off", "OFF":
			return false
		}
	}
	return defaultVal
}
