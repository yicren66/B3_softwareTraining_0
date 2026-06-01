// Package middleware provides HTTP middleware for the API Gateway.
package middleware

import (
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ulule/limiter/v3"
	mgin "github.com/ulule/limiter/v3/drivers/middleware/gin"
	"github.com/ulule/limiter/v3/drivers/store/memory"

	"github.com/jujube-platform/api-gateway/internal/auth"
	"github.com/jujube-platform/api-gateway/internal/config"
)

// CORSMiddleware handles Cross-Origin Resource Sharing.
func CORSMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, PATCH, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Origin, Content-Type, Accept, Authorization, X-Request-ID")
		c.Header("Access-Control-Expose-Headers", "Content-Length, X-Request-ID")
		c.Header("Access-Control-Max-Age", "86400")

		if c.Request.Method == http.MethodOptions {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}
		c.Next()
	}
}

// RequestIDMiddleware injects or propagates an X-Request-ID header.
func RequestIDMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		requestID := c.GetHeader("X-Request-ID")
		if requestID == "" {
			requestID = generateRequestID()
		}
		c.Set("request_id", requestID)
		c.Header("X-Request-ID", requestID)
		c.Next()
	}
}

// LoggerMiddleware logs every request with method, path, status, and latency.
func LoggerMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		path := c.Request.URL.Path
		query := c.Request.URL.RawQuery

		c.Next()

		latency := time.Since(start)
		statusCode := c.Writer.Status()
		clientIP := c.ClientIP()
		method := c.Request.Method

		if query != "" {
			path = path + "?" + query
		}

		log.Printf("[API Gateway] %s | %3d | %12v | %15s | %s",
			method, statusCode, latency, clientIP, path,
		)
	}
}

// AuthMiddleware validates JWT tokens and attaches user claims to context.
func AuthMiddleware(jwtManager *auth.JWTManager) gin.HandlerFunc {
	return func(c *gin.Context) {
		authHeader := c.GetHeader("Authorization")
		token, err := auth.ExtractBearerToken(authHeader)
		if err != nil {
			c.JSON(http.StatusUnauthorized, gin.H{
				"code":    401,
				"message": err.Error(),
			})
			c.Abort()
			return
		}

		claims, err := jwtManager.ValidateToken(token)
		if err != nil {
			status := http.StatusUnauthorized
			if err == auth.ErrTokenExpired {
				status = http.StatusUnauthorized
			}
			c.JSON(status, gin.H{
				"code":    401,
				"message": err.Error(),
			})
			c.Abort()
			return
		}

		// Attach user info to context
		c.Set("user_id", claims.UserID)
		c.Set("username", claims.Username)
		c.Set("user_role", claims.Role)
		c.Next()
	}
}

// OptionalAuthMiddleware attempts JWT validation but doesn't fail if missing.
// Used for endpoints that work with or without authentication (e.g., public search).
func OptionalAuthMiddleware(jwtManager *auth.JWTManager) gin.HandlerFunc {
	return func(c *gin.Context) {
		authHeader := c.GetHeader("Authorization")
		if authHeader == "" {
			c.Set("user_role", "访客")
			c.Next()
			return
		}

		token, err := auth.ExtractBearerToken(authHeader)
		if err != nil {
			c.Set("user_role", "访客")
			c.Next()
			return
		}

		claims, err := jwtManager.ValidateToken(token)
		if err != nil {
			c.Set("user_role", "访客")
			c.Next()
			return
		}

		c.Set("user_id", claims.UserID)
		c.Set("username", claims.Username)
		c.Set("user_role", claims.Role)
		c.Next()
	}
}

// RequireRole returns middleware that checks the user has at least the given role.
func RequireRole(minRole string) gin.HandlerFunc {
	return func(c *gin.Context) {
		role, exists := c.Get("user_role")
		if !exists {
			c.JSON(http.StatusForbidden, gin.H{
				"code":    403,
				"message": "access denied: no role assigned",
			})
			c.Abort()
			return
		}

		roleStr, ok := role.(string)
		if !ok || !auth.HasMinRole(roleStr, minRole) {
			c.JSON(http.StatusForbidden, gin.H{
				"code":    403,
				"message": "access denied: insufficient privileges",
			})
			c.Abort()
			return
		}
		c.Next()
	}
}

// RateLimitMiddleware applies per-IP rate limiting.
func RateLimitMiddleware(cfg *config.Config) gin.HandlerFunc {
	if !cfg.RateLimitEnabled {
		return func(c *gin.Context) { c.Next() }
	}

	rate := limiter.Rate{
		Period: 1 * time.Minute,
		Limit:  int64(cfg.RateLimitPerMinute),
	}
	store := memory.NewStore()
	instance := limiter.New(store, rate)
	return mgin.NewMiddleware(instance)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func generateRequestID() string {
	return "req-" + strings.ReplaceAll(time.Now().UTC().Format("20060102150405.000000"), ".", "")
}
