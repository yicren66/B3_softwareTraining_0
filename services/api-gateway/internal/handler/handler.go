// Package handler defines all API Gateway routes and wires them to backend proxies.
package handler

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/jujube-platform/api-gateway/internal/auth"
	"github.com/jujube-platform/api-gateway/internal/config"
	"github.com/jujube-platform/api-gateway/internal/middleware"
	"github.com/jujube-platform/api-gateway/internal/proxy"
)

// Handler holds all dependencies for route handling.
type Handler struct {
	cfg        *config.Config
	jwtManager *auth.JWTManager
	proxies    map[string]*proxy.ServiceProxy
}

// New creates a new Handler with configured proxies.
func New(cfg *config.Config) *Handler {
	return &Handler{
		cfg:        cfg,
		jwtManager: auth.NewJWTManager(cfg.JWTSecretKey, cfg.JWTExpireHours, cfg.JWTRefreshHours),
		proxies: map[string]*proxy.ServiceProxy{
			"image-recognition": proxy.NewServiceProxy(cfg.ImageRecognitionURL),
			"knowledge-graph":   proxy.NewServiceProxy(cfg.KnowledgeGraphURL),
			"reasoning-engine":  proxy.NewServiceProxy(cfg.ReasoningEngineURL),
			"model-management":  proxy.NewServiceProxy(cfg.ModelManagementURL),
			"log-statistics":    proxy.NewServiceProxy(cfg.LogStatisticsURL),
		},
	}
}

// SetupRoutes configures all routes on the Gin engine.
func (h *Handler) SetupRoutes(r *gin.Engine) {
	// ------------------------------------------------------------------
	// Global middleware
	// ------------------------------------------------------------------
	r.Use(middleware.CORSMiddleware())
	r.Use(middleware.RequestIDMiddleware())
	r.Use(middleware.LoggerMiddleware())
	r.Use(middleware.RateLimitMiddleware(h.cfg))
	r.Use(gin.Recovery())

	// ------------------------------------------------------------------
	// Public endpoints
	// ------------------------------------------------------------------
	r.GET("/health", h.aggregatedHealth)
	r.GET("/", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"service": "jujube-api-gateway",
			"version": "0.1.0",
			"docs":    "/docs",
		})
	})

	// Auth endpoints (public)
	authGroup := r.Group("/api/v1/auth")
	{
		authGroup.POST("/login", h.login)
		authGroup.POST("/refresh", h.refreshToken)
	}

	// ------------------------------------------------------------------
	// Image Recognition — /api/v1/recognition/*
	// ------------------------------------------------------------------
	ir := r.Group("/api/v1/recognition")
	ir.Use(middleware.AuthMiddleware(h.jwtManager))
	{
		ir.POST("/detect", h.proxyTo("image-recognition", "/api/v1/recognition/single"))
		ir.POST("/batch", h.proxyTo("image-recognition", "/api/v1/recognition/batch"))
		ir.GET("/result/:task_id", h.proxyTo("image-recognition"))
		ir.POST("/feedback", h.proxyTo("image-recognition", "/api/v1/recognition/feedback"))
	}

	// ------------------------------------------------------------------
	// Knowledge Graph — /api/v1/kg/*
	// ------------------------------------------------------------------
	kg := r.Group("/api/v1/kg")
	kg.Use(middleware.OptionalAuthMiddleware(h.jwtManager))
	{
		kg.GET("/entity/:entity_id", h.proxyTo("knowledge-graph"))
		kg.POST("/search", h.proxyTo("knowledge-graph", "/api/v1/kg/search"))
		kg.GET("/recommendation/:pest_id", h.proxyTo("knowledge-graph"))
		kg.POST("/qa", h.proxyTo("knowledge-graph", "/api/v1/kg/qa"))
		kg.GET("/risk/predict", h.proxyTo("knowledge-graph", "/api/v1/kg/risk/predict"))
	}

	// ------------------------------------------------------------------
	// Reasoning Engine — /api/v1/reasoning/*
	// ------------------------------------------------------------------
	reasoning := r.Group("/api/v1/reasoning")
	reasoning.Use(middleware.AuthMiddleware(h.jwtManager))
	{
		reasoning.POST("/prevention-plan", h.proxyTo("reasoning-engine", "/api/v1/reasoning/prevention-plan"))
		reasoning.POST("/qa", h.proxyTo("reasoning-engine", "/api/v1/reasoning/qa"))
	}

	// ------------------------------------------------------------------
	// Model Management — /api/v1/model/*  (admin/专家 only)
	// ------------------------------------------------------------------
	model := r.Group("/api/v1/model")
	model.Use(middleware.AuthMiddleware(h.jwtManager))
	model.Use(middleware.RequireRole("植保专家"))
	{
		model.GET("/versions", h.proxyTo("model-management", "/api/v1/model/versions"))
		model.GET("/versions/:model_type/:version", h.proxyTo("model-management"))
		model.POST("/register", h.proxyTo("model-management", "/api/v1/model/register"))
		model.POST("/deploy", h.proxyTo("model-management", "/api/v1/model/deploy"))
		model.POST("/rollback", h.proxyTo("model-management", "/api/v1/model/rollback"))
		model.POST("/train", h.proxyTo("model-management", "/api/v1/model/train"))
		model.GET("/deployments", h.proxyTo("model-management", "/api/v1/model/deployments"))
		model.POST("/ab-test", h.proxyTo("model-management", "/api/v1/model/ab-test"))
	}

	// ------------------------------------------------------------------
	// Statistics — /api/v1/stats/*  (admin only)
	// ------------------------------------------------------------------
	stats := r.Group("/api/v1/stats")
	stats.Use(middleware.AuthMiddleware(h.jwtManager))
	stats.Use(middleware.RequireRole("工作管理员"))
	{
		stats.GET("/dashboard", h.proxyTo("log-statistics"))
		stats.GET("/recognition-logs", h.proxyTo("log-statistics"))
	}
}

// proxyTo returns a Gin handler that forwards to a named backend service.
// Optional extraPathSegments are appended to the incoming request path.
func (h *Handler) proxyTo(serviceName string, extraPathSegments ...string) gin.HandlerFunc {
	srv := h.proxies[serviceName]
	if srv == nil {
		return func(c *gin.Context) {
			c.JSON(http.StatusInternalServerError, gin.H{
				"code":    500,
				"message": "unknown backend service: " + serviceName,
			})
		}
	}

	return func(c *gin.Context) {
		targetPath := c.Request.URL.Path
		if len(extraPathSegments) > 0 && extraPathSegments[0] != "" {
			targetPath = extraPathSegments[0]
			// If the path has a parameter like :entity_id, replace it
			for _, param := range c.Params {
				targetPath = c.Request.URL.Path // fall back to original path with params
			}
		}
		srv.Forward(c, targetPath)
	}
}

// ---------------------------------------------------------------------------
// Auth handlers
// ---------------------------------------------------------------------------

type loginRequest struct {
	Username string `json:"username" binding:"required"`
	Password string `json:"password" binding:"required"`
}

type loginResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	ExpiresIn    int    `json:"expires_in"`
	TokenType    string `json:"token_type"`
	UserID       string `json:"user_id"`
	Username     string `json:"username"`
	Role         string `json:"role"`
}

func (h *Handler) login(c *gin.Context) {
	var req loginRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"code": 400, "message": "invalid request"})
		return
	}

	// TODO: Validate against user management module / database
	// Stub: accept any non-empty credentials with a default role
	if req.Username == "" || req.Password == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"code": 401, "message": "invalid credentials"})
		return
	}

	// Stub role assignment (production: query from user DB)
	role := "枣农"
	userID := "user-" + req.Username
	if req.Username == "admin" {
		role = "超级管理员"
	} else if req.Username == "expert" {
		role = "植保专家"
	}

	accessToken, err := h.jwtManager.GenerateToken(userID, req.Username, role)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"code": 500, "message": "token generation failed"})
		return
	}

	refreshToken, err := h.jwtManager.GenerateRefreshToken(userID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"code": 500, "message": "token generation failed"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"code":    0,
		"message": "success",
		"data": loginResponse{
			AccessToken:  accessToken,
			RefreshToken: refreshToken,
			ExpiresIn:    h.cfg.JWTExpireHours * 3600,
			TokenType:    "Bearer",
			UserID:       userID,
			Username:     req.Username,
			Role:         role,
		},
	})
}

type refreshRequest struct {
	RefreshToken string `json:"refresh_token" binding:"required"`
}

func (h *Handler) refreshToken(c *gin.Context) {
	var req refreshRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"code": 400, "message": "invalid request"})
		return
	}

	claims, err := h.jwtManager.ValidateToken(req.RefreshToken)
	if err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"code": 401, "message": "invalid refresh token"})
		return
	}

	accessToken, err := h.jwtManager.GenerateToken(claims.UserID, claims.Username, claims.Role)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"code": 500, "message": "token generation failed"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"code":    0,
		"message": "success",
		"data": gin.H{
			"access_token": accessToken,
			"expires_in":   h.cfg.JWTExpireHours * 3600,
			"token_type":   "Bearer",
		},
	})
}

// ---------------------------------------------------------------------------
// Aggregated health check
// ---------------------------------------------------------------------------

func (h *Handler) aggregatedHealth(c *gin.Context) {
	statuses := make(map[string]string)
	allHealthy := true

	for name, srv := range h.proxies {
		if srv.HealthCheck(c) {
			statuses[name] = "healthy"
		} else {
			statuses[name] = "unhealthy"
			allHealthy = false
		}
	}

	status := http.StatusOK
	if !allHealthy {
		status = http.StatusServiceUnavailable
	}

	c.JSON(status, gin.H{
		"service":   "api-gateway",
		"status":    statuses,
		"timestamp": time.Now().UTC().Format(time.RFC3339),
	})
}
