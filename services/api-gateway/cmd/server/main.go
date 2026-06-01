// Package main is the entry point for the API Gateway.
//
// The API Gateway is the central ingress point for all client traffic.  It
// handles authentication (JWT), rate limiting, request routing (reverse
// proxy to microservices), CORS, and aggregated health checks.
//
// Port: 8000 (HTTP) / 9000 (gRPC reserved)
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/jujube-platform/api-gateway/internal/config"
	"github.com/jujube-platform/api-gateway/internal/handler"
)

func main() {
	// ------------------------------------------------------------------
	// 1. Configuration
	// ------------------------------------------------------------------
	cfg := config.Load()

	if cfg.Environment == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	log.Printf("[API Gateway] Starting in %s mode on port %d", cfg.Environment, cfg.HTTPPort)

	// ------------------------------------------------------------------
	// 2. Router
	// ------------------------------------------------------------------
	router := gin.New()

	// Handler setup
	h := handler.New(cfg)
	h.SetupRoutes(router)

	// ------------------------------------------------------------------
	// 3. HTTP Server
	// ------------------------------------------------------------------
	addr := fmt.Sprintf("0.0.0.0:%d", cfg.HTTPPort)
	srv := &http.Server{
		Addr:         addr,
		Handler:      router,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Start server in background
	go func() {
		log.Printf("[API Gateway] Listening on %s", addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("[API Gateway] Failed to start: %v", err)
		}
	}()

	// ------------------------------------------------------------------
	// 4. Graceful shutdown
	// ------------------------------------------------------------------
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	sig := <-quit

	log.Printf("[API Gateway] Received %s — shutting down...", sig)

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Fatalf("[API Gateway] Forced shutdown: %v", err)
	}

	log.Println("[API Gateway] Stopped")
}
