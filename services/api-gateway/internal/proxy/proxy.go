// Package proxy provides reverse-proxy forwarding to backend microservices.
package proxy

import (
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

// ServiceProxy handles forwarding requests to a backend microservice.
type ServiceProxy struct {
	client  *http.Client
	baseURL string
}

// NewServiceProxy creates a proxy for the given backend base URL.
func NewServiceProxy(baseURL string) *ServiceProxy {
	return &ServiceProxy{
		baseURL: strings.TrimRight(baseURL, "/"),
		client: &http.Client{
			Timeout: 30 * time.Second,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 20,
				IdleConnTimeout:     90 * time.Second,
			},
		},
	}
}

// Forward proxies the current request to the backend service.
func (p *ServiceProxy) Forward(c *gin.Context, targetPath string) {
	// Build the backend URL
	fullPath := p.baseURL + targetPath

	// Preserve query string
	if qs := c.Request.URL.RawQuery; qs != "" {
		fullPath += "?" + qs
	}

	// Create proxied request
	proxyReq, err := http.NewRequestWithContext(c.Request.Context(),
		c.Request.Method, fullPath, c.Request.Body)
	if err != nil {
		log.Printf("[Proxy] Failed to build request: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"code":    500,
			"message": "internal proxy error",
		})
		return
	}

	// Copy headers
	for key, values := range c.Request.Header {
		for _, v := range values {
			proxyReq.Header.Add(key, v)
		}
	}

	// Add X-Forwarded headers
	proxyReq.Header.Set("X-Forwarded-For", c.ClientIP())
	proxyReq.Header.Set("X-Forwarded-Host", c.Request.Host)
	proxyReq.Header.Set("X-Forwarded-Proto", "http")

	// Forward user identity from auth middleware
	if userID, exists := c.Get("user_id"); exists {
		proxyReq.Header.Set("X-User-ID", userID.(string))
	}
	if userRole, exists := c.Get("user_role"); exists {
		proxyReq.Header.Set("X-User-Role", userRole.(string))
	}
	if requestID, exists := c.Get("request_id"); exists {
		proxyReq.Header.Set("X-Request-ID", requestID.(string))
	}

	// Execute
	resp, err := p.client.Do(proxyReq)
	if err != nil {
		log.Printf("[Proxy] Backend unreachable (%s): %v", fullPath, err)
		c.JSON(http.StatusBadGateway, gin.H{
			"code":    502,
			"message": "backend service unavailable",
		})
		return
	}
	defer resp.Body.Close()

	// Copy response headers
	for key, values := range resp.Header {
		for _, v := range values {
			c.Header(key, v)
		}
	}

	// Copy status and body
	c.Status(resp.StatusCode)

	// Stream the body (for large responses, avoid buffering entirely)
	if _, err := io.Copy(c.Writer, resp.Body); err != nil {
		log.Printf("[Proxy] Body copy error: %v", err)
	}
}

// HealthCheck probes the backend's /health endpoint.
func (p *ServiceProxy) HealthCheck(c *gin.Context) bool {
	resp, err := p.client.Get(p.baseURL + "/health")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// BuildPath joins path segments safely.
func BuildPath(segments ...string) string {
	var parts []string
	for _, s := range segments {
		s = strings.Trim(s, "/")
		if s != "" {
			parts = append(parts, s)
		}
	}
	return "/" + strings.Join(parts, "/")
}
