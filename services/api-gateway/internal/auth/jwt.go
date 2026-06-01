// Package auth provides JWT token creation, validation, and middleware.
package auth

import (
	"errors"
	"fmt"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

// Claims represents the JWT claims payload.
type Claims struct {
	UserID   string `json:"user_id"`
	Username string `json:"username"`
	Role     string `json:"role"` // 超级管理员 | 工作管理员 | 植保专家 | 枣农 | 访客
	jwt.RegisteredClaims
}

var (
	ErrTokenExpired   = errors.New("token has expired")
	ErrTokenInvalid   = errors.New("token is invalid")
	ErrTokenMissing   = errors.New("authorization header is missing")
	ErrInvalidHeader  = errors.New("invalid authorization header format")
)

// JWTManager handles token creation and validation.
type JWTManager struct {
	secretKey     string
	expireHours   int
	refreshHours  int
}

// NewJWTManager creates a new JWTManager.
func NewJWTManager(secretKey string, expireHours, refreshHours int) *JWTManager {
	return &JWTManager{
		secretKey:    secretKey,
		expireHours:  expireHours,
		refreshHours: refreshHours,
	}
}

// GenerateToken creates a signed JWT access token.
func (m *JWTManager) GenerateToken(userID, username, role string) (string, error) {
	claims := Claims{
		UserID:   userID,
		Username: username,
		Role:     role,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(time.Duration(m.expireHours) * time.Hour)),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
			Issuer:    "jujube-api-gateway",
		},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString([]byte(m.secretKey))
}

// GenerateRefreshToken creates a long-lived refresh token.
func (m *JWTManager) GenerateRefreshToken(userID string) (string, error) {
	claims := jwt.RegisteredClaims{
		Subject:   userID,
		ExpiresAt: jwt.NewNumericDate(time.Now().Add(time.Duration(m.refreshHours) * time.Hour)),
		IssuedAt:  jwt.NewNumericDate(time.Now()),
		Issuer:    "jujube-api-gateway",
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString([]byte(m.secretKey))
}

// ValidateToken parses and validates a JWT token, returning its claims.
func (m *JWTManager) ValidateToken(tokenString string) (*Claims, error) {
	token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return []byte(m.secretKey), nil
	})

	if err != nil {
		if errors.Is(err, jwt.ErrTokenExpired) {
			return nil, ErrTokenExpired
		}
		return nil, ErrTokenInvalid
	}

	claims, ok := token.Claims.(*Claims)
	if !ok || !token.Valid {
		return nil, ErrTokenInvalid
	}

	return claims, nil
}

// ExtractBearerToken extracts the Bearer token from the Authorization header.
func ExtractBearerToken(authHeader string) (string, error) {
	if authHeader == "" {
		return "", ErrTokenMissing
	}
	if len(authHeader) < 7 || authHeader[:7] != "Bearer " {
		return "", ErrInvalidHeader
	}
	return authHeader[7:], nil
}

// RoleHierarchy defines privilege levels for role-based access control.
var RoleHierarchy = map[string]int{
	"超级管理员":   4,
	"工作管理员":   3,
	"植保专家":    2,
	"枣农":      1,
	"访客":      0,
}

// HasMinRole checks if the user's role meets or exceeds the required level.
func HasMinRole(userRole string, minRole string) bool {
	userLevel, ok := RoleHierarchy[userRole]
	if !ok {
		return false
	}
	minLevel, ok := RoleHierarchy[minRole]
	if !ok {
		return false
	}
	return userLevel >= minLevel
}
