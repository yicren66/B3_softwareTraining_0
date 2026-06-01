"""
Shared error codes and exception classes for the Jujube platform.

Error code ranges:
    0        — Success
    40001-40099 — Invalid request / client errors
    40101-40199 — Authentication errors
    40301-40399 — Authorization / permission errors
    40401-40499 — Entity not found errors
    42901-42999 — Rate limiting errors
    50001-50099 — Internal server / model errors
    50301-50399 — Service availability errors
"""

import enum
from typing import Any, Dict, Optional


class ErrorCode(enum.IntEnum):
    """Standardised error codes used across all Jujube services."""

    SUCCESS = 0

    # --- 4xx: Client errors ---
    INVALID_IMAGE_FORMAT = 40001
    IMAGE_TOO_LARGE = 40002
    INVALID_PARAMETER = 40003

    # Authentication
    AUTHENTICATION_FAILED = 40101

    # Authorization
    PERMISSION_DENIED = 40301

    # Not found
    ENTITY_NOT_FOUND = 40401
    TASK_NOT_FOUND = 40402

    # Rate limiting
    RATE_LIMIT_EXCEEDED = 42901

    # --- 5xx: Server errors ---
    MODEL_INFERENCE_ERROR = 50001
    KG_QUERY_ERROR = 50002
    TRAINING_JOB_FAILED = 50003

    SERVICE_UNAVAILABLE = 50301

    def http_status(self) -> int:
        """Map the error code to the closest HTTP status code."""
        if self == ErrorCode.SUCCESS:
            return 200
        if 40001 <= self <= 40099:
            return 400
        if 40101 <= self <= 40199:
            return 401
        if 40301 <= self <= 40399:
            return 403
        if 40401 <= self <= 40499:
            return 404
        if 42901 <= self <= 42999:
            return 429
        if 50301 <= self <= 50399:
            return 503
        return 500


class AppException(Exception):
    """Application-level exception carrying a structured error code."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.code = code
        self.message = message
        self.detail = detail or {}
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "detail": self.detail,
        }

    @property
    def http_status(self) -> int:
        return self.code.http_status()


def build_error_response(
    code: ErrorCode,
    message: str,
    detail: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a standardised error response dictionary.

    This helper is used by HTTP layers (FastAPI, sanic, etc.) to return
    consistent error bodies without importing the AppException class.
    """
    return {
        "code": code.value,
        "message": message,
        "detail": detail or {},
    }
