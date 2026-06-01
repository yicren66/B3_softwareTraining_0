"""
Authentication and authorization utilities.

- JWT token generation & validation (using pyjwt).
- Password hashing (bcrypt via passlib – a lightweight bcrypt wrapper is
  provided as a fallback if passlib is not installed).
- Role-based permission-decorator for use with synchronous code.
"""

import functools
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, List, Optional

import jwt
from pydantic import BaseModel, Field

from common.errors import AppException, ErrorCode

# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------

ROLE_SUPER_ADMIN = "super_admin"
ROLE_WORK_ADMIN = "work_admin"
ROLE_PLANT_EXPERT = "plant_expert"
ROLE_FARMER = "farmer"
ROLE_GUEST = "guest"

SUPPORTED_ROLES = frozenset({
    ROLE_SUPER_ADMIN,
    ROLE_WORK_ADMIN,
    ROLE_PLANT_EXPERT,
    ROLE_FARMER,
    ROLE_GUEST,
})

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TokenPayload(BaseModel):
    """Claims carried inside every JWT issued by the platform."""

    sub: str = Field(..., description="User ID (subject)")
    username: str
    role: str
    county: str = ""
    exp: Optional[datetime] = None  # expiration as a UTC datetime


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------

# Simple, dependency-light bcrypt wrapper.
# If passlib is available it uses passlib; otherwise it shells out to a
# pure-bcrypt implementation (bcrypt package).  We keep the check minimal
# so services can pick what they prefer.

try:
    import bcrypt as _bcrypt_lib

    def hash_password(password: str) -> str:
        salt = _bcrypt_lib.gensalt()
        return _bcrypt_lib.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    def verify_password(plain: str, hashed: str) -> bool:
        return _bcrypt_lib.checkpw(
            plain.encode("utf-8"), hashed.encode("utf-8")
        )

except ImportError:
    # Fallback: SHA-256 based password hashing.
    # NOT suitable for production; install `bcrypt` or `passlib[bcrypt]`.
    def _sha256_hash(password: str, salt: Optional[str] = None) -> str:
        salt = salt or secrets.token_hex(16)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
        return f"sha256${salt}${dk.hex()}"

    def hash_password(password: str) -> str:
        return _sha256_hash(password)

    def verify_password(plain: str, hashed: str) -> bool:
        if hashed.startswith("sha256$"):
            _, salt, _ = hashed.split("$", 2)
            return _sha256_hash(plain, salt) == hashed
        return False


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


def create_access_token(
    payload: TokenPayload,
    secret: str,
    expires_delta: timedelta = timedelta(hours=8),
) -> str:
    """Create a signed JWT access token.

    Args:
        payload: TokenPayload with user claims.
        secret: HMAC secret key.
        expires_delta: Token lifetime (default 8 hours).

    Returns:
        Encoded JWT string.
    """
    now = datetime.now(timezone.utc)
    exp = payload.exp or (now + expires_delta)

    claims = payload.model_dump()
    claims["iat"] = now
    claims["exp"] = exp
    claims["sub"] = payload.sub  # ensure sub is preserved

    return jwt.encode(claims, secret, algorithm="HS256")


def decode_token(token: str, secret: str) -> TokenPayload:
    """Decode and validate a JWT, returning its payload.

    Args:
        token: Raw JWT string.
        secret: HMAC secret key.

    Returns:
        TokenPayload with the decoded claims.

    Raises:
        AppException (AUTHENTICATION_FAILED) on any decode / expiry error.
    """
    try:
        claims = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise AppException(
            ErrorCode.AUTHENTICATION_FAILED,
            "Token has expired.",
        )
    except jwt.InvalidTokenError as exc:
        raise AppException(
            ErrorCode.AUTHENTICATION_FAILED,
            f"Invalid token: {exc}",
        )

    return TokenPayload(
        sub=claims["sub"],
        username=claims.get("username", ""),
        role=claims.get("role", ROLE_GUEST),
        county=claims.get("county", ""),
        exp=claims.get("exp"),
    )


# ---------------------------------------------------------------------------
# Permission decorator
# ---------------------------------------------------------------------------


def require_role(*roles: str) -> Callable:
    """Decorator that checks the caller's role before executing the function.

    Usage::

        @require_role("super_admin", "work_admin")
        def restricted_view(request, *args, **kwargs):
            ...

    The decorated function must receive a keyword argument ``token_payload``
    (a :class:`TokenPayload` instance).  If the role does not match, an
    ``AppException`` with ``PERMISSION_DENIED`` is raised.
    """
    allowed = frozenset(roles)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            token_payload: Optional[TokenPayload] = kwargs.get("token_payload")
            if token_payload is None:
                raise AppException(
                    ErrorCode.PERMISSION_DENIED,
                    "Missing token_payload in decorated function call.",
                )
            if token_payload.role not in allowed:
                raise AppException(
                    ErrorCode.PERMISSION_DENIED,
                    f"Role '{token_payload.role}' is not permitted. "
                    f"Required: {', '.join(sorted(allowed))}.",
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator
