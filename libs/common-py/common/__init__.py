"""
# common — Jujube shared Python library

Exports all key classes and helpers used across platform services.
"""

from common.auth import (
    ROLE_FARMER,
    ROLE_GUEST,
    ROLE_PLANT_EXPERT,
    ROLE_SUPER_ADMIN,
    ROLE_WORK_ADMIN,
    SUPPORTED_ROLES,
    TokenPayload,
    create_access_token,
    decode_token,
    hash_password,
    require_role,
    verify_password,
)
from common.errors import (
    AppException,
    ErrorCode,
    build_error_response,
)
from common.logging import (
    bind_request_id,
    clear_request_context,
    get_logger,
)
from common.metrics import (
    active_users_gauge,
    kg_query_counter,
    kg_query_latency_histogram,
    recognition_latency_histogram,
    recognition_request_counter,
)
from common.db import (
    BaseRepository,
    DatabaseConfig,
    RedisClient,
    RedisConfig,
    cache,
    create_pool,
    get_connection,
)

__all__ = [
    # auth
    "ROLE_FARMER",
    "ROLE_GUEST",
    "ROLE_PLANT_EXPERT",
    "ROLE_SUPER_ADMIN",
    "ROLE_WORK_ADMIN",
    "SUPPORTED_ROLES",
    "TokenPayload",
    "create_access_token",
    "decode_token",
    "hash_password",
    "require_role",
    "verify_password",
    # errors
    "AppException",
    "ErrorCode",
    "build_error_response",
    # logging
    "bind_request_id",
    "clear_request_context",
    "get_logger",
    # metrics
    "active_users_gauge",
    "kg_query_counter",
    "kg_query_latency_histogram",
    "recognition_latency_histogram",
    "recognition_request_counter",
    # db
    "BaseRepository",
    "DatabaseConfig",
    "RedisClient",
    "RedisConfig",
    "cache",
    "create_pool",
    "get_connection",
]
