"""
Database package – PostgreSQL and Redis abstractions.
"""

from common.db.postgres import (
    BaseRepository,
    DatabaseConfig,
    create_pool,
    get_connection,
)
from common.db.redis import (
    RedisConfig,
    RedisClient,
    cache,
)

__all__ = [
    "BaseRepository",
    "DatabaseConfig",
    "create_pool",
    "get_connection",
    "RedisConfig",
    "RedisClient",
    "cache",
]
