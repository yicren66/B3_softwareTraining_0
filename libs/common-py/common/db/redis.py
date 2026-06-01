"""
Redis client wrapper (redis-py async) with JSON serialisation and a TTL cache
decorator.
"""

import functools
import json
import hashlib
from typing import Any, Callable, Dict, Optional, Union

import redis.asyncio as aioredis
from pydantic import BaseModel, Field

from common.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class RedisConfig(BaseModel):
    """Connection parameters for a Redis server."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class RedisClient:
    """Async Redis client with JSON helpers.

    Usage::

        config = RedisConfig(host="localhost", port=6379)
        client = await RedisClient.create(config)
        await client.set("key", {"foo": "bar"})
        value = await client.get("key")  # {"foo": "bar"}
    """

    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis
        self._serializer = json
        self._default_ttl: Optional[int] = None

    @classmethod
    async def create(
        cls,
        config: RedisConfig,
        default_ttl: Optional[int] = None,
    ) -> "RedisClient":
        """Factory that creates and connects the underlying Redis client.

        Args:
            config: Connection parameters.
            default_ttl: Default TTL in seconds for ``set`` calls that do not
                specify an explicit TTL.  ``None`` means no expiry.
        """
        redis = await aioredis.from_url(
            f"redis://{config.host}:{config.port}/{config.db}",
            password=config.password or None,
            decode_responses=True,
        )
        client = cls(redis)
        client._default_ttl = default_ttl
        return client

    async def close(self) -> None:
        await self._redis.close()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def get(self, key: str) -> Optional[Any]:
        """Return the deserialised value at *key*, or ``None``."""
        raw = await self._redis.get(key)
        if raw is None:
            return None
        try:
            return self._serializer.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("redis.get: failed to deserialise key", key=key)
            return raw  # return as-is; might be a plain string

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set *key* to the JSON-serialised *value*.

        Args:
            key: Redis key.
            value: Arbitrary JSON-serialisable object.
            ttl: TTL in seconds. Falls back to the client's ``default_ttl``.
        """
        raw = self._serializer.dumps(value, default=str)
        ex = ttl if ttl is not None else self._default_ttl
        if ex is not None:
            return await self._redis.set(key, raw, ex=ex)
        return await self._redis.set(key, raw)

    async def delete(self, *keys: str) -> int:
        """Delete one or more keys. Returns the number of keys deleted."""
        if not keys:
            return 0
        return await self._redis.delete(*keys)

    async def exists(self, *keys: str) -> int:
        """Return the number of *keys* that exist."""
        return await self._redis.exists(*keys)

    async def expire(self, key: str, ttl: int) -> bool:
        """Set a TTL on an existing key."""
        return await self._redis.expire(key, ttl)

    async def ttl(self, key: str) -> int:
        """Return the remaining TTL of *key* in seconds, or -1 / -2."""
        return await self._redis.ttl(key)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[int] = None,
    ) -> Any:
        """Return cached value or compute + cache it via *factory*."""
        cached = await self.get(key)
        if cached is not None:
            return cached
        value = factory()
        await self.set(key, value, ttl=ttl)
        return value


# ---------------------------------------------------------------------------
# Cache decorator
# ---------------------------------------------------------------------------


def _make_cache_key(prefix: str, args: tuple, kwargs: dict) -> str:
    """Deterministically hash function arguments into a cache key."""
    raw = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{prefix}:{digest}"


def cache(
    ttl: int = 300,
    prefix: Optional[str] = None,
    client_attr: str = "_redis_client",
    ignore: Optional[list] = None,
) -> Callable:
    """Decorator that caches the return value of an async function in Redis.

    The decorated instance must have a ``RedisClient`` stored as an attribute
    (default ``_redis_client``).  The cache key is built from the function
    name and the hashed arguments.

    Args:
        ttl: Cache time-to-live in seconds (default 300).
        prefix: Custom cache-key prefix. Defaults to ``func.__qualname__``.
        client_attr: Name of the attribute holding the ``RedisClient`` instance.
        ignore: List of positional argument indices to skip when building the
            cache key.  Useful for ``self`` or ``request``.
    """
    ignore = ignore or []

    def decorator(func: Callable) -> Callable:
        _prefix = prefix or func.__qualname__

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            client: Optional[RedisClient] = getattr(
                args[0], client_attr, None
            ) if args else None

            if client is None:
                logger.debug(
                    "cache decorator: no RedisClient found, bypassing cache",
                    func=_prefix,
                    client_attr=client_attr,
                )
                return await func(*args, **kwargs)

            # Filter arguments
            filtered_args = tuple(
                a for i, a in enumerate(args) if i not in ignore
            )
            key = _make_cache_key(_prefix, filtered_args, kwargs)

            cached = await client.get(key)
            if cached is not None:
                return cached

            result = await func(*args, **kwargs)
            await client.set(key, result, ttl=ttl)
            return result

        return wrapper

    return decorator
