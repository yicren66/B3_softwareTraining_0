"""
Async PostgreSQL connection pool and base repository using asyncpg.
"""

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Type, TypeVar

import asyncpg
from pydantic import BaseModel, Field

from common.errors import AppException, ErrorCode

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class DatabaseConfig(BaseModel):
    """Connection parameters for an asyncpg pool."""

    host: str = "localhost"
    port: int = 5432
    dbname: str = "jujube"
    user: str = "postgres"
    password: str = ""
    min_size: int = Field(default=5, ge=1)
    max_size: int = Field(default=20, ge=1)


# ---------------------------------------------------------------------------
# Pool management
# ---------------------------------------------------------------------------

_pool: Optional[asyncpg.Pool] = None


async def create_pool(config: DatabaseConfig) -> asyncpg.Pool:
    """Create (or replace) the global asyncpg connection pool.

    This should be called once at application startup.
    """
    global _pool
    if _pool is not None:
        await _pool.close()

    _pool = await asyncpg.create_pool(
        host=config.host,
        port=config.port,
        database=config.dbname,
        user=config.user,
        password=config.password,
        min_size=config.min_size,
        max_size=config.max_size,
    )
    return _pool


async def close_pool() -> None:
    """Gracefully close the global pool (call at shutdown)."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


@asynccontextmanager
async def get_connection() -> AsyncIterator[asyncpg.Connection]:
    """Async context manager that yields a connection from the pool."""
    if _pool is None:
        raise AppException(
            ErrorCode.SERVICE_UNAVAILABLE,
            "Database pool has not been initialised. Call create_pool() first.",
        )
    async with _pool.acquire() as conn:
        yield conn


# ---------------------------------------------------------------------------
# Base repository
# ---------------------------------------------------------------------------

T = TypeVar("T")


class BaseRepository:
    """Generic repository providing common CRUD operations.

    Subclass this and set ``table_name`` plus any custom queries.
    """

    table_name: str = ""
    id_column: str = "id"

    def __init__(self, conn: asyncpg.Connection) -> None:
        self.conn = conn

    async def find_by_id(self, id_value: Any) -> Optional[asyncpg.Record]:
        query = f"SELECT * FROM {self.table_name} WHERE {self.id_column} = $1"
        return await self.conn.fetchrow(query, id_value)

    async def find_all(
        self,
        limit: int = 100,
        offset: int = 0,
        order_by: Optional[str] = None,
    ) -> List[asyncpg.Record]:
        query = f"SELECT * FROM {self.table_name}"
        if order_by:
            query += f" ORDER BY {order_by}"
        query += " LIMIT $1 OFFSET $2"
        return await self.conn.fetch(query, limit, offset)

    async def find_where(
        self,
        conditions: Dict[str, Any],
        limit: int = 100,
        offset: int = 0,
    ) -> List[asyncpg.Record]:
        if not conditions:
            return await self.find_all(limit=limit, offset=offset)

        clauses = [f"{col} = ${i + 1}" for i, col in enumerate(conditions.keys())]
        values = list(conditions.values())
        query = f"SELECT * FROM {self.table_name} WHERE {' AND '.join(clauses)}"
        query += f" LIMIT ${len(values) + 1} OFFSET ${len(values) + 2}"
        return await self.conn.fetch(query, *values, limit, offset)

    async def insert(self, data: Dict[str, Any]) -> asyncpg.Record:
        columns = list(data.keys())
        placeholders = [f"${i + 1}" for i in range(len(columns))]
        values = list(data.values())

        query = (
            f"INSERT INTO {self.table_name} ({', '.join(columns)}) "
            f"VALUES ({', '.join(placeholders)}) "
            f"RETURNING *"
        )
        return await self.conn.fetchrow(query, *values)

    async def update(self, id_value: Any, data: Dict[str, Any]) -> asyncpg.Record:
        set_clauses = [f"{col} = ${i + 1}" for i, col in enumerate(data.keys())]
        values = list(data.values())
        values.append(id_value)

        query = (
            f"UPDATE {self.table_name} "
            f"SET {', '.join(set_clauses)} "
            f"WHERE {self.id_column} = ${len(values)} "
            f"RETURNING *"
        )
        return await self.conn.fetchrow(query, *values)

    async def delete(self, id_value: Any) -> bool:
        query = (
            f"DELETE FROM {self.table_name} "
            f"WHERE {self.id_column} = $1 "
            f"RETURNING {self.id_column}"
        )
        result = await self.conn.fetchrow(query, id_value)
        return result is not None

    async def count(self, conditions: Optional[Dict[str, Any]] = None) -> int:
        if conditions:
            clauses = [f"{col} = ${i + 1}" for i, col in enumerate(conditions.keys())]
            values = list(conditions.values())
            query = f"SELECT COUNT(*) FROM {self.table_name} WHERE {' AND '.join(clauses)}"
            row = await self.conn.fetchrow(query, *values)
        else:
            query = f"SELECT COUNT(*) FROM {self.table_name}"
            row = await self.conn.fetchrow(query)
        return row["count"] if row else 0
