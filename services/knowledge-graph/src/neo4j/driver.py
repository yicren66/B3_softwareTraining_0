"""
Neo4j driver lifecycle manager.

Provides a singleton driver with connection pooling, health-check,
and graceful shutdown. All Cypher queries should go through this module.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from config import settings

logger = logging.getLogger(__name__)

_driver: Optional[AsyncDriver] = None


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def get_driver() -> AsyncDriver:
    """Return the singleton driver, creating it if needed."""
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            max_connection_lifetime=settings.NEO4J_MAX_CONNECTION_LIFETIME,
            max_connection_pool_size=settings.NEO4J_MAX_CONNECTION_POOL_SIZE,
            connection_acquisition_timeout=settings.NEO4J_CONNECTION_ACQUISITION_TIMEOUT,
        )
        logger.info(
            "Neo4j driver created — uri=%s db=%s pool=%d",
            settings.NEO4J_URI,
            settings.NEO4J_DATABASE,
            settings.NEO4J_MAX_CONNECTION_POOL_SIZE,
        )
    return _driver


async def close_driver() -> None:
    """Gracefully close the driver."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None
        logger.info("Neo4j driver closed.")


async def health_check() -> bool:
    """Return True if Neo4j is reachable."""
    try:
        driver = await get_driver()
        async with driver.session(database=settings.NEO4J_DATABASE) as session:
            await session.run("RETURN 1")
        return True
    except (Neo4jError, ServiceUnavailable, OSError) as e:
        logger.warning("Neo4j health check failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Session context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def get_session() -> AsyncGenerator:
    """Async context manager yielding a Neo4j session."""
    driver = await get_driver()
    async with driver.session(database=settings.NEO4J_DATABASE) as session:
        yield session


# ---------------------------------------------------------------------------
# Common queries
# ---------------------------------------------------------------------------


async def get_entity_by_id(entity_id: str) -> Optional[dict]:
    """Fetch a single entity with its outgoing relations."""
    query = """
    MATCH (e {entity_id: $entity_id})
    OPTIONAL MATCH (e)-[r]->(target)
    RETURN e,
           collect(DISTINCT {
             relation_type: type(r),
             target_entity_id: target.entity_id,
             target_name: coalesce(target.name_cn, target.stage_name, target.name_cn, ''),
             target_type: labels(target)[0],
             properties: properties(r)
           }) as relations
    """
    async with get_session() as session:
        result = await session.run(query, entity_id=entity_id)
        record = await result.single()
        if record is None:
            return None
        entity_node = record["e"]
        return {
            "entity_id": entity_node.get("entity_id"),
            "type": list(entity_node.labels)[0] if entity_node.labels else "Unknown",
            "name_cn": entity_node.get("name_cn", ""),
            "category": entity_node.get("category", ""),
            "description": entity_node.get("description", ""),
            "scientific_name": entity_node.get("scientific_name", ""),
            "typical_symptoms": entity_node.get("typical_symptoms", ""),
            "severity_levels": entity_node.get("severity_levels", ""),
            "relations": record.get("relations", []),
        }


async def fulltext_search(
    query_text: str,
    limit: int = 10,
) -> list[dict]:
    """Full-text search across PestDisease, Symptom, and PreventionMethod indexes."""
    cypher = """
    CALL db.index.fulltext.queryNodes('pest_disease_fulltext', $query)
    YIELD node, score
    RETURN node.entity_id AS entity_id,
           node.name_cn AS name,
           node.category AS category,
           node.description AS description,
           score
    ORDER BY score DESC
    LIMIT $limit
    """
    async with get_session() as session:
        result = await session.run(cypher, query=query_text, limit=limit)
        records = await result.data()
        return records
