"""
Knowledge Graph Service — Configuration.

All Neo4j connection, embedding model, and serving parameters.
Values can be overridden via environment variables (prefix KG_).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env(key: str, default: str) -> str:
    return os.environ.get(f"KG_{key}", default)


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(f"KG_{key}")
    return int(val) if val is not None else default


def _env_float(key: str, default: float) -> float:
    val = os.environ.get(f"KG_{key}")
    return float(val) if val is not None else default


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(f"KG_{key}")
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    # ---- Neo4j ----
    NEO4J_URI: str = _env("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = _env("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = _env("NEO4J_PASSWORD", "changeme")
    NEO4J_DATABASE: str = _env("NEO4J_DATABASE", "neo4j")
    NEO4J_MAX_CONNECTION_LIFETIME: int = _env_int("NEO4J_MAX_CONN_LIFETIME", 3600)
    NEO4J_MAX_CONNECTION_POOL_SIZE: int = _env_int("NEO4J_MAX_CONN_POOL", 50)
    NEO4J_CONNECTION_ACQUISITION_TIMEOUT: int = _env_int("NEO4J_CONN_ACQ_TIMEOUT", 60)

    # ---- Semantic Search ----
    EMBEDDING_MODEL_NAME: str = _env(
        "EMBEDDING_MODEL_NAME", "shibing624/text2vec-base-chinese"
    )
    EMBEDDING_DIM: int = _env_int("EMBEDDING_DIM", 768)
    EMBEDDING_CACHE_SIZE: int = _env_int("EMBEDDING_CACHE_SIZE", 10000)
    SEMANTIC_SIMILARITY_THRESHOLD: float = _env_float("SEMANTIC_SIMILARITY_THRESHOLD", 0.6)

    # ---- Search ----
    MAX_SEARCH_RESULTS: int = _env_int("MAX_SEARCH_RESULTS", 20)
    DEFAULT_SEARCH_RESULTS: int = _env_int("DEFAULT_SEARCH_RESULTS", 10)

    # ---- Recommendation ----
    RECOMMENDATION_MAX_ITEMS: int = _env_int("RECOMMENDATION_MAX_ITEMS", 10)

    # ---- Risk Prediction ----
    RISK_HIGH_THRESHOLD: float = _env_float("RISK_HIGH_THRESHOLD", 0.7)
    RISK_ALERT_ENABLED: bool = _env_bool("RISK_ALERT_ENABLED", True)

    # ---- Serving ----
    HTTP_PORT: int = _env_int("HTTP_PORT", 8002)
    GRPC_PORT: int = _env_int("GRPC_PORT", 9002)
    METRICS_ENABLED: bool = _env_bool("METRICS_ENABLED", True)

    # ---- Logging ----
    LOG_LEVEL: str = _env("LOG_LEVEL", "INFO")


# Singleton
settings = Settings()
