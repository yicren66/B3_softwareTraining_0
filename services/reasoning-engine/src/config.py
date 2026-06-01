"""
Reasoning Engine — Configuration.

All service endpoints, model paths, and runtime settings.
Values can be overridden via environment variables (prefix RE_).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


def _env(key: str, default: str) -> str:
    return os.environ.get(f"RE_{key}", default)


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(f"RE_{key}")
    return int(val) if val is not None else default


def _env_float(key: str, default: float) -> float:
    val = os.environ.get(f"RE_{key}")
    return float(val) if val is not None else default


class UserRole(str, Enum):
    FARMER = "枣农"
    EXPERT = "植保专家"
    ADMIN = "管理员"
    VISITOR = "访客"


@dataclass(frozen=True)
class Settings:
    # ---- Internal services ----
    KG_SERVICE_URL: str = _env("KG_SERVICE_URL", "http://knowledge-graph:8002")
    KG_SERVICE_TIMEOUT: int = _env_int("KG_SERVICE_TIMEOUT", 10)  # seconds
    RECOGNITION_SERVICE_URL: str = _env("RECOGNITION_SERVICE_URL", "http://image-recognition:8001")
    WEATHER_SERVICE_URL: str = _env("WEATHER_SERVICE_URL", "")

    # ---- QA ----
    MAX_QA_HISTORY: int = _env_int("MAX_QA_HISTORY", 10)
    QA_CONFIDENCE_THRESHOLD: float = _env_float("QA_CONFIDENCE_THRESHOLD", 0.5)

    # ---- Prevention plan ----
    PLAN_MAX_ITEMS_PER_CATEGORY: int = _env_int("PLAN_MAX_ITEMS_PER_CATEGORY", 5)

    # ---- Serving ----
    HTTP_PORT: int = _env_int("HTTP_PORT", 8003)
    GRPC_PORT: int = _env_int("GRPC_PORT", 9003)

    # ---- Logging ----
    LOG_LEVEL: str = _env("LOG_LEVEL", "INFO")


settings = Settings()
