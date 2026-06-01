"""
Model Management Service — Configuration.

Values can be overridden via environment variables (prefix MM_).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(key: str, default: str) -> str:
    return os.environ.get(f"MM_{key}", default)


def _env_int(key: str, default: int) -> int:
    return int(_env(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(_env(key, str(default)))


@dataclass(frozen=True)
class Settings:
    # ---- Model storage ----
    MODEL_REGISTRY_ROOT: str = _env("MODEL_REGISTRY_ROOT", "/models")

    # ---- Database ----
    DB_HOST: str = _env("DB_HOST", "postgres")
    DB_PORT: int = _env_int("DB_PORT", 5432)
    DB_NAME: str = _env("DB_NAME", "jujube_platform")
    DB_USER: str = _env("DB_USER", "jujube")
    DB_PASSWORD: str = _env("DB_PASSWORD", "changeme")

    # ---- Deployment ----
    GRAY_RELEASE_DEFAULT_PERCENT: int = _env_int("GRAY_RELEASE_DEFAULT_PERCENT", 10)
    MAX_VERSIONS_RETAINED: int = _env_int("MAX_VERSIONS_RETAINED", 20)

    # ---- Internal services ----
    IMAGE_RECOGNITION_URL: str = _env("IMAGE_RECOGNITION_URL", "http://image-recognition:8001")

    # ---- Serving ----
    HTTP_PORT: int = _env_int("HTTP_PORT", 8004)
    GRPC_PORT: int = _env_int("GRPC_PORT", 9004)

    # ---- Logging ----
    LOG_LEVEL: str = _env("LOG_LEVEL", "INFO")


settings = Settings()
