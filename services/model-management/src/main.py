"""
Model Management Service — Entry Point.

FastAPI application for model registry, version management, deployment,
and A/B testing. Runs on port 8004.
"""

from __future__ import annotations

import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import settings
from api.routes import router as mm_router

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("model-management")

app = FastAPI(
    title="Jujube Model Management Service",
    description="AI模型版本管理、部署、灰度发布与A/B测试服务 — SYS-02 §Model Management",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(mm_router)


@app.get("/", include_in_schema=False)
async def root():
    return {"service": "model-management", "docs": "/docs", "health": "/health"}


def main():
    logger.info("Model Management Service starting on port %d...", settings.HTTP_PORT)
    uvicorn.run("main:app", host="0.0.0.0", port=settings.HTTP_PORT,
                log_level=settings.LOG_LEVEL.lower())


if __name__ == "__main__":
    main()
