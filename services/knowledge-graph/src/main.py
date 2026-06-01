"""
Knowledge Graph Service — Entry Point.

FastAPI application serving the KG REST API on port 8002.
Connects to Neo4j for graph queries and sentence-transformers for semantic search.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import settings
from neo4j.driver import get_driver, close_driver, health_check
from api.routes import router as kg_router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("kg-service")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    # Startup
    logger.info("Knowledge Graph Service starting...")
    try:
        driver = await get_driver()
        ok = await health_check()
        if ok:
            logger.info("Neo4j connection verified.")
        else:
            logger.warning("Neo4j is NOT reachable — service will start but queries may fail.")
    except Exception as e:
        logger.error("Failed to initialise Neo4j driver: %s", e)

    yield

    # Shutdown
    logger.info("Knowledge Graph Service shutting down...")
    await close_driver()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Jujube Knowledge Graph Service",
    description="枣树病虫害知识图谱查询、检索、推荐与风险预测服务 — SYS-02 §KG",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins in dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(kg_router)

# Root redirect
@app.get("/", include_in_schema=False)
async def root():
    return {"service": "knowledge-graph", "docs": "/docs", "health": "/health"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.HTTP_PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
