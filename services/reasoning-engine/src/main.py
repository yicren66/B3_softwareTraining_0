"""
Reasoning Engine — Entry Point.

FastAPI application serving the reasoning & QA REST API on port 8003.
Integrates with the Knowledge Graph service for recommendations and the
Image Recognition service for diagnosis context.
"""

from __future__ import annotations

import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import settings
from api.routes import router as reasoning_router

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("reasoning-engine")


app = FastAPI(
    title="Jujube Reasoning Engine",
    description="枣树病虫害防治推理、智能问答与个性化推荐服务 — SYS-02 §Reasoning",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(reasoning_router)


@app.get("/", include_in_schema=False)
async def root():
    return {"service": "reasoning-engine", "docs": "/docs", "health": "/health"}


def main():
    logger.info("Reasoning Engine starting on port %d...", settings.HTTP_PORT)
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.HTTP_PORT,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
