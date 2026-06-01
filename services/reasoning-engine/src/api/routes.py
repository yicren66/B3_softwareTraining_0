"""
Reasoning Engine — REST API Routes.

Implements endpoints defined in SRS-SYS02 §6.1.3 and libs/proto/reasoning.proto:
  POST /api/v1/reasoning/prevention-plan
  POST /api/v1/reasoning/qa
  GET  /health
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from config import settings
from engine.qa_engine import answer_question
from engine.prevention_planner import generate_prevention_plan

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/reasoning", tags=["Reasoning Engine"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class PreventionPlanRequest(BaseModel):
    pest_id: str = Field(..., description="KG entity ID of the pest/disease")
    pest_name: str = Field(default="", description="Pest/disease name")
    county: str = Field(default="", description="County name")
    growth_stage: str = Field(default="", description="Current growth stage")
    temperature: float = Field(default=25.0)
    humidity: float = Field(default=60.0)
    user_role: str = Field(default="枣农")
    confidence: float = Field(default=0.0, description="Recognition confidence")
    severity: str = Field(default="", description="Severity: 轻度/中度/重度")


class QARequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    user_role: str = Field(default="枣农")
    conversation_id: str = Field(default="")
    county: str = Field(default="")
    previous_recognition: Optional[dict] = None


class APIResponse(BaseModel):
    code: int = 0
    message: str = "success"
    data: Optional[dict] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/prevention-plan",
    response_model=APIResponse,
    summary="生成个性化防治方案",
    description="结合识别结果、气候参数和物候期，生成角色差异化防治方案。",
)
async def prevention_plan(req: PreventionPlanRequest):
    t0 = time.perf_counter()
    data = await generate_prevention_plan(
        pest_id=req.pest_id,
        pest_name=req.pest_name,
        county=req.county,
        growth_stage=req.growth_stage,
        temperature=req.temperature,
        humidity=req.humidity,
        user_role=req.user_role,
        confidence=req.confidence,
        severity=req.severity,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("POST /prevention-plan pest=%s role=%s — %.1fms",
                req.pest_name or req.pest_id, req.user_role, elapsed)
    return APIResponse(data=data)


@router.post(
    "/qa",
    response_model=APIResponse,
    summary="智能问答",
    description="多轮对话智能问答，支持角色差异化输出，自动引用知识图谱来源。",
)
async def qa(req: QARequest):
    t0 = time.perf_counter()
    data = await answer_question(
        question=req.question,
        user_role=req.user_role,
        conversation_id=req.conversation_id,
        county=req.county,
        previous_recognition=req.previous_recognition,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("POST /qa role=%s — %.1fms", req.user_role, elapsed)
    return APIResponse(data=data)


@router.get("/health", summary="服务健康检查")
async def health():
    return {"status": "healthy", "service": "reasoning-engine"}
