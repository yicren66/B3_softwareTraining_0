"""
Knowledge Graph Service — REST API Routes.

Implements the endpoints defined in SRS-SYS02 §6.1.2 and libs/proto/kg.proto:
  GET  /api/v1/kg/entity/{entity_id}
  POST /api/v1/kg/search
  GET  /api/v1/kg/recommendation/{pest_id}
  POST /api/v1/kg/qa
  GET  /api/v1/kg/risk/predict
  GET  /health
  GET  /metrics
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.schemas import (
    EntityResponse,
    EntityRelation,
    SearchRequest,
    SearchResponse,
    SearchResult,
    ParsedIntent,
    PreventionSummary,
    IntentRecommendation,
    RecommendationRequest,
    RecommendationResponse,
    RiskRequest,
    RiskResponse,
    HealthStatus,
    APIResponse,
)
from neo4j.driver import get_entity_by_id, fulltext_search, health_check
from search.semantic_search import (
    classify_intent,
    extract_pest_entities,
    generate_intent_recommendations,
)
from recommendation.engine import get_prevention_recommendations
from risk.predictor import predict_risks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/kg", tags=["Knowledge Graph"])


# ---------------------------------------------------------------------------
# GET /entity/{entity_id}
# ---------------------------------------------------------------------------


@router.get(
    "/entity/{entity_id}",
    response_model=APIResponse,
    summary="查询实体详情",
    description="根据实体ID获取知识图谱中实体的完整信息，包括症状、防治方法、关联生长阶段等。",
)
async def get_entity(entity_id: str):
    t0 = time.perf_counter()
    entity = await get_entity_by_id(entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")

    relations = [
        EntityRelation(**rel)
        for rel in entity.pop("relations", [])
    ]
    data = EntityResponse(**entity, relations=relations)
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("GET /entity/%s — %.1fms", entity_id, elapsed)

    return APIResponse(data=data.model_dump())


# ---------------------------------------------------------------------------
# POST /search
# ---------------------------------------------------------------------------


@router.post(
    "/search",
    response_model=APIResponse,
    summary="自然语言语义检索",
    description="支持口语化自然语言问题检索，基于词向量语义识别算法将用户意图映射到知识图谱实体和关系。",
)
async def search_knowledge(req: SearchRequest):
    t0 = time.perf_counter()

    # 1. Parse intent
    intent, intent_conf, _ = classify_intent(req.query)

    # 2. Extract pest entities from query
    mapped_pests = extract_pest_entities(req.query)
    entities = list(set(p["name"] for p in mapped_pests))

    parsed_intent = ParsedIntent(
        intent=intent,
        confidence=intent_conf,
        entities=entities,
        mapped_pests=mapped_pests,
    )

    # 3. Full-text search in Neo4j
    fts_results = await fulltext_search(req.query, limit=req.max_results)

    results: list[SearchResult] = []
    for row in fts_results:
        results.append(SearchResult(
            entity_id=row.get("entity_id", ""),
            name=row.get("name", ""),
            relevance_score=round(float(row.get("score", 0)), 4),
            summary=row.get("description", "")[:200] if row.get("description") else "",
        ))

    # 4. Intent-based recommendations
    intent_recs: list[IntentRecommendation] = []
    if req.include_intent_recommendation:
        recs = generate_intent_recommendations(req.query, intent, mapped_pests)
        for r in recs:
            intent_recs.append(IntentRecommendation(**r))

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("POST /search query='%s' intent=%s — %d results, %.1fms",
                req.query[:50], intent, len(results), elapsed)

    return APIResponse(data=SearchResponse(
        query=req.query,
        parsed_intent=parsed_intent,
        results=results,
        intent_recommendations=intent_recs,
    ).model_dump())


# ---------------------------------------------------------------------------
# GET /recommendation/{pest_id}
# ---------------------------------------------------------------------------


@router.get(
    "/recommendation/{pest_id}",
    response_model=APIResponse,
    summary="获取综合防治推荐方案",
    description="获取指定病虫害的综合防治推荐方案，包含生防/物防/人防分类。",
)
async def get_recommendation(
    pest_id: str,
    county: str = Query(default="", description="县市名称"),
    growth_stage: str = Query(default="", description="当前物候期"),
    temperature: float = Query(default=25.0, description="当前温度(°C)"),
    humidity: float = Query(default=60.0, description="当前湿度(%)"),
):
    t0 = time.perf_counter()
    data = await get_prevention_recommendations(
        pest_id=pest_id,
        county=county,
        growth_stage=growth_stage,
        temperature=temperature,
        humidity=humidity,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("GET /recommendation/%s — %.1fms", pest_id, elapsed)

    return APIResponse(data=data)


# ---------------------------------------------------------------------------
# POST /qa
# ---------------------------------------------------------------------------


@router.post(
    "/qa",
    response_model=APIResponse,
    summary="智能问答接口",
    description="提供面向枣农、植保专家等不同角色的智能问答，支持多轮对话。",
)
async def question_answer(
    question: str = Query(..., description="自然语言问题"),
    user_role: str = Query(default="枣农", description="用户角色"),
    conversation_id: str = Query(default="", description="会话ID（多轮对话）"),
    county: str = Query(default="", description="所在县市"),
):
    t0 = time.perf_counter()

    # 1. Parse intent and extract entities
    intent, intent_conf, _ = classify_intent(question)
    mapped_pests = extract_pest_entities(question)

    # 2. Search knowledge graph for relevant entities
    search_results = await fulltext_search(question, limit=5)

    # 3. Build answer based on intent and role
    answer = _build_qa_answer(question, intent, mapped_pests, search_results, user_role, county)

    # 4. Sources from KG
    sources = []
    for sr in search_results[:3]:
        sources.append({
            "entity_id": sr.get("entity_id", ""),
            "name": sr.get("name", ""),
            "relation": "语义匹配",
        })

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("POST /qa intent=%s role=%s — %.1fms", intent, user_role, elapsed)

    return APIResponse(data={
        "conversation_id": conversation_id or f"conv_{int(time.time())}",
        "question": question,
        "answer": answer,
        "sources": sources,
        "confidence": intent_conf,
        "follow_up_suggestions": _follow_up_suggestions(intent, mapped_pests),
    })


# ---------------------------------------------------------------------------
# GET /risk/predict
# ---------------------------------------------------------------------------


@router.get(
    "/risk/predict",
    response_model=APIResponse,
    summary="病虫害爆发风险预测",
    description="结合物候期和气候数据，预测当前阶段可能爆发的病虫害风险。",
)
async def risk_predict(
    county: str = Query(..., description="县市名称"),
    growth_stage: str = Query(default="", description="当前物候期"),
    temperature: float = Query(default=25.0, description="当前温度(°C)"),
    humidity: float = Query(default=60.0, description="当前湿度(%)"),
    precipitation_7d: float = Query(default=0.0, description="近7日降水(mm)"),
):
    t0 = time.perf_counter()
    data = await predict_risks(
        county=county,
        growth_stage=growth_stage,
        temperature=temperature,
        humidity=humidity,
        precipitation_7d=precipitation_7d,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("GET /risk/predict county=%s stage=%s — %.1fms, alerts=%d",
                county, growth_stage, elapsed, len(data.get("alert_pests", [])))

    return APIResponse(data=data)


# ---------------------------------------------------------------------------
# Health & metrics
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthStatus, summary="服务健康检查")
async def health():
    neo4j_ok = await health_check()
    return HealthStatus(
        is_ready=neo4j_ok,
        neo4j_connected=neo4j_ok,
    )


# ---------------------------------------------------------------------------
# QA answer builder (role-differentiated)
# ---------------------------------------------------------------------------


def _build_qa_answer(
    question: str,
    intent: str,
    mapped_pests: list[dict],
    search_results: list[dict],
    user_role: str,
    county: str,
) -> str:
    """Construct a role-appropriate answer from KG search results."""

    if not search_results and not mapped_pests:
        if user_role == "枣农":
            return "您的问题我暂时无法精确匹配。建议您拍摄病虫害照片上传识别，或联系当地植保技术人员进行田间诊断。"
        elif user_role == "植保专家":
            return f"知识图谱中未检索到与「{question}」高度匹配的实体。可考虑扩展检索范围或人工补充相关知识点。"
        return f"未检索到与「{question}」匹配的知识条目，请检查查询表述或联系系统管理员更新知识库。"

    # If pests matched directly
    if mapped_pests:
        pest_names = "、".join(p["name"] for p in mapped_pests[:3])

        if intent == "find_treatment":
            if user_role == "枣农":
                return (
                    f"针对{pest_names}，建议采取以下措施：\n"
                    f"1. 及时清除病残体，加强果园通风透光；\n"
                    f"2. 悬挂诱虫板、安装杀虫灯等物理防治；\n"
                    f"3. 必要时选用推荐药剂，严格按说明书使用，注意安全间隔期。\n"
                    f"详细防治方案请在平台查看完整推荐。"
                )
            elif user_role == "植保专家":
                return (
                    f"关于{pest_names}的综合防治方案：\n"
                    f"建议以农业防治为基础，优先采用生物防治手段，配合物理防治，"
                    f"在达到防治指标时科学选用化学药剂。具体药剂选择、施药时期和剂量"
                    f"可参考知识图谱中的详细推荐。"
                )
            return f"针对{pest_names}，请在平台查看完整的综合防治推荐方案。"

        elif intent == "identify_pest":
            if search_results:
                top = search_results[0]
                return (
                    f"根据您的描述，「{question}」可能与{top.get('name', '未知病虫害')}相关"
                    f"（匹配度{top.get('score', 0):.0%}）。\n"
                    f"建议对照详细症状描述进行确认，或上传病虫害照片进行AI识别。"
                )

    # Fallback: use search results
    if search_results:
        names = "、".join(r.get("name", "") for r in search_results[:3])
        return f"已为您检索到以下相关知识条目：{names}。请点击查看详细信息或提出更具体的问题。"

    return "请提供更多病虫害症状描述，以便进行更准确的诊断。"


def _follow_up_suggestions(intent: str, mapped_pests: list[dict]) -> list[str]:
    """Generate contextual follow-up question suggestions."""
    suggestions: list[str] = []

    if intent == "identify_pest":
        suggestions = [
            "该病虫害的典型症状是什么？",
            "如何与其他相似病虫害区分？",
            "目前处于哪个生长阶段？",
        ]
    elif intent == "find_treatment":
        suggestions = [
            "有什么生物防治方法？",
            "推荐使用什么药剂？",
            "如何预防再次发生？",
        ]
    elif intent == "risk_check":
        suggestions = [
            "当前季节需要注意哪些病虫害？",
            "如何进行预防性防治？",
            "查看完整的风险预测报告",
        ]

    if mapped_pests:
        pest_name = mapped_pests[0]["name"]
        suggestions.insert(0, f"查看{pest_name}的详细防治方案")

    return suggestions[:4]
