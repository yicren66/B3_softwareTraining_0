"""
Multi-turn Q&A engine with KG-backed answers and role differentiation.

SRS-SYS02 §4.2.3 KR-04 — 智能问答交互
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Optional

import httpx

from config import settings, UserRole

logger = logging.getLogger(__name__)

# In-memory conversation store (replace with Redis in production)
_conversations: dict[str, list[dict]] = defaultdict(list)

# Similar pest differentiation knowledge (SRS KR-02)
SIMILAR_PEST_PAIRS = {
    ("枣炭疽病", "枣褐斑病"): "枣炭疽病病斑有轮纹状黑色小点，果实受害为主；枣褐斑病病斑边缘深褐中央灰白，叶片受害更明显。",
    ("枣缩果病", "枣果腐病"): "枣缩果病果实干缩不腐烂，果皮红褐色；枣果腐病果实软腐有霉层和酸臭味。",
    ("枣红蜘蛛", "枣瘿螨"): "枣红蜘蛛在叶背结网，叶片呈灰白色；枣瘿螨使叶片形成泡状虫瘿，无蛛网。",
    ("枣尺蠖", "枣镰翅小卷蛾"): "枣尺蠖幼虫暴食叶片造成缺刻甚至吃光全叶；枣镰翅小卷蛾幼虫吐丝卷叶在内取食。",
}


# ---------------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------------


async def answer_question(
    question: str,
    user_role: str = "枣农",
    conversation_id: str = "",
    county: str = "",
    previous_recognition: Optional[dict] = None,
) -> dict[str, Any]:
    """Answer a natural-language question with KG-backed knowledge.

    Supports multi-turn conversations by storing context per conversation_id.
    """
    # Manage conversation history
    if not conversation_id:
        import uuid
        conversation_id = str(uuid.uuid4())

    history = _conversations[conversation_id]
    if len(history) > settings.MAX_QA_HISTORY:
        history = history[-settings.MAX_QA_HISTORY:]

    # 1. Check for differentiation questions
    diff_answer = _check_similar_pest_differentiation(question)
    if diff_answer:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": diff_answer})
        _conversations[conversation_id] = history
        return _build_qa_response(conversation_id, question, diff_answer, [], 0.95)

    # 2. Query KG service for relevant knowledge
    kg_results = await _query_kg_search(question, user_role)

    # 3. Build answer with KG context
    answer, sources, confidence = _build_answer(
        question=question,
        kg_results=kg_results,
        user_role=user_role,
        county=county,
        previous_recognition=previous_recognition,
        history=history,
    )

    # 4. Generate follow-up suggestions
    follow_ups = _generate_follow_ups(question, kg_results, user_role)

    # 5. Store in conversation history
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    _conversations[conversation_id] = history

    return _build_qa_response(conversation_id, question, answer, sources, confidence, follow_ups)


# ---------------------------------------------------------------------------
# KG integration
# ---------------------------------------------------------------------------


async def _query_kg_search(query: str, user_role: str) -> dict:
    """Call the KG service semantic search endpoint."""
    url = f"{settings.KG_SERVICE_URL}/api/v1/kg/search"
    payload = {
        "query": query,
        "user_role": user_role,
        "max_results": 5,
        "include_intent_recommendation": True,
    }
    try:
        async with httpx.AsyncClient(timeout=settings.KG_SERVICE_TIMEOUT) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                return resp.json().get("data", {})
    except httpx.RequestError as e:
        logger.warning("KG service unavailable for QA: %s", e)
    return {}


# ---------------------------------------------------------------------------
# Answer builders
# ---------------------------------------------------------------------------


def _check_similar_pest_differentiation(question: str) -> Optional[str]:
    """Check if the user is asking to differentiate between similar pests."""
    for (pest_a, pest_b), explanation in SIMILAR_PEST_PAIRS.items():
        if pest_a in question and pest_b in question:
            return f"{pest_a}与{pest_b}的区别：\n{explanation}"
        # Single pest differentiation intent
        if "区别" in question or "区分" in question or "辨别" in question:
            if pest_a in question:
                return f"关于{pest_a}的鉴别要点：\n{explanation}"
    return None


def _build_answer(
    question: str,
    kg_results: dict,
    user_role: str,
    county: str,
    previous_recognition: Optional[dict],
    history: list[dict],
) -> tuple[str, list[dict], float]:
    """Build a contextual answer from KG results."""
    results = kg_results.get("results", [])
    parsed_intent = kg_results.get("parsed_intent", {})
    intent = parsed_intent.get("intent", "")

    sources: list[dict] = []
    for r in results[:3]:
        sources.append({
            "entity_id": r.get("entity_id", ""),
            "name": r.get("name", ""),
            "relation": "知识图谱匹配",
        })

    confidence = parsed_intent.get("confidence", 0.5)

    # If results found, build from KG
    if results:
        top = results[0]
        entity_name = top.get("name", "")
        summary = top.get("summary", "")

        if intent == "find_treatment":
            if user_role == "枣农":
                answer = (
                    f"关于「{entity_name}」的防治，建议如下：\n"
                    f"1. 农业防治：加强果园管理，及时清除病残体；\n"
                    f"2. 物理防治：悬挂诱虫板、安装杀虫灯；\n"
                    f"3. 必要时选用推荐药剂进行化学防治，注意安全间隔期。\n\n"
                    f"详细方案请在平台查看完整推荐。"
                )
            else:
                answer = (
                    f"「{entity_name}」的综合防治方案已从知识图谱检索到。\n"
                    f"{summary}\n\n"
                    f"建议结合田间实际发病情况，按综合防治（IPM）原则制定分级防控方案。"
                )
        elif intent == "identify_pest":
            answer = f"根据您的描述，这可能与「{entity_name}」相关。\n{summary}\n建议对照完整症状描述进行确认，或上传病虫害照片进行AI识别。"
        elif intent == "risk_check":
            answer = f"关于风险预警，已为您检索到「{entity_name}」的相关信息。\n{summary}\n建议关注当地气象预报和植保部门发布的病虫情报。"
        else:
            answer = f"已为您检索到「{entity_name}」的相关知识。\n{summary}"

        confidence = max(confidence, top.get("relevance_score", 0.5))
    else:
        # No KG results — use recognition context if available
        if previous_recognition:
            pest = previous_recognition.get("pest", "未知病虫害")
            conf = previous_recognition.get("confidence", 0)
            answer = (
                f"基于您最近上传的图像识别结果（{pest}，置信度{conf:.0%}），"
                f"建议您查看该病虫害的详细信息和防治方案。"
                f"如需更具体的建议，请提供更多症状描述。"
            )
        else:
            answer = _fallback_answer(question, user_role)

    return answer, sources, confidence


def _fallback_answer(question: str, user_role: str) -> str:
    """Fallback when no KG results match."""
    if user_role == "枣农":
        return (
            "抱歉，我暂时无法准确回答您的问题。建议您：\n"
            "1. 拍摄病虫害清晰照片，使用平台的「图像识别」功能进行诊断；\n"
            "2. 联系当地植保技术人员进行田间诊断；\n"
            "3. 使用更具体的关键词重新提问（如病虫害名称、症状描述等）。"
        )
    return f"知识图谱中未检索到与「{question}」高度匹配的结果。建议扩展检索范围或从其他数据源获取信息。"


def _generate_follow_ups(
    question: str,
    kg_results: dict,
    user_role: str,
) -> list[str]:
    """Generate contextual follow-up suggestions."""
    suggestions = []
    results = kg_results.get("results", [])

    if results:
        top_name = results[0].get("name", "")
        suggestions.append(f"{top_name}的典型症状是什么？")
        suggestions.append(f"如何防治{top_name}？")
        suggestions.append(f"{top_name}在什么季节最容易发生？")

    suggestions.extend([
        "当前季节需要重点防范哪些病虫害？",
        "如何进行果园的日常病虫害巡查？",
    ])

    return suggestions[:4]


def _build_qa_response(
    conversation_id: str,
    question: str,
    answer: str,
    sources: list[dict],
    confidence: float,
    follow_ups: Optional[list[str]] = None,
) -> dict:
    return {
        "conversation_id": conversation_id,
        "question": question,
        "answer": answer,
        "sources": sources,
        "confidence": round(confidence, 4),
        "follow_up_suggestions": follow_ups or [],
    }
