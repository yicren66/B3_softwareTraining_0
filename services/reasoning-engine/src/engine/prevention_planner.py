"""
Personalised prevention plan generator.

Combines recognition results, weather/climate data, growth stage, and KG
recommendations to produce role-differentiated, actionable prevention plans.

SRS-SYS02 §4.2.3 KR-01, §4.2.4 KP-01~KP-03
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from config import settings, UserRole

logger = logging.getLogger(__name__)

# Prevention categories (SRS: 生防/物防/人防 framework)
CATEGORY_LABELS = {
    "生物防治": {"cn": "生物防治（生防）", "icon": "🦋", "priority": 1},
    "物理防治": {"cn": "物理防治（物防）", "icon": "💡", "priority": 2},
    "农业防治": {"cn": "农业/人工防治（人防）", "icon": "👨‍🌾", "priority": 0},
    "化学防治": {"cn": "化学防治", "icon": "⚗️", "priority": 3},
}

# Role-specific templates
ROLE_TEMPLATES = {
    UserRole.FARMER: {
        "explanation_style": "简洁易懂，突出操作步骤",
        "detail_level": "low",
    },
    UserRole.EXPERT: {
        "explanation_style": "包含技术细节和科学依据",
        "detail_level": "high",
    },
    UserRole.ADMIN: {
        "explanation_style": "侧重区域防控调度和资源统筹",
        "detail_level": "medium",
    },
    UserRole.VISITOR: {
        "explanation_style": "基础信息展示",
        "detail_level": "low",
    },
}


# ---------------------------------------------------------------------------
# Prevention plan generation
# ---------------------------------------------------------------------------


async def generate_prevention_plan(
    pest_id: str,
    pest_name: str = "",
    county: str = "",
    growth_stage: str = "",
    temperature: float = 25.0,
    humidity: float = 60.0,
    user_role: str = "枣农",
    confidence: float = 0.0,
    severity: str = "",
) -> dict[str, Any]:
    """Generate a complete personalised prevention plan.

    Args:
        pest_id: KG entity ID of identified pest/disease.
        pest_name: Human-readable name.
        county: User's county.
        growth_stage: Current jujube growth stage.
        temperature: Current temperature (°C).
        humidity: Current humidity (%).
        user_role: 枣农 | 植保专家 | 管理员 | 访客.
        confidence: Recognition confidence [0, 1].
        severity: 轻度 | 中度 | 重度.

    Returns:
        Dict with personalised_explanation and targeted_plan.
    """
    # 1. Fetch KG recommendations
    kg_recs = await _fetch_kg_recommendations(
        pest_id=pest_id,
        county=county,
        growth_stage=growth_stage,
        temperature=temperature,
        humidity=humidity,
    )

    # 2. Build personalised explanation
    explanation = _build_explanation(
        pest_name=pest_name,
        confidence=confidence,
        severity=severity,
        growth_stage=growth_stage,
        temperature=temperature,
        humidity=humidity,
        user_role=user_role,
    )

    # 3. Build targeted plan with priority
    plan = _build_targeted_plan(
        kg_recs=kg_recs,
        user_role=user_role,
        severity=severity,
        growth_stage=growth_stage,
    )

    return {
        "explanation": explanation,
        "plan": plan,
    }


# ---------------------------------------------------------------------------
# KG integration
# ---------------------------------------------------------------------------


async def _fetch_kg_recommendations(
    pest_id: str,
    county: str,
    growth_stage: str,
    temperature: float,
    humidity: float,
) -> dict:
    """Call the Knowledge Graph service for prevention recommendations."""
    url = f"{settings.KG_SERVICE_URL}/api/v1/kg/recommendation/{pest_id}"
    params = {
        "county": county,
        "growth_stage": growth_stage,
        "temperature": temperature,
        "humidity": humidity,
    }
    try:
        async with httpx.AsyncClient(timeout=settings.KG_SERVICE_TIMEOUT) as client:
            resp = await client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("data", {})
            logger.warning("KG service returned %d for %s", resp.status_code, pest_id)
    except httpx.RequestError as e:
        logger.warning("KG service unavailable: %s", e)

    return {}


# ---------------------------------------------------------------------------
# Explanation builder
# ---------------------------------------------------------------------------


def _build_explanation(
    pest_name: str,
    confidence: float,
    severity: str,
    growth_stage: str,
    temperature: float,
    humidity: float,
    user_role: str,
) -> dict:
    """Build role-appropriate personalised disease explanation."""
    role_cfg = ROLE_TEMPLATES.get(UserRole(user_role), ROLE_TEMPLATES[UserRole.FARMER])

    # Cause analysis
    cause_parts = [f"经AI识别，您的枣树可能感染了{pest_name}"]
    if confidence > 0:
        cause_parts.append(f"（置信度{confidence:.0%}）")

    if severity == "重度":
        cause_parts.append("，病情已达到重度水平")
    elif severity == "中度":
        cause_parts.append("，病情处于中度发展阶段")
    else:
        cause_parts.append("，病情处于初期阶段")

    if growth_stage:
        cause_parts.append(f"，当前正处于枣树{growth_stage}")

    cause_analysis = "".join(cause_parts) + "。"

    # Expert: add environmental analysis
    if role_cfg["detail_level"] == "high":
        env_analysis = (
            f"环境条件分析：当前温度{temperature:.0f}°C、湿度{humidity:.0f}%，"
        )
        if temperature > 28 and humidity > 75:
            env_analysis += "高温高湿条件有利于病害扩展蔓延，需重点防范。"
        elif temperature > 30 and humidity < 40:
            env_analysis += "高温干燥条件有利于螨类等害虫繁殖，需关注虫口密度变化。"
        else:
            env_analysis += "当前气候条件大致处于正常范围。"
        cause_analysis += "\n" + env_analysis

    # Harm description based on stage and severity
    stage_harm = _describe_stage_harm(pest_name, severity, growth_stage, user_role)

    return {
        "pest_name": pest_name,
        "cause_analysis": cause_analysis,
        "current_stage_harm_description": stage_harm,
    }


def _describe_stage_harm(
    pest_name: str,
    severity: str,
    growth_stage: str,
    user_role: str,
) -> str:
    """Describe the current harm level in the context of the growth stage."""
    role_cfg = ROLE_TEMPLATES.get(UserRole(user_role), ROLE_TEMPLATES[UserRole.FARMER])

    parts = []

    if severity == "重度":
        parts.append(f"{pest_name}已对枣树造成严重危害")
        if growth_stage in ("花期", "幼果期"):
            parts.append("，当前正值关键生育期，如不及时控制将导致严重减产甚至绝收")
        elif growth_stage in ("膨大期", "成熟期"):
            parts.append("，将直接影响果实品质和商品价值")
        else:
            parts.append("，需立即采取综合防控措施")
    elif severity == "中度":
        parts.append(f"{pest_name}正在发展中")
        parts.append(f"，建议在{growth_stage or '当前阶段'}及时防治，防止病情进一步恶化")
    else:
        parts.append(f"{pest_name}处于初发阶段，及时处理可有效控制")

    parts.append("。")

    # Expert: add technical details
    if role_cfg["detail_level"] == "high":
        parts.append(
            f"建议结合田间实际发病情况和气象预报，"
            f"制定分阶段精准防控方案。注意轮换用药，延缓抗药性产生。"
        )

    return "".join(parts)


# ---------------------------------------------------------------------------
# Targeted plan builder
# ---------------------------------------------------------------------------


def _build_targeted_plan(
    kg_recs: dict,
    user_role: str,
    severity: str,
    growth_stage: str,
) -> dict:
    """Build categorised prevention plan from KG recommendations."""
    role_cfg = ROLE_TEMPLATES.get(UserRole(user_role), ROLE_TEMPLATES[UserRole.FARMER])

    categories: dict[str, dict] = {}
    kg_categories = kg_recs.get("recommendations", {})

    for cat_name, cat_data in kg_categories.items():
        items = cat_data.get("items", [])[:settings.PLAN_MAX_ITEMS_PER_CATEGORY]
        if not items:
            continue

        label_info = CATEGORY_LABELS.get(cat_name, {"cn": cat_name, "icon": "", "priority": 99})

        # For farmers: simplify item descriptions
        if role_cfg["detail_level"] == "low":
            for item in items:
                if len(item.get("details", "")) > 80:
                    item["details"] = item["details"][:80] + "..."

        categories[cat_name] = {
            "items": items,
        }

    # Priority sequence based on severity and growth stage
    priority = _determine_priority_sequence(categories.keys(), severity, growth_stage)

    # Build integrated plan text
    integrated_plan = _build_integrated_plan_text(
        categories, priority, severity, growth_stage, user_role
    )

    # Risk warning
    risk_warning = ""
    if severity == "重度":
        risk_warning = "⚠️ 病情已达到重度水平，建议立即采取联防联控措施，并通报周边枣农协同防治。"
    elif severity == "中度":
        risk_warning = "⚠️ 病情中度发展，建议在3-5天内完成首轮防治。"

    return {
        "categories": categories,
        "priority_sequence_json": str(priority),
        "integrated_plan": integrated_plan,
        "risk_warning": risk_warning,
    }


def _determine_priority_sequence(
    category_names: list[str],
    severity: str,
    growth_stage: str,
) -> list[str]:
    """Determine prevention priority based on severity and growth stage."""
    base_priority = ["农业防治", "生物防治", "物理防治", "化学防治"]

    if severity == "重度":
        # Elevate chemical and biological for severe cases
        base_priority = ["化学防治", "生物防治", "农业防治", "物理防治"]

    if growth_stage in ("花期",):
        # During flowering, minimize chemical impact
        base_priority = ["生物防治", "农业防治", "物理防治", "化学防治"]

    # Filter to only what's available
    return [c for c in base_priority if c in category_names]


def _build_integrated_plan_text(
    categories: dict,
    priority: list[str],
    severity: str,
    growth_stage: str,
    user_role: str,
) -> str:
    """Compose a human-readable integrated prevention plan."""
    parts = []

    if user_role == "枣农":
        parts.append("📋 综合防治建议：\n")
        for cat_name in priority:
            if cat_name in categories:
                label = CATEGORY_LABELS.get(cat_name, {}).get("cn", cat_name)
                methods = categories[cat_name].get("items", [])
                if methods:
                    method_names = "、".join(m["method"] for m in methods[:2])
                    parts.append(f"• {label}：{method_names}")
        parts.append(f"\n💡 提醒：防治时请注意药剂安全间隔期，{growth_stage or '生长期'}避免使用高毒农药。")

    elif user_role == "植保专家":
        parts.append("综合防控技术方案：\n")
        for cat_name in priority:
            if cat_name in categories:
                label = CATEGORY_LABELS.get(cat_name, {}).get("cn", cat_name)
                methods = categories[cat_name].get("items", [])
                if methods:
                    parts.append(f"\n【{label}】")
                    for m in methods[:3]:
                        parts.append(f"  - {m['method']}（效果：{m.get('effectiveness', '—')}）")

    else:  # Admin
        parts.append("区域防控调度建议：\n")
        parts.append(f"病情等级：{severity}，建议启动联防联控预案。")
        parts.append("优先调度资源：")
        for cat_name in priority[:3]:
            if cat_name in categories:
                label = CATEGORY_LABELS.get(cat_name, {}).get("cn", cat_name)
                parts.append(f"  • {label}")

    return "\n".join(parts)
