"""
Prevention & treatment recommendation engine.

Queries the knowledge graph for prevention methods associated with a given
pest/disease, categorises them by type (农业防治/化学防治/物理防治/生物防治),
and generates a comprehensive integrated strategy.
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j.driver import get_entity_by_id, get_session
from config import settings

logger = logging.getLogger(__name__)

# Strategy priority by pest/disease category and severity context
CATEGORY_PRIORITY = ["农业防治", "生物防治", "物理防治", "化学防治"]

# Comprehensive strategy templates per control type
STRATEGY_TEMPLATES = {
    "农业防治": "以农业措施为基础，通过{methods}等手段改善果园生态环境，降低病虫害发生基数。",
    "生物防治": "优先采用{methods}等生物防治手段，保护天敌，减少化学农药使用。",
    "物理防治": "配合{methods}等物理防治措施，有效降低害虫种群密度。",
    "化学防治": "在必要时选用{methods}等高效低毒药剂进行化学防治，严格掌握安全间隔期。",
}


# ---------------------------------------------------------------------------
# Recommendation query
# ---------------------------------------------------------------------------


async def get_prevention_recommendations(
    pest_id: str,
    county: str = "",
    growth_stage: str = "",
    temperature: float = 25.0,
    humidity: float = 60.0,
) -> dict[str, Any]:
    """Retrieve prevention recommendations categorised by method type.

    Args:
        pest_id: KG entity ID (e.g. KG-ENT-003 for 枣树锈病).
        county: County name for regional context.
        growth_stage: Current jujube growth stage (e.g. 幼果期).
        temperature: Current temperature in °C.
        humidity: Current relative humidity in %.

    Returns:
        Dict with pest info, categorised recommendations, and comprehensive strategy.
    """
    # 1. Fetch pest entity
    pest = await get_entity_by_id(pest_id)
    if pest is None:
        logger.warning("Pest entity %s not found in KG.", pest_id)
        return _empty_recommendation(pest_id)

    pest_name = pest.get("name_cn", pest_id)

    # 2. Query prevention methods via KG relations
    cypher = """
    MATCH (p:PestDisease {entity_id: $pest_id})-[r]->(m:PreventionMethod)
    RETURN m.name_cn AS method_name,
           m.category AS category,
           m.description AS description,
           m.applicable_season AS season,
           type(r) AS relation_type
    UNION
    MATCH (p:PestDisease {entity_id: $pest_id})-[r]->(pesticide:Pesticide)
    RETURN pesticide.name_cn AS method_name,
           '化学防治' AS category,
           coalesce(pesticide.description, '') AS description,
           '' AS season,
           type(r) AS relation_type
    """
    methods: dict[str, list[dict]] = {
        "农业防治": [],
        "化学防治": [],
        "物理防治": [],
        "生物防治": [],
    }

    async with get_session() as session:
        result = await session.run(cypher, pest_id=pest_id)
        async for record in result:
            cat = record.get("category", "农业防治")
            if cat not in methods:
                cat = "农业防治"
            methods[cat].append({
                "method": record.get("method_name", ""),
                "effectiveness": _assess_effectiveness(cat, growth_stage, temperature, humidity),
                "cost": _estimate_cost(cat),
                "safety_interval_days": _safety_interval(cat),
                "details": record.get("description", ""),
            })

    # 3. If no KG relations found, provide generic recommendations
    if not any(methods.values()):
        methods = _get_generic_recommendations(pest_name, growth_stage)

    # 4. Build comprehensive strategy
    strategy = _build_comprehensive_strategy(methods, pest_name, temperature, humidity)

    return {
        "pest_id": pest_id,
        "pest_name": pest_name,
        "recommendations": {
            cat: {"items": items[:settings.RECOMMENDATION_MAX_ITEMS]}
            for cat, items in methods.items()
        },
        "comprehensive_strategy": strategy,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assess_effectiveness(category: str, stage: str, temp: float, humidity: float) -> str:
    """Heuristic effectiveness assessment."""
    if category == "生物防治":
        return "高 — 对环境和天敌友好"
    elif category == "化学防治":
        if stage in ("花期", "幼果期"):
            return "中 — 花期/幼果期须谨慎用药"
        return "高 — 快速压低病虫害种群密度"
    elif category == "物理防治":
        return "中 — 作为辅助手段效果良好"
    return "中"


def _estimate_cost(category: str) -> str:
    costs = {"农业防治": "低", "物理防治": "低-中", "生物防治": "中", "化学防治": "中-高"}
    return costs.get(category, "中")


def _safety_interval(category: str) -> int:
    intervals = {"农业防治": 0, "物理防治": 0, "生物防治": 3, "化学防治": 7}
    return intervals.get(category, 7)


def _build_comprehensive_strategy(
    methods: dict[str, list[dict]],
    pest_name: str,
    temperature: float,
    humidity: float,
) -> dict:
    """Generate the integrated prevention strategy."""
    priority = CATEGORY_PRIORITY.copy()

    # Adjust priority based on conditions
    if temperature > 30 and humidity > 80:
        # Hot and humid — disease risk high, elevate chemical options
        priority = ["农业防治", "化学防治", "生物防治", "物理防治"]
    elif humidity < 40:
        # Dry — mite risk, elevate biological
        priority = ["生物防治", "农业防治", "化学防治", "物理防治"]

    integrated_plan_parts = []
    for cat in priority:
        items = methods.get(cat, [])
        if items:
            method_names = "、".join(item["method"] for item in items[:3])
            integrated_plan_parts.append(
                f"【{cat}】{method_names}"
            )

    risk_warning = ""
    if temperature > 30 and humidity > 80:
        risk_warning = f"当前高温高湿条件利于{pest_name}爆发，建议每3-5天巡查一次。"
    elif temperature > 28:
        risk_warning = f"当前温度偏高，注意{pest_name}发生动态，及时采取预防措施。"

    return {
        "priority_sequence": priority,
        "integrated_plan": "；".join(integrated_plan_parts) if integrated_plan_parts else f"针对{pest_name}的综合防治方案请咨询植保专家。",
        "risk_warning": risk_warning,
    }


def _get_generic_recommendations(pest_name: str, growth_stage: str) -> dict[str, list[dict]]:
    """Fallback generic recommendations when no KG relations exist."""
    return {
        "农业防治": [{
            "method": "加强果园管理，合理修剪通风透光",
            "effectiveness": "高", "cost": "低", "safety_interval_days": 0,
            "details": "清除病虫枝、枯枝、落叶，减少越冬病虫源。"
        }],
        "生物防治": [{
            "method": "保护和利用天敌，使用生物源农药",
            "effectiveness": "中", "cost": "中", "safety_interval_days": 3,
            "details": "释放赤眼蜂、草蛉等天敌昆虫，或使用Bt制剂、白僵菌等。"
        }],
        "物理防治": [{
            "method": "悬挂诱虫板，安装频振式杀虫灯",
            "effectiveness": "中", "cost": "低-中", "safety_interval_days": 0,
            "details": "利用害虫趋光性、趋色性诱杀成虫，降低下一代虫口密度。"
        }],
        "化学防治": [{
            "method": "选用高效低毒化学药剂轮换使用",
            "effectiveness": "高", "cost": "中-高", "safety_interval_days": 7,
            "details": "根据病虫害种类选择针对性药剂，严格按推荐剂量使用，注意安全间隔期。"
        }],
    }


def _empty_recommendation(pest_id: str) -> dict:
    return {
        "pest_id": pest_id,
        "pest_name": pest_id,
        "recommendations": {},
        "comprehensive_strategy": {
            "priority_sequence": CATEGORY_PRIORITY,
            "integrated_plan": "暂无针对性防治方案，请联系植保专家。",
            "risk_warning": "",
        },
    }
