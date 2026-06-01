"""
Pest outbreak risk predictor based on growth stage, climate, and KG data.

Queries the knowledge graph for pest-climate and pest-growth-cycle relations
to estimate outbreak risk for each pest/disease in a given region and phenology.
"""

from __future__ import annotations

import logging
from typing import Any

from config import settings
from neo4j.driver import get_session

logger = logging.getLogger(__name__)

# Default climate thresholds per pest category
DEFAULT_CLIMATE_RISK = {
    "病害": {"temp_range": (25, 35), "humidity_threshold": 75, "precip_factor": 1.5},
    "虫害": {"temp_range": (20, 30), "humidity_threshold": 50, "precip_factor": 0.8},
}


# ---------------------------------------------------------------------------
# Risk prediction
# ---------------------------------------------------------------------------


async def predict_risks(
    county: str,
    growth_stage: str = "",
    temperature: float = 25.0,
    humidity: float = 60.0,
    precipitation_7d: float = 0.0,
) -> dict[str, Any]:
    """Predict pest/disease outbreak risks for a given location and conditions.

    Args:
        county: County name (e.g. 沧县).
        growth_stage: Current growth stage (e.g. 幼果期).
        temperature: Current temperature (°C).
        humidity: Current relative humidity (%).
        precipitation_7d: Total precipitation in last 7 days (mm).

    Returns:
        Risk assessment dict with per-pest risk scores, alerts, and recommendations.
    """
    # 1. Query KG for pests active in this growth stage
    cypher = """
    MATCH (p:PestDisease)
    WHERE p.entity_id STARTS WITH 'KG-ENT-'
    OPTIONAL MATCH (p)-[r1]->(gc:GrowthCycle)
    WHERE $growth_stage = '' OR gc.stage_name = $growth_stage
    OPTIONAL MATCH (p)-[r2]->(c:ClimateCondition)
    RETURN p.entity_id AS entity_id,
           p.name_cn AS name,
           p.category AS category,
           p.description AS description,
           collect(DISTINCT gc.stage_name) AS related_stages,
           collect(DISTINCT c.name_cn) AS related_climate
    ORDER BY p.entity_id
    """

    async with get_session() as session:
        result = await session.run(
            cypher,
            growth_stage=growth_stage,
        )
        pests = await result.data()

    if not pests:
        # Fallback: query all core pests
        cypher_all = """
        MATCH (p:PestDisease)
        WHERE p.is_core = 'true' OR p.is_core = true
        RETURN p.entity_id AS entity_id,
               p.name_cn AS name,
               p.category AS category,
               p.description AS description
        LIMIT 30
        """
        async with get_session() as session:
            result = await session.run(cypher_all)
            pests = await result.data()

    # 2. Score each pest
    assessments: list[dict] = []
    alert_pests: list[str] = []

    for pest in pests:
        pest_id = pest.get("entity_id", "")
        pest_name = pest.get("name", pest_id)
        category = pest.get("category", "病害")
        related_stages = pest.get("related_stages", []) or []
        related_climate = pest.get("related_climate", []) or []

        risk_score = _calculate_risk_score(
            category=category,
            growth_stage=growth_stage,
            related_stages=related_stages,
            related_climate=related_climate,
            temperature=temperature,
            humidity=humidity,
            precipitation_7d=precipitation_7d,
        )

        risk_level = _score_to_level(risk_score)

        if risk_level == "高":
            alert_pests.append(pest_name)

        assessments.append({
            "pest_id": pest_id,
            "pest_name": pest_name,
            "risk_level": risk_level,
            "risk_score": round(risk_score, 3),
            "reasoning": _build_reasoning(
                pest_name, category, risk_score,
                growth_stage, temperature, humidity
            ),
            "recommended_action": _recommend_action(risk_level, pest_name, growth_stage),
        })

    # Sort by risk score descending
    assessments.sort(key=lambda x: x["risk_score"], reverse=True)

    alert_triggered = len(alert_pests) > 0 and settings.RISK_ALERT_ENABLED

    return {
        "county": county,
        "current_growth_stage": growth_stage or "未知",
        "current_weather": {
            "temperature": temperature,
            "humidity": humidity,
            "precipitation_7d": precipitation_7d,
            "weather_trend": _weather_trend(temperature, humidity),
        },
        "risk_assessment": assessments,
        "alert_triggered": alert_triggered,
        "alert_pests": alert_pests,
    }


# ---------------------------------------------------------------------------
# Risk scoring
# ---------------------------------------------------------------------------


def _calculate_risk_score(
    category: str,
    growth_stage: str,
    related_stages: list[str],
    related_climate: list[str],
    temperature: float,
    humidity: float,
    precipitation_7d: float,
) -> float:
    """Calculate a 0-1 risk score based on environmental factors."""
    defaults = DEFAULT_CLIMATE_RISK.get(category, DEFAULT_CLIMATE_RISK["病害"])
    score = 0.3  # base risk

    # Temperature match
    t_min, t_max = defaults["temp_range"]
    if t_min <= temperature <= t_max:
        score += 0.2
    elif abs(temperature - t_min) < 5 or abs(temperature - t_max) < 5:
        score += 0.1

    # Humidity
    if category == "病害":
        if humidity > defaults["humidity_threshold"]:
            score += 0.2
        elif humidity > defaults["humidity_threshold"] - 15:
            score += 0.1
    else:  # 虫害
        if humidity < defaults["humidity_threshold"]:
            score += 0.15  # mites thrive in dry conditions

    # Precipitation
    if precipitation_7d > 50:
        score += 0.15 if category == "病害" else -0.05

    # Growth stage match
    if growth_stage and growth_stage in related_stages:
        score += 0.15

    # Climate correlation
    if related_climate:
        score += 0.05 * min(len(related_climate), 3)

    return min(0.99, max(0.05, score))


def _score_to_level(score: float) -> str:
    if score >= settings.RISK_HIGH_THRESHOLD:
        return "高"
    elif score >= 0.4:
        return "中"
    return "低"


def _build_reasoning(
    name: str,
    category: str,
    score: float,
    stage: str,
    temp: float,
    humidity: float,
) -> str:
    parts = [f"{name}（{category}）"]
    if score >= 0.7:
        parts.append("当前环境条件非常有利于其发生和蔓延")
    elif score >= 0.4:
        parts.append("当前环境条件较有利于其发生")
    else:
        parts.append("当前环境条件不太利于其发生")

    if stage:
        parts.append(f"当前{stage}是关注期")
    parts.append(f"温度{temp:.0f}°C，湿度{humidity:.0f}%")
    return "；".join(parts)


def _recommend_action(risk_level: str, name: str, stage: str) -> str:
    if risk_level == "高":
        return f"立即对{name}进行田间调查，采取综合防控措施，必要时启动联防联治。"
    elif risk_level == "中":
        return f"加强{name}田间监测，每3-5天巡查一次，提前准备防治物资。"
    return f"常规监测即可，关注天气变化和{stage or '物候'}进展。"


def _weather_trend(temperature: float, humidity: float) -> str:
    if temperature > 30 and humidity > 80:
        return "高温高湿，病害风险升高"
    elif temperature > 30 and humidity < 40:
        return "高温干燥，螨类风险升高"
    elif temperature < 15:
        return "低温，病虫害活动减弱"
    return "气候条件适中"
