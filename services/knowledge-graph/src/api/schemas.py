"""
Pydantic models for the Knowledge Graph REST API.

Matches the gRPC messages defined in libs/proto/kg.proto.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------


class EntityRelation(BaseModel):
    relation_type: str = Field(..., description="e.g. '危害', '防治方法为'")
    target_entity_id: str
    target_name: str = ""
    target_type: str = ""
    confidence: float = 1.0
    properties: dict[str, str] = Field(default_factory=dict)


class EntityResponse(BaseModel):
    entity_id: str
    type: str
    name_cn: str
    category: str = ""
    description: str = ""
    scientific_name: str = ""
    typical_symptoms: str = ""
    severity_levels: str = ""
    relations: list[EntityRelation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Natural language query")
    user_role: str = Field(default="枣农", description="枣农 | 植保专家 | 管理员")
    max_results: int = Field(default=10, ge=1, le=50)
    include_intent_recommendation: bool = False


class ParsedIntent(BaseModel):
    intent: str = ""  # e.g. "identify_pest", "find_treatment", "risk_check"
    confidence: float = 0.0
    entities: list[str] = Field(default_factory=list)
    mapped_pests: list[dict] = Field(default_factory=list)


class PreventionSummary(BaseModel):
    name: str
    category: str = ""  # 农业防治 | 化学防治 | 物理防治 | 生物防治


class SearchResult(BaseModel):
    entity_id: str
    name: str
    relevance_score: float
    summary: str = ""
    prevention_methods: list[PreventionSummary] = Field(default_factory=list)


class IntentRecommendation(BaseModel):
    entity_id: str
    name: str
    reason: str = ""
    relevance_score: float = 0.0


class SearchResponse(BaseModel):
    query: str
    parsed_intent: ParsedIntent = Field(default_factory=ParsedIntent)
    results: list[SearchResult] = Field(default_factory=list)
    intent_recommendations: list[IntentRecommendation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------


class RecommendationRequest(BaseModel):
    pest_id: str = Field(..., description="KG entity ID of the pest/disease")
    county: str = Field(default="", description="County name for regional context")
    growth_stage: str = Field(default="", description="Current jujube growth stage")
    temperature: float = Field(default=25.0)
    humidity: float = Field(default=60.0)


class PreventionItem(BaseModel):
    method: str
    effectiveness: str = ""
    cost: str = ""
    safety_interval_days: int = 7
    details: str = ""


class PreventionCategory(BaseModel):
    items: list[PreventionItem] = Field(default_factory=list)


class ComprehensiveStrategy(BaseModel):
    priority_sequence: list[str] = Field(default_factory=list)
    integrated_plan: str = ""
    risk_warning: str = ""


class RecommendationResponse(BaseModel):
    pest_id: str
    pest_name: str = ""
    recommendations: dict[str, PreventionCategory] = Field(default_factory=dict)
    comprehensive_strategy: ComprehensiveStrategy = Field(default_factory=ComprehensiveStrategy)


# ---------------------------------------------------------------------------
# Risk Prediction
# ---------------------------------------------------------------------------


class RiskRequest(BaseModel):
    county: str = Field(..., description="County name")
    growth_stage: str = Field(default="", description="Current growth stage")


class CurrentWeather(BaseModel):
    temperature: float = 25.0
    humidity: float = 60.0
    precipitation_7d: float = 0.0
    weather_trend: str = ""


class RiskAssessment(BaseModel):
    pest_id: str
    pest_name: str
    risk_level: str  # 低 | 中 | 高
    risk_score: float
    reasoning: str = ""
    recommended_action: str = ""


class RiskResponse(BaseModel):
    county: str
    current_growth_stage: str = ""
    current_weather: CurrentWeather = Field(default_factory=CurrentWeather)
    risk_assessment: list[RiskAssessment] = Field(default_factory=list)
    alert_triggered: bool = False
    alert_pests: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthStatus(BaseModel):
    is_ready: bool
    entity_count: int = 0
    triple_count: int = 0
    neo4j_connected: bool = False


# ---------------------------------------------------------------------------
# Generic API wrapper
# ---------------------------------------------------------------------------


class APIResponse(BaseModel):
    code: int = 0
    message: str = "success"
    data: Optional[Any] = None
