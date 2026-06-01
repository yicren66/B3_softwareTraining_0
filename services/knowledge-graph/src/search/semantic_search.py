"""
Semantic search engine using sentence-transformers for Chinese text.

Maps natural-language queries (e.g. "枣叶发黄是什么病") to knowledge-graph
entities via embedding similarity and jieba keyword extraction.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import numpy as np
import jieba

from config import settings

logger = logging.getLogger(__name__)

# Lazy-loaded embedding model
_embedding_model: Optional[object] = None

# Intent keywords for rule-based intent classification
INTENT_PATTERNS: dict[str, list[str]] = {
    "identify_pest": [
        "是什么病", "什么虫", "什么病虫害", "识别", "诊断", "判断",
        "这是", "是不是", "属于", "哪种",
    ],
    "find_treatment": [
        "怎么治", "防治", "打什么药", "用什么药", "怎么办", "处理",
        "治疗", "防治方法", "措施", "对策", "农药", "杀虫",
    ],
    "risk_check": [
        "会不会爆发", "风险", "预警", "预防", "会不会有", "可能",
        "季节", "什么时候", "容易得",
    ],
    "differentiate": [
        "区别", "区分", "辨别", "哪个是", "还是", "鉴别", "差异", "不同",
    ],
    "growth_stage": [
        "什么阶段", "什么时候", "物候期", "生长", "萌芽", "花期", "幼果",
    ],
}

# Direct pest/disease name mapping for keyword extraction
PEST_KEYWORDS: dict[str, str] = {
    "枣炭疽病": "KG-ENT-001", "炭疽病": "KG-ENT-001",
    "枣疯病": "KG-ENT-002", "疯病": "KG-ENT-002",
    "枣树锈病": "KG-ENT-003", "锈病": "KG-ENT-003",
    "枣缩果病": "KG-ENT-004", "缩果病": "KG-ENT-004",
    "枣果腐病": "KG-ENT-005", "果腐病": "KG-ENT-005",
    "枣褐斑病": "KG-ENT-006", "褐斑病": "KG-ENT-006",
    "枣叶黑斑病": "KG-ENT-007", "黑斑病": "KG-ENT-007",
    "枣芽象甲": "KG-ENT-008", "芽象甲": "KG-ENT-008",
    "枣瘿蚊": "KG-ENT-009", "瘿蚊": "KG-ENT-009",
    "桃小食心虫": "KG-ENT-010", "食心虫": "KG-ENT-010",
    "绿盲蝽": "KG-ENT-011", "盲蝽": "KG-ENT-011",
    "枣尺蠖": "KG-ENT-012", "尺蠖": "KG-ENT-012",
    "枣镰翅小卷蛾": "KG-ENT-013", "小卷蛾": "KG-ENT-013", "卷叶蛾": "KG-ENT-013",
    "枣红蜘蛛": "KG-ENT-014", "红蜘蛛": "KG-ENT-014",
    "枣龟蜡蚧": "KG-ENT-015", "龟蜡蚧": "KG-ENT-015", "蜡蚧": "KG-ENT-015",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _get_embedding_model():
    """Lazy-load the sentence-transformers model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            logger.info("Embedding model loaded: %s (dim=%d)", settings.EMBEDDING_MODEL_NAME, settings.EMBEDDING_DIM)
        except Exception:
            logger.warning("Could not load SentenceTransformer; using fallback keyword search.")
            _embedding_model = None
    return _embedding_model


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------


def classify_intent(query: str) -> tuple[str, float, list[str]]:
    """Rule-based intent classification from natural language query.

    Returns:
        (intent_label, confidence, extracted_keyword_entities)
    """
    scores: dict[str, int] = {}
    for intent, keywords in INTENT_PATTERNS.items():
        score = sum(1 for kw in keywords if kw in query)
        if score > 0:
            scores[intent] = score

    if not scores:
        return ("find_treatment", 0.5, [])  # default intent

    top_intent = max(scores, key=scores.get)
    confidence = min(0.95, scores[top_intent] / max(len(INTENT_PATTERNS[top_intent]), 1) * 2.0)
    return top_intent, round(confidence, 3), []


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------


def extract_pest_entities(query: str) -> list[dict]:
    """Extract pest/disease name mentions and map to KG entity IDs."""
    matched: list[dict] = []
    seen_ids: set[str] = set()

    # Direct keyword matching
    for keyword, entity_id in PEST_KEYWORDS.items():
        if keyword in query:
            if entity_id not in seen_ids:
                seen_ids.add(entity_id)
                matched.append({
                    "entity_id": entity_id,
                    "name": keyword,
                    "match_confidence": 1.0 if len(keyword) >= 3 else 0.7,
                })

    # Jieba tokenization for partial matching
    tokens = set(jieba.lcut(query))
    for keyword, entity_id in PEST_KEYWORDS.items():
        if entity_id not in seen_ids:
            kw_tokens = set(jieba.lcut(keyword))
            overlap = tokens & kw_tokens
            if len(overlap) >= 2:
                seen_ids.add(entity_id)
                matched.append({
                    "entity_id": entity_id,
                    "name": keyword,
                    "match_confidence": 0.6,
                })

    return matched


# ---------------------------------------------------------------------------
# Query → embedding → entity similarity
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1024)
def _encode_cached(text: str) -> Optional[np.ndarray]:
    """Cache embeddings for frequently asked queries."""
    model = _get_embedding_model()
    if model is None:
        return None
    return model.encode(text, convert_to_numpy=True)


def compute_query_embedding(query: str) -> Optional[list[float]]:
    """Encode a query string into an embedding vector."""
    emb = _encode_cached(query)
    if emb is None:
        return None
    return emb.tolist()


# ---------------------------------------------------------------------------
# Intent recommendation
# ---------------------------------------------------------------------------


def generate_intent_recommendations(
    query: str,
    intent: str,
    matched_pests: list[dict],
) -> list[dict]:
    """Based on intent and matched pests, suggest related knowledge graph entities."""
    recommendations: list[dict] = []

    if intent == "identify_pest":
        # Recommend similar pests for differential diagnosis
        for pest in matched_pests[:3]:
            recommendations.append({
                "entity_id": pest["entity_id"],
                "name": pest["name"],
                "reason": "可能与您的描述相符，请查看详细症状进行比对",
                "relevance_score": pest["match_confidence"],
            })

    elif intent == "find_treatment":
        # Point directly to treatment recommendations
        for pest in matched_pests[:3]:
            recommendations.append({
                "entity_id": pest["entity_id"],
                "name": pest["name"],
                "reason": "查看该病虫害的综合防治方案",
                "relevance_score": 0.9,
            })

    elif intent == "risk_check":
        recommendations.append({
            "entity_id": "KG-STR-003",
            "name": "生物防治策略",
            "reason": "了解生物防治手段降低病虫害风险",
            "relevance_score": 0.7,
        })

    return recommendations
