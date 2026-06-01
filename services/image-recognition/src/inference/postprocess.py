"""
Result postprocessing: label mapping, confidence calibration, severity
thresholds, and formatting into the API-facing RecognitionData dict.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import settings
from inference.engine import RecognitionResult, SmallTarget

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Knowledge-graph entity IDs — synced with kg-construction/seed_data/pests_diseases.csv
# ---------------------------------------------------------------------------

KG_ENTITY_PREFIX = "KG-ENT"


# ---------------------------------------------------------------------------
# Disease & pest label map  — 「7病8虫」15类核心防控对象 (SRS-SYS02 §4.1.1 IR-02)
# ---------------------------------------------------------------------------
# 7 diseases (0-6):  枣炭疽病, 枣疯病, 枣树锈病, 枣缩果病, 枣果腐病, 枣褐斑病, 枣叶黑斑病
# 8 pests    (7-14): 枣芽象甲, 枣瘿蚊, 桃小食心虫, 绿盲蝽, 枣尺蠖, 枣镰翅小卷蛾, 枣红蜘蛛, 枣龟蜡蚧
# ---------------------------------------------------------------------------

DISEASE_MAP: Dict[int, Dict[str, Any]] = {
    0: {
        "name_cn": "枣炭疽病",
        "name_en": "Jujube Anthracnose",
        "scientific_name": "Colletotrichum gloeosporioides",
        "category": "病害",
        "affected_part": "果实,叶片",
        "symptoms": "果实出现褐色凹陷病斑；病斑上有轮纹状黑色小点；叶片出现不规则褐斑；严重时果实腐烂脱落。",
        "kg_entity_id": "KG-ENT-001",
    },
    1: {
        "name_cn": "枣疯病",
        "name_en": "Jujube Witches' Broom",
        "scientific_name": "Candidatus Phytoplasma ziziphi",
        "category": "病害",
        "affected_part": "枝叶,花器",
        "symptoms": "枝叶丛生呈扫帚状；花器退化成叶状结构；节间缩短；叶片变小黄化；病树逐渐枯死。",
        "kg_entity_id": "KG-ENT-002",
    },
    2: {
        "name_cn": "枣树锈病",
        "name_en": "Jujube Rust",
        "scientific_name": "Phakopsora ziziphi-vulgaris",
        "category": "病害",
        "affected_part": "叶片",
        "symptoms": "叶片背面出现黄褐色锈状粉末孢子堆；叶片正面出现褪绿黄斑；严重时叶片枯黄早落；树势衰弱。",
        "kg_entity_id": "KG-ENT-003",
    },
    3: {
        "name_cn": "枣缩果病",
        "name_en": "Jujube Fruit Shrink Disease",
        "scientific_name": "Alternaria alternata",
        "category": "病害",
        "affected_part": "果实",
        "symptoms": "果实表面出现淡褐色小斑；病斑迅速扩大果肉坏死；果实皱缩干瘪；果皮变为红褐色至黑褐色；病果易脱落。",
        "kg_entity_id": "KG-ENT-004",
    },
    4: {
        "name_cn": "枣果腐病",
        "name_en": "Jujube Fruit Rot",
        "scientific_name": "Fusarium spp.",
        "category": "病害",
        "affected_part": "果实",
        "symptoms": "果实表面出现水渍状斑；病斑扩大后果肉软腐；病部产生白色或粉色霉层；果实腐烂脱落；有酸臭味。",
        "kg_entity_id": "KG-ENT-005",
    },
    5: {
        "name_cn": "枣褐斑病",
        "name_en": "Jujube Brown Spot",
        "scientific_name": "Phoma spp.",
        "category": "病害",
        "affected_part": "叶片,果实",
        "symptoms": "叶片出现圆形或不规则褐色斑点；病斑边缘深褐色中央灰白色；严重时病斑连片叶片焦枯；果实出现褐色凹陷斑。",
        "kg_entity_id": "KG-ENT-006",
    },
    6: {
        "name_cn": "枣叶黑斑病",
        "name_en": "Jujube Black Leaf Spot",
        "scientific_name": "Alternaria spp.",
        "category": "病害",
        "affected_part": "叶片,新梢",
        "symptoms": "叶片出现黑色圆形小斑点；病斑逐渐扩大边缘有黄色晕圈；严重时叶片布满黑斑；病叶早落；新梢生长受阻。",
        "kg_entity_id": "KG-ENT-007",
    },
    # --- 8 pests (虫害) ---
    7: {
        "name_cn": "枣芽象甲",
        "name_en": "Jujube Bud Weevil",
        "scientific_name": "Scythropus yasumatsui",
        "category": "虫害",
        "affected_part": "嫩芽,幼叶,幼果",
        "symptoms": "成虫啃食嫩芽和幼叶；叶片出现缺刻和孔洞；严重时嫩芽被吃光；幼果被害出现疤痕；树体生长受阻。",
        "kg_entity_id": "KG-ENT-008",
    },
    8: {
        "name_cn": "枣瘿蚊",
        "name_en": "Jujube Gall Midge",
        "scientific_name": "Dasineura datifolia",
        "category": "虫害",
        "affected_part": "嫩叶,新梢",
        "symptoms": "叶片卷曲形成不规则虫瘿；虫瘿初为绿色后变红褐色；叶片畸形扭曲；严重时叶片枯焦早落；新梢生长受抑制。",
        "kg_entity_id": "KG-ENT-009",
    },
    9: {
        "name_cn": "桃小食心虫",
        "name_en": "Peach Fruit Borer",
        "scientific_name": "Carposina sasakii",
        "category": "虫害",
        "affected_part": "果实",
        "symptoms": "幼虫蛀入果实取食果肉；果面有蛀入孔和虫粪；果实内部充满虫粪；被害果提前变红脱落；虫道内可见幼虫。",
        "kg_entity_id": "KG-ENT-010",
    },
    10: {
        "name_cn": "绿盲蝽",
        "name_en": "Green Plant Bug",
        "scientific_name": "Apolygus lucorum",
        "category": "虫害",
        "affected_part": "嫩叶,嫩芽,幼果",
        "symptoms": "嫩叶被刺吸后出现小孔；叶片展开后孔洞扩大呈破叶状；幼果被刺吸后出现凹陷斑；果实畸形硬化；嫩芽枯萎。",
        "kg_entity_id": "KG-ENT-011",
    },
    11: {
        "name_cn": "枣尺蠖",
        "name_en": "Jujube Looper",
        "scientific_name": "Chihuo zao",
        "category": "虫害",
        "affected_part": "叶片",
        "symptoms": "幼虫咀嚼叶片造成缺刻；大龄幼虫可吃光全叶仅留主脉；严重时整株叶片被食尽；树势极度衰弱；当年和次年产量锐减。",
        "kg_entity_id": "KG-ENT-012",
    },
    12: {
        "name_cn": "枣镰翅小卷蛾",
        "name_en": "Jujube Leaf Roller",
        "scientific_name": "Ancylis sativa",
        "category": "虫害",
        "affected_part": "叶片,花芽",
        "symptoms": "幼虫吐丝将嫩叶纵卷；卷叶内取食叶肉留下表皮；叶片枯黄；严重时全树叶片卷曲；影响花芽分化和产量。",
        "kg_entity_id": "KG-ENT-013",
    },
    13: {
        "name_cn": "枣红蜘蛛",
        "name_en": "Jujube Red Spider Mite",
        "scientific_name": "Tetranychus cinnabarinus",
        "category": "虫害",
        "affected_part": "叶片",
        "symptoms": "叶片出现褪绿小点；严重时叶片呈灰白色；叶背面有细密蛛网；叶片焦枯早落；树势衰弱果实品质下降。",
        "kg_entity_id": "KG-ENT-014",
    },
    14: {
        "name_cn": "枣龟蜡蚧",
        "name_en": "Jujube Wax Scale",
        "scientific_name": "Ceroplastes japonicus",
        "category": "虫害",
        "affected_part": "枝干,叶片",
        "symptoms": "枝干和叶片上附着白色或灰白色蜡壳；叶片褪绿发黄；排泄物诱发煤污病；枝干生长受阻；严重时枝条枯死。",
        "kg_entity_id": "KG-ENT-015",
    },
}


# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------

SEVERITY_MAP: Dict[int, Dict[str, Any]] = {
    0: {"label_cn": "健康", "label_en": "Healthy", "level": 0,
        "description": "未见明显病虫害症状"},
    1: {"label_cn": "轻度", "label_en": "Mild", "level": 1,
        "description": "病虫害初发，危害面积<10%，建议密切关注"},
    2: {"label_cn": "中度", "label_en": "Moderate", "level": 2,
        "description": "病虫害发展，危害面积10%-30%，建议及时防治"},
    3: {"label_cn": "重度", "label_en": "Severe", "level": 3,
        "description": "病虫害严重，危害面积>30%，需立即采取综合防控措施"},
}


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------


def map_class_idx_to_name(idx: int) -> Tuple[str, str, str, str, str]:
    """Map a predicted class index to human-readable metadata.

    Returns:
        (name_cn, category, affected_part, symptoms, kg_entity_id)
    """
    entry = DISEASE_MAP.get(idx)
    if entry is None:
        logger.warning("Unknown class index %d — returning 'unknown'.", idx)
        return (
            "未知", "unknown", "",
            "Unrecognised disease class.",
            f"{KG_ENTITY_PREFIX}.unknown",
        )
    return (
        entry["name_cn"],
        entry["category"],
        entry["affected_part"],
        entry["symptoms"],
        entry["kg_entity_id"],
    )


def map_severity(idx: int) -> Dict[str, Any]:
    """Map severity index → label dict."""
    return SEVERITY_MAP.get(idx, SEVERITY_MAP[0])


# ---------------------------------------------------------------------------
# Confidence calibration (temperature scaling)
# ---------------------------------------------------------------------------


def calibrate_confidence(
    probs: List[float], temperature: Optional[float] = None
) -> List[float]:
    """Apply temperature scaling to raw probabilities.

    Args:
        probs: Raw probability vector (should sum to ~1).
        temperature: T > 1 flattens, T < 1 sharpens. Uses config default if None.

    Returns:
        List[float]: Calibrated probabilities.
    """
    T = temperature if temperature is not None else settings.TEMPERATURE
    if T <= 0:
        raise ValueError(f"Temperature must be > 0, got {T}.")
    arr = np.array(probs, dtype=np.float64)
    # Clamp to avoid log(0)
    arr = np.clip(arr, 1e-9, 1.0)
    logits = np.log(arr)
    scaled = logits / T
    calibrated = np.exp(scaled - np.max(scaled))
    calibrated /= calibrated.sum()
    return calibrated.tolist()


# ---------------------------------------------------------------------------
# Severity threshold logic
# ---------------------------------------------------------------------------


def apply_severity_threshold(
    severity_idx: int,
    severity_probs: List[float],
    confidence_threshold: Optional[float] = None,
) -> Tuple[int, float]:
    """If the top severity probability is below the confidence threshold,
    default to 'Mild' (idx=1) as a conservative estimate.

    Returns:
        (adjusted_severity_idx, top_severity_confidence)
    """
    threshold = (
        confidence_threshold
        if confidence_threshold is not None
        else settings.CONFIDENCE_THRESHOLD
    )
    top_conf = severity_probs[severity_idx] if severity_probs else 0.0
    if top_conf < threshold:
        logger.debug(
            "Severity confidence %.3f below threshold %.3f; defaulting to Mild.",
            top_conf, threshold,
        )
        return 1, severity_probs[1] if len(severity_probs) > 1 else top_conf
    return severity_idx, top_conf


# ---------------------------------------------------------------------------
# Format → API schema
# ---------------------------------------------------------------------------


def format_recognition_result(result: RecognitionResult) -> Dict[str, Any]:
    """Convert a raw RecognitionResult into the API-facing RecognitionData dict.

    This is the canonical schema expected by the gRPC / REST consumers.
    """
    # Calibrate
    calib_class_probs = calibrate_confidence(result.class_probs)
    calib_sev_probs = calibrate_confidence(result.severity_probs)

    # Map labels
    name_cn, category, affected_part, symptoms, kg_id = map_class_idx_to_name(
        result.class_idx
    )
    sev = map_severity(result.severity_idx)
    adj_sev_idx, sev_conf = apply_severity_threshold(
        result.severity_idx, calib_sev_probs
    )
    adj_sev = map_severity(adj_sev_idx)

    # Small targets
    targets_out: List[Dict[str, Any]] = []
    for t in result.small_targets:
        targets_out.append({
            "bbox": list(t.bbox),
            "class_id": t.class_id,
            "confidence": round(t.confidence, 4),
            "class_name": t.class_name or "pest_spot",
        })

    return {
        "prediction": {
            "class_idx": result.class_idx,
            "name_cn": name_cn,
            "name_en": DISEASE_MAP.get(result.class_idx, {}).get("name_en", "Unknown"),
            "category": category,
            "affected_part": affected_part,
            "symptoms": symptoms,
            "confidence": round(
                float(calib_class_probs[result.class_idx]), 4
            ),
            "class_probabilities": [
                round(float(p), 4) for p in calib_class_probs
            ],
        },
        "severity": {
            "level": adj_sev["level"],
            "label_cn": adj_sev["label_cn"],
            "label_en": adj_sev["label_en"],
            "confidence": round(sev_conf, 4),
            "probabilities": [round(float(p), 4) for p in calib_sev_probs],
        },
        "small_targets": targets_out,
        "target_count": len(targets_out),
        "knowledge_graph": {
            "entity_id": kg_id,
            "disease_entity": kg_id,
        },
        "latency": {
            "total_ms": round(result.latency_ms, 2),
            "preprocess_ms": round(result.preprocess_ms, 2),
            "classify_ms": round(result.classify_ms, 2),
            "detect_ms": round(result.detect_ms, 2),
            "severity_ms": round(result.severity_ms, 2),
        },
    }


# ---------------------------------------------------------------------------
# Quick summary helper
# ---------------------------------------------------------------------------


def result_summary(result: Dict[str, Any]) -> str:
    """Return a one-line human-readable summary."""
    p = result["prediction"]
    s = result["severity"]
    return (
        f"[{p['name_cn']}] {p['category']} — confidence={p['confidence']:.2%} "
        f"| severity={s['label_cn']} (L{s['level']}) "
        f"| targets={result['target_count']} "
        f"| latency={result['latency']['total_ms']:.1f}ms"
    )
