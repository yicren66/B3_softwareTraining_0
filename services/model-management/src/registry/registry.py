"""
Model registry — version tracking, deployment state, and A/B testing.

Stores model metadata in JSON files on disk with the following structure:
  /models/
    classifier/
      v1.0.0/
        jujube_classifier.pth
        metrics.json
      v1.1.0/
        ...
    detector/
      v1.0.0/
        ...
    severity/
      v1.0.0/
        ...
    registry.json        — master index of all models & versions

SRS-SYS02 §4.1.2 IM-04 — 模型版本管理
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from config import settings

logger = logging.getLogger(__name__)

REGISTRY_FILE = Path(settings.MODEL_REGISTRY_ROOT) / "registry.json"


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


@dataclass
class ModelMetrics:
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    inference_time_ms: float = 0.0
    model_size_mb: float = 0.0


@dataclass
class ModelVersion:
    version: str
    model_type: str  # classifier | detector | severity
    created_at: str
    artifact_path: str
    dataset_version: str = ""
    framework: str = "pytorch"
    metrics: ModelMetrics = field(default_factory=ModelMetrics)
    status: str = "registered"  # registered | staging | production | deprecated | rolled_back
    onnx_path: str = ""
    tensorrt_path: str = ""
    description: str = ""
    trained_samples: int = 0
    notes: str = ""


@dataclass
class DeploymentState:
    model_type: str
    current_version: str
    previous_version: str = ""
    gray_percent: int = 0  # 0 = full deployment, >0 = gray release percent
    deployed_at: str = ""
    deployed_by: str = ""


# ---------------------------------------------------------------------------
# Registry persistence
# ---------------------------------------------------------------------------


def _ensure_registry_dir():
    Path(settings.MODEL_REGISTRY_ROOT).mkdir(parents=True, exist_ok=True)


def _load_registry() -> dict:
    """Load the master registry from disk."""
    _ensure_registry_dir()
    if REGISTRY_FILE.exists():
        try:
            return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Corrupt registry.json, starting fresh: %s", e)
    return {"models": {}, "deployments": {}, "updated_at": ""}


def _save_registry(reg: dict) -> None:
    """Persist the master registry to disk atomically."""
    _ensure_registry_dir()
    reg["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = REGISTRY_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(REGISTRY_FILE)


# ---------------------------------------------------------------------------
# Model version CRUD
# ---------------------------------------------------------------------------


def list_models(model_type: Optional[str] = None) -> list[dict]:
    """List all registered models, optionally filtered by type."""
    reg = _load_registry()
    models = reg.get("models", {})
    result = []
    for mtype, versions in models.items():
        if model_type and mtype != model_type:
            continue
        for ver, info in versions.items():
            result.append({"model_type": mtype, "version": ver, **info})
    result.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return result


def register_model(
    model_type: str,
    version: str,
    artifact_path: str,
    dataset_version: str = "",
    framework: str = "pytorch",
    metrics: Optional[dict] = None,
    description: str = "",
    trained_samples: int = 0,
    onnx_path: str = "",
    tensorrt_path: str = "",
) -> dict:
    """Register a new model version in the registry."""
    reg = _load_registry()

    if model_type not in reg["models"]:
        reg["models"][model_type] = {}

    if version in reg["models"][model_type]:
        raise ValueError(f"Version {version} already exists for {model_type}")

    entry = ModelVersion(
        version=version,
        model_type=model_type,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        artifact_path=artifact_path,
        dataset_version=dataset_version,
        framework=framework,
        metrics=ModelMetrics(**(metrics or {})),
        description=description,
        trained_samples=trained_samples,
        onnx_path=onnx_path,
        tensorrt_path=tensorrt_path,
    )
    reg["models"][model_type][version] = asdict(entry)
    _save_registry(reg)

    logger.info("Registered %s/%s — f1=%.4f acc=%.4f",
                model_type, version,
                entry.metrics.f1_score, entry.metrics.accuracy)
    return asdict(entry)


def get_version(model_type: str, version: str) -> Optional[dict]:
    """Get a specific model version."""
    reg = _load_registry()
    return reg.get("models", {}).get(model_type, {}).get(version)


def deprecate_version(model_type: str, version: str) -> dict:
    """Mark a model version as deprecated."""
    reg = _load_registry()
    if model_type not in reg["models"] or version not in reg["models"][model_type]:
        raise ValueError(f"Model {model_type}/{version} not found")
    reg["models"][model_type][version]["status"] = "deprecated"
    _save_registry(reg)
    return reg["models"][model_type][version]


# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------


def deploy_model(
    model_type: str,
    version: str,
    gray_percent: int = 0,
    deployed_by: str = "admin",
) -> dict:
    """Deploy a model version to production (with optional gray release).

    Args:
        model_type: classifier | detector | severity
        version: Version string to deploy
        gray_percent: 0 = full production, 1-99 = gray release percentage
        deployed_by: Who triggered the deployment
    """
    reg = _load_registry()

    # Validate version exists
    ver_info = reg.get("models", {}).get(model_type, {}).get(version)
    if ver_info is None:
        raise ValueError(f"Model {model_type}/{version} not found in registry")

    # Record previous deployment
    prev = reg.get("deployments", {}).get(model_type, {})

    # Mark previous production version as rolled_back
    if prev and prev.get("current_version"):
        old_ver = prev["current_version"]
        if old_ver in reg.get("models", {}).get(model_type, {}):
            reg["models"][model_type][old_ver]["status"] = "rolled_back"

    # Update new version status
    ver_info["status"] = "production"
    if gray_percent > 0:
        ver_info["status"] = f"gray_{gray_percent}pct"

    # Update deployment state
    deployment = DeploymentState(
        model_type=model_type,
        current_version=version,
        previous_version=prev.get("current_version", ""),
        gray_percent=gray_percent,
        deployed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        deployed_by=deployed_by,
    )
    reg["deployments"][model_type] = asdict(deployment)
    _save_registry(reg)

    # Notify image-recognition service to hot-reload
    _notify_reload(model_type, version)

    logger.info("Deployed %s/%s (gray=%d%%) by %s", model_type, version, gray_percent, deployed_by)
    return asdict(deployment)


def rollback_model(model_type: str, deployed_by: str = "admin") -> dict:
    """Rollback to the previous production version."""
    reg = _load_registry()
    deployment = reg.get("deployments", {}).get(model_type)
    if not deployment or not deployment.get("previous_version"):
        raise ValueError(f"No previous version to rollback to for {model_type}")

    prev_ver = deployment["previous_version"]

    # Mark current as rolled_back
    cur_ver = deployment["current_version"]
    if cur_ver in reg.get("models", {}).get(model_type, {}):
        reg["models"][model_type][cur_ver]["status"] = "rolled_back"

    # Re-activate previous
    if prev_ver in reg.get("models", {}).get(model_type, {}):
        reg["models"][model_type][prev_ver]["status"] = "production"

    # Update deployment
    new_deployment = DeploymentState(
        model_type=model_type,
        current_version=prev_ver,
        previous_version=cur_ver,
        deployed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        deployed_by=deployed_by,
    )
    reg["deployments"][model_type] = asdict(new_deployment)
    _save_registry(reg)

    _notify_reload(model_type, prev_ver)
    logger.info("Rolled back %s → %s by %s", model_type, prev_ver, deployed_by)
    return asdict(new_deployment)


def get_deployment_status() -> dict:
    """Get current deployment status for all model types."""
    reg = _load_registry()
    return reg.get("deployments", {})


# ---------------------------------------------------------------------------
# Image Recognition service notification
# ---------------------------------------------------------------------------


def _notify_reload(model_type: str, version: str) -> None:
    """Notify the image-recognition service to hot-reload the model."""
    try:
        import httpx
        url = f"{settings.IMAGE_RECOGNITION_URL}/admin/reload"
        httpx.post(url, json={"model_type": model_type, "version": version}, timeout=5)
    except Exception as e:
        logger.warning("Failed to notify image-recognition service: %s", e)


# ---------------------------------------------------------------------------
# A/B test helper
# ---------------------------------------------------------------------------


def start_ab_test(model_type: str, version_a: str, version_b: str, traffic_split: float = 0.5) -> dict:
    """Configure A/B testing between two versions.

    Args:
        model_type: Model type to test.
        version_a: Control version (current production).
        version_b: Treatment version.
        traffic_split: Fraction of traffic to version_b (0-1).
    """
    reg = _load_registry()

    for ver in [version_a, version_b]:
        if ver not in reg.get("models", {}).get(model_type, {}):
            raise ValueError(f"Version {ver} not found for {model_type}")

    gray_pct = int(traffic_split * 100)

    # Deploy version_a with traffic routing
    reg["models"][model_type][version_a]["status"] = "ab_control"
    reg["models"][model_type][version_b]["status"] = f"ab_treatment_{gray_pct}pct"

    deployment = DeploymentState(
        model_type=model_type,
        current_version=f"{version_a}|{version_b}",
        gray_percent=gray_pct,
        deployed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        deployed_by="ab_test",
    )
    reg["deployments"][f"{model_type}_ab"] = asdict(deployment)
    _save_registry(reg)

    return asdict(deployment)
