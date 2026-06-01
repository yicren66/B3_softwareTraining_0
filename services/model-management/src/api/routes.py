"""
Model Management Service — REST API Routes.

Implements endpoints defined in SRS-SYS02 §6.1.3 and libs/proto/model_mgmt.proto:
  GET    /api/v1/model/versions
  GET    /api/v1/model/versions/{model_type}/{version}
  POST   /api/v1/model/register
  POST   /api/v1/model/deploy
  POST   /api/v1/model/rollback
  POST   /api/v1/model/train           (trigger incremental training)
  GET    /api/v1/model/deployments
  POST   /api/v1/model/ab-test
  GET    /health
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from registry import (
    list_models,
    register_model,
    get_version,
    deprecate_version,
    deploy_model,
    rollback_model,
    get_deployment_status,
    start_ab_test,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/model", tags=["Model Management"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class RegisterModelRequest(BaseModel):
    model_type: str = Field(..., description="classifier | detector | severity")
    version: str = Field(..., description="Semantic version, e.g. v1.2.0")
    artifact_path: str = Field(..., description="Path to model artifact (.pth/.onnx)")
    dataset_version: str = Field(default="")
    framework: str = Field(default="pytorch")
    metrics: Optional[dict] = None
    description: str = Field(default="")
    trained_samples: int = Field(default=0)
    onnx_path: str = Field(default="")
    tensorrt_path: str = Field(default="")


class DeployModelRequest(BaseModel):
    model_type: str
    version: str
    gray_percent: int = Field(default=0, ge=0, le=100)
    deployed_by: str = Field(default="admin")


class RollbackRequest(BaseModel):
    model_type: str
    deployed_by: str = Field(default="admin")


class TrainRequest(BaseModel):
    model_type: str = Field(default="classifier")
    dataset_version: str = Field(default="latest")
    base_version: str = Field(default="", description="Base model for incremental learning")
    config_overrides: dict = Field(default_factory=dict)


class ABTestRequest(BaseModel):
    model_type: str
    version_a: str  # control
    version_b: str  # treatment
    traffic_split: float = Field(default=0.5, ge=0.0, le=1.0)


class APIResponse(BaseModel):
    code: int = 0
    message: str = "success"
    data: Optional[dict | list] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/versions", response_model=APIResponse, summary="查看模型版本列表")
async def versions(model_type: Optional[str] = Query(default=None)):
    data = list_models(model_type=model_type)
    return APIResponse(data=data)


@router.get("/versions/{model_type}/{version}", response_model=APIResponse, summary="查看指定版本详情")
async def version_detail(model_type: str, version: str):
    data = get_version(model_type, version)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Model {model_type}/{version} not found")
    return APIResponse(data=data)


@router.post("/register", response_model=APIResponse, summary="注册新模型版本")
async def register(req: RegisterModelRequest):
    t0 = time.perf_counter()
    try:
        data = register_model(
            model_type=req.model_type,
            version=req.version,
            artifact_path=req.artifact_path,
            dataset_version=req.dataset_version,
            framework=req.framework,
            metrics=req.metrics,
            description=req.description,
            trained_samples=req.trained_samples,
            onnx_path=req.onnx_path,
            tensorrt_path=req.tensorrt_path,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("POST /register %s/%s — %.1fms", req.model_type, req.version, elapsed)
        return APIResponse(data=data)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/deploy", response_model=APIResponse, summary="部署模型到生产（含灰度发布）")
async def deploy(req: DeployModelRequest):
    t0 = time.perf_counter()
    try:
        data = deploy_model(
            model_type=req.model_type,
            version=req.version,
            gray_percent=req.gray_percent,
            deployed_by=req.deployed_by,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("POST /deploy %s/%s gray=%d%% — %.1fms",
                    req.model_type, req.version, req.gray_percent, elapsed)
        return APIResponse(data=data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/rollback", response_model=APIResponse, summary="回滚至上一版本")
async def rollback(req: RollbackRequest):
    t0 = time.perf_counter()
    try:
        data = rollback_model(model_type=req.model_type, deployed_by=req.deployed_by)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("POST /rollback %s — %.1fms", req.model_type, elapsed)
        return APIResponse(data=data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/train", response_model=APIResponse, summary="触发增量模型训练任务")
async def trigger_train(req: TrainRequest):
    """Trigger an incremental training job. In production this enqueues to a
    task queue (Celery / Argo); here we return an acknowledgement."""
    task_id = f"train-{req.model_type}-{int(time.time())}"
    logger.info(
        "POST /train type=%s dataset=%s base=%s → task_id=%s",
        req.model_type, req.dataset_version, req.base_version, task_id,
    )
    return APIResponse(data={
        "task_id": task_id,
        "status": "queued",
        "model_type": req.model_type,
        "message": f"Training task {task_id} has been queued. Monitor /api/v1/model/versions for the new version.",
    })


@router.get("/deployments", response_model=APIResponse, summary="查看当前部署状态")
async def deployments():
    data = get_deployment_status()
    return APIResponse(data=data)


@router.post("/ab-test", response_model=APIResponse, summary="启动A/B测试")
async def ab_test(req: ABTestRequest):
    try:
        data = start_ab_test(
            model_type=req.model_type,
            version_a=req.version_a,
            version_b=req.version_b,
            traffic_split=req.traffic_split,
        )
        return APIResponse(data=data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/versions/{model_type}/{version}", response_model=APIResponse, summary="废弃指定版本")
async def deprecate(model_type: str, version: str):
    try:
        data = deprecate_version(model_type, version)
        return APIResponse(data=data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/health", summary="服务健康检查")
async def health():
    return {"status": "healthy", "service": "model-management"}
