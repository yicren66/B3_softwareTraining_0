# Image Recognition Service

病害/虫害图像识别与严重程度评估服务 —— Jujube Platform 核心推理微服务。

## 架构概览

```
                    ┌─────────────────────────────────┐
                    │         gRPC :9001               │
                    │   ClassifyImage / BatchClassify  │
                    │         HealthCheck              │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │       InferenceEngine            │
                    │                                  │
                    │  1. Preprocess (transforms)      │
                    │  2. Classify  (JujubeClassifier) │
                    │  3. Detect   (SmallTarget FPN)   │
                    │  4. Severity (SeverityClassifier)│
                    │  5. Postprocess → RecognitionData│
                    └──────────────┬──────────────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            │                      │                      │
     ┌──────▼──────┐      ┌───────▼───────┐      ┌───────▼───────┐
     │  PyTorch    │      │  ONNX Runtime │      │   TensorRT    │
     │  (.pth)     │      │  (.onnx)      │      │   (.trt)      │
     └─────────────┘      └───────────────┘      └───────────────┘
```

## 目录结构

```
services/image-recognition/
├── Dockerfile
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── config.py                       # 集中配置（IR_* 环境变量）
    ├── main.py                         # FastAPI + gRPC 入口
    ├── models/
    │   ├── __init__.py
    │   └── model_loader.py             # 模型加载/预热/热重载
    ├── preprocessing/
    │   ├── __init__.py
    │   └── transforms.py               # 图像预处理管线
    └── inference/
        ├── __init__.py
        ├── engine.py                   # 推理引擎（多后端）
        └── postprocess.py              # 结果后处理与标签映射
```

## 快速开始

### 开发环境

```bash
# 安装依赖
cd services/image-recognition
pip install -r requirements.txt

# 以开发模式启动（不需要 GPU）
IR_USE_GPU=false IR_INFERENCE_BACKEND=pytorch \
  python -m services.image-recognition.src.main
```

服务启动后：
- **HTTP 健康检查**: http://localhost:8001/health
- **Prometheus 指标**: http://localhost:8001/metrics
- **就绪探测**: http://localhost:8001/ready
- **存活探测**: http://localhost:8001/live
- **gRPC**: localhost:9001

### Docker 部署

```bash
# 构建镜像
docker build -t jujube/image-recognition:latest \
  -f services/image-recognition/Dockerfile .

# 运行（需要 NVIDIA GPU）
docker run --gpus all \
  -p 9001:9001 -p 8001:8001 \
  -v /path/to/models:/app/models_registry:ro \
  jujube/image-recognition:latest
```

### Docker Compose（全平台）

```bash
make dev-up
```

## 配置

所有配置通过 `IR_` 前缀的环境变量覆盖，参见 `src/config.py`。

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `IR_INFERENCE_BACKEND` | `onnxruntime` | 推理后端：`onnxruntime` / `tensorrt` / `pytorch` |
| `IR_USE_GPU` | `true` | 是否使用 GPU |
| `IR_GPU_ID` | `0` | GPU 设备编号 |
| `IR_MODEL_PATH` | `/models/classifier/jujube_classifier.pth` | 分类模型路径 |
| `IR_ONNX_MODEL_PATH` | `/models/classifier/jujube_classifier.onnx` | ONNX 模型路径 |
| `IR_TENSORRT_ENGINE_PATH` | `/models/classifier/jujube_classifier.trt` | TensorRT 引擎路径 |
| `IR_DETECTOR_MODEL_PATH` | `/models/detector/small_target_fpn.pth` | 小目标检测模型 |
| `IR_CONFIDENCE_THRESHOLD` | `0.5` | 置信度阈值 |
| `IR_BATCH_SIZE` | `8` | 默认批次大小 |
| `IR_MAX_BATCH_SIZE` | `32` | 最大批次大小 |
| `IR_HTTP_PORT` | `8001` | HTTP 指标/健康检查端口 |
| `IR_GRPC_PORT` | `9001` | gRPC 服务端口 |

## API

### gRPC 接口

服务实现 `jujube.recognition.ImageRecognition`（定义见 `libs/proto/recognition.proto`）。

#### ClassifyImage
单张图像分类，返回病害名称、类别、置信度、严重程度、小目标检测结果。

```protobuf
rpc ClassifyImage(ClassifyRequest) returns (ClassifyResponse);
```

**请求示例**:
```json
{
  "image_data": "<raw bytes>",
  "image_format": "JPEG",
  "location": {"latitude": 37.5, "longitude": 112.7},
  "include_kg_link": true
}
```

**响应示例**:
```json
{
  "disease_name": "枣疯病",
  "category": "phytoplasma",
  "confidence": 0.93,
  "severity": "中度",
  "severity_confidence": 0.78,
  "affected_part": "branch,leaf",
  "symptoms": "Proliferation of small branches, yellowing...",
  "small_target_detected": true,
  "kg_entity_id": "jujube.disease.witches_broom",
  "latency_ms": 42
}
```

#### BatchClassify
批量异步分类，返回任务 ID 用于轮询。

```protobuf
rpc BatchClassify(BatchClassifyRequest) returns (BatchClassifyResponse);
```

### REST 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查（含模型状态） |
| GET | `/ready` | Kubernetes 就绪探测 |
| GET | `/live` | Kubernetes 存活探测 |
| GET | `/metrics` | Prometheus 指标 |

## 支持的病害类别（20 类）

| ID | 中文名 | 类别 | 受害部位 |
|----|--------|------|----------|
| 0 | 枣锈病 | 真菌 | 叶片 |
| 1 | 枣疯病 | 植原体 | 枝条、叶片 |
| 2 | 枣炭疽病 | 真菌 | 果实、叶片 |
| 3 | 枣缩果病 | 真菌 | 果实 |
| 4 | 枣黑斑病 | 真菌 | 叶片、果实 |
| 5 | 枣裂果病 | 生理性 | 果实 |
| 6 | 枣褐斑病 | 真菌 | 叶片 |
| 7 | 枣白粉病 | 真菌 | 叶片、果实 |
| 8 | 枣根腐病 | 真菌 | 根部 |
| 9 | 枣干腐病 | 真菌 | 树干、枝条 |
| 10 | 枣叶斑病 | 真菌 | 叶片 |
| 11 | 细菌性穿孔病 | 细菌 | 叶片 |
| 12 | 枣花叶病 | 病毒 | 叶片 |
| 13 | 枣红蜘蛛 | 虫害 | 叶片 |
| 14 | 枣食心虫 | 虫害 | 果实 |
| 15 | 枣尺蠖 | 虫害 | 叶片 |
| 16 | 枣黏虫 | 虫害 | 叶片 |
| 17 | 枣龟蜡蚧 | 虫害 | 枝条 |
| 18 | 日灼病 | 生理性 | 果实、树皮 |
| 19 | 缺铁黄化 | 营养 | 叶片 |

## 严重程度

| 级别 | 中文 | 说明 |
|------|------|------|
| 0 | 健康 | 无明显病害症状 |
| 1 | 轻度 | 轻微症状，无需立即处理 |
| 2 | 中度 | 明显症状，建议防治 |
| 3 | 重度 | 严重感染，需紧急处理 |

## 模型热重载

服务启动后会监控 `/app/models_registry` 目录（配置项 `IR_MODEL_WATCH_INTERVAL`，默认 30 秒），
当检测到模型文件更新时自动重新加载并预热新模型，无需重启服务。

## Prometheus 指标

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `ir_requests_total` | Counter | 总请求数（按方法和状态分标签） |
| `ir_request_latency_seconds` | Histogram | 端到端延迟分布 |
| `ir_model_loaded` | Gauge | 各子模型加载状态 |
| `ir_gpu_memory_allocated_mb` | Gauge | GPU 显存已分配量 |
| `ir_gpu_memory_reserved_mb` | Gauge | GPU 显存预留量 |
| `ir_inference_errors_total` | Counter | 推理错误总数 |

## 测试

```bash
# 运行全部集成测试
pytest tests/ -v

# 仅运行图像识别服务测试
pytest tests/integration/ -v -k "test_config or test_preprocessing or test_model_loader or test_postprocess or test_inference"

# 带覆盖率
pytest tests/ --cov=services/image-recognition/src --cov-report=html
```
