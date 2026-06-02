# Jujube Platform — 枣林病虫害智能识别与知识图谱平台

本项目是面向枣林场景的多服务 AI 平台，覆盖 **病虫害图像识别、知识图谱查询、智能推理与防治推荐、模型版本管理、日志统计与运维部署** 等能力。项目目标与《需求规格说明书-子系统一病虫害识别和知识图谱模型基础软件-修改.docx》保持一致，核心识别对象为枣树 **「7病8虫」15类病虫害**。

> 当前仓库已经具备较完整的代码骨架、数据种子、训练脚本、服务实现、Docker/Kubernetes 部署配置和测试用例；但要“一键完整运行”仍存在若干工程级缺口，详见本文末尾 [当前还缺少什么](#当前还缺少什么整个工程才能完整运行起来)。

---

## 1. 系统总体架构

```text
                              ┌─────────────────────────────┐
                              │      Nginx / Ingress         │
                              │  TLS终止、反向代理、统一入口 │
                              └─────────────┬───────────────┘
                                            │
                              ┌─────────────▼───────────────┐
                              │        API Gateway           │
                              │ Go/Gin, JWT, RBAC, Proxy     │
                              │          :8000               │
                              └──┬──────┬──────┬──────┬─────┘
                                 │      │      │      │
        ┌────────────────────────┤      │      │      ├──────────────────┐
        │                        │      │      │                        │
  ┌─────▼──────┐  ┌──────▼─────┐ │ ┌────▼────┐ │ ┌────────▼──────┐ ┌───▼──────────┐
  │  Image     │  │ Knowledge  │ │ │Reasoning│ │ │   Model       │ │ Log & Stats  │
  │Recognition │  │  Graph     │ │ │ Engine  │ │ │ Management    │ │   Service    │
  │HTTP :8001  │  │HTTP :8002  │ │ │HTTP:8003│ │ │HTTP :8004     │ │HTTP/gRPC:8005│
  │gRPC :9001  │  │gRPC :9002* │ │ │gRPC:9003*│ │ │gRPC :9004*    │ │gRPC :9005    │
  └─────┬──────┘  └──────┬─────┘ │ └────┬────┘ │ └───────┬───────┘ └───┬──────────┘
        │                │       │      │      │         │              │
        │         ┌──────▼───┐   │      │      │         │    ┌─────────▼──────┐
        │         │  Neo4j 5 │   │      │      │         │    │ PostgreSQL 14  │
        │         │  :7687   │   │      │      │         │    │     :5432      │
        │         └──────────┘   │      │      │         │    └────────────────┘
        │                        │      │      │         │
        └────────────────────────┴──────┴──────┴─────────┘
                           ┌───────▼──────┐
                           │    Redis 7   │
                           │    :6379     │
                           └──────────────┘
```

说明：

- `*` 标记的部分目前主要是接口契约或预留端口，部分 Python 服务当前实现以 HTTP/FastAPI 为主，gRPC 服务端尚未完整接入生成代码。
- API Gateway 负责统一入口、JWT 鉴权、角色权限、限流和服务代理。
- Image Recognition Service 负责图像预处理、分类、小目标检测、严重程度判断和结果结构化输出。
- Knowledge Graph Service 负责 Neo4j 图谱查询、语义检索、防治推荐和风险预测。
- Reasoning Engine 负责个性化病情解释、防治方案生成、角色差异化输出和多轮问答。
- Model Management Service 负责模型注册、版本管理、部署、回滚、灰度发布和 A/B 测试。
- Log & Statistics Service 负责识别日志、专家反馈、统计分析和仪表盘数据。

---

## 2. 核心业务分类：「7病8虫」15类

模型训练配置、识别后处理、知识图谱实体 ID 当前均按以下 15 类对齐：

| 类别ID | 名称 | 类别 | KG实体ID |
|---:|---|---|---|
| 0 | 枣炭疽病 | 病害 | `KG-ENT-001` |
| 1 | 枣疯病 | 病害 | `KG-ENT-002` |
| 2 | 枣树锈病 | 病害 | `KG-ENT-003` |
| 3 | 枣缩果病 | 病害 | `KG-ENT-004` |
| 4 | 枣果腐病 | 病害 | `KG-ENT-005` |
| 5 | 枣褐斑病 | 病害 | `KG-ENT-006` |
| 6 | 枣叶黑斑病 | 病害 | `KG-ENT-007` |
| 7 | 枣芽象甲 | 虫害 | `KG-ENT-008` |
| 8 | 枣瘿蚊 | 虫害 | `KG-ENT-009` |
| 9 | 桃小食心虫 | 虫害 | `KG-ENT-010` |
| 10 | 绿盲蝽 | 虫害 | `KG-ENT-011` |
| 11 | 枣尺蠖 | 虫害 | `KG-ENT-012` |
| 12 | 枣镰翅小卷蛾 | 虫害 | `KG-ENT-013` |
| 13 | 枣红蜘蛛 | 虫害 | `KG-ENT-014` |
| 14 | 枣龟蜡蚧 | 虫害 | `KG-ENT-015` |

---

## 3. 快速开始

### 3.1 环境要求

| 工具 | 用途 | 建议版本 |
|---|---|---|
| Docker | 构建与运行容器 | Docker Engine + Compose v2 |
| Python | 本地运行 Python 服务和测试 | 3.10+ / 3.11+ |
| Go | 构建 API Gateway 与 Log Statistics | 1.22+ |
| Neo4j | 知识图谱数据库 | 5.x |
| PostgreSQL | 日志、模型元数据等结构化数据 | 14+ |
| Redis | 缓存、队列、状态辅助 | 7.x |
| NVIDIA Container Toolkit | GPU 推理，可选 | 有 GPU 时需要 |
| buf | 生成 protobuf/gRPC 代码，可选 | 需要补齐配置后使用 |

### 3.2 配置环境变量

```bash
cp .env.example .env
# 修改 .env 中的密码、JWT_SECRET_KEY、端口等配置
```

重要变量：

| 变量 | 说明 |
|---|---|
| `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | PostgreSQL 连接信息 |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | Neo4j 连接信息 |
| `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD` | Redis 连接信息 |
| `JWT_SECRET_KEY` | API Gateway JWT 签名密钥，生产环境必须改成随机长字符串 |
| `API_GATEWAY_PORT` | API Gateway 对外端口，默认 8000 |
| `IR_USE_GPU` | Image Recognition 是否使用 GPU，普通开发机建议设为 `false` |
| `IR_INFERENCE_BACKEND` | 推理后端：`pytorch` / `onnxruntime` / `tensorrt` |

### 3.3 理想启动方式

```bash
make dev-up
curl http://localhost:8000/health
make dev-logs
```

或直接：

```bash
docker compose up -d --build
```

> 注意：当前仓库还存在若干运行阻塞点，直接执行 `make dev-up` 很可能失败。请先阅读 [当前还缺少什么](#当前还缺少什么整个工程才能完整运行起来)。

### 3.4 当前已验证的测试

已验证通过的本地测试子集：

```bash
python -m pytest \
  tests/integration/test_config.py \
  tests/integration/test_preprocessing.py \
  tests/integration/test_postprocess.py \
  tests/integration/test_api_contracts.py \
  -v
```

当前结果：

```text
83 passed
```

---

## 4. 顶层目录与文件说明

```text
.
├── .env.example              # 环境变量模板，复制为 .env 后用于 docker-compose 和服务运行
├── .gitignore                # Git 忽略规则：模型权重、数据集、缓存、日志、虚拟环境等
├── Makefile                  # 常用命令入口：dev-up、dev-down、test、lint、deploy 等
├── README.md                 # 项目说明文档（当前文件）
├── docker-compose.yml        # 本地开发环境多容器编排配置
├── pyproject.toml            # Python 工具配置：pytest、mypy、black、isort、ruff、coverage 等
├── deploy/                   # 部署相关配置：Docker Nginx、Kubernetes、数据库初始化脚本
├── docs/                     # 项目文档预留目录：API、架构、数据库文档等
├── kg-construction/          # 知识图谱构建资产：本体、抽取、向量、种子数据
├── libs/                     # 跨服务共享库：Python公共库、protobuf接口定义
├── ml-models/                # 机器学习训练、评估、导出、蒸馏、增量学习相关代码
├── scripts/                  # 项目脚本预留目录
├── services/                 # 各微服务源码
├── tests/                    # 测试用例：集成、端到端、性能测试
└── txt/                      # 原始需求、任务书、提取文本等项目材料
```

---

## 5. `deploy/` 部署目录说明

```text
deploy/
├── docker/
│   ├── nginx/
│   │   └── nginx.conf
│   └── prometheus/
├── kubernetes/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secrets.yaml
│   ├── secrets.yaml.example
│   ├── ingress.yaml
│   ├── api-gateway/
│   ├── image-recognition/
│   ├── knowledge-graph/
│   ├── reasoning-engine/
│   ├── model-management/
│   ├── log-statistics/
│   ├── neo4j/
│   ├── postgresql/
│   └── redis/
└── scripts/
    ├── init_neo4j.sh
    └── init_postgres.sh
```

| 路径/文件 | 作用 |
|---|---|
| `deploy/docker/nginx/nginx.conf` | 本地或生产风格 Nginx 反向代理配置，用于转发到 API Gateway |
| `deploy/docker/prometheus/` | Prometheus 配置预留目录，目前主要作为监控扩展位置 |
| `deploy/kubernetes/namespace.yaml` | 创建 Kubernetes 命名空间 |
| `deploy/kubernetes/configmap.yaml` | 集群中的非敏感配置，如数据库地址、服务地址、推理端口等 |
| `deploy/kubernetes/secrets.yaml.example` | 敏感配置模板，如数据库密码、JWT密钥等；生产环境应复制并替换真实值 |
| `deploy/kubernetes/secrets.yaml` | 当前仓库中的 Secret 配置文件；生产环境应谨慎处理，不建议提交真实密钥 |
| `deploy/kubernetes/ingress.yaml` | Kubernetes Ingress 配置，负责外部流量入口和 TLS |
| `deploy/kubernetes/api-gateway/` | API Gateway 的 Deployment、Service 等资源 |
| `deploy/kubernetes/image-recognition/` | 图像识别服务的 Deployment、Service、HPA、PVC、ServiceMonitor 等资源 |
| `deploy/kubernetes/knowledge-graph/` | 知识图谱服务的 Deployment、Service 资源 |
| `deploy/kubernetes/reasoning-engine/` | 推理引擎服务的 Deployment、Service 资源 |
| `deploy/kubernetes/model-management/` | 模型管理服务的 Deployment、Service 资源 |
| `deploy/kubernetes/log-statistics/` | 日志统计服务的 Deployment、Service 资源 |
| `deploy/kubernetes/neo4j/` | Neo4j 的 StatefulSet/Deployment/Service 等资源 |
| `deploy/kubernetes/postgresql/` | PostgreSQL 的 StatefulSet/Deployment/Service 等资源 |
| `deploy/kubernetes/redis/` | Redis 的 Kubernetes 配置预留目录 |
| `deploy/scripts/init_postgres.sh` | PostgreSQL 初始化脚本：创建扩展、`log_statistics` schema、识别日志表、模型注册表 |
| `deploy/scripts/init_neo4j.sh` | Neo4j 初始化脚本：计划执行知识图谱 Cypher 迁移文件；当前 docker-compose 中尚未真正触发执行 |

---

## 6. `kg-construction/` 知识图谱构建目录说明

```text
kg-construction/
├── embeddings/       # 图谱实体、文本语料向量化结果预留目录
├── extraction/       # 从文档/文本中抽取实体关系的管线预留目录
├── ontology/         # 知识图谱本体定义预留目录
└── seed_data/        # 已整理的知识图谱种子数据 CSV
```

`seed_data/` 文件说明：

| 文件 | 作用 |
|---|---|
| `pests_diseases.csv` | 病虫害实体数据，覆盖核心「7病8虫」及扩展病虫害，包含中文名、英文名、类别、学名、症状、严重程度等 |
| `prevention_methods.csv` | 防治方法实体数据，包括农业防治、化学防治、物理防治、生物防治等 |
| `pesticides.csv` | 农药/生防制剂实体数据 |
| `growth_cycles.csv` | 枣树物候期实体数据：萌芽期、展叶期、花期、幼果期、膨大期、成熟期、休眠期 |
| `symptoms.csv` | 症状特征实体数据 |
| `climate_conditions.csv` | 气候条件实体数据，如高温、高湿、干旱、多雨等 |
| `pest_symptom_relations.csv` | 病虫害与症状之间的关系 |
| `pest_pesticide_relations.csv` | 病虫害与农药/制剂之间的关系 |
| `pest_prevention_relations.csv` | 病虫害与防治方法之间的关系 |
| `pest_climate_relations.csv` | 病虫害与气候条件之间的关系 |
| `pest_growth_cycle_relations.csv` | 病虫害与物候期之间的关系 |

---

## 7. `libs/` 共享库与接口定义

```text
libs/
├── common-py/
│   ├── setup.py
│   └── common/
│       ├── __init__.py
│       ├── auth.py
│       ├── errors.py
│       ├── logging.py
│       ├── metrics.py
│       └── db/
│           ├── __init__.py
│           ├── postgres.py
│           └── redis.py
└── proto/
    ├── common.proto
    ├── recognition.proto
    ├── kg.proto
    ├── reasoning.proto
    ├── model_mgmt.proto
    └── log_statistics.proto
```

### 7.1 `libs/common-py/`

| 文件 | 作用 |
|---|---|
| `setup.py` | 将公共 Python 库安装为可编辑包，供各 Python 微服务复用 |
| `common/auth.py` | JWT/认证相关公共逻辑 |
| `common/errors.py` | 统一异常、错误码或错误响应封装 |
| `common/logging.py` | 日志格式、日志初始化等公共工具 |
| `common/metrics.py` | Prometheus 指标封装或通用指标工具 |
| `common/db/postgres.py` | PostgreSQL 连接与访问公共工具 |
| `common/db/redis.py` | Redis 连接与访问公共工具 |

### 7.2 `libs/proto/`

| 文件 | 作用 |
|---|---|
| `common.proto` | 通用消息结构，如地理位置、分页、通用响应等 |
| `recognition.proto` | 图像识别服务 gRPC 契约：单图识别、批量识别、健康检查 |
| `kg.proto` | 知识图谱服务 gRPC 契约：实体查询、语义搜索、防治推荐、风险预测 |
| `reasoning.proto` | 推理引擎 gRPC 契约：个性化防治计划、智能问答 |
| `model_mgmt.proto` | 模型管理 gRPC 契约：训练、版本、部署、回滚等 |
| `log_statistics.proto` | 日志统计服务 gRPC 契约：识别日志、反馈、统计数据、健康检查 |

> 当前问题：Makefile 中 `PROTO_DIR := proto`，但实际 proto 文件位于 `libs/proto/`；同时仓库没有 `buf.yaml` / `buf.gen.yaml`。因此 `make proto-gen` 当前不能直接工作。

---

## 8. `ml-models/` 模型训练与算法目录

```text
ml-models/
├── classification/
│   ├── config.yaml
│   ├── train.py
│   ├── datasets/
│   ├── eval/
│   ├── losses/
│   └── models/
├── distillation/
│   └── student_models/
├── export/
├── incremental/
└── small-target/
    └── models/
```

### 8.1 `classification/`

| 文件/目录 | 作用 |
|---|---|
| `config.yaml` | 分类模型训练配置：模型结构、输入尺寸、训练参数、增强策略、早停、checkpoint、蒸馏、类别名等；当前已对齐「7病8虫」15类 |
| `train.py` | 主训练脚本，支持 YAML 配置、DDP、多 GPU、AMP 混合精度、早停、top-k checkpoint、TensorBoard、ONNX 导出 |
| `datasets/jujube_dataset.py` | PyTorch Dataset：读取 CSV/JSON 标注、图像、类别、严重程度、bbox，支持训练/验证/测试划分 |
| `datasets/augmentations.py` | Albumentations 数据增强：随机裁剪、旋转、亮度对比度、噪声、小目标增强等 |
| `losses/focal_loss.py` | 多分类 Focal Loss，用于类别不均衡场景 |
| `losses/distillation_loss.py` | 知识蒸馏损失，支持软标签、特征蒸馏、注意力迁移等 |
| `models/jujube_classifier.py` | 分类模型定义，基于 `timm` backbone（默认 EfficientNet-B4）+ 分类头 |
| `eval/evaluate.py` | 模型评估脚本，计算准确率、精确率、召回率、F1 等指标 |

### 8.2 其他模型目录

| 目录 | 作用 |
|---|---|
| `distillation/student_models/` | 轻量化学生模型预留目录，用于模型蒸馏和移动端部署 |
| `export/` | 模型导出预留目录，如 ONNX、TensorRT、TFLite 等 |
| `incremental/` | 增量学习预留目录，用于模型持续迭代、专家纠错数据回流训练 |
| `small-target/models/` | 小目标检测模型预留目录，用于红蜘蛛卵粒、锈斑、产卵痕等细微目标检测 |

---

## 9. `services/` 微服务目录说明

```text
services/
├── api-gateway/
├── image-recognition/
├── knowledge-graph/
├── reasoning-engine/
├── model-management/
└── log-statistics/
```

### 9.1 `services/api-gateway/` — API 网关

技术栈：Go 1.22 + Gin + JWT。

```text
services/api-gateway/
├── Dockerfile
├── go.mod
├── cmd/server/main.go
├── internal/
│   ├── auth/jwt.go
│   ├── config/config.go
│   ├── handler/handler.go
│   ├── middleware/middleware.go
│   └── proxy/proxy.go
└── src/                # 早期/预留目录，目前核心实现位于 internal/ 与 cmd/
```

| 文件 | 作用 |
|---|---|
| `go.mod` | API Gateway 独立 Go module 依赖定义 |
| `Dockerfile` | 多阶段构建 Go 二进制并打包为 Alpine 运行镜像 |
| `cmd/server/main.go` | 程序入口：加载配置、初始化 Gin、注册路由、启动 HTTP 服务、优雅关闭 |
| `internal/config/config.go` | 环境变量配置读取：服务地址、JWT、限流、端口等 |
| `internal/auth/jwt.go` | JWT 生成、校验、Bearer token 提取、角色层级判断 |
| `internal/middleware/middleware.go` | CORS、Request ID、日志、JWT鉴权、可选鉴权、RBAC、限流中间件 |
| `internal/proxy/proxy.go` | 反向代理实现，将网关请求转发到后端微服务，并透传用户身份和请求ID |
| `internal/handler/handler.go` | 路由注册：认证、识别、知识图谱、推理、模型管理、统计等 API |

当前网关设计的主要路由：

| 路由 | 后端服务 | 权限 |
|---|---|---|
| `POST /api/v1/auth/login` | API Gateway 内部 | 公开 |
| `POST /api/v1/auth/refresh` | API Gateway 内部 | 公开 |
| `/api/v1/recognition/*` | Image Recognition | 登录用户 |
| `/api/v1/kg/*` | Knowledge Graph | 可选登录，访客可查 |
| `/api/v1/reasoning/*` | Reasoning Engine | 登录用户 |
| `/api/v1/model/*` | Model Management | 植保专家及以上 |
| `/api/v1/stats/*` | Log Statistics | 工作管理员及以上 |

### 9.2 `services/image-recognition/` — 图像识别服务

技术栈：Python + FastAPI + PyTorch/ONNXRuntime/TensorRT + Prometheus。

```text
services/image-recognition/
├── Dockerfile
├── README.md
├── requirements.txt
├── models_registry/          # 模型权重挂载/预留目录
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── inference/
│   │   ├── engine.py
│   │   └── postprocess.py
│   ├── models/
│   │   └── model_loader.py
│   └── preprocessing/
│       └── transforms.py
└── tests/
```

| 文件 | 作用 |
|---|---|
| `requirements.txt` | 图像识别服务依赖：torch、torchvision、onnxruntime、Pillow、FastAPI、gRPC、Prometheus等 |
| `Dockerfile` | GPU 运行镜像，安装 Python 依赖，复制服务代码和公共库 |
| `src/config.py` | 服务配置：模型路径、输入尺寸、批量大小、推理后端、GPU、HTTP/gRPC端口、日志等 |
| `src/main.py` | FastAPI 入口：健康检查、metrics、readiness/liveness、加载模型、启动 gRPC 预留服务 |
| `src/inference/engine.py` | 端到端推理流水线：预处理 → 分类 → 小目标检测 → 严重程度评估 → 结果聚合 |
| `src/inference/postprocess.py` | 识别结果后处理：15类标签映射、KG实体映射、置信度校准、严重程度阈值、API格式化 |
| `src/models/model_loader.py` | 模型加载管理：PyTorch/ONNX/TensorRT加载、stub模型、热更新、GPU内存状态 |
| `src/preprocessing/transforms.py` | 图像校验、打开、resize、归一化、batch拼接等预处理函数 |

### 9.3 `services/knowledge-graph/` — 知识图谱服务

技术栈：Python + FastAPI + Neo4j + jieba + sentence-transformers。

```text
services/knowledge-graph/
├── Dockerfile
├── requirements.txt
├── src/
│   ├── config.py
│   ├── main.py
│   ├── api/
│   │   ├── schemas.py
│   │   └── routes.py
│   ├── neo4j/
│   │   ├── driver.py
│   │   └── migrations/
│   ├── search/
│   │   └── semantic_search.py
│   ├── recommendation/
│   │   └── engine.py
│   └── risk/
│       └── predictor.py
└── tests/
```

| 文件/目录 | 作用 |
|---|---|
| `requirements.txt` | Neo4j驱动、FastAPI、gRPC、sentence-transformers、jieba、Pydantic、Prometheus等依赖 |
| `Dockerfile` | Python 运行镜像，安装依赖，复制知识图谱服务代码 |
| `src/config.py` | Neo4j连接、语义模型、搜索阈值、风险预测阈值、端口、日志配置 |
| `src/main.py` | FastAPI 应用入口，初始化 Neo4j driver，挂载 API 路由 |
| `src/api/schemas.py` | Pydantic 模型：实体、关系、搜索、推荐、风险预测、健康检查等响应结构 |
| `src/api/routes.py` | REST API：实体查询、语义搜索、防治推荐、问答、风险预测、健康检查 |
| `src/neo4j/driver.py` | Neo4j 连接池、健康检查、实体查询、全文检索等基础访问方法 |
| `src/neo4j/migrations/001_initial_ontology.cypher` | 初始本体约束、唯一性约束、全文索引 |
| `src/neo4j/migrations/002_constraints_indexes.cypher` | 额外索引、组合索引和查询优化索引 |
| `src/neo4j/migrations/003_seed_data.cypher` | 知识图谱种子数据，目标覆盖≥30种病虫害、≥2000三元组 |
| `src/search/semantic_search.py` | 语义搜索与意图识别：关键词抽取、jieba分词、病虫害实体映射、推荐意图判断 |
| `src/recommendation/engine.py` | 防治推荐引擎：农业/化学/物理/生物防治分类、综合策略生成 |
| `src/risk/predictor.py` | 风险预测：结合物候期、气温、湿度、降水等生成风险等级与预警 |

### 9.4 `services/reasoning-engine/` — 推理与问答服务

技术栈：Python + FastAPI + HTTP 调用 KG 服务。

```text
services/reasoning-engine/
├── Dockerfile
├── requirements.txt
├── src/
│   ├── config.py
│   ├── main.py
│   ├── api/routes.py
│   └── engine/
│       ├── prevention_planner.py
│       └── qa_engine.py
└── tests/
```

| 文件 | 作用 |
|---|---|
| `requirements.txt` | FastAPI、httpx、gRPC、sentence-transformers、jieba、Pydantic等依赖 |
| `Dockerfile` | Python 运行镜像，复制推理服务代码 |
| `src/config.py` | KG服务地址、识别服务地址、天气服务地址、QA历史长度、端口和日志配置 |
| `src/main.py` | FastAPI 应用入口，挂载推理 API |
| `src/api/routes.py` | REST API：`/prevention-plan` 个性化防治方案、`/qa` 智能问答、健康检查 |
| `src/engine/prevention_planner.py` | 个性化防治方案生成：结合识别结果、物候期、气候和用户角色输出方案 |
| `src/engine/qa_engine.py` | 多轮问答引擎：知识图谱检索、相似病虫害辨别、角色差异化回答、上下文记忆 |

### 9.5 `services/model-management/` — 模型管理服务

技术栈：Python + FastAPI + JSON文件型模型注册表。

```text
services/model-management/
├── Dockerfile
├── requirements.txt
├── src/
│   ├── config.py
│   ├── main.py
│   ├── api/routes.py
│   └── registry/registry.py
└── tests/
```

| 文件 | 作用 |
|---|---|
| `requirements.txt` | FastAPI、gRPC、Pydantic、PostgreSQL/SQLAlchemy、Prometheus等依赖 |
| `Dockerfile` | Python 运行镜像，创建 `/models` 目录用于模型注册表和模型文件 |
| `src/config.py` | 模型目录、数据库连接、灰度发布比例、镜像识别服务地址、端口和日志配置 |
| `src/main.py` | FastAPI 应用入口，挂载模型管理 API |
| `src/api/routes.py` | REST API：版本列表、注册模型、部署、回滚、训练任务、A/B测试、废弃版本 |
| `src/registry/registry.py` | 模型注册表核心逻辑：JSON持久化、版本CRUD、部署状态、灰度发布、回滚、A/B测试 |

### 9.6 `services/log-statistics/` — 日志统计服务

技术栈：Go + PostgreSQL + Redis + gRPC + Prometheus。

```text
services/log-statistics/
├── cmd/server/main.go
├── internal/
│   ├── grpc/server.go
│   ├── grpc/proto/types.go
│   ├── handler/log_handler.go
│   ├── handler/stats_handler.go
│   ├── model/recognition_log.go
│   ├── repository/postgres.go
│   └── service/log_service.go
└── migrations/postgres/
    ├── 001_init.up.sql
    └── 001_init.down.sql
```

| 文件 | 作用 |
|---|---|
| `cmd/server/main.go` | 服务入口：加载配置、连接PostgreSQL/Redis、启动gRPC和HTTP健康/metrics服务、优雅关闭 |
| `internal/model/recognition_log.go` | 识别日志、专家反馈、统计响应等领域模型 |
| `internal/repository/postgres.go` | PostgreSQL 数据访问层：写入识别日志、更新反馈、查询统计 |
| `internal/service/log_service.go` | 业务服务层：日志记录、反馈处理、统计聚合、健康检查 |
| `internal/grpc/server.go` | gRPC server 封装和启动逻辑 |
| `internal/grpc/proto/types.go` | 临时/手写 proto 类型，占位替代生成代码 |
| `internal/handler/log_handler.go` | gRPC 日志写入与反馈处理 handler |
| `internal/handler/stats_handler.go` | gRPC 统计查询与健康检查 handler |
| `migrations/postgres/001_init.up.sql` | PostgreSQL 表结构初始化 SQL |
| `migrations/postgres/001_init.down.sql` | PostgreSQL 回滚 SQL |

> 当前问题：`services/log-statistics/` 下缺少 `Dockerfile` 和 `go.mod`，但 `docker-compose.yml` 引用了 `services/log-statistics/Dockerfile`，因此该服务当前无法通过 Docker Compose 构建。

---

## 10. `tests/` 测试目录说明

```text
tests/
├── __init__.py
├── conftest.py
├── integration/
├── e2e/
└── performance/
```

| 路径/文件 | 作用 |
|---|---|
| `tests/conftest.py` | pytest 共享 fixture：合成测试图片、批量图片、模型路径mock、推理引擎stub、服务TestClient等 |
| `tests/integration/test_config.py` | 图像识别配置测试：默认值、环境变量覆盖、后端枚举、设备选择 |
| `tests/integration/test_preprocessing.py` | 图像预处理测试：读取、校验、resize、归一化、batch拼接 |
| `tests/integration/test_postprocess.py` | 识别后处理测试：15类标签映射、严重程度映射、置信度校准、结果格式化 |
| `tests/integration/test_model_loader.py` | 模型加载器测试：stub模型、路径、状态、热更新等 |
| `tests/integration/test_inference_engine.py` | 推理引擎测试：分类、小目标检测、严重程度、端到端流水线 |
| `tests/integration/test_kg_service.py` | 知识图谱服务测试：实体查询、语义搜索、防治推荐、风险预测 |
| `tests/integration/test_reasoning_engine.py` | 推理引擎服务测试：个性化防治、角色差异化、多轮QA、相似病虫害辨别 |
| `tests/integration/test_api_contracts.py` | API契约和SRS验收项测试：接口、15类分类、性能指标常量、功能需求覆盖 |
| `tests/e2e/` | 端到端测试预留目录 |
| `tests/performance/` | 性能测试预留目录 |

---

## 11. `txt/` 项目文档材料

| 文件 | 作用 |
|---|---|
| `需求规格说明书-子系统一病虫害识别和知识图谱模型基础软件-修改.docx` | 当前开发依据的需求规格说明书 Word 版 |
| `需求规格说明书-子系统一病虫害识别和知识图谱模型基础软件-修改.txt` | 从 Word 提取的文本版本，便于检索和对照 |
| `横向项目研究或开发-任务书.docx` | 项目任务书 Word 版 |
| `横向项目研究或开发-任务书.txt` | 任务书提取文本 |
| `横向项目研究或开发-end.docx` | 项目相关终稿或阶段性文档 |
| `*.原始备份.docx` / `*.校验提取.txt` | 原始备份与校验提取文本，用于追溯文档内容 |

---

## 12. 主要 API 说明

### 12.1 API Gateway 统一入口

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 聚合后端服务健康状态 |
| `POST` | `/api/v1/auth/login` | 登录并签发JWT token（当前为stub逻辑） |
| `POST` | `/api/v1/auth/refresh` | 刷新token（当前逻辑仍需完善） |
| `POST` | `/api/v1/recognition/detect` | 图像识别入口，代理到 Image Recognition |
| `POST` | `/api/v1/recognition/batch` | 批量图像识别 |
| `GET` | `/api/v1/recognition/result/{task_id}` | 查询异步批量识别结果 |
| `POST` | `/api/v1/recognition/feedback` | 用户反馈/专家纠错 |
| `GET` | `/api/v1/kg/entity/{entity_id}` | 查询知识图谱实体 |
| `POST` | `/api/v1/kg/search` | 知识图谱语义搜索 |
| `GET` | `/api/v1/kg/recommendation/{pest_id}` | 获取防治推荐 |
| `POST` | `/api/v1/kg/qa` | 知识图谱问答 |
| `GET` | `/api/v1/kg/risk/predict` | 病虫害风险预测 |
| `POST` | `/api/v1/reasoning/prevention-plan` | 个性化防治方案 |
| `POST` | `/api/v1/reasoning/qa` | 推理引擎多轮问答 |
| `GET` | `/api/v1/model/versions` | 模型版本列表 |
| `POST` | `/api/v1/model/register` | 注册模型版本 |
| `POST` | `/api/v1/model/deploy` | 部署模型版本 |
| `POST` | `/api/v1/model/rollback` | 回滚模型版本 |
| `POST` | `/api/v1/model/train` | 触发增量训练任务 |
| `GET` | `/api/v1/stats/dashboard` | 统计仪表盘数据 |

### 12.2 Python 服务自带文档

FastAPI 服务理论上会提供：

| 服务 | 文档地址 |
|---|---|
| Image Recognition | `http://localhost:8001/docs` |
| Knowledge Graph | `http://localhost:8002/docs` |
| Reasoning Engine | `http://localhost:8003/docs` |
| Model Management | `http://localhost:8004/docs` |

> 当前 API Gateway 自身没有真正接入 Swagger/OpenAPI 路由，`/docs` 目前不是可用的网关级文档页面。

---

## 13. 常用命令

| 命令 | 作用 |
|---|---|
| `make help` | 查看 Makefile 可用命令 |
| `make dev-up` | 使用 docker compose 构建并启动全部服务 |
| `make dev-down` | 停止并删除容器、网络和 volume |
| `make dev-logs` | 查看所有服务日志 |
| `make build-images` | 构建全部 Docker 镜像 |
| `make lint` | 运行 Go/Python lint |
| `make format` | 格式化 Go/Python 代码 |
| `make deploy-staging` | 应用 Kubernetes staging 配置 |
| `make deploy-prod` | 应用 Kubernetes production 配置 |
| `make clean` | 清理本地缓存、pyc、log等 |
| `make clean-docker` | 删除项目 Docker 容器、镜像和volume |

> 当前 Makefile 的 `test`/`test-integration` 主要写成 Go 测试命令，但仓库测试主体是 pytest；建议后续修正为 Python + Go 混合测试命令。

---

## 14. 当前还缺少什么整个工程才能完整运行起来

下面是当前从“代码已有”到“工程可一键运行”的主要缺口，按阻塞程度排序。

### 14.1 Docker Compose 端口和健康检查不一致（高优先级）

当前多个服务内部 HTTP 端口与 docker-compose 暴露/健康检查端口不一致：

| 服务 | 实际 HTTP 端口 | 预留 gRPC 端口 | docker-compose 当前映射/健康检查 | 问题 |
|---|---:|---:|---|---|
| Image Recognition | 8001 | 9001 | 暴露/检查 9001 | `/health` 是 HTTP，应该访问 8001，不是 gRPC 9001 |
| Knowledge Graph | 8002 | 9002 | 暴露/检查 9002 | 服务当前主要是 FastAPI HTTP，健康检查应走 8002 |
| Reasoning Engine | 8003 | 9003 | 暴露/检查 9003 | 同上，HTTP 健康检查应走 8003 |
| Model Management | 8004 | 9004 | 暴露/检查 9004 | 同上，HTTP 健康检查应走 8004 |
| Log Statistics | 8005 | 9005 | 暴露/检查 9005 | Go服务HTTP健康在8005，gRPC在9005 |

需要修复：

- `docker-compose.yml` 中为每个服务同时暴露 HTTP 和 gRPC 端口，或至少健康检查使用 HTTP 端口。
- 例如 Image Recognition 应改成类似：

```yaml
ports:
  - "8001:8001"
  - "9001:9001"
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
```

### 14.2 Python Docker ENTRYPOINT 使用了带连字符的模块名（高优先级）

多个 Dockerfile 使用类似：

```dockerfile
ENTRYPOINT ["python3", "-m", "services.knowledge-graph.src.main"]
```

但 Python 模块名不能包含 `-`，因此以下服务容器会启动失败：

- `services/image-recognition/Dockerfile`
- `services/knowledge-graph/Dockerfile`
- `services/reasoning-engine/Dockerfile`
- `services/model-management/Dockerfile`

建议修复方式之一：

```dockerfile
WORKDIR /app/services/knowledge-graph/src
ENV PYTHONPATH=/app/services/knowledge-graph/src:/app/libs/common-py
ENTRYPOINT ["python3", "main.py"]
```

或改为没有连字符的包名，例如 `knowledge_graph`、`reasoning_engine`、`model_management`。

### 14.3 Knowledge Graph 本地包名 `neo4j` 与官方 `neo4j` 包冲突（高优先级）

当前知识图谱服务目录：

```text
services/knowledge-graph/src/neo4j/
```

同时代码中需要导入官方 Neo4j Python 驱动：

```python
from neo4j import AsyncGraphDatabase, AsyncDriver
```

这会与本地 `neo4j` 包发生命名冲突，可能导致循环导入或导入错误。建议：

- 将本地目录 `src/neo4j/` 改名为 `src/kg_neo4j/` 或 `src/graph_store/`。
- 同步修改所有导入：

```python
from kg_neo4j.driver import get_entity_by_id, get_session
```

### 14.4 Log Statistics 缺少 Dockerfile 和 go.mod（高优先级）

`docker-compose.yml` 中引用：

```yaml
services/log-statistics/Dockerfile
```

但当前仓库中未发现该文件；并且 `services/log-statistics/` 也没有独立 `go.mod`。因此该服务无法构建。

需要补齐：

- `services/log-statistics/go.mod`
- `services/log-statistics/Dockerfile`
- 依赖版本：`pgx/v5`、`redis/go-redis/v9`、`zerolog`、`prometheus/client_golang`、`grpc` 等

### 14.5 Go API Gateway 当前可能无法编译（高优先级）

已发现几个潜在编译问题：

- `internal/proxy/proxy.go` 导入了 `net/url` 但未使用。
- `internal/handler/handler.go` 中 `for _, param := range c.Params` 的 `param` 未使用，Go 会编译失败。
- 当前本地环境没有 `go` 命令，尚未完成真实编译验证。

需要在安装 Go 1.22+ 后执行：

```bash
cd services/api-gateway
go mod tidy
go test ./...
go build ./cmd/server
```

### 14.6 Protobuf 生成链路未完成（中高优先级）

当前问题：

- proto 文件位于 `libs/proto/`。
- Makefile 写的是 `PROTO_DIR := proto`。
- 仓库没有 `buf.yaml` / `buf.gen.yaml`。
- Python/Go 服务当前大多使用手写占位类型，尚未接入生成的 gRPC 代码。

需要补齐：

- `buf.yaml`
- `buf.gen.yaml`
- 统一生成目录，例如：
  - Go：`pkg/pb/`
  - Python：各服务 `src/proto/` 或统一 `libs/generated-py/`
- 修改服务代码接入生成的 `*_pb2.py`、`*_pb2_grpc.py`、Go pb 类型。

### 14.7 Neo4j 初始化脚本没有真正执行（中高优先级）

`deploy/scripts/init_neo4j.sh` 目前被挂载为：

```yaml
./deploy/scripts/init_neo4j.sh:/init-neo4j.sh:ro
```

但 Neo4j 官方镜像不会自动执行 `/init-neo4j.sh`。并且脚本内部引用：

```bash
/app/services/knowledge-graph/src/neo4j/migrations/*.cypher
```

该路径没有挂载进 Neo4j 容器。

建议修复方式：

- 新增一个 `neo4j-migrate` 一次性容器，等待 Neo4j ready 后执行 Cypher。
- 或在 Knowledge Graph Service 启动时自动执行迁移。
- 或将 migrations 挂载进 Neo4j 容器并改写 entrypoint。

### 14.8 模型权重和真实训练数据缺失（中高优先级）

当前系统可以用 stub 模型跑测试，但真实识别需要：

| 资源 | 预期位置 | 说明 |
|---|---|---|
| 分类模型权重 | `models_registry/classifier/jujube_classifier.pth` 或 `.onnx` | 对应「7病8虫」15类 |
| 小目标检测模型 | `models_registry/detector/small_target_fpn.pth` 或 `.onnx` | 红蜘蛛卵粒、锈斑、产卵痕等小目标 |
| 严重程度模型 | `models_registry/severity/severity_classifier.pth` 或 `.onnx` | 健康/轻度/中度/重度 |
| 标注训练集 | `datasets/` 或 Docker volume | SRS要求初期≥2万张标注图像，每类≥1000张 |

如果没有权重，服务只能使用 stub 模型，结果没有业务意义。

### 14.9 `.env.example` 与服务实际变量不完全一致（中优先级）

示例：

- API Gateway 代码读取 `IMAGE_RECOGNITION_URL`、`KNOWLEDGE_GRAPH_URL` 等 HTTP URL。
- `.env.example` 当前主要提供 `IR_SERVICE_ADDR=image-recognition:9001` 这类 gRPC 地址。
- Python 服务分别使用 `KG_`、`RE_`、`MM_`、`IR_` 前缀变量。

建议补齐 `.env.example`：

```env
IMAGE_RECOGNITION_URL=http://image-recognition:8001
KNOWLEDGE_GRAPH_URL=http://knowledge-graph:8002
REASONING_ENGINE_URL=http://reasoning-engine:8003
MODEL_MANAGEMENT_URL=http://model-management:8004
LOG_STATISTICS_URL=http://log-statistics:8005

IR_HTTP_PORT=8001
KG_HTTP_PORT=8002
RE_HTTP_PORT=8003
MM_HTTP_PORT=8004
```

### 14.10 Makefile 测试命令与实际项目不匹配（中优先级）

当前 Makefile：

```makefile
test-unit:
	go test $(TEST_FLAGS) -short ./...

test-integration:
	go test $(TEST_FLAGS) -tags=integration ./test/integration/...
```

但当前测试主要是 Python pytest，且目录为 `tests/integration/`，不是 `test/integration/`。

建议改为：

```makefile
test-python:
	python -m pytest tests -v

test-go-api:
	cd services/api-gateway && go test ./...

test-go-log:
	cd services/log-statistics && go test ./...

test: test-python test-go-api test-go-log
```

### 14.11 API Gateway 文档未接入 Swagger（中优先级）

README 旧版本提到：

```text
API Gateway at /docs
```

但当前 Go/Gin 网关没有实际注册 Swagger 路由。需要：

- 增加 `swag init` 生成 docs。
- 注册 `ginSwagger.WrapHandler(swaggerFiles.Handler)`。
- 或改为统一由 OpenAPI YAML/Markdown 文档维护。

### 14.12 认证和用户系统仍是 stub（中优先级）

API Gateway 当前登录逻辑：

- 任意非空用户名/密码都可登录。
- 用户角色通过用户名 `admin` / `expert` 简单判断。
- 未对接真实用户管理模块。
- refresh token 逻辑仍需完善。

要用于真实部署，需要接入用户表或外部 IAM/OAuth2。

### 14.13 统计服务和图像识别服务之间尚未形成完整闭环（中优先级）

SRS要求每次识别自动记录日志、专家纠错回流训练数据。目前：

- Log Statistics Service 已有日志模型、repository、handler。
- Image Recognition 推理服务尚未在成功识别后调用 Log Statistics 记录日志。
- 专家反馈到训练队列/增量学习仍是接口和结构，闭环需要补齐。

### 14.14 Kubernetes 配置仍需要整体校验（中优先级）

当前 K8s 文件较全，但还需要：

- 校验镜像名是否与 Dockerfile 构建一致。
- 校验端口、Service targetPort 与实际 HTTP/gRPC 端口一致。
- 校验 Secret/ConfigMap 中变量名与服务读取一致。
- 补齐 Redis 配置。
- 补齐 Log Statistics Dockerfile/go.mod 后再部署。
- 确认 Neo4j Enterprise license 策略。

---

## 15. 建议的下一步修复顺序

如果目标是“尽快让整个工程 docker compose 跑起来”，建议按以下顺序：

1. **修复 Python Dockerfile ENTRYPOINT 与 PYTHONPATH**  
   让 Image Recognition、KG、Reasoning、Model Management 容器能启动。

2. **修复 docker-compose 端口和健康检查**  
   所有 HTTP `/health` 改到 8001~8005；必要时同时映射 gRPC 9001~9005。

3. **重命名 Knowledge Graph 的本地 `neo4j` 包**  
   避免与官方 Neo4j Python driver 冲突。

4. **补齐 Log Statistics 的 Dockerfile 和 go.mod**  
   让 docker compose 不再因缺文件失败。

5. **安装 Go 并修复 API Gateway 编译错误**  
   运行 `go mod tidy && go test ./... && go build ./cmd/server`。

6. **补齐 Neo4j 迁移执行机制**  
   使用一次性迁移容器或 KG 服务启动时迁移。

7. **修正 `.env.example` 与 Makefile**  
   保证环境变量名、端口、测试命令与真实代码一致。

8. **接入真实模型权重和数据集**  
   替换 stub 模型，训练并导出 `pth/onnx/trt`。

9. **补齐 protobuf 生成链路与 gRPC 接入**  
   使用 `libs/proto` 作为唯一契约来源，生成 Go/Python 代码。

10. **补齐日志闭环和专家纠错回流**  
    Image Recognition → Log Statistics → Model Management/Incremental Training。

---

## 16. 当前工程状态一句话总结

当前项目已经具备 **需求规格说明书要求的大部分模块代码与测试骨架**，尤其是「7病8虫」分类、图像识别流水线、知识图谱种子数据、KG查询、推理推荐、模型管理、API Gateway 和日志统计等主体均已形成；但它还处于 **“接近可运行的工程集成阶段”**，主要缺口集中在 **Docker启动方式、端口/健康检查、Go服务构建、Neo4j迁移执行、protobuf生成、真实模型权重与数据闭环**。这些问题修复后，工程才能稳定地通过 `docker compose up -d --build` 完整启动并进行端到端联调。

---

## 17. License

Proprietary — all rights reserved.
