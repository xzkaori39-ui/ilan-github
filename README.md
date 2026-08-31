# i兰 / iLAN

## 从 RAG 到记忆型、自进化 Agentic RAG

i兰（iLAN）是一个面向校园制度、办事流程与服务文档的知识服务系统，提供可追溯、可评测、可持续优化的智能问答能力。

项目沿着一条清晰的工程主链条逐步演进：

```text
基础 RAG
  ↓
BM25 + 向量混合检索
  ↓
GraphRAG + Neo4j：扩充跨文档证据链
  ↓
Agentic RAG：意图识别、改写、检索、回答、校验、反思
  ↓
记忆系统 + Loop Engineering：反馈驱动的策略自进化
```

## 核心链路

```text
文档上传/解析 → 清洗、切片、元数据抽取、向量化
        → BM25 + 向量检索 + Reranker
        → Neo4j 图增强（可选）
        → Python Harness / pi-agent Runtime
        → Intent → Rewrite → Retrieval → Answer → Verify → Reflect
        → 用户答案、原始文档引用、运行 Trace
        → 反馈采集 → 记忆更新 → Skills / Hooks / Rules 优化
```

原有混合 RAG 始终是主体链路。GraphRAG 不替换普通检索，而是在 Neo4j 可用时，根据实体、关系和跨文档连接补充少量图证据，用于增强多文档、间接关系问题的证据链与可解释性。图服务不可用时，系统会明确记录并提示当前回答未经过图增强，安全回退到普通 RAG。

## 项目演进

### 1. 基础 RAG：先建立可靠检索链路

- BM25 处理精确词和制度术语匹配。
- 向量检索处理语义相似问题。
- Reranker 对候选片段重新排序。
- 回答引用原始文档片段，并进行引用正确性校验。

### 2. GraphRAG：扩充证据链而不是制造黑盒答案

项目引入 Microsoft GraphRAG 和 Neo4j，将文档中的实体、关系、文本单元和社区组织为知识图谱。在线链路保留原始 Chunk 作为最终可引用事实源，图谱只提供可回溯的扩展路径和补充证据。

支持实体与关系扩展、跨文档证据连接、图路径记录、Neo4j 可视化，以及 Graph off/on 受控对照评测。

### 3. Agentic RAG：把一次问答拆成可观察的协作流程

Python Harness 和 pi-agent Runtime 协同完成 Intent、Rewrite、Retrieval、Answer、Verify、Reflect 等阶段。Python 侧负责权限、事实源、流程编排和策略约束；pi-agent 负责受控的 Agent loop、模型调用和工具调用。

在 Agentic RAG 的固定 DAG 跑稳之后，我没有把 GraphRAG 当成一条替代链路删掉，而是把它留在 Retrieval 后面做受控的证据扩展。因为制度问题很多时候不是某一条条款能单独回答：比如问“跨学院转专业后原来的选课和奖学金资格怎么处理”，关键词检索能找到几个相关片段，但不一定能识别“学生—学院—专业—学籍状态—奖学金办法”之间的关系。离线阶段我用 Microsoft GraphRAG 从已授权文本块抽取实体、关系、文本单元和社区，再把轻量图投影导入 Neo4j；在线阶段先走原来的 BM25、向量和 Reranker 找到种子 Chunk，再按实体和关系扩展少量跨文档候选，最后仍然回 MongoDB 校验 `doc_id/chunk_id/version`、文档是否 `active`，只把能回到原文的条款放进证据集。这样图谱负责补齐跨文档证据链和发现实体间的间接关系，Agentic RAG 负责在固定边界内编排和校验，两者都不直接替代官方事实；Neo4j 或图投影不可用时就记录状态并回退到原有混合 RAG，不影响正常问答。

### 4. 记忆与自进化：让系统从反馈中改进

iLAN 将不同生命周期和可信等级的知识分层管理：

- **工作记忆**：当前会话上下文与短期任务状态
- **情景记忆**：历史问答、反馈和运行事件
- **用户记忆**：经用户同意保存的偏好与个人语义信息
- **组织记忆**：绑定官方文档和部门范围的组织知识
- **学习记忆**：可评测、可灰度、可回滚的 Skills、Hooks、Rules 和策略版本

Loop Engineering 采用 `Execute → Observe → Reflect → Adapt → Deploy` 闭环：收集失败案例和用户反馈，分析根因，生成候选策略，在沙箱或灰度环境中验证后，再由管理员审核发布。“自进化”是可追踪、可回滚、有人机协同边界的工程闭环，而不是无约束地修改生产系统。

## 主要能力

- **混合检索**：BM25 + 向量检索 + 本地开源 BGE Reranker
- **GraphRAG**：Neo4j 知识图谱、实体关系扩展、图证据可视化
- **Agentic RAG**：多阶段 Agent 协作与受控工具调用
- **可追溯回答**：引用原始文档 Chunk，支持引用正确性校验
- **记忆系统**：工作、情景、用户、组织和学习记忆分层治理
- **反馈自进化**：Skills / Hooks / Rules、人工审核、灰度和回滚
- **工程化评测**：Recall@K、MRR、nDCG、证据完整率、Graph Rescue、P50/P95 时延
- **完整部署**：FastAPI、Next.js、MongoDB、Redis、Neo4j、Docker Compose

## 架构组件

```text
Next.js Web
    ↓ REST
FastAPI / Python Harness ───── pi-agent Runtime
    ├── MongoDB：文档、Chunk、向量、事实与记忆
    ├── Redis：会话状态与异步作业队列
    ├── Neo4j：GraphRAG 图投影与图查询
    └── Loop Engine：反馈、评测、策略版本与灰度发布
```

| 组件 | 作用 |
|---|---|
| `backend/` | FastAPI、文档处理、检索、Harness、记忆和 Loop |
| `services/pi-agent/` | pi-agent Runtime 与 Agent 协作执行 |
| `web/` | Next.js 聊天界面、管理台、评测和图谱视图 |
| `graphrag/` | GraphRAG 配置、Prompt 与构图工作区 |
| `third_party/microsoft-graphrag/` | Microsoft GraphRAG 源码镜像 |
| `demo_data/` | 一份项目自行编写的虚构演示文档 |

## 快速启动

前提：Docker Engine / Docker Desktop 与 Docker Compose v2。

```bash
git clone <your-repository-url> ilan
cd ilan
cp .env.example .env
# 分别填写对话模型 DEEPSEEK_* 与 Embedding/重排 RELAY_* 配置
docker compose up --build -d
docker compose ps
```

默认使用 Mongo 向量存储，适合快速演示。需要规模化原生向量检索时，将 `.env` 中
`VECTOR_BACKEND=milvus`（并确保 `MILVUS_DIMENSION` 与 `EMBEDDING_DIM` 一致），Compose
会同时启动 Milvus、etcd 与 MinIO。Milvus 暂时不可用时，后端会记录 warning 并回退 Mongo，
不会把失败静默降级为内存库。

访问：

- Web：<http://localhost:8080>
- API 文档：<http://localhost:8000/docs>
- Neo4j Browser：<http://localhost:7474>

演示账号仅在 `SEED_DEMO_USERS=true` 时创建：`admin / admin123`。公网部署前请关闭演示账号并替换所有默认密钥。

## 导入示例文档

示例文档不会在启动时自动写入数据库。启动完成后，显式执行：

```bash
docker compose exec backend python -m scripts.seed_data
docker compose exec backend python -m scripts.ingest_department_files --base /app/demo_data
```

首次导入后可切换 `VECTOR_BACKEND=milvus` 并重启后端；已有文档重新向量化写入 Milvus。

随后可以在 Web 中提问：“课程满额后怎么办？”、“成绩复核的时限是多久？”。

## GraphRAG 与评测

默认 `GRAPH_ENABLED=false`，先使用普通 RAG 验证系统。准备好授权文档、GraphRAG 投影和 Neo4j 后，再设置 `GRAPH_ENABLED=true` 并重启服务。详细流程见 [graphrag/README.md](graphrag/README.md)。

公开版提供一份基于虚构示例文档的 6 题评测集，用于验证 Graph off/on 链路。两组固定模型、Embedding、Reranker、Prompt 和 Top-K，仅切换图增强开关。

评测支持 Recall@K、MRR、nDCG@K、引用正确率、答案关键项覆盖率、证据集完整率、图证据精度、图路径有效率、Bridge 命中率、Graph Rescue、失败率和 P50/P95 时延。示例结果只用于验证软件链路，不能代表真实业务效果。

## 开发与验证

```bash
# 后端（Python 3.11）
cd backend
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pytest -q

# 前端（Node.js 22+）
cd ../web
npm ci
npm run build

# pi-agent
cd ../services/pi-agent
npm ci
npm run build

# Compose 配置检查
cd ../..
docker compose config
```

## 发布边界

- 不提交 `.env`、API Key、数据库卷、上传文件或真实政策/手册。
- 不提交 `graphrag/input`、`output`、`cache`、`runs`、`artifacts`、`backups` 等语料衍生物。
- 公开仓库只保留虚构示例文档；真实知识库请在本地或私有部署中导入。
- 所有组织知识应绑定有效文档 Chunk，并遵循权限、版本和归档治理。
