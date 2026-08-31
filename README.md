# i兰 / iLAN

面向校园服务文档的 Agentic RAG 与 GraphRAG 演示系统。项目覆盖文档入库、混合检索、本地开源重排、可引用问答、会话/组织记忆、异步 Worker、GraphRAG 图增强与 Graph off/on 离线评测。

> 本仓库不包含任何真实学校知识库、账号密钥、数据库卷、GraphRAG 运行产物或依赖私有语料的评测金标。`demo_data/` 中仅有一份项目自行编写的虚构示例文档。

## 架构

```text
Next.js Web → FastAPI / Python Harness → MongoDB（文档、向量、记忆）
                                  ├── Redis Stream Worker（入库、评测、Loop）
                                  ├── Neo4j（可选 GraphRAG 图投影）
                                  └── pi-agent（受控 Agent 运行时）
```

GraphRAG 的线上适配位于 `backend/app/graph/`；Microsoft GraphRAG 的源码镜像位于 `third_party/microsoft-graphrag/`，仅供对照与离线构图参考，许可证见其 `LICENSE`。

## 快速启动

前提：Docker Engine / Docker Desktop 与 Docker Compose v2。

```bash
git clone <your-repository-url> ilan
cd ilan
cp .env.example .env
# 编辑 .env：至少填写 DEEPSEEK_* 与 RELAY_*，并替换所有 replace-with-* 密码/密钥
docker compose up --build -d
docker compose ps
```

访问：

- Web：<http://localhost:8080>
- API 文档：<http://localhost:8000/docs>
- Neo4j Browser：<http://localhost:7474>（仅绑定本机）

演示账号仅在 `SEED_DEMO_USERS=true` 时创建：`admin / admin123`。公网部署前应关闭该开关并创建自己的管理员。

## 导入唯一示例知识库

Compose 启动后，示例文档会以只读方式挂载到 `/app/demo_data`，不会自动进入数据库。执行以下命令显式导入：

```bash
docker compose exec backend python -m scripts.seed_data
docker compose exec backend python -m scripts.ingest_department_files --base /app/demo_data
```

然后在 Web 中以管理员身份上传其他已获授权的文档，或就示例内容提问，例如“课程满额后怎么办？”、“成绩复核的时限是多久？”。

## 图增强与评测

默认 `GRAPH_ENABLED=false`，普通 RAG 可直接运行。准备好 Neo4j 和 GraphRAG 投影后，再设置 `GRAPH_ENABLED=true` 并重启服务；不可用时系统会记录/提示未经过图增强，避免误导。

公开版带有 6 条虚构示例评测题，可在超级管理员的 RAG 评测页运行 Graph off/on 对照。指标与自建金标方法见 [backend/evaluation/README.md](backend/evaluation/README.md)。示例结果只用于链路验证，不能用于宣称真实业务质量。

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

# Compose 配置
cd ..
docker compose config
```

## 发布边界

- 不提交 `.env`、API Key、Mongo/Redis/Neo4j 卷、上传文件、真实政策/手册或运行日志。
- 不提交 `graphrag/input`、`output`、`cache`、`runs`、`artifacts`、`backups` 等语料衍生物。
- 对真实知识库自行建立独立评测集：题干保持自然、金标不发送给模型、固定语料与模型配置后再比较 Graph off/on。
