# iLAN GraphRAG 工作区

此目录保存 iLAN 与 Microsoft GraphRAG 对接所需的公开配置和 Prompt；它与线上 FastAPI 问答服务分离。

线上链路始终以原有混合 RAG 为主体。Neo4j 只在 `GRAPH_ENABLED=true` 且图投影可用时补充少量可追溯候选证据；图服务不可用时，系统回退到普通 RAG 并记录该状态。

## 构图流程

1. 将已获授权、已入库的 Mongo 文本块导出为官方 GraphRAG JSONL 输入：

   ```bash
   PYTHONPATH=backend python -m scripts.export_graphrag_input
   ```

2. 在独立运行目录中复制 `settings.yaml`、`prompts/`、私有 `.env` 与 JSONL，然后使用 Microsoft GraphRAG 构图：

   ```bash
   graphrag index --root /path/to/graphrag-run --method standard
   ```

3. 将官方 Parquet 输出转换为 iLAN 的小型 JSON 投影并导入 Neo4j：

   ```bash
   PYTHONPATH=backend python -m scripts.export_graphrag_artifacts \
     --input /path/to/graphrag-run/output \
     --output /path/to/projection.json
   PYTHONPATH=backend python -m scripts.import_graphrag_projection \
     --projection /path/to/projection.json
   ```

`input/`、`output/`、`cache/`、`logs/`、`artifacts/`、`runs/` 和 `backups/` 全部被 Git 忽略。不要向仓库提交任何真实文档、Parquet、图投影或运行日志。
