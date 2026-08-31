# iLAN 评测方法

公开仓库仅提供 `real_document_qa.json`：它对应 `demo_data/campus_service_demo.md` 的 6 条虚构问题，用于验证“文档入库 → 检索 → 生成 → 引用核验 → Graph off/on 对照”链路。

运行前先导入示例文档：

```bash
docker compose exec backend python -m scripts.seed_data
docker compose exec backend python -m scripts.ingest_department_files --base /app/demo_data
```

随后可在超级管理员的“RAG 评测”面板发起离线作业，或直接运行：

```bash
docker compose exec backend python -m scripts.evaluate_rag --dataset evaluation/real_document_qa.json
```

评测框架计算 Recall@K、MRR、nDCG@K、引用正确率、答案关键项覆盖率、失败率及 P50/P95 时延；GraphRAG 的图路径、桥接证据与救援指标支持通过 `app.evaluation.dataset.GraphEvalDataset` 接入自建隐藏金标。

不要将该示例的数字用于真实业务效果声明。实际评测应使用已获授权的文档、独立编写的自然语言问题、冻结的语料版本和不暴露给模型的金标。
