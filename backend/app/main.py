"""文枢后端入口（FastAPI）。"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest
from starlette.responses import Response

from app.api.router import api_router, root_router
from app.config import get_settings
from app.deps import build_container
from app.utils.logging import get_logger, setup_logging
from app.loop.default_skills import seed_default_skills

logger = get_logger(__name__)


async def hydrate_milvus_vectors(store, vector_store, embeddings, chunks, dimension: int) -> tuple[int, int]:
    """将已有 Mongo 向量迁移到 Milvus，仅为确实缺失的 Chunk 调用 Embedding API。"""
    saved = {row.get("_id"): row.get("vector") for row in await store.find("vector_embeddings")}
    missing: list[dict] = []
    pending: list[dict] = []
    reused = 0
    for chunk in chunks:
        vector = saved.get(chunk["embedding_id"])
        metadata = {
            "doc_id": chunk["doc_id"], "dept_id": chunk["dept_id"], "chunk_index": chunk["chunk_index"],
        }
        if isinstance(vector, list) and len(vector) == dimension:
            pending.append({"id": chunk["embedding_id"], "vector": vector, **metadata})
            reused += 1
        else:
            missing.append(chunk)

    if missing:
        vectors = await embeddings.embed([chunk["content"] for chunk in missing])
        for chunk, vector in zip(missing, vectors):
            pending.append({
                "id": chunk["embedding_id"], "vector": vector,
                "doc_id": chunk["doc_id"], "dept_id": chunk["dept_id"], "chunk_index": chunk["chunk_index"],
            })
    if hasattr(vector_store, "add_many"):
        await vector_store.add_many(pending)
    else:
        for row in pending:
            await vector_store.add(row.pop("id"), row.pop("vector"), row)
    return reused, len(missing)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    container = build_container(settings)
    app.state.container = container

    # 连接外部依赖（memory 模式跳过）
    if container.mongo is not None:
        try:
            await container.mongo.connect()
        except Exception as exc:  # noqa: BLE001
            logger.error("MongoDB 连接失败: %s", exc)
    if hasattr(container.session_store, "connect"):
        try:
            await container.session_store.connect()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis 连接失败(%s)，会话回退内存", exc)

    # 种子默认 Rules / Hooks
    await container.rule_engine.seed_defaults()
    await container.hook_engine.seed_defaults()
    seeded_skills = await seed_default_skills(container.store)
    if seeded_skills:
        logger.info("已创建 %d 个可执行基线 Skills", seeded_skills)

    # 种子账号（学生/管理员）
    await container.auth.seed_users()

    # 回填部门 Loop 阶段与审核统计字段（兼容旧数据）
    await _backfill_departments(container)

    # 重建内存检索索引（BM25 + 向量）：MongoDB 中已入库的文档可跨进程/重启被检索
    try:
        chunks = await container.store.list_active_chunks()
        if chunks:
            for c in chunks:
                container.bm25.add(c)
            # 共享 Mongo 向量库已持久化，无需每个 Pod 启动时重复嵌入全量文档。
            # 切 Milvus 时优先复制已有 Mongo 向量，避免一次迁移重新消耗 Embedding API 配额。
            actual_vector_backend = getattr(container.vector_store, "backend_name", settings.vector_backend)
            if actual_vector_backend == "milvus":
                reused, embedded = await hydrate_milvus_vectors(
                    container.store, container.vector_store, container.embeddings, chunks, settings.milvus_dimension,
                )
                logger.info("Milvus 向量同步完成: %d 复用，%d 新嵌入", reused, embedded)
            elif actual_vector_backend != "mongo":
                texts = [c["content"] for c in chunks]
                vectors = await container.embeddings.embed(texts)
                for c, v in zip(chunks, vectors):
                    await container.vector_store.add(
                        c["embedding_id"],
                        v,
                        {"doc_id": c["doc_id"], "dept_id": c["dept_id"], "chunk_index": c["chunk_index"]},
                    )
            logger.info("重建检索索引完成: %d chunks (BM25 + 向量)", len(chunks))
        else:
            logger.info("无已入库文档，跳过检索索引重建")
    except Exception as exc:  # noqa: BLE001
        logger.warning("重建检索索引失败(%s)，检索可能不完整", exc)

    logger.info("%s 启动完成 (storage=%s)", settings.app_name, settings.storage_mode)
    yield

    # 关闭
    if container.mongo is not None:
        await container.mongo.close()
    if hasattr(container.session_store, "close"):
        await container.session_store.close()
    await container.pi_runtime.close()
    logger.info("应用已关闭")


async def _backfill_departments(container) -> None:
    """为已存在的部门补充 loop_phase / review_stats / fade_out 字段（幂等）。"""
    from datetime import datetime, timezone

    try:
        microelectronics = await container.store.get_department("dept_weidianzi")
        if microelectronics is None:
            await container.store.upsert_department({
                "_id": "dept_weidianzi",
                "name": "微电子学院",
                "name_en": "School of Microelectronics",
                "category": "academic",
                "admin_users": [],
                "agent_config": {},
                "loop_phase": "human_in_loop",
                "review_stats": {"total": 0, "correct": 0, "accuracy": 0.0},
            })
            logger.info("已创建部门目录节点: 微电子学院")
        for dept in await container.store.list_departments():
            changed = False
            if "loop_phase" not in dept:
                dept["loop_phase"] = "human_in_loop"
                changed = True
            if "review_stats" not in dept:
                dept["review_stats"] = {"total": 0, "correct": 0, "accuracy": 0.0}
                changed = True
            if "admin_users" not in dept:
                dept["admin_users"] = []
                changed = True
            if changed:
                dept["updated_at"] = datetime.now(timezone.utc).isoformat()
                await container.store.upsert_department(dept)
        logger.info("部门 Loop 阶段字段回填完成")
    except Exception as exc:  # noqa: BLE001
        logger.warning("部门字段回填失败: %s", exc)


settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

# CORS：显式来源列表；通配符来源不允许携带凭据（浏览器规范）
_cors_origins = settings.cors_origin_list
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials="*" not in _cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(root_router)
app.include_router(api_router)


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain; version=0.0.4")


@app.get("/")
async def index():
    return {"app": settings.app_name, "docs": "/docs", "health": "/healthz"}
