"""Redis Stream worker：处理异步文档入库、Loop 唤醒等作业。"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.deps import build_container
from app.evaluation.runner import RAGEvaluationRunner
from app.utils.logging import setup_logging


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def process(container, job):
    payload = job.get("payload") or {}
    if job["type"] == "ingest_document":
        path = Path(payload["path"])
        try:
            doc = await container.indexer.ingest(
                path, payload["dept_id"], payload["uploaded_by"]
            )
            # 持久文件使用 UUID 命名，恢复原始文件名用于展示和版本判断。
            doc.setdefault("source", {})["file_name"] = payload.get("original_name", path.name)
            await container.store.update_document(doc["_id"], {"source": doc["source"]})
            relations = await container.conflict_detector.run_for_document(doc)
            review = await container.review_engine.create_review_order(doc)
            return {"document_id": doc["_id"], "relations": len(relations), "review_id": (review or {}).get("_id")}
        finally:
            path.unlink(missing_ok=True)
            try:
                path.parent.rmdir()
            except OSError:
                pass
    if job["type"] in {"feedback_received", "run_loop"}:
        async def progress(stage, detail):
            await container.job_queue.update_progress(job["_id"], {
                "stage": stage, "detail": detail,
            })

        result = await container.loop_engine.run_cycle(progress_callback=progress)
        result["memory_retention"] = await container.memory_retention.prune_expired()
        return result
    if job["type"] == "evaluation_run":
        evaluation_id = payload["evaluation_id"]

        async def progress(stage, detail):
            await container.job_queue.update_progress(job["_id"], {"stage": stage, "detail": detail})
            evaluation = await container.store.get("rag_evaluations", evaluation_id)
            if evaluation:
                evaluation.update({"status": "running", "stage": stage, "updated_at": _now()})
                await container.store.upsert("rag_evaluations", evaluation)

        report = await RAGEvaluationRunner(container.settings, dataset_id=payload.get("dataset_id", "real_document_qa")).run(
            evaluation_id, payload["requested_by"], progress,
        )
        existing = await container.store.get("rag_evaluations", evaluation_id) or {}
        report["created_at"] = existing.get("created_at", _now())
        await container.store.upsert("rag_evaluations", report)
        return {"evaluation_id": evaluation_id, "status": "completed"}
    raise ValueError(f"未知作业类型: {job['type']}")


async def main():
    settings = get_settings()
    setup_logging(settings.log_level)
    container = build_container(settings)
    if container.mongo is not None:
        await container.mongo.connect()
    if hasattr(container.session_store, "connect"):
        await container.session_store.connect()
    while True:
        for job in await container.job_queue.next_jobs():
            try:
                result = await process(container, job)
                await container.job_queue.finish(job, "completed", result)
            except Exception as exc:
                await container.job_queue.finish(job, "failed", {"error": str(exc)})
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(main())
