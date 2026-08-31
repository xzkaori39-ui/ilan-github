"""Retrieval Agent：执行混合检索（BM25 + 向量 + Rerank），返回 top-k chunks。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from app.llm.embeddings import EmbeddingClient
from app.retrieval.hybrid import HybridRetriever
from app.utils.logging import get_logger

logger = get_logger(__name__)

GRAPH_CANDIDATE_MULTIPLIER = 10


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[dict[str, Any]]
    graph_status: str


class RetrievalAgent:
    def __init__(
        self, hybrid: HybridRetriever, embeddings: EmbeddingClient, store, graph_expander=None,
        graph_rerank_min_score: float = 0.1,
    ) -> None:
        self.hybrid = hybrid
        self.embeddings = embeddings
        self.store = store
        self.graph_expander = graph_expander
        self.graph_rerank_min_score = float(graph_rerank_min_score)

    async def retrieve(self, queries: list[str], dept_ids: Optional[list[str]] = None, top_k: int = 5) -> list[dict[str, Any]]:
        """Compatibility view for callers that need only answer-context chunks."""
        return (await self.retrieve_with_graph_status(queries, dept_ids, top_k)).chunks

    async def retrieve_with_graph_status(
        self, queries: list[str], dept_ids: Optional[list[str]] = None, top_k: int = 5,
    ) -> RetrievalResult:
        """多 query × 多部门并行检索，合并去重。"""
        if not queries:
            return RetrievalResult([], "disabled" if self.graph_expander is None else "no_new_evidence")
        dept_ids = dept_ids or [None]
        seen: dict[str, dict[str, Any]] = {}

        for query in queries:
            vec = await self.embeddings.embed_query(query)
            for dept_id in dept_ids:
                if dept_id == "dept_all":
                    dept_id = None
                hits = await self.hybrid.retrieve(query, vec, dept_id=dept_id)
                for h in hits:
                    id_ = h.get("id", "")
                    if id_ and id_ not in seen:
                        seen[id_] = h

        # 检索后统一从持久化存储回填完整 chunk，避免向量库只返回 metadata
        # 导致正文、关键词和章节信息缺失；同时在返回前强制校验文档仍为 active。
        chunks: list[dict[str, Any]] = []
        for chunk_id, hit in seen.items():
            stored = await self.store.get("chunks", chunk_id)
            if not stored:
                continue
            doc = await self.store.get_document(stored.get("doc_id", ""))
            if not doc or doc.get("status") != "active":
                continue
            full = dict(hit)
            full.update(stored)
            full["id"] = chunk_id
            full["doc_title"] = doc.get("title", "")
            chunks.append(full)
        chunks.sort(key=lambda x: x.get("rerank_score", x.get("_rrf", x.get("score", 0.0))), reverse=True)
        primary_chunks = chunks[:top_k]
        if self.graph_expander is None:
            for chunk in primary_chunks:
                chunk["graph_status"] = "disabled"
            return RetrievalResult(primary_chunks, "disabled")
        try:
            graph_limit = max(int(getattr(self.graph_expander, "limit", 2) or 0), 0)
            expansion = await self.graph_expander.expand(
                primary_chunks,
                query=" ".join(queries),
                candidate_limit=max(graph_limit * GRAPH_CANDIDATE_MULTIPLIER, graph_limit),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("图增强检索失败，使用主检索结果: %s", exc)
            for chunk in primary_chunks:
                chunk["graph_status"] = "fallback_unavailable"
            return RetrievalResult(primary_chunks, "fallback_unavailable")
        existing_ids = {str(chunk.get("id", "")) for chunk in primary_chunks}
        supplemental = [
            chunk for chunk in expansion.chunks
            if str(chunk.get("id", "")) and str(chunk.get("id", "")) not in existing_ids
        ]
        graph_status = expansion.status
        if graph_status == "expanded" and supplemental:
            try:
                ranked_supplemental = await self.hybrid.reranker.rerank(
                    " ".join(queries), supplemental, top_k=graph_limit,
                )
                supplemental = [
                    chunk for chunk in ranked_supplemental
                    if float(chunk.get("rerank_score", 0.0) or 0.0) >= self.graph_rerank_min_score
                ]
            except Exception as exc:  # noqa: BLE001
                logger.warning("图补充证据重排失败，拒绝图补充: %s", exc)
                supplemental = []
            if supplemental:
                for chunk in supplemental:
                    chunk["graph_rerank_score"] = float(chunk.get("rerank_score", 0.0) or 0.0)
            else:
                graph_status = "no_new_evidence"
        elif graph_status == "expanded":
            graph_status = "no_new_evidence"
        for chunk in primary_chunks + supplemental:
            chunk["graph_status"] = graph_status
        return RetrievalResult(primary_chunks + supplemental, graph_status)
