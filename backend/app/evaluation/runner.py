"""RAG 离线回放的指标计算与执行器。"""
from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from app.config import Settings
from app.deps import build_container
from app.evaluation.dataset import GraphEvalCase, load_evaluation_dataset
from app.evaluation.metrics import aggregate_graph_metrics, build_paired_graph_comparison, score_graph_case
from app.harness.agents.answer_agent import ANSWER_PROMPT
from app.harness.agents.intent_agent import INTENT_PROMPT
from app.harness.agents.query_rewriter import REWRITE_PROMPT
from app.harness.agents.verifier_agent import VERIFY_PROMPT

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "evaluation" / "real_document_qa.json"
EVALUATION_DATASETS = {
    "real_document_qa": DEFAULT_DATASET,
}
# Public releases include only the synthetic demo dataset.  Graph-sensitive
# evaluation assets require corpus-specific hidden gold and are deliberately
# kept outside this repository; the graph scoring framework remains reusable.
GRAPH_SENSITIVE_DATASET_IDS: set[str] = set()
QUALITY_METRIC_KEYS = ("recall_at_k", "mrr", "citation_correctness", "answer_key_coverage")


@dataclass(frozen=True)
class EvaluationCorpus:
    """Frozen, embedding-ready corpus shared by both evaluation profiles."""

    chunks: list[dict[str, Any]]
    vectors: list[list[float]]
    fingerprint: str


def _corpus_fingerprint(chunks: list[dict[str, Any]]) -> str:
    stable_rows = [
        {
            "id": str(chunk.get("_id", "")),
            "doc_id": str(chunk.get("doc_id", "")),
            "dept_id": str(chunk.get("dept_id", "")),
            "chunk_index": int(chunk.get("chunk_index", 0) or 0),
            "content": str(chunk.get("content", "")),
        }
        for chunk in chunks
    ]
    stable_rows.sort(key=lambda row: row["id"])
    payload = json.dumps(stable_rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def prepare_evaluation_corpus(store, embeddings) -> EvaluationCorpus:
    """Embed the active corpus once, before graph-off/graph-on replay begins."""
    chunks = await store.list_active_chunks()
    if not chunks:
        raise RuntimeError("没有可评测的 active chunks")
    vectors = await embeddings.embed([str(chunk.get("content", "")) for chunk in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError("评测语料向量数量与 active chunks 不一致")
    return EvaluationCorpus(chunks=chunks, vectors=vectors, fingerprint=_corpus_fingerprint(chunks))


async def install_evaluation_corpus(vector_store, corpus: EvaluationCorpus) -> None:
    """Seed an isolated profile vector store from an already-frozen corpus."""
    for chunk, vector in zip(corpus.chunks, corpus.vectors):
        await vector_store.add(
            str(chunk["_id"]), vector,
            {"doc_id": chunk["doc_id"], "dept_id": chunk["dept_id"], "chunk_index": chunk["chunk_index"]},
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_evaluation_dataset(dataset_id: str) -> Path:
    """Return only a checked-in evaluation asset, never an arbitrary filesystem path."""
    try:
        return EVALUATION_DATASETS[dataset_id]
    except KeyError as exc:
        raise ValueError(f"unknown evaluation dataset: {dataset_id}") from exc


def _round(value: float) -> float:
    return round(value, 4)


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * percentile
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower))


def _prompt_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256(template.encode("utf-8")).hexdigest()
        for name, template in {
            "answer": ANSWER_PROMPT,
            "intent": INTENT_PROMPT,
            "rewrite": REWRITE_PROMPT,
            "verify": VERIFY_PROMPT,
        }.items()
    }


def _config_snapshot(settings: Settings, reranker_enabled: bool, graph_enabled: bool) -> dict[str, Any]:
    return {
        "model": settings.deepseek_model,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "reranker_enabled": reranker_enabled,
        "reranker_provider": settings.reranker_provider,
        "reranker_model": settings.reranker_model if reranker_enabled else None,
        "graph_enabled": graph_enabled,
        "graph_expansion_limit": settings.graph_expansion_limit if graph_enabled else 0,
        "graph_rerank_min_score": settings.graph_rerank_min_score if graph_enabled else None,
        "hybrid_topk": settings.hybrid_topk,
        "bm25_top": settings.bm25_top,
        "vector_top": settings.vector_top,
        "prompt_hashes": _prompt_hashes(),
    }


def build_profile_configs(settings: Settings) -> list[dict[str, Any]]:
    return [
        {
            "name": "baseline_no_graph",
            "label": "当前基线（关闭图增强）",
            "config": _config_snapshot(settings, settings.reranker_enabled, False),
        },
        {
            "name": "candidate_graph_enabled",
            "label": "候选方案（开启图增强）",
            "config": _config_snapshot(settings, settings.reranker_enabled, True),
        },
    ]


def aggregate_profile_details(details: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    count = max(len(details), 1)
    ranks = [int(detail.get("rank") or 0) for detail in details]
    latencies = [int(detail.get("latency_ms") or 0) for detail in details]
    all_usage_available = bool(details) and all(
        detail.get(key) is not None
        for detail in details
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost_yuan")
    )
    metrics: dict[str, Any] = {
        "recall_at_k": _round(sum(1 for rank in ranks if 1 <= rank <= top_k) / count),
        "mrr": _round(sum(1.0 / rank if rank else 0.0 for rank in ranks) / count),
        "ndcg_at_k": _round(sum(1.0 / math.log2(rank + 1) if 1 <= rank <= top_k else 0.0 for rank in ranks) / count),
        "citation_correctness": _round(sum(float(detail.get("citation_correctness") or 0.0) for detail in details) / count),
        "answer_key_coverage": _round(sum(float(detail.get("answer_key_coverage") or 0.0) for detail in details) / count),
        "avg_latency_ms": round(sum(latencies) / count),
        "p50_latency_ms": _percentile(latencies, 0.5),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "failure_rate": _round(sum(1 for detail in details if not detail.get("success")) / count),
        "graph_usage_rate": _round(sum(1 for detail in details if int(detail.get("graph_evidence_count") or 0) > 0) / count),
        "avg_graph_evidence": _round(sum(int(detail.get("graph_evidence_count") or 0) for detail in details) / count),
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "cost_yuan": None,
    }
    if all_usage_available:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            metrics[key] = sum(int(detail[key]) for detail in details)
        metrics["cost_yuan"] = round(sum(float(detail["cost_yuan"]) for detail in details), 6)
    return metrics


def build_comparison(
    baseline: dict[str, Any], candidate: dict[str, Any], candidate_name: str = "candidate_graph_enabled",
) -> dict[str, Any]:
    deltas = {
        key: _round(float(candidate.get(key, 0.0)) - float(baseline.get(key, 0.0)))
        for key in QUALITY_METRIC_KEYS
    }
    recommendation = "consider_candidate" if all(value >= 0 for value in deltas.values()) and any(
        value > 0 for value in deltas.values()
    ) else "keep_baseline"
    return {"candidate": candidate_name, "deltas": deltas, "recommendation": recommendation}


def build_graph_groups(details: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    """Keep graph-challenge and ordinary-control outcomes auditable side by side."""
    return {
        "all": aggregate_graph_metrics(details, graph_sensitive=None),
        "graph_sensitive": aggregate_graph_metrics(details, graph_sensitive=True),
        "control": aggregate_graph_metrics(details, graph_sensitive=False),
    }


def _normalize(text: str) -> str:
    return "".join(text.lower().split()).replace("：", ":").replace("－", "-").replace("‑", "-")


def _source_file(hit: dict[str, Any], docs: dict[str, dict[str, Any]]) -> str:
    return str((docs.get(hit.get("doc_id", ""), {}).get("source") or {}).get("file_name", ""))


class RAGEvaluationRunner:
    def __init__(self, settings: Settings, dataset_id: str = "real_document_qa") -> None:
        self.settings = settings
        self.dataset_id = dataset_id

    async def run(
        self,
        evaluation_id: str,
        requested_by: str,
        progress_callback: Callable[[str, str], Awaitable[None]],
    ) -> dict[str, Any]:
        dataset_path = resolve_evaluation_dataset(self.dataset_id)
        graph_dataset = self.dataset_id in GRAPH_SENSITIVE_DATASET_IDS
        if graph_dataset:
            loaded_dataset = load_evaluation_dataset(dataset_path)
            if loaded_dataset.top_k != self.settings.hybrid_topk:
                raise ValueError(
                    "graph-sensitive dataset top_k must match the running hybrid_topk "
                    f"({loaded_dataset.top_k} != {self.settings.hybrid_topk})"
                )
            cases: list[dict[str, Any] | GraphEvalCase] = list(loaded_dataset.cases)
            dataset_summary = {
                "schema_version": loaded_dataset.schema_version,
                "graph_sensitive_cases": sum(case.graph_sensitive for case in loaded_dataset.cases),
                "control_cases": sum(not case.graph_sensitive for case in loaded_dataset.cases),
            }
        else:
            cases = json.loads(dataset_path.read_text(encoding="utf-8"))
            dataset_summary = {"schema_version": None, "graph_sensitive_cases": 0, "control_cases": len(cases)}
        await progress_callback("preparing", "正在冻结评测语料与向量")
        corpus = await self._prepare_corpus()
        profiles: list[dict[str, Any]] = []
        for profile in build_profile_configs(self.settings):
            await progress_callback(profile["name"], "正在运行固定校园问答集")
            details = await self._run_profile(cases, profile["config"], progress_callback, corpus)
            metrics = aggregate_profile_details(details, self.settings.hybrid_topk)
            if graph_dataset:
                metrics.update(aggregate_graph_metrics(details, graph_sensitive=None))
            profiles.append({
                **profile,
                "details": details,
                "failed_cases": sum(1 for row in details if not row["success"]),
                "metrics": metrics,
                "groups": build_graph_groups(details) if graph_dataset else {},
            })
        await progress_callback("persisting", "正在保存评测报告")
        return {
            "_id": evaluation_id,
            "status": "completed",
            "dataset": dataset_path.name,
            "dataset_id": self.dataset_id,
            "dataset_summary": dataset_summary,
            "case_count": len(cases),
            "top_k": self.settings.hybrid_topk,
            "corpus_fingerprint": corpus.fingerprint,
            "profiles": profiles,
            "comparison": {
                **build_comparison(
                profiles[0]["metrics"], profiles[1]["metrics"], candidate_name=profiles[1]["name"],
                ),
                "graph": build_paired_graph_comparison(profiles[0]["details"], profiles[1]["details"])
                if graph_dataset else None,
            },
            "created_by": requested_by,
            "completed_at": _now(),
        }

    async def _prepare_corpus(self) -> EvaluationCorpus:
        container = build_container(self.settings)
        if container.mongo is not None:
            await container.mongo.connect()
        try:
            return await prepare_evaluation_corpus(container.store, container.embeddings)
        finally:
            if container.mongo is not None:
                await container.mongo.close()

    async def _run_profile(
        self,
        cases: list[dict[str, Any] | GraphEvalCase],
        config: dict[str, Any],
        progress_callback: Callable[[str, str], Awaitable[None]],
        corpus: EvaluationCorpus,
    ) -> list[dict[str, Any]]:
        profile_settings = self.settings.model_copy(update={
            "reranker_enabled": bool(config["reranker_enabled"]),
            "graph_enabled": bool(config["graph_enabled"]),
        })
        container = build_container(profile_settings)
        if container.mongo is not None:
            await container.mongo.connect()
        if hasattr(container.session_store, "connect"):
            await container.session_store.connect()
        try:
            current_chunks = await container.store.list_active_chunks()
            if _corpus_fingerprint(current_chunks) != corpus.fingerprint:
                raise RuntimeError("评测期间 active corpus 已变化；请重新发起评测")
            container.bm25.index(corpus.chunks)
            await install_evaluation_corpus(container.vector_store, corpus)
            docs = {doc["_id"]: doc for doc in await container.store.list_documents(status="active")}
            rules = await container.rule_engine.active_rules()
            return [await self._run_case(container, docs, rules, case, progress_callback) for case in cases]
        finally:
            if container.mongo is not None:
                await container.mongo.close()

    async def _run_case(self, container, docs, rules, case: dict[str, Any] | GraphEvalCase, progress_callback) -> dict[str, Any]:
        started = time.perf_counter()
        graph_case = isinstance(case, GraphEvalCase)
        case_id = case.id if graph_case else case["id"]
        query = case.query if graph_case else case["query"]
        try:
            await progress_callback("case", f"正在评测 {case_id}")
            dept_ids = case.dept_ids if graph_case else [case["dept_id"]]
            retrieval = await container.retrieval_agent.retrieve_with_graph_status(
                [query], dept_ids, top_k=self.settings.hybrid_topk,
            )
            hits = retrieval.chunks
            if graph_case:
                relevant_chunk_ids = {
                    chunk_id for evidence_set in case.required_evidence_sets for chunk_id in evidence_set
                }
                rank = next((index for index, hit in enumerate(hits, 1) if str(hit.get("id", "")) in relevant_chunk_ids), 0)
            else:
                relevant = set(case["relevant_files"])
                rank = next((index for index, hit in enumerate(hits, 1) if _source_file(hit, docs) in relevant), 0)
            answer = await container.orchestrator.answer_agent.generate(query, hits, rules=rules)
            valid = 0
            for citation in answer.citations:
                chunk = await container.store.get("chunks", f"{citation.doc_id}:{citation.chunk_index}")
                doc = docs.get(citation.doc_id)
                if chunk and doc and citation.snippet in chunk.get("content", ""):
                    valid += 1
            citation_score = valid / len(answer.citations) if answer.citations else 0.0
            expected = case.expected_terms if graph_case else case.get("expected_terms") or []
            content = _normalize(answer.content)
            coverage = sum(1 for term in expected if _normalize(term) in content) / len(expected) if expected else 1.0
            detail: dict[str, Any] = {
                "id": case_id, "dept_id": ",".join(case.dept_ids) if graph_case else case["dept_id"], "query": query, "rank": rank,
                "hit_at_k": bool(rank and rank <= self.settings.hybrid_topk),
                "retrieved_files": [_source_file(hit, docs) for hit in hits],
                "citation_correctness": _round(citation_score), "answer_key_coverage": _round(coverage),
                "graph_evidence_count": sum(1 for hit in hits if hit.get("retrieval_source") == "graph"),
                "graph_status": retrieval.graph_status,
                "latency_ms": round((time.perf_counter() - started) * 1000), "success": True,
                "prompt_tokens": None, "completion_tokens": None, "total_tokens": None, "cost_yuan": None,
            }
            if graph_case:
                graph_chunks = [str(hit.get("id", "")) for hit in hits if hit.get("retrieval_source") == "graph"]
                graph_paths = [hit.get("graph_path") or {} for hit in hits if hit.get("retrieval_source") == "graph"]
                retrieved_chunk_ids = [str(hit.get("id", "")) for hit in hits if hit.get("id")]
                detail.update({
                    "retrieved_chunk_ids": retrieved_chunk_ids,
                    "graph_added_chunk_ids": graph_chunks,
                    "graph_paths": graph_paths,
                })
                detail.update(score_graph_case(case, retrieved_chunk_ids, graph_chunks, graph_paths, retrieval.graph_status))
            return detail
        except Exception as exc:  # noqa: BLE001
            detail = {
                "id": case_id, "dept_id": ",".join(case.dept_ids) if graph_case else case["dept_id"], "query": query, "rank": 0,
                "hit_at_k": False, "retrieved_files": [], "citation_correctness": 0.0,
                "answer_key_coverage": 0.0, "latency_ms": round((time.perf_counter() - started) * 1000),
                "graph_evidence_count": 0,
                "success": False, "error": str(exc), "prompt_tokens": None, "completion_tokens": None,
                "total_tokens": None, "cost_yuan": None,
            }
            if graph_case:
                detail.update({
                    "retrieved_chunk_ids": [], "graph_added_chunk_ids": [], "graph_paths": [],
                    **score_graph_case(case, [], [], [], "fallback_unavailable"),
                })
            return detail
