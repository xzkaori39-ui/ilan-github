"""RAG 离线评测的确定性指标与推荐规则。"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.evaluation.dataset import GraphEvalCase
from app.evaluation.runner import (
    RAGEvaluationRunner,
    aggregate_profile_details,
    build_comparison,
    build_graph_groups,
    build_profile_configs,
    resolve_evaluation_dataset,
)
from app.config import Settings
from app.retrieval.vector_store import MemoryVectorStore


def test_aggregate_profile_details_computes_retrieval_quality_and_engineering_metrics():
    metrics = aggregate_profile_details(
        [
            {
                "rank": 1,
                "citation_correctness": 1.0,
                "answer_key_coverage": 1.0,
                "latency_ms": 100,
                "success": True,
            },
            {
                "rank": 3,
                "citation_correctness": 0.5,
                "answer_key_coverage": 0.5,
                "latency_ms": 300,
                "success": True,
            },
            {
                "rank": 0,
                "citation_correctness": 0.0,
                "answer_key_coverage": 0.0,
                "latency_ms": 500,
                "success": False,
            },
        ],
        top_k=5,
    )

    assert metrics == {
        "recall_at_k": 0.6667,
        "mrr": 0.4444,
        "ndcg_at_k": 0.5,
        "citation_correctness": 0.5,
        "answer_key_coverage": 0.5,
        "avg_latency_ms": 300,
        "p50_latency_ms": 300,
        "p95_latency_ms": 480,
        "failure_rate": 0.3333,
        "graph_usage_rate": 0.0,
        "avg_graph_evidence": 0.0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "cost_yuan": None,
    }


def test_aggregate_profile_details_sums_token_cost_only_when_every_case_supplies_usage():
    metrics = aggregate_profile_details(
        [
            {
                "rank": 1,
                "citation_correctness": 1.0,
                "answer_key_coverage": 1.0,
                "latency_ms": 100,
                "success": True,
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
                "cost_yuan": 0.02,
            },
            {
                "rank": 1,
                "citation_correctness": 1.0,
                "answer_key_coverage": 1.0,
                "latency_ms": 200,
                "success": True,
                "prompt_tokens": 18,
                "completion_tokens": 12,
                "total_tokens": 30,
                "cost_yuan": 0.03,
            },
        ],
        top_k=5,
    )

    assert metrics["prompt_tokens"] == 30
    assert metrics["completion_tokens"] == 20
    assert metrics["total_tokens"] == 50
    assert metrics["cost_yuan"] == 0.05


def test_aggregate_profile_details_reports_graph_usage_and_added_evidence():
    metrics = aggregate_profile_details(
        [
            {"rank": 1, "citation_correctness": 1.0, "answer_key_coverage": 1.0, "latency_ms": 100, "success": True, "graph_evidence_count": 2},
            {"rank": 1, "citation_correctness": 1.0, "answer_key_coverage": 1.0, "latency_ms": 100, "success": True, "graph_evidence_count": 0},
            {"rank": 1, "citation_correctness": 1.0, "answer_key_coverage": 1.0, "latency_ms": 100, "success": True, "graph_evidence_count": 1},
        ],
        top_k=5,
    )

    assert metrics["graph_usage_rate"] == 0.6667
    assert metrics["avg_graph_evidence"] == 1.0


def test_comparison_only_recommends_candidate_when_all_quality_metrics_do_not_drop_and_one_improves():
    baseline = {
        "recall_at_k": 0.8,
        "mrr": 0.6,
        "citation_correctness": 0.8,
        "answer_key_coverage": 0.7,
    }
    candidate = {**baseline, "mrr": 0.7}

    assert build_comparison(baseline, candidate)["recommendation"] == "consider_candidate"
    assert build_comparison(candidate, baseline)["recommendation"] == "keep_baseline"


def test_profile_configs_compare_graph_on_off_without_changing_reranker_or_exposing_secrets():
    profiles = build_profile_configs(
        Settings(
            deepseek_api_key="do-not-store",
            relay_api_key="do-not-store",
            deepseek_model="gpt-5.4",
            embedding_model="text-embedding-v4",
            reranker_provider="local",
            reranker_model="BAAI/bge-reranker-v2-m3",
            reranker_enabled=True,
        )
    )

    assert [profile["name"] for profile in profiles] == ["baseline_no_graph", "candidate_graph_enabled"]
    assert profiles[0]["config"]["reranker_enabled"] is True
    assert profiles[1]["config"]["reranker_enabled"] is True
    assert profiles[0]["config"]["graph_enabled"] is False
    assert profiles[1]["config"]["graph_enabled"] is True
    assert "deepseek_api_key" not in str(profiles)
    assert "relay_api_key" not in str(profiles)


def test_graph_evaluation_groups_keep_challenge_and_control_metrics_separate():
    details = [
        {
            "graph_sensitive": True, "success": True, "evidence_set_complete": False,
            "bridge_evidence_hit": None, "graph_evidence_precision": None, "graph_path_valid": None,
            "distractor_hit": False, "graph_status": "disabled",
        },
        {
            "graph_sensitive": False, "success": True, "evidence_set_complete": True,
            "bridge_evidence_hit": None, "graph_evidence_precision": None, "graph_path_valid": None,
            "distractor_hit": False, "graph_status": "disabled",
        },
    ]

    groups = build_graph_groups(details)

    assert groups["all"]["case_count"] == 2
    assert groups["graph_sensitive"]["case_count"] == 1
    assert groups["control"]["evidence_set_complete_in_answer_context"] == 1.0


def test_evaluation_dataset_selection_is_an_explicit_allowlist():
    assert resolve_evaluation_dataset("real_document_qa").name == "real_document_qa.json"

    import pytest

    with pytest.raises(ValueError, match="unknown evaluation dataset"):
        resolve_evaluation_dataset("../../secret")


def test_graph_runner_persists_retrieval_ids_before_scoring_them():
    case = GraphEvalCase.model_validate({
        "id": "graph_case", "query": "问题", "dept_ids": ["dept_all"], "question_type": "cross_document_rule",
        "graph_sensitive": True, "required_evidence_sets": [["doc-a:0", "doc-b:0"]],
        "bridge_chunk_ids": ["doc-b:0"], "gold_entity_keys": ["entity-a", "entity-b"],
        "allowed_relation_path_sets": [[{
            "source_entity_key": "entity-a", "relationship_key": "rel-a-b", "target_entity_key": "entity-b",
        }]], "distractor_chunk_ids": [], "required_facts": ["事实"], "expected_terms": ["答案"],
        "source_review": {"reviewed_by": "maintainer", "reviewed_at": "2026-08-28"},
    })

    class RetrievalAgent:
        async def retrieve_with_graph_status(self, *_args, **_kwargs):
            return SimpleNamespace(chunks=[
                {"id": "doc-a:0", "doc_id": "doc-a", "chunk_index": 0},
                {"id": "doc-b:0", "doc_id": "doc-b", "chunk_index": 0, "retrieval_source": "graph",
                 "graph_path": {"path_type": "entity_relation", "entity_keys": ["entity-a", "entity-b"], "relationship_keys": ["rel-a-b"]}},
            ], graph_status="expanded")

    class AnswerAgent:
        async def generate(self, *_args, **_kwargs):
            return SimpleNamespace(content="答案", citations=[])

    container = SimpleNamespace(
        retrieval_agent=RetrievalAgent(),
        orchestrator=SimpleNamespace(answer_agent=AnswerAgent()),
        store=SimpleNamespace(),
    )

    async def progress(*_args):
        return None

    detail = asyncio.run(RAGEvaluationRunner(Settings())._run_case(container, {}, [], case, progress))

    assert detail["success"] is True
    assert detail["retrieved_chunk_ids"] == ["doc-a:0", "doc-b:0"]


def test_evaluation_corpus_embeds_once_and_can_seed_both_profiles_without_new_embedding_calls():
    from app.evaluation.runner import install_evaluation_corpus, prepare_evaluation_corpus
    from app.storage.store import MemoryStore

    class Embeddings:
        def __init__(self):
            self.calls: list[list[str]] = []

        async def embed(self, texts):
            self.calls.append(list(texts))
            return [[float(index)] for index, _text in enumerate(texts, start=1)]

    async def exercise():
        store = MemoryStore()
        await store.insert_document({"_id": "doc", "status": "active"})
        await store.insert_chunks([
            {"_id": "doc:0", "doc_id": "doc", "dept_id": "dept_jwc", "chunk_index": 0, "content": "第一条"},
            {"_id": "doc:1", "doc_id": "doc", "dept_id": "dept_jwc", "chunk_index": 1, "content": "第二条"},
        ])
        embeddings = Embeddings()
        corpus = await prepare_evaluation_corpus(store, embeddings)
        off_store, on_store = MemoryVectorStore(), MemoryVectorStore()
        await install_evaluation_corpus(off_store, corpus)
        await install_evaluation_corpus(on_store, corpus)
        return embeddings.calls, await off_store.count(), await on_store.count()

    calls, off_count, on_count = asyncio.run(exercise())

    assert calls == [["第一条", "第二条"]]
    assert (off_count, on_count) == (2, 2)
