"""Deterministic, hidden-gold GraphRAG evaluation metrics."""
from __future__ import annotations

from app.evaluation.dataset import GraphEvalCase
from app.evaluation.metrics import (
    aggregate_graph_metrics,
    build_paired_graph_comparison,
    score_graph_case,
)


def _case() -> GraphEvalCase:
    return GraphEvalCase.model_validate({
        "id": "graph_multi_001", "query": "跨文档问题", "dept_ids": ["dept_all"],
        "question_type": "cross_document_rule", "graph_sensitive": True,
        "required_evidence_sets": [["doc-a:0", "doc-b:3"]], "bridge_chunk_ids": ["doc-b:3"],
        "gold_entity_keys": ["entity-a", "entity-b"],
        "allowed_relation_path_sets": [[{
            "source_entity_key": "entity-a", "relationship_key": "rel-a-b", "target_entity_key": "entity-b",
        }]],
        "distractor_chunk_ids": ["doc-c:2"], "required_facts": ["事实一"], "expected_terms": ["术语一"],
        "source_review": {"reviewed_by": "maintainer", "reviewed_at": "2026-08-28"},
    })


def test_score_marks_complete_bridge_precision_path_and_distractor():
    score = score_graph_case(
        _case(), ["doc-a:0", "doc-b:3", "doc-c:2"], ["doc-b:3"], [{
            "path_type": "entity_relation", "entity_keys": ["entity-a", "entity-b"],
            "relationship_keys": ["rel-a-b"],
        }], "expanded",
    )

    assert score == {
        "graph_sensitive": True,
        "evidence_set_complete": True,
        "bridge_evidence_hit": True,
        "graph_evidence_precision": 1.0,
        "graph_path_valid": True,
        "distractor_hit": True,
        "graph_status": "expanded",
    }


def test_score_accepts_reverse_traversal_of_same_relation():
    score = score_graph_case(
        _case(), ["doc-a:0", "doc-b:3"], ["doc-b:3"], [{
            "path_type": "entity_relation_query", "entity_keys": ["entity-b", "entity-a"],
            "relationship_keys": ["rel-a-b"],
        }], "expanded",
    )

    assert score["graph_path_valid"] is True


def test_paired_rescue_counts_only_successful_off_incomplete_to_on_complete_transition():
    comparison = build_paired_graph_comparison(
        [
            {"id": "a", "graph_sensitive": True, "success": True, "evidence_set_complete": False},
            {"id": "b", "graph_sensitive": True, "success": True, "evidence_set_complete": False},
            {"id": "c", "graph_sensitive": True, "success": False, "evidence_set_complete": False},
        ],
        [
            {"id": "a", "graph_sensitive": True, "success": True, "evidence_set_complete": True},
            {"id": "b", "graph_sensitive": True, "success": True, "evidence_set_complete": False},
            {"id": "c", "graph_sensitive": True, "success": True, "evidence_set_complete": True},
        ],
    )

    assert comparison == {
        "graph_rescue_numerator": 1,
        "graph_rescue_denominator": 2,
        "graph_rescue_rate": 0.5,
    }


def test_aggregate_uses_null_for_metrics_without_applicable_graph_evidence():
    metrics = aggregate_graph_metrics([
        {
            "graph_sensitive": True, "success": True, "evidence_set_complete": False,
            "bridge_evidence_hit": False, "graph_evidence_precision": None, "graph_path_valid": None,
            "distractor_hit": False, "graph_status": "no_new_evidence",
        },
        {
            "graph_sensitive": False, "success": True, "evidence_set_complete": True,
            "bridge_evidence_hit": None, "graph_evidence_precision": None, "graph_path_valid": None,
            "distractor_hit": False, "graph_status": "disabled",
        },
    ], graph_sensitive=True)

    assert metrics == {
        "case_count": 1,
        "applicable_case_count": 1,
        "evidence_set_complete_in_answer_context": 0.0,
        "bridge_evidence_hit_rate": 0.0,
        "graph_evidence_precision": None,
        "graph_path_validity_rate": None,
        "distractor_resistance_rate": 1.0,
        "graph_fallback_rate": 0.0,
    }


def test_aggregate_does_not_treat_graph_disabled_baseline_as_zero_fallback():
    metrics = aggregate_graph_metrics([
        {
            "graph_sensitive": True, "success": True, "evidence_set_complete": False,
            "bridge_evidence_hit": None, "graph_evidence_precision": None, "graph_path_valid": None,
            "distractor_hit": False, "graph_status": "disabled",
        },
    ], graph_sensitive=True)

    assert metrics["graph_fallback_rate"] is None


def test_no_new_evidence_counts_as_missed_bridge_when_graph_query_completed():
    score = score_graph_case(_case(), ["doc-a:0"], [], [], "no_new_evidence")

    assert score["bridge_evidence_hit"] is False
