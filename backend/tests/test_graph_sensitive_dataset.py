"""Graph-sensitive evaluation asset contract."""
from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from app.evaluation.dataset import (
    GraphEvalCase,
    GraphEvalDataset,
    RelationHop,
    SourceReview,
    validate_dataset_references,
)


def _case(**changes):
    value = {
        "id": "graph_multi_001",
        "query": "跨文档问题",
        "dept_ids": ["dept_all"],
        "question_type": "cross_document_rule",
        "graph_sensitive": True,
        "required_evidence_sets": [["doc-a:0", "doc-b:1"]],
        "bridge_chunk_ids": ["doc-b:1"],
        "gold_entity_keys": ["entity-a", "entity-b"],
        "allowed_relation_path_sets": [[{
            "source_entity_key": "entity-a",
            "relationship_key": "rel-a-b",
            "target_entity_key": "entity-b",
        }]],
        "distractor_chunk_ids": ["doc-c:2"],
        "required_facts": ["事实一", "事实二"],
        "expected_terms": ["术语一", "术语二"],
        "source_review": {"reviewed_by": "maintainer", "reviewed_at": "2026-08-28"},
    }
    value.update(changes)
    return value


def test_graph_sensitive_case_rejects_single_document_gold_evidence():
    with pytest.raises(ValidationError, match="two documents"):
        GraphEvalCase.model_validate(_case(required_evidence_sets=[["doc-a:0", "doc-a:1"]]))


def test_graph_sensitive_case_rejects_bridge_distractor_overlap():
    with pytest.raises(ValidationError, match="must not overlap"):
        GraphEvalCase.model_validate(_case(distractor_chunk_ids=["doc-b:1"]))


def test_dataset_rejects_duplicate_case_id():
    first = GraphEvalCase.model_validate(_case())
    second = GraphEvalCase.model_validate(_case())
    with pytest.raises(ValidationError, match="duplicate case id"):
        GraphEvalDataset(dataset_id="graph_sensitive_test", schema_version="1.0", top_k=5, cases=[first, second])


def test_reference_validation_reports_unknown_chunk_entity_and_relationship():
    case = GraphEvalCase.model_validate(_case())
    dataset = GraphEvalDataset(dataset_id="graph_sensitive_test", schema_version="1.0", top_k=5, cases=[case])

    errors = validate_dataset_references(dataset, {"doc-a:0"}, {"entity-a"}, {"rel-a-b"})

    assert any("unknown chunk" in error for error in errors)
    assert any("unknown entity" in error for error in errors)
    assert errors == sorted(errors)


def test_case_models_keep_normalized_typed_gold_fields():
    case = GraphEvalCase.model_validate(_case())

    assert case.source_review == SourceReview(reviewed_by="maintainer", reviewed_at=date(2026, 8, 28))
    assert case.allowed_relation_path_sets == [[RelationHop(
        source_entity_key="entity-a", relationship_key="rel-a-b", target_entity_key="entity-b",
    )]]
