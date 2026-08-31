"""Versioned, hidden-gold contracts for graph-sensitive RAG evaluation."""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


def _document_id(chunk_id: str) -> str:
    """Extract the stable document id from the existing ``document:index`` chunk id."""
    return chunk_id.rsplit(":", 1)[0]


class RelationHop(BaseModel):
    source_entity_key: str = Field(min_length=1)
    relationship_key: str = Field(min_length=1)
    target_entity_key: str = Field(min_length=1)


class SourceReview(BaseModel):
    reviewed_by: str = Field(min_length=1)
    reviewed_at: date


class GraphEvalCase(BaseModel):
    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    dept_ids: list[str] = Field(min_length=1)
    question_type: str = Field(min_length=1)
    # The question wording is authored from a real user role, while gold stays hidden.
    persona: Literal["student", "teacher"] = "student"
    graph_sensitive: bool
    required_evidence_sets: list[list[str]] = Field(min_length=1)
    bridge_chunk_ids: list[str]
    gold_entity_keys: list[str]
    allowed_relation_path_sets: list[list[RelationHop]]
    distractor_chunk_ids: list[str] = Field(default_factory=list)
    required_facts: list[str] = Field(min_length=1)
    expected_terms: list[str] = Field(min_length=1)
    source_review: SourceReview

    @model_validator(mode="after")
    def validate_gold(self) -> "GraphEvalCase":
        evidence_ids = {chunk_id for evidence_set in self.required_evidence_sets for chunk_id in evidence_set}
        if any(not chunk_id for chunk_id in evidence_ids):
            raise ValueError("gold chunk ids must not be empty")
        if any(len(evidence_set) != len(set(evidence_set)) for evidence_set in self.required_evidence_sets):
            raise ValueError("gold evidence set must not contain duplicate chunk ids")
        if self.graph_sensitive and any(
            len({_document_id(chunk_id) for chunk_id in evidence_set}) < 2
            for evidence_set in self.required_evidence_sets
        ):
            raise ValueError("graph-sensitive evidence must span two documents")
        if set(self.bridge_chunk_ids) & set(self.distractor_chunk_ids):
            raise ValueError("bridge and distractor chunks must not overlap")
        if evidence_ids & set(self.distractor_chunk_ids):
            raise ValueError("evidence and distractor chunks must not overlap")
        if self.graph_sensitive and not self.allowed_relation_path_sets:
            raise ValueError("graph-sensitive case needs an allowed relation path")
        if self.graph_sensitive and not self.gold_entity_keys:
            raise ValueError("graph-sensitive case needs gold entity keys")
        if any(not path for path in self.allowed_relation_path_sets):
            raise ValueError("relation path sets must not be empty")
        return self


class GraphEvalDataset(BaseModel):
    dataset_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    top_k: int = Field(ge=1)
    cases: list[GraphEvalCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_case_ids(self) -> "GraphEvalDataset":
        case_ids = [case.id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("duplicate case id")
        return self


def load_evaluation_dataset(path: Path) -> GraphEvalDataset:
    """Read one self-contained GraphRAG evaluation asset.

    Public users should keep gold assets outside model inputs and version them
    with their authorized corpus.  The repository does not ship corpus-bound
    overlays or hidden gold.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return GraphEvalDataset.model_validate(payload)


def validate_dataset_references(
    dataset: GraphEvalDataset,
    active_chunk_ids: set[str],
    entity_keys: set[str],
    relationship_keys: set[str],
) -> list[str]:
    """Return stable, aggregate-safe errors for references absent from current projections."""
    errors: list[str] = []
    for case in dataset.cases:
        chunk_ids = {
            chunk_id
            for evidence_set in case.required_evidence_sets
            for chunk_id in evidence_set
        } | set(case.bridge_chunk_ids) | set(case.distractor_chunk_ids)
        for chunk_id in sorted(chunk_ids - active_chunk_ids):
            errors.append(f"{case.id}: unknown chunk {chunk_id}")
        for entity_key in sorted(set(case.gold_entity_keys) - entity_keys):
            errors.append(f"{case.id}: unknown entity {entity_key}")
        path_relationship_keys = {
            hop.relationship_key
            for path in case.allowed_relation_path_sets
            for hop in path
        }
        for relationship_key in sorted(path_relationship_keys - relationship_keys):
            errors.append(f"{case.id}: unknown relationship {relationship_key}")
    return sorted(errors)

