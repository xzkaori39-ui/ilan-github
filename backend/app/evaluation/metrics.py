"""Pure, deterministic metrics for hidden-gold GraphRAG evaluation assets."""
from __future__ import annotations

from typing import Any

from app.evaluation.dataset import GraphEvalCase


def _round(value: float) -> float:
    return round(value, 4)


def _path_hops(path: dict[str, Any]) -> set[tuple[str, str, str]]:
    if not str(path.get("path_type", "")).startswith("entity_relation"):
        return set()
    entity_keys = [str(key) for key in path.get("entity_keys") or []]
    relationship_keys = [str(key) for key in path.get("relationship_keys") or []]
    if len(entity_keys) != 2 or len(relationship_keys) != 1:
        return set()
    return {(entity_keys[0], relationship_keys[0], entity_keys[1])}


def score_graph_case(
    case: GraphEvalCase,
    retrieved_chunk_ids: list[str],
    graph_chunk_ids: list[str],
    graph_paths: list[dict[str, Any]],
    graph_status: str,
) -> dict[str, bool | float | None | str]:
    """Score one result without exposing or sending gold fields to the Agent."""
    retrieved = set(retrieved_chunk_ids)
    graph_added = set(graph_chunk_ids)
    evidence_ids = {chunk_id for evidence_set in case.required_evidence_sets for chunk_id in evidence_set}
    evidence_complete = any(set(evidence_set).issubset(retrieved) for evidence_set in case.required_evidence_sets)
    entity_paths = [path for path in graph_paths if str(path.get("path_type", "")).startswith("entity_relation")]
    observed_hops = set().union(*(_path_hops(path) for path in entity_paths)) if entity_paths else set()
    expected_paths = []
    for path in case.allowed_relation_path_sets:
        hops = {(hop.source_entity_key, hop.relationship_key, hop.target_entity_key) for hop in path}
        # GraphRAG relationships are traversed as undirected edges during
        # retrieval.  Accept the reverse endpoint order for path validity,
        # while keeping the exact relationship key as the semantic anchor.
        hops |= {(target, relationship, source) for source, relationship, target in hops}
        expected_paths.append(hops)
    if not case.graph_sensitive:
        bridge_hit: bool | None = None
        path_valid: bool | None = None
    else:
        bridge_hit = (
            bool(graph_added & set(case.bridge_chunk_ids))
            if graph_status == "expanded"
            else False if graph_status == "no_new_evidence" else None
        )
        path_valid = any(expected.issubset(observed_hops) for expected in expected_paths) if entity_paths else None
    observed_hops_undirected = observed_hops | {
        (target, relationship, source) for source, relationship, target in observed_hops
    }
    return {
        "graph_sensitive": case.graph_sensitive,
        "evidence_set_complete": evidence_complete,
        "bridge_evidence_hit": bridge_hit,
        "graph_evidence_precision": _round(len(graph_added & evidence_ids) / len(graph_added)) if graph_added else None,
        "graph_path_valid": (
            any(expected.issubset(observed_hops_undirected) for expected in expected_paths)
            if entity_paths else None
        ) if case.graph_sensitive else path_valid,
        "distractor_hit": bool(retrieved & set(case.distractor_chunk_ids)),
        "graph_status": graph_status,
    }


def _rate(values: list[bool | float]) -> float | None:
    if not values:
        return None
    return _round(sum(float(value) for value in values) / len(values))


def aggregate_graph_metrics(
    details: list[dict[str, Any]], graph_sensitive: bool | None,
) -> dict[str, float | int | None]:
    """Aggregate all cases or a graph-sensitive/control subset without invented denominators."""
    rows = [detail for detail in details if graph_sensitive is None or detail.get("graph_sensitive") is graph_sensitive]
    bridge_hits = [detail["bridge_evidence_hit"] for detail in rows if detail.get("bridge_evidence_hit") is not None]
    precision = [float(detail["graph_evidence_precision"]) for detail in rows if detail.get("graph_evidence_precision") is not None]
    path_validity = [detail["graph_path_valid"] for detail in rows if detail.get("graph_path_valid") is not None]
    evidence_complete = [bool(detail.get("evidence_set_complete")) for detail in rows]
    distractor_resistance = [not bool(detail.get("distractor_hit")) for detail in rows]
    return {
        "case_count": len(rows),
        "applicable_case_count": len(rows),
        "evidence_set_complete_in_answer_context": _rate(evidence_complete),
        "bridge_evidence_hit_rate": _rate(bridge_hits),
        "graph_evidence_precision": _rate(precision),
        "graph_path_validity_rate": _rate(path_validity),
        "distractor_resistance_rate": _rate(distractor_resistance),
        "graph_fallback_rate": _rate([
            detail.get("graph_status") == "fallback_unavailable"
            for detail in rows
            if detail.get("graph_status") in {"expanded", "no_new_evidence", "fallback_unavailable"}
        ]),
    }


def build_paired_graph_comparison(
    off_details: list[dict[str, Any]], on_details: list[dict[str, Any]],
) -> dict[str, float | int | None]:
    """Measure only cases actually rescued from incomplete off to complete on evidence."""
    on_by_id = {str(detail.get("id", "")): detail for detail in on_details}
    candidates = [
        (off, on_by_id.get(str(off.get("id", ""))))
        for off in off_details
    ]
    eligible = [
        (off, on)
        for off, on in candidates
        if on
        and bool(off.get("graph_sensitive"))
        and bool(on.get("graph_sensitive"))
        and bool(off.get("success"))
        and bool(on.get("success"))
        and not bool(off.get("evidence_set_complete"))
    ]
    rescued = sum(1 for _off, on in eligible if bool(on.get("evidence_set_complete")))
    denominator = len(eligible)
    return {
        "graph_rescue_numerator": rescued,
        "graph_rescue_denominator": denominator,
        "graph_rescue_rate": _round(rescued / denominator) if denominator else None,
    }
