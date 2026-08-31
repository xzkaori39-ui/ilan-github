"""Bounded, inspectable graph expansion for the existing RAG retrieval chain."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class GraphExpansionResult:
    status: str
    chunks: list[dict[str, Any]]


def graph_query_terms(query: str) -> list[str]:
    """Generate ordered lexical anchors, keeping domain phrases ahead of n-grams."""
    stop_terms = {
        "什么", "哪些", "如何", "需要", "有关", "相关", "这个", "那个", "学生",
        "学校", "大学", "手册", "流程", "规定", "说明", "介绍", "准备",
    }
    ordered: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        term = term.strip().lower()
        if len(term) < 2 or term in stop_terms or term in seen:
            return
        seen.add(term)
        ordered.append(term)

    # Quoted spans are usually the user's intended entity anchors.  Add them
    # before sentence-level n-grams so a long introductory phrase cannot
    # consume the bounded term budget and hide the actual entities.
    quoted = re.findall(r"[“\"「『]([^”\"」』]{2,})[”\"」』]", query)
    for phrase in quoted:
        add(phrase)
    if quoted:
        # When users quote entities, ignore the surrounding instruction text
        # (often containing ``学生手册``/``东华大学``) and derive n-grams only
        # from the quoted anchors themselves.
        for phrase in quoted:
            for size in (4, 3, 2):
                for index in range(max(len(phrase) - size, -1), -1, -1):
                    add(phrase[index:index + size])
        return ordered[:32]
    for token in re.findall(r"[a-z0-9][a-z0-9_-]{1,}", query.lower()):
        add(token)
    for raw_run in re.findall(r"[\u4e00-\u9fff]{2,}", query):
        run = re.sub(r"(?:需要准备什么|需要什么|是什么|哪些|如何|什么)$", "", raw_run)
        add(run)
        for size in (4, 3, 2):
            for index in range(max(len(run) - size, -1), -1, -1):
                add(run[index:index + size])
    return ordered[:32]


def _path_rank(path: dict[str, Any]) -> tuple[float, float, str]:
    # A path matching both concrete query entities outranks one that only
    # matched a description or one endpoint.
    relevance = (
        100.0 * float(path.get("direct_entity_match_count", 0) or 0)
        + float(path.get("query_match_count", 0) or 0)
    )
    degree = float(path.get("source_degree", 0) or 0) + float(path.get("target_degree", 0) or 0)
    specificity = 1.0 / (1.0 + degree)
    return relevance, specificity * float(path.get("score", 0.0) or 0.0), str(path.get("target_chunk_id", ""))


def _is_generic_entity_path(path: dict[str, Any]) -> bool:
    return any(_is_generic_entity_key(str(entity_key)) for entity_key in path.get("entity_keys") or [])


def _is_high_degree_generic_path(path: dict[str, Any]) -> bool:
    """Drop generic hub paths unless the query explicitly matches the path."""
    if not _is_generic_entity_path(path):
        return False
    degree = max(
        int(path.get("source_degree", 0) or 0),
        int(path.get("target_degree", 0) or 0),
    )
    return degree >= 100 and int(path.get("query_match_count", 0) or 0) <= 1


def _entity_label(entity_key: str) -> str:
    """Return the human-readable part of a typed graph entity key."""
    return str(entity_key).split(":", 1)[-1].strip().lower()


def _is_generic_entity_key(entity_key: str) -> bool:
    """Identify hubs that are too broad to justify graph expansion on their own."""
    label = _entity_label(entity_key)
    if label in {"学校", "大学", "学院", "东华大学", "学生手册", "本科生手册", "研究生手册"}:
        return True
    # Institution-prefixed handbooks/regulations are common noisy bridges in
    # the imported corpus (e.g. ``东华大学学生宿舍管理办法``).
    return label.startswith(("学校", "大学", "东华大学")) and (
        "手册" in label or "管理办法" in label
    )


def _path_has_direct_entity_overlap(path: dict[str, Any], terms: list[str]) -> bool:
    """Require a non-generic path endpoint to directly overlap a query term.

    Neo4j also scores matches in descriptions and relationship text.  That is
    useful for ranking, but it can make a generic institution-to-policy edge
    look relevant.  Expansion therefore requires lexical overlap on a
    concrete entity label itself.
    """
    if not terms:
        return True
    entity_keys = [str(key) for key in path.get("entity_keys") or []]
    if not str(path.get("path_type", "")).startswith("entity_relation") or not entity_keys:
        return False
    specific_labels = [
        _entity_label(key) for key in entity_keys if not _is_generic_entity_key(key)
    ]
    return any(
        term in label or label in term
        for label in specific_labels
        for term in terms
        if term
    )


def _direct_entity_match_count(path: dict[str, Any], terms: list[str]) -> int:
    labels = [
        _entity_label(str(key)) for key in path.get("entity_keys") or []
        if not _is_generic_entity_key(str(key))
    ]
    return sum(
        any(term in label or label in term for term in terms if term)
        for label in labels
    )


class GraphExpander:
    def __init__(self, graph_store, store, limit: int = 2) -> None:
        self.graph_store = graph_store
        self.store = store
        self.limit = limit

    async def expand(
        self, seeds: list[dict[str, Any]], query: str = "", candidate_limit: int | None = None,
    ) -> GraphExpansionResult:
        """Return active graph candidates without changing the primary hit ranking.

        ``candidate_limit`` lets the retrieval-side content gate inspect a
        bounded pool before it selects the final ``self.limit`` answer
        evidence.  Direct callers keep the original final-limit behavior.
        """
        if getattr(self.graph_store, "enabled", True) is False:
            return GraphExpansionResult("disabled", [])
        seed_ids = [str(chunk.get("id", "")) for chunk in seeds if chunk.get("id")]
        max_candidates = max(self.limit * 10, self.limit)
        selected_limit = self.limit if candidate_limit is None else min(max(int(candidate_limit), 0), max_candidates)
        if not seed_ids or selected_limit <= 0:
            return GraphExpansionResult("no_new_evidence", [])
        terms = graph_query_terms(query)
        result = await self.graph_store.neighbor_chunks_with_paths(seed_ids, max_candidates, terms)
        if result.get("unavailable"):
            return GraphExpansionResult("fallback_unavailable", [])

        paths = list(result.get("paths") or [])
        if terms:
            paths = [path for path in paths if _path_has_direct_entity_overlap(path, terms)]
            for path in paths:
                path["direct_entity_match_count"] = _direct_entity_match_count(path, terms)
        if any(int(path.get("query_match_count", 0) or 0) > 0 for path in paths):
            paths = [path for path in paths if int(path.get("query_match_count", 0) or 0) > 0]
        specific_paths = [path for path in paths if not _is_generic_entity_path(path)]
        if specific_paths:
            paths = specific_paths
        paths.sort(key=_path_rank, reverse=True)

        expanded: list[dict[str, Any]] = []
        seen_ids = set(seed_ids)
        seen_entity_pairs: set[tuple[str, ...]] = set()
        for path in paths:
            if len(expanded) >= selected_limit:
                break
            if _is_high_degree_generic_path(path):
                continue
            chunk_id = str(path.get("target_chunk_id", ""))
            if not chunk_id or chunk_id in seen_ids:
                continue
            entity_pair = tuple(sorted(str(key) for key in path.get("entity_keys") or []))
            if entity_pair and entity_pair in seen_entity_pairs:
                continue
            chunk = await self.store.get("chunks", chunk_id)
            if not chunk:
                continue
            doc = await self.store.get_document(chunk.get("doc_id", ""))
            if not doc or doc.get("status") != "active":
                continue
            seen_ids.add(chunk_id)
            if entity_pair:
                seen_entity_pairs.add(entity_pair)
            item = dict(chunk)
            item.update({
                "id": chunk_id,
                "graph_score": round(float(path.get("score", 0.0)), 4),
                "retrieval_source": "graph",
                "graph_path": {
                    "seed_chunk_id": str(path.get("seed_chunk_id", "")),
                    "target_chunk_id": chunk_id,
                    "path_type": str(path.get("path_type", "topic_fallback")),
                    "entity_keys": [str(key) for key in path.get("entity_keys") or []],
                    "relationship_keys": [str(key) for key in path.get("relationship_keys") or []],
                },
            })
            expanded.append(item)
        return GraphExpansionResult("expanded" if expanded else "no_new_evidence", expanded)
