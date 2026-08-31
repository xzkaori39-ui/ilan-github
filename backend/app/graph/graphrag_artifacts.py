"""Normalize Microsoft GraphRAG tables into a Neo4j-safe projection."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _ids(value: Any, allowed: set[str] | None = None) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Iterable):
        return []
    result: list[str] = []
    for item in value:
        identifier = _text(item)
        if not identifier or (allowed is not None and identifier not in allowed) or identifier in result:
            continue
        result.append(identifier)
    return result


def _entity_key(entity_type: str, title: str) -> str:
    return f"{entity_type.casefold()}:{title.casefold()}"


def _community_key(row: dict[str, Any]) -> str:
    return f"{int(row.get('level', 0))}:{int(row.get('community', 0))}"


def build_projection(
    *,
    text_units: Iterable[dict[str, Any]],
    entities: Iterable[dict[str, Any]],
    relationships: Iterable[dict[str, Any]],
    communities: Iterable[dict[str, Any]],
    community_reports: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Keep only GraphRAG records that can lead back to a Mongo chunk."""
    projected_text_units: list[dict[str, Any]] = []
    for row in text_units:
        identifier = _text(row.get("id"))
        chunk_id = _text(row.get("document_id"))
        if identifier and chunk_id:
            projected_text_units.append({"id": identifier, "chunk_id": chunk_id})
    text_unit_ids = {row["id"] for row in projected_text_units}

    projected_entities: list[dict[str, Any]] = []
    entity_id_to_key: dict[str, str] = {}
    titles: list[str] = []
    for row in entities:
        identifier = _text(row.get("id"))
        title = _text(row.get("title"))
        entity_type = _text(row.get("type"))
        evidence_ids = _ids(row.get("text_unit_ids"), text_unit_ids)
        if not identifier or not title or not entity_type or not evidence_ids:
            continue
        key = _entity_key(entity_type, title)
        entity_id_to_key[identifier] = key
        titles.append(title)
        projected_entities.append({
            "id": identifier,
            "key": key,
            "name": title,
            "type": entity_type,
            "description": _text(row.get("description")),
            "text_unit_ids": evidence_ids,
        })
    title_counts = Counter(titles)
    entity_key_by_title = {
        entity["name"]: entity["key"]
        for entity in projected_entities
        if title_counts[entity["name"]] == 1
    }

    projected_relationships: list[dict[str, Any]] = []
    for row in relationships:
        identifier = _text(row.get("id"))
        source_key = entity_key_by_title.get(_text(row.get("source")))
        target_key = entity_key_by_title.get(_text(row.get("target")))
        description = _text(row.get("description"))
        evidence_ids = _ids(row.get("text_unit_ids"), text_unit_ids)
        if not identifier or not source_key or not target_key or not description or not evidence_ids:
            continue
        projected_relationships.append({
            "id": identifier,
            "key": f"{source_key}|{target_key}|{description}",
            "source_key": source_key,
            "target_key": target_key,
            "description": description,
            "weight": float(row.get("weight", 0.0)),
            "text_unit_ids": evidence_ids,
        })

    projected_communities: list[dict[str, Any]] = []
    for row in communities:
        entity_keys = [entity_id_to_key[identifier] for identifier in _ids(row.get("entity_ids")) if identifier in entity_id_to_key]
        evidence_ids = _ids(row.get("text_unit_ids"), text_unit_ids)
        if not entity_keys or not evidence_ids:
            continue
        projected_communities.append({
            "key": _community_key(row),
            "community": int(row.get("community", 0)),
            "level": int(row.get("level", 0)),
            "title": _text(row.get("title")),
            "entity_keys": entity_keys,
            "text_unit_ids": evidence_ids,
        })
    community_keys = {community["key"] for community in projected_communities}

    projected_reports: list[dict[str, Any]] = []
    for row in community_reports:
        community_key = _community_key(row)
        if community_key not in community_keys:
            continue
        projected_reports.append({
            "community_key": community_key,
            "title": _text(row.get("title")),
            "summary": _text(row.get("summary")),
            "full_content": _text(row.get("full_content")),
        })

    return {
        "text_units": projected_text_units,
        "entities": projected_entities,
        "relationships": projected_relationships,
        "communities": projected_communities,
        "community_reports": projected_reports,
    }
