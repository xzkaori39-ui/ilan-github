"""Stable input contract between i兰 Mongo chunks and Microsoft GraphRAG."""
from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
from pathlib import Path
from typing import Any


def build_input_rows(
    documents: Iterable[dict[str, Any]], chunks: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return one GraphRAG input row per active, citable Mongo chunk.

    The Mongo chunk id is deliberately the upstream document id. GraphRAG keeps
    that id on each generated TextUnit, which lets the later Neo4j projection
    lead an answer back to the original i兰 citation target.
    """
    active_documents = {
        str(document.get("_id", "")): document
        for document in documents
        if document.get("status") == "active" and document.get("_id")
    }
    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        doc_id = str(chunk.get("doc_id", ""))
        document = active_documents.get(doc_id)
        chunk_id = str(chunk.get("_id", ""))
        content = str(chunk.get("content", "")).strip()
        if document is None or not chunk_id or not content:
            continue
        section_title = str(chunk.get("section_title", "")).strip()
        document_title = str(document.get("title", "")).strip() or "未命名文档"
        title = f"{document_title}｜{section_title}" if section_title else document_title
        dept_id = str(chunk.get("dept_id") or document.get("dept_id", ""))
        chunk_index = int(chunk.get("chunk_index", 0))
        rows.append({
            "id": chunk_id,
            "text": content,
            "title": title,
            "creation_date": str(document.get("created_at", "")),
            # These fields must be top-level: GraphRAG's structured reader
            # wraps each input row as raw_data, and its metadata prepender
            # therefore cannot read values nested under our raw_data field.
            "dept_id": dept_id,
            "section_title": section_title,
            "raw_data": {
                "source_chunk_id": chunk_id,
                "source_document_id": doc_id,
                "dept_id": dept_id,
                "chunk_index": chunk_index,
                "section_title": section_title,
            },
        })
    return sorted(
        rows,
        key=lambda row: (
            row["raw_data"]["dept_id"],
            row["raw_data"]["source_document_id"],
            row["raw_data"]["chunk_index"],
            row["id"],
        ),
    )


async def export_active_chunks(store, output_path: str | Path) -> dict[str, Any]:
    """Write active citable chunks as JSONL plus a deterministic manifest.

    JSONL is an official GraphRAG input format and avoids adding PyArrow to the
    running i兰 backend. GraphRAG itself writes its derived knowledge model as
    Parquet in its separate environment.
    """
    output = Path(output_path)
    documents = await store.list_documents(status="active")
    chunks = await store.list_active_chunks()
    rows = build_input_rows(documents, chunks)
    serialized_rows = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]
    content = "\n".join(serialized_rows) + ("\n" if serialized_rows else "")
    content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "format": "jsonl",
        "row_count": len(rows),
        "content_sha256": content_sha256,
        "source": "ilan.mongo.active_chunks",
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "path": str(output), "manifest_path": str(manifest_path)}
