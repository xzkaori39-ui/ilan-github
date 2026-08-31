"""Tests for the stable Mongo-chunk to Microsoft GraphRAG export contract."""
from __future__ import annotations

import json

import pytest

from app.graph.graphrag_export import build_input_rows, export_active_chunks
from app.storage.store import MemoryStore


def test_build_input_rows_keeps_citable_mongo_chunk_ids_and_skips_inactive_documents():
    rows = build_input_rows(
        documents=[
            {
                "_id": "doc_active",
                "dept_id": "dept_xsc",
                "title": "示例服务指南",
                "status": "active",
                "created_at": "2026-08-28T00:00:00+00:00",
            },
            {"_id": "doc_archived", "dept_id": "dept_xsc", "title": "旧手册", "status": "archived"},
        ],
        chunks=[
            {
                "_id": "doc_active:1",
                "doc_id": "doc_active",
                "dept_id": "dept_xsc",
                "chunk_index": 1,
                "section_title": "学籍管理",
                "content": "第二段正文",
            },
            {
                "_id": "doc_archived:0",
                "doc_id": "doc_archived",
                "dept_id": "dept_xsc",
                "chunk_index": 0,
                "content": "不得被导出",
            },
            {
                "_id": "doc_active:0",
                "doc_id": "doc_active",
                "dept_id": "dept_xsc",
                "chunk_index": 0,
                "section_title": "总则",
                "content": "第一段正文",
            },
        ],
    )

    assert [row["id"] for row in rows] == ["doc_active:0", "doc_active:1"]
    assert [row["text"] for row in rows] == ["第一段正文", "第二段正文"]
    assert rows[0]["title"] == "示例服务指南｜总则"
    assert rows[0]["creation_date"] == "2026-08-28T00:00:00+00:00"
    assert rows[0]["dept_id"] == "dept_xsc"
    assert rows[0]["section_title"] == "总则"
    assert rows[0]["raw_data"] == {
        "source_chunk_id": "doc_active:0",
        "source_document_id": "doc_active",
        "dept_id": "dept_xsc",
        "chunk_index": 0,
        "section_title": "总则",
    }


@pytest.mark.asyncio
async def test_export_active_chunks_writes_jsonl_and_a_content_manifest(tmp_path):
    store = MemoryStore()
    await store.insert_document({
        "_id": "doc_1", "dept_id": "dept_all", "title": "示例服务指南",
        "status": "active", "created_at": "2026-08-28T00:00:00+00:00",
    })
    await store.insert_chunks([{
        "_id": "doc_1:0", "doc_id": "doc_1", "dept_id": "dept_all",
        "chunk_index": 0, "section_title": "培养方案", "content": "培养方案正文",
    }])

    result = await export_active_chunks(store, tmp_path / "active_chunks.jsonl")

    assert result["row_count"] == 1
    assert len(result["content_sha256"]) == 64
    assert json.loads((tmp_path / "active_chunks.jsonl").read_text(encoding="utf-8")) == {
        "id": "doc_1:0", "text": "培养方案正文", "title": "示例服务指南｜培养方案",
        "creation_date": "2026-08-28T00:00:00+00:00",
        "dept_id": "dept_all", "section_title": "培养方案",
        "raw_data": {
            "source_chunk_id": "doc_1:0", "source_document_id": "doc_1", "dept_id": "dept_all",
            "chunk_index": 0, "section_title": "培养方案",
        },
    }
    manifest = json.loads((tmp_path / "active_chunks.manifest.json").read_text(encoding="utf-8"))
    assert manifest["row_count"] == 1
    assert manifest["content_sha256"] == result["content_sha256"]
