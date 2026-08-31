from __future__ import annotations

import pytest

from app.retrieval.vector_store import MilvusVectorStore, MongoVectorStore, build_vector_store
from app.storage.store import MemoryStore


class FakeTargetVectorStore:
    backend_name = "milvus"

    def __init__(self):
        self.rows = []

    async def add(self, id_, vector, metadata):
        self.rows.append((id_, vector, metadata))


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    async def embed(self, texts):
        self.calls.append(texts)
        return [[0.0, 1.0, 0.0] for _ in texts]


class FakeMilvusClient:
    def __init__(self):
        self.created = []
        self.rows = []
        self.deleted_filters = []

    def has_collection(self, collection_name):
        return bool(self.created)

    def create_collection(self, collection_name, schema, index_params):
        self.created.append((collection_name, schema, index_params))

    def insert(self, collection_name, data):
        self.rows.extend(data)
        return {"insert_count": len(data)}

    def search(self, collection_name, data, limit, output_fields, filter=None):
        rows = [row for row in self.rows if not filter or row["dept_id"] == "dept_a"]
        return [[
            {"id": row["id"], "distance": 0.12, "entity": {k: row[k] for k in output_fields}}
            for row in rows[:limit]
        ]]

    def delete(self, collection_name, filter):
        self.deleted_filters.append(filter)
        doc_id = filter.split('==', 1)[1].strip().strip('"')
        self.rows = [row for row in self.rows if row["doc_id"] != doc_id]
        return {"delete_count": 1}

    def get_collection_stats(self, collection_name):
        return {"row_count": len(self.rows)}


@pytest.mark.asyncio
async def test_milvus_store_creates_collection_writes_searches_filters_and_deletes():
    client = FakeMilvusClient()
    store = MilvusVectorStore(
        uri="http://milvus:19530", collection_name="ilan_chunks", dimension=3, client=client,
    )

    await store.add("doc-a:0", [1.0, 0.0, 0.0], {"doc_id": "doc-a", "dept_id": "dept_a", "chunk_index": 0})
    await store.add("doc-b:0", [0.0, 1.0, 0.0], {"doc_id": "doc-b", "dept_id": "dept_b", "chunk_index": 0})

    assert len(client.created) == 1
    hits = await store.search([1.0, 0.0, 0.0], top_k=5, dept_id="dept_a")
    assert hits == [{"id": "doc-a:0", "score": 0.12, "doc_id": "doc-a", "dept_id": "dept_a", "chunk_index": 0}]
    assert await store.count() == 2

    await store.delete_by_doc("doc-a")
    assert await store.count() == 1
    assert client.deleted_filters == ['doc_id == "doc-a"']


def test_milvus_initialization_falls_back_to_mongo_store_when_unavailable(monkeypatch):
    class BrokenMilvus:
        def __init__(self, **_kwargs):
            raise RuntimeError("milvus offline")

    monkeypatch.setattr("app.retrieval.vector_store.MilvusVectorStore", BrokenMilvus)
    fallback = build_vector_store(
        "milvus", MemoryStore(), milvus_uri="http://milvus:19530", milvus_collection="ilan_chunks", milvus_dimension=3,
    )

    assert isinstance(fallback, MongoVectorStore)


@pytest.mark.asyncio
async def test_milvus_migration_reuses_mongo_vectors_before_requesting_embeddings():
    from app.main import hydrate_milvus_vectors

    store = MemoryStore()
    await store.upsert("vector_embeddings", {
        "_id": "doc-a:0", "vector": [1.0, 0.0, 0.0], "doc_id": "doc-a", "dept_id": "dept_a", "chunk_index": 0,
    })
    target = FakeTargetVectorStore()
    embeddings = FakeEmbeddings()
    chunks = [
        {"embedding_id": "doc-a:0", "content": "已存在", "doc_id": "doc-a", "dept_id": "dept_a", "chunk_index": 0},
        {"embedding_id": "doc-b:0", "content": "缺失", "doc_id": "doc-b", "dept_id": "dept_b", "chunk_index": 0},
    ]

    reused, embedded = await hydrate_milvus_vectors(store, target, embeddings, chunks, dimension=3)

    assert (reused, embedded) == (1, 1)
    assert embeddings.calls == [["缺失"]]
    assert [row[0] for row in target.rows] == ["doc-a:0", "doc-b:0"]

