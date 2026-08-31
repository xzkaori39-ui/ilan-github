"""测试混合检索（内存向量 + BM25 + 启发式重排）。"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.llm.embeddings import EmbeddingClient
from app.retrieval.bm25 import BM25Index
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker, HeuristicReranker, build_reranker
from app.retrieval.vector_store import MemoryVectorStore
from app.storage.store import MemoryStore


@pytest.mark.asyncio
async def test_hybrid_retrieve(embeddings):
    bm25 = BM25Index()
    vs = MemoryVectorStore()
    docs = [
        {"_id": "c1", "doc_id": "d1", "dept_id": "dept_jwc", "chunk_index": 0, "content": "本科生选课管理办法规定选课时间。"},
        {"_id": "c2", "doc_id": "d2", "dept_id": "dept_jwc", "chunk_index": 0, "content": "退课应当在开课后两周内申请。"},
        {"_id": "c3", "doc_id": "d3", "dept_id": "dept_cwc", "chunk_index": 0, "content": "学费缴纳方式与时间安排。"},
    ]
    bm25.index(docs)
    vecs = await embeddings.embed([d["content"] for d in docs])
    for d, v in zip(docs, vecs):
        await vs.add(d["_id"], v, {"doc_id": d["doc_id"], "dept_id": d["dept_id"]})

    hybrid = HybridRetriever(bm25=bm25, vector_store=vs, reranker=HeuristicReranker(), top_k=3)
    query_vec = await embeddings.embed_query("选课时间是什么时候")
    hits = await hybrid.retrieve("选课时间是什么时候", query_vec)
    assert hits, "应返回检索结果"
    assert hits[0]["id"] in {"c1", "c2", "c3"}


@pytest.mark.asyncio
async def test_vector_store_cosine(embeddings):
    vs = MemoryVectorStore()
    v = await embeddings.embed(["本科生选课管理办法"])
    await vs.add("a", v[0], {"dept_id": "dept_jwc"})
    hits = await vs.search(v[0], top_k=1)
    assert hits and hits[0]["id"] == "a"
    assert hits[0]["score"] > 0.9


def test_build_reranker_local_provider_overrides_relay(monkeypatch):
    """本地 provider 必须优先于已配置的中转站 key。"""
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", object())
    reranker = build_reranker(enabled=True, provider="local", relay=object())
    assert isinstance(reranker, CrossEncoderReranker)


@pytest.mark.asyncio
async def test_local_reranker_loads_only_from_cached_model_files(monkeypatch):
    calls: list[dict] = []

    class FakeCrossEncoder:
        def __init__(self, _model_name, **kwargs):
            calls.append(kwargs)

        def predict(self, _pairs):
            return []

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=FakeCrossEncoder))
    reranker = CrossEncoderReranker("BAAI/bge-reranker-v2-m3")

    assert await reranker.rerank("选课", []) == []
    assert calls == [{"local_files_only": True}]


@pytest.mark.asyncio
async def test_relay_embeddings_are_batched_to_ten_and_keep_input_order():
    calls: list[list[str]] = []

    class FakeRelay:
        async def embed(self, texts, model):
            calls.append(texts)
            return [[float(index)] for index, _ in enumerate(texts, start=sum(map(len, calls[:-1])))]

    client = EmbeddingClient(Settings(embedding_provider="relay", relay_api_key="present"), relay=FakeRelay())
    vectors = await client.embed([f"chunk-{index}" for index in range(24)])

    assert [len(batch) for batch in calls] == [10, 10, 4]
    assert vectors == [[float(index)] for index in range(24)]


@pytest.mark.asyncio
async def test_retrieval_hydrates_vector_only_hit_and_filters_archived(embeddings):
    from app.harness.agents.retrieval_agent import RetrievalAgent

    store = MemoryStore()
    bm25 = BM25Index()
    vs = MemoryVectorStore()
    hybrid = HybridRetriever(bm25=bm25, vector_store=vs, reranker=HeuristicReranker(), top_k=5)
    agent = RetrievalAgent(hybrid, embeddings, store)
    for doc_id, status in (("active-doc", "active"), ("old-doc", "archived")):
        await store.insert_document({"_id": doc_id, "dept_id": "dept_jwc", "title": doc_id, "status": status})
        chunk = {
            "_id": f"{doc_id}:0", "doc_id": doc_id, "dept_id": "dept_jwc",
            "chunk_index": 0, "content": "研究生开题报告不少于五千字", "keywords": ["开题"],
            "section_path": [], "section_title": "要求",
        }
        await store.insert_chunks([chunk])
        vector = await embeddings.embed_query(chunk["content"])
        await vs.add(chunk["_id"], vector, {"doc_id": doc_id, "dept_id": "dept_jwc", "chunk_index": 0})
    hits = await agent.retrieve(["研究生开题报告不少于五千字"], ["dept_jwc"], top_k=5)
    assert len(hits) == 1
    assert hits[0]["doc_id"] == "active-doc"
    assert hits[0]["content"]


@pytest.mark.asyncio
async def test_retrieval_keeps_primary_hits_and_appends_bounded_graph_evidence():
    from app.harness.agents.retrieval_agent import RetrievalAgent
    from app.graph.expander import GraphExpansionResult

    class Embeddings:
        async def embed_query(self, _query):
            return [0.0]

    class Hybrid:
        reranker = HeuristicReranker()

        async def retrieve(self, _query, _vector, dept_id=None):
            assert dept_id == "dept_jwc"
            return [{"id": "doc_primary:0", "rerank_score": 0.9}]

    class GraphExpander:
        async def expand(self, seeds, query="", candidate_limit=None):
            assert candidate_limit == 20
            chunks = [{
                "id": "doc_graph:0", "doc_id": "doc_graph", "dept_id": "dept_jwc", "chunk_index": 0,
                    "content": "图谱补充证据", "retrieval_source": "graph", "graph_score": 1.0, "score": 0.2,
                "graph_path": {"relationship_keys": ["rel-1"]},
            }] if [seed["id"] for seed in seeds] == ["doc_primary:0"] else []
            return GraphExpansionResult("expanded" if chunks else "no_new_evidence", chunks)

    store = MemoryStore()
    await store.insert_document({"_id": "doc_primary", "status": "active", "title": "主检索文档"})
    await store.insert_chunks([{
        "_id": "doc_primary:0", "doc_id": "doc_primary", "dept_id": "dept_jwc", "chunk_index": 0,
        "content": "主检索证据",
    }])
    agent = RetrievalAgent(Hybrid(), Embeddings(), store, graph_expander=GraphExpander())

    result = await agent.retrieve_with_graph_status(["问题"], ["dept_jwc"], top_k=1)

    assert result.graph_status == "expanded"
    assert [(hit["id"], hit.get("retrieval_source")) for hit in result.chunks] == [
        ("doc_primary:0", None),
        ("doc_graph:0", "graph"),
    ]


@pytest.mark.asyncio
async def test_retrieval_keeps_primary_hits_when_graph_falls_back():
    from app.graph.expander import GraphExpansionResult
    from app.harness.agents.retrieval_agent import RetrievalAgent

    class Embeddings:
        async def embed_query(self, _query):
            return [0.0]

    class Hybrid:
        reranker = HeuristicReranker()

        async def retrieve(self, _query, _vector, dept_id=None):
            return [{"id": "doc_primary:0", "rerank_score": 0.9}]

    class GraphExpander:
        async def expand(self, _seeds, query=""):
            return GraphExpansionResult("fallback_unavailable", [])

    store = MemoryStore()
    await store.insert_document({"_id": "doc_primary", "status": "active", "title": "主检索文档"})
    await store.insert_chunks([{
        "_id": "doc_primary:0", "doc_id": "doc_primary", "dept_id": "dept_jwc", "chunk_index": 0,
        "content": "主检索证据",
    }])
    result = await RetrievalAgent(Hybrid(), Embeddings(), store, graph_expander=GraphExpander()).retrieve_with_graph_status(
        ["问题"], ["dept_jwc"], top_k=1,
    )

    assert result.graph_status == "fallback_unavailable"
    assert [chunk["id"] for chunk in result.chunks] == ["doc_primary:0"]


@pytest.mark.asyncio
async def test_retrieval_content_reranks_graph_candidates_without_reordering_primary_hits():
    """A weak graph path must not bypass the query-to-content reranker."""
    from app.graph.expander import GraphExpansionResult
    from app.harness.agents.retrieval_agent import RetrievalAgent

    class Embeddings:
        async def embed_query(self, _query):
            return [0.0]

    class Reranker:
        async def rerank(self, query, candidates, top_k=5):
            assert query == "研究生开题答辩需要准备什么"
            assert [candidate["id"] for candidate in candidates] == ["doc_graph:noise", "doc_graph:focus", "doc_graph:extra"]
            scores = {"doc_graph:noise": 0.01, "doc_graph:focus": 0.93, "doc_graph:extra": 0.80}
            for candidate in candidates:
                candidate["rerank_score"] = scores[candidate["id"]]
            return [candidates[1], candidates[2]][:top_k]

    class Hybrid:
        reranker = Reranker()

        async def retrieve(self, _query, _vector, dept_id=None):
            return [{"id": "doc_primary:0", "rerank_score": 0.99}]

    class GraphExpander:
        async def expand(self, _seeds, query="", candidate_limit=None):
            assert query == "研究生开题答辩需要准备什么"
            assert candidate_limit == 20
            return GraphExpansionResult("expanded", [
                {"id": "doc_graph:noise", "content": "校园手册总则", "retrieval_source": "graph"},
                {"id": "doc_graph:focus", "content": "开题答辩材料清单", "retrieval_source": "graph"},
                {"id": "doc_graph:extra", "content": "研究生答辩流程", "retrieval_source": "graph"},
            ])

    store = MemoryStore()
    await store.insert_document({"_id": "doc_primary", "status": "active", "title": "主检索文档"})
    await store.insert_chunks([{
        "_id": "doc_primary:0", "doc_id": "doc_primary", "dept_id": "dept_jwc", "chunk_index": 0,
        "content": "主检索证据",
    }])

    result = await RetrievalAgent(Hybrid(), Embeddings(), store, graph_expander=GraphExpander()).retrieve_with_graph_status(
        ["研究生开题答辩需要准备什么"], ["dept_jwc"], top_k=1,
    )

    assert [chunk["id"] for chunk in result.chunks] == ["doc_primary:0", "doc_graph:focus", "doc_graph:extra"]
    assert [chunk["graph_rerank_score"] for chunk in result.chunks[1:]] == [0.93, 0.8]


@pytest.mark.asyncio
async def test_retrieval_returns_no_new_graph_evidence_when_content_gate_rejects_every_candidate():
    """Graph availability is not enough: rejected content must not be sent to the answer agent."""
    from app.graph.expander import GraphExpansionResult
    from app.harness.agents.retrieval_agent import RetrievalAgent

    class Embeddings:
        async def embed_query(self, _query):
            return [0.0]

    class Reranker:
        async def rerank(self, _query, _candidates, top_k=5):
            assert top_k == 2
            return []

    class Hybrid:
        reranker = Reranker()

        async def retrieve(self, _query, _vector, dept_id=None):
            return [{"id": "doc_primary:0", "rerank_score": 0.99}]

    class GraphExpander:
        async def expand(self, _seeds, query="", candidate_limit=None):
            return GraphExpansionResult("expanded", [{
                "id": "doc_graph:noise", "content": "无关内容", "retrieval_source": "graph",
            }])

    store = MemoryStore()
    await store.insert_document({"_id": "doc_primary", "status": "active", "title": "主检索文档"})
    await store.insert_chunks([{
        "_id": "doc_primary:0", "doc_id": "doc_primary", "dept_id": "dept_jwc", "chunk_index": 0,
        "content": "主检索证据",
    }])

    result = await RetrievalAgent(Hybrid(), Embeddings(), store, graph_expander=GraphExpander()).retrieve_with_graph_status(
        ["问题"], ["dept_jwc"], top_k=1,
    )

    assert result.graph_status == "no_new_evidence"
    assert [chunk["id"] for chunk in result.chunks] == ["doc_primary:0"]


@pytest.mark.asyncio
async def test_retrieval_applies_a_minimum_content_relevance_score_to_graph_evidence():
    from app.graph.expander import GraphExpansionResult
    from app.harness.agents.retrieval_agent import RetrievalAgent

    class Embeddings:
        async def embed_query(self, _query):
            return [0.0]

    class Reranker:
        async def rerank(self, _query, candidates, top_k=2):
            for candidate in candidates:
                candidate["rerank_score"] = 0.09 if candidate["id"] == "weak" else 0.11
            return sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)[:top_k]

    class Hybrid:
        reranker = Reranker()

        async def retrieve(self, _query, _vector, dept_id=None):
            return [{"id": "primary", "rerank_score": 0.9}]

    class GraphExpander:
        limit = 2

        async def expand(self, _seeds, query="", candidate_limit=None):
            return GraphExpansionResult("expanded", [
                {"id": "weak", "content": "轻微相关", "retrieval_source": "graph"},
                {"id": "strong", "content": "强相关", "retrieval_source": "graph"},
            ])

    store = MemoryStore()
    await store.insert_document({"_id": "primary", "status": "active", "title": "主文档"})
    await store.insert_chunks([{"_id": "primary", "doc_id": "primary", "dept_id": "dept_jwc", "chunk_index": 0, "content": "主证据"}])

    result = await RetrievalAgent(
        Hybrid(), Embeddings(), store, graph_expander=GraphExpander(), graph_rerank_min_score=0.1,
    ).retrieve_with_graph_status(["问题"], ["dept_jwc"], top_k=1)

    assert [chunk["id"] for chunk in result.chunks] == ["primary", "strong"]
    assert result.chunks[1]["graph_rerank_score"] == 0.11
