"""图增强检索的去重、状态过滤与降级。"""
from __future__ import annotations

import pytest

from app.graph.expander import GraphExpander, graph_query_terms
from app.storage.store import MemoryStore


class _Graph:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids

    async def neighbor_chunks_with_paths(self, seed_ids: list[str], limit: int, _query_terms: list[str]) -> dict:
        return {
            "unavailable": False,
            "paths": [
                {
                    "seed_chunk_id": seed_ids[0], "target_chunk_id": chunk_id,
                    "path_type": "entity_relation", "entity_keys": ["entity-a", "entity-b"],
                    "relationship_keys": ["rel-a-b"], "score": 2.0,
                }
                for chunk_id in self.ids[:limit]
            ],
        }


@pytest.mark.asyncio
async def test_graph_expander_returns_only_active_non_seed_chunks_in_graph_order():
    store = MemoryStore()
    await store.insert_document({"_id": "doc_active", "status": "active"})
    await store.insert_document({"_id": "doc_archived", "status": "archived"})
    await store.insert_chunks([
        {"_id": "doc_active:0", "doc_id": "doc_active", "dept_id": "dept_xsc", "chunk_index": 0, "content": "seed"},
        {"_id": "doc_active:1", "doc_id": "doc_active", "dept_id": "dept_xsc", "chunk_index": 1, "content": "related"},
        {"_id": "doc_archived:0", "doc_id": "doc_archived", "dept_id": "dept_xsc", "chunk_index": 0, "content": "old"},
    ])
    expander = GraphExpander(_Graph(["doc_active:0", "doc_active:1", "doc_archived:0"]), store, limit=2)

    result = await expander.expand([{"id": "doc_active:0"}])

    assert result.status == "expanded"
    assert result.chunks == [{
        "_id": "doc_active:1", "id": "doc_active:1", "doc_id": "doc_active", "dept_id": "dept_xsc",
        "chunk_index": 1, "content": "related", "graph_score": 2.0, "retrieval_source": "graph",
        "graph_path": {
            "seed_chunk_id": "doc_active:0", "target_chunk_id": "doc_active:1", "path_type": "entity_relation",
            "entity_keys": ["entity-a", "entity-b"], "relationship_keys": ["rel-a-b"],
        },
    }]


@pytest.mark.asyncio
async def test_graph_expander_can_expose_a_larger_candidate_pool_for_the_content_gate():
    """A retrieval-side gate must see more paths than the final answer limit."""
    class CandidateGraph:
        async def neighbor_chunks_with_paths(self, seed_ids, limit, _query_terms):
            return {"unavailable": False, "paths": [
                {
                    "seed_chunk_id": seed_ids[0], "target_chunk_id": f"doc:{index}",
                    "path_type": "entity_relation", "entity_keys": ["entity:seed", f"entity:{index}"],
                    "relationship_keys": [f"rel:{index}"], "score": float(4 - index),
                }
                for index in range(1, limit + 1)
            ]}

    store = MemoryStore()
    await store.insert_document({"_id": "doc", "status": "active"})
    await store.insert_chunks([
        {"_id": "doc:1", "doc_id": "doc", "chunk_index": 1, "content": "候选一"},
        {"_id": "doc:2", "doc_id": "doc", "chunk_index": 2, "content": "候选二"},
        {"_id": "doc:3", "doc_id": "doc", "chunk_index": 3, "content": "候选三"},
    ])

    result = await GraphExpander(CandidateGraph(), store, limit=1).expand(
        [{"id": "doc:0"}], candidate_limit=3,
    )

    assert [chunk["id"] for chunk in result.chunks] == ["doc:1", "doc:2", "doc:3"]


def test_graph_query_terms_prioritize_complete_domain_phrases_over_arbitrary_ngrams():
    terms = graph_query_terms("研究生开题答辩需要准备什么")

    assert terms[:2] == ["研究生开题答辩", "开题答辩"]
    assert "学生" not in terms


def test_graph_query_terms_keep_quoted_entities_before_sentence_ngrams():
    terms = graph_query_terms(
        "请结合东华大学研究生与本科生学生手册，说明“学生申诉”与“学校所在地省级教育行政部门”在相关制度中的职责衔接。"
    )

    assert terms[:2] == ["学生申诉", "学校所在地省级教育行政部门"]


@pytest.mark.asyncio
async def test_graph_expander_rejects_high_degree_generic_path_without_specific_query_match():
    class NoisyGraph:
        enabled = True

        async def neighbor_chunks_with_paths(self, _seed_ids, _limit, _query_terms):
            return {"unavailable": False, "paths": [
                {
                    "seed_chunk_id": "doc:0", "target_chunk_id": "doc:1", "path_type": "entity_relation",
                    "entity_keys": ["organization:学校", "policy:学生手册"], "relationship_keys": ["generic"],
                    "score": 100.0, "query_match_count": 0, "source_degree": 500, "target_degree": 500,
                },
            ]}

    store = MemoryStore()
    await store.insert_document({"_id": "doc", "status": "active"})
    await store.insert_chunks([{"_id": "doc:1", "doc_id": "doc", "chunk_index": 1, "content": "学校总则"}])

    result = await GraphExpander(NoisyGraph(), store, limit=1).expand(
        [{"id": "doc:0"}], query="开题答辩流程",
    )

    assert result.status == "no_new_evidence"
    assert result.chunks == []


@pytest.mark.asyncio
async def test_graph_expander_returns_empty_when_seed_is_empty():
    result = await GraphExpander(_Graph(["doc:1"]), MemoryStore(), limit=2).expand([])
    assert result.status == "no_new_evidence"
    assert result.chunks == []


@pytest.mark.asyncio
async def test_graph_expander_marks_graph_store_unavailable_instead_of_empty_success():
    class _UnavailableGraph:
        async def neighbor_chunks_with_paths(self, _seed_ids: list[str], _limit: int, _query_terms: list[str]) -> dict:
            return {"unavailable": True, "paths": []}

    result = await GraphExpander(_UnavailableGraph(), MemoryStore(), limit=2).expand([{"id": "doc:0"}])

    assert result.status == "fallback_unavailable"
    assert result.chunks == []


@pytest.mark.asyncio
async def test_graph_expander_prioritizes_query_relevant_specific_path_over_generic_high_degree_path():
    class RankedGraph:
        enabled = True

        async def neighbor_chunks_with_paths(self, _seed_ids: list[str], _limit: int, query_terms: list[str]) -> dict:
            assert "开题" in query_terms
            return {
                "unavailable": False,
                "paths": [
                    {
                        "seed_chunk_id": "doc:0", "target_chunk_id": "doc:1", "path_type": "entity_relation",
                        "entity_keys": ["organization:东华大学", "policy:学生手册"], "relationship_keys": ["generic"],
                        "score": 100.0, "query_match_count": 0, "source_degree": 500, "target_degree": 500,
                    },
                    {
                        "seed_chunk_id": "doc:0", "target_chunk_id": "doc:2", "path_type": "entity_relation",
                        "entity_keys": ["process:开题答辩", "role:研究生"], "relationship_keys": ["specific"],
                        "score": 2.0, "query_match_count": 2, "source_degree": 2, "target_degree": 3,
                    },
                ],
            }

    store = MemoryStore()
    await store.insert_document({"_id": "doc", "status": "active"})
    await store.insert_chunks([
        {"_id": "doc:1", "doc_id": "doc", "chunk_index": 1, "content": "泛化内容"},
        {"_id": "doc:2", "doc_id": "doc", "chunk_index": 2, "content": "开题答辩流程"},
    ])

    result = await GraphExpander(RankedGraph(), store, limit=1).expand(
        [{"id": "doc:0"}], query="研究生开题答辩需要准备什么？",
    )

    assert [chunk["id"] for chunk in result.chunks] == ["doc:2"]
    assert result.chunks[0]["graph_path"]["relationship_keys"] == ["specific"]


@pytest.mark.asyncio
async def test_graph_expander_filters_generic_handbook_path_even_when_its_text_matches_more_query_terms():
    class GenericDominatesGraph:
        enabled = True

        async def neighbor_chunks_with_paths(self, _seed_ids, _limit, _query_terms):
            return {"unavailable": False, "paths": [
                {
                    "seed_chunk_id": "doc:0", "target_chunk_id": "doc:1", "path_type": "entity_relation",
                    "entity_keys": ["policy:研究生学生手册", "organization:学生社团"], "relationship_keys": ["generic"],
                    "score": 9.0, "query_match_count": 4, "source_degree": 25, "target_degree": 8,
                },
                {
                    "seed_chunk_id": "doc:0", "target_chunk_id": "doc:2", "path_type": "entity_relation",
                    "entity_keys": ["process:开题答辩", "role:研究生"], "relationship_keys": ["specific"],
                    "score": 2.0, "query_match_count": 1, "source_degree": 2, "target_degree": 3,
                },
            ]}

    store = MemoryStore()
    await store.insert_document({"_id": "doc", "status": "active"})
    await store.insert_chunks([
        {"_id": "doc:1", "doc_id": "doc", "chunk_index": 1, "content": "学生手册"},
        {"_id": "doc:2", "doc_id": "doc", "chunk_index": 2, "content": "开题答辩"},
    ])

    result = await GraphExpander(GenericDominatesGraph(), store, limit=1).expand([{"id": "doc:0"}], query="开题答辩流程")

    assert [chunk["id"] for chunk in result.chunks] == ["doc:2"]


@pytest.mark.asyncio
async def test_graph_expander_requires_direct_entity_overlap_with_query_terms():
    class IndirectOnlyGraph:
        enabled = True

        async def neighbor_chunks_with_paths(self, _seed_ids, _limit, _query_terms):
            return {"unavailable": False, "paths": [
                {
                    "seed_chunk_id": "doc:0", "target_chunk_id": "doc:1", "path_type": "entity_relation",
                    "entity_keys": ["organization:东华大学", "policy:学生宿舍管理办法"],
                    "relationship_keys": ["generic"], "score": 99.0, "query_match_count": 3,
                },
            ]}

    store = MemoryStore()
    await store.insert_document({"_id": "doc", "status": "active"})
    await store.insert_chunks([{"_id": "doc:1", "doc_id": "doc", "chunk_index": 1, "content": "宿舍"}])

    result = await GraphExpander(IndirectOnlyGraph(), store, limit=1).expand(
        [{"id": "doc:0"}], query="研究生开题答辩流程",
    )

    assert result.status == "no_new_evidence"
    assert result.chunks == []


@pytest.mark.asyncio
async def test_graph_expander_rejects_university_to_policy_generic_bridge_even_when_university_matches_query():
    class UniversityPolicyGraph:
        enabled = True

        async def neighbor_chunks_with_paths(self, _seed_ids, _limit, _query_terms):
            return {"unavailable": False, "paths": [
                {
                    "seed_chunk_id": "doc:0", "target_chunk_id": "doc:1", "path_type": "entity_relation",
                    "entity_keys": ["organization:东华大学", "policy:东华大学学生宿舍管理办法"],
                    "relationship_keys": ["generic"], "score": 99.0, "query_match_count": 1,
                },
            ]}

    store = MemoryStore()
    await store.insert_document({"_id": "doc", "status": "active"})
    await store.insert_chunks([{"_id": "doc:1", "doc_id": "doc", "chunk_index": 1, "content": "宿舍"}])

    result = await GraphExpander(UniversityPolicyGraph(), store, limit=1).expand(
        [{"id": "doc:0"}], query="东华大学研究生开题答辩流程",
    )

    assert result.status == "no_new_evidence"
    assert result.chunks == []


@pytest.mark.asyncio
async def test_graph_expander_keeps_distinct_evidence_for_the_same_relationship():
    class DuplicateRelationGraph:
        enabled = True

        async def neighbor_chunks_with_paths(self, _seed_ids, _limit, _query_terms):
            return {"unavailable": False, "paths": [
                {
                    "seed_chunk_id": "doc:0", "target_chunk_id": "doc:1", "path_type": "entity_relation",
                    "entity_keys": ["procedure:开题答辩", "department:研究生院"],
                    "relationship_keys": ["same-relation"], "score": 9.0, "query_match_count": 2,
                },
                {
                    "seed_chunk_id": "doc:0", "target_chunk_id": "doc:2", "path_type": "entity_relation",
                    "entity_keys": ["procedure:开题答辩", "department:教务处"],
                    "relationship_keys": ["same-relation"], "score": 5.0, "query_match_count": 2,
                },
            ]}

    store = MemoryStore()
    await store.insert_document({"_id": "doc", "status": "active"})
    await store.insert_chunks([
        {"_id": "doc:1", "doc_id": "doc", "chunk_index": 1, "content": "最优关系证据"},
        {"_id": "doc:2", "doc_id": "doc", "chunk_index": 2, "content": "重复关系证据"},
    ])

    result = await GraphExpander(DuplicateRelationGraph(), store, limit=2).expand(
        [{"id": "doc:0"}], query="开题答辩",
    )

    assert [chunk["id"] for chunk in result.chunks] == ["doc:1", "doc:2"]


@pytest.mark.asyncio
async def test_graph_expander_keeps_low_score_relation_when_other_path_filters_pass():
    class LowWeightGraph:
        enabled = True

        async def neighbor_chunks_with_paths(self, _seed_ids, _limit, _query_terms):
            return {"unavailable": False, "paths": [{
                "seed_chunk_id": "doc:0", "target_chunk_id": "doc:1", "path_type": "entity_relation",
                "entity_keys": ["procedure:开题答辩", "department:研究生院"],
                "relationship_keys": ["weak-relation"], "score": 7.0, "query_match_count": 2,
            }]}

    store = MemoryStore()
    await store.insert_document({"_id": "doc", "status": "active"})
    await store.insert_chunks([{"_id": "doc:1", "doc_id": "doc", "chunk_index": 1, "content": "低权重关系"}])

    result = await GraphExpander(LowWeightGraph(), store, limit=1).expand(
        [{"id": "doc:0"}], query="开题答辩",
    )

    assert result.status == "expanded"
    assert [chunk["id"] for chunk in result.chunks] == ["doc:1"]
