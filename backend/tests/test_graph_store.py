"""Neo4j 图谱投影的确定性数据与降级行为。"""
from __future__ import annotations

import pytest

from app.graph.store import GraphStore, build_document_projection


class _AsyncResult:
    def __init__(self, records=None):
        self.records = records or []

    def __aiter__(self):
        async def iterator():
            for record in self.records:
                yield record
        return iterator()

    async def single(self):
        return self.records[0] if self.records else None


class _Session:
    def __init__(self, calls):
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def run(self, query, **params):
        self.calls.append((query, params))
        return _AsyncResult()


class _Driver:
    def __init__(self):
        self.calls = []

    async def verify_connectivity(self):
        return None

    def session(self, **_kwargs):
        return _Session(self.calls)

    async def close(self):
        return None


class _NeighborSession(_Session):
    async def run(self, query, **params):
        self.calls.append((query, params))
        if "RELATES_TO" in query:
            return _AsyncResult([{
                "seed_chunk_id": "doc_1:0", "target_chunk_id": "doc_1:1", "score": 8.0,
                "source_entity_key": "entity:source", "relationship_key": "rel-1", "target_entity_key": "entity:target",
            }])
        return _AsyncResult()


class _NeighborDriver(_Driver):
    def session(self, **_kwargs):
        return _NeighborSession(self.calls)


def test_document_projection_keeps_citable_chunk_and_deduplicates_topics():
    projection = build_document_projection(
        {"_id": "dept_xsc", "name": "学生处"},
        {"_id": "doc_1", "title": "本科生手册", "dept_id": "dept_xsc", "status": "active"},
        [
            {"_id": "doc_1:0", "chunk_index": 0, "section_title": "学籍", "keywords": ["学籍", "处分"]},
            {"_id": "doc_1:1", "chunk_index": 1, "section_title": "处分", "keywords": ["处分", "申诉"]},
        ],
    )

    assert projection["department"] == {"id": "dept_xsc", "name": "学生处"}
    assert [chunk["id"] for chunk in projection["chunks"]] == ["doc_1:0", "doc_1:1"]
    assert projection["topics"] == ["处分", "学籍", "申诉"]


@pytest.mark.asyncio
async def test_disabled_graph_store_returns_safe_empty_results_without_driver():
    graph = GraphStore(enabled=False)

    assert await graph.health() == {"enabled": False, "connected": False}
    assert await graph.neighbor_chunk_ids(["doc_1:0"], limit=2) == []
    assert await graph.summary() == {"enabled": False, "connected": False, "nodes": {}, "edges": 0}


@pytest.mark.asyncio
async def test_graph_store_visualization_returns_bounded_nodes_and_edges():
    class ViewSession(_Session):
        async def run(self, query, **params):
            self.calls.append((query, params))
            if "RETURN elementId(n)" in query:
                return _AsyncResult([{
                    "id": "n1", "label": "Entity", "name": "教务处", "properties": {"type": "department"},
                }, {
                    "id": "n2", "label": "Entity", "name": "选课", "properties": {"type": "procedure"},
                }])
            return _AsyncResult([{"source": "n1", "target": "n2", "type": "RELATES_TO", "key": "r1", "properties": {"weight": 2.0}}])

    class ViewDriver(_Driver):
        def session(self, **_kwargs):
            return ViewSession(self.calls)

    graph = GraphStore(enabled=True, driver=ViewDriver())
    result = await graph.visualization(query="选课", label="Entity", node_limit=2, edge_limit=3)

    assert result["connected"] is True
    assert result["nodes"][0]["name"] == "教务处"
    assert result["edges"] == [{"source": "n1", "target": "n2", "type": "RELATES_TO", "key": "r1", "properties": {"weight": 2.0}}]
    assert any(params["search_query"] == "选课" and params["label"] == "Entity" for query, params in graph._driver.calls if "elementId(n)" in query)


@pytest.mark.asyncio
async def test_disabled_graph_store_visualization_is_explicitly_unavailable():
    result = await GraphStore(enabled=False).visualization()
    assert result == {"enabled": False, "connected": False, "nodes": [], "edges": [], "message": "图谱服务不可用"}


@pytest.mark.asyncio
async def test_graph_store_imports_official_projection_with_chunk_evidence():
    driver = _Driver()
    graph = GraphStore(enabled=True, driver=driver)
    projection = {
        "text_units": [{"id": "tu-1", "chunk_id": "doc_1:0"}],
        "entities": [{
            "id": "entity-1", "key": "department:学生处", "name": "学生处", "type": "DEPARTMENT",
            "description": "学生事务部门", "text_unit_ids": ["tu-1"],
        }],
        "relationships": [{
            "id": "rel-1", "key": "department:学生处|policy:学生管理规定|执行", "source_key": "department:学生处",
            "target_key": "policy:学生管理规定", "description": "执行", "weight": 9.0, "text_unit_ids": ["tu-1"],
        }],
        "communities": [],
        "community_reports": [],
    }

    assert await graph.import_graphrag_projection(projection) is True

    payloads = [params.get("rows") for _query, params in driver.calls if "rows" in params]
    assert projection["text_units"] in payloads
    assert projection["entities"] in payloads
    assert any("HAS_TEXT_UNIT" in query for query, _params in driver.calls)
    assert any("MENTIONS" in query for query, _params in driver.calls)
    assert any("RELATES_TO" in query and "relationship.text_unit_ids" in query for query, _params in driver.calls)


@pytest.mark.asyncio
async def test_graph_store_prefers_official_entity_path_for_neighbor_chunks():
    driver = _NeighborDriver()
    graph = GraphStore(enabled=True, driver=driver)

    result = await graph.neighbor_chunks_with_paths(["doc_1:0"], limit=2)

    assert result == {
        "unavailable": False,
        "paths": [{
            "seed_chunk_id": "doc_1:0", "target_chunk_id": "doc_1:1", "path_type": "entity_relation",
            "entity_keys": ["entity:source", "entity:target"], "relationship_keys": ["rel-1"], "score": 8.0,
            "query_match_count": 0, "direct_entity_match_count": 0, "source_degree": 0, "target_degree": 0,
        }],
    }
    assert await graph.neighbor_chunk_ids(["doc_1:0"], limit=2) == ["doc_1:1"]

    assert any("HAS_TEXT_UNIT" in query and "RELATES_TO" in query for query, _params in driver.calls)
    assert any("UNWIND relationship.text_unit_ids" in query for query, _params in driver.calls)


@pytest.mark.asyncio
async def test_graph_store_passes_query_terms_and_degree_aware_ranking_to_neo4j():
    driver = _NeighborDriver()
    graph = GraphStore(enabled=True, driver=driver)

    result = await graph.neighbor_chunks_with_paths(["doc_1:0"], limit=10, query_terms=["开题", "答辩"])

    assert result["unavailable"] is False

    query, params = next((query, params) for query, params in driver.calls if "RELATES_TO" in query)
    assert params["query_terms"] == ["开题", "答辩"]
    assert "source_degree" in query
    assert "target_degree" in query
    assert "query_match_count" in query


@pytest.mark.asyncio
async def test_graph_store_skips_global_entity_lookup_when_seed_path_has_direct_match():
    class DirectSeedSession(_Session):
        async def run(self, query, **params):
            self.calls.append((query, params))
            if "RELATES_TO" in query:
                return _AsyncResult([{
                    "seed_chunk_id": "doc_1:0", "target_chunk_id": "doc_1:1", "score": 8.0,
                    "source_entity_key": "entity:source", "relationship_key": "rel-1",
                    "target_entity_key": "entity:target", "query_match_count": 1,
                    "direct_entity_match_count": 1,
                }])
            return _AsyncResult()

    class DirectSeedDriver(_Driver):
        def session(self, **_kwargs):
            return DirectSeedSession(self.calls)

    driver = DirectSeedDriver()
    graph = GraphStore(enabled=True, driver=driver)

    result = await graph.neighbor_chunks_with_paths(["doc_1:0"], limit=2, query_terms=["目标实体"])

    assert result["unavailable"] is False
    assert len([query for query, _params in driver.calls if "RELATES_TO" in query]) == 1
