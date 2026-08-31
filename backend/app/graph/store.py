"""MongoDB 知识库到 Neo4j 的可降级图谱投影。"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)


def build_document_projection(
    department: dict[str, Any], document: dict[str, Any], chunks: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """将 Mongo 文档转为无 LLM、可重复写入的图谱 payload。"""
    projected_chunks = [
        {
            "id": str(chunk["_id"]),
            "index": int(chunk.get("chunk_index", 0)),
            "section_title": str(chunk.get("section_title", "")),
            "topics": sorted({str(topic).strip() for topic in chunk.get("keywords", []) if str(topic).strip()}),
        }
        for chunk in chunks
    ]
    projected_chunks.sort(key=lambda chunk: (chunk["index"], chunk["id"]))
    return {
        "department": {"id": str(department.get("_id", "")), "name": str(department.get("name", ""))},
        "document": {
            "id": str(document.get("_id", "")),
            "title": str(document.get("title", "")),
            "dept_id": str(document.get("dept_id", "")),
            "status": str(document.get("status", "")),
        },
        "chunks": projected_chunks,
        "topics": sorted({topic for chunk in projected_chunks for topic in chunk["topics"]}),
    }


class GraphStore:
    """Neo4j 投影；不可用时所有调用安全地退化为空结果。"""

    def __init__(
        self,
        enabled: bool,
        uri: str = "",
        username: str = "",
        password: str = "",
        database: str = "neo4j",
        driver=None,
    ) -> None:
        self.enabled = enabled
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self._driver = driver
        self._connected = False

    async def connect(self) -> bool:
        if not self.enabled:
            return False
        if self._connected and self._driver is not None:
            return True
        try:
            if self._driver is None:
                from neo4j import AsyncGraphDatabase

                self._driver = AsyncGraphDatabase.driver(self.uri, auth=(self.username, self.password))
            await self._driver.verify_connectivity()
            await self._ensure_schema()
            self._connected = True
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Neo4j 不可用，图增强检索已降级: %s", exc)
            self._connected = False
            if self._driver is not None:
                try:
                    await self._driver.close()
                except Exception:  # noqa: BLE001
                    pass
                self._driver = None
            return False

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
        self._driver = None
        self._connected = False

    async def health(self) -> dict[str, bool]:
        connected = await self.connect()
        return {"enabled": self.enabled, "connected": connected}

    async def sync_document(
        self, department: dict[str, Any], document: dict[str, Any], chunks: list[dict[str, Any]],
    ) -> bool:
        if not await self.connect():
            return False
        payload = build_document_projection(department, document, chunks)
        chunk_topics = [
            {"chunk_id": chunk["id"], "topics": chunk["topics"]}
            for chunk in payload["chunks"]
        ]
        try:
            async with self._driver.session(database=self.database) as session:
                await session.run(
                    "MERGE (dept:Department {id: $id}) SET dept.name = $name",
                    **payload["department"],
                )
                await session.run(
                    "MERGE (doc:Document {id: $id}) "
                    "SET doc.title = $title, doc.status = $status, doc.dept_id = $dept_id "
                    "WITH doc MATCH (dept:Department {id: $dept_id}) MERGE (dept)-[:OWNS]->(doc)",
                    **payload["document"],
                )
                await session.run(
                    "MATCH (doc:Document {id: $doc_id}) "
                    "UNWIND $chunks AS row "
                    "MERGE (chunk:Chunk {id: row.id}) "
                    "SET chunk.index = row.index, chunk.section_title = row.section_title "
                    "MERGE (doc)-[:CONTAINS]->(chunk)",
                    doc_id=payload["document"]["id"], chunks=payload["chunks"],
                )
                await session.run(
                    "UNWIND $rows AS row "
                    "MATCH (chunk:Chunk {id: row.chunk_id}) "
                    "UNWIND row.topics AS name "
                    "MERGE (topic:Topic {name: name}) "
                    "MERGE (chunk)-[:MENTIONS]->(topic)",
                    rows=chunk_topics,
                )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Neo4j 文档同步失败(%s): %s", document.get("title", ""), exc)
            self._connected = False
            return False

    async def import_graphrag_projection(self, projection: dict[str, list[dict[str, Any]]]) -> bool:
        """Idempotently project official GraphRAG artifacts without changing Mongo records."""
        if not await self.connect():
            return False
        rows = {name: projection.get(name, []) for name in (
            "text_units", "entities", "relationships", "communities", "community_reports",
        )}
        try:
            async with self._driver.session(database=self.database) as session:
                await session.run(
                    "UNWIND $rows AS row "
                    "MERGE (chunk:Chunk {id: row.chunk_id}) "
                    "MERGE (unit:TextUnit {id: row.id}) "
                    "SET unit.chunk_id = row.chunk_id "
                    "MERGE (chunk)-[:HAS_TEXT_UNIT]->(unit)",
                    rows=rows["text_units"],
                )
                await session.run(
                    "UNWIND $rows AS row "
                    "MERGE (entity:Entity {key: row.key}) "
                    "SET entity.name = row.name, entity.type = row.type, entity.description = row.description",
                    rows=rows["entities"],
                )
                await session.run(
                    "UNWIND $rows AS row "
                    "MATCH (entity:Entity {key: row.key}) "
                    "UNWIND row.text_unit_ids AS text_unit_id "
                    "MATCH (unit:TextUnit {id: text_unit_id}) "
                    "MERGE (unit)-[:MENTIONS]->(entity)",
                    rows=rows["entities"],
                )
                await session.run(
                    "UNWIND $rows AS row "
                    "MATCH (source:Entity {key: row.source_key}) "
                    "MATCH (target:Entity {key: row.target_key}) "
                    "MERGE (source)-[relationship:RELATES_TO {key: row.key}]->(target) "
                    "SET relationship.description = row.description, relationship.weight = row.weight, "
                    "relationship.text_unit_ids = row.text_unit_ids",
                    rows=rows["relationships"],
                )
                await session.run(
                    "UNWIND $rows AS row "
                    "MERGE (community:Community {key: row.key}) "
                    "SET community.title = row.title, community.level = row.level, community.community = row.community "
                    "WITH community, row UNWIND row.entity_keys AS entity_key "
                    "MATCH (entity:Entity {key: entity_key}) "
                    "MERGE (entity)-[:IN_COMMUNITY]->(community)",
                    rows=rows["communities"],
                )
                await session.run(
                    "UNWIND $rows AS row "
                    "MATCH (community:Community {key: row.community_key}) "
                    "MERGE (report:CommunityReport {community_key: row.community_key}) "
                    "SET report.title = row.title, report.summary = row.summary, report.full_content = row.full_content "
                    "MERGE (community)-[:HAS_REPORT]->(report)",
                    rows=rows["community_reports"],
                )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Neo4j 官方 GraphRAG 产物导入失败: %s", exc)
            self._connected = False
            return False

    async def delete_document(self, doc_id: str) -> bool:
        if not await self.connect():
            return False
        try:
            async with self._driver.session(database=self.database) as session:
                await session.run(
                    "MATCH (doc:Document {id: $doc_id}) "
                    "OPTIONAL MATCH (doc)-[:CONTAINS]->(chunk:Chunk) "
                    "DETACH DELETE doc, chunk",
                    doc_id=doc_id,
                )
                await session.run("MATCH (topic:Topic) WHERE NOT (topic)<-[:MENTIONS]-() DELETE topic")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Neo4j 文档删除同步失败(%s): %s", doc_id, exc)
            return False

    async def neighbor_chunks_with_paths(
        self, seed_ids: list[str], limit: int = 2, query_terms: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return bounded graph neighbors plus the exact traversal provenance.

        ``unavailable`` is deliberately distinct from an available graph that
        simply has no new evidence.  Callers can therefore tell users when a
        normal-RAG fallback was used instead of claiming graph enhancement.
        """
        if not seed_ids or limit <= 0:
            return {"unavailable": False, "paths": []}
        if not await self.connect():
            return {"unavailable": True, "paths": []}
        try:
            async with self._driver.session(database=self.database) as session:
                normalized_terms = [term.lower() for term in (query_terms or []) if term]
                anchor_terms = [
                    term for term in normalized_terms
                    if term not in {"学校", "大学", "东华大学", "学生手册", "本科生手册", "研究生手册"}
                    and "学生手册" not in term
                ]
                # Use the edge's own evidence units instead of every chunk
                # mentioning the target entity; this keeps paths citable.
                entity_result = await session.run(
                    "MATCH (seed:Chunk) WHERE seed.id IN $seed_ids "
                    "MATCH (seed)-[:HAS_TEXT_UNIT]->(:TextUnit)-[:MENTIONS]->(source:Entity) "
                    "MATCH (source)-[relationship:RELATES_TO]-(target:Entity) "
                    "UNWIND relationship.text_unit_ids AS relationship_text_unit_id "
                    "MATCH (relationship_unit:TextUnit {id: relationship_text_unit_id}) "
                    "MATCH (relationship_unit)<-[:HAS_TEXT_UNIT]-(neighbor:Chunk) "
                    "WHERE NOT neighbor.id IN $seed_ids "
                    "CALL (source) { MATCH (:TextUnit)-[:MENTIONS]->(source) RETURN count(*) AS source_degree } "
                    "CALL (target) { MATCH (:TextUnit)-[:MENTIONS]->(target) RETURN count(*) AS target_degree } "
                    "WITH seed, neighbor, source, relationship, target, source_degree, target_degree, "
                    "[term IN $query_terms WHERE toLower(coalesce(source.name, '') + ' ' + "
                    "coalesce(source.description, '') + ' ' + coalesce(relationship.description, '') + ' ' + "
                    "coalesce(target.name, '') + ' ' + coalesce(target.description, '')) CONTAINS term] AS matches, "
                    "[term IN $anchor_terms WHERE toLower(coalesce(source.name, '')) CONTAINS term "
                    "OR toLower(coalesce(target.name, '')) CONTAINS term] AS direct_matches "
                    "RETURN seed.id AS seed_chunk_id, neighbor.id AS target_chunk_id, "
                    "source.key AS source_entity_key, relationship.key AS relationship_key, "
                    "target.key AS target_entity_key, coalesce(relationship.weight, 1.0) AS score, "
                    "size(matches) AS query_match_count, size(direct_matches) AS direct_entity_match_count, "
                    "source_degree, target_degree "
                    "ORDER BY query_match_count DESC, "
                    "score / sqrt(toFloat(1 + source_degree) * toFloat(1 + target_degree)) DESC, target_chunk_id ASC LIMIT $limit",
                    seed_ids=seed_ids, limit=limit, query_terms=normalized_terms, anchor_terms=anchor_terms,
                )
                entity_paths = [{
                    "seed_chunk_id": str(record["seed_chunk_id"]),
                    "target_chunk_id": str(record["target_chunk_id"]),
                    "path_type": "entity_relation",
                    "entity_keys": [str(record["source_entity_key"]), str(record["target_entity_key"])],
                    "relationship_keys": [str(record["relationship_key"])],
                    "score": float(record["score"]),
                    "query_match_count": int(record.get("query_match_count", 0) or 0),
                    "source_degree": int(record.get("source_degree", 0) or 0),
                    "target_degree": int(record.get("target_degree", 0) or 0),
                    "direct_entity_match_count": int(record.get("direct_entity_match_count", 0) or 0),
                } async for record in entity_result]
                # If the initial seed chunks do not mention the queried
                # entities, a seed-only traversal cannot reach the relevant
                # bridge.  Anchor a second bounded lookup on entity names and
                # use the relationship's own evidence units as citable chunks.
                seed_has_direct_match = any(
                    int(path.get("direct_entity_match_count", 0) or 0) > 0
                    for path in entity_paths
                )
                if anchor_terms and not seed_has_direct_match:
                    query_entity_result = await session.run(
                        "MATCH (source:Entity)-[relationship:RELATES_TO]-(target:Entity) "
                        "WITH source, relationship, target, "
                        "[term IN $anchor_terms WHERE toLower(coalesce(source.name, '')) CONTAINS term "
                        "OR toLower(coalesce(target.name, '')) CONTAINS term] AS matches "
                        "WHERE size(matches) > 0 "
                        "UNWIND relationship.text_unit_ids AS relationship_text_unit_id "
                        "MATCH (relationship_unit:TextUnit {id: relationship_text_unit_id}) "
                        "MATCH (relationship_unit)<-[:HAS_TEXT_UNIT]-(neighbor:Chunk) "
                        "WHERE NOT neighbor.id IN $seed_ids "
                        "RETURN $seed_ids[0] AS seed_chunk_id, neighbor.id AS target_chunk_id, "
                        "source.key AS source_entity_key, relationship.key AS relationship_key, "
                        "target.key AS target_entity_key, coalesce(relationship.weight, 1.0) AS score, "
                        "size(matches) AS query_match_count, 0 AS source_degree, 0 AS target_degree "
                        "ORDER BY query_match_count DESC, score DESC, target_chunk_id ASC LIMIT $global_limit",
                        seed_ids=seed_ids, limit=limit, global_limit=max(limit * 5, limit), anchor_terms=anchor_terms,
                    )
                    entity_paths.extend([{
                        "seed_chunk_id": str(record["seed_chunk_id"]),
                        "target_chunk_id": str(record["target_chunk_id"]),
                        "path_type": "entity_relation_query",
                        "entity_keys": [str(record["source_entity_key"]), str(record["target_entity_key"])],
                        "relationship_keys": [str(record["relationship_key"])],
                        "score": float(record["score"]),
                        "query_match_count": int(record.get("query_match_count", 0) or 0),
                        "source_degree": int(record.get("source_degree", 0) or 0),
                        "target_degree": int(record.get("target_degree", 0) or 0),
                    } async for record in query_entity_result])
                if entity_paths:
                    return {"unavailable": False, "paths": entity_paths}
                result = await session.run(
                    "MATCH (seed:Chunk) WHERE seed.id IN $seed_ids "
                    "MATCH (seed)-[:MENTIONS]->(:Topic)<-[:MENTIONS]-(neighbor:Chunk) "
                    "WHERE NOT neighbor.id IN $seed_ids "
                    "RETURN seed.id AS seed_chunk_id, neighbor.id AS target_chunk_id, count(*) AS score "
                    "ORDER BY score DESC, target_chunk_id ASC LIMIT $limit",
                    seed_ids=seed_ids, limit=limit,
                )
                topic_paths = [{
                    "seed_chunk_id": str(record["seed_chunk_id"]),
                    "target_chunk_id": str(record["target_chunk_id"]),
                    "path_type": "topic_fallback",
                    "entity_keys": [],
                    "relationship_keys": [],
                    "score": float(record["score"]),
                } async for record in result]
                return {"unavailable": False, "paths": topic_paths}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Neo4j 邻居查询失败，使用普通 RAG: %s", exc)
            self._connected = False
            return {"unavailable": True, "paths": []}

    async def neighbor_chunk_ids(self, seed_ids: list[str], limit: int = 2) -> list[str]:
        """Backward-compatible id-only view for legacy callers."""
        result = await self.neighbor_chunks_with_paths(seed_ids, limit)
        return [str(path["target_chunk_id"]) for path in result["paths"]]

    async def summary(self) -> dict[str, Any]:
        if not await self.connect():
            return {"enabled": self.enabled, "connected": False, "nodes": {}, "edges": 0}
        try:
            async with self._driver.session(database=self.database) as session:
                node_result = await session.run(
                    "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY label"
                )
                nodes = {record["label"]: record["count"] async for record in node_result}
                edge_result = await session.run("MATCH ()-[r]->() RETURN count(r) AS count")
                edge = await edge_result.single()
                return {"enabled": True, "connected": True, "nodes": nodes, "edges": edge["count"] if edge else 0}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Neo4j 摘要查询失败: %s", exc)
            return {"enabled": True, "connected": False, "nodes": {}, "edges": 0}

    async def visualization(
        self, query: str = "", label: str = "", node_limit: int = 80, edge_limit: int = 180,
    ) -> dict[str, Any]:
        """Return a small, read-only subgraph for the admin visualization.

        This intentionally keeps the payload bounded: the UI is for inspecting
        graph evidence, not exporting the entire Neo4j database.
        """
        node_limit = max(1, min(int(node_limit), 200))
        edge_limit = max(0, min(int(edge_limit), 500))
        if not await self.connect():
            return {"enabled": self.enabled, "connected": False, "nodes": [], "edges": [], "message": "图谱服务不可用"}
        try:
            async with self._driver.session(database=self.database) as session:
                node_result = await session.run(
                    "MATCH (n) "
                    "WHERE ($label = '' OR $label IN labels(n)) "
                    "AND ($search_query = '' OR any(value IN [coalesce(n.name, ''), coalesce(n.title, ''), "
                    "coalesce(n.key, ''), coalesce(n.id, '')] WHERE toLower(toString(value)) CONTAINS toLower($search_query))) "
                    "WITH n, count { (n)--() } AS degree "
                    "RETURN elementId(n) AS id, labels(n)[0] AS label, "
                    "coalesce(n.name, n.title, n.key, n.id, elementId(n)) AS name, properties(n) AS properties "
                    "ORDER BY degree DESC, name LIMIT $node_limit",
                    search_query=query.strip(), label=label.strip(), node_limit=node_limit,
                )
                nodes = []
                async for record in node_result:
                    nodes.append({
                        "id": str(record["id"]),
                        "label": str(record.get("label") or "Node"),
                        "name": str(record.get("name") or record["id"]),
                        "properties": dict(record.get("properties") or {}),
                    })
                ids = [node["id"] for node in nodes]
                edges = []
                if ids and edge_limit:
                    edge_result = await session.run(
                        "MATCH (a)-[r]-(b) "
                        "WHERE elementId(a) IN $ids AND elementId(b) IN $ids AND elementId(a) < elementId(b) "
                        "RETURN elementId(a) AS source, elementId(b) AS target, type(r) AS type, "
                        "coalesce(r.key, '') AS key, properties(r) AS properties "
                        "ORDER BY type, source, target LIMIT $edge_limit",
                        ids=ids, edge_limit=edge_limit,
                    )
                    async for record in edge_result:
                        edges.append({
                            "source": str(record["source"]),
                            "target": str(record["target"]),
                            "type": str(record.get("type") or "RELATED"),
                            "key": str(record.get("key") or ""),
                            "properties": dict(record.get("properties") or {}),
                        })
                return {"enabled": True, "connected": True, "nodes": nodes, "edges": edges, "message": "ok"}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Neo4j 图谱可视化查询失败: %s", exc)
            self._connected = False
            return {"enabled": True, "connected": False, "nodes": [], "edges": [], "message": "图谱服务不可用"}

    async def _ensure_schema(self) -> None:
        async with self._driver.session(database=self.database) as session:
            for statement in (
                "CREATE CONSTRAINT department_id IF NOT EXISTS FOR (n:Department) REQUIRE n.id IS UNIQUE",
                "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE",
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE",
                "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (n:Topic) REQUIRE n.name IS UNIQUE",
                "CREATE CONSTRAINT text_unit_id IF NOT EXISTS FOR (n:TextUnit) REQUIRE n.id IS UNIQUE",
                "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (n:Entity) REQUIRE n.key IS UNIQUE",
                "CREATE CONSTRAINT community_key IF NOT EXISTS FOR (n:Community) REQUIRE n.key IS UNIQUE",
                "CREATE CONSTRAINT community_report_key IF NOT EXISTS FOR (n:CommunityReport) REQUIRE n.community_key IS UNIQUE",
            ):
                await session.run(statement)
