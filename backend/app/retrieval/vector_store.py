"""向量存储：内存实现（开发零运维）+ Chroma/Mongo/Milvus。

接口统一：add / search / delete_by_doc。
"""
from __future__ import annotations

import math
import asyncio
from typing import Any, Optional

from app.utils.logging import get_logger

logger = get_logger(__name__)


class VectorStore:
    """向量存储接口。"""

    async def add(self, id_: str, vector: list[float], metadata: dict[str, Any]) -> None: ...
    async def search(self, vector: list[float], top_k: int = 10, dept_id: Optional[str] = None) -> list[dict[str, Any]]: ...
    async def delete_by_doc(self, doc_id: str) -> None: ...
    async def count(self) -> int: ...


class MemoryVectorStore(VectorStore):
    """内存向量库：cosine 相似度（向量需已归一化；未归一化时自动归一）。"""

    backend_name = "memory"

    def __init__(self) -> None:
        self._vecs: dict[str, list[float]] = {}
        self._meta: dict[str, dict[str, Any]] = {}

    async def add(self, id_: str, vector: list[float], metadata: dict[str, Any]) -> None:
        self._vecs[id_] = self._normalize(vector)
        self._meta[id_] = dict(metadata)

    async def search(self, vector: list[float], top_k: int = 10, dept_id: Optional[str] = None) -> list[dict[str, Any]]:
        q = self._normalize(vector)
        scores: list[tuple[float, str]] = []
        for id_, v in self._vecs.items():
            if dept_id and self._meta[id_].get("dept_id") != dept_id:
                continue
            scores.append((self._dot(q, v), id_))
        scores.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, id_ in scores[:top_k]:
            out.append({"id": id_, "score": float(score), **self._meta[id_]})
        return out

    async def delete_by_doc(self, doc_id: str) -> None:
        for id_ in [i for i, m in self._meta.items() if m.get("doc_id") == doc_id]:
            self._vecs.pop(id_, None)
            self._meta.pop(id_, None)

    async def count(self) -> int:
        return len(self._vecs)

    @staticmethod
    def _normalize(v: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    @staticmethod
    def _dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))


class ChromaVectorStore(VectorStore):
    """Chroma 实现（可选，需 chromadb）。"""

    backend_name = "chroma"

    def __init__(self, collection_name: str = "wenshu_chunks") -> None:
        import chromadb  # 延迟导入

        self._client = chromadb.PersistentClient(path="./chroma_data")
        self._coll = self._client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    async def add(self, id_: str, vector: list[float], metadata: dict[str, Any]) -> None:
        self._coll.upsert(ids=[id_], embeddings=[vector], metadatas=[metadata])

    async def search(self, vector: list[float], top_k: int = 10, dept_id: Optional[str] = None) -> list[dict[str, Any]]:
        where = {"dept_id": dept_id} if dept_id else None
        res = self._coll.query(query_embeddings=[vector], n_results=top_k, where=where)
        out = []
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        for id_, dist, meta in zip(ids, dists, metas):
            out.append({"id": id_, "score": float(1.0 - dist), **(meta or {})})
        return out

    async def delete_by_doc(self, doc_id: str) -> None:
        self._coll.delete(where={"doc_id": doc_id})

    async def count(self) -> int:
        return self._coll.count()


class MongoVectorStore(VectorStore):
    """MongoDB 共享向量存储。

    适用于当前 MVP/中小数据量，所有部门 Pod 共享同一份向量与版本状态；
    超过百万 chunk 时可无缝替换为 Milvus 实现。
    """

    backend_name = "mongo"

    def __init__(self, store) -> None:
        self.store = store

    async def add(self, id_: str, vector: list[float], metadata: dict[str, Any]) -> None:
        await self.store.upsert("vector_embeddings", {
            "_id": id_, "vector": self._normalize(vector), **metadata,
        })

    async def search(self, vector: list[float], top_k: int = 10, dept_id: Optional[str] = None) -> list[dict[str, Any]]:
        query = {"dept_id": dept_id} if dept_id else None
        rows = await self.store.find("vector_embeddings", query)
        normalized = self._normalize(vector)
        scored = [
            (self._dot(normalized, row.get("vector") or []), row) for row in rows
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {"id": row["_id"], "score": float(score), **{k: v for k, v in row.items() if k not in {"_id", "vector"}}}
            for score, row in scored[:top_k]
        ]

    async def delete_by_doc(self, doc_id: str) -> None:
        for row in await self.store.find("vector_embeddings", {"doc_id": doc_id}):
            await self.store.delete("vector_embeddings", row["_id"])

    async def count(self) -> int:
        return await self.store.count("vector_embeddings")

    @staticmethod
    def _normalize(v: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    @staticmethod
    def _dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))


class MilvusVectorStore(VectorStore):
    """Milvus 向量存储（Milvus 2.4+ ``MilvusClient`` API）。

    pymilvus 延迟导入，因此 Mongo/内存模式不要求安装或连接 Milvus。
    同步 SDK 调用统一放到线程中，避免阻塞 FastAPI 事件循环。
    """

    backend_name = "milvus"

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        collection_name: str = "ilan_chunks",
        dimension: int = 3072,
        token: str = "",
        client: Any | None = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError("Milvus 向量维度必须为正数")
        self.uri = uri
        self.collection_name = collection_name
        self.dimension = dimension
        if client is None:
            try:
                from pymilvus import MilvusClient  # type: ignore
            except ImportError as exc:
                raise RuntimeError("Milvus 后端需要安装 pymilvus") from exc
            kwargs = {"uri": uri}
            if token:
                kwargs["token"] = token
            client = MilvusClient(**kwargs)
        self._client = client
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self._client.has_collection(collection_name=self.collection_name):
            return
        # 允许轻量测试替身只实现集合/数据操作，不强制复刻 SDK schema builder。
        if not hasattr(self._client, "create_schema"):
            self._client.create_collection(collection_name=self.collection_name, schema=None, index_params=None)
            return
        try:
            from pymilvus import DataType  # type: ignore

            schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=512)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dimension)
            index_params = self._client.prepare_index_params()
            index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        except ImportError as exc:
            # 测试/兼容客户端可自行实现 schema；真实客户端没有 pymilvus 时应明确失败。
            if self._client.__class__.__module__.startswith("tests") or self._client.__class__.__name__.startswith("Fake"):
                self._client.create_collection(collection_name=self.collection_name, schema=None, index_params=None)
                return
            raise RuntimeError("Milvus 后端需要安装 pymilvus") from exc
        self._client.create_collection(
            collection_name=self.collection_name, schema=schema, index_params=index_params
        )
        if hasattr(self._client, "load_collection"):
            self._client.load_collection(collection_name=self.collection_name)

    @staticmethod
    def _validate_vector(vector: list[float], dimension: int) -> list[float]:
        if len(vector) != dimension:
            raise ValueError(f"向量维度不匹配：期望 {dimension}，实际 {len(vector)}")
        return [float(value) for value in vector]

    async def add(self, id_: str, vector: list[float], metadata: dict[str, Any]) -> None:
        row = {"id": id_, "vector": self._validate_vector(vector, self.dimension), **metadata}
        await self.add_many([row])

    async def add_many(self, rows: list[dict[str, Any]]) -> None:
        """批量写入并只 flush 一次，供历史 Mongo→Milvus 迁移使用。"""
        if not rows:
            return
        prepared = [
            {**row, "vector": self._validate_vector(row["vector"], self.dimension)} for row in rows
        ]

        def write() -> None:
            # upsert 避免应用重启重建索引时产生重复主键；Fake client 仅实现 insert。
            if hasattr(self._client, "upsert"):
                self._client.upsert(collection_name=self.collection_name, data=prepared)
            else:
                self._client.insert(collection_name=self.collection_name, data=prepared)
            if hasattr(self._client, "flush"):
                self._client.flush(collection_name=self.collection_name)

        await asyncio.to_thread(write)

    async def search(self, vector: list[float], top_k: int = 10, dept_id: Optional[str] = None) -> list[dict[str, Any]]:
        query = self._validate_vector(vector, self.dimension)
        filter_expr = self._filter_expr("dept_id", dept_id) if dept_id else None
        fields = ["doc_id", "dept_id", "chunk_index"]

        def run_search():
            return self._client.search(
                collection_name=self.collection_name,
                data=[query],
                limit=top_k,
                output_fields=fields,
                filter=filter_expr,
            )

        results = await asyncio.to_thread(run_search)
        hits = results[0] if results else []
        output: list[dict[str, Any]] = []
        for hit in hits:
            if isinstance(hit, dict):
                id_ = hit.get("id")
                distance = hit.get("distance", hit.get("score", 0.0))
                entity = hit.get("entity") or {}
                if not entity:
                    entity = {field: hit[field] for field in fields if field in hit}
            else:
                id_ = getattr(hit, "id", None)
                distance = getattr(hit, "distance", getattr(hit, "score", 0.0))
                entity = getattr(hit, "entity", {}) or {}
            # Milvus COSINE 返回值本身就是相似度（1=完全相同，0=正交），与项目 score 语义一致。
            output.append({"id": id_, "score": float(distance), **dict(entity)})
        return output

    async def delete_by_doc(self, doc_id: str) -> None:
        def delete() -> None:
            self._client.delete(collection_name=self.collection_name, filter=self._filter_expr("doc_id", doc_id))
            if hasattr(self._client, "flush"):
                self._client.flush(collection_name=self.collection_name)
        await asyncio.to_thread(delete)

    async def count(self) -> int:
        # get_collection_stats 的 row_count 在 delete 后可能短暂滞后；优先走精确聚合。
        if hasattr(self._client, "query"):
            def exact_count():
                rows = self._client.query(
                    collection_name=self.collection_name,
                    filter='id != ""',
                    output_fields=["count(*)"],
                )
                return int((rows or [{}])[0].get("count(*)", 0))
            return await asyncio.to_thread(exact_count)
        stats = await asyncio.to_thread(self._client.get_collection_stats, collection_name=self.collection_name)
        return int(stats.get("row_count", 0))

    @staticmethod
    def _filter_expr(field: str, value: str) -> str:
        # Milvus 表达式字符串使用双引号；转义反斜杠和双引号，避免注入/解析错误。
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
        return f'{field} == "{escaped}"'


def build_vector_store(
    backend: str,
    store=None,
    *,
    milvus_uri: str = "http://localhost:19530",
    milvus_collection: str = "ilan_chunks",
    milvus_dimension: int = 3072,
    milvus_token: str = "",
) -> VectorStore:
    if backend == "mongo" and store is not None:
        return MongoVectorStore(store)
    if backend == "milvus":
        try:
            return MilvusVectorStore(
                uri=milvus_uri,
                collection_name=milvus_collection,
                dimension=milvus_dimension,
                token=milvus_token,
            )
        except Exception as exc:  # noqa: BLE001
            if store is not None:
                logger.warning("Milvus 初始化失败(%s)，回退 Mongo 向量库", exc)
                return MongoVectorStore(store)
            logger.warning("Milvus 初始化失败(%s)，回退内存向量库", exc)
    if backend == "chroma":
        try:
            return ChromaVectorStore()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chroma 初始化失败(%s)，回退内存向量库", exc)
    return MemoryVectorStore()
