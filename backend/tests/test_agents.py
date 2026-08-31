"""测试 Agent 的无 LLM 回退逻辑。"""
from __future__ import annotations

from app.harness.base import Answer, VerificationResult
from app.harness.orchestrator import Orchestrator
from app.harness.agents.intent_agent import IntentAgent
from app.harness.agents.query_rewriter import QueryRewriter
from app.harness.agents.verifier_agent import VerifierAgent


class _FakeLLM:
    """模拟 LLM，complete 抛错以测试回退。"""

    def __init__(self):
        self.calls = 0

    async def complete(self, *a, **kw):
        self.calls += 1
        raise RuntimeError("no llm")

    async def complete_json(self, *a, **kw):
        self.calls += 1
        raise RuntimeError("no llm")


def test_intent_fallback():
    store = _MemStore()
    agent = IntentAgent(_FakeLLM(), store)
    import asyncio

    intent = asyncio.run(agent.infer("退课截止时间是第几周？", "u1", "上一轮事项：退课"))
    assert intent.type == "deadline_query"
    assert "dept_jwc" in intent.depts


def test_query_rewriter_fallback():
    store = _MemStore()
    rw = QueryRewriter(_FakeLLM(), store)
    import asyncio

    queries = asyncio.run(rw.rewrite("退课怎么办", None))
    assert queries and queries[0] == "退课怎么办"


def test_verifier_heuristic():
    v = VerifierAgent(_FakeLLM())
    import asyncio

    answer = Answer(content="根据规定应于第8周前退课", citations=[])
    result = asyncio.run(v.verify("退课时间", answer, []))
    assert isinstance(result, VerificationResult)


def test_chat_response_exposes_truthful_graph_fallback_status():
    response = Orchestrator._to_response(
        "session-1", Answer(content="普通 RAG 回答"), [{"graph_status": "fallback_unavailable"}], "other",
    )

    assert response["graph_status"] == "fallback_unavailable"


def test_answer_context_keeps_bounded_graph_evidence_after_primary_top_k():
    primary = [{"id": f"primary:{index}"} for index in range(5)]
    graph = [{"id": f"graph:{index}", "retrieval_source": "graph"} for index in range(2)]

    context = Orchestrator._compose_answer_context([], primary + graph, top_k=5, graph_limit=2)

    assert [chunk["id"] for chunk in context] == [
        "primary:0", "primary:1", "primary:2", "primary:3", "primary:4", "graph:0", "graph:1",
    ]


class _MemStore:
    async def list_departments(self):
        return [{"_id": "dept_jwc", "name": "教务处"}, {"_id": "dept_cwc", "name": "财务处"}]

    async def get_user_profile(self, user_id):
        return None

    async def list_glossary(self):
        return []
