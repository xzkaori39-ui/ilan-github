"""RAG 评测管理端接口的鉴权、去重与部门隔离。"""
from __future__ import annotations

import asyncio
import json

from fastapi.testclient import TestClient

from app.main import app
from app.api.routes.admin import _evaluation_for_scope


def _login(client: TestClient, username: str) -> str:
    response = client.post("/api/v1/auth/login", json={"username": username, "password": "admin123"})
    assert response.status_code == 200
    return response.json()["data"]["token"]


def test_department_admin_cannot_enqueue_global_evaluation():
    with TestClient(app) as client:
        token = _login(client, "jwc_admin")
        response = client.post("/api/v1/admin/evaluations/run", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 403


def test_system_admin_can_enqueue_the_public_demo_dataset():
    with TestClient(app) as client:
        token = _login(client, "admin")
        response = client.post(
            "/api/v1/admin/evaluations/run",
            json={"dataset_id": "real_document_qa"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert response.json()["data"]["dataset_id"] == "real_document_qa"


def test_evaluation_read_response_redacts_hidden_gold_path_fields():
    with TestClient(app) as client:
        token = _login(client, "admin")
        store = app.state.container.store
        asyncio.run(store.upsert("rag_evaluations", {
            "_id": "eval_test_redaction", "status": "completed", "created_at": "2026-08-28T00:02:00+00:00",
            "profiles": [{"name": "baseline", "details": [{
                "id": "case", "dept_id": "dept_all", "query": "q", "graph_paths": [{"relationship_keys": ["hidden"]}],
                "retrieved_chunk_ids": ["hidden-chunk"], "allowed_relation_path_sets": [["hidden-gold"]],
            }], "metrics": {}}],
        }))
        try:
            response = client.get("/api/v1/admin/evaluations/latest", headers={"Authorization": f"Bearer {token}"})
            assert response.status_code == 200
            rendered = json.dumps(response.json()["data"])
            assert "allowed_relation_path_sets" not in rendered
            assert "graph_paths" not in rendered
            assert "hidden-chunk" not in rendered
        finally:
            asyncio.run(store.delete("rag_evaluations", "eval_test_redaction"))


def test_system_admin_run_rejects_duplicate_and_department_latest_hides_other_details():
    with TestClient(app) as client:
        super_token = _login(client, "admin")
        dept_token = _login(client, "jwc_admin")
        store = app.state.container.store
        asyncio.run(store.upsert("async_jobs", {
            "_id": "job_eval_test_busy", "type": "evaluation_run", "status": "queued", "created_at": "2026-08-28T00:00:00+00:00",
        }))
        try:
            duplicate = client.post("/api/v1/admin/evaluations/run", headers={"Authorization": f"Bearer {super_token}"})
            assert duplicate.status_code == 409

            asyncio.run(store.upsert("rag_evaluations", {
                "_id": "eval_test_scoped", "status": "completed", "created_at": "2026-08-28T00:01:00+00:00",
                "profiles": [{"name": "baseline", "details": [
                    {"id": "jwc", "dept_id": "dept_jwc"}, {"id": "cwc", "dept_id": "dept_cwc"},
                ], "metrics": {}}],
            }))
            latest = client.get("/api/v1/admin/evaluations/latest", headers={"Authorization": f"Bearer {dept_token}"})
            assert latest.status_code == 200
            details = latest.json()["data"]["profiles"][0]["details"]
            assert details == [{"id": "jwc", "dept_id": "dept_jwc"}]
        finally:
            asyncio.run(store.delete("async_jobs", "job_eval_test_busy"))
            asyncio.run(store.delete("rag_evaluations", "eval_test_scoped"))


def test_department_scope_recomputes_graph_groups_from_only_visible_cases():
    evaluation = {
        "top_k": 5,
        "profiles": [{
            "name": "baseline_no_graph",
            "metrics": {},
            "groups": {"graph_sensitive": {"case_count": 99}},
            "details": [
                {
                    "id": "jwc", "dept_id": "dept_jwc", "graph_sensitive": True, "success": True,
                    "rank": 1, "citation_correctness": 1.0, "answer_key_coverage": 1.0, "latency_ms": 1,
                    "evidence_set_complete": True, "bridge_evidence_hit": False,
                    "graph_evidence_precision": None, "graph_path_valid": None, "distractor_hit": False,
                    "graph_status": "no_new_evidence",
                },
                {
                    "id": "other", "dept_id": "dept_cwc", "graph_sensitive": False, "success": True,
                    "rank": 1, "citation_correctness": 1.0, "answer_key_coverage": 1.0, "latency_ms": 1,
                    "evidence_set_complete": True, "bridge_evidence_hit": None,
                    "graph_evidence_precision": None, "graph_path_valid": None, "distractor_hit": False,
                    "graph_status": "disabled",
                },
            ],
        }],
    }

    scoped = _evaluation_for_scope(evaluation, "dept_jwc")

    assert scoped["profiles"][0]["groups"]["all"]["case_count"] == 1
    assert scoped["profiles"][0]["groups"]["graph_sensitive"]["case_count"] == 1
    assert scoped["profiles"][0]["groups"]["control"]["case_count"] == 0
