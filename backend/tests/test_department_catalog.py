"""部门目录与大文件导入的配置回归测试。"""
from __future__ import annotations

import pytest
from pathlib import Path

from app.config import Settings
from app.harness.agents.dept_router import DeptRouter
from app.main import _backfill_departments
from app.storage.store import MemoryStore
from scripts.ingest_department_files import DEPT_MAP, resolve_dept
from scripts.seed_data import DEPARTMENTS


class _Container:
    def __init__(self) -> None:
        self.store = MemoryStore()


def test_default_upload_limit_accepts_the_92mb_graduate_handbook():
    assert Settings().max_upload_mb >= 128


def test_microelectronics_folder_maps_to_a_dedicated_department():
    assert resolve_dept(Path("/tmp/department_files/微电子学院/微电子学院学生手册.pdf")) == "dept_weidianzi"


def test_independent_college_folders_map_to_their_own_departments():
    assert resolve_dept(Path("/tmp/department_files/中法学院/心理测评操作说明.pdf")) == "dept_zfxy"
    assert resolve_dept(Path("/tmp/department_files/微电子学院/微电子学院学生手册.pdf")) == "dept_weidianzi"
    assert DEPT_MAP["中法学院"] != DEPT_MAP["微电子学院"]


def test_seed_catalog_contains_the_two_independent_college_nodes():
    seeded_ids = {department["_id"] for department in DEPARTMENTS}

    assert {"dept_zfxy", "dept_weidianzi"} <= seeded_ids


@pytest.mark.asyncio
async def test_student_questions_route_to_the_matching_independent_college():
    store = MemoryStore()
    for department in DEPARTMENTS:
        await store.upsert_department(department)
    router = DeptRouter(llm=None, store=store)

    microelectronics = await router.route("微电子学院学生手册中的实验室规范在哪里查看？")
    sino_french = await router.route("中法学院赴法交换和行李寄存如何办理？")

    assert microelectronics["dept_ids"] == ["dept_weidianzi"]
    assert microelectronics["dept_names"] == ["微电子学院"]
    assert microelectronics["matched_by"] == "keyword"
    assert sino_french["dept_ids"] == ["dept_zfxy"]
    assert sino_french["dept_names"] == ["中法学院"]
    assert sino_french["matched_by"] == "keyword"


@pytest.mark.asyncio
async def test_startup_creates_microelectronics_department_when_missing():
    container = _Container()

    await _backfill_departments(container)

    assert await container.store.get_department("dept_weidianzi") == {
        "_id": "dept_weidianzi",
        "name": "微电子学院",
        "name_en": "School of Microelectronics",
        "category": "academic",
        "admin_users": [],
        "agent_config": {},
        "loop_phase": "human_in_loop",
        "review_stats": {"total": 0, "correct": 0, "accuracy": 0.0},
    }
