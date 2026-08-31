"""Official GraphRAG artifact normalization contracts."""
from __future__ import annotations

from app.graph.graphrag_artifacts import build_projection


def test_build_projection_keeps_only_citable_entity_paths_and_stable_keys():
    projection = build_projection(
        text_units=[
            {"id": "tu-1", "document_id": "doc_a:0"},
            {"id": "tu-2", "document_id": "doc_a:1"},
        ],
        entities=[
            {
                "id": "entity-dept",
                "title": "学生处",
                "type": "DEPARTMENT",
                "description": "负责学生事务的部门",
                "text_unit_ids": ["tu-1", "missing-tu"],
            },
            {
                "id": "entity-policy",
                "title": "学生管理规定",
                "type": "POLICY",
                "description": "学生管理制度",
                "text_unit_ids": ["tu-2"],
            },
        ],
        relationships=[
            {
                "id": "rel-1",
                "source": "学生处",
                "target": "学生管理规定",
                "description": "学生处执行该规定",
                "weight": 9.0,
                "text_unit_ids": ["tu-1", "missing-tu"],
            },
            {
                "id": "rel-ignored",
                "source": "不存在",
                "target": "学生管理规定",
                "description": "没有可导入端点",
                "weight": 1.0,
                "text_unit_ids": ["tu-1"],
            },
        ],
        communities=[
            {
                "community": 7,
                "level": 0,
                "title": "学生管理",
                "entity_ids": ["entity-dept", "entity-policy"],
                "text_unit_ids": ["tu-1", "missing-tu"],
            },
        ],
        community_reports=[
            {
                "community": 7,
                "level": 0,
                "title": "学生管理洞察",
                "summary": "学生事务与制度的关联",
                "full_content": "完整洞察",
            },
        ],
    )

    assert projection["text_units"] == [
        {"id": "tu-1", "chunk_id": "doc_a:0"},
        {"id": "tu-2", "chunk_id": "doc_a:1"},
    ]
    assert projection["entities"] == [
        {
            "id": "entity-dept",
            "key": "department:学生处",
            "name": "学生处",
            "type": "DEPARTMENT",
            "description": "负责学生事务的部门",
            "text_unit_ids": ["tu-1"],
        },
        {
            "id": "entity-policy",
            "key": "policy:学生管理规定",
            "name": "学生管理规定",
            "type": "POLICY",
            "description": "学生管理制度",
            "text_unit_ids": ["tu-2"],
        },
    ]
    assert projection["relationships"] == [
        {
            "id": "rel-1",
            "key": "department:学生处|policy:学生管理规定|学生处执行该规定",
            "source_key": "department:学生处",
            "target_key": "policy:学生管理规定",
            "description": "学生处执行该规定",
            "weight": 9.0,
            "text_unit_ids": ["tu-1"],
        },
    ]
    assert projection["communities"] == [{
        "key": "0:7",
        "community": 7,
        "level": 0,
        "title": "学生管理",
        "entity_keys": ["department:学生处", "policy:学生管理规定"],
        "text_unit_ids": ["tu-1"],
    }]
    assert projection["community_reports"] == [{
        "community_key": "0:7",
        "title": "学生管理洞察",
        "summary": "学生事务与制度的关联",
        "full_content": "完整洞察",
    }]
