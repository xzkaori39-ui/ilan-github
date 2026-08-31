"""测试 fixtures。"""
from __future__ import annotations

import os

# Tests must never inherit the deployment Mongo/Redis/API settings.  This also
# keeps TestClient and its storage operations on an in-process test backend.
os.environ["STORAGE_MODE"] = "memory"
os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["GRAPH_ENABLED"] = "false"
os.environ["PI_AGENT_ENABLED"] = "false"

import pytest

from app.config import get_settings
from app.deps import build_container
from app.llm.embeddings import EmbeddingClient
from app.pipeline.parser import DocumentParser


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def container(settings):
    """离线容器（memory 存储 + hash 向量 + 无真实 LLM）。"""
    return build_container(settings)


@pytest.fixture
def fresh_container(settings):
    """每个测试独立的离线容器。"""
    return build_container(settings)


@pytest.fixture
def parser():
    return DocumentParser()


@pytest.fixture
def embeddings(settings):
    return EmbeddingClient(settings, relay=None)
