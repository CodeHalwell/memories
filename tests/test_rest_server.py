"""Tests for the REST server adapter.

Memory logic is covered by test_service.py; here we verify the HTTP wiring:
the dependency guard and the request/response mapping for each endpoint.
"""

import importlib.util

import pytest

from agent_memory.integrations import rest_server
from agent_memory.service import MemoryService

FASTAPI_INSTALLED = (
    importlib.util.find_spec("fastapi") is not None
    and importlib.util.find_spec("httpx") is not None
)


def test_module_imports_without_server_extra():
    assert hasattr(rest_server, "create_app")


@pytest.mark.skipif(FASTAPI_INSTALLED, reason="server extra is installed")
def test_create_app_without_fastapi_raises_actionable_error():
    with pytest.raises(ImportError, match=r"agent-memory\[server\]"):
        rest_server.create_app(MemoryService(load_embeddings=False))


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    service = MemoryService(data_dir=tmp_path, load_embeddings=False)
    app = rest_server.create_app(service)
    with TestClient(app) as c:  # runs lifespan (initialize/close)
        yield c


pytestmark = pytest.mark.skipif(
    not FASTAPI_INSTALLED, reason="requires the 'server' extra + httpx"
)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_save_and_retrieve_roundtrip(client):
    save = client.post("/memories", json={
        "content": "The user prefers dark roast coffee",
        "session_id": "s1", "turn": 1, "namespace": "u1",
    })
    assert save.status_code == 200
    assert save.json()["saved"] is True

    retrieve = client.post("/retrieve", json={"query": "coffee preference", "namespace": "u1"})
    assert retrieve.status_code == 200
    body = retrieve.json()
    assert body["count"] >= 1
    assert any("coffee" in m["content"] for m in body["memories"])


def test_retrieve_is_namespace_isolated_over_http(client):
    client.post("/memories", json={"content": "secret alpha", "session_id": "s", "turn": 1, "namespace": "a"})
    other = client.post("/retrieve", json={"query": "secret alpha", "namespace": "b"})
    assert other.json()["count"] == 0


def test_get_memory_and_404(client):
    save = client.post("/memories", json={
        "content": "remember this fact", "session_id": "s1", "turn": 1, "namespace": "u1",
    })
    mem_id = save.json()["memory"]["id"]

    found = client.get(f"/memories/{mem_id}", params={"namespace": "u1"})
    assert found.status_code == 200
    assert found.json()["id"] == mem_id

    missing = client.get("/memories/nope", params={"namespace": "u1"})
    assert missing.status_code == 404
    # Wrong namespace is also a 404 (isolation).
    wrong_ns = client.get(f"/memories/{mem_id}", params={"namespace": "other"})
    assert wrong_ns.status_code == 404
