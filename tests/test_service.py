"""Tests for the transport-agnostic MemoryService facade (lite profile)."""

import pytest
import pytest_asyncio

from agent_memory.service import MemoryService, memory_to_dict
from agent_memory.models import Memory


@pytest_asyncio.fixture
async def service(tmp_path):
    svc = MemoryService(data_dir=tmp_path, load_embeddings=False)
    await svc.initialize()
    yield svc
    await svc.close()


def test_memory_to_dict_is_json_safe():
    mem = Memory(
        content="hello", session_id="s1", turn=2, salience=0.6,
        keywords=[("alpha", 0.9), ("beta", 0.4)],
    )
    d = memory_to_dict(mem)
    assert d["content"] == "hello"
    assert d["session_id"] == "s1"
    assert d["turn"] == 2
    assert d["keywords"] == ["alpha", "beta"]
    # No binary/internal fields leak out.
    assert "spatial_embedding" not in d
    assert "vector_id" not in d


async def test_initialize_is_idempotent(tmp_path):
    svc = MemoryService(data_dir=tmp_path, load_embeddings=False)
    await svc.initialize()
    await svc.initialize()  # second call is a no-op
    assert svc._initialized is True
    await svc.close()


async def test_save_first_turn_persists(service):
    result = await service.save_turn(
        content="The user is planning a trip to Kyoto in spring.",
        session_id="s1", turn=1,
    )
    assert result["saved"] is True
    assert result["memory"] is not None
    assert result["memory"]["session_id"] == "s1"
    assert "Kyoto" in result["memory"]["content"]


async def test_retrieve_returns_saved_memory(service):
    await service.save_turn(
        content="The user is planning a trip to Kyoto in spring.",
        session_id="s1", turn=1,
    )
    result = await service.retrieve("Kyoto trip", session_id="s1")
    assert result["query"] == "Kyoto trip"
    assert result["count"] >= 1
    assert any("Kyoto" in m["content"] for m in result["memories"])


async def test_get_memory_roundtrip(service):
    saved = await service.save_turn(content="Remember the wifi password is hunter2", session_id="s1", turn=1)
    mem_id = saved["memory"]["id"]
    fetched = await service.get_memory(mem_id)
    assert fetched is not None
    assert fetched["id"] == mem_id


async def test_get_missing_memory_returns_none(service):
    assert await service.get_memory("does-not-exist") is None
