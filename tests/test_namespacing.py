"""Tests for multi-tenant namespace isolation (§2.4 of the integration roadmap).

Isolation is enforced at the SQLite query layer (and, with embeddings, the
vector payload filter). These tests exercise the lite profile, where the SQLite
layer is the sole enforcement point — the strongest guarantee to verify.
"""

import aiosqlite
import pytest
import pytest_asyncio

from agent_memory.models import Memory
from agent_memory.service import MemoryService
from agent_memory.storage.sqlite_store import SQLiteStore


@pytest_asyncio.fixture
async def service(tmp_path):
    svc = MemoryService(data_dir=tmp_path, load_embeddings=False)
    await svc.initialize()
    yield svc
    await svc.close()


async def test_retrieve_is_namespace_isolated(service):
    await service.save_turn(content="Alice's secret project is codename Falcon", session_id="s", turn=1, namespace="alice")
    await service.save_turn(content="Bob's secret project is codename Otter", session_id="s", turn=1, namespace="bob")

    alice = await service.retrieve("secret project codename", namespace="alice")
    assert alice["count"] >= 1
    assert all(m["namespace"] == "alice" for m in alice["memories"])
    assert any("Falcon" in m["content"] for m in alice["memories"])
    assert not any("Otter" in m["content"] for m in alice["memories"])

    bob = await service.retrieve("secret project codename", namespace="bob")
    assert all(m["namespace"] == "bob" for m in bob["memories"])
    assert not any("Falcon" in m["content"] for m in bob["memories"])


async def test_default_namespace_does_not_see_tenants(service):
    await service.save_turn(content="Alice data about quarterly revenue", session_id="s", turn=1, namespace="alice")
    default = await service.retrieve("quarterly revenue")  # namespace defaults to "default"
    assert default["count"] == 0


async def test_get_memory_cross_namespace_returns_none(service):
    saved = await service.save_turn(content="namespaced note", session_id="s", turn=1, namespace="alice")
    mem_id = saved["memory"]["id"]

    assert await service.get_memory(mem_id, namespace="alice") is not None
    assert await service.get_memory(mem_id, namespace="bob") is None
    # Without a namespace filter, the lookup is unrestricted.
    assert await service.get_memory(mem_id) is not None


@pytest_asyncio.fixture
async def store(tmp_path):
    s = SQLiteStore(tmp_path / "ns.db")
    await s.initialize()
    yield s
    await s.close()


async def test_sqlite_search_respects_namespace(store):
    await store.save_memory(Memory(id="a1", content="apples and oranges", raw_log_id="r", namespace="alice", session_id="s", turn=1))
    await store.save_memory(Memory(id="b1", content="apples and bananas", raw_log_id="r", namespace="bob", session_id="s", turn=1))

    alice_hits = await store.search_by_content(["apples"], namespace="alice")
    assert {m.id for m in alice_hits} == {"a1"}

    # No namespace filter returns both.
    all_hits = await store.search_by_content(["apples"])
    assert {m.id for m in all_hits} == {"a1", "b1"}


async def test_migration_adds_namespace_to_legacy_db(tmp_path):
    """A database created before the namespace column must be migrated in place,
    with existing rows defaulting to the 'default' namespace."""
    db_path = tmp_path / "legacy.db"

    # Build a legacy 'memories' table without the namespace column.
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute(
            """CREATE TABLE memories (
                id TEXT PRIMARY KEY, created_at TEXT, updated_at TEXT, content TEXT,
                summary TEXT, raw_log_id TEXT, session_id TEXT, turn INTEGER,
                valence REAL, arousal REAL, surprise REAL, salience REAL,
                access_count INTEGER, last_accessed TEXT, decay_score REAL,
                compaction_gen INTEGER, tier TEXT, fast_pathed INTEGER,
                is_semantic INTEGER, graph_node_id TEXT, vector_id TEXT,
                spatial_embedding BLOB, scene_description TEXT
            )"""
        )
        await db.execute(
            "INSERT INTO memories (id, created_at, updated_at, content, raw_log_id, session_id, turn, decay_score, tier) "
            "VALUES ('legacy-1','t','t','old memory','r','s',1,1.0,'hot')"
        )
        await db.commit()

    # Opening via SQLiteStore must migrate and read the legacy row as 'default'.
    store = SQLiteStore(db_path)
    await store.initialize()
    try:
        mem = await store.get_memory("legacy-1")
        assert mem is not None
        assert mem.namespace == "default"
        # And the new namespace filter works on the migrated row.
        assert await store.get_memory("legacy-1", namespace="default") is not None
        assert await store.get_memory("legacy-1", namespace="other") is None
    finally:
        await store.close()
