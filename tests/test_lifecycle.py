"""End-to-end lifecycle tests through the real save -> retrieve -> compact flow.

Each test here corresponds to a defect found in the research review, where
the earlier unit tests passed only because they set internal state by hand.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from agent_memory import MemoryManager
from agent_memory.core.decay import compute_decay, decay_for_memory
from agent_memory.models import Memory

OLD = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()


async def _age_all(mgr, when: str = OLD) -> None:
    await mgr.sqlite.db.execute(
        "UPDATE memories SET created_at = ?, event_time = ?, last_accessed = NULL", (when, when),
    )
    await mgr.sqlite.db.commit()


# ── Decay ──

def test_decay_is_one_when_fresh_and_falls_with_time():
    now = datetime.now(timezone.utc)
    assert compute_decay(now, access_count=0) == 1.0
    assert compute_decay(now, access_count=5) == 1.0  # use never lowers the score
    week = compute_decay(now - timedelta(days=7), access_count=0)
    assert 0.45 < week < 0.55  # one half-life


def test_access_count_slows_decay():
    t = datetime.now(timezone.utc) - timedelta(days=14)
    assert compute_decay(t, access_count=20) > compute_decay(t, access_count=0)


def test_decay_for_memory_uses_event_time_when_never_accessed():
    mem = Memory(content="x", raw_log_id="r", event_time=OLD, created_at=OLD)
    assert decay_for_memory(mem) < 0.01
    mem.last_accessed = datetime.now(timezone.utc).isoformat()
    assert decay_for_memory(mem) == 1.0


async def test_retrieval_refreshes_decay_rather_than_lowering_it(lite_manager):
    await lite_manager.process_turn("s1", 1, "The user prefers metric units for all recipes")
    hits = await lite_manager.retrieve("metric units recipes")
    assert hits and hits[0].kind == "memory"
    stored = await lite_manager.sqlite.get_memory(hits[0].id)
    assert stored.decay_score == 1.0 and stored.access_count == 1


# ── Compaction ──

async def test_compaction_selects_and_merges_stale_memories(lite_manager, stub_llm):
    stub_llm(salience=0.0, answer_for=lambda src: "supplier")
    for i in range(6):
        await lite_manager.process_turn(f"s{i % 2}", i, f"billing invoice note {i} for the supplier")
    await _age_all(lite_manager)
    result = await lite_manager.run_compaction()
    assert result.memories_reviewed > 0
    assert result.memories_merged >= 1
    async with lite_manager.sqlite.db.execute(
        "SELECT COUNT(*) AS n FROM memories WHERE is_semantic = 1 AND tier = 'warm'"
    ) as cur:
        assert (await cur.fetchone())["n"] >= 1


async def test_merge_rejected_when_facts_would_be_lost(lite_manager, stub_llm):
    # The validator's answer does not appear in the merged text -> no merge.
    stub_llm(salience=0.0, answer_for=lambda src: "Zanzibar")
    for i in range(4):
        await lite_manager.process_turn(f"s{i % 2}", i, f"billing invoice note {i} for the supplier")
    await _age_all(lite_manager)
    result = await lite_manager.run_compaction()
    assert result.memories_reviewed > 0
    assert result.memories_merged == 0


async def test_compaction_never_merges_across_namespaces(lite_manager, stub_llm):
    stub_llm(salience=0.0, answer_for=lambda src: src.split()[0])
    for ns in ("alice", "bob"):
        for i in range(2):
            mem = Memory(
                content=f"{ns} billing address note {i}", raw_log_id=f"raw-{ns}-{i}",
                namespace=ns, session_id=f"{ns}-s{i}", turn=i, salience=0.05,
                created_at=OLD, event_time=OLD, keywords=[("billing", 1.0), ("address", 1.0)],
            )
            await lite_manager.sqlite.save_memory(mem)
    await lite_manager.run_compaction()
    async with lite_manager.sqlite.db.execute(
        "SELECT source_memory_ids, resulting_memory_id FROM compaction_merges"
    ) as cur:
        merges = [dict(r) for r in await cur.fetchall()]
    assert merges, "expected at least one within-namespace merge"
    for m in merges:
        sources = [await lite_manager.sqlite.get_memory(i) for i in json.loads(m["source_memory_ids"])]
        result = await lite_manager.sqlite.get_memory(m["resulting_memory_id"])
        assert {s.namespace for s in sources} == {result.namespace}
    assert await lite_manager.retrieve("billing address", namespace="default") == []


async def test_dedupe_collapses_identical_memories_without_llm(lite_manager):
    for i in range(3):
        mem = Memory(
            content="Remember the wifi password is hunter2", raw_log_id=f"r{i}",
            session_id=f"s{i}", turn=0, salience=0.4 + i * 0.1,
        )
        await lite_manager.sqlite.save_memory(mem)
    result = await lite_manager.run_compaction()
    assert result.memories_pruned == 2
    assert await lite_manager.sqlite.count_memories(tier="cold") == 2


# ── Save path ──

async def test_explicit_instruction_saved_without_llm(lite_manager):
    await lite_manager.process_turn("s1", 0, "hello")
    mem = await lite_manager.process_turn("s1", 1, "Remember this: I am allergic to penicillin")
    assert mem is not None and mem.fast_pathed


async def test_lexical_policy_keeps_facts_and_skips_acknowledgements(lite_manager):
    await lite_manager.process_turn("s1", 0, "hi")
    fact = await lite_manager.process_turn("s1", 1, "My daughter's birthday is on 4 March and she loves dinosaurs")
    ack = await lite_manager.process_turn("s1", 2, "ok thanks!")
    assert fact is not None
    assert ack is None


async def test_decisions_log_features_and_provenance(lite_manager):
    await lite_manager.process_turn("s1", 0, "hi")
    await lite_manager.process_turn("s1", 1, "ok")
    async with lite_manager.sqlite.db.execute(
        "SELECT policy_source, features FROM save_decisions ORDER BY decided_at"
    ) as cur:
        rows = [dict(r) for r in await cur.fetchall()]
    assert rows[0]["policy_source"] == "rule:first_turn"
    assert rows[1]["policy_source"] == "heuristic:lexical"
    assert "novelty" in json.loads(rows[1]["features"])


async def test_first_turn_detection_survives_restart(tmp_path):
    cfg = {"llm_enabled": False, "policy_auto_train": False}
    m1 = MemoryManager(data_dir=tmp_path, config=cfg)
    await m1.initialize(load_embeddings=False)
    await m1.process_turn("s1", 0, "hello")
    await m1.close()
    m2 = MemoryManager(data_dir=tmp_path, config=cfg)
    await m2.initialize(load_embeddings=False)
    assert await m2.process_turn("s1", 1, "ok cool") is None
    await m2.close()


async def test_skipped_turn_is_still_retrievable_as_raw(lite_manager):
    await lite_manager.process_turn("s1", 0, "hi")
    assert await lite_manager.process_turn("s1", 1, "ok, zucchini") is None
    hits = await lite_manager.retrieve("zucchini")
    assert hits and hits[0].kind == "raw" and hits[0].id.startswith("raw:")
    fetched = await lite_manager.get_memory(hits[0].id)
    assert fetched is not None and "zucchini" in fetched.content


# ── Retrieval signals ──

async def test_grep_layer_uses_manager_log_dir_and_scores_content(tmp_path):
    mgr = MemoryManager(data_dir=tmp_path, config={
        "llm_enabled": False, "policy_auto_train": False, "retrieval_layers": ["grep"],
    })
    await mgr.initialize(load_embeddings=False)
    try:
        await mgr.process_turn("s1", 1, "zebra crossing on Abbey Road")
        await mgr.process_turn("s1", 2, "a note about gardening and roses")
        assert [h.content for h in await mgr.retrieve("zebra")] == ["zebra crossing on Abbey Road"]
        # No content match -> no results (JSON keys and unrelated lines don't count).
        assert await mgr.retrieve("what is my favourite colour") == []
        # Regex metacharacters are searched literally.
        assert await mgr.retrieve("price (in USD") == []
    finally:
        await mgr.close()


async def test_retrieval_miss_registers_as_gap(lite_manager):
    await lite_manager.process_turn("s1", 0, "a note about gardening")
    await lite_manager.retrieve("quantum chromodynamics lattice", session_id="s1")
    gaps = await lite_manager.sqlite.get_failed_retrieval_keywords("s1")
    assert "quantum" in gaps and "chromodynamics" in gaps


async def test_retrieval_logs_ranked_hits_and_turn(lite_manager):
    await lite_manager.process_turn("s1", 0, "the boiler service is due in October")
    hits = await lite_manager.retrieve("boiler service", session_id="s1", turn=3)
    rid = hits[0].retrieval_id
    async with lite_manager.sqlite.db.execute(
        "SELECT turn, namespace, miss FROM retrieval_decisions WHERE id = ?", (rid,)
    ) as cur:
        row = dict(await cur.fetchone())
    assert row == {"turn": 3, "namespace": "default", "miss": 0}
    logged = await lite_manager.sqlite.get_retrieval_hits(rid)
    assert logged[0]["rank"] == 1 and logged[0]["memory_id"] == hits[0].id


async def test_time_filter(lite_manager):
    await lite_manager.process_turn("s1", 0, "holiday in Lisbon", timestamp="2023-05-01T10:00:00+00:00")
    await lite_manager.process_turn("s1", 1, "holiday in Porto", timestamp="2024-05-01T10:00:00+00:00")
    hits = await lite_manager.retrieve("holiday", since="2024-01-01T00:00:00+00:00")
    assert [h.content for h in hits] == ["holiday in Porto"]


# ── Deletion ──

async def test_delete_memory_and_purge_raw(lite_manager):
    mem = await lite_manager.process_turn("s1", 0, "my locker code is 4471")
    assert await lite_manager.delete_memory(mem.id, purge_raw=True)
    assert await lite_manager.sqlite.get_memory(mem.id) is None
    assert await lite_manager.retrieve("locker code") == []


async def test_purge_namespace_removes_everything(lite_manager):
    await lite_manager.process_turn("s", 0, "Alice's codename is Falcon", namespace="alice")
    await lite_manager.process_turn("s", 0, "Bob's codename is Otter", namespace="bob")
    info = await lite_manager.purge_namespace("alice")
    assert info["memories_deleted"] == 1 and info["raw_turns_deleted"] == 1
    assert await lite_manager.retrieve("codename", namespace="alice") == []
    assert await lite_manager.retrieve("codename", namespace="bob")


async def test_jsonl_logs_are_namespaced(lite_manager):
    await lite_manager.process_turn("s", 0, "one", namespace="alice")
    await lite_manager.process_turn("s", 0, "two", namespace="bob")
    alice = list(lite_manager.jsonl.iter_session("s", namespace="alice"))
    assert [e.content for e in alice] == ["one"]


# ── Graph ──

async def test_entities_are_indexed_and_retrievable(lite_manager, stub_llm):
    pytest.importorskip("kuzu")
    stub_llm(entities=[{"name": "Caroline", "type": "person"}])
    await lite_manager.process_turn("s1", 0, "She went to the support group yesterday")
    hits = await lite_manager.retrieve("What did Caroline do?")
    assert hits and "entity" in hits[0].matched_layers


async def test_write_time_links_connect_similar_memories(edge_manager):
    pytest.importorskip("kuzu")
    edge_manager.config["link_similarity_threshold"] = 0.5
    a = await edge_manager.process_turn("s1", 0, "the quarterly budget review meeting is on Friday")
    b = await edge_manager.process_turn("s2", 0, "the quarterly budget review meeting moved to Monday")
    related = {r["id"] for r in edge_manager.graph.get_related_memories(a.id, max_depth=1)}
    assert b.id in related
