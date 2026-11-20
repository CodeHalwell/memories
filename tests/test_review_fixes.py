"""Regressions for the automated review findings on PR #10."""

from __future__ import annotations

from agent_memory.config import make_config
from agent_memory.core.compaction import _near_duplicate, _normalised
from agent_memory.models import Memory
from agent_memory.policy.controller import PolicyController
from agent_memory.policy.decision_model import (
    FEATURE_NAMES,
    LogisticDecisionModel,
    build_save_features,
)
from agent_memory.policy.outcome_assessor import assess_save_outcomes
from agent_memory.storage.jsonl_log import JSONLLogger
from agent_memory.text import fts_query


# P1: namespaces that slug identically must not share log directories.
def test_namespace_dirs_do_not_collide(tmp_path):
    log = JSONLLogger(tmp_path)
    assert log.namespace_dir("tenant/a") != log.namespace_dir("tenant_a")


async def test_purge_keeps_other_tenants_logs(lite_manager):
    await lite_manager.process_turn("s", 0, "alpha secret", namespace="tenant/a")
    await lite_manager.process_turn("s", 0, "beta secret", namespace="tenant_a")
    await lite_manager.purge_namespace("tenant/a")
    remaining = list(lite_manager.jsonl.iter_session("s", namespace="tenant_a"))
    assert [e.content for e in remaining] == ["beta secret"]


# Raw-only retrieval must not depend on curation.
async def test_raw_only_retrieval_returns_raw_candidates(lite_manager):
    mem = await lite_manager.process_turn("s1", 0, "the spare key is under the blue flowerpot")
    assert mem is not None  # curated (first turn)
    ranked = await lite_manager.replay_raw("where is the spare key", "default", 3)
    assert ranked == [mem.raw_log_id]
    trace = await lite_manager._retrieval.retrieve_detailed(
        "spare key", layers=["raw"], record_access=False,
    )
    assert trace.memories[0].kind == "raw"


# Promotion is judged on the threshold the runtime actually uses.
def test_runtime_threshold_is_the_trained_threshold():
    ctrl = PolicyController(config=make_config())
    model = LogisticDecisionModel(list(FEATURE_NAMES), [0.0] * len(FEATURE_NAMES),
                                  [1.0] * len(FEATURE_NAMES), [0.0] * len(FEATURE_NAMES),
                                  bias=0.2, threshold=0.5)  # p = 0.55
    ctrl.set_model(model)
    feats = build_save_features("I live in Leeds", recent_save_rate=0.9)
    verdict = ctrl.decide_save(feats)
    assert verdict.threshold == 0.5 and verdict.save


# Replay ranks only turns that existed when the query was made.
async def test_replay_ignores_turns_logged_after_the_query(lite_manager):
    await lite_manager.process_turn("s1", 0, "hello")
    await lite_manager.process_turn("s1", 1, "my locker code is 4471")
    await lite_manager.retrieve("what is my locker code", session_id="s1")
    # Logged afterwards and a better lexical match for the query.
    await lite_manager.process_turn("s1", 2, "what is my locker code")
    await assess_save_outcomes(lite_manager.sqlite, lookback_days=0, config=lite_manager.config)
    async with lite_manager.sqlite.db.execute(
        "SELECT turn, outcome_useful FROM save_decisions ORDER BY turn"
    ) as cur:
        labels = {r["turn"]: r["outcome_useful"] for r in await cur.fetchall()}
    assert labels[1] == 1 and labels[2] == 0


# Time windows are applied before layer limits.
async def test_time_window_finds_matches_beyond_layer_depth(tmp_path):
    from agent_memory import MemoryManager

    mgr = MemoryManager(data_dir=tmp_path, config={
        "llm_enabled": False, "policy_auto_train": False, "layer_depth": 2,
        "retrieval_layers": ["raw"],
    })
    await mgr.initialize(load_embeddings=False)
    try:
        for i in range(5):
            await mgr.process_turn("s", i, f"holiday plans holiday {i}", timestamp=f"2024-0{i + 1}-01T00:00:00Z")
        await mgr.process_turn("s", 9, "holiday plans", timestamp="2020-06-01T00:00:00Z")
        hits = await mgr.retrieve("holiday plans", until="2021-01-01T00:00:00Z")
        assert [h.content for h in hits] == ["holiday plans"]
    finally:
        await mgr.close()


# Non-Latin scripts are searchable.
async def test_non_latin_queries_match(lite_manager):
    await lite_manager.process_turn("s1", 0, "Я живу в Москве")
    hits = await lite_manager.retrieve("Москве")
    assert hits and "Москве" in hits[0].content
    assert fts_query("東京") is not None


# Deduplication keeps distinct non-Latin memories and contradictory facts.
def test_dedupe_keys_preserve_non_latin_text():
    assert _normalised("Я живу в Москве") != _normalised("東京に住んでいます")
    assert _normalised("東京に住んでいます")


def test_negation_is_not_a_duplicate():
    a = Memory(content="I am allergic to penicillin", raw_log_id="a")
    b = Memory(content="I am not allergic to penicillin", raw_log_id="b")
    c = Memory(content="I am allergic to penicillin.", raw_log_id="c")
    assert not _near_duplicate(a, b, 0.9)
    assert _near_duplicate(a, c, 0.9)


async def test_dedupe_cycle_keeps_contradictions_and_non_latin(lite_manager):
    for i, text in enumerate([
        "I am allergic to penicillin", "I am not allergic to penicillin",
        "Я живу в Москве", "東京に住んでいます",
    ]):
        await lite_manager.sqlite.save_memory(
            Memory(content=text, raw_log_id=f"r{i}", session_id=f"s{i}", turn=0, salience=0.4),
        )
    result = await lite_manager.run_compaction()
    assert result.memories_pruned == 0


# Time-based follow-ups stay within the namespace.
async def test_followups_are_namespace_scoped(lite_manager):
    s = lite_manager.sqlite
    common = dict(session_id="", turn=None, layers_queried=["raw"], graph_depth=2,
                  mood_weight=0.0, top_k=5, memory_ids=["m"], return_count=1)
    await s.log_retrieval_decision(decision_id="a1", query="gate code", namespace="a",
                                   decided_at="2026-01-01T10:00:00+00:00", **common)
    await s.log_retrieval_decision(decision_id="b1", query="gate code again", namespace="b",
                                   decided_at="2026-01-01T10:01:00+00:00", **common)
    assert await s.get_retrieval_followups_by_time("", "2026-01-01T10:00:00+00:00", 10, "a1", namespace="a") == []
    assert await s.get_retrieval_followups_by_time("", "2026-01-01T10:00:00+00:00", 10, "a1", namespace="b") == ["gate code again"]
