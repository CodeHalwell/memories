"""Tests for the save-decision model, labelling, training and promotion (A5)."""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone

import numpy as np

from agent_memory.config import make_config
from agent_memory.policy.controller import PolicyController
from agent_memory.policy.decision_model import (
    FEATURE_NAMES,
    LexicalHeuristicModel,
    LogisticDecisionModel,
    build_save_features,
)
from agent_memory.policy.outcome_assessor import assess_save_outcomes
from agent_memory.policy.training import (
    binary_metrics,
    roc_auc,
    train_from_rows,
    train_save_policy,
)


def _synthetic_rows(n=400, seed=0, deployed_good=False):
    """Rows whose usefulness depends on first_person_fact and n_terms."""
    rng = random.Random(seed)
    rows = []
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        fact = rng.random() < 0.4
        terms = rng.random()
        useful = int(fact and terms > 0.3 or rng.random() < 0.05)
        feats = build_save_features("x " * int(terms * 30), turn=i)
        feats["first_person_fact"] = float(fact)
        feats["n_terms"] = terms
        decision = ("save" if useful else "skip") if deployed_good else ("save" if rng.random() < 0.5 else "skip")
        rows.append({
            "features": feats, "outcome_useful": useful, "decision": decision,
            "policy_prob": 0.5, "decided_at": (t0 + timedelta(minutes=i)).isoformat(),
        })
    return rows


def test_features_are_complete_and_bounded():
    f = build_save_features("I moved to Leeds on 3 May 2024, I love it!", role="user", turn=12,
                            llm={"should_save": True, "confidence": 0.8})
    assert set(f) == set(FEATURE_NAMES)
    assert all(-0.01 <= v <= 1.5 for v in f.values())
    assert f["first_person_fact"] == f["has_date"] == f["preference"] == f["is_user"] == 1.0
    assert f["llm_available"] == 1.0 and f["llm_confidence"] == 0.8


def test_lexical_prior_ranks_facts_above_acknowledgements():
    m = LexicalHeuristicModel()
    fact = m.predict_proba(build_save_features("My sister Jo lives in Glasgow and works as a nurse"))
    ack = m.predict_proba(build_save_features("ok thanks!"))
    assert fact > m.threshold > ack


def test_logistic_fit_learns_signal_and_roundtrips(tmp_path):
    rows = _synthetic_rows()
    model = LogisticDecisionModel.fit([r["features"] for r in rows], [r["outcome_useful"] for r in rows])
    probs = model.predict_proba_batch([r["features"] for r in rows])
    assert roc_auc(probs, [r["outcome_useful"] for r in rows]) > 0.85
    path = tmp_path / "m.json"
    model.save(path)
    again = LogisticDecisionModel.load(path)
    assert np.allclose(again.predict_proba_batch([rows[0]["features"]]), probs[:1])


def test_metrics_helpers():
    assert roc_auc([0.1, 0.9], [0, 1]) == 1.0
    assert roc_auc([0.5, 0.5], [0, 1]) == 0.5
    assert roc_auc([0.2, 0.3], [1, 1]) is None
    m = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0], beta=1.0)
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_controller_hard_rules_beat_any_model():
    ctrl = PolicyController(config=make_config())
    never = LogisticDecisionModel(list(FEATURE_NAMES), [0.0] * len(FEATURE_NAMES),
                                  [1.0] * len(FEATURE_NAMES), [0.0] * len(FEATURE_NAMES), -50.0)
    ctrl.set_model(never)
    explicit = build_save_features("remember this: gate code 1234", explicit_marker=True)
    assert ctrl.decide_save(explicit).decision == "fast_path"
    plain = build_save_features("I live in Leeds")
    verdict = ctrl.decide_save(plain)
    assert verdict.decision == "skip" and verdict.source.startswith("learned:")


def test_controller_falls_back_by_availability():
    ctrl = PolicyController(config=make_config())
    no_llm = build_save_features("I live in Leeds")
    assert ctrl.decide_save(no_llm).source == "heuristic:lexical"
    with_llm = build_save_features("I live in Leeds", llm={"should_save": True, "confidence": 0.9})
    v = ctrl.decide_save(with_llm)
    assert v.source == "heuristic:llm" and v.save


def test_promotion_gate_promotes_only_when_better():
    cfg = make_config({"policy_min_class_examples": 5})
    # Deployed policy was random -> a model that learns the signal beats it.
    model, report = train_from_rows(_synthetic_rows(deployed_good=False), cfg, min_examples=50)
    assert report.trained and report.promoted, report.reason
    # Deployed policy was already perfect -> nothing to gain, no promotion.
    _, report = train_from_rows(_synthetic_rows(deployed_good=True), cfg, min_examples=50)
    assert report.trained and not report.promoted


def test_training_refuses_thin_data():
    cfg = make_config()
    model, report = train_from_rows(_synthetic_rows(n=50), cfg)
    assert model is None and not report.trained


async def test_outcomes_label_saves_and_skips_by_replay(lite_manager):
    await lite_manager.process_turn("s1", 0, "hello")
    await lite_manager.process_turn("s1", 1, "ok, the spare key is under the blue flowerpot")
    await lite_manager.process_turn("s1", 2, "ok thanks!")
    await lite_manager.retrieve("where is the spare key", session_id="s2")
    labelled = await assess_save_outcomes(lite_manager.sqlite, lookback_days=0, config=lite_manager.config)
    assert labelled == 3
    async with lite_manager.sqlite.db.execute(
        "SELECT turn, decision, outcome_useful FROM save_decisions ORDER BY turn"
    ) as cur:
        rows = {r["turn"]: dict(r) for r in await cur.fetchall()}
    assert rows[1]["outcome_useful"] == 1          # needed later, whether or not it was saved
    assert rows[2]["outcome_useful"] == 0


async def test_unused_decisions_wait_for_the_window(lite_manager):
    await lite_manager.process_turn("s1", 0, "hello")
    assert await assess_save_outcomes(lite_manager.sqlite, lookback_days=14) == 0


async def test_record_response_marks_used_items(lite_manager):
    await lite_manager.process_turn("s1", 0, "The dentist appointment is on Tuesday at 3pm with Dr Patel")
    hits = await lite_manager.retrieve("dentist appointment")
    rows = await lite_manager.record_response(
        hits[0].retrieval_id, "Your dentist appointment with Dr Patel is Tuesday at 3pm.",
    )
    assert rows[0]["used"] is True
    assert hits[0].raw_log_id in await lite_manager.sqlite.get_used_raw_log_ids("default")


async def test_train_policy_end_to_end_writes_candidate_and_audit(lite_manager):
    rng = random.Random(1)
    for i in range(60):
        text = (f"My colleague number {i} works in Leeds on project {i}" if rng.random() < 0.5
                else "ok thanks")
        await lite_manager.process_turn(f"s{i}", 1, "hi")
        await lite_manager.process_turn(f"s{i}", 2, text)
    await lite_manager.retrieve("colleague works Leeds project")
    await assess_save_outcomes(lite_manager.sqlite, lookback_days=0, config=lite_manager.config)
    lite_manager.config["policy_min_class_examples"] = 5
    report = await train_save_policy(
        lite_manager.sqlite, lite_manager.policy_dir, config=lite_manager.config, force=True,
    )
    assert report.trained
    assert report.model_path and json.loads(open(report.model_path).read())["feature_names"]
    run = await lite_manager.sqlite.last_policy_training_run()
    assert run is not None and run["promoted"] == int(report.promoted)
