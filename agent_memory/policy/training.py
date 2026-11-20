"""Training and promotion of the learned save-decision model (A5).

Pipeline:

  1. Build a dataset from ``save_decisions`` rows that have logged features and
     an outcome label (see ``outcome_assessor``).
  2. Split by time: the oldest ``1 - policy_validation_fraction`` trains, the
     newest validates — the model is always judged on decisions it could not
     have seen, in the order they would have happened.
  3. Fit an L2 logistic regression; choose its threshold on validation to
     maximise F-beta (recall-weighted, beta = ``policy_fbeta``) within the
     save-rate bounds.
  4. Evaluate the full policy (hard rules + model) against the policy that
     actually made those decisions, on the same validation rows.
  5. Promote (write ``save_model.json``) only if the learned policy beats the
     deployed one by ``policy_promotion_margin`` in F-beta and has AUC of at
     least ``policy_min_auc``. Otherwise the model is kept as a candidate.

Every run is recorded in ``policy_training_runs``.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from agent_memory.config import MEMORY_CONFIG
from agent_memory.policy.controller import (
    ACTIVE_MODEL_FILE,
    CANDIDATE_MODEL_FILE,
    POLICY_HARD_CONSTRAINTS,
)
from agent_memory.policy.decision_model import FEATURE_NAMES, LogisticDecisionModel
from agent_memory.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


@dataclass
class TrainingReport:
    trained: bool
    promoted: bool
    reason: str
    n_train: int = 0
    n_validation: int = 0
    metrics: dict = field(default_factory=dict)
    model_path: str | None = None
    top_features: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trained": self.trained, "promoted": self.promoted, "reason": self.reason,
            "n_train": self.n_train, "n_validation": self.n_validation,
            "metrics": self.metrics, "model_path": self.model_path,
            "top_features": self.top_features,
        }


# ── Metrics ──

def roc_auc(scores, labels) -> float | None:
    """Rank-based AUC (Mann–Whitney U). None if only one class is present."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    if pos == 0 or neg == 0:
        return None
    order = s.argsort()
    ranks = np.empty(len(s))
    # average ranks for ties
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def binary_metrics(pred, labels, beta: float = 2.0) -> dict:
    p = np.asarray(pred, dtype=int)
    y = np.asarray(labels, dtype=int)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta * beta
    fbeta = (1 + b2) * precision * recall / (b2 * precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4), "recall": round(recall, 4),
        f"f{beta:g}": round(fbeta, 4), "fbeta": round(fbeta, 4),
        "save_rate": round(float(p.mean()) if len(p) else 0.0, 4),
    }


def forced_save(features: dict, config: dict) -> bool:
    """Hard-rule saves that any policy makes regardless of its model."""
    if features.get("explicit_marker"):
        return True
    if features.get("is_first_turn") and config.get("always_save_first_turn", True):
        return True
    return bool(
        features.get("llm_available")
        and features.get("arousal", 0.0) > config["fast_path_arousal"]
        and features.get("surprise", 0.0) > config["fast_path_surprise"]
    )


def choose_threshold(probs, labels, forced, beta: float, bounds: tuple[float, float]) -> float:
    """Threshold maximising F-beta of the full policy, preferring save rates
    within ``bounds``."""
    probs = np.asarray(probs)
    best_in, best_any = (-1.0, 0.5), (-1.0, 0.5)
    for t in np.unique(np.concatenate([np.linspace(0.05, 0.95, 19), probs])):
        pred = np.where(forced, 1, (probs >= t).astype(int))
        m = binary_metrics(pred, labels, beta)
        cand = (m["fbeta"], float(t))
        best_any = max(best_any, cand)
        if bounds[0] <= m["save_rate"] <= bounds[1]:
            best_in = max(best_in, cand)
    return (best_in if best_in[0] >= 0 else best_any)[1]


# ── Dataset ──

async def load_save_dataset(sqlite: SQLiteStore) -> list[dict]:
    """Labelled decisions with logged features, oldest first."""
    rows = await sqlite.export_save_policy_data()
    return [r for r in rows if r.get("features")]


def train_from_rows(
    rows: list[dict],
    config: dict | None = None,
    force: bool = False,
    min_examples: int | None = None,
) -> tuple[LogisticDecisionModel | None, TrainingReport]:
    """Fit, evaluate and decide promotion for a dataset (pure function)."""
    cfg = config or MEMORY_CONFIG
    min_examples = cfg["policy_min_save_examples"] if min_examples is None else min_examples
    beta = cfg.get("policy_fbeta", 2.0)
    labels_all = [int(r["outcome_useful"]) for r in rows]
    n_pos = sum(labels_all)
    n_neg = len(labels_all) - n_pos
    min_class = cfg.get("policy_min_class_examples", 20)

    if len(rows) < min_examples and not force:
        return None, TrainingReport(False, False, f"{len(rows)} labelled examples < {min_examples} required")
    if min(n_pos, n_neg) < min_class:
        return None, TrainingReport(False, False, f"class balance too thin (pos={n_pos}, neg={n_neg})")

    rows = sorted(rows, key=lambda r: r.get("decided_at") or "")
    n_val = max(int(len(rows) * cfg.get("policy_validation_fraction", 0.25)), 1)
    train, val = rows[:-n_val], rows[-n_val:]
    y_train = [int(r["outcome_useful"]) for r in train]
    y_val = [int(r["outcome_useful"]) for r in val]
    if len(set(y_train)) < 2:
        return None, TrainingReport(False, False, "training split has a single class")

    model = LogisticDecisionModel.fit([r["features"] for r in train], y_train, FEATURE_NAMES)
    val_features = [r["features"] for r in val]
    probs = model.predict_proba_batch(val_features)
    forced = np.array([forced_save(f, cfg) for f in val_features])
    bounds = (POLICY_HARD_CONSTRAINTS["min_save_rate"], POLICY_HARD_CONSTRAINTS["max_save_rate"])

    # Choose the threshold on the *training* split so validation stays held out.
    train_probs = model.predict_proba_batch([r["features"] for r in train])
    train_forced = np.array([forced_save(r["features"], cfg) for r in train])
    model.threshold = choose_threshold(train_probs, y_train, train_forced, beta, bounds)

    learned_pred = np.where(forced, 1, (probs >= model.threshold).astype(int))
    deployed_pred = np.array([int(r["decision"] in ("save", "fast_path")) for r in val])
    deployed_scores = np.array([
        r["policy_prob"] if r.get("policy_prob") is not None else float(d)
        for r, d in zip(val, deployed_pred)
    ])

    learned_m = binary_metrics(learned_pred, y_val, beta)
    deployed_m = binary_metrics(deployed_pred, y_val, beta)
    learned_m["auc"] = roc_auc(probs, y_val)
    deployed_m["auc"] = roc_auc(deployed_scores, y_val)
    metrics = {
        "learned": learned_m, "deployed": deployed_m,
        "threshold": round(model.threshold, 4),
        "validation_positive_rate": round(float(np.mean(y_val)), 4),
        "n_pos": n_pos, "n_neg": n_neg,
    }
    model.metrics = metrics
    model.version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    model.trained_at = datetime.now(timezone.utc).isoformat()

    margin = cfg.get("policy_promotion_margin", 0.01)
    auc_ok = learned_m["auc"] is not None and learned_m["auc"] >= cfg.get("policy_min_auc", 0.6)
    beats = learned_m["fbeta"] >= deployed_m["fbeta"] + margin
    promote = auc_ok and beats
    if promote:
        reason = (f"promoted: F{beta:g} {learned_m['fbeta']:.3f} vs deployed "
                  f"{deployed_m['fbeta']:.3f}, AUC {learned_m['auc']:.3f}")
    else:
        reason = (f"not promoted: F{beta:g} {learned_m['fbeta']:.3f} vs deployed "
                  f"{deployed_m['fbeta']:.3f} (margin {margin}), AUC {learned_m['auc']}")
    return model, TrainingReport(
        trained=True, promoted=promote, reason=reason, n_train=len(train),
        n_validation=len(val), metrics=metrics, top_features=model.explain(),
    )


async def train_save_policy(
    sqlite: SQLiteStore,
    model_dir: Path,
    config: dict | None = None,
    force: bool = False,
    min_examples: int | None = None,
) -> TrainingReport:
    """Train from the store, write the model, record the run.

    ``force`` trains below ``policy_min_save_examples`` (it never bypasses the
    promotion gate).
    """
    cfg = config or MEMORY_CONFIG
    rows = await load_save_dataset(sqlite)
    model, report = train_from_rows(rows, cfg, force=force, min_examples=min_examples)
    now = datetime.now(timezone.utc).isoformat()
    if model is not None:
        path = model_dir / (ACTIVE_MODEL_FILE if report.promoted else CANDIDATE_MODEL_FILE)
        model.save(path)
        report.model_path = str(path)
    await sqlite.log_policy_training_run(
        run_id=str(uuid.uuid4()), trained_at=now, model_type="logistic",
        n_train=report.n_train, n_validation=report.n_validation,
        metrics=report.metrics, promoted=report.promoted,
        model_path=report.model_path, notes=report.reason,
    )
    logger.info("Save-policy training: %s", report.reason)
    return report


async def should_retrain(sqlite: SQLiteStore, config: dict | None = None) -> bool:
    """Enough labelled data, and at least 20% more than at the last run."""
    cfg = config or MEMORY_CONFIG
    n = await sqlite.count_labelled_save_decisions()
    if n < cfg["policy_min_save_examples"]:
        return False
    last = await sqlite.last_policy_training_run()
    if last is None:
        return True
    previous = (last.get("n_train") or 0) + (last.get("n_validation") or 0)
    return n >= previous * 1.2


def format_report(report: TrainingReport) -> str:
    return json.dumps(report.to_dict(), indent=2)
