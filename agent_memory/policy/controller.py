"""Policy controller (A5): the single place save decisions are made.

Order of precedence for a save decision:

  1. Hard rules (cannot be overridden by any model):
     explicit "remember this"-style instructions, the first turn of a session
     (when ``always_save_first_turn``), and high-arousal + high-surprise events.
  2. The active decision model:
     - a promoted learned model (``LogisticDecisionModel``), if one exists;
     - otherwise the LLM heuristic when an LLM evaluation is available;
     - otherwise the lexical heuristic.

A skipped turn is not lost: every turn is logged and indexed in the raw-turn
index, so "skip" means "not curated into a memory", and outcome labels can be
computed for skips as well as saves.

Retrieval and compaction parameters are still heuristic; their decisions and
outcomes are logged (``retrieval_decisions``/``retrieval_hits``) for a future
learned retrieval policy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from agent_memory.config import MEMORY_CONFIG
from agent_memory.policy.decision_model import (
    DecisionModel,
    LexicalHeuristicModel,
    LLMHeuristicModel,
    LogisticDecisionModel,
)

logger = logging.getLogger(__name__)

POLICY_HARD_CONSTRAINTS = {
    # Save constraints
    "min_save_rate": 0.05,
    "max_save_rate": 0.50,
    "fast_path_override": True,

    # Retrieval constraints
    "min_layers": 1,
    "max_graph_depth": 4,
    "max_top_k": 20,

    # Compaction constraints
    "never_delete_raw_logs": True,
    "never_compact_fast_path_gen0": True,
    "require_merge_validation": True,
}

ACTIVE_MODEL_FILE = "save_model.json"
CANDIDATE_MODEL_FILE = "save_model.candidate.json"


@dataclass
class PolicyState:
    """State vector for policy decisions (A5.2)."""
    turn_number: int = 0
    session_length: int = 0
    time_since_last_save: float = 0.0
    content_length: int = 0
    emotional_valence: float = 0.0
    emotional_arousal: float = 0.0
    emotional_surprise: float = 0.0
    hot_tier_count: int = 0
    recent_retrieval_hit_rate: float = 0.0
    retrieval_gap_score: float = 0.0
    graph_node_count: int = 0
    avg_edge_degree: float = 0.0
    orphan_memory_count: int = 0
    days_since_last_compaction: float = 0.0
    pending_merge_candidates: int = 0


@dataclass
class PolicyDecision:
    """Outcome of :meth:`PolicyController.decide_save`."""
    save: bool
    decision: str          # "save" | "skip" | "fast_path"
    prob: float
    threshold: float
    source: str            # e.g. "rule:explicit", "heuristic:llm", "learned:<version>"
    reason: str


class PolicyController:
    """Applies hard rules, then the active decision model."""

    def __init__(self, config: dict | None = None, model_dir: Path | None = None) -> None:
        self.constraints = POLICY_HARD_CONSTRAINTS
        self.config = config or MEMORY_CONFIG
        self.model_dir = model_dir
        self.llm_model = LLMHeuristicModel(threshold=self.config["save_confidence_threshold"])
        self.lexical_model = LexicalHeuristicModel()
        self.learned: LogisticDecisionModel | None = None
        self.reload()

    # ── Model management ──

    @property
    def active_model_path(self) -> Path | None:
        return self.model_dir / ACTIVE_MODEL_FILE if self.model_dir else None

    def reload(self) -> None:
        """(Re)load the promoted learned model, if any."""
        self.learned = None
        path = self.active_model_path
        if path and path.exists():
            try:
                self.learned = LogisticDecisionModel.load(path)
                logger.info("Loaded learned save model %s", self.learned.name)
            except Exception:
                logger.exception("Could not load learned save model at %s; using heuristics", path)

    def set_model(self, model: LogisticDecisionModel | None) -> None:
        self.learned = model

    @property
    def active_model(self) -> DecisionModel:
        return self.learned or self.llm_model

    # ── Save decisions ──

    def decide_save(
        self, features: dict[str, float], llm_threshold: float | None = None,
    ) -> PolicyDecision:
        cfg = self.config
        if features.get("explicit_marker"):
            return PolicyDecision(True, "fast_path", 1.0, 0.0, "rule:explicit",
                                  "Explicit instruction to remember")
        if features.get("is_first_turn") and cfg.get("always_save_first_turn", True):
            return PolicyDecision(True, "fast_path", 1.0, 0.0, "rule:first_turn",
                                  "First turn of session — always saved")
        if (features.get("llm_available")
                and features.get("arousal", 0.0) > cfg["fast_path_arousal"]
                and features.get("surprise", 0.0) > cfg["fast_path_surprise"]):
            return PolicyDecision(True, "fast_path", 1.0, 0.0, "rule:emotional",
                                  "High arousal and surprise")

        model: DecisionModel
        if self.learned is not None:
            model = self.learned
            # Save-rate bounds are enforced when the threshold is chosen in
            # training, so the threshold evaluated by the promotion gate is the
            # threshold used here.
            threshold = self.learned.threshold
        elif features.get("llm_available"):
            model = self.llm_model
            threshold = cfg["save_confidence_threshold"] if llm_threshold is None else llm_threshold
        else:
            model = self.lexical_model
            threshold = self.lexical_model.threshold

        prob = model.predict_proba(features)
        save = prob >= threshold
        return PolicyDecision(
            save=save, decision="save" if save else "skip", prob=round(prob, 4),
            threshold=round(threshold, 4), source=model.name,
            reason=f"{model.name}: p={prob:.2f} {'>=' if save else '<'} {threshold:.2f}",
        )

    # ── Legacy / heuristic interfaces ──

    def should_save(self, state: PolicyState, llm_confidence: float) -> bool:
        """Decide whether to save from LLM confidence alone (heuristic — v1)."""
        return llm_confidence >= self.config["save_confidence_threshold"]

    def retrieval_config(self, state: PolicyState) -> dict:
        """Return retrieval parameters (heuristic)."""
        return {
            "layers": self.config["retrieval_layers"],
            "graph_depth": min(
                self.config["graph_traversal_depth"],
                self.constraints["max_graph_depth"],
            ),
            "mood_weight": self.config["mood_congruent_weight"],
            "top_k": min(
                self.config.get("retrieval_top_k", self.config["top_k_per_layer"]),
                self.constraints["max_top_k"],
            ),
        }

    def compaction_priority(self, state: PolicyState) -> float:
        """Return compaction urgency score (heuristic)."""
        if state.hot_tier_count > self.config["hot_tier_threshold"]:
            return 1.0
        return state.hot_tier_count / max(self.config["hot_tier_threshold"], 1)
