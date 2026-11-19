"""Save-decision models (A5).

A decision model maps a feature vector describing one logged turn to the
probability that the turn will be needed later. The policy controller turns
that probability into a save/skip decision, applying hard rules first.

Three models are provided:

  - :class:`LLMHeuristicModel` — the original behaviour: trust the LLM curator's
    ``should_save`` and ``confidence``.
  - :class:`LexicalHeuristicModel` — a hand-weighted logistic prior over cheap
    lexical features, used when no LLM is available (previously every turn but
    the first was dropped in that case).
  - :class:`LogisticDecisionModel` — learned from logged decisions and their
    outcome labels (see ``agent_memory.policy.training``). Pure numpy, so it
    adds no dependency and trains in milliseconds; serialised as JSON.

Every feature is computed at decision time and logged with the decision, so the
learned model is trained on exactly the inputs it will see in deployment.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from agent_memory import text as T

FEATURE_NAMES: list[str] = [
    "log_length",
    "n_terms",
    "has_number",
    "has_date",
    "first_person_fact",
    "preference",
    "question",
    "acknowledgement",
    "is_user",
    "is_first_turn",
    "turn_position",
    "novelty",
    "gap_overlap",
    "recent_save_rate",
    "explicit_marker",
    "llm_available",
    "llm_should_save",
    "llm_confidence",
    "llm_salience",
    "arousal",
    "surprise",
]


def build_save_features(
    content: str,
    *,
    role: str = "assistant",
    turn: int = 0,
    is_first_turn: bool = False,
    novelty: float = 1.0,
    gap_overlap: float = 0.0,
    recent_save_rate: float = 0.0,
    explicit_marker: bool = False,
    llm: dict | None = None,
) -> dict[str, float]:
    """Feature vector for one turn. Values are roughly in [0, 1.5]."""
    terms = T.content_terms(content)
    llm = llm or {}
    has_llm = bool(llm)
    return {
        "log_length": min(math.log1p(len(content or "")) / 7.0, 1.5),
        "n_terms": min(len(terms) / 20.0, 1.5),
        "has_number": float(T.has_number(content)),
        "has_date": float(T.has_date_reference(content)),
        "first_person_fact": float(T.has_first_person_fact(content)),
        "preference": float(T.has_preference(content)),
        "question": float(T.is_question(content)),
        "acknowledgement": float(T.is_acknowledgement(content)),
        "is_user": float(role == "user"),
        "is_first_turn": float(is_first_turn),
        "turn_position": min(math.log1p(max(turn, 0)) / 5.0, 1.5),
        "novelty": float(min(max(novelty, 0.0), 1.0)),
        "gap_overlap": float(min(max(gap_overlap, 0.0), 1.0)),
        "recent_save_rate": float(min(max(recent_save_rate, 0.0), 1.0)),
        "explicit_marker": float(explicit_marker),
        "llm_available": float(has_llm),
        "llm_should_save": float(bool(llm.get("should_save", False))) if has_llm else 0.0,
        "llm_confidence": _clip01(llm.get("confidence", 0.0)) if has_llm else 0.0,
        "llm_salience": _clip01(llm.get("salience", 0.0)) if has_llm else 0.0,
        "arousal": _clip01(llm.get("arousal", 0.0)) if has_llm else 0.0,
        "surprise": _clip01(llm.get("surprise", 0.0)) if has_llm else 0.0,
    }


def _clip01(v) -> float:
    try:
        return float(min(max(float(v), 0.0), 1.0))
    except (TypeError, ValueError):
        return 0.0


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class DecisionModel(Protocol):
    """Anything that scores a feature dict with a save probability."""

    name: str
    threshold: float

    def predict_proba(self, features: dict[str, float]) -> float: ...


@dataclass
class LLMHeuristicModel:
    """The original policy: save when the LLM says so with enough confidence."""

    threshold: float = 0.5
    name: str = "heuristic:llm"

    def predict_proba(self, features: dict[str, float]) -> float:
        if not features.get("llm_should_save"):
            return 0.0
        return float(features.get("llm_confidence", 0.0))


@dataclass
class LexicalHeuristicModel:
    """Hand-set logistic prior over LLM-free features.

    Favours self-disclosures, preferences, numbers and dates, and longer novel
    content; disfavours greetings/acknowledgements and bare questions. The
    weights are a prior, not fitted — the learned model replaces this once
    there is enough labelled data.
    """

    threshold: float = 0.5
    name: str = "heuristic:lexical"
    weights: dict[str, float] = field(default_factory=lambda: {
        "first_person_fact": 1.0,
        "preference": 0.8,
        "has_number": 0.6,
        "has_date": 0.6,
        "n_terms": 1.2,
        "novelty": 0.8,
        "acknowledgement": -2.5,
        "question": -0.6,
    })
    bias: float = -1.2

    def predict_proba(self, features: dict[str, float]) -> float:
        z = self.bias + sum(w * features.get(k, 0.0) for k, w in self.weights.items())
        return float(_sigmoid(z))


@dataclass
class LogisticDecisionModel:
    """L2-regularised logistic regression over :data:`FEATURE_NAMES`."""

    feature_names: list[str]
    mean: list[float]
    std: list[float]
    weights: list[float]
    bias: float
    threshold: float = 0.5
    version: str = ""
    trained_at: str = ""
    metrics: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"learned:{self.version or 'unversioned'}"

    def _matrix(self, rows: list[dict[str, float]]) -> np.ndarray:
        x = np.array([[r.get(f, 0.0) for f in self.feature_names] for r in rows], dtype=float)
        return (x - np.array(self.mean)) / np.array(self.std)

    def predict_proba_batch(self, rows: list[dict[str, float]]) -> np.ndarray:
        if not rows:
            return np.zeros(0)
        return _sigmoid(self._matrix(rows) @ np.array(self.weights) + self.bias)

    def predict_proba(self, features: dict[str, float]) -> float:
        return float(self.predict_proba_batch([features])[0])

    @classmethod
    def fit(
        cls,
        rows: list[dict[str, float]],
        labels: list[int],
        feature_names: list[str] | None = None,
        l2: float = 1.0,
        balanced: bool = True,
        max_iter: int = 50,
    ) -> "LogisticDecisionModel":
        """Fit with Newton–Raphson (IRLS). Class-balanced sample weights by default."""
        names = list(feature_names or FEATURE_NAMES)
        x_raw = np.array([[r.get(f, 0.0) for f in names] for r in rows], dtype=float)
        y = np.array(labels, dtype=float)
        mean = x_raw.mean(axis=0)
        std = x_raw.std(axis=0)
        std[std < 1e-9] = 1.0  # constant features stay at zero after scaling
        x = (x_raw - mean) / std
        n, d = x.shape
        xb = np.hstack([x, np.ones((n, 1))])
        if balanced and 0 < y.sum() < n:
            pos = y.sum()
            sw = np.where(y == 1, n / (2 * pos), n / (2 * (n - pos)))
        else:
            sw = np.ones(n)
        beta = np.zeros(d + 1)
        reg = np.full(d + 1, l2)
        reg[-1] = 0.0  # do not penalise the intercept
        for _ in range(max_iter):
            p = _sigmoid(xb @ beta)
            grad = xb.T @ (sw * (p - y)) + reg * beta
            w = sw * p * (1 - p)
            hess = (xb * w[:, None]).T @ xb + np.diag(reg)
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hess, grad, rcond=None)[0]
            beta -= step
            if np.max(np.abs(step)) < 1e-6:
                break
        return cls(
            feature_names=names, mean=mean.tolist(), std=std.tolist(),
            weights=beta[:-1].tolist(), bias=float(beta[-1]),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LogisticDecisionModel":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "LogisticDecisionModel":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def explain(self, top: int = 8) -> list[tuple[str, float]]:
        """Largest standardised coefficients (feature, weight)."""
        pairs = sorted(zip(self.feature_names, self.weights), key=lambda kv: -abs(kv[1]))
        return [(k, round(v, 3)) for k, v in pairs[:top]]
