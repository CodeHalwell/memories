"""LoCoMo retrieval evaluation — LLM-free, so it runs anywhere.

LoCoMo (Maharana et al., 2024) annotates every question with the dialogue
turns that contain its evidence. That allows a retrieval-only metric that needs
no answer model or judge: **evidence recall@k** — the fraction of a question's
evidence turns found in the top-k retrieved items. Published LoCoMo QA scores
move by several points with the judge and answer prompt alone, so a
retrieval metric is the cleaner way to compare memory configurations.

Two experiments:

``retrieval``  Ingest each conversation turn by turn (in timestamp order, one
               namespace per conversation), then ask every non-adversarial
               question and score evidence recall@k for several system
               configurations, including the raw-history baseline the
               literature says every memory system must beat.

``policy``     Cross-validated evaluation of the save-decision model. The
               system labels its own decisions by replaying the questions
               against the raw-turn index (no access to the evidence
               annotations); a logistic model is trained on those labels for
               the training conversations and judged on held-out conversations
               against the *annotated* evidence turns. Reported against the
               lexical heuristic it would replace.

The dataset is not bundled (CC BY-NC 4.0). Download ``locomo10.json`` from
https://github.com/snap-research/locomo and pass its path.

Usage::

    python -m agent_memory.eval.locomo --data locomo10.json
    python -m agent_memory.eval.locomo --data locomo10.json --experiment policy
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Settings shared by every configuration: no LLM (the harness must run
# offline and be reproducible), no recency term (the conversations are
# historical, so wall-clock recency is meaningless), no automatic training.
BASE_OVERRIDES = {
    "llm_enabled": False,
    "recency_weight": 0.0,
    "mood_congruent_weight": 0.0,
    "policy_auto_train": False,
    "policy_logging_enabled": True,
    "reranker_model": None,   # systems opt in explicitly via "_reranker"
}

SYSTEMS: dict[str, dict] = {
    # Raw-history baselines: every turn indexed, no curation involved.
    "raw_bm25": {"retrieval_layers": ["raw"]},
    "raw_dense": {"retrieval_layers": ["raw_semantic"]},
    "raw_hybrid": {"retrieval_layers": ["raw", "raw_semantic"]},
    # The pre-change architecture's reach: only curated memories are searchable.
    "curated_only": {"retrieval_layers": ["keyword", "semantic", "entity"]},
    # Everything fused (the default configuration).
    "full": {},
    # With a cross-encoder reranking the fused top-50.
    "raw_hybrid_rerank": {"retrieval_layers": ["raw", "raw_semantic"], "_reranker": "minilm"},
    "full_rerank": {"_reranker": "minilm"},
}


@dataclass
class Turn:
    dia_id: str
    speaker: str
    text: str
    session: int
    timestamp: str


@dataclass
class QA:
    question: str
    answer: str
    evidence: list[str]
    category: int


@dataclass
class Conversation:
    sample_id: str
    turns: list[Turn]
    questions: list[QA] = field(default_factory=list)


CATEGORY_NAMES = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}

_DT_RE = re.compile(r"(\d{1,2}):(\d{2})\s*([ap]m)\s+on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", re.I)


def parse_locomo_datetime(value: str) -> str:
    """'1:56 pm on 8 May, 2023' -> '2023-05-08T13:56:00+00:00'."""
    m = _DT_RE.search(value or "")
    if not m:
        return datetime(2000, 1, 1, tzinfo=timezone.utc).isoformat()
    hour, minute, ampm, day, month, year = m.groups()
    hour = int(hour) % 12 + (12 if ampm.lower() == "pm" else 0)
    month_num = datetime.strptime(month[:3].title(), "%b").month
    return datetime(int(year), month_num, int(day), hour, int(minute), tzinfo=timezone.utc).isoformat()


def _norm_dia(d: str) -> str:
    m = re.match(r"D:?(\d+):(\d+)", str(d).strip())
    return f"D{int(m.group(1))}:{int(m.group(2))}" if m else str(d).strip()


def load_locomo(path: str | Path) -> list[Conversation]:
    data = json.loads(Path(path).read_text())
    convs = []
    for sample in data:
        c = sample["conversation"]
        sessions = sorted(
            int(k.split("_")[1]) for k in c
            if re.fullmatch(r"session_\d+", k) and isinstance(c[k], list)
        )
        turns: list[Turn] = []
        for s in sessions:
            ts = parse_locomo_datetime(c.get(f"session_{s}_date_time", ""))
            for t in c[f"session_{s}"]:
                text = t.get("text", "")
                if t.get("blip_caption"):
                    text = f"{text} [shares a photo: {t['blip_caption']}]"
                turns.append(Turn(_norm_dia(t["dia_id"]), t["speaker"], text, s, ts))
        qas = [
            QA(str(q.get("question", "")), str(q.get("answer", q.get("adversarial_answer", ""))),
               [_norm_dia(e) for e in q.get("evidence", [])], int(q.get("category", 0)))
            for q in sample.get("qa", [])
        ]
        convs.append(Conversation(sample["sample_id"], turns, qas))
    return convs


_EMBEDDERS: dict = {}
_RERANKERS: dict = {}


def make_embedder(name: str):
    """Embedders are cached so a model is loaded once per run."""
    if name not in _EMBEDDERS:
        if name == "hashing":
            from agent_memory.embeddings.base import HashingTextEmbedder

            _EMBEDDERS[name] = HashingTextEmbedder(dimension=512)
        else:
            from agent_memory.embeddings.text_embedder import TextEmbedder

            _EMBEDDERS[name] = TextEmbedder(name if name != "minilm" else "all-MiniLM-L6-v2")
    return _EMBEDDERS[name]


def make_reranker(name: str | None):
    if not name:
        return None
    if name not in _RERANKERS:
        from agent_memory.rerank import CrossEncoderReranker

        _RERANKERS[name] = CrossEncoderReranker(
            name if name != "minilm" else "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    return _RERANKERS[name]


async def build_manager(workdir: Path, overrides: dict, embedder, embeddings: bool, reranker=None):
    from agent_memory import MemoryManager, NullVisualEmbedder

    overrides = dict(overrides)
    reranker = make_reranker(overrides.pop("_reranker", None)) or reranker
    mgr = MemoryManager(
        data_dir=workdir, text_embedder=embedder, visual_embedder=NullVisualEmbedder(),
        config={**BASE_OVERRIDES, **overrides}, reranker=reranker,
    )
    await mgr.initialize(load_embeddings=embeddings)
    return mgr


async def ingest(mgr, conv: Conversation) -> dict[str, str]:
    """Feed a conversation turn by turn; return raw_log_id -> dia_id."""
    for i, t in enumerate(conv.turns):
        await mgr.process_turn(
            session_id=f"{conv.sample_id}-s{t.session}", turn=i,
            content=f"{t.speaker}: {t.text}", role=t.speaker,
            namespace=conv.sample_id, timestamp=t.timestamp,
        )
    by_turn = {i: t.dia_id for i, t in enumerate(conv.turns)}
    out: dict[str, str] = {}
    async with mgr.sqlite.db.execute(
        "SELECT id, turn FROM raw_log_index WHERE namespace = ?", (conv.sample_id,)
    ) as cur:
        async for row in cur:
            out[row["id"]] = by_turn[row["turn"]]
    return out


async def score_questions(mgr, conv: Conversation, raw_to_dia, ks: list[int]) -> list[dict]:
    kmax = max(ks)
    results = []
    for qa in conv.questions:
        if qa.category == 5 or not qa.evidence:
            continue
        hits = await mgr.retrieve(
            qa.question, session_id=f"{conv.sample_id}-eval", top_k=kmax, namespace=conv.sample_id,
        )
        ranked: list[str] = []
        for h in hits:
            dia = raw_to_dia.get(h.raw_log_id)
            if dia and dia not in ranked:
                ranked.append(dia)
        ev = set(qa.evidence)
        row = {"category": CATEGORY_NAMES.get(qa.category, str(qa.category))}
        for k in ks:
            found = ev & set(ranked[:k])
            row[f"recall@{k}"] = len(found) / len(ev)
            row[f"hit@{k}"] = float(bool(found))
        results.append(row)
    return results


def summarise(rows: list[dict], ks: list[int]) -> dict:
    out: dict = {"n": len(rows)}
    for k in ks:
        out[f"recall@{k}"] = round(float(np.mean([r[f"recall@{k}"] for r in rows])), 4) if rows else 0.0
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    out["by_category"] = {
        c: {"n": len(v), **{f"recall@{k}": round(float(np.mean([r[f"recall@{k}"] for r in v])), 4) for k in ks}}
        for c, v in sorted(by_cat.items())
    }
    return out


async def run_retrieval(convs, systems: list[str], ks: list[int], embedder_name: str) -> dict:
    results: dict = {}
    for name in systems:
        overrides = SYSTEMS[name]
        layers = overrides.get("retrieval_layers")
        needs_vectors = layers is None or any("semantic" in l for l in layers)
        rows: list[dict] = []
        saved = total = 0
        t0 = time.time()
        for conv in convs:
            with tempfile.TemporaryDirectory() as d:
                mgr = await build_manager(Path(d), overrides, make_embedder(embedder_name), needs_vectors)
                try:
                    raw_to_dia = await ingest(mgr, conv)
                    rows += await score_questions(mgr, conv, raw_to_dia, ks)
                    saved += await mgr.sqlite.count_memories()
                    total += len(conv.turns)
                finally:
                    await mgr.close()
        results[name] = {**summarise(rows, ks), "curated_fraction": round(saved / max(total, 1), 4),
                         "seconds": round(time.time() - t0, 1)}
        logger.info("%s: %s", name, {k: v for k, v in results[name].items() if k != "by_category"})
    return results


# ── Decision-model experiment ──

LABEL_SOURCES = ["replay@1", "replay@3", "replay@5", "rerank@1", "rerank@2", "rerank@3",
                 "attribution", "attribution+rerank@1"]


async def collect_policy_data(convs, embedder_name: str, rerank: bool = True) -> dict[str, dict]:
    """Ingest with the default (heuristic) policy, ask every question, record
    the gold answer as the agent's response (simulating reader feedback via
    ``record_response``), and collect per-decision features with several
    candidate labels plus the annotated-evidence (oracle) label."""
    out: dict[str, dict] = {}
    for conv in convs:
        with tempfile.TemporaryDirectory() as d:
            mgr = await build_manager(Path(d), {}, make_embedder(embedder_name), True)
            try:
                raw_to_dia = await ingest(mgr, conv)
                # Ask every question (as a deployed agent would), recording
                # the gold answer as its response — simulated reader feedback.
                for qa in conv.questions:
                    if qa.category == 5 or not qa.evidence:
                        continue
                    hits = await mgr.retrieve(qa.question, session_id=f"{conv.sample_id}-eval",
                                              top_k=10, namespace=conv.sample_id)
                    if hits:
                        await mgr.record_response(hits[0].retrieval_id, qa.answer)
                # Best replay rank of each turn over all (later) queries.
                best_rank: dict[str, int] = {}
                for q in await mgr.sqlite.get_retrieval_queries(namespace=conv.sample_id):
                    for rank, (rid, _s) in enumerate(
                        await mgr.sqlite.search_raw_fts(q["query"], limit=5, namespace=conv.sample_id), 1,
                    ):
                        best_rank[rid] = min(best_rank.get(rid, 99), rank)
                # The production labelling path with a cross-encoder configured.
                rr_rank: dict[str, int] = {}
                if rerank:
                    mgr._retrieval.reranker = make_reranker("minilm")
                for q in (await mgr.sqlite.get_retrieval_queries(namespace=conv.sample_id) if rerank else []):
                    for rank, rid in enumerate(await mgr.replay_raw(q["query"], conv.sample_id, 3), 1):
                        rr_rank[rid] = min(rr_rank.get(rid, 99), rank)
                used = await mgr.sqlite.get_used_raw_log_ids(conv.sample_id)
                evidence = {e for qa in conv.questions if qa.category != 5 for e in qa.evidence}
                rows = []
                async with mgr.sqlite.db.execute(
                    "SELECT raw_log_id, decided_at, decision, policy_prob, features "
                    "FROM save_decisions WHERE namespace = ? ORDER BY decided_at", (conv.sample_id,)
                ) as cur:
                    async for r in cur:
                        rank = best_rank.get(r["raw_log_id"], 99)
                        rr = rr_rank.get(r["raw_log_id"], 99)
                        att = int(r["raw_log_id"] in used)
                        rows.append({
                            "features": json.loads(r["features"]),
                            "labels": {
                                "replay@1": int(rank <= 1), "replay@3": int(rank <= 3),
                                "replay@5": int(rank <= 5), "attribution": att,
                                "rerank@1": int(rr <= 1), "rerank@2": int(rr <= 2),
                                "rerank@3": int(rr <= 3),
                                "attribution+rerank@1": int(att or rr <= 1),
                            },
                            "oracle_label": int(raw_to_dia.get(r["raw_log_id"]) in evidence),
                            "decision": r["decision"], "policy_prob": r["policy_prob"],
                        })
                out[conv.sample_id] = {"rows": rows}
            finally:
                await mgr.close()
    return out


def _metrics_at(pred, probs, oracle, beta) -> dict:
    from agent_memory.policy.training import binary_metrics, roc_auc

    m = binary_metrics(pred, oracle, beta)
    auc = roc_auc(probs, oracle) if probs is not None else None
    m["auc"] = round(auc, 4) if auc is not None else None
    return m


def gate_analysis(data: dict, sources: list[str], cfg: dict) -> dict:
    """Would the production promotion gate promote a model trained on each
    label source — and would that have been right?

    The gate runs exactly as in deployment: decisions in time order (here,
    conversation by conversation), train on the oldest 75%, compare the
    learned and deployed policies on the newest 25% *using the same label
    source*. The verdict is then checked against the annotated evidence.
    """
    from agent_memory.policy.training import binary_metrics, train_from_rows

    beta = cfg["policy_fbeta"]
    ordered = [r for cid in sorted(data) for r in data[cid]["rows"]]
    for i, r in enumerate(ordered):
        r["_order"] = f"{i:08d}"
    n_val = max(int(len(ordered) * cfg["policy_validation_fraction"]), 1)
    val = ordered[-n_val:]
    out = {}
    for src in sources:
        rows = [{"features": r["features"], "outcome_useful": r["labels"][src],
                 "decision": r["decision"], "policy_prob": r["policy_prob"],
                 "decided_at": r["_order"]} for r in ordered]
        model, report = train_from_rows(rows, cfg, force=True, min_examples=0)
        if model is None:
            out[src] = {"trained": False, "reason": report.reason}
            continue
        feats = [r["features"] for r in val]
        probs = model.predict_proba_batch(feats)
        from agent_memory.policy.training import forced_save

        forced = np.array([forced_save(f, cfg) for f in feats])
        learned = np.where(forced, 1, (probs >= model.threshold).astype(int))
        deployed = [int(r["decision"] in ("save", "fast_path")) for r in val]
        oracle = [r["oracle_label"] for r in val]
        lo = binary_metrics(learned, oracle, beta)["fbeta"]
        do = binary_metrics(deployed, oracle, beta)["fbeta"]
        out[src] = {
            "gate_promotes": report.promoted,
            "gate_fbeta_learned_vs_deployed": [report.metrics["learned"]["fbeta"],
                                               report.metrics["deployed"]["fbeta"]],
            "oracle_fbeta_learned_vs_deployed": [lo, do],
            "promotion_correct": report.promoted == (lo > do),
        }
    return out


async def run_policy(
    convs, embedder_name: str, folds: int = 5, seed: int = 0, labels: list[str] | None = None,
    dump: str | None = None, load: str | None = None,
) -> dict:
    """K-fold over conversations. For each label source, train on the training
    conversations' labels and score on held-out conversations against the
    annotated evidence. Also compares every policy at the heuristic's own save
    rate, so differences are not just "saves more"."""
    from agent_memory.config import make_config
    from agent_memory.policy.controller import POLICY_HARD_CONSTRAINTS
    from agent_memory.policy.decision_model import FEATURE_NAMES, LogisticDecisionModel
    from agent_memory.policy.training import binary_metrics, choose_threshold, forced_save

    cfg = make_config(BASE_OVERRIDES)
    beta = cfg["policy_fbeta"]
    bounds = (POLICY_HARD_CONSTRAINTS["min_save_rate"], POLICY_HARD_CONSTRAINTS["max_save_rate"])
    sources = labels or LABEL_SOURCES
    if load:
        data = json.loads(Path(load).read_text())
    else:
        data = await collect_policy_data(convs, embedder_name, rerank=any("rerank" in x for x in sources))
    if dump:
        Path(dump).write_text(json.dumps(data))
    ids = sorted(data)
    random.Random(seed).shuffle(ids)
    fold_of = {cid: i % folds for i, cid in enumerate(ids)}
    all_rows = [r for cid in ids for r in data[cid]["rows"]]
    oracle_all = np.array([r["oracle_label"] for r in all_rows])

    label_quality = {
        src: binary_metrics([r["labels"][src] for r in all_rows], oracle_all, 1.0)
        for src in sources
    }

    def label_of(r, src):
        return r["oracle_label"] if src == "oracle" else r["labels"][src]

    policies: dict = {}
    for src in [*sources, "oracle"]:
        pred_all, prob_all, matched_all, oracle = [], [], [], []
        for f in range(folds):
            train = [r for cid in ids if fold_of[cid] != f for r in data[cid]["rows"]]
            test = [r for cid in ids if fold_of[cid] == f for r in data[cid]["rows"]]
            y_tr = [label_of(r, src) for r in train]
            if not test or len(set(y_tr)) < 2:
                continue
            model = LogisticDecisionModel.fit([r["features"] for r in train], y_tr, FEATURE_NAMES)
            tr_probs = model.predict_proba_batch([r["features"] for r in train])
            tr_forced = np.array([forced_save(r["features"], cfg) for r in train])
            model.threshold = choose_threshold(tr_probs, y_tr, tr_forced, beta, bounds)
            te_feats = [r["features"] for r in test]
            probs = model.predict_proba_batch(te_feats)
            forced = np.array([forced_save(x, cfg) for x in te_feats])
            pred_all += list(np.where(forced, 1, (probs >= model.threshold).astype(int)))
            # Same number of saves as the deployed heuristic made on this fold.
            n_save = sum(r["decision"] in ("save", "fast_path") for r in test)
            order = np.argsort(-(probs + forced * 10.0))
            matched = np.zeros(len(test), dtype=int)
            matched[order[:n_save]] = 1
            matched_all += list(matched)
            prob_all += list(probs)
            oracle += [r["oracle_label"] for r in test]
        policies[f"learned[{src}]"] = {
            **_metrics_at(pred_all, prob_all, oracle, beta),
            "at_heuristic_rate": binary_metrics(matched_all, oracle, beta),
        }

    deployed_pred = [int(r["decision"] in ("save", "fast_path")) for r in all_rows]
    deployed_prob = [r["policy_prob"] for r in all_rows]
    policies["deployed_lexical_heuristic"] = _metrics_at(deployed_pred, deployed_prob, oracle_all, beta)
    rate = float(np.mean(deployed_pred))
    rand = (np.random.default_rng(seed).random(len(all_rows)) < rate).astype(int)
    policies["random_at_heuristic_rate"] = _metrics_at(rand, None, oracle_all, beta)

    best = max(sources, key=lambda s: policies[f"learned[{s}]"]["auc"] or 0)
    final = LogisticDecisionModel.fit(
        [r["features"] for r in all_rows], [r["labels"][best] for r in all_rows], FEATURE_NAMES,
    )
    return {
        "turns": len(all_rows),
        "evidence_rate": round(float(oracle_all.mean()), 4),
        "label_quality_vs_evidence": label_quality,
        "folds": folds,
        "policies": policies,
        "best_label_source": best,
        "top_features[best]": final.explain(10),
        "promotion_gate": gate_analysis(data, sources, cfg),
    }


def _print_table(results: dict, ks: list[int]) -> None:
    head = f"{'system':<14}" + "".join(f"{'R@' + str(k):>9}" for k in ks) + f"{'curated':>9}{'secs':>8}"
    print(head)
    for name, r in results.items():
        print(f"{name:<14}" + "".join(f"{r[f'recall@{k}']:>9.3f}" for k in ks)
              + f"{r['curated_fraction']:>9.2f}{r['seconds']:>8.1f}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="path to locomo10.json")
    ap.add_argument("--experiment", choices=["retrieval", "policy", "both"], default="retrieval")
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--k", default="1,5,10,20")
    ap.add_argument("--embedder", default="hashing", help="hashing | minilm | <sentence-transformers model>")
    ap.add_argument("--limit", type=int, default=None, help="only the first N conversations")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--dump", default=None, help="policy: save collected decision rows to this JSON")
    ap.add_argument("--load", default=None, help="policy: reuse rows saved with --dump (no ingestion)")
    ap.add_argument("--labels", default=None,
                    help=f"label sources for --experiment policy (default: {','.join(LABEL_SOURCES)})")
    ap.add_argument("--out", default=None, help="write JSON results here")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    convs = load_locomo(args.data)[: args.limit]
    ks = [int(k) for k in args.k.split(",")]
    report: dict = {"conversations": len(convs), "embedder": args.embedder}
    if args.experiment in ("retrieval", "both"):
        systems = [s.strip() for s in args.systems.split(",") if s.strip()]
        report["retrieval"] = asyncio.run(run_retrieval(convs, systems, ks, args.embedder))
        _print_table(report["retrieval"], ks)
    if args.experiment in ("policy", "both"):
        labels = args.labels.split(",") if args.labels else None
        report["policy"] = asyncio.run(run_policy(
            convs, args.embedder, folds=args.folds, labels=labels, dump=args.dump, load=args.load,
        ))
        print(json.dumps(report["policy"], indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
