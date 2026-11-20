"""Multi-layer retrieval with reciprocal-rank fusion.

First-stage layers (each returns a ranked list):

  - ``keyword``      BM25 (SQLite FTS5) over curated memories: content,
                     summary, keywords and entities
  - ``raw``          BM25 over *every* logged turn, curated or not
  - ``semantic``     vector similarity over curated memories
  - ``raw_semantic`` vector similarity over every logged turn
  - ``entity``       memories naming an entity that the query names
  - ``grep``         ripgrep over the JSONL logs (optional; superseded by ``raw``)
  - ``visual``       CLIP similarity over scene descriptions (optional)
  - ``graph``        neighbours (RELATES_TO and shared entities) of the top
                     fused memories, added as a further ranked list

Raw-turn hits on a turn that was curated count towards that memory, so the same
evidence found through several layers reinforces one candidate. Lists are
combined with reciprocal-rank fusion (scores on different scales are never
summed), the top candidates' graph neighbours are added as a further ranked
list, and an optional reranker rescores the head of the list.

Final score = normalised fused relevance + small recency, importance and mood
terms (additive, in the style of Generative Agents). Decay can lower a
memory's rank but never multiplies its relevance away.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from agent_memory.config import LOG_DIR, MEMORY_CONFIG
from agent_memory.core.decay import compute_decay, decay_for_memory, parse_time
from agent_memory.models import Memory
from agent_memory.storage.graph_store import GraphStore
from agent_memory.storage.sqlite_store import SQLiteStore
from agent_memory.storage.vector_store import VectorStore
from agent_memory.text import content_terms

logger = logging.getLogger(__name__)

_LEXICAL_LAYERS = {"keyword", "raw", "entity", "grep"}
_MEMORY_LAYERS = {"keyword", "semantic", "entity", "graph", "visual"}


@dataclass
class RetrievalTrace:
    """Retrieval results plus the signals logged for policy learning."""
    memories: list[Memory]
    layers_used: list[str] = field(default_factory=list)
    lexical_hit: bool = False
    top_semantic: float | None = None
    top_score: float | None = None
    miss: bool = True


class RetrievalEngine:
    """Orchestrates multi-layer retrieval across all storage backends."""

    def __init__(
        self,
        sqlite: SQLiteStore,
        graph: GraphStore,
        vector: VectorStore,
        text_embedder,
        visual_embedder=None,
        embeddings_available: bool = True,
        config: dict | None = None,
        log_dir_for: Callable[[str], Path] | None = None,
        reranker=None,
        graph_available: bool = True,
        llm_enabled: bool = True,
    ) -> None:
        self.sqlite = sqlite
        self.graph = graph
        self.vector = vector
        self.text_embedder = text_embedder
        self.visual_embedder = visual_embedder
        # When False (lite profile), the vector layers are skipped.
        self.embeddings_available = embeddings_available
        self.config = config or MEMORY_CONFIG
        self.log_dir_for = log_dir_for or (lambda _ns: LOG_DIR)
        self.reranker = reranker
        self.graph_available = graph_available
        self.llm_enabled = llm_enabled

    async def retrieve(
        self,
        query: str,
        session_id: str | None = None,
        top_k: int | None = None,
        namespace: str = "default",
        enable_mood_congruent: bool = True,
        enable_visual: bool = True,
        since: str | None = None,
        until: str | None = None,
    ) -> list[Memory]:
        trace = await self.retrieve_detailed(
            query, session_id=session_id, top_k=top_k, namespace=namespace,
            enable_mood_congruent=enable_mood_congruent, enable_visual=enable_visual,
            since=since, until=until,
        )
        return trace.memories

    async def retrieve_detailed(
        self,
        query: str,
        session_id: str | None = None,
        top_k: int | None = None,
        namespace: str = "default",
        enable_mood_congruent: bool = True,
        enable_visual: bool = True,
        since: str | None = None,
        until: str | None = None,
        record_access: bool = True,
        layers: list[str] | None = None,
    ) -> RetrievalTrace:
        """Run all retrieval layers and return ranked, deduplicated results.

        Results are isolated to ``namespace``: every layer filters by it, and
        loading enforces it again as a safety net. ``since``/``until`` (ISO
        timestamps) restrict results to events in that window.
        """
        cfg = self.config
        top_k = top_k or cfg["retrieval_top_k"]
        depth = max(cfg["layer_depth"], top_k * 2)
        # Time windows are applied inside the lexical layers' SQL (before their
        # LIMIT); vector layers cannot filter on time, so they over-fetch.
        t_since, t_until = parse_time(since), parse_time(until)
        window = (
            t_since.isoformat() if t_since else None,
            t_until.isoformat() if t_until else None,
        )
        vec_depth = depth * 5 if (t_since or t_until) else depth
        active = list(layers if layers is not None else cfg["retrieval_layers"])

        # Mood-congruent weighting needs an LLM call per query; skip it when no
        # LLM is available or its weight is zero.
        mood_task = None
        if enable_mood_congruent and self.llm_enabled and cfg.get("mood_congruent_weight", 0) > 0:
            from agent_memory.emotion.scorer import score_emotion
            mood_task = asyncio.create_task(score_emotion(query))

        query_vector = None
        if self.embeddings_available and ({"semantic", "raw_semantic"} & set(active)):
            try:
                query_vector = self.text_embedder.embed(query)
            except Exception:
                logger.warning("Query embedding failed; skipping vector layers")

        tasks: list[tuple[str, asyncio.Task]] = []

        def add(name: str, coro) -> None:
            tasks.append((name, asyncio.create_task(coro)))

        if "keyword" in active:
            add("keyword", self._keyword_layer(query, depth, namespace, window))
        if "raw" in active:
            add("raw", self._raw_layer(query, depth, namespace, window))
        if "entity" in active:
            add("entity", self._entity_layer(query, depth, namespace, window))
        if "grep" in active:
            add("grep", self._grep_layer(query, depth, namespace))
        semantic_scores: list[float] = []
        if query_vector is not None and "semantic" in active:
            add("semantic", self._semantic_layer(query_vector, vec_depth, namespace, semantic_scores))
        if (query_vector is not None and "raw_semantic" in active
                and cfg.get("index_raw_vectors", True)):
            add("raw_semantic", self._raw_semantic_layer(query_vector, vec_depth, namespace, semantic_scores))
        if enable_visual and self.visual_embedder:
            add("visual", self._visual_layer(query, depth, namespace))

        results = await asyncio.gather(*(t for _, t in tasks), return_exceptions=True)
        layer_lists: dict[str, list[tuple[tuple[str, str], float]]] = {}
        for (name, _t), res in zip(tasks, results):
            if isinstance(res, Exception):
                logger.warning("Retrieval layer '%s' failed: %s", name, res)
                continue
            layer_lists[name] = res

        # Raw hits on curated turns count for the memory itself.
        raw_ids = {key[1] for items in layer_lists.values() for key, _ in items if key[0] == "r"}
        # With only raw layers active (outcome replay, raw-history baselines),
        # keep raw turns as raw candidates so ranking does not depend on which
        # turns were curated or later consolidated.
        raw_to_mem = (
            await self.sqlite.memory_ids_for_raw(sorted(raw_ids), namespace=namespace)
            if set(active) & _MEMORY_LAYERS else {}
        )

        def canonical(key: tuple[str, str]) -> tuple[str, str]:
            if key[0] == "r" and key[1] in raw_to_mem:
                return ("m", raw_to_mem[key[1]])
            return key

        fused: dict[tuple[str, str], float] = defaultdict(float)
        matched: dict[tuple[str, str], set[str]] = defaultdict(set)
        rrf_k = cfg.get("rrf_k", 60)
        weights = cfg.get("layer_weights", {})
        for name, items in layer_lists.items():
            seen: set[tuple[str, str]] = set()
            rank = 0
            for key, _score in items:
                ck = canonical(key)
                if ck in seen:
                    continue
                seen.add(ck)
                fused[ck] += weights.get(name, 1.0) / (rrf_k + rank + 1)
                matched[ck].add(name)
                rank += 1

        lexical_hit = any(layer_lists.get(n) for n in _LEXICAL_LAYERS)
        top_semantic = max(semantic_scores) if semantic_scores else None

        # Graph expansion from the strongest curated candidates.
        if self.graph_available and fused and "graph" in active:
            await self._expand_graph(fused, matched, rrf_k, weights.get("graph", 0.5))

        candidates = await self._load_candidates(fused, matched, namespace, since, until)

        context_emotion = None
        if mood_task is not None:
            try:
                context_emotion = await mood_task
            except Exception:
                logger.debug("Mood scoring failed, proceeding without")

        ranked = self._score(candidates, fused, context_emotion)
        if self.reranker is not None and ranked:
            ranked = self._rerank(query, ranked)
        ranked.sort(key=lambda x: x[1], reverse=True)
        top = ranked[:top_k]

        now = datetime.now(timezone.utc).isoformat()
        accesses = []
        out: list[Memory] = []
        for mem, score in top:
            mem.score = round(score, 6)
            if mem.kind == "memory":
                layers = mem.matched_layers
                if layers == ["graph"]:
                    access_type = "graph_traversal"
                elif {"semantic", "raw_semantic"} & set(layers):
                    access_type = "vector"
                else:
                    access_type = "primary"
                accesses.append((mem.id, access_type, session_id, query))
                mem.access_count += 1
                mem.last_accessed = now
                mem.decay_score = compute_decay(
                    datetime.now(timezone.utc), mem.access_count,
                    arousal=mem.arousal, surprise=mem.surprise,
                    is_semantic=mem.is_semantic, config=cfg,
                )
            out.append(mem)
        if record_access and accesses:
            await self.sqlite.record_accesses(accesses, now)

        floor = cfg.get("semantic_min_score", 0.0)
        miss = not lexical_hit and (top_semantic is None or top_semantic < floor)
        return RetrievalTrace(
            memories=out,
            layers_used=sorted(layer_lists),
            lexical_hit=lexical_hit,
            top_semantic=top_semantic,
            top_score=out[0].score if out else None,
            miss=miss or not out,
        )

    # ── Fusion helpers ──

    async def _expand_graph(self, fused, matched, rrf_k: int, weight: float) -> None:
        cfg = self.config
        seeds = [
            k for k, _ in sorted(fused.items(), key=lambda kv: -kv[1]) if k[0] == "m"
        ][:cfg.get("graph_expansion_seeds", 5)]
        expansion: list[str] = []
        for seed in seeds:
            try:
                related = self.graph.get_related_memories(
                    seed[1], max_depth=cfg["graph_traversal_depth"],
                )
                shared = self.graph.get_entity_neighbours(seed[1], limit=5)
            except Exception:
                logger.debug("Graph expansion failed for %s", seed[1])
                continue
            for rel in related:
                expansion.append(rel["id"])
            for rel in shared:
                expansion.append(rel["id"])
        rank = 0
        seen: set[str] = set()
        for mid in expansion:
            if mid in seen:
                continue
            seen.add(mid)
            key = ("m", mid)
            fused[key] += weight / (rrf_k + rank + 1)
            matched[key].add("graph")
            rank += 1

    async def _load_candidates(
        self, fused, matched, namespace: str, since: str | None, until: str | None,
    ) -> list[Memory]:
        mem_ids = [k[1] for k in fused if k[0] == "m"]
        raw_ids = [k[1] for k in fused if k[0] == "r"]
        # Passing namespace enforces tenant isolation as a final safety net.
        mems = await self.sqlite.get_memories(mem_ids, namespace=namespace)
        raws = await self.sqlite.get_raw_turns(raw_ids)
        t_since, t_until = parse_time(since), parse_time(until)

        def in_window(ts: str | None) -> bool:
            if t_since is None and t_until is None:
                return True
            t = parse_time(ts)
            if t is None:
                return False
            return (t_since is None or t >= t_since) and (t_until is None or t <= t_until)

        out: list[Memory] = []
        for mid, mem in mems.items():
            if not in_window(mem.event_time or mem.created_at):
                continue
            mem.kind = "memory"
            mem.matched_layers = sorted(matched[("m", mid)])
            out.append(mem)
        for rid, row in raws.items():
            if row.get("namespace", "default") != namespace or not in_window(row["timestamp"]):
                continue
            out.append(Memory(
                id=f"raw:{rid}",
                created_at=row["timestamp"],
                updated_at=row["timestamp"],
                content=row.get("content") or "",
                raw_log_id=rid,
                namespace=namespace,
                session_id=row["session_id"],
                turn=row["turn"],
                salience=0.0,
                tier="raw",
                role=row.get("role") or "assistant",
                event_time=row["timestamp"],
                kind="raw",
                matched_layers=sorted(matched[("r", rid)]),
            ))
        return out

    def _score(self, candidates: list[Memory], fused, context_emotion) -> list[tuple[Memory, float]]:
        cfg = self.config
        if not candidates:
            return []
        max_fused = max(fused.values()) or 1.0
        w_rec = cfg.get("recency_weight", 0.0)
        w_imp = cfg.get("importance_weight", 0.0)
        w_mood = cfg.get("mood_congruent_weight", 0.0)
        cold = cfg.get("cold_tier_weight", 1.0)
        now = datetime.now(timezone.utc)
        scored = []
        for mem in candidates:
            key = ("r", mem.raw_log_id) if mem.kind == "raw" else ("m", mem.id)
            rel = fused.get(key, 0.0) / max_fused
            score = rel + w_rec * decay_for_memory(mem, now=now, config=cfg)
            if mem.kind == "memory":
                score += w_imp * mem.salience
                if context_emotion:
                    valence_sim = 1.0 - abs(context_emotion["valence"] - mem.valence) / 2.0
                    arousal_sim = 1.0 - abs(context_emotion["arousal"] - mem.arousal)
                    score += w_mood * (valence_sim + arousal_sim) / 2.0
                if mem.tier == "cold":
                    score *= cold
            scored.append((mem, score))
        return scored

    def _rerank(self, query: str, ranked: list[tuple[Memory, float]]) -> list[tuple[Memory, float]]:
        cfg = self.config
        ranked.sort(key=lambda x: x[1], reverse=True)
        depth = cfg.get("rerank_depth", 50)
        head, tail = ranked[:depth], ranked[depth:]
        try:
            scores = self.reranker.score(query, [m.content for m, _ in head])
        except Exception:
            logger.warning("Reranker failed; keeping fused order")
            return ranked
        lo, hi = min(scores), max(scores)
        span = (hi - lo) or 1.0
        now = datetime.now(timezone.utc)
        out = []
        for (mem, old), s in zip(head, scores):
            new = (s - lo) / span + cfg.get("recency_weight", 0.0) * decay_for_memory(mem, now=now, config=cfg)
            if mem.kind == "memory":
                new += cfg.get("importance_weight", 0.0) * mem.salience
                if mem.tier == "cold":
                    new *= cfg.get("cold_tier_weight", 1.0)
            # Keep the reranked head above the un-reranked tail.
            out.append((mem, new + 1.0))
        return out + tail

    # ── Layer implementations ──
    # Each returns [((kind, id), score), ...] best first; kind is "m" (memory) or "r" (raw).

    async def _keyword_layer(self, query: str, limit: int, namespace: str = "default", window=(None, None)):
        hits = await self.sqlite.search_memories_fts(
            query, limit=limit, namespace=namespace, since=window[0], until=window[1],
        )
        return [(("m", mid), s) for mid, s in hits]

    async def _raw_layer(self, query: str, limit: int, namespace: str = "default", window=(None, None)):
        hits = await self.sqlite.search_raw_fts(
            query, limit=limit, namespace=namespace, since=window[0], until=window[1],
        )
        return [(("r", rid), s) for rid, s in hits]

    async def _entity_layer(self, query: str, limit: int, namespace: str = "default", window=(None, None)):
        hits = await self.sqlite.search_entities(
            query, limit=limit, namespace=namespace, since=window[0], until=window[1],
        )
        return [(("m", mid), s) for mid, s in hits]

    async def _semantic_layer(self, query_vector, limit: int, namespace: str, scores_out: list):
        results = self.vector.search_text(query_vector, limit=limit, namespace=namespace)
        floor = self.config.get("semantic_min_score", 0.0)
        scores_out.extend(r["score"] for r in results)
        return [(("m", r["memory_id"]), r["score"]) for r in results if r["score"] >= floor]

    async def _raw_semantic_layer(self, query_vector, limit: int, namespace: str, scores_out: list):
        results = self.vector.search_raw(query_vector, limit=limit, namespace=namespace)
        floor = self.config.get("semantic_min_score", 0.0)
        scores_out.extend(r["score"] for r in results)
        return [(("r", r["raw_log_id"]), r["score"]) for r in results if r["score"] >= floor]

    async def _visual_layer(self, query: str, limit: int, namespace: str = "default"):
        if not self.visual_embedder:
            return []
        query_vector = self.visual_embedder.embed(query)
        results = self.vector.search_visual(query_vector, limit=limit, namespace=namespace)
        return [(("m", r["memory_id"]), r["score"]) for r in results]

    async def _grep_layer(self, query: str, limit: int, namespace: str = "default"):
        """Search the namespace's JSONL logs with ripgrep (fixed strings).

        Matches are verified against the entry's *content* (not the JSON keys)
        and scored by the fraction of distinct query terms present.
        """
        terms = list(dict.fromkeys(content_terms(query)))[:8]
        if not terms:
            return []
        log_dir = Path(self.log_dir_for(namespace))
        if not log_dir.exists():
            return []
        args = ["rg", "--json", "-i", "-F"]
        if namespace == "default":
            args += ["--max-depth", "1"]  # other namespaces live in sub-directories
        for t in terms:
            args += ["-e", t]
        args.append(str(log_dir))
        try:
            proc = await asyncio.create_subprocess_exec(
                *args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
        except FileNotFoundError:
            logger.debug("ripgrep not found, skipping grep layer")
            return []

        scores: dict[str, float] = {}
        for line in stdout.decode(errors="replace").splitlines():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") != "match":
                continue
            text = obj.get("data", {}).get("lines", {}).get("text", "")
            try:
                entry = json.loads(text)
            except json.JSONDecodeError:
                continue
            if entry.get("namespace", "default") != namespace or not entry.get("id"):
                continue
            content_set = set(content_terms(entry.get("content", "")))
            frac = sum(t in content_set for t in terms) / len(terms)
            if frac > 0:
                scores[entry["id"]] = max(scores.get(entry["id"], 0.0), frac)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:limit]
        return [(("r", rid), s) for rid, s in ranked]
