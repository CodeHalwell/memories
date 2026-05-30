"""Three-layer retrieval stack with mood-congruent weighting.

Layers:
  1. Grep — raw log search via ripgrep subprocess
  2. Keyword — SQLite keyword search weighted by decay score
  3. Semantic — Qdrant vector similarity search

Results from all layers are merged, deduplicated, and re-ranked. Graph
traversal expands the result set along RELATES_TO edges. The visual
channel provides an independent retrieval path via CLIP embeddings.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import uuid
from datetime import datetime, timezone

from agent_memory.config import LOG_DIR, MEMORY_CONFIG
from agent_memory.core.decay import compute_decay
from agent_memory.embeddings.text_embedder import TextEmbedder
from agent_memory.embeddings.visual_embedder import VisualEmbedder
from agent_memory.emotion.scorer import score_emotion
from agent_memory.models import Memory
from agent_memory.storage.graph_store import GraphStore
from agent_memory.storage.sqlite_store import SQLiteStore
from agent_memory.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Orchestrates multi-layer retrieval across all storage backends."""

    def __init__(
        self,
        sqlite: SQLiteStore,
        graph: GraphStore,
        vector: VectorStore,
        text_embedder: TextEmbedder,
        visual_embedder: VisualEmbedder | None = None,
        embeddings_available: bool = True,
    ) -> None:
        self.sqlite = sqlite
        self.graph = graph
        self.vector = vector
        self.text_embedder = text_embedder
        self.visual_embedder = visual_embedder
        # When False (lite profile), the semantic vector layer is skipped and
        # retrieval relies on the grep + keyword + graph layers only.
        self.embeddings_available = embeddings_available
        self.config = MEMORY_CONFIG

    async def retrieve(
        self,
        query: str,
        session_id: str | None = None,
        top_k: int | None = None,
        enable_mood_congruent: bool = True,
        enable_visual: bool = True,
    ) -> list[Memory]:
        """Run all retrieval layers and return ranked, deduplicated memories."""
        top_k = top_k or self.config["top_k_per_layer"]

        # Score current context for mood-congruent weighting
        context_emotion = None
        if enable_mood_congruent:
            try:
                context_emotion = await score_emotion(query)
            except Exception:
                logger.debug("Mood scoring failed, proceeding without")

        # Run layers concurrently. Each task is paired with its layer name so
        # the merge step stays correct regardless of which layers are active.
        # The semantic layer is included only when embeddings are available
        # (the lite profile skips it).
        layer_tasks: list[tuple[str, asyncio.Task]] = [
            ("grep", asyncio.create_task(self._grep_layer(query, top_k))),
            ("keyword", asyncio.create_task(self._keyword_layer(query, top_k))),
        ]
        if self.embeddings_available:
            layer_tasks.append(
                ("semantic", asyncio.create_task(self._semantic_layer(query, top_k)))
            )
        if enable_visual and self.visual_embedder:
            layer_tasks.append(
                ("visual", asyncio.create_task(self._visual_layer(query, top_k)))
            )

        results = await asyncio.gather(
            *(t for _, t in layer_tasks), return_exceptions=True
        )

        # Merge all candidates
        candidates: dict[str, _Candidate] = {}

        for (layer_name, _task), result in zip(layer_tasks, results):
            if isinstance(result, Exception):
                logger.warning("Retrieval layer '%s' failed: %s", layer_name, result)
                continue
            for mem_id, score in result:
                if mem_id in candidates:
                    candidates[mem_id].score += score
                    candidates[mem_id].layers.add(layer_name)
                else:
                    candidates[mem_id] = _Candidate(
                        memory_id=mem_id, score=score, layers={layer_name},
                    )

        # Graph traversal expansion
        graph_expanded = set()
        for mem_id in list(candidates.keys()):
            related = self.graph.get_related_memories(
                mem_id, max_depth=self.config["graph_traversal_depth"],
            )
            for rel in related:
                rid = rel["id"]
                if rid not in candidates and rid not in graph_expanded:
                    graph_expanded.add(rid)
                    # Score inversely proportional to depth
                    depth_score = 1.0 / (rel["depth"] + 1) * (rel.get("salience", 0.5))
                    candidates[rid] = _Candidate(
                        memory_id=rid, score=depth_score, layers={"graph_traversal"},
                    )

        # Load full memories from SQLite
        memories: list[tuple[Memory, float]] = []
        for cand in candidates.values():
            mem = await self.sqlite.get_memory(cand.memory_id)
            if mem is None:
                continue

            score = cand.score

            # Mood-congruent boosting
            if context_emotion and enable_mood_congruent:
                mood_weight = self.config["mood_congruent_weight"]
                valence_sim = 1.0 - abs(context_emotion["valence"] - mem.valence) / 2.0
                arousal_sim = 1.0 - abs(context_emotion["arousal"] - mem.arousal)
                mood_bonus = (valence_sim + arousal_sim) / 2.0 * mood_weight
                score += mood_bonus

            # Factor in decay
            score *= mem.decay_score

            memories.append((mem, score))

        # Sort by score descending
        memories.sort(key=lambda x: x[1], reverse=True)

        # Log access and update decay for returned memories
        now = datetime.now(timezone.utc).isoformat()
        result_memories = []
        for mem, _score in memories[:top_k * 2]:  # return more than per-layer top_k
            # Determine access type
            cand = candidates.get(mem.id)
            access_type = "primary"
            if cand and "graph_traversal" in cand.layers:
                access_type = "graph_traversal"
            elif cand and "grep" in cand.layers and len(cand.layers) == 1:
                access_type = "grep_entrypoint"
            elif cand and "semantic" in cand.layers:
                access_type = "vector"

            mem.access_count += 1
            mem.last_accessed = now
            mem.decay_score = compute_decay(
                datetime.fromisoformat(now), mem.access_count,
                arousal=mem.arousal, surprise=mem.surprise,
                is_semantic=mem.is_semantic,
            )

            await self.sqlite.log_access(
                access_id=str(uuid.uuid4()),
                memory_id=mem.id,
                accessed_at=now,
                access_type=access_type,
                session_id=session_id,
                query=query,
            )
            await self.sqlite.update_memory_access(
                mem.id, mem.decay_score, mem.access_count, now,
            )

            result_memories.append(mem)

        return result_memories

    # ── Layer implementations ──

    async def _grep_layer(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Search raw logs with ripgrep and map hits to memory IDs."""
        log_dir = str(LOG_DIR)
        terms = query.split()[:5]  # Use first 5 words as search terms
        pattern = "|".join(terms)

        try:
            proc = await asyncio.create_subprocess_exec(
                "rg", "--json", "-i", "-e", pattern, log_dir,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
        except FileNotFoundError:
            logger.debug("ripgrep not found, skipping grep layer")
            return []

        if not stdout:
            return []

        import json
        hits: dict[str, int] = {}  # raw_log_id -> hit count
        for line in stdout.decode().split("\n"):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if obj.get("type") == "match":
                    data = obj.get("data", {})
                    text = data.get("lines", {}).get("text", "")
                    # Extract ID from the JSONL line
                    try:
                        entry = json.loads(text)
                        entry_id = entry.get("id", "")
                        if entry_id:
                            hits[entry_id] = hits.get(entry_id, 0) + 1
                    except json.JSONDecodeError:
                        pass
            except json.JSONDecodeError:
                pass

        # Map raw_log_ids to memory_ids via SQLite
        results: list[tuple[str, float]] = []
        for raw_id, count in sorted(hits.items(), key=lambda x: x[1], reverse=True)[:limit]:
            ref = await self.sqlite.get_raw_log_ref(raw_id)
            if ref:
                # Find memory by raw_log_id
                async with self.sqlite.db.execute(
                    "SELECT id FROM memories WHERE raw_log_id = ?", (raw_id,)
                ) as cur:
                    row = await cur.fetchone()
                    if row:
                        results.append((row["id"], float(count)))

        return results

    async def _keyword_layer(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Search by keywords extracted from the query.

        Combines two signals: extracted-keyword matches (high precision) and a
        content substring fallback (high recall — also finds memories with no
        extracted keywords, e.g. fast-path saves). The content fallback is what
        keeps this layer effective on the lite/edge profile, where the semantic
        vector layer is unavailable.
        """
        keywords = [w.lower().strip() for w in query.split() if len(w) > 2]
        if not keywords:
            return []

        scores: dict[str, float] = {}
        kw_memories = await self.sqlite.search_by_keywords(keywords, limit=limit)
        for m in kw_memories:
            scores[m.id] = max(scores.get(m.id, 0.0), m.decay_score)

        content_memories = await self.sqlite.search_by_content(keywords, limit=limit)
        for m in content_memories:
            # Slightly discount pure content matches relative to keyword hits.
            scores[m.id] = max(scores.get(m.id, 0.0), m.decay_score * 0.9)

        return list(scores.items())

    async def _semantic_layer(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Search by vector similarity in text embedding space."""
        query_vector = self.text_embedder.embed(query)
        results = self.vector.search_text(query_vector, limit=limit)
        return [(r["memory_id"], r["score"]) for r in results]

    async def _visual_layer(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Search by visual/spatial similarity using CLIP embeddings."""
        if not self.visual_embedder:
            return []
        query_vector = self.visual_embedder.embed(query)
        results = self.vector.search_visual(query_vector, limit=limit)
        return [(r["memory_id"], r["score"]) for r in results]


class _Candidate:
    """Internal candidate tracking during retrieval merge."""

    __slots__ = ("memory_id", "score", "layers")

    def __init__(self, memory_id: str, score: float, layers: set[str]) -> None:
        self.memory_id = memory_id
        self.score = score
        self.layers = layers
