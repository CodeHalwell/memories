"""Compaction ("sleep cycle"): deduplicate, then carefully consolidate.

Order of operations in a cycle:

  1. Refresh cached decay scores for the current time.
  2. Deduplicate: near-identical memories within a namespace are collapsed
     onto the most valuable copy (no LLM needed). Research on agent memory
     consolidation finds deduplication carries most of the benefit, while
     summarising clusters destroys the specifics later questions need.
  3. Consolidate low-value candidates that share keywords, *within one
     namespace*, into a fact-preserving merged memory — only if the merge
     passes validation (A2.4): generated (question, answer) pairs from the
     sources must still be answerable from the merged text, and synthetic
     queries must still retrieve it about as well as the originals.
  4. Tier promotion.

Sources are never deleted: they move to the cold tier (still retrievable, at
reduced weight) with EVOLVED_FROM lineage edges.

A2.3: Generation gap guard — only merge memories within 1 generation of each other.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from agent_memory.config import MEMORY_CONFIG
from agent_memory.llm.client import llm_available, llm_complete_json
from agent_memory.models import CompactionResult, Memory, MergeValidation
from agent_memory.storage.graph_store import GraphStore
from agent_memory.storage.sqlite_store import SQLiteStore
from agent_memory.storage.vector_store import VectorStore
from agent_memory.text import coverage, jaccard, negations, term_set, tokens

logger = logging.getLogger(__name__)


def compaction_score(memory: Memory) -> float:
    """Score a memory for compaction candidacy.

    Low decay + low salience = good candidate for compaction.
    """
    return (1 - memory.decay_score) * 0.6 + (1 - memory.salience) * 0.4


def _keyword_overlap(mem_a: Memory, mem_b: Memory) -> float:
    """Compute keyword overlap ratio between two memories."""
    kw_a = {kw for kw, _ in mem_a.keywords}
    kw_b = {kw for kw, _ in mem_b.keywords}
    if not kw_a or not kw_b:
        return 0.0
    intersection = kw_a & kw_b
    union = kw_a | kw_b
    return len(intersection) / len(union) if union else 0.0


def _can_merge(group: list[Memory], config: dict) -> bool:
    """Check if a group of memories can be merged (no exclusion conditions)."""
    if len({m.namespace for m in group}) > 1:
        return False  # never merge across tenants
    for i, a in enumerate(group):
        for b in group[i + 1:]:
            # Opposite valence exclusion
            if (a.valence * b.valence < 0 and
                    abs(a.valence - b.valence) > config["valence_merge_exclusion_delta"]):
                return False
            # Either is fast_pathed gen-0
            if (a.fast_pathed and a.compaction_gen == 0) or \
               (b.fast_pathed and b.compaction_gen == 0):
                return False
            # A2.3: Generation gap guard — only merge within 1 generation
            if abs(a.compaction_gen - b.compaction_gen) > config.get("max_generation_gap_for_merge", 1):
                return False
    return True


def _group_by_keywords(candidates: list[Memory], threshold: float) -> list[list[Memory]]:
    """Group memories by keyword overlap using greedy clustering.

    Groups never span namespaces.
    """
    if not candidates:
        return []

    used = set()
    groups: list[list[Memory]] = []

    for i, mem_a in enumerate(candidates):
        if i in used:
            continue
        group = [mem_a]
        used.add(i)
        for j, mem_b in enumerate(candidates):
            if j in used or mem_b.namespace != mem_a.namespace:
                continue
            # Check overlap with all current group members
            overlaps = [_keyword_overlap(g, mem_b) for g in group]
            if overlaps and min(overlaps) >= threshold:
                group.append(mem_b)
                used.add(j)
        if len(group) > 1:
            groups.append(group)

    return groups


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    return float(dot / norm) if norm > 0 else 0.0


def _normalised(text: str) -> str:
    """Lowercased words in any script, single-spaced (punctuation dropped)."""
    return " ".join(re.findall(r"[^\W_]+", (text or "").lower()))


def _answer_preserved(answer: str, merged_terms: set[str], merged_norm: str) -> bool:
    """An answer survives if its normalised text appears verbatim, or nearly
    all of its content words do."""
    norm = _normalised(answer)
    if norm and norm in merged_norm:
        return True
    return coverage(term_set(answer), merged_terms) >= 0.8


async def validate_merge(
    source_memories: list[Memory],
    candidate_content: str,
    text_embedder=None,
    n_queries: int = 5,
    degradation_tolerance: float = 0.15,
    min_answer_preservation: float = 1.0,
) -> MergeValidation:
    """Validate a merge before committing it (A2.4).

    Generates (question, answer) pairs from the sources, then requires that
    (a) the answers still appear in the merged text — the fact-preservation
    check — and (b) with an embedder, the questions retrieve the merged
    memory about as well as the best source (embedding degradation within
    tolerance). Fails closed if questions cannot be generated.
    """
    source_text = "\n\n".join(m.content for m in source_memories)
    prompt = (
        f"Generate {n_queries} questions with answers for this content:\n\n"
        f"<source_text>\n{source_text}\n</source_text>"
    )
    try:
        generated = await llm_complete_json(prompt, system=MEMORY_CONFIG["prompts"]["synthetic_query"])
        if isinstance(generated, dict):
            generated = next((v for v in generated.values() if isinstance(v, list)), [])
        if not isinstance(generated, list):
            generated = []
    except Exception:
        logger.warning("Synthetic question generation failed, failing validation to be safe")
        return MergeValidation(passed=False, queries_tested=[])

    queries: list[str] = []
    answers: list[str] = []
    for item in generated:
        if isinstance(item, dict) and item.get("question"):
            queries.append(str(item["question"]))
            if item.get("answer") not in (None, ""):
                answers.append(str(item["answer"]))
        elif isinstance(item, str):
            queries.append(item)
    if not queries:
        return MergeValidation(passed=False, queries_tested=[])

    merged_terms = term_set(candidate_content)
    merged_norm = _normalised(candidate_content)
    preserved = sum(_answer_preserved(a, merged_terms, merged_norm) for a in answers)
    answers_ok = (not answers) or (preserved / len(answers) >= min_answer_preservation)

    avg_merged = avg_source = degradation = 0.0
    embed_ok = True
    if text_embedder is not None:
        candidate_emb = text_embedder.embed(candidate_content)
        source_embs = [text_embedder.embed(m.content) for m in source_memories]
        merged_scores, source_scores = [], []
        for query in queries:
            query_emb = text_embedder.embed(query)
            merged_scores.append(_cosine_similarity(query_emb, candidate_emb))
            source_scores.append(max(_cosine_similarity(query_emb, se) for se in source_embs))
        avg_merged = sum(merged_scores) / len(merged_scores)
        avg_source = sum(source_scores) / len(source_scores)
        degradation = avg_source - avg_merged
        embed_ok = degradation < degradation_tolerance

    return MergeValidation(
        passed=answers_ok and embed_ok,
        avg_source_score=avg_source,
        avg_merged_score=avg_merged,
        degradation=degradation,
        queries_tested=queries,
        answers_checked=len(answers),
        answers_preserved=preserved,
    )


def _near_duplicate(a: Memory, b: Memory, threshold: float) -> bool:
    """Near-identical wording *and* the same polarity.

    Compares all words (stopwords included) so that "I am allergic" and "I am
    not allergic" are never collapsed, and rejects pairs whose negation words
    differ outright.
    """
    if negations(a.content) != negations(b.content):
        return False
    return jaccard(set(tokens(a.content)), set(tokens(b.content))) >= threshold


class CompactionEngine:
    """Runs compaction cycles — deduplication, then validated consolidation."""

    def __init__(
        self,
        sqlite: SQLiteStore,
        graph: GraphStore,
        vector: VectorStore,
        text_embedder=None,
        visual_embedder=None,
        config: dict | None = None,
        graph_available: bool = True,
        vectors_available: bool = True,
    ) -> None:
        self.sqlite = sqlite
        self.graph = graph
        self.vector = vector
        # Only pass an embedder that is actually usable (None on the lite profile).
        self.text_embedder = text_embedder
        self.visual_embedder = visual_embedder
        self.config = config or MEMORY_CONFIG
        self.graph_available = graph_available
        self.vectors_available = vectors_available

    async def run(self, trigger: str = "scheduled") -> CompactionResult:
        """Execute a full compaction cycle (see module docstring)."""
        result = CompactionResult(trigger=trigger)
        await self.sqlite.refresh_decay_scores(config=self.config)

        deduped = 0
        for ns in await self.sqlite.list_namespaces():
            deduped += await self._dedupe_namespace(ns, result.id)
        result.memories_pruned = deduped

        candidates = await self.sqlite.get_compaction_candidates(
            threshold=self.config["compaction_candidate_threshold"],
        )
        result.memories_reviewed = len(candidates)

        # Filter by graph edge count (structurally important anchors). Write-time
        # "similar" links are excluded: near-duplicates are what we want to merge.
        filtered = []
        for mem in candidates:
            if self.graph_available and mem.graph_node_id:
                try:
                    if self.graph.get_edge_count(mem.id, exclude_types=("similar",)) > 3:
                        continue
                except Exception:
                    pass
            filtered.append(mem)

        groups = _group_by_keywords(filtered, self.config["keyword_overlap_merge_threshold"])

        merged_count = 0
        if groups and llm_available(self.config):
            for group in groups:
                if not _can_merge(group, self.config):
                    continue
                if await self._merge_group(group, result.id):
                    merged_count += 1
        result.memories_merged = merged_count

        # Tier promotion: move hot memories exceeding threshold to warm
        hot_count = await self.sqlite.count_memories(tier="hot")
        threshold = self.config["hot_tier_threshold"]
        if hot_count > threshold:
            await self._promote_tier(hot_count - threshold)

        result.notes = (
            f"Reviewed {result.memories_reviewed}, deduplicated {deduped}, "
            f"merged into {merged_count} semantic memories"
        )
        await self.sqlite.log_compaction_run(result)
        logger.info(
            "Compaction complete: reviewed=%d, deduplicated=%d, merged=%d",
            result.memories_reviewed, deduped, merged_count,
        )
        return result

    # ── Deduplication ──

    async def _dedupe_namespace(self, namespace: str, compaction_id: str) -> int:
        """Collapse near-identical active memories onto the most valuable copy."""
        mems = await self.sqlite.list_active_memories(namespace)
        if len(mems) < 2:
            return 0
        thr = self.config.get("dedupe_jaccard_threshold", 0.9)
        terms = {m.id: term_set(m.content) for m in mems}
        # Bucket by exact normalised content, then compare within shared-term
        # buckets for near-duplicates (bounded to keep this roughly linear).
        by_hash: dict[str, list[Memory]] = defaultdict(list)
        for m in mems:
            key = _normalised(m.content)
            if key:  # an empty key is not evidence of duplication
                by_hash[hashlib.sha1(key.encode()).hexdigest()].append(m)
        groups = [g for g in by_hash.values() if len(g) > 1]

        rare_index: dict[str, list[Memory]] = defaultdict(list)
        for m in mems:
            for t in sorted(terms[m.id], key=len, reverse=True)[:3]:
                rare_index[t].append(m)
        grouped = {m.id for g in groups for m in g}
        for bucket in rare_index.values():
            if len(bucket) < 2 or len(bucket) > 50:
                continue
            for i, a in enumerate(bucket):
                if a.id in grouped:
                    continue
                dupes = [a]
                for b in bucket[i + 1:]:
                    if b.id not in grouped and _near_duplicate(a, b, thr):
                        dupes.append(b)
                if len(dupes) > 1:
                    groups.append(dupes)
                    grouped.update(m.id for m in dupes)

        removed = 0
        now = datetime.now(timezone.utc).isoformat()
        for group in groups:
            keep = max(group, key=lambda m: (m.fast_pathed, m.salience, m.access_count, m.created_at))
            for dup in group:
                # Hard constraint: fast-path gen-0 memories are never compacted.
                if dup.id == keep.id or (dup.fast_pathed and dup.compaction_gen == 0):
                    continue
                await self.sqlite.update_memory_tier(dup.id, "cold")
                if self.graph_available:
                    try:
                        self.graph.update_memory_tier(dup.id, "cold")
                        self.graph.add_evolved_from(keep.id, dup.id, compaction_id=compaction_id, created_at=now)
                    except Exception:
                        logger.debug("Graph update failed during dedupe of %s", dup.id)
                await self.sqlite.log_compaction_merge(compaction_id, [dup.id], keep.id, validation_passed=True)
                removed += 1
        return removed

    # ── Consolidation ──

    async def _merge_group(self, group: list[Memory], compaction_id: str) -> Memory | None:
        """Merge a group of memories into a single semantic memory."""
        sources = "\n\n".join(
            f"Memory {i+1} (salience={m.salience}, valence={m.valence}, "
            f"when={m.event_time or m.created_at}):\n<memory_{i+1}>\n{m.content}\n</memory_{i+1}>"
            for i, m in enumerate(group)
        )
        prompt = f"""Combine these {len(group)} related memories into one memory without losing any fact:

{sources}

Respond with JSON only."""

        try:
            result = await llm_complete_json(prompt, system=self.config["prompts"]["merge"])
        except Exception:
            logger.exception("LLM merge failed for group of %d memories", len(group))
            return None

        merged_content = str(result.get("content", "") or "")
        if not merged_content.strip():
            return None

        # A2.4: Validate merge (fact preservation + retrieval degradation)
        validation = await validate_merge(
            source_memories=group,
            candidate_content=merged_content,
            text_embedder=self.text_embedder,
            n_queries=self.config.get("merge_validation_queries", 5),
            degradation_tolerance=self.config.get("merge_degradation_tolerance", 0.15),
            min_answer_preservation=self.config.get("merge_min_answer_preservation", 1.0),
        )
        if not validation.passed:
            logger.info(
                "Merge validation failed (answers %d/%d, degradation=%.3f); keeping %d sources",
                validation.answers_preserved, validation.answers_checked,
                validation.degradation, len(group),
            )
            return None

        now = datetime.now(timezone.utc).isoformat()
        new_id = str(uuid.uuid4())
        max_gen = max(m.compaction_gen for m in group)
        namespace = group[0].namespace

        keywords = [
            (str(kw["keyword"]).lower(), float(kw.get("weight", 1.0)))
            for kw in result.get("keywords", []) if isinstance(kw, dict) and kw.get("keyword")
        ][:self.config["max_keywords_per_memory"]]
        if not keywords:
            merged_kw: dict[str, float] = {}
            for m in group:
                for kw, w in m.keywords:
                    merged_kw[kw] = max(merged_kw.get(kw, 0.0), w)
            keywords = sorted(merged_kw.items(), key=lambda kv: -kv[1])[:self.config["max_keywords_per_memory"]]
        entities = list(dict.fromkeys(e for m in group for e in m.entities))[:8]
        event_times = [m.event_time for m in group if m.event_time]

        new_mem = Memory(
            id=new_id,
            created_at=now,
            updated_at=now,
            content=merged_content,
            summary=result.get("summary"),
            raw_log_id=group[0].raw_log_id,  # link to first source; all sources via lineage
            namespace=namespace,
            session_id=group[0].session_id,
            turn=group[0].turn,
            valence=float(result.get("valence", 0.0) or 0.0),
            arousal=float(result.get("arousal", 0.0) or 0.0),
            salience=float(result.get("salience", max(m.salience for m in group)) or 0.5),
            compaction_gen=max_gen + 1,
            tier="warm",
            is_semantic=True,
            keywords=keywords,
            entities=entities,
            role="summary",
            event_time=max(event_times) if event_times else None,
        )

        await self.sqlite.save_memory(new_mem)

        source_ids = [m.id for m in group]
        if self.graph_available:
            self.graph.add_memory_node(
                memory_id=new_id, summary=new_mem.summary or "", tier="warm",
                salience=new_mem.salience, valence=new_mem.valence,
                compaction_gen=new_mem.compaction_gen, created_at=now, namespace=namespace,
            )
            await self.sqlite.update_memory_graph_ref(new_id, new_id)
            for src in group:
                self.graph.add_evolved_from(new_id, src.id, compaction_id=compaction_id, created_at=now)
            self.graph.replicate_edges_to_new_node(source_ids, new_id)

        # Move source memories to cold tier (still retrievable, lower weight)
        for src in group:
            await self.sqlite.update_memory_tier(src.id, "cold")
            if self.graph_available and src.graph_node_id:
                self.graph.update_memory_tier(src.id, "cold")

        if self.text_embedder is not None and self.vectors_available:
            vector = self.text_embedder.embed(new_mem.content)
            point_id = self.vector.upsert_text_vector(
                memory_id=new_id, vector=vector, tier="warm",
                valence=new_mem.valence, arousal=new_mem.arousal,
                session_id=new_mem.session_id, created_at=now, namespace=namespace,
            )
            await self.sqlite.update_memory_vector_ref(new_id, point_id)

        await self.sqlite.log_compaction_merge(
            compaction_id, source_ids, new_id,
            validation_passed=validation.passed,
            avg_source_score=validation.avg_source_score,
            avg_merged_score=validation.avg_merged_score,
            degradation=validation.degradation,
        )
        return new_mem

    async def _promote_tier(self, count: int) -> None:
        """Move the oldest/lowest-decay hot memories to warm tier."""
        async with self.sqlite.db.execute(
            "SELECT id, graph_node_id FROM memories WHERE tier = 'hot' "
            "ORDER BY decay_score ASC LIMIT ?",
            (count,),
        ) as cur:
            rows = [dict(r) async for r in cur]

        for row in rows:
            await self.sqlite.update_memory_tier(row["id"], "warm")
            if self.graph_available and row.get("graph_node_id"):
                self.graph.update_memory_tier(row["id"], "warm")
