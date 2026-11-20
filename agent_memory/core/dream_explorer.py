"""Exploratory association discovery during sleep (A3).

During the compaction sleep cycle, look for non-obvious connections between
memories that were encoded separately (in different sessions) and are not yet
linked, and ask the LLM to classify the relationship. Mimics REM-like
stochastic association discovery.

Strategies (all strictly within one namespace — edges never cross tenants):

  1. ``cluster_bridge`` — sample anchor memories and look up their nearest
     neighbours in vector space; neighbours from other sessions that are not
     yet connected in the graph are "latent" connections.
  2. ``random_walk`` — sample random pairs from different sessions and keep
     those above the similarity threshold. Cheap, but rarely productive with
     real embeddings (random pairs are seldom similar), so it only runs with
     the walk budget left over after bridging.

Committing an edge requires an LLM classification other than "unrelated",
so this phase is skipped when no LLM is available.
"""

from __future__ import annotations

import logging
import random
import uuid
from collections import defaultdict
from datetime import datetime, timezone

from agent_memory.config import MEMORY_CONFIG
from agent_memory.llm.client import llm_complete
from agent_memory.models import DiscoveredEdge
from agent_memory.storage.graph_store import GraphStore
from agent_memory.storage.sqlite_store import SQLiteStore
from agent_memory.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

_VALID_RELATIONS = {"caused", "supports", "contradicts", "precedes", "part_of", "analogous", "unrelated"}


async def classify_relationship(
    mem_a_content: str, mem_b_content: str,
) -> str:
    """Ask the LLM to classify the relationship between two memories."""
    prompt = (
        "Memory A:\n<memory_a>\n"
        f"{mem_a_content}\n"
        "</memory_a>\n\n"
        "Memory B:\n<memory_b>\n"
        f"{mem_b_content}\n"
        "</memory_b>\n\n"
        "What is the relationship between Memory A and Memory B?"
    )
    try:
        response = await llm_complete(prompt, system=MEMORY_CONFIG["prompts"]["classify_relationship"], temperature=0.1)
        result = response.strip().lower()
        return result if result in _VALID_RELATIONS else "unrelated"
    except Exception:
        logger.debug("Relationship classification failed")
        return "unrelated"


async def exploratory_walk(
    sqlite: SQLiteStore,
    graph: GraphStore,
    vector: VectorStore,
    text_embedder=None,
    n_walks: int | None = None,
    similarity_threshold: float | None = None,
    max_new_edges: int | None = None,
    rng: random.Random | None = None,
) -> list[DiscoveredEdge]:
    """Discover candidate edges between unlinked, similar memories.

    Returns discovered edges; the caller commits them to the graph.
    """
    cfg = MEMORY_CONFIG
    n_walks = n_walks or cfg["dream_walk_count"]
    similarity_threshold = similarity_threshold or cfg["dream_similarity_threshold"]
    max_new_edges = max_new_edges or cfg["dream_max_new_edges"]
    rng = rng or random.Random()

    discovered: list[DiscoveredEdge] = []
    if not text_embedder:
        return discovered

    by_ns: dict[str, list[dict]] = defaultdict(list)
    for row in await sqlite.get_memories_with_vectors(tiers=["hot", "warm"]):
        by_ns[row.get("namespace") or "default"].append(row)
    pools = [rows for rows in by_ns.values() if len(rows) >= 2]
    if not pools:
        return discovered

    seen_pairs: set[frozenset] = set()

    async def consider(a: dict, b_id: str, sim: float, method: str) -> None:
        pair = frozenset((a["id"], b_id))
        if pair in seen_pairs or a["id"] == b_id:
            return
        seen_pairs.add(pair)
        if graph.path_exists(a["id"], b_id, max_hops=1):
            return
        mem_a = await sqlite.get_memory(a["id"])
        mem_b = await sqlite.get_memory(b_id, namespace=mem_a.namespace if mem_a else None)
        if not mem_a or not mem_b or mem_a.session_id == mem_b.session_id:
            return
        rel_type = await classify_relationship(mem_a.content, mem_b.content)
        if rel_type != "unrelated":
            discovered.append(DiscoveredEdge(
                source_id=a["id"], target_id=b_id, similarity=sim,
                relationship_type=rel_type, discovery_method=method,
            ))

    # Strategy 1: nearest-neighbour bridges (most of the budget).
    bridge_budget = max(1, (n_walks * 3) // 4)
    for _ in range(bridge_budget):
        if len(discovered) >= max_new_edges:
            break
        pool = rng.choice(pools)
        anchor = rng.choice(pool)
        try:
            neighbours = vector.nearest_memories(
                anchor["vector_id"], limit=5, namespace=anchor.get("namespace") or "default",
            )
        except Exception:
            continue
        for nb in neighbours:
            if nb["score"] >= similarity_threshold:
                await consider(anchor, nb["memory_id"], nb["score"], "cluster_bridge")
                if len(discovered) >= max_new_edges:
                    break

    # Strategy 2: random pairs within a namespace.
    for _ in range(n_walks - bridge_budget):
        if len(discovered) >= max_new_edges:
            break
        pool = rng.choice(pools)
        a, b = rng.sample(pool, 2)
        if a["session_id"] == b["session_id"]:
            continue
        sim = vector.similarity(a["vector_id"], b["vector_id"])
        if sim is None or sim < similarity_threshold:
            continue
        await consider(a, b["id"], sim, "random_walk")

    return discovered[:max_new_edges]


async def commit_discoveries(
    discoveries: list[DiscoveredEdge],
    graph: GraphStore,
    sqlite: SQLiteStore,
    run_id: str | None = None,
) -> int:
    """Commit discovered edges to the graph and log the exploration run.

    Returns the number of edges committed.
    """
    now = datetime.now(timezone.utc).isoformat()
    run_id = run_id or str(uuid.uuid4())
    strategies = sorted({d.discovery_method for d in discoveries})
    committed = 0

    for edge in discoveries:
        edge_id = str(uuid.uuid4())
        try:
            graph.add_relates_to(
                from_id=edge.source_id,
                to_id=edge.target_id,
                weight=edge.similarity,
                relationship_type=edge.relationship_type,
                created_at=now,
            )
            committed += 1
            ok = True
        except Exception:
            logger.debug("Failed to commit edge %s -> %s", edge.source_id, edge.target_id)
            ok = False
        await sqlite.log_dream_edge(
            edge_id=edge_id,
            run_id=run_id,
            source_id=edge.source_id,
            target_id=edge.target_id,
            similarity=edge.similarity,
            relationship_type=edge.relationship_type,
            discovery_method=edge.discovery_method,
            committed=ok,
        )

    await sqlite.log_dream_run(
        run_id=run_id,
        ran_at=now,
        n_walks=len(discoveries),
        edges_discovered=len(discoveries),
        edges_committed=committed,
        strategies=strategies,
        notes=f"Committed {committed}/{len(discoveries)} edges",
    )

    logger.info("Dream exploration: committed %d/%d edges", committed, len(discoveries))
    return committed
