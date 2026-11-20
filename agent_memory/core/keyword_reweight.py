"""Graph-informed keyword reweighting (A2.5).

Keywords that appear in memories connected by RELATES_TO edges are
structurally more important than ones in isolated memories. This module
runs a lightweight reweighting pass during compaction, boosting weights
for keywords shared across well-connected memory clusters.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from agent_memory.storage.graph_store import GraphStore
from agent_memory.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


async def reweight_keywords_from_graph(
    sqlite: SQLiteStore,
    graph: GraphStore,
    max_hops: int = 2,
    max_memories_per_keyword: int = 50,
) -> int:
    """Adjust keyword weights based on graph connectivity.

    Keywords shared across well-connected memories get boosted.
    Returns the number of keyword weights updated.

    ``max_memories_per_keyword`` caps the number of memories evaluated per
    keyword to avoid O(n²) graph queries for very common keywords.
    """
    # Get all keywords with their memory associations
    rows = await sqlite.get_all_keywords_with_memories()
    if not rows:
        return 0

    # Group by (namespace, keyword): connectivity is measured within a tenant
    # only, so one tenant's graph never changes another's keyword weights.
    keyword_index: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        keyword_index[(row.get("namespace") or "default", row["keyword"])].append(row)

    # Collect all weight updates, then commit in a single batch
    pending_updates: list[tuple[float, str, str]] = []

    for (_ns, keyword), entries in keyword_index.items():
        if len(entries) < 2:
            continue

        # Cap memories per keyword to avoid O(n²) graph queries.
        # Sort to ensure deterministic and meaningful selection:
        #   - higher-weight memories first
        #   - tie-broken by memory_id for stability
        sorted_entries = sorted(entries, key=lambda e: (-e["base_weight"], e["memory_id"]))
        capped_entries = sorted_entries[:max_memories_per_keyword]
        memory_ids = [e["memory_id"] for e in capped_entries]

        # Check graph connectivity between memories sharing this keyword
        connected_pairs = 0
        total_pairs = 0
        for i, m1 in enumerate(memory_ids):
            for m2 in memory_ids[i + 1:]:
                total_pairs += 1
                if graph.path_exists(m1, m2, max_hops=max_hops):
                    connected_pairs += 1

        if total_pairs == 0:
            continue

        connectivity_ratio = connected_pairs / total_pairs
        # Scale: 0.0 connectivity = base weight, 1.0 = +50%. Computed from the
        # stored base weight, so repeated cycles are idempotent rather than
        # ratcheting every weight up to 1.0.
        boost = 1.0 + 0.5 * connectivity_ratio

        for entry in capped_entries:
            new_weight = round(min(entry["base_weight"] * boost, 1.0), 6)
            if abs(new_weight - entry["weight"]) > 1e-9:
                pending_updates.append((new_weight, entry["memory_id"], keyword))

    if pending_updates:
        await sqlite.batch_update_keyword_weights(pending_updates)

    updated = len(pending_updates)
    logger.info("Keyword reweighting: updated %d weights", updated)
    return updated
