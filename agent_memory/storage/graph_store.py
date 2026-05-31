"""Kuzu graph database for semantic relationships between memories and entities.

Encodes RELATES_TO, MENTIONS, and EVOLVED_FROM edges. Enables multi-hop
reasoning at retrieval time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from agent_memory.config import GRAPH_DIR

if TYPE_CHECKING:
    import kuzu

logger = logging.getLogger(__name__)


class GraphStore:
    """Kuzu-backed graph for memory relationships.

    The ``kuzu`` import is deferred to ``initialize()`` so this module can be
    imported on profiles without the ``graph`` extra installed (the graph layer
    is optional on the lite/edge profile).
    """

    def __init__(self, graph_dir: Path | None = None) -> None:
        self.graph_dir = graph_dir or GRAPH_DIR
        self._db: kuzu.Database | None = None
        self._conn: kuzu.Connection | None = None

    def initialize(self) -> None:
        try:
            import kuzu
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "GraphStore requires the 'graph' extra. "
                "Install with: pip install agent-memory[graph]"
            ) from exc
        # Kuzu creates the database directory itself; only ensure the parent exists
        self.graph_dir.parent.mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(str(self.graph_dir))
        self._conn = kuzu.Connection(self._db)
        self._create_schema()

    def close(self) -> None:
        # kuzu doesn't require explicit close but we clear refs
        self._conn = None
        self._db = None

    @property
    def conn(self) -> kuzu.Connection:
        assert self._conn is not None, "GraphStore not initialized — call initialize() first"
        return self._conn

    def _create_schema(self) -> None:
        """Create node and relationship tables if they don't exist."""
        stmts = [
            # Node tables
            """CREATE NODE TABLE IF NOT EXISTS Memory (
                id STRING,
                summary STRING,
                tier STRING,
                salience DOUBLE,
                valence DOUBLE,
                compaction_gen INT64,
                created_at STRING,
                namespace STRING,
                PRIMARY KEY (id)
            )""",
            """CREATE NODE TABLE IF NOT EXISTS Entity (
                id STRING,
                name STRING,
                type STRING,
                PRIMARY KEY (id)
            )""",
            # Relationship tables
            """CREATE REL TABLE IF NOT EXISTS RELATES_TO (
                FROM Memory TO Memory,
                weight DOUBLE,
                relationship_type STRING,
                created_at STRING
            )""",
            """CREATE REL TABLE IF NOT EXISTS MENTIONS (
                FROM Memory TO Entity,
                weight DOUBLE
            )""",
            """CREATE REL TABLE IF NOT EXISTS EVOLVED_FROM (
                FROM Memory TO Memory,
                compaction_id STRING,
                created_at STRING
            )""",
        ]
        for stmt in stmts:
            try:
                self.conn.execute(stmt)
            except Exception as e:
                # Table may already exist — kuzu raises on duplicate creation
                logger.debug("Schema statement skipped: %s", e)

    # ── Node operations ──

    def add_memory_node(
        self, memory_id: str, summary: str, tier: str = "hot",
        salience: float = 0.5, valence: float = 0.0,
        compaction_gen: int = 0, created_at: str = "", namespace: str = "default",
    ) -> None:
        self.conn.execute(
            "MERGE (m:Memory {id: $id}) SET m.summary = $summary, m.tier = $tier, "
            "m.salience = $salience, m.valence = $valence, "
            "m.compaction_gen = $compaction_gen, m.created_at = $created_at, "
            "m.namespace = $namespace",
            {
                "id": memory_id, "summary": summary or "", "tier": tier,
                "salience": salience, "valence": valence,
                "compaction_gen": compaction_gen, "created_at": created_at,
                "namespace": namespace,
            },
        )

    def add_entity_node(self, entity_id: str, name: str, entity_type: str) -> None:
        self.conn.execute(
            "MERGE (e:Entity {id: $id}) SET e.name = $name, e.type = $type",
            {"id": entity_id, "name": name, "type": entity_type},
        )

    # ── Edge operations ──

    def add_relates_to(
        self, from_id: str, to_id: str, weight: float = 1.0,
        relationship_type: str = "supports", created_at: str = "",
    ) -> None:
        self.conn.execute(
            "MATCH (a:Memory {id: $from_id}), (b:Memory {id: $to_id}) "
            "CREATE (a)-[:RELATES_TO {weight: $weight, relationship_type: $rtype, created_at: $cat}]->(b)",
            {"from_id": from_id, "to_id": to_id, "weight": weight, "rtype": relationship_type, "cat": created_at},
        )

    def add_mentions(self, memory_id: str, entity_id: str, weight: float = 1.0) -> None:
        self.conn.execute(
            "MATCH (m:Memory {id: $mid}), (e:Entity {id: $eid}) "
            "CREATE (m)-[:MENTIONS {weight: $weight}]->(e)",
            {"mid": memory_id, "eid": entity_id, "weight": weight},
        )

    def add_evolved_from(
        self, new_id: str, source_id: str, compaction_id: str = "", created_at: str = "",
    ) -> None:
        self.conn.execute(
            "MATCH (a:Memory {id: $new_id}), (b:Memory {id: $src_id}) "
            "CREATE (a)-[:EVOLVED_FROM {compaction_id: $cid, created_at: $cat}]->(b)",
            {"new_id": new_id, "src_id": source_id, "cid": compaction_id, "cat": created_at},
        )

    # ── Queries ──

    def get_related_memories(
        self, memory_id: str, max_depth: int = 2, min_weight: float = 0.0,
    ) -> list[dict]:
        """Traverse RELATES_TO edges up to max_depth hops from a memory node.

        Returns list of dicts with keys: id, summary, tier, salience, depth, weight.
        """
        query = (
            "MATCH (start:Memory {id: $id})-[r:RELATES_TO*1.." + str(max_depth) + "]->(m:Memory) "
            "RETURN DISTINCT m.id AS id, m.summary AS summary, m.tier AS tier, "
            "m.salience AS salience, length(r) AS depth"
        )
        result = self.conn.execute(query, {"id": memory_id})
        rows = []
        while result.has_next():
            row = result.get_next()
            rows.append({
                "id": row[0], "summary": row[1], "tier": row[2],
                "salience": row[3], "depth": row[4],
            })
        return rows

    def get_memory_entities(self, memory_id: str) -> list[dict]:
        """Get all entities mentioned by a memory."""
        result = self.conn.execute(
            "MATCH (m:Memory {id: $id})-[r:MENTIONS]->(e:Entity) "
            "RETURN e.id AS id, e.name AS name, e.type AS type, r.weight AS weight",
            {"id": memory_id},
        )
        rows = []
        while result.has_next():
            row = result.get_next()
            rows.append({"id": row[0], "name": row[1], "type": row[2], "weight": row[3]})
        return rows

    def get_evolution_lineage(self, memory_id: str) -> list[dict]:
        """Trace the full lineage of a compacted memory back to originals."""
        result = self.conn.execute(
            "MATCH (m:Memory {id: $id})-[r:EVOLVED_FROM*1..10]->(orig:Memory) "
            "RETURN orig.id AS id, orig.summary AS summary, orig.compaction_gen AS gen, "
            "length(r) AS depth ORDER BY depth",
            {"id": memory_id},
        )
        rows = []
        while result.has_next():
            row = result.get_next()
            rows.append({"id": row[0], "summary": row[1], "gen": row[2], "depth": row[3]})
        return rows

    def get_edge_count(self, memory_id: str) -> int:
        """Count RELATES_TO edges connected to a memory (both directions)."""
        result = self.conn.execute(
            "MATCH (m:Memory {id: $id})-[r:RELATES_TO]-() RETURN count(r) AS cnt",
            {"id": memory_id},
        )
        if result.has_next():
            return result.get_next()[0]
        return 0

    def replicate_edges_to_new_node(self, source_ids: list[str], new_id: str) -> None:
        """Copy all RELATES_TO edges from source memories to a new compacted memory node.

        Skips edges between source nodes (they are being merged).
        """
        src_set = set(source_ids)
        for src_id in source_ids:
            # Outgoing edges
            result = self.conn.execute(
                "MATCH (a:Memory {id: $id})-[r:RELATES_TO]->(b:Memory) "
                "RETURN b.id AS target_id, r.weight AS weight, "
                "r.relationship_type AS rtype, r.created_at AS cat",
                {"id": src_id},
            )
            while result.has_next():
                row = result.get_next()
                target = row[0]
                if target not in src_set and target != new_id:
                    try:
                        self.add_relates_to(new_id, target, row[1], row[2], row[3])
                    except Exception:
                        pass  # dedup — edge may already exist

            # Incoming edges
            result = self.conn.execute(
                "MATCH (a:Memory)-[r:RELATES_TO]->(b:Memory {id: $id}) "
                "RETURN a.id AS source_id, r.weight AS weight, "
                "r.relationship_type AS rtype, r.created_at AS cat",
                {"id": src_id},
            )
            while result.has_next():
                row = result.get_next()
                source = row[0]
                if source not in src_set and source != new_id:
                    try:
                        self.add_relates_to(source, new_id, row[1], row[2], row[3])
                    except Exception:
                        pass

    def path_exists(self, from_id: str, to_id: str, max_hops: int = 2) -> bool:
        """Check if a path exists between two memory nodes via RELATES_TO edges."""
        try:
            query = (
                "MATCH (a:Memory {id: $from_id})-[r:RELATES_TO*1.." + str(max_hops) + "]->(b:Memory {id: $to_id}) "
                "RETURN count(r) AS cnt LIMIT 1"
            )
            result = self.conn.execute(query, {"from_id": from_id, "to_id": to_id})
            if result.has_next():
                return result.get_next()[0] > 0
        except Exception:
            pass
        return False

    def update_memory_tier(self, memory_id: str, tier: str) -> None:
        self.conn.execute(
            "MATCH (m:Memory {id: $id}) SET m.tier = $tier",
            {"id": memory_id, "tier": tier},
        )
