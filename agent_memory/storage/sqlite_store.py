"""SQLite storage for memory metadata, access tracking, and compaction history."""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite

from agent_memory.config import DB_PATH
from agent_memory.models import CompactionResult, Memory, SaveDecision

_SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_log_index (
    id          TEXT PRIMARY KEY,
    namespace   TEXT NOT NULL DEFAULT 'default',
    session_id  TEXT NOT NULL,
    turn        INTEGER NOT NULL,
    timestamp   TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    byte_offset INTEGER
);

CREATE TABLE IF NOT EXISTS memories (
    id                TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    content           TEXT NOT NULL,
    summary           TEXT,
    raw_log_id        TEXT NOT NULL,
    namespace         TEXT NOT NULL DEFAULT 'default',
    session_id        TEXT NOT NULL,
    turn              INTEGER NOT NULL,
    valence           REAL,
    arousal           REAL,
    surprise          REAL,
    salience          REAL DEFAULT 0.5,
    access_count      INTEGER DEFAULT 0,
    last_accessed     TEXT,
    decay_score       REAL DEFAULT 1.0,
    compaction_gen    INTEGER DEFAULT 0,
    tier              TEXT DEFAULT 'hot',
    fast_pathed       INTEGER DEFAULT 0,
    is_semantic       INTEGER DEFAULT 0,
    graph_node_id     TEXT,
    vector_id         TEXT,
    spatial_embedding BLOB,
    scene_description TEXT,
    FOREIGN KEY (raw_log_id) REFERENCES raw_log_index(id)
);

CREATE TABLE IF NOT EXISTS memory_keywords (
    memory_id   TEXT NOT NULL,
    keyword     TEXT NOT NULL,
    weight      REAL DEFAULT 1.0,
    PRIMARY KEY (memory_id, keyword),
    FOREIGN KEY (memory_id) REFERENCES memories(id)
);

CREATE INDEX IF NOT EXISTS idx_keyword ON memory_keywords(keyword);
-- The memories(namespace) index is created in _migrate(), after the namespace
-- column is ensured (it may be absent on databases predating that column).

CREATE TABLE IF NOT EXISTS memory_access_log (
    id          TEXT PRIMARY KEY,
    memory_id   TEXT NOT NULL,
    accessed_at TEXT NOT NULL,
    access_type TEXT NOT NULL,
    session_id  TEXT,
    query       TEXT,
    FOREIGN KEY (memory_id) REFERENCES memories(id)
);

CREATE TABLE IF NOT EXISTS compaction_runs (
    id                  TEXT PRIMARY KEY,
    ran_at              TEXT NOT NULL,
    trigger             TEXT,
    memories_reviewed   INTEGER,
    memories_merged     INTEGER,
    memories_pruned     INTEGER,
    notes               TEXT,
    keywords_updated    INTEGER DEFAULT 0,
    edges_discovered    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS compaction_merges (
    compaction_id         TEXT NOT NULL,
    source_memory_ids     TEXT NOT NULL,
    resulting_memory_id   TEXT NOT NULL,
    validation_passed     INTEGER,
    avg_source_score      REAL,
    avg_merged_score      REAL,
    degradation           REAL,
    FOREIGN KEY (compaction_id) REFERENCES compaction_runs(id)
);

CREATE TABLE IF NOT EXISTS save_decisions (
    id                  TEXT PRIMARY KEY,
    raw_log_id          TEXT NOT NULL,
    session_id          TEXT NOT NULL,
    turn                INTEGER NOT NULL,
    decided_at          TEXT NOT NULL,
    decision            TEXT NOT NULL,
    reason              TEXT,
    confidence          REAL,
    gap_triggered       INTEGER DEFAULT 0,
    threshold_used      REAL,
    outcome_useful      INTEGER,
    outcome_assessed_at TEXT
);

-- A4: Retrieval decision logging for policy training
CREATE TABLE IF NOT EXISTS retrieval_decisions (
    id                  TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL,
    turn                INTEGER,
    query               TEXT NOT NULL,
    decided_at          TEXT NOT NULL,
    layers_queried      TEXT NOT NULL,
    graph_depth         INTEGER,
    mood_weight         REAL,
    top_k               INTEGER,
    memories_returned   TEXT NOT NULL,
    return_count        INTEGER NOT NULL,
    outcome_helpful     INTEGER,
    outcome_assessed_at TEXT
);

-- A3: Dream exploration logging
CREATE TABLE IF NOT EXISTS dream_exploration_runs (
    id                TEXT PRIMARY KEY,
    ran_at            TEXT NOT NULL,
    n_walks           INTEGER,
    edges_discovered  INTEGER,
    edges_committed   INTEGER,
    strategies_used   TEXT,
    notes             TEXT
);

CREATE TABLE IF NOT EXISTS dream_discovered_edges (
    id                  TEXT PRIMARY KEY,
    exploration_run_id  TEXT NOT NULL,
    source_memory_id    TEXT NOT NULL,
    target_memory_id    TEXT NOT NULL,
    similarity          REAL,
    relationship_type   TEXT,
    discovery_method    TEXT,
    committed           INTEGER DEFAULT 0,
    FOREIGN KEY (exploration_run_id) REFERENCES dream_exploration_runs(id)
);
"""


class SQLiteStore:
    """Async SQLite store for memory metadata."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DB_PATH
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._migrate()
        await self._db.commit()

    async def _migrate(self) -> None:
        """Apply additive schema migrations to pre-existing databases.

        ``CREATE TABLE IF NOT EXISTS`` does not alter an already-existing table,
        so columns added after a database was first created must be backfilled
        here. All migrations are additive and idempotent.
        """
        await self._add_column_if_missing(
            "memories", "namespace", "TEXT NOT NULL DEFAULT 'default'"
        )
        await self._add_column_if_missing(
            "raw_log_index", "namespace", "TEXT NOT NULL DEFAULT 'default'"
        )
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace)"
        )

    async def _add_column_if_missing(self, table: str, column: str, decl: str) -> None:
        async with self.db.execute(f"PRAGMA table_info({table})") as cur:
            existing = {row["name"] async for row in cur}
        if column not in existing:
            await self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        assert self._db is not None, "SQLiteStore not initialized — call initialize() first"
        return self._db

    # ── Raw log index ──

    async def index_raw_log(
        self, entry_id: str, session_id: str, turn: int,
        timestamp: str, file_path: str, byte_offset: int,
        namespace: str = "default",
    ) -> None:
        await self.db.execute(
            "INSERT OR IGNORE INTO raw_log_index "
            "(id, namespace, session_id, turn, timestamp, file_path, byte_offset) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (entry_id, namespace, session_id, turn, timestamp, file_path, byte_offset),
        )
        await self.db.commit()

    async def get_raw_log_ref(self, entry_id: str) -> dict | None:
        async with self.db.execute(
            "SELECT * FROM raw_log_index WHERE id = ?", (entry_id,)
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    # ── Memories ──

    async def save_memory(self, mem: Memory) -> None:
        await self.db.execute(
            """INSERT OR REPLACE INTO memories
            (id, created_at, updated_at, content, summary, raw_log_id, namespace, session_id, turn,
             valence, arousal, surprise, salience, access_count, last_accessed, decay_score,
             compaction_gen, tier, fast_pathed, is_semantic, graph_node_id, vector_id,
             spatial_embedding, scene_description)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                mem.id, mem.created_at, mem.updated_at, mem.content, mem.summary,
                mem.raw_log_id, mem.namespace, mem.session_id, mem.turn,
                mem.valence, mem.arousal, mem.surprise, mem.salience,
                mem.access_count, mem.last_accessed, mem.decay_score,
                mem.compaction_gen, mem.tier, int(mem.fast_pathed), int(mem.is_semantic),
                mem.graph_node_id, mem.vector_id,
                mem.spatial_embedding, mem.scene_description,
            ),
        )
        # Save keywords
        await self.db.execute("DELETE FROM memory_keywords WHERE memory_id = ?", (mem.id,))
        for kw, weight in mem.keywords:
            await self.db.execute(
                "INSERT INTO memory_keywords (memory_id, keyword, weight) VALUES (?, ?, ?)",
                (mem.id, kw, weight),
            )
        await self.db.commit()

    async def get_memory(
        self, memory_id: str, namespace: str | None = None,
    ) -> Memory | None:
        if namespace is None:
            sql, params = "SELECT * FROM memories WHERE id = ?", (memory_id,)
        else:
            sql = "SELECT * FROM memories WHERE id = ? AND namespace = ?"
            params = (memory_id, namespace)
        async with self.db.execute(sql, params) as cur:
            row = await cur.fetchone()
            if not row:
                return None
            mem = _row_to_memory(dict(row))

        # Load keywords
        async with self.db.execute(
            "SELECT keyword, weight FROM memory_keywords WHERE memory_id = ?", (memory_id,)
        ) as cur:
            mem.keywords = [(r["keyword"], r["weight"]) async for r in cur]
        return mem

    async def list_memories(
        self, tier: str | None = None, limit: int = 100, offset: int = 0,
    ) -> list[Memory]:
        if tier:
            sql = "SELECT * FROM memories WHERE tier = ? ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params: tuple = (tier, limit, offset)
        else:
            sql = "SELECT * FROM memories ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params = (limit, offset)

        memories = []
        async with self.db.execute(sql, params) as cur:
            async for row in cur:
                memories.append(_row_to_memory(dict(row)))

        # Batch-load keywords
        if memories:
            ids = [m.id for m in memories]
            placeholders = ",".join("?" for _ in ids)
            kw_map: dict[str, list[tuple[str, float]]] = {mid: [] for mid in ids}
            async with self.db.execute(
                f"SELECT memory_id, keyword, weight FROM memory_keywords WHERE memory_id IN ({placeholders})",
                ids,
            ) as cur:
                async for row in cur:
                    kw_map[row["memory_id"]].append((row["keyword"], row["weight"]))
            for m in memories:
                m.keywords = kw_map.get(m.id, [])

        return memories

    async def count_memories(self, tier: str | None = None) -> int:
        if tier:
            sql = "SELECT COUNT(*) as cnt FROM memories WHERE tier = ?"
            params: tuple = (tier,)
        else:
            sql = "SELECT COUNT(*) as cnt FROM memories"
            params = ()
        async with self.db.execute(sql, params) as cur:
            row = await cur.fetchone()
            return row["cnt"] if row else 0

    async def update_memory_access(
        self, memory_id: str, decay_score: float, access_count: int, last_accessed: str,
    ) -> None:
        await self.db.execute(
            "UPDATE memories SET decay_score = ?, access_count = ?, last_accessed = ?, updated_at = ? WHERE id = ?",
            (decay_score, access_count, last_accessed, last_accessed, memory_id),
        )
        await self.db.commit()

    async def update_memory_tier(self, memory_id: str, tier: str) -> None:
        await self.db.execute(
            "UPDATE memories SET tier = ? WHERE id = ?", (tier, memory_id)
        )
        await self.db.commit()

    async def update_memory_graph_ref(self, memory_id: str, graph_node_id: str) -> None:
        await self.db.execute(
            "UPDATE memories SET graph_node_id = ? WHERE id = ?", (graph_node_id, memory_id)
        )
        await self.db.commit()

    async def update_memory_vector_ref(self, memory_id: str, vector_id: str) -> None:
        await self.db.execute(
            "UPDATE memories SET vector_id = ? WHERE id = ?", (vector_id, memory_id)
        )
        await self.db.commit()

    async def update_memory_visual(
        self, memory_id: str, scene_description: str, spatial_embedding: bytes,
    ) -> None:
        await self.db.execute(
            "UPDATE memories SET scene_description = ?, spatial_embedding = ? WHERE id = ?",
            (scene_description, spatial_embedding, memory_id),
        )
        await self.db.commit()

    # ── Keyword search ──

    async def search_by_keywords(
        self, keywords: list[str], limit: int = 10, namespace: str | None = None,
    ) -> list[Memory]:
        if not keywords:
            return []
        placeholders = ",".join("?" for _ in keywords)
        ns_clause = "" if namespace is None else "AND m.namespace = ?"
        ns_params: tuple = () if namespace is None else (namespace,)
        sql = f"""
            SELECT m.*, SUM(mk.weight) as match_score
            FROM memories m
            JOIN memory_keywords mk ON m.id = mk.memory_id
            WHERE mk.keyword IN ({placeholders}) {ns_clause}
            GROUP BY m.id
            ORDER BY match_score * m.decay_score DESC
            LIMIT ?
        """
        memories = []
        async with self.db.execute(sql, (*keywords, *ns_params, limit)) as cur:
            async for row in cur:
                memories.append(_row_to_memory(dict(row)))

        # Load keywords for results
        if memories:
            ids = [m.id for m in memories]
            ph = ",".join("?" for _ in ids)
            kw_map: dict[str, list[tuple[str, float]]] = {mid: [] for mid in ids}
            async with self.db.execute(
                f"SELECT memory_id, keyword, weight FROM memory_keywords WHERE memory_id IN ({ph})",
                ids,
            ) as cur:
                async for row in cur:
                    kw_map[row["memory_id"]].append((row["keyword"], row["weight"]))
            for m in memories:
                m.keywords = kw_map.get(m.id, [])

        return memories

    async def search_by_content(
        self, terms: list[str], limit: int = 10, namespace: str | None = None,
    ) -> list[Memory]:
        """Substring search over memory content for the given terms (OR).

        A dependency-free fallback that retrieves memories even when no keywords
        were extracted (e.g. fast-path/first-turn saves) and no embeddings are
        available. This is what keeps retrieval working on the lite/edge profile,
        where the semantic vector layer is absent. Results are ranked by the
        number of distinct terms matched, weighted by decay score.
        """
        terms = [t for t in (t.strip() for t in terms) if t]
        if not terms:
            return []
        # One LIKE clause per term; score by how many distinct terms match.
        # The match-count expression can't be referenced by alias inside WHERE
        # on all SQLite builds, so it is repeated (and its params supplied twice).
        match_expr = " + ".join(
            "(CASE WHEN m.content LIKE ? THEN 1 ELSE 0 END)" for _ in terms
        )
        like_params: list[object] = [f"%{t}%" for t in terms]
        ns_clause = "" if namespace is None else "AND m.namespace = ?"
        ns_params: list[object] = [] if namespace is None else [namespace]
        sql = f"""
            SELECT m.*, ({match_expr}) AS match_count
            FROM memories m
            WHERE ({match_expr}) > 0 {ns_clause}
            ORDER BY match_count * m.decay_score DESC
            LIMIT ?
        """
        all_params = [*like_params, *like_params, *ns_params, limit]
        memories: list[Memory] = []
        async with self.db.execute(sql, all_params) as cur:
            async for row in cur:
                d = dict(row)
                d.pop("match_count", None)
                memories.append(_row_to_memory(d))

        # Load keywords for results (mirror search_by_keywords).
        if memories:
            ids = [m.id for m in memories]
            ph = ",".join("?" for _ in ids)
            kw_map: dict[str, list[tuple[str, float]]] = {mid: [] for mid in ids}
            async with self.db.execute(
                f"SELECT memory_id, keyword, weight FROM memory_keywords WHERE memory_id IN ({ph})",
                ids,
            ) as cur:
                async for row in cur:
                    kw_map[row["memory_id"]].append((row["keyword"], row["weight"]))
            for m in memories:
                m.keywords = kw_map.get(m.id, [])

        return memories

    async def update_keyword_weight(self, memory_id: str, keyword: str, weight: float) -> None:
        """Update a single keyword weight (used by keyword reweighting — A2.5)."""
        await self.db.execute(
            "UPDATE memory_keywords SET weight = MIN(?, 1.0) WHERE memory_id = ? AND keyword = ?",
            (weight, memory_id, keyword),
        )
        await self.db.commit()

    async def batch_update_keyword_weights(
        self, updates: list[tuple[float, str, str]],
    ) -> None:
        """Update multiple keyword weights in a single transaction (A2.5).

        Each tuple is (new_weight, memory_id, keyword).
        """
        await self.db.executemany(
            "UPDATE memory_keywords SET weight = MIN(?, 1.0) WHERE memory_id = ? AND keyword = ?",
            updates,
        )
        await self.db.commit()

    async def get_all_keywords_with_memories(self, tiers: list[str] | None = None) -> list[dict]:
        """Return all keyword-memory associations for active tiers (A2.5)."""
        tiers_to_use = tiers or ["hot", "warm"]
        placeholders = ",".join("?" for _ in tiers_to_use)
        sql = f"""
            SELECT mk.keyword, mk.memory_id, mk.weight
            FROM memory_keywords mk
            JOIN memories m ON mk.memory_id = m.id
            WHERE m.tier IN ({placeholders})
        """
        rows = []
        async with self.db.execute(sql, tiers_to_use) as cur:
            async for row in cur:
                rows.append({"keyword": row["keyword"], "memory_id": row["memory_id"], "weight": row["weight"]})
        return rows

    # ── Access log ──

    async def log_access(
        self, access_id: str, memory_id: str, accessed_at: str,
        access_type: str, session_id: str | None = None, query: str | None = None,
    ) -> None:
        await self.db.execute(
            "INSERT INTO memory_access_log (id, memory_id, accessed_at, access_type, session_id, query) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (access_id, memory_id, accessed_at, access_type, session_id, query),
        )
        await self.db.commit()

    async def get_recent_access_queries(
        self, session_id: str, limit: int = 20,
    ) -> list[str]:
        """Return recent retrieval queries for a session (A2.1 gap detection)."""
        async with self.db.execute(
            "SELECT query FROM memory_access_log "
            "WHERE session_id = ? AND query IS NOT NULL "
            "ORDER BY accessed_at DESC LIMIT ?",
            (session_id, limit),
        ) as cur:
            return [row["query"] async for row in cur]

    async def get_failed_retrieval_keywords(self, session_id: str, lookback: int = 20) -> list[str]:
        """Identify keywords from queries that yielded no or only cold-tier results (A2.1).

        Returns keywords that represent retrieval gaps.
        """
        # Get recent queries
        queries = await self.get_recent_access_queries(session_id, limit=lookback)
        if not queries:
            return []

        # Check which queries only returned cold-tier or no results
        gap_keywords: list[str] = []
        for query in queries:
            async with self.db.execute(
                """SELECT m.tier FROM memory_access_log mal
                   JOIN memories m ON mal.memory_id = m.id
                   WHERE mal.query = ? AND mal.session_id = ?""",
                (query, session_id),
            ) as cur:
                tiers = [row["tier"] async for row in cur]

            # If no results or only cold-tier, this is a gap
            if not tiers or all(t == "cold" for t in tiers):
                words = [w.lower().strip() for w in query.split() if len(w) > 2]
                gap_keywords.extend(words)

        return list(set(gap_keywords))

    # ── Save decisions ──

    async def log_save_decision(self, dec: SaveDecision) -> None:
        await self.db.execute(
            "INSERT INTO save_decisions "
            "(id, raw_log_id, session_id, turn, decided_at, decision, reason, confidence, gap_triggered, threshold_used) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (dec.id, dec.raw_log_id, dec.session_id, dec.turn, dec.decided_at,
             dec.decision, dec.reason, dec.confidence, int(dec.gap_triggered), dec.threshold_used),
        )
        await self.db.commit()

    async def update_save_outcome(self, decision_id: str, useful: bool, assessed_at: str) -> None:
        """Mark whether a saved memory turned out to be useful (A4)."""
        await self.db.execute(
            "UPDATE save_decisions SET outcome_useful = ?, outcome_assessed_at = ? WHERE id = ?",
            (int(useful), assessed_at, decision_id),
        )
        await self.db.commit()

    async def get_unassessed_save_decisions(self, lookback_days: int = 30) -> list[dict]:
        """Get save decisions that haven't been assessed yet (A4)."""
        async with self.db.execute(
            """SELECT sd.id, sd.raw_log_id, m.id as memory_id, m.access_count
               FROM save_decisions sd
               LEFT JOIN memories m ON m.raw_log_id = sd.raw_log_id
               WHERE sd.decision IN ('save', 'fast_path')
                 AND sd.id NOT IN (
                     SELECT id FROM save_decisions WHERE outcome_useful IS NOT NULL
                 )
                 AND sd.decided_at < datetime('now', ?)""",
            (f"-{lookback_days} days",),
        ) as cur:
            return [dict(row) async for row in cur]

    # ── Retrieval decisions (A4) ──

    async def log_retrieval_decision(
        self, decision_id: str, session_id: str, turn: int | None,
        query: str, decided_at: str, layers_queried: list[str],
        graph_depth: int, mood_weight: float, top_k: int,
        memory_ids: list[str], return_count: int,
    ) -> None:
        await self.db.execute(
            "INSERT INTO retrieval_decisions "
            "(id, session_id, turn, query, decided_at, layers_queried, graph_depth, "
            "mood_weight, top_k, memories_returned, return_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (decision_id, session_id or "", turn, query, decided_at,
             json.dumps(layers_queried), graph_depth, mood_weight, top_k,
             json.dumps(memory_ids), return_count),
        )
        await self.db.commit()

    async def get_unassessed_retrieval_decisions(self) -> list[dict]:
        """Get retrieval decisions not yet assessed (A4)."""
        async with self.db.execute(
            """SELECT id, session_id, turn, query
               FROM retrieval_decisions
               WHERE outcome_helpful IS NULL
                 AND decided_at < datetime('now', '-1 hour')"""
        ) as cur:
            return [dict(row) async for row in cur]

    async def update_retrieval_outcome(self, decision_id: str, helpful: bool, assessed_at: str) -> None:
        await self.db.execute(
            "UPDATE retrieval_decisions SET outcome_helpful = ?, outcome_assessed_at = ? WHERE id = ?",
            (int(helpful), assessed_at, decision_id),
        )
        await self.db.commit()

    async def get_retrieval_followups(self, session_id: str, turn: int, window: int = 3) -> list[str]:
        """Get follow-up queries within a turn window (A4 outcome assessment)."""
        async with self.db.execute(
            "SELECT query FROM retrieval_decisions WHERE session_id = ? AND turn > ? AND turn <= ?",
            (session_id, turn, turn + window),
        ) as cur:
            return [row["query"] async for row in cur]

    # ── Dream exploration logging (A3) ──

    async def log_dream_run(
        self, run_id: str, ran_at: str, n_walks: int,
        edges_discovered: int, edges_committed: int,
        strategies: list[str], notes: str | None = None,
    ) -> None:
        await self.db.execute(
            "INSERT INTO dream_exploration_runs "
            "(id, ran_at, n_walks, edges_discovered, edges_committed, strategies_used, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, ran_at, n_walks, edges_discovered, edges_committed,
             json.dumps(strategies), notes),
        )
        await self.db.commit()

    async def log_dream_edge(
        self, edge_id: str, run_id: str, source_id: str, target_id: str,
        similarity: float, relationship_type: str, discovery_method: str,
        committed: bool = False,
    ) -> None:
        await self.db.execute(
            "INSERT INTO dream_discovered_edges "
            "(id, exploration_run_id, source_memory_id, target_memory_id, "
            "similarity, relationship_type, discovery_method, committed) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (edge_id, run_id, source_id, target_id, similarity,
             relationship_type, discovery_method, int(committed)),
        )
        await self.db.commit()

    # ── Compaction ──

    async def log_compaction_run(self, result: CompactionResult) -> None:
        await self.db.execute(
            "INSERT INTO compaction_runs "
            "(id, ran_at, trigger, memories_reviewed, memories_merged, memories_pruned, "
            "notes, keywords_updated, edges_discovered) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (result.id, result.ran_at, result.trigger, result.memories_reviewed,
             result.memories_merged, result.memories_pruned, result.notes,
             result.keywords_updated, result.edges_discovered),
        )
        await self.db.commit()

    async def log_compaction_merge(
        self, compaction_id: str, source_ids: list[str], resulting_id: str,
        validation_passed: bool | None = None,
        avg_source_score: float | None = None,
        avg_merged_score: float | None = None,
        degradation: float | None = None,
    ) -> None:
        await self.db.execute(
            "INSERT INTO compaction_merges "
            "(compaction_id, source_memory_ids, resulting_memory_id, "
            "validation_passed, avg_source_score, avg_merged_score, degradation) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (compaction_id, json.dumps(source_ids), resulting_id,
             int(validation_passed) if validation_passed is not None else None,
             avg_source_score, avg_merged_score, degradation),
        )
        await self.db.commit()

    async def get_compaction_candidates(self, threshold: float = 0.7) -> list[Memory]:
        """Return memories that are candidates for compaction."""
        sql = """
            SELECT m.* FROM memories m
            LEFT JOIN (
                SELECT memory_id, COUNT(*) as edge_count
                FROM memory_keywords
                GROUP BY memory_id
            ) kc ON m.id = kc.memory_id
            WHERE m.tier = 'hot'
              AND m.fast_pathed = 0
              AND NOT (m.compaction_gen = 0 AND m.access_count > 5)
            ORDER BY ((1 - m.decay_score) * 0.6 + (1 - m.salience) * 0.4) DESC
        """
        candidates = []
        async with self.db.execute(sql) as cur:
            async for row in cur:
                mem = _row_to_memory(dict(row))
                score = (1 - mem.decay_score) * 0.6 + (1 - mem.salience) * 0.4
                if score > threshold:
                    candidates.append(mem)

        # Load keywords
        if candidates:
            ids = [m.id for m in candidates]
            ph = ",".join("?" for _ in ids)
            kw_map: dict[str, list[tuple[str, float]]] = {mid: [] for mid in ids}
            async with self.db.execute(
                f"SELECT memory_id, keyword, weight FROM memory_keywords WHERE memory_id IN ({ph})",
                ids,
            ) as cur:
                async for row in cur:
                    kw_map[row["memory_id"]].append((row["keyword"], row["weight"]))
            for m in candidates:
                m.keywords = kw_map.get(m.id, [])

        return candidates

    # ── Policy data export (A4.4) ──

    async def export_save_policy_data(self) -> list[dict]:
        """Export assessed save decisions for policy training."""
        rows = []
        async with self.db.execute(
            """SELECT sd.confidence, sd.decision, sd.gap_triggered,
                      m.valence, m.arousal, m.surprise, m.salience,
                      sd.outcome_useful
               FROM save_decisions sd
               LEFT JOIN memories m ON m.raw_log_id = sd.raw_log_id
               WHERE sd.outcome_useful IS NOT NULL"""
        ) as cur:
            async for row in cur:
                rows.append(dict(row))
        return rows

    async def export_retrieval_policy_data(self) -> list[dict]:
        """Export assessed retrieval decisions for policy training."""
        rows = []
        async with self.db.execute(
            """SELECT layers_queried, graph_depth, mood_weight, top_k,
                      return_count, outcome_helpful
               FROM retrieval_decisions
               WHERE outcome_helpful IS NOT NULL"""
        ) as cur:
            async for row in cur:
                rows.append(dict(row))
        return rows

    async def get_memories_with_vectors(self, tiers: list[str] | None = None) -> list[dict]:
        """Get memories that have vector embeddings (for dream explorer)."""
        tier_clause = ""
        params: tuple = ()
        if tiers:
            placeholders = ",".join("?" for _ in tiers)
            tier_clause = f"AND tier IN ({placeholders})"
            params = tuple(tiers)
        sql = f"SELECT id, session_id, vector_id FROM memories WHERE vector_id IS NOT NULL {tier_clause}"
        rows = []
        async with self.db.execute(sql, params) as cur:
            async for row in cur:
                rows.append(dict(row))
        return rows


def _row_to_memory(row: dict) -> Memory:
    """Convert a SQLite row dict to a Memory dataclass."""
    return Memory(
        id=row["id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        content=row["content"],
        summary=row.get("summary"),
        raw_log_id=row["raw_log_id"],
        namespace=row.get("namespace", "default"),
        session_id=row["session_id"],
        turn=row["turn"],
        valence=row.get("valence", 0.0),
        arousal=row.get("arousal", 0.0),
        surprise=row.get("surprise", 0.0),
        salience=row.get("salience", 0.5),
        access_count=row.get("access_count", 0),
        last_accessed=row.get("last_accessed"),
        decay_score=row.get("decay_score", 1.0),
        compaction_gen=row.get("compaction_gen", 0),
        tier=row.get("tier", "hot"),
        fast_pathed=bool(row.get("fast_pathed", 0)),
        is_semantic=bool(row.get("is_semantic", 0)),
        graph_node_id=row.get("graph_node_id"),
        vector_id=row.get("vector_id"),
        spatial_embedding=row.get("spatial_embedding"),
        scene_description=row.get("scene_description"),
    )
