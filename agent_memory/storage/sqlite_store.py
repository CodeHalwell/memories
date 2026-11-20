"""SQLite storage for memory metadata, indexes, access tracking and policy data.

Besides the memory records themselves this store holds two FTS5 (BM25) indexes:

  - ``memories_fts``  — curated memories (content, summary, keywords, entities)
  - ``raw_turns_fts`` — **every** logged turn, curated or not

The raw-turn index is what makes a "skip" decision recoverable: a turn that was
never curated into a memory is still retrievable as raw evidence. It also gives
the decision model curation-independent outcome labels (see
``agent_memory.policy.outcome_assessor``).

FTS5 ships with the SQLite in standard CPython builds. If it is missing, the
store falls back to LIKE-based search and ``fts_enabled`` is False.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from agent_memory.config import DB_PATH
from agent_memory.models import CompactionResult, Memory, SaveDecision
from agent_memory.text import content_terms, fts_query

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_log_index (
    id          TEXT PRIMARY KEY,
    namespace   TEXT NOT NULL DEFAULT 'default',
    session_id  TEXT NOT NULL,
    turn        INTEGER NOT NULL,
    timestamp   TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    byte_offset INTEGER,
    role        TEXT DEFAULT 'assistant',
    purged      INTEGER DEFAULT 0
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
    role              TEXT DEFAULT 'assistant',
    event_time        TEXT,
    FOREIGN KEY (raw_log_id) REFERENCES raw_log_index(id)
);

CREATE TABLE IF NOT EXISTS memory_keywords (
    memory_id   TEXT NOT NULL,
    keyword     TEXT NOT NULL,
    weight      REAL DEFAULT 1.0,
    base_weight REAL,
    PRIMARY KEY (memory_id, keyword),
    FOREIGN KEY (memory_id) REFERENCES memories(id)
);

CREATE INDEX IF NOT EXISTS idx_keyword ON memory_keywords(keyword);
-- Indexes on columns that may be absent from databases created by older
-- versions are created in _migrate(), after the columns are ensured.

CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id   TEXT NOT NULL,
    namespace   TEXT NOT NULL DEFAULT 'default',
    entity_key  TEXT NOT NULL,
    name        TEXT NOT NULL,
    type        TEXT,
    PRIMARY KEY (memory_id, entity_key)
);

CREATE INDEX IF NOT EXISTS idx_entity_key ON memory_entities(namespace, entity_key);

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
    outcome_assessed_at TEXT,
    namespace           TEXT DEFAULT 'default',
    role                TEXT,
    policy_prob         REAL,
    policy_source       TEXT,
    features            TEXT
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
    outcome_assessed_at TEXT,
    namespace           TEXT DEFAULT 'default',
    top_score           REAL,
    lexical_hit         INTEGER,
    miss                INTEGER
);

-- The ranked list each retrieval returned (memories and raw turns).
CREATE TABLE IF NOT EXISTS retrieval_hits (
    retrieval_id  TEXT NOT NULL,
    rank          INTEGER NOT NULL,
    memory_id     TEXT,
    raw_log_id    TEXT,
    score         REAL,
    layers        TEXT,
    namespace     TEXT DEFAULT 'default',
    retrieved_at  TEXT NOT NULL,
    PRIMARY KEY (retrieval_id, rank)
);

CREATE INDEX IF NOT EXISTS idx_hits_raw ON retrieval_hits(raw_log_id);
CREATE INDEX IF NOT EXISTS idx_hits_memory ON retrieval_hits(memory_id);

-- Usage attribution: was a retrieved item actually used in the response?
CREATE TABLE IF NOT EXISTS memory_usage (
    retrieval_id  TEXT NOT NULL,
    memory_id     TEXT,
    raw_log_id    TEXT,
    used          INTEGER NOT NULL,
    overlap       REAL,
    source        TEXT,
    recorded_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_usage_retrieval ON memory_usage(retrieval_id);

-- A5: decision-model training runs (audit trail for promotions)
CREATE TABLE IF NOT EXISTS policy_training_runs (
    id            TEXT PRIMARY KEY,
    trained_at    TEXT NOT NULL,
    model_type    TEXT NOT NULL,
    n_train       INTEGER,
    n_validation  INTEGER,
    metrics       TEXT,
    promoted      INTEGER DEFAULT 0,
    model_path    TEXT,
    notes         TEXT
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

# Columns added after the first release, applied to pre-existing databases.
_ADDITIVE_COLUMNS: list[tuple[str, str, str]] = [
    ("memories", "namespace", "TEXT NOT NULL DEFAULT 'default'"),
    ("memories", "role", "TEXT DEFAULT 'assistant'"),
    ("memories", "event_time", "TEXT"),
    ("raw_log_index", "namespace", "TEXT NOT NULL DEFAULT 'default'"),
    ("raw_log_index", "role", "TEXT DEFAULT 'assistant'"),
    ("raw_log_index", "purged", "INTEGER DEFAULT 0"),
    ("memory_keywords", "base_weight", "REAL"),
    ("save_decisions", "namespace", "TEXT DEFAULT 'default'"),
    ("save_decisions", "role", "TEXT"),
    ("save_decisions", "policy_prob", "REAL"),
    ("save_decisions", "policy_source", "TEXT"),
    ("save_decisions", "features", "TEXT"),
    ("retrieval_decisions", "namespace", "TEXT DEFAULT 'default'"),
    ("retrieval_decisions", "top_score", "REAL"),
    ("retrieval_decisions", "lexical_hit", "INTEGER"),
    ("retrieval_decisions", "miss", "INTEGER"),
]

_MEMORY_COLUMNS = (
    "id, created_at, updated_at, content, summary, raw_log_id, namespace, session_id, turn, "
    "valence, arousal, surprise, salience, access_count, last_accessed, decay_score, "
    "compaction_gen, tier, fast_pathed, is_semantic, graph_node_id, vector_id, "
    "spatial_embedding, scene_description, role, event_time"
)


def entity_key(name: str) -> str:
    """Normalised lookup key for an entity name ("Dr. Jane O'Neil" -> "dr jane o neil")."""
    return " ".join(re.findall(r"[^\W_]+", (name or "").lower()))


class SQLiteStore:
    """Async SQLite store for memory metadata."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DB_PATH
        self._db: aiosqlite.Connection | None = None
        self.fts_enabled = False

    async def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.executescript(_SCHEMA)
        await self._migrate()
        await self._init_fts()
        await self._db.commit()

    async def _migrate(self) -> None:
        """Apply additive schema migrations to pre-existing databases.

        ``CREATE TABLE IF NOT EXISTS`` does not alter an already-existing table,
        so columns added after a database was first created must be backfilled
        here. All migrations are additive and idempotent.
        """
        for table, column, decl in _ADDITIVE_COLUMNS:
            await self._add_column_if_missing(table, column, decl)
        await self.db.execute(
            "UPDATE memory_keywords SET base_weight = weight WHERE base_weight IS NULL"
        )
        for stmt in (
            "CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace)",
            "CREATE INDEX IF NOT EXISTS idx_memories_raw ON memories(raw_log_id)",
            "CREATE INDEX IF NOT EXISTS idx_raw_session ON raw_log_index(namespace, session_id)",
            "CREATE INDEX IF NOT EXISTS idx_retrieval_ns ON retrieval_decisions(namespace, decided_at)",
            "CREATE INDEX IF NOT EXISTS idx_save_raw ON save_decisions(raw_log_id)",
        ):
            await self.db.execute(stmt)

    async def _init_fts(self) -> None:
        try:
            await self.db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5("
                "content, summary, keywords, entities, tokenize='porter unicode61')"
            )
            await self.db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS raw_turns_fts USING fts5("
                "content, tokenize='porter unicode61')"
            )
            self.fts_enabled = True
        except Exception:  # pragma: no cover - SQLite built without FTS5
            logger.warning("SQLite FTS5 unavailable; falling back to LIKE search")
            self.fts_enabled = False
            return
        # Backfill the memory index for databases created before it existed.
        async with self.db.execute("SELECT COUNT(*) AS n FROM memories_fts") as cur:
            indexed = (await cur.fetchone())["n"]
        async with self.db.execute("SELECT COUNT(*) AS n FROM memories") as cur:
            total = (await cur.fetchone())["n"]
        if total and not indexed:
            async with self.db.execute("SELECT id FROM memories") as cur:
                ids = [r["id"] async for r in cur]
            for mid in ids:
                await self._index_memory_fts(mid)

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
        namespace: str = "default", content: str | None = None,
        role: str = "assistant",
    ) -> None:
        """Index a raw log entry; with ``content``, also add it to the raw-turn
        full-text index so it is retrievable whether or not it is curated."""
        cur = await self.db.execute(
            "INSERT OR IGNORE INTO raw_log_index "
            "(id, namespace, session_id, turn, timestamp, file_path, byte_offset, role) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (entry_id, namespace, session_id, turn, timestamp, file_path, byte_offset, role),
        )
        if content and self.fts_enabled and cur.rowcount:
            await self.db.execute(
                "INSERT INTO raw_turns_fts(rowid, content) VALUES (?, ?)",
                (cur.lastrowid, content),
            )
        await self.db.commit()

    async def raw_index_size(self) -> int:
        if not self.fts_enabled:
            return 0
        async with self.db.execute("SELECT COUNT(*) AS n FROM raw_turns_fts") as cur:
            return (await cur.fetchone())["n"]

    async def iter_raw_refs(self) -> list[dict]:
        async with self.db.execute(
            "SELECT rowid, * FROM raw_log_index WHERE purged = 0 ORDER BY rowid"
        ) as cur:
            return [dict(r) async for r in cur]

    async def add_raw_fts(self, rowid: int, content: str) -> None:
        if self.fts_enabled:
            await self.db.execute(
                "INSERT INTO raw_turns_fts(rowid, content) VALUES (?, ?)", (rowid, content),
            )

    async def get_raw_log_ref(self, entry_id: str) -> dict | None:
        async with self.db.execute(
            "SELECT * FROM raw_log_index WHERE id = ?", (entry_id,)
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def session_has_prior_turns(
        self, session_id: str, namespace: str, exclude_id: str,
    ) -> bool:
        """True if the session already has logged turns (restart-safe first-turn check)."""
        async with self.db.execute(
            "SELECT 1 FROM raw_log_index WHERE session_id = ? AND namespace = ? AND id != ? LIMIT 1",
            (session_id, namespace, exclude_id),
        ) as cur:
            return await cur.fetchone() is not None

    async def get_raw_turns(self, raw_ids: list[str]) -> dict[str, dict]:
        """Load raw turns (with content from the FTS index) keyed by raw log id."""
        if not raw_ids:
            return {}
        ph = ",".join("?" for _ in raw_ids)
        out: dict[str, dict] = {}
        content_expr = (
            "(SELECT content FROM raw_turns_fts WHERE raw_turns_fts.rowid = r.rowid)"
            if self.fts_enabled else "NULL"
        )
        async with self.db.execute(
            f"SELECT r.*, {content_expr} AS content FROM raw_log_index r "
            f"WHERE r.id IN ({ph}) AND r.purged = 0",
            raw_ids,
        ) as cur:
            async for row in cur:
                out[row["id"]] = dict(row)
        return out

    async def search_raw_fts(
        self, query: str, limit: int = 30, namespace: str | None = None,
        since: str | None = None, until: str | None = None,
    ) -> list[tuple[str, float]]:
        """BM25 search over every logged turn. Returns (raw_log_id, score).

        ``since``/``until`` (normalised ISO timestamps) restrict turns by when
        they happened, before the LIMIT is applied.
        """
        match = fts_query(query)
        if not match:
            return []
        if not self.fts_enabled:
            return await self._search_raw_like(query, limit, namespace)
        clauses, params = _scope_clauses("r.namespace", "r.timestamp", namespace, since, until)
        sql = f"""
            SELECT r.id AS id, -bm25(raw_turns_fts) AS score
            FROM raw_turns_fts JOIN raw_log_index r ON r.rowid = raw_turns_fts.rowid
            WHERE raw_turns_fts MATCH ? AND r.purged = 0 {clauses}
            ORDER BY bm25(raw_turns_fts) LIMIT ?
        """
        params = [match, *params, limit]
        async with self.db.execute(sql, params) as cur:
            return [(row["id"], float(row["score"])) async for row in cur]

    async def _search_raw_like(  # pragma: no cover - only without FTS5
        self, query: str, limit: int, namespace: str | None,
    ) -> list[tuple[str, float]]:
        # Raw-turn text lives only in the FTS index; without FTS5 the raw layer
        # is unavailable (curated memories still fall back to LIKE search).
        return []

    async def memory_ids_for_raw(
        self, raw_ids: list[str], namespace: str | None = None,
    ) -> dict[str, str]:
        """Map raw log ids to the curated memory derived from them (if any).

        Prefers the active (non-cold), most-consolidated memory when several
        share a raw log id.
        """
        if not raw_ids:
            return {}
        ph = ",".join("?" for _ in raw_ids)
        ns_clause = "" if namespace is None else "AND namespace = ?"
        params = [*raw_ids, *([] if namespace is None else [namespace])]
        out: dict[str, tuple[int, int, str]] = {}
        async with self.db.execute(
            f"SELECT id, raw_log_id, tier, compaction_gen FROM memories "
            f"WHERE raw_log_id IN ({ph}) {ns_clause}",
            params,
        ) as cur:
            async for row in cur:
                rank = (0 if row["tier"] == "cold" else 1, row["compaction_gen"] or 0)
                prev = out.get(row["raw_log_id"])
                if prev is None or rank > prev[:2]:
                    out[row["raw_log_id"]] = (*rank, row["id"])
        return {raw: v[2] for raw, v in out.items()}

    # ── Memories ──

    async def save_memory(self, mem: Memory) -> None:
        values = (
            mem.id, mem.created_at, mem.updated_at, mem.content, mem.summary,
            mem.raw_log_id, mem.namespace, mem.session_id, mem.turn,
            mem.valence, mem.arousal, mem.surprise, mem.salience,
            mem.access_count, mem.last_accessed, mem.decay_score,
            mem.compaction_gen, mem.tier, int(mem.fast_pathed), int(mem.is_semantic),
            mem.graph_node_id, mem.vector_id,
            mem.spatial_embedding, mem.scene_description, mem.role, mem.event_time,
        )
        cols = [c.strip() for c in _MEMORY_COLUMNS.split(",")]
        updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c != "id")
        # Upsert (not INSERT OR REPLACE) so the rowid — the FTS key — is stable.
        await self.db.execute(
            f"INSERT INTO memories ({_MEMORY_COLUMNS}) "
            f"VALUES ({','.join('?' for _ in cols)}) "
            f"ON CONFLICT(id) DO UPDATE SET {updates}",
            values,
        )
        # Keywords: keep the original (base) weight of keywords that already
        # existed, so graph reweighting stays idempotent across re-saves.
        async with self.db.execute(
            "SELECT keyword, base_weight FROM memory_keywords WHERE memory_id = ?", (mem.id,)
        ) as cur:
            bases = {r["keyword"]: r["base_weight"] async for r in cur}
        await self.db.execute("DELETE FROM memory_keywords WHERE memory_id = ?", (mem.id,))
        for kw, weight in dict(mem.keywords).items():
            base = bases.get(kw)
            await self.db.execute(
                "INSERT INTO memory_keywords (memory_id, keyword, weight, base_weight) "
                "VALUES (?, ?, ?, ?)",
                (mem.id, kw, weight, base if base is not None else weight),
            )
        await self.db.execute("DELETE FROM memory_entities WHERE memory_id = ?", (mem.id,))
        for name, etype in mem.entities:
            key = entity_key(name)
            if key:
                await self.db.execute(
                    "INSERT OR IGNORE INTO memory_entities "
                    "(memory_id, namespace, entity_key, name, type) VALUES (?, ?, ?, ?, ?)",
                    (mem.id, mem.namespace, key, name, etype),
                )
        await self._index_memory_fts(mem.id)
        await self.db.commit()

    async def _index_memory_fts(self, memory_id: str) -> None:
        if not self.fts_enabled:
            return
        async with self.db.execute(
            "SELECT rowid, content, summary FROM memories WHERE id = ?", (memory_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return
        async with self.db.execute(
            "SELECT keyword FROM memory_keywords WHERE memory_id = ?", (memory_id,)
        ) as cur:
            kws = " ".join([r["keyword"].replace("_", " ") async for r in cur])
        async with self.db.execute(
            "SELECT name FROM memory_entities WHERE memory_id = ?", (memory_id,)
        ) as cur:
            ents = " ".join([r["name"] async for r in cur])
        await self.db.execute("DELETE FROM memories_fts WHERE rowid = ?", (row["rowid"],))
        await self.db.execute(
            "INSERT INTO memories_fts(rowid, content, summary, keywords, entities) "
            "VALUES (?, ?, ?, ?, ?)",
            (row["rowid"], row["content"], row["summary"] or "", kws, ents),
        )

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
        await self._attach_keywords([mem])
        return mem

    async def get_memories(
        self, memory_ids: list[str], namespace: str | None = None,
    ) -> dict[str, Memory]:
        """Batch-load memories (with keywords/entities), optionally namespace-scoped."""
        if not memory_ids:
            return {}
        ph = ",".join("?" for _ in memory_ids)
        ns_clause = "" if namespace is None else "AND namespace = ?"
        params = [*memory_ids, *([] if namespace is None else [namespace])]
        out: dict[str, Memory] = {}
        async with self.db.execute(
            f"SELECT * FROM memories WHERE id IN ({ph}) {ns_clause}", params,
        ) as cur:
            async for row in cur:
                out[row["id"]] = _row_to_memory(dict(row))
        await self._attach_keywords(list(out.values()))
        return out

    async def _attach_keywords(self, memories: list[Memory]) -> None:
        if not memories:
            return
        ids = [m.id for m in memories]
        ph = ",".join("?" for _ in ids)
        kw_map: dict[str, list[tuple[str, float]]] = {mid: [] for mid in ids}
        async with self.db.execute(
            f"SELECT memory_id, keyword, weight FROM memory_keywords WHERE memory_id IN ({ph})",
            ids,
        ) as cur:
            async for row in cur:
                kw_map[row["memory_id"]].append((row["keyword"], row["weight"]))
        ent_map: dict[str, list[tuple[str, str]]] = {mid: [] for mid in ids}
        async with self.db.execute(
            f"SELECT memory_id, name, type FROM memory_entities WHERE memory_id IN ({ph})",
            ids,
        ) as cur:
            async for row in cur:
                ent_map[row["memory_id"]].append((row["name"], row["type"] or ""))
        for m in memories:
            m.keywords = kw_map.get(m.id, [])
            m.entities = ent_map.get(m.id, [])

    async def list_memories(
        self, tier: str | None = None, limit: int = 100, offset: int = 0,
        namespace: str | None = None,
    ) -> list[Memory]:
        clauses, params = [], []
        if tier:
            clauses.append("tier = ?")
            params.append(tier)
        if namespace is not None:
            clauses.append("namespace = ?")
            params.append(namespace)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM memories {where} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        memories = []
        async with self.db.execute(sql, (*params, limit, offset)) as cur:
            async for row in cur:
                memories.append(_row_to_memory(dict(row)))
        await self._attach_keywords(memories)
        return memories

    async def list_namespaces(self) -> list[str]:
        async with self.db.execute("SELECT DISTINCT namespace FROM memories") as cur:
            return [r["namespace"] async for r in cur]

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

    async def record_accesses(
        self, accesses: list[tuple[str, str, str | None, str | None]], now: str,
    ) -> None:
        """Record a batch of accesses in one transaction.

        Each item is (memory_id, access_type, session_id, query). Access resets
        recency, so the cached decay score becomes 1.0.
        """
        import uuid

        for memory_id, access_type, session_id, query in accesses:
            await self.db.execute(
                "INSERT INTO memory_access_log (id, memory_id, accessed_at, access_type, session_id, query) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), memory_id, now, access_type, session_id, query),
            )
            await self.db.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed = ?, "
                "decay_score = 1.0, updated_at = ? WHERE id = ?",
                (now, now, memory_id),
            )
        await self.db.commit()

    async def refresh_decay_scores(self, now: datetime | None = None, config: dict | None = None) -> int:
        """Recompute the cached ``decay_score`` of every memory for ``now``."""
        from agent_memory.core.decay import decay_for_memory

        now = now or datetime.now(timezone.utc)
        updates = []
        async with self.db.execute(
            "SELECT id, last_accessed, event_time, created_at, access_count, arousal, "
            "surprise, is_semantic FROM memories"
        ) as cur:
            async for row in cur:
                mem = _row_to_memory({**dict(row), "content": "", "raw_log_id": "",
                                      "session_id": "", "turn": 0, "updated_at": ""})
                updates.append((decay_for_memory(mem, now=now, config=config), row["id"]))
        await self.db.executemany("UPDATE memories SET decay_score = ? WHERE id = ?", updates)
        await self.db.commit()
        return len(updates)

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

    # ── Deletion ──

    async def delete_memory(self, memory_id: str, namespace: str | None = None) -> Memory | None:
        """Delete a curated memory and its index rows. Returns what was deleted
        (so callers can remove graph/vector entries), or None if absent."""
        mem = await self.get_memory(memory_id, namespace=namespace)
        if mem is None:
            return None
        if self.fts_enabled:
            await self.db.execute(
                "DELETE FROM memories_fts WHERE rowid = (SELECT rowid FROM memories WHERE id = ?)",
                (memory_id,),
            )
        for sql in (
            "DELETE FROM memory_keywords WHERE memory_id = ?",
            "DELETE FROM memory_entities WHERE memory_id = ?",
            "DELETE FROM memory_access_log WHERE memory_id = ?",
            "DELETE FROM memories WHERE id = ?",
        ):
            await self.db.execute(sql, (memory_id,))
        await self.db.commit()
        return mem

    async def purge_raw(self, raw_log_id: str) -> bool:
        """Remove a raw turn from the retrievable index. The JSONL line is kept
        (the log is append-only) but the turn is never surfaced again."""
        async with self.db.execute(
            "SELECT rowid FROM raw_log_index WHERE id = ?", (raw_log_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return False
        if self.fts_enabled:
            await self.db.execute("DELETE FROM raw_turns_fts WHERE rowid = ?", (row["rowid"],))
        await self.db.execute("UPDATE raw_log_index SET purged = 1 WHERE id = ?", (raw_log_id,))
        await self.db.commit()
        return True

    async def purge_namespace(self, namespace: str) -> dict:
        """Delete every memory, index row and decision log of a tenant.

        Returns the deleted memory ids, vector ids and log files so the caller
        can clean the graph, vector store and JSONL logs.
        """
        async with self.db.execute(
            "SELECT id, vector_id FROM memories WHERE namespace = ?", (namespace,)
        ) as cur:
            mems = [dict(r) async for r in cur]
        for m in mems:
            await self.delete_memory(m["id"])
        async with self.db.execute(
            "SELECT rowid, id, file_path FROM raw_log_index WHERE namespace = ?", (namespace,)
        ) as cur:
            raws = [dict(r) async for r in cur]
        if self.fts_enabled:
            await self.db.executemany(
                "DELETE FROM raw_turns_fts WHERE rowid = ?", [(r["rowid"],) for r in raws]
            )
        await self.db.execute("DELETE FROM raw_log_index WHERE namespace = ?", (namespace,))
        async with self.db.execute(
            "SELECT id FROM retrieval_decisions WHERE namespace = ?", (namespace,)
        ) as cur:
            rids = [r["id"] async for r in cur]
        await self.db.executemany(
            "DELETE FROM memory_usage WHERE retrieval_id = ?", [(r,) for r in rids]
        )
        for sql in (
            "DELETE FROM retrieval_hits WHERE namespace = ?",
            "DELETE FROM retrieval_decisions WHERE namespace = ?",
            "DELETE FROM save_decisions WHERE namespace = ?",
        ):
            await self.db.execute(sql, (namespace,))
        await self.db.commit()
        return {
            "memory_ids": [m["id"] for m in mems],
            "vector_ids": [m["vector_id"] for m in mems if m["vector_id"]],
            "raw_log_ids": [r["id"] for r in raws],
            "files": sorted({r["file_path"] for r in raws}),
        }

    # ── Keyword / full-text search ──

    async def search_memories_fts(
        self, query: str, limit: int = 30, namespace: str | None = None,
        since: str | None = None, until: str | None = None,
    ) -> list[tuple[str, float]]:
        """BM25 search over curated memories (content, summary, keywords,
        entities). Returns (memory_id, score), best first."""
        match = fts_query(query)
        if not match:
            return []
        if not self.fts_enabled:
            terms = content_terms(query)
            mems = await self.search_by_content(terms, limit=limit, namespace=namespace)
            return [(m.id, 1.0 / (i + 1)) for i, m in enumerate(mems)]
        clauses, params = _scope_clauses(
            "m.namespace", "COALESCE(m.event_time, m.created_at)", namespace, since, until,
        )
        sql = f"""
            SELECT m.id AS id, -bm25(memories_fts, 1.0, 0.5, 2.0, 2.0) AS score
            FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid
            WHERE memories_fts MATCH ? {clauses}
            ORDER BY bm25(memories_fts, 1.0, 0.5, 2.0, 2.0) LIMIT ?
        """
        params = [match, *params, limit]
        async with self.db.execute(sql, params) as cur:
            return [(row["id"], float(row["score"])) async for row in cur]

    async def search_entities(
        self, query: str, limit: int = 30, namespace: str | None = None,
        since: str | None = None, until: str | None = None,
    ) -> list[tuple[str, float]]:
        """Memories naming an entity mentioned in ``query`` (unigram/bigram match)."""
        toks = entity_key(query).split()
        keys = set(toks) | {f"{a} {b}" for a, b in zip(toks, toks[1:])}
        keys = {k for k in keys if (len(k) > 2 or not k.isascii()) and k not in {"the", "and", "you"}}
        if not keys:
            return []
        ph = ",".join("?" for _ in keys)
        clauses, scope = _scope_clauses(
            "e.namespace", "COALESCE(m.event_time, m.created_at)", namespace, since, until,
        )
        params = [*keys, *scope, limit]
        async with self.db.execute(
            f"SELECT e.memory_id AS memory_id, COUNT(*) AS n FROM memory_entities e "
            f"JOIN memories m ON m.id = e.memory_id "
            f"WHERE e.entity_key IN ({ph}) {clauses} "
            f"GROUP BY e.memory_id ORDER BY n DESC LIMIT ?",
            params,
        ) as cur:
            return [(row["memory_id"], float(row["n"])) async for row in cur]

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
                d = dict(row)
                d.pop("match_score", None)
                memories.append(_row_to_memory(d))
        await self._attach_keywords(memories)
        return memories

    async def search_by_content(
        self, terms: list[str], limit: int = 10, namespace: str | None = None,
    ) -> list[Memory]:
        """Substring search over memory content for the given terms (OR).

        Fallback for SQLite builds without FTS5. Results are ranked by the
        number of distinct terms matched, weighted by decay score.
        """
        terms = [t for t in (t.strip() for t in terms) if t]
        if not terms:
            return []
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
        await self._attach_keywords(memories)
        return memories

    async def most_similar_text(
        self, text: str, namespace: str, exclude_raw_id: str | None = None,
    ) -> str | None:
        """Content of the curated memory most lexically similar to ``text``
        (used for the decision model's novelty feature)."""
        hits = await self.search_memories_fts(text, limit=3, namespace=namespace)
        for mid, _score in hits:
            mem = await self.get_memory(mid)
            if mem and mem.raw_log_id != exclude_raw_id:
                return mem.content
        return None

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
            SELECT mk.keyword, mk.memory_id, mk.weight,
                   COALESCE(mk.base_weight, mk.weight) AS base_weight, m.namespace
            FROM memory_keywords mk
            JOIN memories m ON mk.memory_id = m.id
            WHERE m.tier IN ({placeholders})
        """
        async with self.db.execute(sql, tiers_to_use) as cur:
            return [dict(row) async for row in cur]

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
        """Return recent retrieval queries for a session."""
        async with self.db.execute(
            "SELECT query FROM retrieval_decisions WHERE session_id = ? "
            "ORDER BY decided_at DESC LIMIT ?",
            (session_id, limit),
        ) as cur:
            return [row["query"] async for row in cur]

    async def get_failed_retrieval_keywords(
        self, session_id: str, lookback: int = 20, namespace: str | None = None,
    ) -> list[str]:
        """Keywords from recent queries in this session that found nothing
        relevant (A2.1).

        A retrieval is a miss when no lexical layer matched and no vector hit
        cleared the similarity floor (recorded at query time), or — for rows
        logged before that was recorded — when it returned nothing.
        """
        ns_clause = "" if namespace is None else "AND namespace = ?"
        params = [session_id, *([] if namespace is None else [namespace]), lookback]
        async with self.db.execute(
            f"SELECT query, miss, return_count FROM retrieval_decisions "
            f"WHERE session_id = ? {ns_clause} ORDER BY decided_at DESC LIMIT ?",
            params,
        ) as cur:
            rows = [dict(r) async for r in cur]
        gap: set[str] = set()
        for r in rows:
            missed = r["miss"] == 1 or (r["miss"] is None and not r["return_count"])
            if missed:
                gap.update(content_terms(r["query"]))
        return sorted(gap)

    # ── Save decisions ──

    async def log_save_decision(self, dec: SaveDecision) -> None:
        await self.db.execute(
            "INSERT INTO save_decisions "
            "(id, raw_log_id, session_id, turn, decided_at, decision, reason, confidence, "
            "gap_triggered, threshold_used, namespace, role, policy_prob, policy_source, features) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (dec.id, dec.raw_log_id, dec.session_id, dec.turn, dec.decided_at,
             dec.decision, dec.reason, dec.confidence, int(dec.gap_triggered), dec.threshold_used,
             dec.namespace, dec.role, dec.policy_prob, dec.policy_source,
             json.dumps(dec.features) if dec.features is not None else None),
        )
        await self.db.commit()

    async def update_save_outcome(self, decision_id: str, useful: bool, assessed_at: str) -> None:
        """Mark whether a saved memory turned out to be useful (A4)."""
        await self.db.execute(
            "UPDATE save_decisions SET outcome_useful = ?, outcome_assessed_at = ? WHERE id = ?",
            (int(useful), assessed_at, decision_id),
        )
        await self.db.commit()

    async def batch_update_save_outcomes(
        self, updates: list[tuple[bool, str, str]],
    ) -> None:
        """Each tuple is (useful, assessed_at, decision_id)."""
        await self.db.executemany(
            "UPDATE save_decisions SET outcome_useful = ?, outcome_assessed_at = ? WHERE id = ?",
            [(int(u), at, did) for u, at, did in updates],
        )
        await self.db.commit()

    async def get_unassessed_save_decisions(self, lookback_days: int = 30) -> list[dict]:
        """Saved decisions older than ``lookback_days`` with no outcome yet (legacy helper)."""
        async with self.db.execute(
            """SELECT sd.id, sd.raw_log_id, m.id as memory_id, m.access_count
               FROM save_decisions sd
               LEFT JOIN memories m ON m.raw_log_id = sd.raw_log_id
               WHERE sd.decision IN ('save', 'fast_path')
                 AND sd.outcome_useful IS NULL
                 AND sd.decided_at < datetime('now', ?)""",
            (f"-{lookback_days} days",),
        ) as cur:
            return [dict(row) async for row in cur]

    async def get_unlabelled_save_decisions(self) -> list[dict]:
        """All save/skip decisions without an outcome label, oldest first."""
        async with self.db.execute(
            "SELECT id, raw_log_id, namespace, decided_at, decision FROM save_decisions "
            "WHERE outcome_useful IS NULL ORDER BY decided_at"
        ) as cur:
            return [dict(r) async for r in cur]

    async def decision_times(self, namespace: str) -> dict[str, str]:
        """raw_log_id -> when its save decision was made (i.e. when it was logged)."""
        async with self.db.execute(
            "SELECT raw_log_id, decided_at FROM save_decisions WHERE namespace = ?", (namespace,)
        ) as cur:
            return {r["raw_log_id"]: r["decided_at"] async for r in cur}

    async def recent_save_rate(self, session_id: str, namespace: str, limit: int = 20) -> float:
        async with self.db.execute(
            "SELECT decision FROM save_decisions WHERE session_id = ? AND namespace = ? "
            "ORDER BY decided_at DESC LIMIT ?",
            (session_id, namespace, limit),
        ) as cur:
            rows = [r["decision"] async for r in cur]
        if not rows:
            return 0.0
        return sum(d in ("save", "fast_path") for d in rows) / len(rows)

    # ── Retrieval decisions (A4) ──

    async def log_retrieval_decision(
        self, decision_id: str, session_id: str, turn: int | None,
        query: str, decided_at: str, layers_queried: list[str],
        graph_depth: int, mood_weight: float, top_k: int,
        memory_ids: list[str], return_count: int,
        namespace: str = "default", top_score: float | None = None,
        lexical_hit: bool | None = None, miss: bool | None = None,
        hits: list[dict] | None = None,
    ) -> None:
        await self.db.execute(
            "INSERT INTO retrieval_decisions "
            "(id, session_id, turn, query, decided_at, layers_queried, graph_depth, "
            "mood_weight, top_k, memories_returned, return_count, namespace, top_score, "
            "lexical_hit, miss) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (decision_id, session_id or "", turn, query, decided_at,
             json.dumps(layers_queried), graph_depth, mood_weight, top_k,
             json.dumps(memory_ids), return_count, namespace, top_score,
             None if lexical_hit is None else int(lexical_hit),
             None if miss is None else int(miss)),
        )
        for h in hits or []:
            await self.db.execute(
                "INSERT INTO retrieval_hits "
                "(retrieval_id, rank, memory_id, raw_log_id, score, layers, namespace, retrieved_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (decision_id, h["rank"], h.get("memory_id"), h.get("raw_log_id"),
                 h.get("score"), json.dumps(h.get("layers", [])), namespace, decided_at),
            )
        await self.db.commit()

    async def get_retrieval_hits(self, retrieval_id: str) -> list[dict]:
        async with self.db.execute(
            "SELECT h.*, rd.namespace AS rd_namespace FROM retrieval_hits h "
            "JOIN retrieval_decisions rd ON rd.id = h.retrieval_id "
            "WHERE h.retrieval_id = ? ORDER BY h.rank",
            (retrieval_id,),
        ) as cur:
            return [dict(r) async for r in cur]

    async def get_retrieval_queries(
        self, namespace: str | None = None, since: str | None = None,
    ) -> list[dict]:
        clauses, params = [], []
        if namespace is not None:
            clauses.append("namespace = ?")
            params.append(namespace)
        if since is not None:
            clauses.append("decided_at >= ?")
            params.append(since)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        async with self.db.execute(
            f"SELECT id, namespace, query, decided_at FROM retrieval_decisions {where} "
            f"ORDER BY decided_at",
            params,
        ) as cur:
            return [dict(r) async for r in cur]

    async def log_usage(
        self, retrieval_id: str, rows: list[dict], recorded_at: str,
    ) -> None:
        """Rows: {memory_id, raw_log_id, used, overlap, source}."""
        for r in rows:
            await self.db.execute(
                "INSERT INTO memory_usage (retrieval_id, memory_id, raw_log_id, used, overlap, "
                "source, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (retrieval_id, r.get("memory_id"), r.get("raw_log_id"), int(r["used"]),
                 r.get("overlap"), r.get("source"), recorded_at),
            )
        await self.db.commit()

    async def get_used_raw_log_ids(self, namespace: str | None = None) -> set[str]:
        """Raw log ids whose turn (directly, via its memory, or via a merged
        memory it was consolidated into) was attributed as used."""
        ns_clause = "" if namespace is None else "AND rd.namespace = ?"
        params = [] if namespace is None else [namespace]
        used_raw: set[str] = set()
        used_mem: set[str] = set()
        async with self.db.execute(
            f"SELECT u.memory_id, u.raw_log_id FROM memory_usage u "
            f"JOIN retrieval_decisions rd ON rd.id = u.retrieval_id "
            f"WHERE u.used = 1 {ns_clause}",
            params,
        ) as cur:
            async for r in cur:
                if r["raw_log_id"]:
                    used_raw.add(r["raw_log_id"])
                if r["memory_id"]:
                    used_mem.add(r["memory_id"])
        if used_mem:
            sources = await self.merge_sources(list(used_mem))
            all_mem = used_mem | sources
            mems = await self.get_memories(list(all_mem))
            used_raw.update(m.raw_log_id for m in mems.values())
        return used_raw

    async def merge_sources(self, memory_ids: list[str]) -> set[str]:
        """All memories that were (transitively) consolidated into ``memory_ids``."""
        out: set[str] = set()
        frontier = set(memory_ids)
        while frontier:
            ph = ",".join("?" for _ in frontier)
            nxt: set[str] = set()
            async with self.db.execute(
                f"SELECT source_memory_ids FROM compaction_merges WHERE resulting_memory_id IN ({ph})",
                list(frontier),
            ) as cur:
                async for r in cur:
                    for sid in json.loads(r["source_memory_ids"]):
                        if sid not in out:
                            nxt.add(sid)
            out |= nxt
            frontier = nxt
        return out

    async def get_unassessed_retrieval_decisions(self, min_age_minutes: int = 60) -> list[dict]:
        """Get retrieval decisions not yet assessed (A4)."""
        async with self.db.execute(
            """SELECT id, session_id, turn, query, decided_at, namespace, return_count
               FROM retrieval_decisions
               WHERE outcome_helpful IS NULL
                 AND decided_at < ?""",
            ((datetime.now(timezone.utc) - _minutes(min_age_minutes)).isoformat(),),
        ) as cur:
            return [dict(row) async for row in cur]

    async def update_retrieval_outcome(self, decision_id: str, helpful: bool, assessed_at: str) -> None:
        await self.db.execute(
            "UPDATE retrieval_decisions SET outcome_helpful = ?, outcome_assessed_at = ? WHERE id = ?",
            (int(helpful), assessed_at, decision_id),
        )
        await self.db.commit()

    async def get_retrieval_followups(
        self, session_id: str, turn: int, window: int = 3, namespace: str | None = None,
    ) -> list[str]:
        """Get follow-up queries within a turn window (A4 outcome assessment)."""
        ns_clause = "" if namespace is None else "AND namespace = ?"
        async with self.db.execute(
            "SELECT query FROM retrieval_decisions WHERE session_id = ? AND turn > ? AND turn <= ? "
            f"{ns_clause}",
            (session_id, turn, turn + window, *([] if namespace is None else [namespace])),
        ) as cur:
            return [row["query"] async for row in cur]

    async def get_retrieval_followups_by_time(
        self, session_id: str, decided_at: str, minutes: int, exclude_id: str,
        namespace: str | None = None,
    ) -> list[str]:
        """Follow-up queries in the same session (and namespace) within
        ``minutes`` after a retrieval."""
        start = decided_at
        from agent_memory.core.decay import parse_time

        t0 = parse_time(decided_at)
        end = (t0 + _minutes(minutes)).isoformat() if t0 else decided_at
        ns_clause = "" if namespace is None else "AND namespace = ?"
        async with self.db.execute(
            "SELECT query FROM retrieval_decisions WHERE session_id = ? AND id != ? "
            f"AND decided_at > ? AND decided_at <= ? {ns_clause}",
            (session_id, exclude_id, start, end, *([] if namespace is None else [namespace])),
        ) as cur:
            return [row["query"] async for row in cur]

    async def get_usage_for_retrieval(self, retrieval_id: str) -> list[int]:
        async with self.db.execute(
            "SELECT used FROM memory_usage WHERE retrieval_id = ?", (retrieval_id,)
        ) as cur:
            return [r["used"] async for r in cur]

    # ── Policy training audit ──

    async def log_policy_training_run(
        self, run_id: str, trained_at: str, model_type: str, n_train: int,
        n_validation: int, metrics: dict, promoted: bool, model_path: str | None,
        notes: str | None = None,
    ) -> None:
        await self.db.execute(
            "INSERT INTO policy_training_runs (id, trained_at, model_type, n_train, "
            "n_validation, metrics, promoted, model_path, notes) VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, trained_at, model_type, n_train, n_validation, json.dumps(metrics),
             int(promoted), model_path, notes),
        )
        await self.db.commit()

    async def last_policy_training_run(self) -> dict | None:
        async with self.db.execute(
            "SELECT * FROM policy_training_runs ORDER BY trained_at DESC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def count_labelled_save_decisions(self) -> int:
        async with self.db.execute(
            "SELECT COUNT(*) AS n FROM save_decisions "
            "WHERE outcome_useful IS NOT NULL AND features IS NOT NULL"
        ) as cur:
            return (await cur.fetchone())["n"]

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
        """Return hot memories whose compaction score exceeds ``threshold``.

        Uses the cached ``decay_score``; the compaction engine refreshes it for
        the current time (``refresh_decay_scores``) before calling this.
        """
        sql = """
            SELECT m.* FROM memories m
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
        await self._attach_keywords(candidates)
        return candidates

    async def list_active_memories(self, namespace: str) -> list[Memory]:
        """Hot and warm memories of one namespace (for deduplication)."""
        memories = []
        async with self.db.execute(
            "SELECT * FROM memories WHERE namespace = ? AND tier IN ('hot', 'warm') "
            "ORDER BY created_at",
            (namespace,),
        ) as cur:
            async for row in cur:
                memories.append(_row_to_memory(dict(row)))
        await self._attach_keywords(memories)
        return memories

    # ── Policy data export (A4.4) ──

    async def export_save_policy_data(self) -> list[dict]:
        """Export assessed save decisions (with logged features) for training."""
        rows = []
        async with self.db.execute(
            """SELECT sd.id, sd.decided_at, sd.namespace, sd.confidence, sd.decision,
                      sd.gap_triggered, sd.policy_prob, sd.policy_source, sd.features,
                      m.valence, m.arousal, m.surprise, m.salience,
                      sd.outcome_useful
               FROM save_decisions sd
               LEFT JOIN memories m ON m.raw_log_id = sd.raw_log_id AND m.compaction_gen = 0
               WHERE sd.outcome_useful IS NOT NULL
               ORDER BY sd.decided_at"""
        ) as cur:
            async for row in cur:
                d = dict(row)
                d["features"] = json.loads(d["features"]) if d["features"] else None
                rows.append(d)
        return rows

    async def export_retrieval_policy_data(self) -> list[dict]:
        """Export assessed retrieval decisions for policy training."""
        rows = []
        async with self.db.execute(
            """SELECT layers_queried, graph_depth, mood_weight, top_k,
                      return_count, top_score, lexical_hit, outcome_helpful
               FROM retrieval_decisions
               WHERE outcome_helpful IS NOT NULL"""
        ) as cur:
            async for row in cur:
                rows.append(dict(row))
        return rows

    async def get_memories_with_vectors(
        self, tiers: list[str] | None = None, namespace: str | None = None,
    ) -> list[dict]:
        """Get memories that have vector embeddings (for dream explorer)."""
        clauses, params = ["vector_id IS NOT NULL"], []
        if tiers:
            clauses.append(f"tier IN ({','.join('?' for _ in tiers)})")
            params.extend(tiers)
        if namespace is not None:
            clauses.append("namespace = ?")
            params.append(namespace)
        sql = (
            "SELECT id, session_id, vector_id, namespace FROM memories "
            f"WHERE {' AND '.join(clauses)}"
        )
        async with self.db.execute(sql, params) as cur:
            return [dict(row) async for row in cur]


def _scope_clauses(
    ns_col: str, time_col: str, namespace: str | None, since: str | None, until: str | None,
) -> tuple[str, list[object]]:
    """SQL ``AND`` clauses (and params) for an optional namespace and time window."""
    clauses, params = [], []
    if namespace is not None:
        clauses.append(f"{ns_col} = ?")
        params.append(namespace)
    if since is not None:
        clauses.append(f"{time_col} >= ?")
        params.append(since)
    if until is not None:
        clauses.append(f"{time_col} <= ?")
        params.append(until)
    return "".join(f" AND {c}" for c in clauses), params


def _minutes(n: int):
    from datetime import timedelta

    return timedelta(minutes=n)


def _row_to_memory(row: dict) -> Memory:
    """Convert a SQLite row dict to a Memory dataclass."""
    return Memory(
        id=row["id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        content=row["content"],
        summary=row.get("summary"),
        raw_log_id=row["raw_log_id"],
        namespace=row.get("namespace") or "default",
        session_id=row["session_id"],
        turn=row["turn"],
        valence=row.get("valence") or 0.0,
        arousal=row.get("arousal") or 0.0,
        surprise=row.get("surprise") or 0.0,
        salience=row.get("salience") if row.get("salience") is not None else 0.5,
        access_count=row.get("access_count") or 0,
        last_accessed=row.get("last_accessed"),
        decay_score=row.get("decay_score") if row.get("decay_score") is not None else 1.0,
        compaction_gen=row.get("compaction_gen") or 0,
        tier=row.get("tier") or "hot",
        fast_pathed=bool(row.get("fast_pathed", 0)),
        is_semantic=bool(row.get("is_semantic", 0)),
        graph_node_id=row.get("graph_node_id"),
        vector_id=row.get("vector_id"),
        spatial_embedding=row.get("spatial_embedding"),
        scene_description=row.get("scene_description"),
        role=row.get("role") or "assistant",
        event_time=row.get("event_time"),
    )


__all__ = ["SQLiteStore", "entity_key"]
