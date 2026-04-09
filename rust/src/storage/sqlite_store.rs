//! SQLite storage for memory metadata, access tracking, and compaction history.

use std::collections::HashMap;
use std::path::PathBuf;

use sqlx::sqlite::{SqlitePool, SqlitePoolOptions, SqliteRow};
use sqlx::Row;

use crate::config::default_db_path;
use crate::models::{CompactionResult, Memory, SaveDecision};

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS raw_log_index (
    id          TEXT PRIMARY KEY,
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
"#;

/// Async SQLite store for memory metadata.
pub struct SQLiteStore {
    db_path: PathBuf,
    pool: Option<SqlitePool>,
}

impl SQLiteStore {
    pub fn new(db_path: Option<PathBuf>) -> Self {
        Self {
            db_path: db_path.unwrap_or_else(default_db_path),
            pool: None,
        }
    }

    pub async fn initialize(&mut self) -> Result<(), sqlx::Error> {
        if let Some(parent) = self.db_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let url = format!("sqlite:{}?mode=rwc", self.db_path.display());
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(&url)
            .await?;
        // Execute schema statements one by one
        for statement in SCHEMA.split(';') {
            let stmt = statement.trim();
            if !stmt.is_empty() {
                sqlx::query(stmt).execute(&pool).await?;
            }
        }
        self.pool = Some(pool);
        Ok(())
    }

    pub async fn close(&mut self) {
        if let Some(pool) = self.pool.take() {
            pool.close().await;
        }
    }

    fn pool(&self) -> &SqlitePool {
        self.pool
            .as_ref()
            .expect("SQLiteStore not initialized — call initialize() first")
    }

    // ── Raw log index ──

    pub async fn index_raw_log(
        &self,
        entry_id: &str,
        session_id: &str,
        turn: i64,
        timestamp: &str,
        file_path: &str,
        byte_offset: i64,
    ) -> Result<(), sqlx::Error> {
        sqlx::query(
            "INSERT OR IGNORE INTO raw_log_index (id, session_id, turn, timestamp, file_path, byte_offset) \
             VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(entry_id)
        .bind(session_id)
        .bind(turn)
        .bind(timestamp)
        .bind(file_path)
        .bind(byte_offset)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    pub async fn get_raw_log_ref(&self, entry_id: &str) -> Result<Option<HashMap<String, serde_json::Value>>, sqlx::Error> {
        let row: Option<SqliteRow> = sqlx::query("SELECT * FROM raw_log_index WHERE id = ?")
            .bind(entry_id)
            .fetch_optional(self.pool())
            .await?;
        Ok(row.map(|r| sqlite_row_to_map(&r)))
    }

    // ── Memories ──

    pub async fn save_memory(&self, mem: &Memory) -> Result<(), sqlx::Error> {
        sqlx::query(
            "INSERT OR REPLACE INTO memories \
             (id, created_at, updated_at, content, summary, raw_log_id, session_id, turn, \
              valence, arousal, surprise, salience, access_count, last_accessed, decay_score, \
              compaction_gen, tier, fast_pathed, is_semantic, graph_node_id, vector_id, \
              spatial_embedding, scene_description) \
             VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        )
        .bind(&mem.id)
        .bind(&mem.created_at)
        .bind(&mem.updated_at)
        .bind(&mem.content)
        .bind(&mem.summary)
        .bind(&mem.raw_log_id)
        .bind(&mem.session_id)
        .bind(mem.turn)
        .bind(mem.valence)
        .bind(mem.arousal)
        .bind(mem.surprise)
        .bind(mem.salience)
        .bind(mem.access_count)
        .bind(&mem.last_accessed)
        .bind(mem.decay_score)
        .bind(mem.compaction_gen)
        .bind(&mem.tier)
        .bind(mem.fast_pathed as i32)
        .bind(mem.is_semantic as i32)
        .bind(&mem.graph_node_id)
        .bind(&mem.vector_id)
        .bind(&mem.spatial_embedding)
        .bind(&mem.scene_description)
        .execute(self.pool())
        .await?;

        // Save keywords
        sqlx::query("DELETE FROM memory_keywords WHERE memory_id = ?")
            .bind(&mem.id)
            .execute(self.pool())
            .await?;

        for (kw, weight) in &mem.keywords {
            sqlx::query(
                "INSERT INTO memory_keywords (memory_id, keyword, weight) VALUES (?, ?, ?)",
            )
            .bind(&mem.id)
            .bind(kw)
            .bind(weight)
            .execute(self.pool())
            .await?;
        }
        Ok(())
    }

    pub async fn get_memory(&self, memory_id: &str) -> Result<Option<Memory>, sqlx::Error> {
        let row: Option<SqliteRow> = sqlx::query("SELECT * FROM memories WHERE id = ?")
            .bind(memory_id)
            .fetch_optional(self.pool())
            .await?;

        let Some(row) = row else {
            return Ok(None);
        };
        let mut mem = row_to_memory(&row);

        // Load keywords
        let kw_rows: Vec<SqliteRow> =
            sqlx::query("SELECT keyword, weight FROM memory_keywords WHERE memory_id = ?")
                .bind(memory_id)
                .fetch_all(self.pool())
                .await?;

        mem.keywords = kw_rows
            .iter()
            .map(|r| {
                let kw: String = r.get("keyword");
                let w: f64 = r.get("weight");
                (kw, w)
            })
            .collect();

        Ok(Some(mem))
    }

    pub async fn list_memories(
        &self,
        tier: Option<&str>,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Memory>, sqlx::Error> {
        let rows: Vec<SqliteRow> = if let Some(tier) = tier {
            sqlx::query(
                "SELECT * FROM memories WHERE tier = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            )
            .bind(tier)
            .bind(limit)
            .bind(offset)
            .fetch_all(self.pool())
            .await?
        } else {
            sqlx::query("SELECT * FROM memories ORDER BY created_at DESC LIMIT ? OFFSET ?")
                .bind(limit)
                .bind(offset)
                .fetch_all(self.pool())
                .await?
        };

        let mut memories: Vec<Memory> = rows.iter().map(row_to_memory).collect();

        // Batch-load keywords
        if !memories.is_empty() {
            let ids: Vec<&str> = memories.iter().map(|m| m.id.as_str()).collect();
            let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!(
                "SELECT memory_id, keyword, weight FROM memory_keywords WHERE memory_id IN ({placeholders})"
            );
            let mut query = sqlx::query(&sql);
            for id in &ids {
                query = query.bind(id);
            }
            let kw_rows: Vec<SqliteRow> = query.fetch_all(self.pool()).await?;

            let mut kw_map: HashMap<String, Vec<(String, f64)>> = HashMap::new();
            for r in &kw_rows {
                let mid: String = r.get("memory_id");
                let kw: String = r.get("keyword");
                let w: f64 = r.get("weight");
                kw_map.entry(mid).or_default().push((kw, w));
            }
            for m in &mut memories {
                if let Some(kws) = kw_map.remove(&m.id) {
                    m.keywords = kws;
                }
            }
        }

        Ok(memories)
    }

    /// Find the memory ID associated with a raw log entry.
    pub async fn find_memory_id_by_raw_log_id(
        &self,
        raw_log_id: &str,
    ) -> Result<Option<String>, sqlx::Error> {
        let row: Option<SqliteRow> =
            sqlx::query("SELECT id FROM memories WHERE raw_log_id = ?")
                .bind(raw_log_id)
                .fetch_optional(self.pool())
                .await?;
        Ok(row.map(|r| r.get("id")))
    }

    pub async fn count_memories(&self, tier: Option<&str>) -> Result<i64, sqlx::Error> {
        let row: SqliteRow = if let Some(tier) = tier {
            sqlx::query("SELECT COUNT(*) as cnt FROM memories WHERE tier = ?")
                .bind(tier)
                .fetch_one(self.pool())
                .await?
        } else {
            sqlx::query("SELECT COUNT(*) as cnt FROM memories")
                .fetch_one(self.pool())
                .await?
        };
        Ok(row.get("cnt"))
    }

    pub async fn update_memory_access(
        &self,
        memory_id: &str,
        decay_score: f64,
        access_count: i64,
        last_accessed: &str,
    ) -> Result<(), sqlx::Error> {
        sqlx::query(
            "UPDATE memories SET decay_score = ?, access_count = ?, last_accessed = ?, updated_at = ? WHERE id = ?",
        )
        .bind(decay_score)
        .bind(access_count)
        .bind(last_accessed)
        .bind(last_accessed)
        .bind(memory_id)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    pub async fn update_memory_tier(&self, memory_id: &str, tier: &str) -> Result<(), sqlx::Error> {
        sqlx::query("UPDATE memories SET tier = ? WHERE id = ?")
            .bind(tier)
            .bind(memory_id)
            .execute(self.pool())
            .await?;
        Ok(())
    }

    pub async fn update_memory_graph_ref(
        &self,
        memory_id: &str,
        graph_node_id: &str,
    ) -> Result<(), sqlx::Error> {
        sqlx::query("UPDATE memories SET graph_node_id = ? WHERE id = ?")
            .bind(graph_node_id)
            .bind(memory_id)
            .execute(self.pool())
            .await?;
        Ok(())
    }

    pub async fn update_memory_vector_ref(
        &self,
        memory_id: &str,
        vector_id: &str,
    ) -> Result<(), sqlx::Error> {
        sqlx::query("UPDATE memories SET vector_id = ? WHERE id = ?")
            .bind(vector_id)
            .bind(memory_id)
            .execute(self.pool())
            .await?;
        Ok(())
    }

    pub async fn update_memory_visual(
        &self,
        memory_id: &str,
        scene_description: &str,
        spatial_embedding: &[u8],
    ) -> Result<(), sqlx::Error> {
        sqlx::query(
            "UPDATE memories SET scene_description = ?, spatial_embedding = ? WHERE id = ?",
        )
        .bind(scene_description)
        .bind(spatial_embedding)
        .bind(memory_id)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    // ── Keyword search ──

    pub async fn search_by_keywords(
        &self,
        keywords: &[String],
        limit: i64,
    ) -> Result<Vec<Memory>, sqlx::Error> {
        if keywords.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = keywords.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT m.*, SUM(mk.weight) as match_score \
             FROM memories m \
             JOIN memory_keywords mk ON m.id = mk.memory_id \
             WHERE mk.keyword IN ({placeholders}) \
             GROUP BY m.id \
             ORDER BY match_score * m.decay_score DESC \
             LIMIT ?"
        );
        let mut query = sqlx::query(&sql);
        for kw in keywords {
            query = query.bind(kw);
        }
        query = query.bind(limit);
        let rows: Vec<SqliteRow> = query.fetch_all(self.pool()).await?;

        let mut memories: Vec<Memory> = rows.iter().map(row_to_memory).collect();

        // Load keywords for results
        if !memories.is_empty() {
            let ids: Vec<&str> = memories.iter().map(|m| m.id.as_str()).collect();
            let ph = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let kw_sql = format!(
                "SELECT memory_id, keyword, weight FROM memory_keywords WHERE memory_id IN ({ph})"
            );
            let mut kw_query = sqlx::query(&kw_sql);
            for id in &ids {
                kw_query = kw_query.bind(id);
            }
            let kw_rows: Vec<SqliteRow> = kw_query.fetch_all(self.pool()).await?;

            let mut kw_map: HashMap<String, Vec<(String, f64)>> = HashMap::new();
            for r in &kw_rows {
                let mid: String = r.get("memory_id");
                let kw: String = r.get("keyword");
                let w: f64 = r.get("weight");
                kw_map.entry(mid).or_default().push((kw, w));
            }
            for m in &mut memories {
                if let Some(kws) = kw_map.remove(&m.id) {
                    m.keywords = kws;
                }
            }
        }

        Ok(memories)
    }

    /// Update a single keyword weight (used by keyword reweighting — A2.5).
    pub async fn update_keyword_weight(
        &self,
        memory_id: &str,
        keyword: &str,
        weight: f64,
    ) -> Result<(), sqlx::Error> {
        sqlx::query(
            "UPDATE memory_keywords SET weight = MIN(?, 1.0) WHERE memory_id = ? AND keyword = ?",
        )
        .bind(weight)
        .bind(memory_id)
        .bind(keyword)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    /// Update multiple keyword weights in a single transaction (A2.5).
    ///
    /// Each tuple is (new_weight, memory_id, keyword).
    pub async fn batch_update_keyword_weights(
        &self,
        updates: &[(f64, String, String)],
    ) -> Result<(), sqlx::Error> {
        for (weight, memory_id, keyword) in updates {
            sqlx::query(
                "UPDATE memory_keywords SET weight = MIN(?, 1.0) WHERE memory_id = ? AND keyword = ?",
            )
            .bind(weight)
            .bind(memory_id)
            .bind(keyword)
            .execute(self.pool())
            .await?;
        }
        Ok(())
    }

    /// Return all keyword-memory associations for active tiers (A2.5).
    pub async fn get_all_keywords_with_memories(
        &self,
        tiers: Option<&[String]>,
    ) -> Result<Vec<HashMap<String, serde_json::Value>>, sqlx::Error> {
        let tiers_to_use: Vec<String> = tiers
            .map(|t| t.to_vec())
            .unwrap_or_else(|| vec!["hot".to_string(), "warm".to_string()]);

        let placeholders = tiers_to_use.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT mk.keyword, mk.memory_id, mk.weight \
             FROM memory_keywords mk \
             JOIN memories m ON mk.memory_id = m.id \
             WHERE m.tier IN ({placeholders})"
        );
        let mut query = sqlx::query(&sql);
        for tier in &tiers_to_use {
            query = query.bind(tier);
        }
        let rows: Vec<SqliteRow> = query.fetch_all(self.pool()).await?;

        let results = rows
            .iter()
            .map(|r| {
                let mut map = HashMap::new();
                let kw: String = r.get("keyword");
                let mid: String = r.get("memory_id");
                let w: f64 = r.get("weight");
                map.insert("keyword".to_string(), serde_json::Value::String(kw));
                map.insert("memory_id".to_string(), serde_json::Value::String(mid));
                map.insert(
                    "weight".to_string(),
                    serde_json::Value::Number(serde_json::Number::from_f64(w).unwrap_or_else(|| serde_json::Number::from(0))),
                );
                map
            })
            .collect();
        Ok(results)
    }

    // ── Access log ──

    pub async fn log_access(
        &self,
        access_id: &str,
        memory_id: &str,
        accessed_at: &str,
        access_type: &str,
        session_id: Option<&str>,
        query: Option<&str>,
    ) -> Result<(), sqlx::Error> {
        sqlx::query(
            "INSERT INTO memory_access_log (id, memory_id, accessed_at, access_type, session_id, query) \
             VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(access_id)
        .bind(memory_id)
        .bind(accessed_at)
        .bind(access_type)
        .bind(session_id)
        .bind(query)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    /// Return recent retrieval queries for a session (A2.1 gap detection).
    pub async fn get_recent_access_queries(
        &self,
        session_id: &str,
        limit: i64,
    ) -> Result<Vec<String>, sqlx::Error> {
        let rows: Vec<SqliteRow> = sqlx::query(
            "SELECT query FROM memory_access_log \
             WHERE session_id = ? AND query IS NOT NULL \
             ORDER BY accessed_at DESC LIMIT ?",
        )
        .bind(session_id)
        .bind(limit)
        .fetch_all(self.pool())
        .await?;

        Ok(rows.iter().map(|r| r.get("query")).collect())
    }

    /// Identify keywords from queries that yielded no or only cold-tier results (A2.1).
    pub async fn get_failed_retrieval_keywords(
        &self,
        session_id: &str,
        lookback: i64,
    ) -> Result<Vec<String>, sqlx::Error> {
        let queries = self.get_recent_access_queries(session_id, lookback).await?;
        if queries.is_empty() {
            return Ok(Vec::new());
        }

        let mut gap_keywords: Vec<String> = Vec::new();

        for query in &queries {
            let rows: Vec<SqliteRow> = sqlx::query(
                "SELECT m.tier FROM memory_access_log mal \
                 JOIN memories m ON mal.memory_id = m.id \
                 WHERE mal.query = ? AND mal.session_id = ?",
            )
            .bind(query)
            .bind(session_id)
            .fetch_all(self.pool())
            .await?;

            let tiers: Vec<String> = rows.iter().map(|r| r.get("tier")).collect();

            // If no results or only cold-tier, this is a gap
            if tiers.is_empty() || tiers.iter().all(|t| t == "cold") {
                let words: Vec<String> = query
                    .split_whitespace()
                    .filter(|w| w.len() > 2)
                    .map(|w| w.to_lowercase())
                    .collect();
                gap_keywords.extend(words);
            }
        }

        // Deduplicate
        gap_keywords.sort();
        gap_keywords.dedup();
        Ok(gap_keywords)
    }

    // ── Save decisions ──

    pub async fn log_save_decision(&self, dec: &SaveDecision) -> Result<(), sqlx::Error> {
        sqlx::query(
            "INSERT INTO save_decisions \
             (id, raw_log_id, session_id, turn, decided_at, decision, reason, confidence, gap_triggered, threshold_used) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&dec.id)
        .bind(&dec.raw_log_id)
        .bind(&dec.session_id)
        .bind(dec.turn)
        .bind(&dec.decided_at)
        .bind(&dec.decision)
        .bind(&dec.reason)
        .bind(dec.confidence)
        .bind(dec.gap_triggered as i32)
        .bind(&dec.threshold_used)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    /// Mark whether a saved memory turned out to be useful (A4).
    pub async fn update_save_outcome(
        &self,
        decision_id: &str,
        useful: bool,
        assessed_at: &str,
    ) -> Result<(), sqlx::Error> {
        sqlx::query(
            "UPDATE save_decisions SET outcome_useful = ?, outcome_assessed_at = ? WHERE id = ?",
        )
        .bind(useful as i32)
        .bind(assessed_at)
        .bind(decision_id)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    /// Get save decisions that haven't been assessed yet (A4).
    pub async fn get_unassessed_save_decisions(
        &self,
        lookback_days: i64,
    ) -> Result<Vec<HashMap<String, serde_json::Value>>, sqlx::Error> {
        let offset = format!("-{lookback_days} days");
        let rows: Vec<SqliteRow> = sqlx::query(
            "SELECT sd.id, sd.raw_log_id, m.id as memory_id, m.access_count \
             FROM save_decisions sd \
             LEFT JOIN memories m ON m.raw_log_id = sd.raw_log_id \
             WHERE sd.decision IN ('save', 'fast_path') \
               AND sd.id NOT IN ( \
                   SELECT id FROM save_decisions WHERE outcome_useful IS NOT NULL \
               ) \
               AND sd.decided_at < datetime('now', ?)",
        )
        .bind(&offset)
        .fetch_all(self.pool())
        .await?;

        Ok(rows.iter().map(|r| sqlite_row_to_map(r)).collect())
    }

    // ── Retrieval decisions (A4) ──

    pub async fn log_retrieval_decision(
        &self,
        decision_id: &str,
        session_id: &str,
        turn: Option<i64>,
        query: &str,
        decided_at: &str,
        layers_queried: &[String],
        graph_depth: i32,
        mood_weight: f64,
        top_k: i32,
        memory_ids: &[String],
        return_count: i32,
    ) -> Result<(), sqlx::Error> {
        let layers_json = serde_json::to_string(layers_queried).unwrap_or_default();
        let memories_json = serde_json::to_string(memory_ids).unwrap_or_default();

        sqlx::query(
            "INSERT INTO retrieval_decisions \
             (id, session_id, turn, query, decided_at, layers_queried, graph_depth, \
              mood_weight, top_k, memories_returned, return_count) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(decision_id)
        .bind(session_id)
        .bind(turn)
        .bind(query)
        .bind(decided_at)
        .bind(&layers_json)
        .bind(graph_depth)
        .bind(mood_weight)
        .bind(top_k)
        .bind(&memories_json)
        .bind(return_count)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    /// Get retrieval decisions not yet assessed (A4).
    pub async fn get_unassessed_retrieval_decisions(
        &self,
    ) -> Result<Vec<HashMap<String, serde_json::Value>>, sqlx::Error> {
        let rows: Vec<SqliteRow> = sqlx::query(
            "SELECT id, session_id, turn, query \
             FROM retrieval_decisions \
             WHERE outcome_helpful IS NULL \
               AND decided_at < datetime('now', '-1 hour')",
        )
        .fetch_all(self.pool())
        .await?;

        Ok(rows.iter().map(|r| sqlite_row_to_map(r)).collect())
    }

    pub async fn update_retrieval_outcome(
        &self,
        decision_id: &str,
        helpful: bool,
        assessed_at: &str,
    ) -> Result<(), sqlx::Error> {
        sqlx::query(
            "UPDATE retrieval_decisions SET outcome_helpful = ?, outcome_assessed_at = ? WHERE id = ?",
        )
        .bind(helpful as i32)
        .bind(assessed_at)
        .bind(decision_id)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    /// Get follow-up queries within a turn window (A4 outcome assessment).
    pub async fn get_retrieval_followups(
        &self,
        session_id: &str,
        turn: i64,
        window: i32,
    ) -> Result<Vec<String>, sqlx::Error> {
        let rows: Vec<SqliteRow> = sqlx::query(
            "SELECT query FROM retrieval_decisions WHERE session_id = ? AND turn > ? AND turn <= ?",
        )
        .bind(session_id)
        .bind(turn)
        .bind(turn + window as i64)
        .fetch_all(self.pool())
        .await?;

        Ok(rows.iter().map(|r| r.get("query")).collect())
    }

    // ── Dream exploration logging (A3) ──

    pub async fn log_dream_run(
        &self,
        run_id: &str,
        ran_at: &str,
        n_walks: i64,
        edges_discovered: i64,
        edges_committed: i64,
        strategies: &[String],
        notes: Option<&str>,
    ) -> Result<(), sqlx::Error> {
        let strategies_json = serde_json::to_string(strategies).unwrap_or_default();

        sqlx::query(
            "INSERT INTO dream_exploration_runs \
             (id, ran_at, n_walks, edges_discovered, edges_committed, strategies_used, notes) \
             VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(run_id)
        .bind(ran_at)
        .bind(n_walks)
        .bind(edges_discovered)
        .bind(edges_committed)
        .bind(&strategies_json)
        .bind(notes)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    pub async fn log_dream_edge(
        &self,
        edge_id: &str,
        run_id: &str,
        source_id: &str,
        target_id: &str,
        similarity: f64,
        relationship_type: &str,
        discovery_method: &str,
        committed: bool,
    ) -> Result<(), sqlx::Error> {
        sqlx::query(
            "INSERT INTO dream_discovered_edges \
             (id, exploration_run_id, source_memory_id, target_memory_id, \
              similarity, relationship_type, discovery_method, committed) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(edge_id)
        .bind(run_id)
        .bind(source_id)
        .bind(target_id)
        .bind(similarity)
        .bind(relationship_type)
        .bind(discovery_method)
        .bind(committed as i32)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    // ── Compaction ──

    pub async fn log_compaction_run(&self, result: &CompactionResult) -> Result<(), sqlx::Error> {
        sqlx::query(
            "INSERT INTO compaction_runs \
             (id, ran_at, trigger, memories_reviewed, memories_merged, memories_pruned, \
              notes, keywords_updated, edges_discovered) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&result.id)
        .bind(&result.ran_at)
        .bind(&result.trigger)
        .bind(result.memories_reviewed)
        .bind(result.memories_merged)
        .bind(result.memories_pruned)
        .bind(&result.notes)
        .bind(result.keywords_updated)
        .bind(result.edges_discovered)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    pub async fn log_compaction_merge(
        &self,
        compaction_id: &str,
        source_ids: &[String],
        resulting_id: &str,
        validation_passed: Option<bool>,
        avg_source_score: Option<f64>,
        avg_merged_score: Option<f64>,
        degradation: Option<f64>,
    ) -> Result<(), sqlx::Error> {
        let source_json = serde_json::to_string(source_ids).unwrap_or_default();
        sqlx::query(
            "INSERT INTO compaction_merges \
             (compaction_id, source_memory_ids, resulting_memory_id, \
              validation_passed, avg_source_score, avg_merged_score, degradation) \
             VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(compaction_id)
        .bind(&source_json)
        .bind(resulting_id)
        .bind(validation_passed.map(|v| v as i32))
        .bind(avg_source_score)
        .bind(avg_merged_score)
        .bind(degradation)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    /// Return memories that are candidates for compaction.
    pub async fn get_compaction_candidates(
        &self,
        threshold: f64,
    ) -> Result<Vec<Memory>, sqlx::Error> {
        let rows: Vec<SqliteRow> = sqlx::query(
            "SELECT m.* FROM memories m \
             LEFT JOIN ( \
                 SELECT memory_id, COUNT(*) as edge_count \
                 FROM memory_keywords \
                 GROUP BY memory_id \
             ) kc ON m.id = kc.memory_id \
             WHERE m.tier = 'hot' \
               AND m.fast_pathed = 0 \
               AND NOT (m.compaction_gen = 0 AND m.access_count > 5) \
             ORDER BY ((1 - m.decay_score) * 0.6 + (1 - m.salience) * 0.4) DESC",
        )
        .fetch_all(self.pool())
        .await?;

        let mut candidates: Vec<Memory> = Vec::new();
        for r in &rows {
            let mem = row_to_memory(r);
            let score = (1.0 - mem.decay_score) * 0.6 + (1.0 - mem.salience) * 0.4;
            if score > threshold {
                candidates.push(mem);
            }
        }

        // Load keywords
        if !candidates.is_empty() {
            let ids: Vec<&str> = candidates.iter().map(|m| m.id.as_str()).collect();
            let ph = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!(
                "SELECT memory_id, keyword, weight FROM memory_keywords WHERE memory_id IN ({ph})"
            );
            let mut query = sqlx::query(&sql);
            for id in &ids {
                query = query.bind(id);
            }
            let kw_rows: Vec<SqliteRow> = query.fetch_all(self.pool()).await?;

            let mut kw_map: HashMap<String, Vec<(String, f64)>> = HashMap::new();
            for r in &kw_rows {
                let mid: String = r.get("memory_id");
                let kw: String = r.get("keyword");
                let w: f64 = r.get("weight");
                kw_map.entry(mid).or_default().push((kw, w));
            }
            for m in &mut candidates {
                if let Some(kws) = kw_map.remove(&m.id) {
                    m.keywords = kws;
                }
            }
        }

        Ok(candidates)
    }

    // ── Policy data export (A4.4) ──

    /// Export assessed save decisions for policy training.
    pub async fn export_save_policy_data(
        &self,
    ) -> Result<Vec<HashMap<String, serde_json::Value>>, sqlx::Error> {
        let rows: Vec<SqliteRow> = sqlx::query(
            "SELECT sd.confidence, sd.decision, sd.gap_triggered, \
                    m.valence, m.arousal, m.surprise, m.salience, \
                    sd.outcome_useful \
             FROM save_decisions sd \
             LEFT JOIN memories m ON m.raw_log_id = sd.raw_log_id \
             WHERE sd.outcome_useful IS NOT NULL",
        )
        .fetch_all(self.pool())
        .await?;

        Ok(rows.iter().map(|r| sqlite_row_to_map(r)).collect())
    }

    /// Export assessed retrieval decisions for policy training.
    pub async fn export_retrieval_policy_data(
        &self,
    ) -> Result<Vec<HashMap<String, serde_json::Value>>, sqlx::Error> {
        let rows: Vec<SqliteRow> = sqlx::query(
            "SELECT layers_queried, graph_depth, mood_weight, top_k, \
                    return_count, outcome_helpful \
             FROM retrieval_decisions \
             WHERE outcome_helpful IS NOT NULL",
        )
        .fetch_all(self.pool())
        .await?;

        Ok(rows.iter().map(|r| sqlite_row_to_map(r)).collect())
    }

    /// Get memories that have vector embeddings (for dream explorer).
    pub async fn get_memories_with_vectors(
        &self,
        tiers: Option<&[String]>,
    ) -> Result<Vec<HashMap<String, serde_json::Value>>, sqlx::Error> {
        let (tier_clause, tiers_to_use) = if let Some(tiers) = tiers {
            let ph = tiers.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            (format!("AND tier IN ({ph})"), tiers.to_vec())
        } else {
            (String::new(), Vec::new())
        };

        let sql = format!(
            "SELECT id, session_id, vector_id FROM memories WHERE vector_id IS NOT NULL {tier_clause}"
        );
        let mut query = sqlx::query(&sql);
        for tier in &tiers_to_use {
            query = query.bind(tier);
        }
        let rows: Vec<SqliteRow> = query.fetch_all(self.pool()).await?;

        Ok(rows.iter().map(|r| sqlite_row_to_map(r)).collect())
    }
}

/// Convert a SQLite row to a Memory struct.
fn row_to_memory(row: &SqliteRow) -> Memory {
    Memory {
        id: row.get("id"),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
        content: row.get("content"),
        summary: row.try_get("summary").ok(),
        raw_log_id: row.get("raw_log_id"),
        session_id: row.get("session_id"),
        turn: row.get("turn"),
        valence: row.try_get("valence").unwrap_or(0.0),
        arousal: row.try_get("arousal").unwrap_or(0.0),
        surprise: row.try_get("surprise").unwrap_or(0.0),
        salience: row.try_get("salience").unwrap_or(0.5),
        access_count: row.try_get("access_count").unwrap_or(0),
        last_accessed: row.try_get("last_accessed").ok(),
        decay_score: row.try_get("decay_score").unwrap_or(1.0),
        compaction_gen: row.try_get("compaction_gen").unwrap_or(0),
        tier: row.try_get("tier").unwrap_or_else(|_| "hot".to_string()),
        fast_pathed: row
            .try_get::<i32, _>("fast_pathed")
            .map(|v| v != 0)
            .unwrap_or(false),
        is_semantic: row
            .try_get::<i32, _>("is_semantic")
            .map(|v| v != 0)
            .unwrap_or(false),
        graph_node_id: row.try_get("graph_node_id").ok(),
        vector_id: row.try_get("vector_id").ok(),
        spatial_embedding: row.try_get("spatial_embedding").ok(),
        scene_description: row.try_get("scene_description").ok(),
        keywords: Vec::new(), // loaded separately
    }
}

/// Generic helper to convert a SQLite row to a HashMap.
fn sqlite_row_to_map(row: &SqliteRow) -> HashMap<String, serde_json::Value> {
    use sqlx::Column;
    let mut map = HashMap::new();
    for col in row.columns() {
        let name = col.name().to_string();
        // Try different types
        if let Ok(v) = row.try_get::<String, _>(name.as_str()) {
            map.insert(name, serde_json::Value::String(v));
        } else if let Ok(v) = row.try_get::<i64, _>(name.as_str()) {
            map.insert(name, serde_json::json!(v));
        } else if let Ok(v) = row.try_get::<f64, _>(name.as_str()) {
            map.insert(
                name,
                serde_json::Value::Number(
                    serde_json::Number::from_f64(v).unwrap_or_else(|| serde_json::Number::from(0)),
                ),
            );
        } else {
            map.insert(name, serde_json::Value::Null);
        }
    }
    map
}
