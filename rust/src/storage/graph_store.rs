//! Graph store for semantic relationships between memories and entities.
//!
//! Since Kuzu doesn't have stable Rust bindings, graph operations are implemented
//! using SQLite with recursive CTEs. Separate tables for nodes (Memory, Entity) and
//! edges (RELATES_TO, MENTIONS, EVOLVED_FROM) with recursive WITH queries for traversal.

use std::collections::HashSet;
use std::path::PathBuf;

use log::debug;
use sqlx::sqlite::{SqlitePool, SqlitePoolOptions, SqliteRow};
use sqlx::Row;

use crate::config::default_graph_dir;

const GRAPH_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS memory_nodes (
    id             TEXT PRIMARY KEY,
    summary        TEXT,
    tier           TEXT,
    salience       REAL,
    valence        REAL,
    compaction_gen  INTEGER,
    created_at     TEXT
);

CREATE TABLE IF NOT EXISTS entity_nodes (
    id    TEXT PRIMARY KEY,
    name  TEXT,
    type  TEXT
);

CREATE TABLE IF NOT EXISTS relates_to_edges (
    from_id           TEXT NOT NULL,
    to_id             TEXT NOT NULL,
    weight            REAL,
    relationship_type TEXT,
    created_at        TEXT,
    PRIMARY KEY (from_id, to_id, relationship_type),
    FOREIGN KEY (from_id) REFERENCES memory_nodes(id),
    FOREIGN KEY (to_id) REFERENCES memory_nodes(id)
);

CREATE TABLE IF NOT EXISTS mentions_edges (
    memory_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    weight    REAL,
    PRIMARY KEY (memory_id, entity_id),
    FOREIGN KEY (memory_id) REFERENCES memory_nodes(id),
    FOREIGN KEY (entity_id) REFERENCES entity_nodes(id)
);

CREATE TABLE IF NOT EXISTS evolved_from_edges (
    new_id        TEXT NOT NULL,
    source_id     TEXT NOT NULL,
    compaction_id TEXT,
    created_at    TEXT,
    PRIMARY KEY (new_id, source_id),
    FOREIGN KEY (new_id) REFERENCES memory_nodes(id),
    FOREIGN KEY (source_id) REFERENCES memory_nodes(id)
);
"#;

/// SQLite-backed graph store for memory relationships.
pub struct GraphStore {
    graph_dir: PathBuf,
    pool: Option<SqlitePool>,
}

impl GraphStore {
    pub fn new(graph_dir: Option<PathBuf>) -> Self {
        Self {
            graph_dir: graph_dir.unwrap_or_else(default_graph_dir),
            pool: None,
        }
    }

    pub async fn initialize(&mut self) -> Result<(), sqlx::Error> {
        std::fs::create_dir_all(&self.graph_dir).ok();
        let db_path = self.graph_dir.join("graph.db");
        let url = format!("sqlite:{}?mode=rwc", db_path.display());
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(&url)
            .await?;

        for statement in GRAPH_SCHEMA.split(';') {
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
            .expect("GraphStore not initialized — call initialize() first")
    }

    // ── Node operations ──

    pub async fn add_memory_node(
        &self,
        memory_id: &str,
        summary: &str,
        tier: &str,
        salience: f64,
        valence: f64,
        compaction_gen: i32,
        created_at: &str,
    ) -> Result<(), sqlx::Error> {
        sqlx::query(
            "INSERT OR REPLACE INTO memory_nodes (id, summary, tier, salience, valence, compaction_gen, created_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(memory_id)
        .bind(summary)
        .bind(tier)
        .bind(salience)
        .bind(valence)
        .bind(compaction_gen)
        .bind(created_at)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    pub async fn add_entity_node(
        &self,
        entity_id: &str,
        name: &str,
        entity_type: &str,
    ) -> Result<(), sqlx::Error> {
        sqlx::query(
            "INSERT OR REPLACE INTO entity_nodes (id, name, type) VALUES (?, ?, ?)",
        )
        .bind(entity_id)
        .bind(name)
        .bind(entity_type)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    // ── Edge operations ──

    pub async fn add_relates_to(
        &self,
        from_id: &str,
        to_id: &str,
        weight: f64,
        relationship_type: &str,
        created_at: &str,
    ) -> Result<(), sqlx::Error> {
        sqlx::query(
            "INSERT OR REPLACE INTO relates_to_edges (from_id, to_id, weight, relationship_type, created_at) \
             VALUES (?, ?, ?, ?, ?)",
        )
        .bind(from_id)
        .bind(to_id)
        .bind(weight)
        .bind(relationship_type)
        .bind(created_at)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    pub async fn add_mentions(
        &self,
        memory_id: &str,
        entity_id: &str,
        weight: f64,
    ) -> Result<(), sqlx::Error> {
        sqlx::query(
            "INSERT OR REPLACE INTO mentions_edges (memory_id, entity_id, weight) VALUES (?, ?, ?)",
        )
        .bind(memory_id)
        .bind(entity_id)
        .bind(weight)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    pub async fn add_evolved_from(
        &self,
        new_id: &str,
        source_id: &str,
        compaction_id: &str,
        created_at: &str,
    ) -> Result<(), sqlx::Error> {
        sqlx::query(
            "INSERT OR REPLACE INTO evolved_from_edges (new_id, source_id, compaction_id, created_at) \
             VALUES (?, ?, ?, ?)",
        )
        .bind(new_id)
        .bind(source_id)
        .bind(compaction_id)
        .bind(created_at)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    // ── Queries ──

    /// Traverse RELATES_TO edges up to `max_depth` hops from a memory node.
    /// Uses recursive CTE for multi-hop traversal.
    pub async fn get_related_memories(
        &self,
        memory_id: &str,
        max_depth: i32,
        min_weight: f64,
    ) -> Result<Vec<RelatedMemory>, sqlx::Error> {
        // max_depth is an i32 parameter, safe to embed as a literal since sqlx
        // does not support binding integers in recursive CTE depth comparisons.
        let max_depth = max_depth.max(0);
        let sql = format!(
            "WITH RECURSIVE traversal(id, depth) AS ( \
                 SELECT to_id, 1 FROM relates_to_edges WHERE from_id = ? AND weight >= ? \
                 UNION \
                 SELECT r.to_id, t.depth + 1 \
                 FROM relates_to_edges r \
                 JOIN traversal t ON r.from_id = t.id \
                 WHERE t.depth < {max_depth} AND r.weight >= ? \
             ) \
             SELECT DISTINCT m.id, m.summary, m.tier, m.salience, t.depth \
             FROM traversal t \
             JOIN memory_nodes m ON t.id = m.id \
             WHERE m.id != ?"
        );

        let rows: Vec<SqliteRow> = sqlx::query(&sql)
            .bind(memory_id)
            .bind(min_weight)
            .bind(min_weight)
            .bind(memory_id)
            .fetch_all(self.pool())
            .await?;

        Ok(rows
            .iter()
            .map(|r| RelatedMemory {
                id: r.get("id"),
                summary: r.try_get("summary").ok(),
                tier: r.try_get("tier").unwrap_or_else(|_| "hot".to_string()),
                salience: r.try_get("salience").unwrap_or(0.5),
                depth: r.try_get("depth").unwrap_or(1),
            })
            .collect())
    }

    /// Get all entities mentioned by a memory.
    pub async fn get_memory_entities(
        &self,
        memory_id: &str,
    ) -> Result<Vec<EntityMention>, sqlx::Error> {
        let rows: Vec<SqliteRow> = sqlx::query(
            "SELECT e.id, e.name, e.type, me.weight \
             FROM mentions_edges me \
             JOIN entity_nodes e ON me.entity_id = e.id \
             WHERE me.memory_id = ?",
        )
        .bind(memory_id)
        .fetch_all(self.pool())
        .await?;

        Ok(rows
            .iter()
            .map(|r| EntityMention {
                id: r.get("id"),
                name: r.get("name"),
                entity_type: r.try_get("type").unwrap_or_default(),
                weight: r.try_get("weight").unwrap_or(1.0),
            })
            .collect())
    }

    /// Trace the full lineage of a compacted memory back to originals.
    pub async fn get_evolution_lineage(
        &self,
        memory_id: &str,
    ) -> Result<Vec<LineageEntry>, sqlx::Error> {
        let rows: Vec<SqliteRow> = sqlx::query(
            "WITH RECURSIVE lineage(id, depth) AS ( \
                 SELECT source_id, 1 FROM evolved_from_edges WHERE new_id = ? \
                 UNION \
                 SELECT ef.source_id, l.depth + 1 \
                 FROM evolved_from_edges ef \
                 JOIN lineage l ON ef.new_id = l.id \
                 WHERE l.depth < 10 \
             ) \
             SELECT m.id, m.summary, m.compaction_gen as gen, l.depth \
             FROM lineage l \
             JOIN memory_nodes m ON l.id = m.id \
             ORDER BY l.depth",
        )
        .bind(memory_id)
        .fetch_all(self.pool())
        .await?;

        Ok(rows
            .iter()
            .map(|r| LineageEntry {
                id: r.get("id"),
                summary: r.try_get("summary").ok(),
                gen: r.try_get("gen").unwrap_or(0),
                depth: r.try_get("depth").unwrap_or(1),
            })
            .collect())
    }

    /// Count RELATES_TO edges connected to a memory (both directions).
    pub async fn get_edge_count(&self, memory_id: &str) -> Result<i64, sqlx::Error> {
        let row: SqliteRow = sqlx::query(
            "SELECT (SELECT COUNT(*) FROM relates_to_edges WHERE from_id = ?) + \
                    (SELECT COUNT(*) FROM relates_to_edges WHERE to_id = ?) AS cnt",
        )
        .bind(memory_id)
        .bind(memory_id)
        .fetch_one(self.pool())
        .await?;
        Ok(row.get("cnt"))
    }

    /// Copy all RELATES_TO edges from source memories to a new compacted memory node.
    /// Skips edges between source nodes (they are being merged).
    pub async fn replicate_edges_to_new_node(
        &self,
        source_ids: &[String],
        new_id: &str,
    ) -> Result<(), sqlx::Error> {
        let src_set: HashSet<&str> = source_ids.iter().map(|s| s.as_str()).collect();

        for src_id in source_ids {
            // Outgoing edges
            let rows: Vec<SqliteRow> = sqlx::query(
                "SELECT to_id, weight, relationship_type, created_at \
                 FROM relates_to_edges WHERE from_id = ?",
            )
            .bind(src_id)
            .fetch_all(self.pool())
            .await?;

            for r in &rows {
                let target: String = r.get("to_id");
                if !src_set.contains(target.as_str()) && target != new_id {
                    let weight: f64 = r.try_get("weight").unwrap_or(1.0);
                    let rtype: String = r.try_get("relationship_type").unwrap_or_default();
                    let cat: String = r.try_get("created_at").unwrap_or_default();
                    if let Err(e) = self.add_relates_to(new_id, &target, weight, &rtype, &cat).await {
                        debug!("Edge replication skipped: {e}");
                    }
                }
            }

            // Incoming edges
            let rows: Vec<SqliteRow> = sqlx::query(
                "SELECT from_id, weight, relationship_type, created_at \
                 FROM relates_to_edges WHERE to_id = ?",
            )
            .bind(src_id)
            .fetch_all(self.pool())
            .await?;

            for r in &rows {
                let source: String = r.get("from_id");
                if !src_set.contains(source.as_str()) && source != new_id {
                    let weight: f64 = r.try_get("weight").unwrap_or(1.0);
                    let rtype: String = r.try_get("relationship_type").unwrap_or_default();
                    let cat: String = r.try_get("created_at").unwrap_or_default();
                    if let Err(e) = self.add_relates_to(&source, new_id, weight, &rtype, &cat).await {
                        debug!("Edge replication skipped: {e}");
                    }
                }
            }
        }
        Ok(())
    }

    /// Check if a path exists between two memory nodes via RELATES_TO edges.
    pub async fn path_exists(
        &self,
        from_id: &str,
        to_id: &str,
        max_hops: i32,
    ) -> Result<bool, sqlx::Error> {
        // max_hops is an i32 parameter, safe to embed as a literal since sqlx
        // does not support binding integers in recursive CTE depth comparisons.
        let max_hops = max_hops.max(0);
        let sql = format!(
            "WITH RECURSIVE reach(id, depth) AS ( \
                 SELECT to_id, 1 FROM relates_to_edges WHERE from_id = ? \
                 UNION \
                 SELECT r.to_id, re.depth + 1 \
                 FROM relates_to_edges r \
                 JOIN reach re ON r.from_id = re.id \
                 WHERE re.depth < {max_hops} \
             ) \
             SELECT COUNT(*) AS cnt FROM reach WHERE id = ? LIMIT 1"
        );

        let row: SqliteRow = sqlx::query(&sql)
            .bind(from_id)
            .bind(to_id)
            .fetch_one(self.pool())
            .await?;

        let cnt: i64 = row.get("cnt");
        Ok(cnt > 0)
    }

    pub async fn update_memory_tier(
        &self,
        memory_id: &str,
        tier: &str,
    ) -> Result<(), sqlx::Error> {
        sqlx::query("UPDATE memory_nodes SET tier = ? WHERE id = ?")
            .bind(tier)
            .bind(memory_id)
            .execute(self.pool())
            .await?;
        Ok(())
    }
}

/// A related memory found via graph traversal.
#[derive(Debug, Clone)]
pub struct RelatedMemory {
    pub id: String,
    pub summary: Option<String>,
    pub tier: String,
    pub salience: f64,
    pub depth: i32,
}

/// An entity mentioned by a memory.
#[derive(Debug, Clone)]
pub struct EntityMention {
    pub id: String,
    pub name: String,
    pub entity_type: String,
    pub weight: f64,
}

/// An entry in a memory's evolution lineage.
#[derive(Debug, Clone)]
pub struct LineageEntry {
    pub id: String,
    pub summary: Option<String>,
    pub gen: i32,
    pub depth: i32,
}
