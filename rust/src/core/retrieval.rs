//! Three-layer retrieval stack with mood-congruent weighting.
//!
//! Layers:
//!   1. Grep — raw log search via ripgrep subprocess
//!   2. Keyword — SQLite keyword search weighted by decay score
//!   3. Semantic — Qdrant vector similarity search
//!
//! Results from all layers are merged, deduplicated, and re-ranked. Graph
//! traversal expands the result set along RELATES_TO edges. The visual
//! channel provides an independent retrieval path via CLIP embeddings.

use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};
use log::{debug, warn};
use uuid::Uuid;

use crate::config::{default_log_dir, MemoryConfig};
use crate::core::decay::compute_decay;
use crate::embeddings::text_embedder::TextEmbedderTrait;
use crate::embeddings::visual_embedder::VisualEmbedderTrait;
use crate::emotion::scorer::score_emotion;
use crate::llm::client::LlmClient;
use crate::models::Memory;
use crate::storage::graph_store::GraphStore;
use crate::storage::sqlite_store::SQLiteStore;
use crate::storage::vector_store::VectorStore;

/// Internal candidate tracking during retrieval merge.
struct Candidate {
    memory_id: String,
    score: f64,
    layers: HashSet<String>,
}

/// Orchestrates multi-layer retrieval across all storage backends.
pub struct RetrievalEngine<'a> {
    sqlite: &'a SQLiteStore,
    graph: &'a GraphStore,
    vector: &'a VectorStore,
    text_embedder: &'a dyn TextEmbedderTrait,
    visual_embedder: Option<&'a dyn VisualEmbedderTrait>,
    llm: &'a LlmClient,
    config: &'a MemoryConfig,
}

impl<'a> RetrievalEngine<'a> {
    pub fn new(
        sqlite: &'a SQLiteStore,
        graph: &'a GraphStore,
        vector: &'a VectorStore,
        text_embedder: &'a dyn TextEmbedderTrait,
        visual_embedder: Option<&'a dyn VisualEmbedderTrait>,
        llm: &'a LlmClient,
        config: &'a MemoryConfig,
    ) -> Self {
        Self {
            sqlite,
            graph,
            vector,
            text_embedder,
            visual_embedder,
            llm,
            config,
        }
    }

    /// Run all retrieval layers and return ranked, deduplicated memories.
    pub async fn retrieve(
        &self,
        query: &str,
        session_id: Option<&str>,
        top_k: Option<usize>,
        enable_mood_congruent: bool,
        enable_visual: bool,
    ) -> Result<Vec<Memory>, Box<dyn std::error::Error + Send + Sync>> {
        let top_k = top_k.unwrap_or(self.config.top_k_per_layer);

        // Score current context for mood-congruent weighting
        let context_emotion = if enable_mood_congruent {
            let scores = score_emotion(query, self.config, self.llm).await;
            if scores.contains_key("valence") {
                Some(scores)
            } else {
                debug!("Mood scoring failed, proceeding without");
                None
            }
        } else {
            None
        };

        // Run layers concurrently
        let (grep_result, keyword_result, semantic_result) = tokio::join!(
            self.grep_layer(query, top_k),
            self.keyword_layer(query, top_k),
            self.semantic_layer(query, top_k),
        );

        let visual_result = if enable_visual && self.visual_embedder.is_some() {
            Some(self.visual_layer(query, top_k).await)
        } else {
            None
        };

        // Merge all candidates
        let mut candidates: HashMap<String, Candidate> = HashMap::new();

        let layer_results: Vec<(&str, Result<Vec<(String, f64)>, Box<dyn std::error::Error + Send + Sync>>)> = {
            let mut v: Vec<(&str, Result<Vec<(String, f64)>, Box<dyn std::error::Error + Send + Sync>>)> = vec![
                ("grep", grep_result),
                ("keyword", keyword_result),
                ("semantic", semantic_result),
            ];
            if let Some(vr) = visual_result {
                v.push(("visual", vr));
            }
            v
        };

        for (layer_name, result) in layer_results {
            match result {
                Ok(hits) => {
                    for (mem_id, score) in hits {
                        if let Some(cand) = candidates.get_mut(&mem_id) {
                            cand.score += score;
                            cand.layers.insert(layer_name.to_string());
                        } else {
                            let mut layers = HashSet::new();
                            layers.insert(layer_name.to_string());
                            candidates.insert(
                                mem_id.clone(),
                                Candidate {
                                    memory_id: mem_id,
                                    score,
                                    layers,
                                },
                            );
                        }
                    }
                }
                Err(e) => {
                    warn!("Retrieval layer {layer_name} failed: {e}");
                }
            }
        }

        // Graph traversal expansion
        let candidate_ids: Vec<String> = candidates.keys().cloned().collect();
        for mem_id in &candidate_ids {
            match self
                .graph
                .get_related_memories(mem_id, self.config.graph_traversal_depth as i32, 0.0)
                .await
            {
                Ok(related) => {
                    for rel in related {
                        if !candidates.contains_key(&rel.id) {
                            let depth_score =
                                1.0 / (rel.depth as f64 + 1.0) * rel.salience;
                            let mut layers = HashSet::new();
                            layers.insert("graph_traversal".to_string());
                            candidates.insert(
                                rel.id.clone(),
                                Candidate {
                                    memory_id: rel.id,
                                    score: depth_score,
                                    layers,
                                },
                            );
                        }
                    }
                }
                Err(e) => {
                    debug!("Graph traversal failed for {mem_id}: {e}");
                }
            }
        }

        // Load full memories from SQLite
        let mut memories: Vec<(Memory, f64)> = Vec::new();
        for cand in candidates.values() {
            let mem = match self.sqlite.get_memory(&cand.memory_id).await {
                Ok(Some(m)) => m,
                _ => continue,
            };

            let mut score = cand.score;

            // Mood-congruent boosting
            if let Some(ref emotion) = context_emotion {
                if enable_mood_congruent {
                    let mood_weight = self.config.mood_congruent_weight;
                    let valence_sim = 1.0
                        - (emotion.get("valence").copied().unwrap_or(0.0) - mem.valence).abs()
                            / 2.0;
                    let arousal_sim = 1.0
                        - (emotion.get("arousal").copied().unwrap_or(0.0) - mem.arousal).abs();
                    let mood_bonus = (valence_sim + arousal_sim) / 2.0 * mood_weight;
                    score += mood_bonus;
                }
            }

            // Factor in decay
            score *= mem.decay_score;

            memories.push((mem, score));
        }

        // Sort by score descending
        memories.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Log access and update decay for returned memories
        let now = Utc::now().to_rfc3339();
        let now_dt: DateTime<Utc> = Utc::now();
        let mut result_memories = Vec::new();

        for (mut mem, _score) in memories.into_iter().take(top_k * 2) {
            let cand = candidates.get(&mem.id);
            let access_type = if cand.is_some_and(|c| c.layers.contains("graph_traversal")) {
                "graph_traversal"
            } else if cand
                .is_some_and(|c| c.layers.contains("grep") && c.layers.len() == 1)
            {
                "grep_entrypoint"
            } else if cand.is_some_and(|c| c.layers.contains("semantic")) {
                "vector"
            } else {
                "primary"
            };

            mem.access_count += 1;
            mem.last_accessed = Some(now.clone());
            mem.decay_score = compute_decay(
                now_dt,
                mem.access_count,
                mem.arousal,
                mem.surprise,
                mem.is_semantic,
                self.config,
            );

            let _ = self
                .sqlite
                .log_access(
                    &Uuid::new_v4().to_string(),
                    &mem.id,
                    &now,
                    access_type,
                    session_id,
                    Some(query),
                )
                .await;

            let _ = self
                .sqlite
                .update_memory_access(&mem.id, mem.decay_score, mem.access_count, &now)
                .await;

            result_memories.push(mem);
        }

        Ok(result_memories)
    }

    // ── Layer implementations ──

    async fn grep_layer(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(String, f64)>, Box<dyn std::error::Error + Send + Sync>> {
        let log_dir = default_log_dir();
        let log_dir_str = log_dir.to_string_lossy();
        let terms: Vec<&str> = query.split_whitespace().take(5).collect();
        let pattern = terms.join("|");

        let output = match tokio::process::Command::new("rg")
            .args(["--json", "-i", "-e", &pattern, &log_dir_str])
            .output()
            .await
        {
            Ok(o) => o,
            Err(_) => {
                debug!("ripgrep not found, skipping grep layer");
                return Ok(Vec::new());
            }
        };

        if output.stdout.is_empty() {
            return Ok(Vec::new());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut hits: HashMap<String, usize> = HashMap::new();

        for line in stdout.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Ok(obj) = serde_json::from_str::<serde_json::Value>(trimmed) {
                if obj.get("type").and_then(|v| v.as_str()) == Some("match") {
                    if let Some(text) = obj
                        .get("data")
                        .and_then(|d| d.get("lines"))
                        .and_then(|l| l.get("text"))
                        .and_then(|t| t.as_str())
                    {
                        if let Ok(entry) = serde_json::from_str::<serde_json::Value>(text) {
                            if let Some(id) = entry.get("id").and_then(|v| v.as_str()) {
                                *hits.entry(id.to_string()).or_insert(0) += 1;
                            }
                        }
                    }
                }
            }
        }

        // Map raw_log_ids to memory_ids via SQLite
        let mut results: Vec<(String, f64)> = Vec::new();
        let mut sorted_hits: Vec<_> = hits.into_iter().collect();
        sorted_hits.sort_by(|a, b| b.1.cmp(&a.1));

        for (raw_id, count) in sorted_hits.into_iter().take(limit) {
            if self.sqlite.get_raw_log_ref(&raw_id).await?.is_some() {
                // Find memory by raw_log_id
                if let Ok(Some(mem)) = self.sqlite.get_memory(&raw_id).await {
                    results.push((mem.id, count as f64));
                }
            }
        }

        Ok(results)
    }

    async fn keyword_layer(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(String, f64)>, Box<dyn std::error::Error + Send + Sync>> {
        let keywords: Vec<String> = query
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .map(|w| w.to_lowercase())
            .collect();

        if keywords.is_empty() {
            return Ok(Vec::new());
        }

        let memories = self.sqlite.search_by_keywords(&keywords, limit as i64).await?;
        Ok(memories.iter().map(|m| (m.id.clone(), m.decay_score)).collect())
    }

    async fn semantic_layer(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(String, f64)>, Box<dyn std::error::Error + Send + Sync>> {
        let query_vector = self.text_embedder.embed(query).await?;
        let results = self.vector.search_text(query_vector, limit, None).await?;
        Ok(results.iter().map(|r| (r.memory_id.clone(), r.score)).collect())
    }

    async fn visual_layer(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(String, f64)>, Box<dyn std::error::Error + Send + Sync>> {
        let embedder = match self.visual_embedder {
            Some(e) => e,
            None => return Ok(Vec::new()),
        };
        let query_vector = embedder.embed(query).await?;
        let results = self.vector.search_visual(query_vector, limit).await?;
        Ok(results.iter().map(|r| (r.memory_id.clone(), r.score)).collect())
    }
}
