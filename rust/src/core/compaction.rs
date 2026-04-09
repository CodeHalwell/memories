//! Compaction scheduler and merge logic.
//!
//! Compaction runs between sessions (the "sleep cycle"). It collapses episodic
//! detail into semantic generalisations, implementing intentional forgetting.
//!
//! Candidate memories are scored, grouped by keyword overlap, and merged when
//! appropriate. Lineage is tracked via EVOLVED_FROM edges so any compacted
//! memory can be traced back to its originals.
//!
//! A2.3: Generation gap guard — only merge memories within 1 generation of each other.
//! A2.4: Merge validation via generative replay — synthetic queries test that
//!       merged memory still surfaces for the same queries as the originals.

use std::collections::HashSet;

use chrono::Utc;
use log::{error, info};
use uuid::Uuid;

use crate::config::MemoryConfig;
use crate::embeddings::text_embedder::TextEmbedderTrait;
use crate::llm::client::LlmClient;
use crate::models::{CompactionResult, Memory, MergeValidation};
use crate::storage::graph_store::GraphStore;
use crate::storage::sqlite_store::SQLiteStore;
use crate::storage::vector_store::{self, VectorStore};

/// Score a memory for compaction candidacy.
///
/// Low decay + low salience = good candidate for compaction.
pub fn compaction_score(memory: &Memory) -> f64 {
    (1.0 - memory.decay_score) * 0.6 + (1.0 - memory.salience) * 0.4
}

/// Compute keyword overlap ratio between two memories (Jaccard similarity).
pub fn keyword_overlap(mem_a: &Memory, mem_b: &Memory) -> f64 {
    let kw_a: HashSet<&str> = mem_a.keywords.iter().map(|(kw, _)| kw.as_str()).collect();
    let kw_b: HashSet<&str> = mem_b.keywords.iter().map(|(kw, _)| kw.as_str()).collect();
    if kw_a.is_empty() || kw_b.is_empty() {
        return 0.0;
    }
    let intersection = kw_a.intersection(&kw_b).count();
    let union = kw_a.union(&kw_b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Check if a group of memories can be merged (no exclusion conditions).
fn can_merge(group: &[Memory], config: &MemoryConfig) -> bool {
    for (i, a) in group.iter().enumerate() {
        for b in &group[i + 1..] {
            // Opposite valence exclusion
            if a.valence * b.valence < 0.0
                && (a.valence - b.valence).abs() > config.valence_merge_exclusion_delta
            {
                return false;
            }
            // Either is fast_pathed gen-0
            if (a.fast_pathed && a.compaction_gen == 0)
                || (b.fast_pathed && b.compaction_gen == 0)
            {
                return false;
            }
            // A2.3: Generation gap guard — only merge within 1 generation
            if (a.compaction_gen - b.compaction_gen).abs() > config.max_generation_gap_for_merge {
                return false;
            }
        }
    }
    true
}

/// Group memories by keyword overlap using greedy clustering.
fn group_by_keywords(candidates: &[Memory], threshold: f64) -> Vec<Vec<usize>> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let mut used: HashSet<usize> = HashSet::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();

    for i in 0..candidates.len() {
        if used.contains(&i) {
            continue;
        }
        let mut group = vec![i];
        used.insert(i);
        for j in 0..candidates.len() {
            if used.contains(&j) {
                continue;
            }
            // Check overlap with all current group members
            let overlaps: Vec<f64> = group
                .iter()
                .map(|&g| keyword_overlap(&candidates[g], &candidates[j]))
                .collect();
            if !overlaps.is_empty() && overlaps.iter().copied().fold(f64::INFINITY, f64::min) >= threshold {
                group.push(j);
                used.insert(j);
            }
        }
        if group.len() > 1 {
            groups.push(group);
        }
    }

    groups
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    vector_store::cosine_similarity(a, b)
}

/// Validate a merge via generative replay (A2.4).
///
/// Generate synthetic queries from source memories, then test whether the
/// candidate merge still retrieves well for those queries.
pub async fn validate_merge(
    source_memories: &[Memory],
    candidate_content: &str,
    text_embedder: &dyn TextEmbedderTrait,
    llm: &LlmClient,
    config: &MemoryConfig,
    n_queries: usize,
    degradation_tolerance: f64,
) -> MergeValidation {
    let source_text: String = source_memories
        .iter()
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");

    let prompt = format!(
        "Generate {n_queries} search queries for this content:\n\n<source_text>\n{source_text}\n</source_text>"
    );

    let queries: Vec<String> = match llm
        .complete_json(&prompt, Some(&config.prompts.synthetic_query), None, None)
        .await
    {
        Ok(value) => {
            if let Some(arr) = value.as_array() {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            } else {
                Vec::new()
            }
        }
        Err(e) => {
            log::warn!("Synthetic query generation failed, failing validation to be safe: {e}");
            return MergeValidation {
                passed: false,
                queries_tested: Vec::new(),
                ..Default::default()
            };
        }
    };

    if queries.is_empty() {
        return MergeValidation {
            passed: true,
            queries_tested: Vec::new(),
            ..Default::default()
        };
    }

    // Embed candidate and sources
    let candidate_emb = match text_embedder.embed(candidate_content).await {
        Ok(v) => v,
        Err(_) => {
            return MergeValidation {
                passed: false,
                ..Default::default()
            }
        }
    };

    let mut source_embs = Vec::new();
    for m in source_memories {
        match text_embedder.embed(&m.content).await {
            Ok(v) => source_embs.push(v),
            Err(_) => {
                return MergeValidation {
                    passed: false,
                    ..Default::default()
                }
            }
        }
    }

    let mut merged_scores = Vec::new();
    let mut source_scores = Vec::new();

    for query in &queries {
        let query_emb = match text_embedder.embed(query).await {
            Ok(v) => v,
            Err(_) => continue,
        };
        merged_scores.push(cosine_similarity(&query_emb, &candidate_emb));
        let best_source = source_embs
            .iter()
            .map(|se| cosine_similarity(&query_emb, se))
            .fold(f64::NEG_INFINITY, f64::max);
        source_scores.push(best_source);
    }

    let avg_merged = if merged_scores.is_empty() {
        0.0
    } else {
        merged_scores.iter().sum::<f64>() / merged_scores.len() as f64
    };
    let avg_source = if source_scores.is_empty() {
        0.0
    } else {
        source_scores.iter().sum::<f64>() / source_scores.len() as f64
    };
    let degradation = avg_source - avg_merged;

    MergeValidation {
        passed: degradation < degradation_tolerance,
        avg_source_score: avg_source,
        avg_merged_score: avg_merged,
        degradation,
        queries_tested: queries,
    }
}

/// Runs compaction cycles — merging low-value memories into generalisations.
pub struct CompactionEngine<'a> {
    sqlite: &'a SQLiteStore,
    graph: &'a GraphStore,
    vector: &'a VectorStore,
    text_embedder: Option<&'a dyn TextEmbedderTrait>,
    llm: &'a LlmClient,
    config: &'a MemoryConfig,
}

impl<'a> CompactionEngine<'a> {
    pub fn new(
        sqlite: &'a SQLiteStore,
        graph: &'a GraphStore,
        vector: &'a VectorStore,
        text_embedder: Option<&'a dyn TextEmbedderTrait>,
        llm: &'a LlmClient,
        config: &'a MemoryConfig,
    ) -> Self {
        Self {
            sqlite,
            graph,
            vector,
            text_embedder,
            llm,
            config,
        }
    }

    /// Execute a full compaction cycle.
    ///
    /// Steps:
    ///   1. Select candidates from hot tier
    ///   2. Filter by hard exclusions (graph edge count)
    ///   3. Group by keyword overlap
    ///   4. Merge eligible groups (with A2.4 validation)
    ///   5. Update tiers
    ///   6. Log the run
    pub async fn run(
        &self,
        trigger: &str,
    ) -> Result<CompactionResult, Box<dyn std::error::Error + Send + Sync>> {
        let mut result = CompactionResult {
            trigger: trigger.to_string(),
            ..Default::default()
        };

        // Get candidates from SQLite
        let candidates = self
            .sqlite
            .get_compaction_candidates(self.config.compaction_candidate_threshold)
            .await?;
        result.memories_reviewed = candidates.len() as i64;

        if candidates.is_empty() {
            info!("Compaction: no candidates found");
            self.sqlite.log_compaction_run(&result).await?;
            return Ok(result);
        }

        // Filter by graph edge count (structurally important anchors)
        let mut filtered = Vec::new();
        for mem in &candidates {
            if mem.graph_node_id.is_some() {
                let edge_count = self.graph.get_edge_count(&mem.id).await.unwrap_or(0);
                if edge_count > 3 {
                    continue;
                }
            }
            filtered.push(mem.clone());
        }

        // Group by keyword overlap
        let groups = group_by_keywords(&filtered, self.config.keyword_overlap_merge_threshold);

        let mut merged_count = 0i64;
        for group_indices in &groups {
            let group: Vec<Memory> = group_indices.iter().map(|&i| filtered[i].clone()).collect();
            if !can_merge(&group, self.config) {
                continue;
            }
            if let Some(_new_mem) = self.merge_group(&group, &result.id).await {
                merged_count += 1;
            }
        }

        result.memories_merged = merged_count;

        // Tier promotion: move hot memories exceeding threshold to warm
        let hot_count = self.sqlite.count_memories(Some("hot")).await?;
        let threshold = self.config.hot_tier_threshold as i64;
        if hot_count > threshold {
            self.promote_tier((hot_count - threshold) as i64).await;
        }

        result.notes = Some(format!(
            "Reviewed {}, merged into {} semantic memories",
            result.memories_reviewed, merged_count
        ));
        self.sqlite.log_compaction_run(&result).await?;

        info!(
            "Compaction complete: reviewed={}, merged={}",
            result.memories_reviewed, merged_count
        );
        Ok(result)
    }

    /// Merge a group of memories into a single semantic memory.
    async fn merge_group(&self, group: &[Memory], compaction_id: &str) -> Option<Memory> {
        let sources: String = group
            .iter()
            .enumerate()
            .map(|(i, m)| {
                format!(
                    "Memory {} (salience={}, valence={}):\n<memory_{}>\n{}\n</memory_{}>",
                    i + 1,
                    m.salience,
                    m.valence,
                    i + 1,
                    m.content,
                    i + 1
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let prompt = format!(
            "Merge these {} related memories into a single generalised memory:\n\n{}\n\nRespond with JSON only.",
            group.len(),
            sources
        );

        let result = match self
            .llm
            .complete_json(&prompt, Some(&self.config.prompts.merge), None, None)
            .await
        {
            Ok(r) => r,
            Err(e) => {
                error!("LLM merge failed for group of {} memories: {e}", group.len());
                return None;
            }
        };

        let merged_content = result
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // A2.4: Validate merge via generative replay
        if let Some(text_embedder) = self.text_embedder {
            let validation = validate_merge(
                group,
                &merged_content,
                text_embedder,
                self.llm,
                self.config,
                self.config.merge_validation_queries,
                self.config.merge_degradation_tolerance,
            )
            .await;

            if !validation.passed {
                info!(
                    "Merge validation failed (degradation={:.3}), skipping group of {} memories",
                    validation.degradation,
                    group.len()
                );
                return None;
            }
        }

        let now = Utc::now().to_rfc3339();
        let new_id = Uuid::new_v4().to_string();
        let max_gen = group.iter().map(|m| m.compaction_gen).max().unwrap_or(0);

        let keywords: Vec<(String, f64)> = result
            .get("keywords")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|kw| {
                        let keyword = kw.get("keyword")?.as_str()?.to_lowercase();
                        let weight = kw.get("weight").and_then(|w| w.as_f64()).unwrap_or(1.0);
                        Some((keyword, weight))
                    })
                    .take(self.config.max_keywords_per_memory)
                    .collect()
            })
            .unwrap_or_default();

        let new_mem = Memory {
            id: new_id.clone(),
            created_at: now.clone(),
            updated_at: now.clone(),
            content: merged_content.clone(),
            summary: result.get("summary").and_then(|v| v.as_str()).map(String::from),
            raw_log_id: group[0].raw_log_id.clone(),
            session_id: group[0].session_id.clone(),
            turn: group[0].turn,
            valence: result.get("valence").and_then(|v| v.as_f64()).unwrap_or(0.0),
            arousal: result.get("arousal").and_then(|v| v.as_f64()).unwrap_or(0.0),
            salience: result.get("salience").and_then(|v| v.as_f64()).unwrap_or(0.5),
            compaction_gen: max_gen + 1,
            tier: "warm".to_string(),
            is_semantic: true,
            keywords,
            ..Default::default()
        };

        // Save to SQLite
        if let Err(e) = self.sqlite.save_memory(&new_mem).await {
            error!("Failed to save merged memory: {e}");
            return None;
        }

        // Create graph node for new memory
        let _ = self
            .graph
            .add_memory_node(
                &new_id,
                new_mem.summary.as_deref().unwrap_or(""),
                "warm",
                new_mem.salience,
                new_mem.valence,
                new_mem.compaction_gen,
                &now,
            )
            .await;
        let _ = self.sqlite.update_memory_graph_ref(&new_id, &new_id).await;

        // Create EVOLVED_FROM edges and replicate RELATES_TO edges
        let source_ids: Vec<String> = group.iter().map(|m| m.id.clone()).collect();
        for src in group {
            let _ = self
                .graph
                .add_evolved_from(&new_id, &src.id, compaction_id, &now)
                .await;
        }

        let _ = self
            .graph
            .replicate_edges_to_new_node(&source_ids, &new_id)
            .await;

        // Move source memories to cold tier
        for src in group {
            let _ = self.sqlite.update_memory_tier(&src.id, "cold").await;
            if src.graph_node_id.is_some() {
                let _ = self.graph.update_memory_tier(&src.id, "cold").await;
            }
        }

        // Create text embedding for new memory
        if let Some(text_embedder) = self.text_embedder {
            if let Ok(vector) = text_embedder.embed(&new_mem.content).await {
                if let Ok(point_id) = self
                    .vector
                    .upsert_text_vector(
                        &new_id,
                        vector,
                        "warm",
                        new_mem.valence,
                        new_mem.arousal,
                        &new_mem.session_id,
                        &now,
                    )
                    .await
                {
                    let _ = self.sqlite.update_memory_vector_ref(&new_id, &point_id).await;
                }
            }
        }

        // Log the merge
        let _ = self
            .sqlite
            .log_compaction_merge(compaction_id, &source_ids, &new_id, None, None, None, None)
            .await;

        Some(new_mem)
    }

    /// Move the oldest/lowest-decay hot memories to warm tier.
    async fn promote_tier(&self, count: i64) {
        let memories = match self.sqlite.list_memories(Some("hot"), count, 0).await {
            Ok(m) => m,
            Err(_) => return,
        };

        for mem in &memories {
            let _ = self.sqlite.update_memory_tier(&mem.id, "warm").await;
            if mem.graph_node_id.is_some() {
                let _ = self.graph.update_memory_tier(&mem.id, "warm").await;
            }
        }
    }
}
