//! Policy controller interface (A5 — stub for v2).
//!
//! This module defines the interface for a learned memory policy that will
//! eventually replace fixed heuristics for save, retrieval, and compaction
//! decisions. For now, it delegates to the existing heuristic logic.
//!
//! The hard constraints defined here cannot be overridden by any learned policy.

use serde::{Deserialize, Serialize};

use crate::config::MemoryConfig;

/// Hard constraints that no learned policy may override.
#[derive(Debug, Clone)]
pub struct PolicyHardConstraints {
    // Save constraints
    pub min_save_rate: f64,
    pub max_save_rate: f64,
    pub fast_path_override: bool,

    // Retrieval constraints
    pub min_layers: usize,
    pub max_graph_depth: usize,
    pub max_top_k: usize,

    // Compaction constraints
    pub never_delete_raw_logs: bool,
    pub never_compact_fast_path_gen0: bool,
    pub require_merge_validation: bool,
}

impl Default for PolicyHardConstraints {
    fn default() -> Self {
        Self {
            min_save_rate: 0.05,
            max_save_rate: 0.50,
            fast_path_override: true,
            min_layers: 1,
            max_graph_depth: 4,
            max_top_k: 20,
            never_delete_raw_logs: true,
            never_compact_fast_path_gen0: true,
            require_merge_validation: true,
        }
    }
}

/// State vector for policy decisions (A5.2).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PolicyState {
    pub turn_number: i64,
    pub session_length: i64,
    pub time_since_last_save: f64,
    pub content_length: usize,
    pub emotional_valence: f64,
    pub emotional_arousal: f64,
    pub emotional_surprise: f64,
    pub hot_tier_count: i64,
    pub recent_retrieval_hit_rate: f64,
    pub retrieval_gap_score: f64,
    pub graph_node_count: i64,
    pub avg_edge_degree: f64,
    pub orphan_memory_count: i64,
    pub days_since_last_compaction: f64,
    pub pending_merge_candidates: i64,
}

/// Stub policy controller — uses heuristics, logs decisions for future training.
///
/// In v2, this will be replaced with a learned model (gradient-boosted trees
/// or a small RL-trained policy network).
pub struct PolicyController {
    pub constraints: PolicyHardConstraints,
    config: MemoryConfig,
}

impl PolicyController {
    pub fn new(config: &MemoryConfig) -> Self {
        Self {
            constraints: PolicyHardConstraints::default(),
            config: config.clone(),
        }
    }

    /// Decide whether to save (heuristic — v1).
    pub fn should_save(&self, _state: &PolicyState, llm_confidence: f64) -> bool {
        llm_confidence >= self.config.save_confidence_threshold
    }

    /// Return retrieval parameters (heuristic — v1).
    pub fn retrieval_config(&self, _state: &PolicyState) -> RetrievalConfig {
        RetrievalConfig {
            layers: self.config.retrieval_layers.clone(),
            graph_depth: self
                .config
                .graph_traversal_depth
                .min(self.constraints.max_graph_depth),
            mood_weight: self.config.mood_congruent_weight,
            top_k: self
                .config
                .top_k_per_layer
                .min(self.constraints.max_top_k),
        }
    }

    /// Return compaction urgency score (heuristic — v1).
    pub fn compaction_priority(&self, state: &PolicyState) -> f64 {
        if state.hot_tier_count > self.config.hot_tier_threshold as i64 {
            return 1.0;
        }
        state.hot_tier_count as f64 / (self.config.hot_tier_threshold as f64).max(1.0)
    }
}

/// Retrieval configuration returned by the policy controller.
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    pub layers: Vec<String>,
    pub graph_depth: usize,
    pub mood_weight: f64,
    pub top_k: usize,
}
