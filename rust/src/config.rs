//! Configuration for the Agent Memory System.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::prompts;

/// Base data directory — all runtime data stored here.
pub fn default_data_dir() -> PathBuf {
    PathBuf::from("data")
}

pub fn default_log_dir() -> PathBuf {
    default_data_dir().join("logs").join("sessions")
}

pub fn default_db_path() -> PathBuf {
    default_data_dir().join("memory.db")
}

pub fn default_graph_dir() -> PathBuf {
    default_data_dir().join("graph")
}

pub fn default_vector_dir() -> PathBuf {
    default_data_dir().join("vectors")
}

pub fn default_policy_data_dir() -> PathBuf {
    default_data_dir().join("policy_data")
}

/// Prompt configuration mapping.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PromptsConfig {
    pub scene_description: String,
    pub save_decision: String,
    pub merge: String,
    pub synthetic_query: String,
    pub emotion: String,
    pub classify_relationship: String,
}

impl Default for PromptsConfig {
    fn default() -> Self {
        Self {
            scene_description: prompts::SCENE_DESCRIPTION_SYSTEM.to_string(),
            save_decision: prompts::SAVE_DECISION_SYSTEM.to_string(),
            merge: prompts::MERGE_SYSTEM.to_string(),
            synthetic_query: prompts::SYNTHETIC_QUERY_SYSTEM.to_string(),
            emotion: prompts::EMOTION_SYSTEM.to_string(),
            classify_relationship: prompts::CLASSIFY_RELATIONSHIP_SYSTEM.to_string(),
        }
    }
}

/// Central configuration for the memory system.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    // LLM
    pub llm_model: String,
    pub llm_temperature: f64,

    // Save thresholds
    pub save_confidence_threshold: f64,
    pub fast_path_arousal: f64,
    pub fast_path_surprise: f64,
    pub max_keywords_per_memory: usize,

    // Save decision — retrieval gap awareness (A2.1)
    pub gap_lookback_turns: usize,
    pub gap_overlap_threshold: f64,
    pub gap_threshold_reduction: f64,

    // Retrieval
    pub retrieval_layers: Vec<String>,
    pub graph_traversal_depth: usize,
    pub mood_congruent_weight: f64,
    pub top_k_per_layer: usize,

    // Compaction
    pub hot_tier_threshold: usize,
    pub compaction_candidate_threshold: f64,
    pub keyword_overlap_merge_threshold: f64,
    pub valence_merge_exclusion_delta: f64,

    // Compaction — merge validation (A2.4)
    pub merge_validation_queries: usize,
    pub merge_degradation_tolerance: f64,

    // Compaction — generation gap guard (A2.3)
    pub max_generation_gap_for_merge: i32,

    // Decay
    pub decay_recency_weight: f64,
    pub decay_frequency_weight: f64,
    pub decay_halflife_days: f64,

    // Visual layer
    pub visual_salience_threshold: f64,
    pub clip_model: String,

    // Embeddings
    pub text_embedding_model: String,

    // Dream exploration (A3)
    pub dream_walk_count: usize,
    pub dream_similarity_threshold: f64,
    pub dream_max_new_edges: usize,
    pub dream_cluster_min_size: usize,
    pub dream_enabled: bool,

    // Policy logging (A4)
    pub policy_logging_enabled: bool,
    pub save_outcome_lookback_days: i64,
    pub retrieval_outcome_followup_turns: i32,
    pub retrieval_outcome_keyword_overlap: f64,

    // Policy training (A5, v2)
    pub policy_min_save_examples: usize,
    pub policy_min_retrieval_examples: usize,

    // System prompts
    pub prompts: PromptsConfig,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            llm_model: "claude-sonnet-4-6".to_string(),
            llm_temperature: 0.2,

            save_confidence_threshold: 0.5,
            fast_path_arousal: 0.85,
            fast_path_surprise: 0.75,
            max_keywords_per_memory: 10,

            gap_lookback_turns: 20,
            gap_overlap_threshold: 0.3,
            gap_threshold_reduction: 0.7,

            retrieval_layers: vec![
                "grep".to_string(),
                "keyword".to_string(),
                "semantic".to_string(),
            ],
            graph_traversal_depth: 2,
            mood_congruent_weight: 0.2,
            top_k_per_layer: 5,

            hot_tier_threshold: 500,
            compaction_candidate_threshold: 0.7,
            keyword_overlap_merge_threshold: 0.6,
            valence_merge_exclusion_delta: 0.6,

            merge_validation_queries: 5,
            merge_degradation_tolerance: 0.15,

            max_generation_gap_for_merge: 1,

            decay_recency_weight: 0.6,
            decay_frequency_weight: 0.4,
            decay_halflife_days: 7.0,

            visual_salience_threshold: 0.7,
            clip_model: "ViT-B-32".to_string(),

            text_embedding_model: "all-MiniLM-L6-v2".to_string(),

            dream_walk_count: 50,
            dream_similarity_threshold: 0.7,
            dream_max_new_edges: 20,
            dream_cluster_min_size: 3,
            dream_enabled: true,

            policy_logging_enabled: true,
            save_outcome_lookback_days: 30,
            retrieval_outcome_followup_turns: 3,
            retrieval_outcome_keyword_overlap: 0.5,

            policy_min_save_examples: 1000,
            policy_min_retrieval_examples: 500,

            prompts: PromptsConfig::default(),
        }
    }
}

impl MemoryConfig {
    /// Parse configuration from a TOML string, using defaults for missing fields.
    pub fn from_toml(toml_str: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(toml_str)
    }

    /// Parse configuration from a TOML file.
    pub fn from_file(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let config = Self::from_toml(&contents)?;
        Ok(config)
    }
}
