//! Data models for the Agent Memory System.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

fn default_uuid() -> String {
    Uuid::new_v4().to_string()
}

fn default_now() -> String {
    Utc::now().to_rfc3339()
}

/// Raw log entry — immutable record of each agent output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawLogEntry {
    #[serde(default = "default_uuid")]
    pub id: String,
    #[serde(default)]
    pub session_id: String,
    #[serde(default)]
    pub turn: i64,
    #[serde(default = "default_now")]
    pub timestamp: String,
    #[serde(default = "default_role")]
    pub role: String,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub token_count: i64,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub provider: String,
}

fn default_role() -> String {
    "assistant".to_string()
}

impl Default for RawLogEntry {
    fn default() -> Self {
        Self {
            id: default_uuid(),
            session_id: String::new(),
            turn: 0,
            timestamp: default_now(),
            role: "assistant".to_string(),
            content: String::new(),
            token_count: 0,
            model: String::new(),
            provider: String::new(),
        }
    }
}

/// Core memory record with emotional metadata, compaction state, and cross-store references.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    #[serde(default = "default_uuid")]
    pub id: String,
    #[serde(default = "default_now")]
    pub created_at: String,
    #[serde(default = "default_now")]
    pub updated_at: String,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub raw_log_id: String,
    #[serde(default)]
    pub session_id: String,
    #[serde(default)]
    pub turn: i64,

    // Emotional metadata
    #[serde(default)]
    pub valence: f64,
    #[serde(default)]
    pub arousal: f64,
    #[serde(default)]
    pub surprise: f64,

    // Salience and access
    #[serde(default = "default_salience")]
    pub salience: f64,
    #[serde(default)]
    pub access_count: i64,
    #[serde(default)]
    pub last_accessed: Option<String>,
    #[serde(default = "default_decay_score")]
    pub decay_score: f64,

    // Compaction state
    #[serde(default)]
    pub compaction_gen: i32,
    #[serde(default = "default_tier")]
    pub tier: String,
    #[serde(default)]
    pub fast_pathed: bool,
    #[serde(default)]
    pub is_semantic: bool,

    // Cross-store references
    #[serde(default)]
    pub graph_node_id: Option<String>,
    #[serde(default)]
    pub vector_id: Option<String>,

    // Visual layer
    #[serde(default)]
    pub spatial_embedding: Option<Vec<u8>>,
    #[serde(default)]
    pub scene_description: Option<String>,

    // Keywords (not stored in main table — separate table)
    #[serde(default)]
    pub keywords: Vec<(String, f64)>,
}

fn default_salience() -> f64 {
    0.5
}

fn default_decay_score() -> f64 {
    1.0
}

fn default_tier() -> String {
    "hot".to_string()
}

impl Default for Memory {
    fn default() -> Self {
        Self {
            id: default_uuid(),
            created_at: default_now(),
            updated_at: default_now(),
            content: String::new(),
            summary: None,
            raw_log_id: String::new(),
            session_id: String::new(),
            turn: 0,
            valence: 0.0,
            arousal: 0.0,
            surprise: 0.0,
            salience: 0.5,
            access_count: 0,
            last_accessed: None,
            decay_score: 1.0,
            compaction_gen: 0,
            tier: "hot".to_string(),
            fast_pathed: false,
            is_semantic: false,
            graph_node_id: None,
            vector_id: None,
            spatial_embedding: None,
            scene_description: None,
            keywords: Vec::new(),
        }
    }
}

/// Save decision log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveDecision {
    #[serde(default = "default_uuid")]
    pub id: String,
    #[serde(default)]
    pub raw_log_id: String,
    #[serde(default)]
    pub session_id: String,
    #[serde(default)]
    pub turn: i64,
    #[serde(default = "default_now")]
    pub decided_at: String,
    #[serde(default = "default_decision")]
    pub decision: String,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub confidence: f64,
    // A2.1 — retrieval gap awareness
    #[serde(default)]
    pub gap_triggered: bool,
    #[serde(default)]
    pub threshold_used: Option<f64>,
}

fn default_decision() -> String {
    "skip".to_string()
}

impl Default for SaveDecision {
    fn default() -> Self {
        Self {
            id: default_uuid(),
            raw_log_id: String::new(),
            session_id: String::new(),
            turn: 0,
            decided_at: default_now(),
            decision: "skip".to_string(),
            reason: None,
            confidence: 0.0,
            gap_triggered: false,
            threshold_used: None,
        }
    }
}

/// Compaction run result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionResult {
    #[serde(default = "default_uuid")]
    pub id: String,
    #[serde(default = "default_now")]
    pub ran_at: String,
    #[serde(default = "default_trigger")]
    pub trigger: String,
    #[serde(default)]
    pub memories_reviewed: i64,
    #[serde(default)]
    pub memories_merged: i64,
    #[serde(default)]
    pub memories_pruned: i64,
    #[serde(default)]
    pub notes: Option<String>,
    // A2.5 / A3 — addendum tracking
    #[serde(default)]
    pub keywords_updated: i64,
    #[serde(default)]
    pub edges_discovered: i64,
}

fn default_trigger() -> String {
    "scheduled".to_string()
}

impl Default for CompactionResult {
    fn default() -> Self {
        Self {
            id: default_uuid(),
            ran_at: default_now(),
            trigger: "scheduled".to_string(),
            memories_reviewed: 0,
            memories_merged: 0,
            memories_pruned: 0,
            notes: None,
            keywords_updated: 0,
            edges_discovered: 0,
        }
    }
}

/// Result of generative replay validation for a compaction merge (A2.4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeValidation {
    pub passed: bool,
    pub avg_source_score: f64,
    pub avg_merged_score: f64,
    pub degradation: f64,
    pub queries_tested: Vec<String>,
}

impl Default for MergeValidation {
    fn default() -> Self {
        Self {
            passed: true,
            avg_source_score: 0.0,
            avg_merged_score: 0.0,
            degradation: 0.0,
            queries_tested: Vec::new(),
        }
    }
}

/// An edge discovered during exploratory graph walks (A3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredEdge {
    #[serde(default)]
    pub source_id: String,
    #[serde(default)]
    pub target_id: String,
    #[serde(default)]
    pub similarity: f64,
    #[serde(default)]
    pub relationship_type: String,
    /// random_walk | cluster_bridge | temporal_proximity
    #[serde(default)]
    pub discovery_method: String,
}

impl Default for DiscoveredEdge {
    fn default() -> Self {
        Self {
            source_id: String::new(),
            target_id: String::new(),
            similarity: 0.0,
            relationship_type: String::new(),
            discovery_method: String::new(),
        }
    }
}
