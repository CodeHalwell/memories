//! Custom error types for the Agent Memory System.

use thiserror::Error;

/// Top-level error type for the Agent Memory System.
#[derive(Error, Debug)]
pub enum AgentMemoryError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] sqlx::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("HTTP request error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("TOML parsing error: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("LLM call failed: {0}")]
    LlmError(String),

    #[error("Embedding error: {0}")]
    EmbeddingError(String),

    #[error("Graph store error: {0}")]
    GraphError(String),

    #[error("Vector store error: {0}")]
    VectorError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Not initialized: {0}")]
    NotInitialized(String),

    #[error("{0}")]
    Other(String),
}
