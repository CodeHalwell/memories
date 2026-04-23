//! Agent Memory System — local-first, cognitively-inspired memory for AI agents.

pub mod config;
pub mod core;
pub mod embeddings;
pub mod emotion;
pub mod error;
pub mod llm;
pub mod models;
pub mod policy;
pub mod prompts;
pub mod storage;

pub use error::AgentMemoryError;
