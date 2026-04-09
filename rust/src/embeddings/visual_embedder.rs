//! Visual/spatial embedding via HTTP API (CLIP-like).
//!
//! Embeds scene descriptions (text) using a CLIP-compatible API to provide
//! an independent retrieval channel based on spatial/perceptual similarity.

use async_trait::async_trait;
use log::info;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::config::MemoryConfig;

/// Trait for visual embedding backends.
#[async_trait]
pub trait VisualEmbedderTrait: Send + Sync {
    /// Embed a scene description text. Returns a list of floats.
    async fn embed(&self, text: &str) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>>;

    /// Embed and return raw bytes for storage in SQLite BLOB column.
    async fn embed_to_bytes(&self, text: &str) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>>;

    /// Convert raw bytes back to float vector.
    fn bytes_to_vector(&self, data: &[u8]) -> Vec<f64>;

    /// Return the embedding dimension.
    fn dimension(&self) -> usize;
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f64>,
}

/// HTTP-based visual embedder using a CLIP-compatible embedding API.
pub struct VisualEmbedder {
    model_name: String,
    api_url: String,
    api_key: Option<String>,
    client: Client,
    dim: usize,
}

impl VisualEmbedder {
    pub fn new(config: &MemoryConfig, api_url: &str, api_key: Option<String>, dim: usize) -> Self {
        info!("Initializing visual embedder with model: {}", config.clip_model);
        Self {
            model_name: config.clip_model.clone(),
            api_url: api_url.to_string(),
            api_key,
            client: Client::new(),
            dim,
        }
    }

    pub fn with_model(model_name: &str, api_url: &str, api_key: Option<String>, dim: usize) -> Self {
        Self {
            model_name: model_name.to_string(),
            api_url: api_url.to_string(),
            api_key,
            client: Client::new(),
            dim,
        }
    }

    async fn call_api(&self, text: &str) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        let body = EmbeddingRequest {
            model: self.model_name.clone(),
            input: vec![text.to_string()],
        };

        let mut req = self.client.post(&self.api_url).json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let resp: EmbeddingResponse = req.send().await?.json().await?;
        resp.data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .ok_or_else(|| "Empty embedding response".into())
    }
}

#[async_trait]
impl VisualEmbedderTrait for VisualEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        self.call_api(text).await
    }

    async fn embed_to_bytes(&self, text: &str) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        let floats = self.embed(text).await?;
        let mut bytes = Vec::with_capacity(floats.len() * 4);
        for f in &floats {
            bytes.extend_from_slice(&(*f as f32).to_le_bytes());
        }
        Ok(bytes)
    }

    fn bytes_to_vector(&self, data: &[u8]) -> Vec<f64> {
        data.chunks_exact(4)
            .map(|chunk| {
                let arr: [u8; 4] = chunk.try_into().unwrap();
                f32::from_le_bytes(arr) as f64
            })
            .collect()
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}
