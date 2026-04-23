//! Text embedding via HTTP API.
//!
//! Calls an OpenAI-compatible embedding API endpoint to produce vector
//! representations of text. Used for semantic similarity search in the
//! vector store.

use async_trait::async_trait;
use log::info;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::config::MemoryConfig;

/// Trait for text embedding backends.
#[async_trait]
pub trait TextEmbedderTrait: Send + Sync {
    /// Embed a single text string.
    async fn embed(&self, text: &str) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>>;

    /// Embed multiple texts.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error + Send + Sync>>;

    /// Return the dimension of the embeddings.
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

/// HTTP-based text embedder using an OpenAI-compatible embedding API.
pub struct TextEmbedder {
    model_name: String,
    api_url: String,
    api_key: Option<String>,
    client: Client,
    dim: usize,
}

impl TextEmbedder {
    pub fn new(config: &MemoryConfig, api_url: &str, api_key: Option<String>, dim: usize) -> Self {
        info!("Initializing text embedder with model: {}", config.text_embedding_model);
        Self {
            model_name: config.text_embedding_model.clone(),
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

    async fn call_api(&self, input: Vec<String>) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error + Send + Sync>> {
        let body = EmbeddingRequest {
            model: self.model_name.clone(),
            input,
        };

        let mut req = self.client.post(&self.api_url).json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let resp: EmbeddingResponse = req.send().await?.json().await?;
        Ok(resp.data.into_iter().map(|d| d.embedding).collect())
    }
}

#[async_trait]
impl TextEmbedderTrait for TextEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        let results = self.call_api(vec![text.to_string()]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| "Empty embedding response".into())
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error + Send + Sync>> {
        self.call_api(texts.to_vec()).await
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}
