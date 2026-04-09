//! LLM client with retry and structured logging.
//!
//! Uses `reqwest` to call OpenAI-compatible HTTP APIs (POST to /v1/chat/completions).
//! Includes retry logic with exponential backoff.

use log::warn;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::MemoryConfig;

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f64,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    content: String,
}

/// HTTP-based LLM client for OpenAI-compatible APIs.
pub struct LlmClient {
    api_url: String,
    api_key: Option<String>,
    default_model: String,
    default_temperature: f64,
    max_retries: usize,
    client: Client,
}

impl LlmClient {
    pub fn new(config: &MemoryConfig, api_url: &str, api_key: Option<String>) -> Self {
        Self {
            api_url: api_url.trim_end_matches('/').to_string(),
            api_key,
            default_model: config.llm_model.clone(),
            default_temperature: config.llm_temperature,
            max_retries: 3,
            client: Client::new(),
        }
    }

    pub fn with_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Send a completion request and return the text response.
    pub async fn complete(
        &self,
        prompt: &str,
        system: Option<&str>,
        model: Option<&str>,
        temperature: Option<f64>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let model = model.unwrap_or(&self.default_model);
        let temperature = temperature.unwrap_or(self.default_temperature);

        let mut messages = Vec::new();
        if let Some(sys) = system {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: sys.to_string(),
            });
        }
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        });

        let request_body = ChatCompletionRequest {
            model: model.to_string(),
            messages,
            temperature,
        };

        let url = format!("{}/v1/chat/completions", self.api_url);

        let mut last_err: Option<Box<dyn std::error::Error + Send + Sync>> = None;

        for attempt in 0..self.max_retries {
            let mut req = self.client.post(&url).json(&request_body);
            if let Some(key) = &self.api_key {
                req = req.bearer_auth(key);
            }

            match req.send().await {
                Ok(resp) => {
                    if resp.status().is_success() {
                        let body: ChatCompletionResponse = resp.json().await?;
                        if let Some(choice) = body.choices.into_iter().next() {
                            return Ok(choice.message.content);
                        }
                        return Ok(String::new());
                    }
                    let status = resp.status();
                    let text = resp.text().await.unwrap_or_default();
                    last_err = Some(format!("LLM API returned {status}: {text}").into());
                }
                Err(e) => {
                    last_err = Some(Box::new(e));
                }
            }

            if attempt < self.max_retries - 1 {
                warn!(
                    "LLM call failed (attempt {}/{}), retrying...",
                    attempt + 1,
                    self.max_retries
                );
                // Exponential backoff: 1s, 2s, 4s, ...
                let delay = std::time::Duration::from_secs(1 << attempt);
                tokio::time::sleep(delay).await;
            }
        }

        Err(last_err.unwrap_or_else(|| "LLM call failed after all retries".into()))
    }

    /// Send a completion request and parse the response as JSON.
    pub async fn complete_json(
        &self,
        prompt: &str,
        system: Option<&str>,
        model: Option<&str>,
        temperature: Option<f64>,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let text = self.complete(prompt, system, model, temperature).await?;

        // Strip markdown code fences if present
        let cleaned = text.trim();
        let cleaned = if cleaned.starts_with("```") {
            let lines: Vec<&str> = cleaned.split('\n').collect();
            let inner: Vec<&str> = lines[1..]
                .iter()
                .filter(|l| !l.trim().starts_with("```"))
                .copied()
                .collect();
            inner.join("\n")
        } else {
            cleaned.to_string()
        };

        let value: Value = serde_json::from_str(&cleaned)?;
        Ok(value)
    }
}
