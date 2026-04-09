//! Qdrant vector store for semantic similarity search via REST API.
//!
//! Uses HTTP calls to Qdrant REST API. Manages two collections:
//!   - memory_text: sentence-transformer embeddings of memory content
//!   - memory_visual: CLIP embeddings of scene descriptions

use log::debug;
use reqwest::Client;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const TEXT_COLLECTION: &str = "memory_text";
pub const VISUAL_COLLECTION: &str = "memory_visual";

/// Qdrant REST API-backed vector store for memory embeddings.
pub struct VectorStore {
    base_url: String,
    client: Client,
}

#[derive(Debug, Serialize)]
struct UpsertRequest {
    points: Vec<PointStruct>,
}

#[derive(Debug, Serialize)]
struct PointStruct {
    id: String,
    vector: Vec<f64>,
    payload: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct SearchRequest {
    vector: Vec<f64>,
    limit: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    filter: Option<serde_json::Value>,
    with_payload: bool,
}

#[derive(Debug, Deserialize)]
struct SearchResponse {
    result: Vec<SearchHit>,
}

#[derive(Debug, Deserialize)]
struct SearchHit {
    #[allow(dead_code)]
    id: serde_json::Value,
    score: f64,
    payload: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct CreateCollectionRequest {
    vectors: VectorParams,
}

#[derive(Debug, Serialize)]
struct VectorParams {
    size: usize,
    distance: String,
}

#[derive(Debug, Deserialize)]
struct RetrieveResponse {
    result: Vec<RetrievePoint>,
}

#[derive(Debug, Deserialize)]
struct RetrievePoint {
    #[allow(dead_code)]
    id: serde_json::Value,
    vector: Option<Vec<f64>>,
}

#[derive(Debug, Serialize)]
struct DeleteRequest {
    filter: serde_json::Value,
}

impl VectorStore {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client: Client::new(),
        }
    }

    /// Initialize and ensure collections exist.
    pub async fn initialize(
        &self,
        text_dim: usize,
        visual_dim: usize,
    ) -> Result<(), reqwest::Error> {
        self.ensure_collection(TEXT_COLLECTION, text_dim).await?;
        self.ensure_collection(VISUAL_COLLECTION, visual_dim).await?;
        Ok(())
    }

    async fn ensure_collection(
        &self,
        name: &str,
        dim: usize,
    ) -> Result<(), reqwest::Error> {
        let url = format!("{}/collections/{name}", self.base_url);

        // Check if collection exists
        let resp = self.client.get(&url).send().await?;
        match resp.status() {
            StatusCode::OK => return Ok(()),
            StatusCode::NOT_FOUND => {}
            _ => {
                resp.error_for_status()?;
            }
        }

        // Create collection
        let body = CreateCollectionRequest {
            vectors: VectorParams {
                size: dim,
                distance: "Cosine".to_string(),
            },
        };
        self.client
            .put(&url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    // ── Text embeddings ──

    /// Insert or update a text embedding. Returns the point ID.
    pub async fn upsert_text_vector(
        &self,
        memory_id: &str,
        vector: Vec<f64>,
        tier: &str,
        valence: f64,
        arousal: f64,
        session_id: &str,
        created_at: &str,
    ) -> Result<String, reqwest::Error> {
        let point_id = Uuid::new_v4().to_string();
        let payload = serde_json::json!({
            "memory_id": memory_id,
            "tier": tier,
            "valence": valence,
            "arousal": arousal,
            "session_id": session_id,
            "created_at": created_at,
        });

        let body = UpsertRequest {
            points: vec![PointStruct {
                id: point_id.clone(),
                vector,
                payload,
            }],
        };

        let url = format!("{}/collections/{TEXT_COLLECTION}/points", self.base_url);
        self.client
            .put(&url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;
        Ok(point_id)
    }

    /// Search for nearest text embeddings.
    pub async fn search_text(
        &self,
        query_vector: Vec<f64>,
        limit: usize,
        tier_filter: Option<&str>,
    ) -> Result<Vec<TextSearchResult>, reqwest::Error> {
        let filter = tier_filter.map(|tier| {
            serde_json::json!({
                "must": [{"key": "tier", "match": {"value": tier}}]
            })
        });

        let body = SearchRequest {
            vector: query_vector,
            limit,
            filter,
            with_payload: true,
        };

        let url = format!("{}/collections/{TEXT_COLLECTION}/points/search", self.base_url);
        let resp: SearchResponse = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        Ok(resp
            .result
            .iter()
            .map(|hit| {
                let payload = hit.payload.as_ref();
                TextSearchResult {
                    memory_id: payload
                        .and_then(|p| p.get("memory_id"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    score: hit.score,
                    tier: payload
                        .and_then(|p| p.get("tier"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("hot")
                        .to_string(),
                    valence: payload
                        .and_then(|p| p.get("valence"))
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0),
                    arousal: payload
                        .and_then(|p| p.get("arousal"))
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0),
                }
            })
            .collect())
    }

    // ── Visual embeddings ──

    /// Insert or update a visual (CLIP) embedding. Returns the point ID.
    pub async fn upsert_visual_vector(
        &self,
        memory_id: &str,
        vector: Vec<f64>,
        session_id: &str,
        created_at: &str,
    ) -> Result<String, reqwest::Error> {
        let point_id = Uuid::new_v4().to_string();
        let payload = serde_json::json!({
            "memory_id": memory_id,
            "session_id": session_id,
            "created_at": created_at,
        });

        let body = UpsertRequest {
            points: vec![PointStruct {
                id: point_id.clone(),
                vector,
                payload,
            }],
        };

        let url = format!("{}/collections/{VISUAL_COLLECTION}/points", self.base_url);
        self.client
            .put(&url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;
        Ok(point_id)
    }

    /// Search for nearest visual embeddings.
    pub async fn search_visual(
        &self,
        query_vector: Vec<f64>,
        limit: usize,
    ) -> Result<Vec<VisualSearchResult>, reqwest::Error> {
        let body = SearchRequest {
            vector: query_vector,
            limit,
            filter: None,
            with_payload: true,
        };

        let url = format!(
            "{}/collections/{VISUAL_COLLECTION}/points/search",
            self.base_url
        );
        let resp: SearchResponse = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        Ok(resp
            .result
            .iter()
            .map(|hit| {
                let payload = hit.payload.as_ref();
                VisualSearchResult {
                    memory_id: payload
                        .and_then(|p| p.get("memory_id"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    score: hit.score,
                }
            })
            .collect())
    }

    /// Compute cosine similarity between two points in the text collection.
    /// Used by dream explorer (A3) for cross-session similarity checks.
    pub async fn similarity(
        &self,
        point_id_a: &str,
        point_id_b: &str,
    ) -> Result<Option<f64>, reqwest::Error> {
        let url = format!(
            "{}/collections/{TEXT_COLLECTION}/points",
            self.base_url
        );

        let body = serde_json::json!({
            "ids": [point_id_a, point_id_b],
            "with_vector": true
        });

        let resp: RetrieveResponse = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        if resp.result.len() < 2 {
            return Ok(None);
        }

        let a = match &resp.result[0].vector {
            Some(v) => v,
            None => return Ok(None),
        };
        let b = match &resp.result[1].vector {
            Some(v) => v,
            None => return Ok(None),
        };

        Ok(Some(cosine_similarity(a, b)))
    }

    /// Delete all points for a given memory_id from a collection.
    pub async fn delete_point(
        &self,
        collection: &str,
        memory_id: &str,
    ) -> Result<(), reqwest::Error> {
        let url = format!(
            "{}/collections/{collection}/points/delete",
            self.base_url
        );

        let body = DeleteRequest {
            filter: serde_json::json!({
                "must": [{"key": "memory_id", "match": {"value": memory_id}}]
            }),
        };

        self.client
            .post(&url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;
        debug!("Deleted points for memory_id={memory_id} from {collection}");
        Ok(())
    }
}

/// Compute cosine similarity between two vectors.
///
/// Returns 0.0 if vectors are empty or have different lengths.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm = norm_a * norm_b;
    if norm > 0.0 {
        dot / norm
    } else {
        0.0
    }
}

/// Result from text embedding search.
#[derive(Debug, Clone)]
pub struct TextSearchResult {
    pub memory_id: String,
    pub score: f64,
    pub tier: String,
    pub valence: f64,
    pub arousal: f64,
}

/// Result from visual embedding search.
#[derive(Debug, Clone)]
pub struct VisualSearchResult {
    pub memory_id: String,
    pub score: f64,
}
