//! Exploratory graph walks during sleep (A3).
//!
//! During the compaction sleep cycle, performs semi-random walks through the
//! memory graph and vector space to discover non-obvious connections between
//! memories encoded separately. This mimics REM-like stochastic association
//! discovery that prevents overfitting to mundane patterns.
//!
//! Two strategies:
//!   1. Random anchor pairs — sample pairs from different sessions, check
//!      semantic similarity, classify relationship via LLM.
//!   2. Cluster bridges — find memories close in vector space but disconnected
//!      in the graph. These are "latent" connections.

use chrono::Utc;
use log::{debug, info};
use rand::seq::SliceRandom;
use uuid::Uuid;

use crate::config::MemoryConfig;
use crate::llm::client::LlmClient;
use crate::models::DiscoveredEdge;
use crate::storage::graph_store::GraphStore;
use crate::storage::sqlite_store::SQLiteStore;
use crate::storage::vector_store::VectorStore;

/// Ask the LLM to classify the relationship between two memories.
pub async fn classify_relationship(
    mem_a_content: &str,
    mem_b_content: &str,
    config: &MemoryConfig,
    llm: &LlmClient,
) -> String {
    let prompt = format!(
        "Memory A:\n<memory_a>\n{mem_a_content}\n</memory_a>\n\n\
         Memory B:\n<memory_b>\n{mem_b_content}\n</memory_b>\n\n\
         What is the relationship between Memory A and Memory B?"
    );

    let valid = [
        "caused",
        "supports",
        "contradicts",
        "precedes",
        "part_of",
        "analogous",
        "unrelated",
    ];

    match llm
        .complete(
            &prompt,
            Some(&config.prompts.classify_relationship),
            None,
            Some(0.1),
        )
        .await
    {
        Ok(response) => {
            let result = response.trim().to_lowercase();
            if valid.contains(&result.as_str()) {
                result
            } else {
                "unrelated".to_string()
            }
        }
        Err(_) => {
            debug!("Relationship classification failed");
            "unrelated".to_string()
        }
    }
}

/// Perform semi-random walks to discover non-obvious memory connections.
///
/// Returns a list of discovered edges. Caller is responsible for committing
/// them to the graph.
pub async fn exploratory_walk(
    sqlite: &SQLiteStore,
    graph: &GraphStore,
    vector: &VectorStore,
    config: &MemoryConfig,
    llm: &LlmClient,
    n_walks: Option<usize>,
    similarity_threshold: Option<f64>,
    max_new_edges: Option<usize>,
) -> Result<Vec<DiscoveredEdge>, Box<dyn std::error::Error + Send + Sync>> {
    let n_walks = n_walks.unwrap_or(config.dream_walk_count);
    let similarity_threshold = similarity_threshold.unwrap_or(config.dream_similarity_threshold);
    let max_new_edges = max_new_edges.unwrap_or(config.dream_max_new_edges);

    let mut discovered: Vec<DiscoveredEdge> = Vec::new();

    // Get all memories with vector embeddings
    let tiers = vec!["hot".to_string(), "warm".to_string()];
    let all_memories = sqlite.get_memories_with_vectors(Some(&tiers)).await?;

    if all_memories.len() < 2 {
        return Ok(discovered);
    }

    let mut rng = rand::thread_rng();

    // Strategy 1: Random anchor pairs
    for _ in 0..n_walks {
        if discovered.len() >= max_new_edges {
            break;
        }

        let pair: Vec<_> = all_memories.choose_multiple(&mut rng, 2).collect();
        let a = pair[0];
        let b = pair[1];

        let a_session = a
            .get("session_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let b_session = b
            .get("session_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Skip if same session (likely already connected)
        if a_session == b_session {
            continue;
        }

        let a_id = a.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let b_id = b.get("id").and_then(|v| v.as_str()).unwrap_or("");

        // Skip if already connected in graph
        match graph.path_exists(a_id, b_id, 1).await {
            Ok(true) => continue,
            Err(_) => continue,
            _ => {}
        }

        // Check semantic similarity via vector store
        let a_vec_id = a
            .get("vector_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let b_vec_id = b
            .get("vector_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let sim = match vector.similarity(a_vec_id, b_vec_id).await {
            Ok(Some(s)) => s,
            _ => continue,
        };

        if sim < similarity_threshold {
            continue;
        }

        // Load full memories for classification
        let mem_a = match sqlite.get_memory(a_id).await {
            Ok(Some(m)) => m,
            _ => continue,
        };
        let mem_b = match sqlite.get_memory(b_id).await {
            Ok(Some(m)) => m,
            _ => continue,
        };

        let rel_type = classify_relationship(&mem_a.content, &mem_b.content, config, llm).await;
        if rel_type != "unrelated" {
            discovered.push(DiscoveredEdge {
                source_id: a_id.to_string(),
                target_id: b_id.to_string(),
                similarity: sim,
                relationship_type: rel_type,
                discovery_method: "random_walk".to_string(),
            });
        }
    }

    discovered.truncate(max_new_edges);
    Ok(discovered)
}

/// Commit discovered edges to the graph and log the exploration run.
///
/// Returns the number of edges committed.
pub async fn commit_discoveries(
    discoveries: &[DiscoveredEdge],
    graph: &GraphStore,
    sqlite: &SQLiteStore,
    run_id: Option<&str>,
) -> Result<i64, Box<dyn std::error::Error + Send + Sync>> {
    let now = Utc::now().to_rfc3339();
    let run_id = run_id
        .map(String::from)
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let strategies: Vec<String> = discoveries
        .iter()
        .map(|d| d.discovery_method.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let mut committed = 0i64;

    for edge in discoveries {
        let edge_id = Uuid::new_v4().to_string();
        match graph
            .add_relates_to(
                &edge.source_id,
                &edge.target_id,
                edge.similarity,
                &edge.relationship_type,
                &now,
            )
            .await
        {
            Ok(_) => {
                committed += 1;
                let _ = sqlite
                    .log_dream_edge(
                        &edge_id,
                        &run_id,
                        &edge.source_id,
                        &edge.target_id,
                        edge.similarity,
                        &edge.relationship_type,
                        &edge.discovery_method,
                        true,
                    )
                    .await;
            }
            Err(_) => {
                debug!(
                    "Failed to commit edge {} -> {}",
                    edge.source_id, edge.target_id
                );
                let _ = sqlite
                    .log_dream_edge(
                        &edge_id,
                        &run_id,
                        &edge.source_id,
                        &edge.target_id,
                        edge.similarity,
                        &edge.relationship_type,
                        &edge.discovery_method,
                        false,
                    )
                    .await;
            }
        }
    }

    let _ = sqlite
        .log_dream_run(
            &run_id,
            &now,
            discoveries.len() as i64,
            discoveries.len() as i64,
            committed,
            &strategies,
            Some(&format!("Committed {committed}/{} edges", discoveries.len())),
        )
        .await;

    info!(
        "Dream exploration: committed {committed}/{} edges",
        discoveries.len()
    );
    Ok(committed)
}
