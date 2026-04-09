//! Graph-informed keyword reweighting (A2.5).
//!
//! Keywords that appear in memories connected by RELATES_TO edges are
//! structurally more important than ones in isolated memories. This module
//! runs a lightweight reweighting pass during compaction, boosting weights
//! for keywords shared across well-connected memory clusters.

use std::collections::HashMap;

use log::info;

use crate::storage::graph_store::GraphStore;
use crate::storage::sqlite_store::SQLiteStore;

/// Adjust keyword weights based on graph connectivity.
///
/// Keywords shared across well-connected memories get boosted.
/// Returns the number of keyword weights updated.
///
/// `max_memories_per_keyword` caps the number of memories evaluated per
/// keyword to avoid O(n²) graph queries for very common keywords.
pub async fn reweight_keywords_from_graph(
    sqlite: &SQLiteStore,
    graph: &GraphStore,
    max_hops: i32,
    max_memories_per_keyword: usize,
) -> Result<i64, Box<dyn std::error::Error + Send + Sync>> {
    let rows = sqlite.get_all_keywords_with_memories(None).await?;
    if rows.is_empty() {
        return Ok(0);
    }

    // Group by keyword
    let mut keyword_index: HashMap<String, Vec<HashMap<String, serde_json::Value>>> =
        HashMap::new();
    for row in rows {
        let keyword = row
            .get("keyword")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        keyword_index.entry(keyword).or_default().push(row);
    }

    // Collect all weight updates, then commit in a single batch
    let mut pending_updates: Vec<(f64, String, String)> = Vec::new();

    for (keyword, entries) in &keyword_index {
        if entries.len() < 2 {
            continue;
        }

        // Cap memories per keyword. Sort by weight descending, then by memory_id.
        let mut sorted_entries = entries.clone();
        sorted_entries.sort_by(|a, b| {
            let wa = a
                .get("weight")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let wb = b
                .get("weight")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let ma = a
                .get("memory_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let mb = b
                .get("memory_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            wb.partial_cmp(&wa)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| ma.cmp(mb))
        });

        let capped_entries: Vec<_> = sorted_entries
            .into_iter()
            .take(max_memories_per_keyword)
            .collect();
        let memory_ids: Vec<String> = capped_entries
            .iter()
            .filter_map(|e| e.get("memory_id").and_then(|v| v.as_str()).map(String::from))
            .collect();

        // Check graph connectivity between memories sharing this keyword
        let mut connected_pairs: usize = 0;
        let mut total_pairs: usize = 0;

        for i in 0..memory_ids.len() {
            for j in (i + 1)..memory_ids.len() {
                total_pairs += 1;
                match graph.path_exists(&memory_ids[i], &memory_ids[j], max_hops).await {
                    Ok(true) => connected_pairs += 1,
                    _ => {}
                }
            }
        }

        if total_pairs == 0 {
            continue;
        }

        let connectivity_ratio = connected_pairs as f64 / total_pairs as f64;

        if connectivity_ratio <= 0.0 {
            continue;
        }

        // Boost: 0.0 connectivity = no change, 1.0 = +50% weight
        let boost = 1.0 + 0.5 * connectivity_ratio;

        for entry in &capped_entries {
            let old_weight = entry
                .get("weight")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let memory_id = entry
                .get("memory_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let new_weight = (old_weight * boost).min(1.0);

            if (new_weight - old_weight).abs() > f64::EPSILON {
                pending_updates.push((new_weight, memory_id.to_string(), keyword.clone()));
            }
        }
    }

    let updated = pending_updates.len() as i64;
    if !pending_updates.is_empty() {
        sqlite.batch_update_keyword_weights(&pending_updates).await?;
    }

    info!("Keyword reweighting: updated {updated} weights");
    Ok(updated)
}
