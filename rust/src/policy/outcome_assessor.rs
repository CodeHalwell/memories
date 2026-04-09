//! Decision-outcome pairing for policy training data (A4.3).
//!
//! Outcomes are assessed asynchronously — either at session end or during the
//! next compaction cycle. Save decisions are assessed by checking whether the
//! saved memory was ever retrieved. Retrieval decisions are assessed by checking
//! whether the agent re-queried the same topic shortly afterward.

use std::collections::HashSet;

use chrono::Utc;
use log::info;

use crate::config::MemoryConfig;
use crate::storage::sqlite_store::SQLiteStore;

/// Assess whether saved memories turned out to be useful.
///
/// A memory is considered useful if it was retrieved at least once within
/// the lookback window. Returns the number of decisions assessed.
pub async fn assess_save_outcomes(
    sqlite: &SQLiteStore,
    config: &MemoryConfig,
    lookback_days: Option<i64>,
) -> Result<i64, Box<dyn std::error::Error + Send + Sync>> {
    let lookback_days = lookback_days.unwrap_or(config.save_outcome_lookback_days);
    let now = Utc::now().to_rfc3339();

    let unassessed = sqlite.get_unassessed_save_decisions(lookback_days).await?;
    let mut updated = 0i64;

    for row in &unassessed {
        let access_count = row
            .get("access_count")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let useful = access_count > 0;
        let id = row
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        sqlite.update_save_outcome(id, useful, &now).await?;
        updated += 1;
    }

    info!("Save outcome assessment: assessed {updated} decisions");
    Ok(updated)
}

/// Assess whether retrievals were helpful.
///
/// Heuristic: if the agent did not re-query the same topic within N turns,
/// the retrieval was probably adequate. Returns the number assessed.
pub async fn assess_retrieval_outcomes(
    sqlite: &SQLiteStore,
    config: &MemoryConfig,
    followup_turns: Option<i32>,
    keyword_overlap_threshold: Option<f64>,
) -> Result<i64, Box<dyn std::error::Error + Send + Sync>> {
    let followup_turns = followup_turns.unwrap_or(config.retrieval_outcome_followup_turns);
    let overlap_threshold =
        keyword_overlap_threshold.unwrap_or(config.retrieval_outcome_keyword_overlap);
    let now = Utc::now().to_rfc3339();

    let unassessed = sqlite.get_unassessed_retrieval_decisions().await?;
    let mut updated = 0i64;

    for row in &unassessed {
        let turn = match row.get("turn").and_then(|v| v.as_i64()) {
            Some(t) => t,
            None => continue,
        };

        let session_id = row
            .get("session_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let query = row
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let followups = sqlite
            .get_retrieval_followups(session_id, turn, followup_turns)
            .await?;

        let original_keywords: HashSet<String> = query
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .map(|w| w.to_lowercase())
            .collect();

        let mut re_queried = false;

        for fu_query in &followups {
            let fu_keywords: HashSet<String> = fu_query
                .split_whitespace()
                .filter(|w| w.len() > 2)
                .map(|w| w.to_lowercase())
                .collect();

            if !original_keywords.is_empty() && !fu_keywords.is_empty() {
                let overlap = original_keywords.intersection(&fu_keywords).count() as f64
                    / original_keywords.len().max(1) as f64;
                if overlap > overlap_threshold {
                    re_queried = true;
                    break;
                }
            }
        }

        let helpful = !re_queried;
        let id = row
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        sqlite.update_retrieval_outcome(id, helpful, &now).await?;
        updated += 1;
    }

    info!("Retrieval outcome assessment: assessed {updated} decisions");
    Ok(updated)
}
