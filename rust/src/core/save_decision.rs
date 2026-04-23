//! LLM-driven save decision and keyword extraction.
//!
//! At the end of each turn, this module determines whether the output should be
//! persisted as a memory. The LLM scores emotional dimensions, extracts keywords,
//! and provides a confidence-weighted save/skip decision.
//!
//! A2.1 Amendment: the save decision is informed by retrieval gaps — if recent
//! queries failed to find relevant memories in a topic area, the save threshold
//! for overlapping content is lowered temporarily.

use log::{debug, error};

use crate::config::MemoryConfig;
use crate::llm::client::LlmClient;
use crate::models::{Memory, RawLogEntry, SaveDecision};
use crate::storage::sqlite_store::SQLiteStore;

const FAST_PATH_PHRASES: &[&str] = &[
    "remember this",
    "don't forget",
    "save this",
    "keep in mind",
];

/// Check if a memory should bypass the LLM save decision.
pub fn is_fast_path(arousal: f64, surprise: f64, content: &str, config: &MemoryConfig) -> bool {
    if arousal > config.fast_path_arousal && surprise > config.fast_path_surprise {
        return true;
    }
    let lower = content.to_lowercase();
    FAST_PATH_PHRASES.iter().any(|phrase| lower.contains(phrase))
}

/// Identify topic areas where recent retrievals returned poor results (A2.1).
pub async fn get_retrieval_gaps(
    sqlite: &SQLiteStore,
    session_id: &str,
    config: &MemoryConfig,
) -> Result<Vec<String>, sqlx::Error> {
    sqlite
        .get_failed_retrieval_keywords(session_id, config.gap_lookback_turns as i64)
        .await
}

/// Compute overlap between content keywords and retrieval gap keywords.
pub fn compute_gap_overlap(content_keywords: &[String], gap_keywords: &[String]) -> f64 {
    if content_keywords.is_empty() || gap_keywords.is_empty() {
        return 0.0;
    }
    use std::collections::HashSet;
    let content_set: HashSet<&str> = content_keywords.iter().map(|s| s.as_str()).collect();
    let gap_set: HashSet<&str> = gap_keywords.iter().map(|s| s.as_str()).collect();
    let intersection = content_set.intersection(&gap_set).count();
    intersection as f64 / content_set.len().max(1) as f64
}

/// Decide whether to save an agent output as a memory.
///
/// Returns (SaveDecision, Option<Memory>) — the decision log entry, and a Memory
/// if the decision is to save.
pub async fn make_save_decision(
    entry: &RawLogEntry,
    is_first_turn: bool,
    sqlite: Option<&SQLiteStore>,
    config: &MemoryConfig,
    llm: &LlmClient,
) -> (SaveDecision, Option<Memory>) {
    // First turn of a session is always saved via fast path
    if is_first_turn {
        let mem = Memory {
            content: entry.content.clone(),
            raw_log_id: entry.id.clone(),
            session_id: entry.session_id.clone(),
            turn: entry.turn,
            salience: 0.7,
            fast_pathed: true,
            ..Default::default()
        };
        let dec = SaveDecision {
            raw_log_id: entry.id.clone(),
            session_id: entry.session_id.clone(),
            turn: entry.turn,
            decision: "fast_path".to_string(),
            reason: Some("First turn of session — always saved".to_string()),
            confidence: 1.0,
            ..Default::default()
        };
        return (dec, Some(mem));
    }

    // Ask LLM for structured evaluation
    let prompt = format!(
        "Evaluate whether this agent output should be saved as a memory:\n\n\
         Session: {}\nTurn: {}\nContent:\n<content>\n{}\n</content>\n\n\
         Respond with JSON only.",
        entry.session_id, entry.turn, entry.content
    );

    let result = match llm
        .complete_json(&prompt, Some(&config.prompts.save_decision), None, None)
        .await
    {
        Ok(r) => r,
        Err(e) => {
            error!("LLM save decision failed, defaulting to skip: {e}");
            let dec = SaveDecision {
                raw_log_id: entry.id.clone(),
                session_id: entry.session_id.clone(),
                turn: entry.turn,
                decision: "skip".to_string(),
                reason: Some("LLM evaluation failed".to_string()),
                confidence: 0.0,
                ..Default::default()
            };
            return (dec, None);
        }
    };

    let confidence = result
        .get("confidence")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let should_save = result
        .get("should_save")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let valence = result
        .get("valence")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let arousal = result
        .get("arousal")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let surprise = result
        .get("surprise")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let salience = result
        .get("salience")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.5);

    // Extract keywords for gap analysis
    let keywords: Vec<(String, f64)> = result
        .get("keywords")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|kw| {
                    let keyword = kw.get("keyword")?.as_str()?.to_lowercase();
                    let weight = kw
                        .get("weight")
                        .and_then(|w| w.as_f64())
                        .unwrap_or(1.0);
                    Some((keyword, weight))
                })
                .take(config.max_keywords_per_memory)
                .collect()
        })
        .unwrap_or_default();

    let content_kw_names: Vec<String> = keywords.iter().map(|(kw, _)| kw.clone()).collect();

    // A2.1: Retrieval gap awareness — lower threshold if content fills a gap
    let mut threshold = config.save_confidence_threshold;
    let mut gap_triggered = false;
    if let Some(sqlite) = sqlite {
        match get_retrieval_gaps(sqlite, &entry.session_id, config).await {
            Ok(gap_keywords) => {
                let gap_overlap = compute_gap_overlap(&content_kw_names, &gap_keywords);
                if gap_overlap > config.gap_overlap_threshold {
                    threshold *= config.gap_threshold_reduction;
                    gap_triggered = true;
                }
            }
            Err(_) => {
                debug!("Gap detection failed, using default threshold");
            }
        }
    }

    // Check fast path conditions from LLM-scored emotions
    let fast_path = is_fast_path(arousal, surprise, &entry.content, config);

    let (decision, final_confidence, final_should_save) = if fast_path {
        ("fast_path".to_string(), confidence.max(0.9), true)
    } else if should_save && confidence >= threshold {
        ("save".to_string(), confidence, true)
    } else {
        ("skip".to_string(), confidence, false)
    };

    let dec = SaveDecision {
        raw_log_id: entry.id.clone(),
        session_id: entry.session_id.clone(),
        turn: entry.turn,
        decision: decision.clone(),
        reason: result.get("reason").and_then(|v| v.as_str()).map(String::from),
        confidence: final_confidence,
        gap_triggered,
        threshold_used: Some(threshold),
        ..Default::default()
    };

    if final_should_save {
        let mem = Memory {
            content: entry.content.clone(),
            summary: result.get("summary").and_then(|v| v.as_str()).map(String::from),
            raw_log_id: entry.id.clone(),
            session_id: entry.session_id.clone(),
            turn: entry.turn,
            valence,
            arousal,
            surprise,
            salience,
            fast_pathed: fast_path,
            keywords,
            ..Default::default()
        };
        (dec, Some(mem))
    } else {
        (dec, None)
    }
}
