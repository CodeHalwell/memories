//! Emotional scoring via LLM.
//!
//! Scores the current context for valence and arousal to support mood-congruent
//! retrieval. Can also re-score existing memories during compaction.

use std::collections::HashMap;

use log::error;

use crate::config::MemoryConfig;
use crate::llm::client::LlmClient;

/// Score the emotional dimensions of a text.
///
/// Returns a map with keys: valence, arousal, surprise.
pub async fn score_emotion(
    text: &str,
    config: &MemoryConfig,
    llm: &LlmClient,
) -> HashMap<String, f64> {
    let prompt = format!("Score the emotional tone of this text:\n\n<text>\n{text}\n</text>");

    match llm
        .complete_json(&prompt, Some(&config.prompts.emotion), None, None)
        .await
    {
        Ok(result) => {
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

            let mut scores = HashMap::new();
            scores.insert("valence".to_string(), clamp(valence, -1.0, 1.0));
            scores.insert("arousal".to_string(), clamp(arousal, 0.0, 1.0));
            scores.insert("surprise".to_string(), clamp(surprise, 0.0, 1.0));
            scores
        }
        Err(e) => {
            error!("Emotion scoring failed, returning neutral: {e}");
            let mut scores = HashMap::new();
            scores.insert("valence".to_string(), 0.0);
            scores.insert("arousal".to_string(), 0.0);
            scores.insert("surprise".to_string(), 0.0);
            scores
        }
    }
}

fn clamp(value: f64, lo: f64, hi: f64) -> f64 {
    value.max(lo).min(hi)
}
