//! Decay scoring for memory access patterns.
//!
//! Implements time-based exponential decay combined with frequency-based
//! persistence. Mirrors the forgetting curve — memories that aren't accessed
//! gradually lose retrieval priority.
//!
//! A2.2 Amendment: emotional salience (arousal + surprise) slows decay, and
//! semantic (compacted) memories have a floor preventing full decay.

use chrono::{DateTime, Utc};

use crate::config::MemoryConfig;

/// Compute a decay score between 0.0 and ~1.0.
///
/// Higher scores indicate more "alive" memories. The score combines:
///   - Recency: exponential decay based on days since last access
///   - Frequency: logarithmic scaling of access count
///   - Emotional boost: high arousal + surprise slows decay (A2.2)
///   - Semantic floor: compacted memories never fully decay (A2.2)
pub fn compute_decay(
    last_accessed: DateTime<Utc>,
    access_count: i64,
    arousal: f64,
    surprise: f64,
    is_semantic: bool,
    config: &MemoryConfig,
) -> f64 {
    let now = Utc::now();
    let days_since = (now - last_accessed)
        .num_seconds()
        .max(0) as f64
        / 86400.0;

    let halflife = config.decay_halflife_days;
    let lambda = if halflife > 0.0 {
        (2.0_f64).ln() / halflife
    } else {
        0.1
    };

    // A2.2: Emotional memories decay more slowly
    // arousal + surprise in [0, 2], so boost is in [1.0, 2.0]
    let emotional_boost = 1.0 + 0.5 * (arousal + surprise);
    let mut recency = (-lambda * days_since / emotional_boost).exp();

    let frequency = (1.0 + access_count as f64).ln() / 10.0;

    // A2.2: Semantic (compacted) memories have a flatter decay curve
    if is_semantic {
        recency = recency.max(0.3);
    }

    let score = config.decay_recency_weight * recency + config.decay_frequency_weight * frequency;
    (score * 10000.0).round() / 10000.0
}
