//! Policy training data export (A4.4).
//!
//! Exports decision-outcome pairs as JSONL files for offline policy model training.
//! Requires sufficient assessed data before export is meaningful.

use std::io::Write;
use std::path::PathBuf;

use log::info;

use crate::config::{default_policy_data_dir, MemoryConfig};
use crate::storage::sqlite_store::SQLiteStore;

/// Result of a policy data export operation.
#[derive(Debug, Clone)]
pub struct ExportResult {
    pub save_examples: usize,
    pub retrieval_examples: usize,
    pub save_path: String,
    pub retrieval_path: String,
    pub ready_for_training: bool,
}

/// Export decision-outcome pairs for offline policy model training.
///
/// Returns metadata about the exported data: example counts and file paths.
pub async fn export_policy_training_data(
    sqlite: &SQLiteStore,
    config: &MemoryConfig,
    output_dir: Option<PathBuf>,
) -> Result<ExportResult, Box<dyn std::error::Error + Send + Sync>> {
    let output_dir = output_dir.unwrap_or_else(default_policy_data_dir);
    std::fs::create_dir_all(&output_dir)?;

    let save_data = sqlite.export_save_policy_data().await?;
    let retrieval_data = sqlite.export_retrieval_policy_data().await?;

    let save_path = output_dir.join("save_policy_data.jsonl");
    let retrieval_path = output_dir.join("retrieval_policy_data.jsonl");

    {
        let mut f = std::fs::File::create(&save_path)?;
        for row in &save_data {
            let line = serde_json::to_string(row)?;
            writeln!(f, "{line}")?;
        }
    }

    {
        let mut f = std::fs::File::create(&retrieval_path)?;
        for row in &retrieval_data {
            let line = serde_json::to_string(row)?;
            writeln!(f, "{line}")?;
        }
    }

    let result = ExportResult {
        save_examples: save_data.len(),
        retrieval_examples: retrieval_data.len(),
        save_path: save_path.to_string_lossy().into_owned(),
        retrieval_path: retrieval_path.to_string_lossy().into_owned(),
        ready_for_training: save_data.len() >= config.policy_min_save_examples
            && retrieval_data.len() >= config.policy_min_retrieval_examples,
    };

    info!(
        "Policy data export: {} save examples, {} retrieval examples (ready={})",
        result.save_examples, result.retrieval_examples, result.ready_for_training
    );

    Ok(result)
}
