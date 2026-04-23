//! Raw JSONL log writer and indexer.
//!
//! Every agent output is appended to a session-specific JSONL file. This is the
//! immutable ground truth — entries are never modified or deleted.

use std::path::{Path, PathBuf};

use crate::config::default_log_dir;
use crate::models::RawLogEntry;

/// Append-only JSONL logger for raw agent outputs.
pub struct JSONLLogger {
    log_dir: PathBuf,
}

impl JSONLLogger {
    pub fn new(log_dir: Option<PathBuf>) -> Result<Self, std::io::Error> {
        let dir = log_dir.unwrap_or_else(default_log_dir);
        std::fs::create_dir_all(&dir)?;
        Ok(Self { log_dir: dir })
    }

    fn session_path(&self, session_id: &str) -> PathBuf {
        // Sanitize session_id to prevent path traversal
        let safe_id = Path::new(session_id)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| session_id.to_string());
        self.log_dir.join(format!("{safe_id}.jsonl"))
    }

    /// Append an entry and return (file_path, byte_offset).
    pub fn append(&self, entry: &RawLogEntry) -> Result<(String, u64), std::io::Error> {
        let path = self.session_path(&entry.session_id);
        let line = serde_json::to_string(entry).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?;

        let byte_offset = if path.exists() {
            std::fs::metadata(&path)?.len()
        } else {
            0
        };

        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        writeln!(file, "{line}")?;

        Ok((path.to_string_lossy().into_owned(), byte_offset))
    }

    /// Read a single entry at the given byte offset.
    pub fn read_entry(&self, file_path: &str, byte_offset: u64) -> Result<RawLogEntry, Box<dyn std::error::Error>> {
        use std::io::{BufRead, Seek, SeekFrom};
        let mut file = std::fs::File::open(file_path)?;
        file.seek(SeekFrom::Start(byte_offset))?;
        let mut reader = std::io::BufReader::new(file);
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let entry: RawLogEntry = serde_json::from_str(line.trim())?;
        Ok(entry)
    }

    /// Yield all entries for a session in order.
    pub fn iter_session(&self, session_id: &str) -> Result<Vec<RawLogEntry>, Box<dyn std::error::Error>> {
        let path = self.session_path(session_id);
        if !path.exists() {
            return Ok(Vec::new());
        }
        use std::io::BufRead;
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let mut entries = Vec::new();
        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                let entry: RawLogEntry = serde_json::from_str(trimmed)?;
                entries.push(entry);
            }
        }
        Ok(entries)
    }

    /// Simple text search within a session log.
    pub fn search(&self, session_id: &str, text: &str) -> Result<Vec<RawLogEntry>, Box<dyn std::error::Error>> {
        let entries = self.iter_session(session_id)?;
        let lower_text = text.to_lowercase();
        let results = entries
            .into_iter()
            .filter(|e| e.content.to_lowercase().contains(&lower_text))
            .collect();
        Ok(results)
    }

    /// Return all session IDs that have log files.
    pub fn list_sessions(&self) -> Result<Vec<String>, std::io::Error> {
        let mut sessions = Vec::new();
        if self.log_dir.exists() {
            let mut entries: Vec<_> = std::fs::read_dir(&self.log_dir)?
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .is_some_and(|ext| ext == "jsonl")
                })
                .collect();
            entries.sort_by_key(|e| e.path());
            for entry in entries {
                if let Some(stem) = entry.path().file_stem() {
                    sessions.push(stem.to_string_lossy().into_owned());
                }
            }
        }
        Ok(sessions)
    }

    /// Return the file size in bytes for a session log.
    pub fn session_size(&self, session_id: &str) -> u64 {
        let path = self.session_path(session_id);
        if path.exists() {
            std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0)
        } else {
            0
        }
    }
}
