// Package storage provides persistence backends for the agent memory system.
package storage

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	agentmemory "github.com/CodeHalwell/Memories/go"
)

// JSONLLogger is an append-only JSONL logger for raw agent outputs.
// This is the immutable ground truth — entries are never modified or deleted.
type JSONLLogger struct {
	LogDir string
}

// NewJSONLLogger creates a JSONLLogger at the given directory.
func NewJSONLLogger(logDir string) (*JSONLLogger, error) {
	if err := os.MkdirAll(logDir, 0o755); err != nil {
		return nil, fmt.Errorf("creating log dir: %w", err)
	}
	return &JSONLLogger{LogDir: logDir}, nil
}

func (l *JSONLLogger) sessionPath(sessionID string) string {
	safeID := filepath.Base(sessionID)
	return filepath.Join(l.LogDir, safeID+".jsonl")
}

// Append writes an entry and returns (filePath, byteOffset).
func (l *JSONLLogger) Append(entry agentmemory.RawLogEntry) (string, int64, error) {
	path := l.sessionPath(entry.SessionID)

	data, err := json.Marshal(entry)
	if err != nil {
		return "", 0, fmt.Errorf("marshalling entry: %w", err)
	}
	line := append(data, '\n')

	var byteOffset int64
	if info, err := os.Stat(path); err == nil {
		byteOffset = info.Size()
	}

	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return "", 0, fmt.Errorf("opening log file: %w", err)
	}
	defer f.Close()

	if _, err := f.Write(line); err != nil {
		return "", 0, fmt.Errorf("writing entry: %w", err)
	}

	return path, byteOffset, nil
}

// ReadEntry reads a single entry at the given byte offset.
func (l *JSONLLogger) ReadEntry(filePath string, byteOffset int64) (agentmemory.RawLogEntry, error) {
	var entry agentmemory.RawLogEntry

	f, err := os.Open(filePath)
	if err != nil {
		return entry, fmt.Errorf("opening log file: %w", err)
	}
	defer f.Close()

	if _, err := f.Seek(byteOffset, 0); err != nil {
		return entry, fmt.Errorf("seeking to offset: %w", err)
	}

	buf := make([]byte, 0, 4096)
	tmp := make([]byte, 1)
	for {
		n, err := f.Read(tmp)
		if n > 0 {
			if tmp[0] == '\n' {
				break
			}
			buf = append(buf, tmp[0])
		}
		if err != nil {
			break
		}
	}

	if err := json.Unmarshal(buf, &entry); err != nil {
		return entry, fmt.Errorf("unmarshalling entry: %w", err)
	}
	return entry, nil
}

// IterSession yields all entries for a session in order via a callback.
func (l *JSONLLogger) IterSession(sessionID string, fn func(agentmemory.RawLogEntry) bool) error {
	path := l.sessionPath(sessionID)
	file, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("opening session file: %w", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var entry agentmemory.RawLogEntry
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			continue
		}
		if !fn(entry) {
			break
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("scanning session file: %w", err)
	}
	return nil
}

// Search performs a case-insensitive text search within a session log.
func (l *JSONLLogger) Search(sessionID, text string) ([]agentmemory.RawLogEntry, error) {
	var results []agentmemory.RawLogEntry
	lower := strings.ToLower(text)

	err := l.IterSession(sessionID, func(entry agentmemory.RawLogEntry) bool {
		if strings.Contains(strings.ToLower(entry.Content), lower) {
			results = append(results, entry)
		}
		return true
	})
	return results, err
}

// ListSessions returns all session IDs that have log files.
func (l *JSONLLogger) ListSessions() ([]string, error) {
	entries, err := filepath.Glob(filepath.Join(l.LogDir, "*.jsonl"))
	if err != nil {
		return nil, err
	}
	var sessions []string
	for _, e := range entries {
		base := filepath.Base(e)
		sessions = append(sessions, strings.TrimSuffix(base, ".jsonl"))
	}
	return sessions, nil
}

// SessionSize returns the file size in bytes for a session log.
func (l *JSONLLogger) SessionSize(sessionID string) int64 {
	path := l.sessionPath(sessionID)
	info, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return info.Size()
}
