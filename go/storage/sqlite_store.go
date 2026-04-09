package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	_ "github.com/mattn/go-sqlite3"

	agentmemory "github.com/CodeHalwell/Memories/go"
)

const schema = `
CREATE TABLE IF NOT EXISTS raw_log_index (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    turn        INTEGER NOT NULL,
    timestamp   TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    byte_offset INTEGER
);

CREATE TABLE IF NOT EXISTS memories (
    id                TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    content           TEXT NOT NULL,
    summary           TEXT,
    raw_log_id        TEXT NOT NULL,
    session_id        TEXT NOT NULL,
    turn              INTEGER NOT NULL,
    valence           REAL,
    arousal           REAL,
    surprise          REAL,
    salience          REAL DEFAULT 0.5,
    access_count      INTEGER DEFAULT 0,
    last_accessed     TEXT,
    decay_score       REAL DEFAULT 1.0,
    compaction_gen    INTEGER DEFAULT 0,
    tier              TEXT DEFAULT 'hot',
    fast_pathed       INTEGER DEFAULT 0,
    is_semantic       INTEGER DEFAULT 0,
    graph_node_id     TEXT,
    vector_id         TEXT,
    spatial_embedding BLOB,
    scene_description TEXT,
    FOREIGN KEY (raw_log_id) REFERENCES raw_log_index(id)
);

CREATE TABLE IF NOT EXISTS memory_keywords (
    memory_id   TEXT NOT NULL,
    keyword     TEXT NOT NULL,
    weight      REAL DEFAULT 1.0,
    PRIMARY KEY (memory_id, keyword),
    FOREIGN KEY (memory_id) REFERENCES memories(id)
);

CREATE INDEX IF NOT EXISTS idx_keyword ON memory_keywords(keyword);

CREATE TABLE IF NOT EXISTS memory_access_log (
    id          TEXT PRIMARY KEY,
    memory_id   TEXT NOT NULL,
    accessed_at TEXT NOT NULL,
    access_type TEXT NOT NULL,
    session_id  TEXT,
    query       TEXT,
    FOREIGN KEY (memory_id) REFERENCES memories(id)
);

CREATE TABLE IF NOT EXISTS compaction_runs (
    id                  TEXT PRIMARY KEY,
    ran_at              TEXT NOT NULL,
    trigger_type        TEXT,
    memories_reviewed   INTEGER,
    memories_merged     INTEGER,
    memories_pruned     INTEGER,
    notes               TEXT,
    keywords_updated    INTEGER DEFAULT 0,
    edges_discovered    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS compaction_merges (
    compaction_id         TEXT NOT NULL,
    source_memory_ids     TEXT NOT NULL,
    resulting_memory_id   TEXT NOT NULL,
    validation_passed     INTEGER,
    avg_source_score      REAL,
    avg_merged_score      REAL,
    degradation           REAL,
    FOREIGN KEY (compaction_id) REFERENCES compaction_runs(id)
);

CREATE TABLE IF NOT EXISTS save_decisions (
    id                  TEXT PRIMARY KEY,
    raw_log_id          TEXT NOT NULL,
    session_id          TEXT NOT NULL,
    turn                INTEGER NOT NULL,
    decided_at          TEXT NOT NULL,
    decision            TEXT NOT NULL,
    reason              TEXT,
    confidence          REAL,
    gap_triggered       INTEGER DEFAULT 0,
    threshold_used      REAL,
    outcome_useful      INTEGER,
    outcome_assessed_at TEXT
);

CREATE TABLE IF NOT EXISTS retrieval_decisions (
    id                  TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL,
    turn                INTEGER,
    query               TEXT NOT NULL,
    decided_at          TEXT NOT NULL,
    layers_queried      TEXT NOT NULL,
    graph_depth         INTEGER,
    mood_weight         REAL,
    top_k               INTEGER,
    memories_returned   TEXT NOT NULL,
    return_count        INTEGER NOT NULL,
    outcome_helpful     INTEGER,
    outcome_assessed_at TEXT
);

CREATE TABLE IF NOT EXISTS dream_exploration_runs (
    id                TEXT PRIMARY KEY,
    ran_at            TEXT NOT NULL,
    n_walks           INTEGER,
    edges_discovered  INTEGER,
    edges_committed   INTEGER,
    strategies_used   TEXT,
    notes             TEXT
);

CREATE TABLE IF NOT EXISTS dream_discovered_edges (
    id                  TEXT PRIMARY KEY,
    exploration_run_id  TEXT NOT NULL,
    source_memory_id    TEXT NOT NULL,
    target_memory_id    TEXT NOT NULL,
    similarity          REAL,
    relationship_type   TEXT,
    discovery_method    TEXT,
    committed           INTEGER DEFAULT 0,
    FOREIGN KEY (exploration_run_id) REFERENCES dream_exploration_runs(id)
);
`

// SQLiteStore provides persistent storage for memory metadata, access tracking, and compaction history.
type SQLiteStore struct {
	dbPath string
	db     *sql.DB
}

// NewSQLiteStore creates a new SQLiteStore targeting the given database path.
func NewSQLiteStore(dbPath string) *SQLiteStore {
	return &SQLiteStore{dbPath: dbPath}
}

// Initialize opens the database and creates the schema.
func (s *SQLiteStore) Initialize(ctx context.Context) error {
	dir := filepath.Dir(s.dbPath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("creating db dir: %w", err)
	}

	db, err := sql.Open("sqlite3", s.dbPath+"?_journal_mode=WAL&_busy_timeout=5000")
	if err != nil {
		return fmt.Errorf("opening database: %w", err)
	}
	s.db = db

	if _, err := s.db.ExecContext(ctx, schema); err != nil {
		return fmt.Errorf("creating schema: %w", err)
	}
	return nil
}

// Close closes the database connection.
func (s *SQLiteStore) Close() error {
	if s.db != nil {
		return s.db.Close()
	}
	return nil
}

// DB returns the underlying *sql.DB. Panics if not initialized.
func (s *SQLiteStore) DB() *sql.DB {
	if s.db == nil {
		panic("SQLiteStore not initialized — call Initialize() first")
	}
	return s.db
}

// ---------- Raw log index ----------

// IndexRawLog inserts a raw log index record.
func (s *SQLiteStore) IndexRawLog(ctx context.Context, entryID, sessionID string, turn int, timestamp, filePath string, byteOffset int64) error {
	_, err := s.DB().ExecContext(ctx,
		"INSERT OR IGNORE INTO raw_log_index (id, session_id, turn, timestamp, file_path, byte_offset) VALUES (?, ?, ?, ?, ?, ?)",
		entryID, sessionID, turn, timestamp, filePath, byteOffset,
	)
	return err
}

// RawLogRef is a row from raw_log_index.
type RawLogRef struct {
	ID         string
	SessionID  string
	Turn       int
	Timestamp  string
	FilePath   string
	ByteOffset int64
}

// GetRawLogRef returns the raw log reference for the given entry ID, or nil if not found.
func (s *SQLiteStore) GetRawLogRef(ctx context.Context, entryID string) (*RawLogRef, error) {
	row := s.DB().QueryRowContext(ctx, "SELECT id, session_id, turn, timestamp, file_path, byte_offset FROM raw_log_index WHERE id = ?", entryID)
	var ref RawLogRef
	err := row.Scan(&ref.ID, &ref.SessionID, &ref.Turn, &ref.Timestamp, &ref.FilePath, &ref.ByteOffset)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &ref, nil
}

// ---------- Memories ----------

// SaveMemory inserts or replaces a memory and its keywords.
func (s *SQLiteStore) SaveMemory(ctx context.Context, mem agentmemory.Memory) error {
	tx, err := s.DB().BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback() //nolint:errcheck

	_, err = tx.ExecContext(ctx,
		`INSERT OR REPLACE INTO memories
		(id, created_at, updated_at, content, summary, raw_log_id, session_id, turn,
		 valence, arousal, surprise, salience, access_count, last_accessed, decay_score,
		 compaction_gen, tier, fast_pathed, is_semantic, graph_node_id, vector_id,
		 spatial_embedding, scene_description)
		VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
		mem.ID, mem.CreatedAt, mem.UpdatedAt, mem.Content, mem.Summary,
		mem.RawLogID, mem.SessionID, mem.Turn,
		mem.Valence, mem.Arousal, mem.Surprise, mem.Salience,
		mem.AccessCount, mem.LastAccessed, mem.DecayScore,
		mem.CompactionGen, mem.Tier, boolToInt(mem.FastPathed), boolToInt(mem.IsSemantic),
		mem.GraphNodeID, mem.VectorID,
		mem.SpatialEmbedding, mem.SceneDescription,
	)
	if err != nil {
		return err
	}

	// Save keywords
	_, err = tx.ExecContext(ctx, "DELETE FROM memory_keywords WHERE memory_id = ?", mem.ID)
	if err != nil {
		return err
	}
	for _, kw := range mem.Keywords {
		_, err = tx.ExecContext(ctx,
			"INSERT INTO memory_keywords (memory_id, keyword, weight) VALUES (?, ?, ?)",
			mem.ID, kw.Keyword, kw.Weight,
		)
		if err != nil {
			return err
		}
	}

	return tx.Commit()
}

// GetMemory retrieves a memory by ID, including its keywords. Returns nil if not found.
func (s *SQLiteStore) GetMemory(ctx context.Context, memoryID string) (*agentmemory.Memory, error) {
	row := s.DB().QueryRowContext(ctx, "SELECT * FROM memories WHERE id = ?", memoryID)
	mem, err := scanMemory(row)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	keywords, err := s.loadKeywords(ctx, memoryID)
	if err != nil {
		return nil, err
	}
	mem.Keywords = keywords

	return mem, nil
}

// ListMemories returns memories, optionally filtered by tier.
func (s *SQLiteStore) ListMemories(ctx context.Context, tier *string, limit, offset int) ([]agentmemory.Memory, error) {
	var rows *sql.Rows
	var err error
	if tier != nil {
		rows, err = s.DB().QueryContext(ctx,
			"SELECT * FROM memories WHERE tier = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
			*tier, limit, offset,
		)
	} else {
		rows, err = s.DB().QueryContext(ctx,
			"SELECT * FROM memories ORDER BY created_at DESC LIMIT ? OFFSET ?",
			limit, offset,
		)
	}
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var memories []agentmemory.Memory
	for rows.Next() {
		mem, err := scanMemoryRows(rows)
		if err != nil {
			return nil, err
		}
		memories = append(memories, *mem)
	}

	if err := s.batchLoadKeywords(ctx, memories); err != nil {
		return nil, err
	}

	return memories, nil
}

// CountMemories returns the count of memories, optionally filtered by tier.
func (s *SQLiteStore) CountMemories(ctx context.Context, tier *string) (int, error) {
	var count int
	var err error
	if tier != nil {
		err = s.DB().QueryRowContext(ctx, "SELECT COUNT(*) FROM memories WHERE tier = ?", *tier).Scan(&count)
	} else {
		err = s.DB().QueryRowContext(ctx, "SELECT COUNT(*) FROM memories").Scan(&count)
	}
	return count, err
}

// UpdateMemoryAccess updates decay, access count and last_accessed for a memory.
func (s *SQLiteStore) UpdateMemoryAccess(ctx context.Context, memoryID string, decayScore float64, accessCount int, lastAccessed string) error {
	_, err := s.DB().ExecContext(ctx,
		"UPDATE memories SET decay_score = ?, access_count = ?, last_accessed = ?, updated_at = ? WHERE id = ?",
		decayScore, accessCount, lastAccessed, lastAccessed, memoryID,
	)
	return err
}

// UpdateMemoryTier updates the tier of a memory.
func (s *SQLiteStore) UpdateMemoryTier(ctx context.Context, memoryID, tier string) error {
	_, err := s.DB().ExecContext(ctx, "UPDATE memories SET tier = ? WHERE id = ?", tier, memoryID)
	return err
}

// UpdateMemoryGraphRef updates the graph node ID reference for a memory.
func (s *SQLiteStore) UpdateMemoryGraphRef(ctx context.Context, memoryID, graphNodeID string) error {
	_, err := s.DB().ExecContext(ctx, "UPDATE memories SET graph_node_id = ? WHERE id = ?", graphNodeID, memoryID)
	return err
}

// UpdateMemoryVectorRef updates the vector point ID reference for a memory.
func (s *SQLiteStore) UpdateMemoryVectorRef(ctx context.Context, memoryID, vectorID string) error {
	_, err := s.DB().ExecContext(ctx, "UPDATE memories SET vector_id = ? WHERE id = ?", vectorID, memoryID)
	return err
}

// UpdateMemoryVisual updates the scene description and spatial embedding for a memory.
func (s *SQLiteStore) UpdateMemoryVisual(ctx context.Context, memoryID, sceneDescription string, spatialEmbedding []byte) error {
	_, err := s.DB().ExecContext(ctx,
		"UPDATE memories SET scene_description = ?, spatial_embedding = ? WHERE id = ?",
		sceneDescription, spatialEmbedding, memoryID,
	)
	return err
}

// ---------- Keyword search ----------

// SearchByKeywords searches memories by keyword match, ranked by weight * decay.
func (s *SQLiteStore) SearchByKeywords(ctx context.Context, keywords []string, limit int) ([]agentmemory.Memory, error) {
	if len(keywords) == 0 {
		return nil, nil
	}
	placeholders := make([]string, len(keywords))
	args := make([]interface{}, len(keywords))
	for i, kw := range keywords {
		placeholders[i] = "?"
		args[i] = kw
	}
	args = append(args, limit)

	query := fmt.Sprintf(`
		SELECT m.id, m.created_at, m.updated_at, m.content, m.summary, m.raw_log_id,
		       m.session_id, m.turn, m.valence, m.arousal, m.surprise, m.salience,
		       m.access_count, m.last_accessed, m.decay_score, m.compaction_gen, m.tier,
		       m.fast_pathed, m.is_semantic, m.graph_node_id, m.vector_id,
		       m.spatial_embedding, m.scene_description,
		       SUM(mk.weight) as match_score
		FROM memories m
		JOIN memory_keywords mk ON m.id = mk.memory_id
		WHERE mk.keyword IN (%s)
		GROUP BY m.id
		ORDER BY match_score * m.decay_score DESC
		LIMIT ?
	`, strings.Join(placeholders, ","))

	rows, err := s.DB().QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var memories []agentmemory.Memory
	for rows.Next() {
		mem, err := scanMemoryWithExtra(rows)
		if err != nil {
			return nil, err
		}
		memories = append(memories, *mem)
	}

	if err := s.batchLoadKeywords(ctx, memories); err != nil {
		return nil, err
	}

	return memories, nil
}

// UpdateKeywordWeight updates a single keyword weight (A2.5).
func (s *SQLiteStore) UpdateKeywordWeight(ctx context.Context, memoryID, keyword string, weight float64) error {
	_, err := s.DB().ExecContext(ctx,
		"UPDATE memory_keywords SET weight = MIN(?, 1.0) WHERE memory_id = ? AND keyword = ?",
		weight, memoryID, keyword,
	)
	return err
}

// KeywordWeightUpdate is a single weight update for batch operations.
type KeywordWeightUpdate struct {
	Weight   float64
	MemoryID string
	Keyword  string
}

// BatchUpdateKeywordWeights updates multiple keyword weights in a single transaction (A2.5).
func (s *SQLiteStore) BatchUpdateKeywordWeights(ctx context.Context, updates []KeywordWeightUpdate) error {
	tx, err := s.DB().BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback() //nolint:errcheck

	stmt, err := tx.PrepareContext(ctx, "UPDATE memory_keywords SET weight = MIN(?, 1.0) WHERE memory_id = ? AND keyword = ?")
	if err != nil {
		return err
	}
	defer stmt.Close()

	for _, u := range updates {
		if _, err := stmt.ExecContext(ctx, u.Weight, u.MemoryID, u.Keyword); err != nil {
			return err
		}
	}
	return tx.Commit()
}

// KeywordMemoryAssociation describes a keyword and its associated memory.
type KeywordMemoryAssociation struct {
	Keyword  string
	MemoryID string
	Weight   float64
}

// GetAllKeywordsWithMemories returns all keyword-memory associations for active tiers (A2.5).
func (s *SQLiteStore) GetAllKeywordsWithMemories(ctx context.Context, tiers []string) ([]KeywordMemoryAssociation, error) {
	if len(tiers) == 0 {
		tiers = []string{"hot", "warm"}
	}
	placeholders := make([]string, len(tiers))
	args := make([]interface{}, len(tiers))
	for i, t := range tiers {
		placeholders[i] = "?"
		args[i] = t
	}
	query := fmt.Sprintf(`
		SELECT mk.keyword, mk.memory_id, mk.weight
		FROM memory_keywords mk
		JOIN memories m ON mk.memory_id = m.id
		WHERE m.tier IN (%s)
	`, strings.Join(placeholders, ","))

	rows, err := s.DB().QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []KeywordMemoryAssociation
	for rows.Next() {
		var a KeywordMemoryAssociation
		if err := rows.Scan(&a.Keyword, &a.MemoryID, &a.Weight); err != nil {
			return nil, err
		}
		results = append(results, a)
	}
	return results, nil
}

// ---------- Access log ----------

// LogAccess logs a memory access event.
func (s *SQLiteStore) LogAccess(ctx context.Context, accessID, memoryID, accessedAt, accessType string, sessionID, query *string) error {
	_, err := s.DB().ExecContext(ctx,
		"INSERT INTO memory_access_log (id, memory_id, accessed_at, access_type, session_id, query) VALUES (?, ?, ?, ?, ?, ?)",
		accessID, memoryID, accessedAt, accessType, sessionID, query,
	)
	return err
}

// GetRecentAccessQueries returns recent retrieval queries for a session (A2.1 gap detection).
func (s *SQLiteStore) GetRecentAccessQueries(ctx context.Context, sessionID string, limit int) ([]string, error) {
	rows, err := s.DB().QueryContext(ctx,
		"SELECT query FROM memory_access_log WHERE session_id = ? AND query IS NOT NULL ORDER BY accessed_at DESC LIMIT ?",
		sessionID, limit,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var queries []string
	for rows.Next() {
		var q string
		if err := rows.Scan(&q); err != nil {
			return nil, err
		}
		queries = append(queries, q)
	}
	return queries, nil
}

// GetFailedRetrievalKeywords identifies keywords from queries that yielded no or only cold-tier results (A2.1).
func (s *SQLiteStore) GetFailedRetrievalKeywords(ctx context.Context, sessionID string, lookback int) ([]string, error) {
	queries, err := s.GetRecentAccessQueries(ctx, sessionID, lookback)
	if err != nil {
		return nil, err
	}
	if len(queries) == 0 {
		return nil, nil
	}

	gapKeywordsSet := make(map[string]struct{})
	for _, q := range queries {
		rows, err := s.DB().QueryContext(ctx,
			`SELECT m.tier FROM memory_access_log mal
			 JOIN memories m ON mal.memory_id = m.id
			 WHERE mal.query = ? AND mal.session_id = ?`,
			q, sessionID,
		)
		if err != nil {
			return nil, err
		}

		var tiers []string
		for rows.Next() {
			var t string
			if err := rows.Scan(&t); err != nil {
				rows.Close()
				return nil, err
			}
			tiers = append(tiers, t)
		}
		rows.Close()

		if len(tiers) == 0 || allCold(tiers) {
			words := strings.Fields(q)
			for _, w := range words {
				w = strings.ToLower(strings.TrimSpace(w))
				if len(w) > 2 {
					gapKeywordsSet[w] = struct{}{}
				}
			}
		}
	}

	gapKeywords := make([]string, 0, len(gapKeywordsSet))
	for k := range gapKeywordsSet {
		gapKeywords = append(gapKeywords, k)
	}
	return gapKeywords, nil
}

// ---------- Save decisions ----------

// LogSaveDecision persists a save decision.
func (s *SQLiteStore) LogSaveDecision(ctx context.Context, dec agentmemory.SaveDecision) error {
	_, err := s.DB().ExecContext(ctx,
		`INSERT INTO save_decisions
		(id, raw_log_id, session_id, turn, decided_at, decision, reason, confidence, gap_triggered, threshold_used)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		dec.ID, dec.RawLogID, dec.SessionID, dec.Turn, dec.DecidedAt,
		dec.Decision, dec.Reason, dec.Confidence, boolToInt(dec.GapTriggered), dec.ThresholdUsed,
	)
	return err
}

// UpdateSaveOutcome marks whether a saved memory turned out to be useful (A4).
func (s *SQLiteStore) UpdateSaveOutcome(ctx context.Context, decisionID string, useful bool, assessedAt string) error {
	_, err := s.DB().ExecContext(ctx,
		"UPDATE save_decisions SET outcome_useful = ?, outcome_assessed_at = ? WHERE id = ?",
		boolToInt(useful), assessedAt, decisionID,
	)
	return err
}

// UnassessedSaveDecision is a row from the unassessed save decisions query.
type UnassessedSaveDecision struct {
	ID          string
	RawLogID    string
	MemoryID    *string
	AccessCount *int
}

// GetUnassessedSaveDecisions returns save decisions not yet assessed (A4).
func (s *SQLiteStore) GetUnassessedSaveDecisions(ctx context.Context, lookbackDays int) ([]UnassessedSaveDecision, error) {
	rows, err := s.DB().QueryContext(ctx,
		`SELECT sd.id, sd.raw_log_id, m.id as memory_id, m.access_count
		 FROM save_decisions sd
		 LEFT JOIN memories m ON m.raw_log_id = sd.raw_log_id
		 WHERE sd.decision IN ('save', 'fast_path')
		   AND sd.id NOT IN (
		       SELECT id FROM save_decisions WHERE outcome_useful IS NOT NULL
		   )
		   AND sd.decided_at < datetime('now', ?)`,
		fmt.Sprintf("-%d days", lookbackDays),
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []UnassessedSaveDecision
	for rows.Next() {
		var r UnassessedSaveDecision
		if err := rows.Scan(&r.ID, &r.RawLogID, &r.MemoryID, &r.AccessCount); err != nil {
			return nil, err
		}
		results = append(results, r)
	}
	return results, nil
}

// ---------- Retrieval decisions (A4) ----------

// LogRetrievalDecision logs a retrieval decision for policy training.
func (s *SQLiteStore) LogRetrievalDecision(
	ctx context.Context,
	decisionID, sessionID string, turn *int, query, decidedAt string,
	layersQueried []string, graphDepth int, moodWeight float64, topK int,
	memoryIDs []string, returnCount int,
) error {
	layersJSON, _ := json.Marshal(layersQueried)
	memoriesJSON, _ := json.Marshal(memoryIDs)
	_, err := s.DB().ExecContext(ctx,
		`INSERT INTO retrieval_decisions
		(id, session_id, turn, query, decided_at, layers_queried, graph_depth,
		 mood_weight, top_k, memories_returned, return_count)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		decisionID, sessionID, turn, query, decidedAt,
		string(layersJSON), graphDepth, moodWeight, topK,
		string(memoriesJSON), returnCount,
	)
	return err
}

// UnassessedRetrievalDecision is a row from the unassessed retrieval decisions query.
type UnassessedRetrievalDecision struct {
	ID        string
	SessionID string
	Turn      *int
	Query     string
}

// GetUnassessedRetrievalDecisions returns retrieval decisions not yet assessed (A4).
func (s *SQLiteStore) GetUnassessedRetrievalDecisions(ctx context.Context) ([]UnassessedRetrievalDecision, error) {
	rows, err := s.DB().QueryContext(ctx,
		`SELECT id, session_id, turn, query
		 FROM retrieval_decisions
		 WHERE outcome_helpful IS NULL
		   AND decided_at < datetime('now', '-1 hour')`,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []UnassessedRetrievalDecision
	for rows.Next() {
		var r UnassessedRetrievalDecision
		if err := rows.Scan(&r.ID, &r.SessionID, &r.Turn, &r.Query); err != nil {
			return nil, err
		}
		results = append(results, r)
	}
	return results, nil
}

// UpdateRetrievalOutcome marks a retrieval decision as helpful or not (A4).
func (s *SQLiteStore) UpdateRetrievalOutcome(ctx context.Context, decisionID string, helpful bool, assessedAt string) error {
	_, err := s.DB().ExecContext(ctx,
		"UPDATE retrieval_decisions SET outcome_helpful = ?, outcome_assessed_at = ? WHERE id = ?",
		boolToInt(helpful), assessedAt, decisionID,
	)
	return err
}

// GetRetrievalFollowups returns follow-up queries within a turn window (A4 outcome assessment).
func (s *SQLiteStore) GetRetrievalFollowups(ctx context.Context, sessionID string, turn, window int) ([]string, error) {
	rows, err := s.DB().QueryContext(ctx,
		"SELECT query FROM retrieval_decisions WHERE session_id = ? AND turn > ? AND turn <= ?",
		sessionID, turn, turn+window,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var queries []string
	for rows.Next() {
		var q string
		if err := rows.Scan(&q); err != nil {
			return nil, err
		}
		queries = append(queries, q)
	}
	return queries, nil
}

// ---------- Dream exploration logging (A3) ----------

// LogDreamRun logs a dream exploration run.
func (s *SQLiteStore) LogDreamRun(ctx context.Context, runID, ranAt string, nWalks, edgesDiscovered, edgesCommitted int, strategies []string, notes *string) error {
	strategiesJSON, _ := json.Marshal(strategies)
	_, err := s.DB().ExecContext(ctx,
		`INSERT INTO dream_exploration_runs
		(id, ran_at, n_walks, edges_discovered, edges_committed, strategies_used, notes)
		VALUES (?, ?, ?, ?, ?, ?, ?)`,
		runID, ranAt, nWalks, edgesDiscovered, edgesCommitted, string(strategiesJSON), notes,
	)
	return err
}

// LogDreamEdge logs a discovered edge from a dream exploration run.
func (s *SQLiteStore) LogDreamEdge(ctx context.Context, edgeID, runID, sourceID, targetID string, similarity float64, relationshipType, discoveryMethod string, committed bool) error {
	_, err := s.DB().ExecContext(ctx,
		`INSERT INTO dream_discovered_edges
		(id, exploration_run_id, source_memory_id, target_memory_id,
		 similarity, relationship_type, discovery_method, committed)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		edgeID, runID, sourceID, targetID, similarity,
		relationshipType, discoveryMethod, boolToInt(committed),
	)
	return err
}

// ---------- Compaction ----------

// LogCompactionRun logs a compaction run result.
func (s *SQLiteStore) LogCompactionRun(ctx context.Context, result agentmemory.CompactionResult) error {
	_, err := s.DB().ExecContext(ctx,
		`INSERT INTO compaction_runs
		(id, ran_at, trigger_type, memories_reviewed, memories_merged, memories_pruned,
		 notes, keywords_updated, edges_discovered)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		result.ID, result.RanAt, result.Trigger, result.MemoriesReviewed,
		result.MemoriesMerged, result.MemoriesPruned, result.Notes,
		result.KeywordsUpdated, result.EdgesDiscovered,
	)
	return err
}

// LogCompactionMerge logs a single merge within a compaction run.
func (s *SQLiteStore) LogCompactionMerge(
	ctx context.Context, compactionID string, sourceIDs []string, resultingID string,
	validationPassed *bool, avgSourceScore, avgMergedScore, degradation *float64,
) error {
	sourceJSON, _ := json.Marshal(sourceIDs)
	var vpInt *int
	if validationPassed != nil {
		v := boolToInt(*validationPassed)
		vpInt = &v
	}
	_, err := s.DB().ExecContext(ctx,
		`INSERT INTO compaction_merges
		(compaction_id, source_memory_ids, resulting_memory_id,
		 validation_passed, avg_source_score, avg_merged_score, degradation)
		VALUES (?, ?, ?, ?, ?, ?, ?)`,
		compactionID, string(sourceJSON), resultingID,
		vpInt, avgSourceScore, avgMergedScore, degradation,
	)
	return err
}

// GetCompactionCandidates returns memories that are candidates for compaction.
func (s *SQLiteStore) GetCompactionCandidates(ctx context.Context, threshold float64) ([]agentmemory.Memory, error) {
	rows, err := s.DB().QueryContext(ctx,
		`SELECT * FROM memories
		 WHERE tier = 'hot'
		   AND fast_pathed = 0
		   AND NOT (compaction_gen = 0 AND access_count > 5)
		 ORDER BY ((1 - decay_score) * 0.6 + (1 - salience) * 0.4) DESC`,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var candidates []agentmemory.Memory
	for rows.Next() {
		mem, err := scanMemoryRows(rows)
		if err != nil {
			return nil, err
		}
		score := (1-mem.DecayScore)*0.6 + (1-mem.Salience)*0.4
		if score > threshold {
			candidates = append(candidates, *mem)
		}
	}

	if err := s.batchLoadKeywords(ctx, candidates); err != nil {
		return nil, err
	}

	return candidates, nil
}

// ---------- Policy data export (A4.4) ----------

// ExportSavePolicyData returns assessed save decisions for policy training.
func (s *SQLiteStore) ExportSavePolicyData(ctx context.Context) ([]map[string]interface{}, error) {
	rows, err := s.DB().QueryContext(ctx,
		`SELECT sd.confidence, sd.decision, sd.gap_triggered,
		        m.valence, m.arousal, m.surprise, m.salience,
		        sd.outcome_useful
		 FROM save_decisions sd
		 LEFT JOIN memories m ON m.raw_log_id = sd.raw_log_id
		 WHERE sd.outcome_useful IS NOT NULL`,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []map[string]interface{}
	for rows.Next() {
		var confidence, valence, arousal, surprise, salience sql.NullFloat64
		var decision sql.NullString
		var gapTriggered, outcomeUseful sql.NullInt64
		if err := rows.Scan(&confidence, &decision, &gapTriggered,
			&valence, &arousal, &surprise, &salience, &outcomeUseful); err != nil {
			return nil, err
		}
		row := map[string]interface{}{
			"confidence":     nullFloat(confidence),
			"decision":       nullString(decision),
			"gap_triggered":  nullInt(gapTriggered),
			"valence":        nullFloat(valence),
			"arousal":        nullFloat(arousal),
			"surprise":       nullFloat(surprise),
			"salience":       nullFloat(salience),
			"outcome_useful": nullInt(outcomeUseful),
		}
		results = append(results, row)
	}
	return results, nil
}

// ExportRetrievalPolicyData returns assessed retrieval decisions for policy training.
func (s *SQLiteStore) ExportRetrievalPolicyData(ctx context.Context) ([]map[string]interface{}, error) {
	rows, err := s.DB().QueryContext(ctx,
		`SELECT layers_queried, graph_depth, mood_weight, top_k,
		        return_count, outcome_helpful
		 FROM retrieval_decisions
		 WHERE outcome_helpful IS NOT NULL`,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []map[string]interface{}
	for rows.Next() {
		var layersQueried sql.NullString
		var graphDepth, topK, returnCount, outcomeHelpful sql.NullInt64
		var moodWeight sql.NullFloat64
		if err := rows.Scan(&layersQueried, &graphDepth, &moodWeight, &topK, &returnCount, &outcomeHelpful); err != nil {
			return nil, err
		}
		row := map[string]interface{}{
			"layers_queried":  nullString(layersQueried),
			"graph_depth":     nullInt(graphDepth),
			"mood_weight":     nullFloat(moodWeight),
			"top_k":           nullInt(topK),
			"return_count":    nullInt(returnCount),
			"outcome_helpful": nullInt(outcomeHelpful),
		}
		results = append(results, row)
	}
	return results, nil
}

// MemoryVectorInfo holds memory ID, session, and vector ID for dream exploration.
type MemoryVectorInfo struct {
	ID        string
	SessionID string
	VectorID  string
}

// GetMemoriesWithVectors returns memories that have vector embeddings (for dream explorer).
func (s *SQLiteStore) GetMemoriesWithVectors(ctx context.Context, tiers []string) ([]MemoryVectorInfo, error) {
	query := "SELECT id, session_id, vector_id FROM memories WHERE vector_id IS NOT NULL"
	var args []interface{}
	if len(tiers) > 0 {
		ph := make([]string, len(tiers))
		for i, t := range tiers {
			ph[i] = "?"
			args = append(args, t)
		}
		query += fmt.Sprintf(" AND tier IN (%s)", strings.Join(ph, ","))
	}

	rows, err := s.DB().QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []MemoryVectorInfo
	for rows.Next() {
		var r MemoryVectorInfo
		if err := rows.Scan(&r.ID, &r.SessionID, &r.VectorID); err != nil {
			return nil, err
		}
		results = append(results, r)
	}
	return results, nil
}

// FindMemoryByRawLogID finds a memory by its raw log ID.
func (s *SQLiteStore) FindMemoryByRawLogID(ctx context.Context, rawLogID string) (*string, error) {
	var memID string
	err := s.DB().QueryRowContext(ctx, "SELECT id FROM memories WHERE raw_log_id = ?", rawLogID).Scan(&memID)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &memID, nil
}

// GetLowestDecayHotMemories returns the IDs and graph refs of the lowest-decay hot memories.
func (s *SQLiteStore) GetLowestDecayHotMemories(ctx context.Context, count int) ([]struct{ ID string; GraphNodeID *string }, error) {
	rows, err := s.DB().QueryContext(ctx,
		"SELECT id, graph_node_id FROM memories WHERE tier = 'hot' ORDER BY decay_score ASC LIMIT ?",
		count,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []struct{ ID string; GraphNodeID *string }
	for rows.Next() {
		var r struct{ ID string; GraphNodeID *string }
		if err := rows.Scan(&r.ID, &r.GraphNodeID); err != nil {
			return nil, err
		}
		results = append(results, r)
	}
	return results, nil
}

// ---------- Helpers ----------

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

func allCold(tiers []string) bool {
	for _, t := range tiers {
		if t != "cold" {
			return false
		}
	}
	return true
}

func nullFloat(n sql.NullFloat64) interface{} {
	if n.Valid {
		return n.Float64
	}
	return nil
}

func nullString(n sql.NullString) interface{} {
	if n.Valid {
		return n.String
	}
	return nil
}

func nullInt(n sql.NullInt64) interface{} {
	if n.Valid {
		return n.Int64
	}
	return nil
}

func (s *SQLiteStore) loadKeywords(ctx context.Context, memoryID string) ([]agentmemory.Keyword, error) {
	rows, err := s.DB().QueryContext(ctx,
		"SELECT keyword, weight FROM memory_keywords WHERE memory_id = ?", memoryID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var kws []agentmemory.Keyword
	for rows.Next() {
		var kw agentmemory.Keyword
		if err := rows.Scan(&kw.Keyword, &kw.Weight); err != nil {
			return nil, err
		}
		kws = append(kws, kw)
	}
	return kws, nil
}

func (s *SQLiteStore) batchLoadKeywords(ctx context.Context, memories []agentmemory.Memory) error {
	if len(memories) == 0 {
		return nil
	}

	ids := make([]string, len(memories))
	idxMap := make(map[string]int)
	for i, m := range memories {
		ids[i] = m.ID
		idxMap[m.ID] = i
	}

	ph := make([]string, len(ids))
	args := make([]interface{}, len(ids))
	for i, id := range ids {
		ph[i] = "?"
		args[i] = id
	}

	rows, err := s.DB().QueryContext(ctx,
		fmt.Sprintf("SELECT memory_id, keyword, weight FROM memory_keywords WHERE memory_id IN (%s)", strings.Join(ph, ",")),
		args...,
	)
	if err != nil {
		return err
	}
	defer rows.Close()

	for rows.Next() {
		var mid, kw string
		var w float64
		if err := rows.Scan(&mid, &kw, &w); err != nil {
			return err
		}
		if idx, ok := idxMap[mid]; ok {
			memories[idx].Keywords = append(memories[idx].Keywords, agentmemory.Keyword{Keyword: kw, Weight: w})
		}
	}
	return nil
}

// scanMemory scans a single row from the memories table.
func scanMemory(row *sql.Row) (*agentmemory.Memory, error) {
	var m agentmemory.Memory
	var fastPathed, isSemantic int
	err := row.Scan(
		&m.ID, &m.CreatedAt, &m.UpdatedAt, &m.Content, &m.Summary,
		&m.RawLogID, &m.SessionID, &m.Turn,
		&m.Valence, &m.Arousal, &m.Surprise, &m.Salience,
		&m.AccessCount, &m.LastAccessed, &m.DecayScore,
		&m.CompactionGen, &m.Tier, &fastPathed, &isSemantic,
		&m.GraphNodeID, &m.VectorID,
		&m.SpatialEmbedding, &m.SceneDescription,
	)
	m.FastPathed = fastPathed != 0
	m.IsSemantic = isSemantic != 0
	return &m, err
}

// scanMemoryRows scans a row from a *sql.Rows cursor over the memories table.
func scanMemoryRows(rows *sql.Rows) (*agentmemory.Memory, error) {
	var m agentmemory.Memory
	var fastPathed, isSemantic int
	err := rows.Scan(
		&m.ID, &m.CreatedAt, &m.UpdatedAt, &m.Content, &m.Summary,
		&m.RawLogID, &m.SessionID, &m.Turn,
		&m.Valence, &m.Arousal, &m.Surprise, &m.Salience,
		&m.AccessCount, &m.LastAccessed, &m.DecayScore,
		&m.CompactionGen, &m.Tier, &fastPathed, &isSemantic,
		&m.GraphNodeID, &m.VectorID,
		&m.SpatialEmbedding, &m.SceneDescription,
	)
	m.FastPathed = fastPathed != 0
	m.IsSemantic = isSemantic != 0
	return &m, err
}

// scanMemoryWithExtra scans a row that has an extra trailing column (e.g. match_score).
func scanMemoryWithExtra(rows *sql.Rows) (*agentmemory.Memory, error) {
	var m agentmemory.Memory
	var fastPathed, isSemantic int
	var extra sql.NullFloat64
	err := rows.Scan(
		&m.ID, &m.CreatedAt, &m.UpdatedAt, &m.Content, &m.Summary,
		&m.RawLogID, &m.SessionID, &m.Turn,
		&m.Valence, &m.Arousal, &m.Surprise, &m.Salience,
		&m.AccessCount, &m.LastAccessed, &m.DecayScore,
		&m.CompactionGen, &m.Tier, &fastPathed, &isSemantic,
		&m.GraphNodeID, &m.VectorID,
		&m.SpatialEmbedding, &m.SceneDescription,
		&extra,
	)
	m.FastPathed = fastPathed != 0
	m.IsSemantic = isSemantic != 0
	return &m, err
}
