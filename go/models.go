package agentmemory

import (
	"time"

	"github.com/google/uuid"
)

// Keyword represents a keyword with its relevance weight.
type Keyword struct {
	Keyword string  `json:"keyword"`
	Weight  float64 `json:"weight"`
}

// RawLogEntry is an immutable raw agent output record.
type RawLogEntry struct {
	ID         string `json:"id"`
	SessionID  string `json:"session_id"`
	Turn       int    `json:"turn"`
	Timestamp  string `json:"timestamp"`
	Role       string `json:"role"`
	Content    string `json:"content"`
	TokenCount int    `json:"token_count"`
	Model      string `json:"model"`
	Provider   string `json:"provider"`
}

// NewRawLogEntry creates a RawLogEntry with auto-generated ID and timestamp.
func NewRawLogEntry() RawLogEntry {
	return RawLogEntry{
		ID:        uuid.New().String(),
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		Role:      "assistant",
	}
}

// Memory is the primary memory data structure with emotional metadata, salience, and decay.
type Memory struct {
	ID        string `json:"id"`
	CreatedAt string `json:"created_at"`
	UpdatedAt string `json:"updated_at"`
	Content   string `json:"content"`
	Summary   *string `json:"summary,omitempty"`
	RawLogID  string `json:"raw_log_id"`
	SessionID string `json:"session_id"`
	Turn      int    `json:"turn"`

	// Emotional metadata
	Valence  float64 `json:"valence"`
	Arousal  float64 `json:"arousal"`
	Surprise float64 `json:"surprise"`

	// Salience and access
	Salience     float64 `json:"salience"`
	AccessCount  int     `json:"access_count"`
	LastAccessed *string `json:"last_accessed,omitempty"`
	DecayScore   float64 `json:"decay_score"`

	// Compaction state
	CompactionGen int    `json:"compaction_gen"`
	Tier          string `json:"tier"`
	FastPathed    bool   `json:"fast_pathed"`
	IsSemantic    bool   `json:"is_semantic"`

	// Cross-store references
	GraphNodeID *string `json:"graph_node_id,omitempty"`
	VectorID    *string `json:"vector_id,omitempty"`

	// Visual layer
	SpatialEmbedding []byte  `json:"spatial_embedding,omitempty"`
	SceneDescription *string `json:"scene_description,omitempty"`

	// Keywords (not stored in main table — separate table)
	Keywords []Keyword `json:"keywords,omitempty"`
}

// NewMemory creates a Memory with auto-generated ID and timestamps.
func NewMemory() Memory {
	now := NowISO()
	return Memory{
		ID:         uuid.New().String(),
		CreatedAt:  now,
		UpdatedAt:  now,
		Salience:   0.5,
		DecayScore: 1.0,
		Tier:       "hot",
	}
}

// SaveDecision logs a save/skip/fast_path decision for a raw log entry.
type SaveDecision struct {
	ID           string  `json:"id"`
	RawLogID     string  `json:"raw_log_id"`
	SessionID    string  `json:"session_id"`
	Turn         int     `json:"turn"`
	DecidedAt    string  `json:"decided_at"`
	Decision     string  `json:"decision"` // save | skip | fast_path
	Reason       *string `json:"reason,omitempty"`
	Confidence   float64 `json:"confidence"`
	// A2.1 — retrieval gap awareness
	GapTriggered  bool     `json:"gap_triggered"`
	ThresholdUsed *float64 `json:"threshold_used,omitempty"`
}

// NewSaveDecision creates a SaveDecision with auto-generated ID and timestamp.
func NewSaveDecision() SaveDecision {
	return SaveDecision{
		ID:        uuid.New().String(),
		DecidedAt: NowISO(),
		Decision:  "skip",
	}
}

// CompactionResult tracks the outcome of a compaction run.
type CompactionResult struct {
	ID               string  `json:"id"`
	RanAt            string  `json:"ran_at"`
	Trigger          string  `json:"trigger"`
	MemoriesReviewed int     `json:"memories_reviewed"`
	MemoriesMerged   int     `json:"memories_merged"`
	MemoriesPruned   int     `json:"memories_pruned"`
	Notes            *string `json:"notes,omitempty"`
	// A2.5 / A3 — addendum tracking
	KeywordsUpdated  int `json:"keywords_updated"`
	EdgesDiscovered  int `json:"edges_discovered"`
}

// NewCompactionResult creates a CompactionResult with auto-generated ID and timestamp.
func NewCompactionResult() CompactionResult {
	return CompactionResult{
		ID:      uuid.New().String(),
		RanAt:   NowISO(),
		Trigger: "scheduled",
	}
}

// MergeValidation holds the result of generative replay validation for a compaction merge (A2.4).
type MergeValidation struct {
	Passed         bool     `json:"passed"`
	AvgSourceScore float64  `json:"avg_source_score"`
	AvgMergedScore float64  `json:"avg_merged_score"`
	Degradation    float64  `json:"degradation"`
	QueriesTested  []string `json:"queries_tested"`
}

// DiscoveredEdge represents an edge discovered during exploratory graph walks (A3).
type DiscoveredEdge struct {
	SourceID         string  `json:"source_id"`
	TargetID         string  `json:"target_id"`
	Similarity       float64 `json:"similarity"`
	RelationshipType string  `json:"relationship_type"`
	DiscoveryMethod  string  `json:"discovery_method"` // random_walk | cluster_bridge | temporal_proximity
}

// EmotionScores holds emotional dimension scores.
type EmotionScores struct {
	Valence  float64 `json:"valence"`
	Arousal  float64 `json:"arousal"`
	Surprise float64 `json:"surprise"`
}
