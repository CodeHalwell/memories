package agentmemory

import (
	"path/filepath"
	"time"
)

// DefaultDataDir is the default base directory for all runtime data.
const DefaultDataDir = "data"

// Prompts groups all system prompt strings used by LLM operations.
type Prompts struct {
	SceneDescription      string `json:"scene_description"`
	SaveDecision          string `json:"save_decision"`
	Merge                 string `json:"merge"`
	SyntheticQuery        string `json:"synthetic_query"`
	Emotion               string `json:"emotion"`
	ClassifyRelationship  string `json:"classify_relationship"`
}

// MemoryConfig holds all tunable parameters for the agent memory system.
type MemoryConfig struct {
	// Base data directory — all runtime data stored here.
	DataDir       string `json:"data_dir"`
	LogDir        string `json:"log_dir"`
	DBPath        string `json:"db_path"`
	GraphDir      string `json:"graph_dir"`
	VectorDir     string `json:"vector_dir"`
	PolicyDataDir string `json:"policy_data_dir"`

	// LLM
	LLMModel       string  `json:"llm_model"`
	LLMTemperature float64 `json:"llm_temperature"`
	LLMBaseURL     string  `json:"llm_base_url"`
	LLMAPIKey      string  `json:"llm_api_key"`

	// Save thresholds
	SaveConfidenceThreshold float64 `json:"save_confidence_threshold"`
	FastPathArousal         float64 `json:"fast_path_arousal"`
	FastPathSurprise        float64 `json:"fast_path_surprise"`
	MaxKeywordsPerMemory    int     `json:"max_keywords_per_memory"`

	// Save decision — retrieval gap awareness (A2.1)
	GapLookbackTurns    int     `json:"gap_lookback_turns"`
	GapOverlapThreshold float64 `json:"gap_overlap_threshold"`
	GapThresholdReduction float64 `json:"gap_threshold_reduction"`

	// Retrieval
	RetrievalLayers      []string `json:"retrieval_layers"`
	GraphTraversalDepth  int      `json:"graph_traversal_depth"`
	MoodCongruentWeight  float64  `json:"mood_congruent_weight"`
	TopKPerLayer         int      `json:"top_k_per_layer"`

	// Compaction
	HotTierThreshold              int     `json:"hot_tier_threshold"`
	CompactionCandidateThreshold  float64 `json:"compaction_candidate_threshold"`
	KeywordOverlapMergeThreshold  float64 `json:"keyword_overlap_merge_threshold"`
	ValenceMergeExclusionDelta    float64 `json:"valence_merge_exclusion_delta"`

	// Compaction — merge validation (A2.4)
	MergeValidationQueries    int     `json:"merge_validation_queries"`
	MergeDegradationTolerance float64 `json:"merge_degradation_tolerance"`

	// Compaction — generation gap guard (A2.3)
	MaxGenerationGapForMerge int `json:"max_generation_gap_for_merge"`

	// Decay
	DecayRecencyWeight   float64 `json:"decay_recency_weight"`
	DecayFrequencyWeight float64 `json:"decay_frequency_weight"`
	DecayHalflifeDays    float64 `json:"decay_halflife_days"`

	// Visual layer
	VisualSalienceThreshold float64 `json:"visual_salience_threshold"`
	CLIPModel               string  `json:"clip_model"`

	// Embeddings
	TextEmbeddingModel string `json:"text_embedding_model"`
	TextEmbeddingDim   int    `json:"text_embedding_dim"`
	VisualEmbeddingDim int    `json:"visual_embedding_dim"`

	// Embedding API
	EmbeddingBaseURL string `json:"embedding_base_url"`
	EmbeddingAPIKey  string `json:"embedding_api_key"`

	// Qdrant
	QdrantURL string `json:"qdrant_url"`

	// Dream exploration (A3)
	DreamWalkCount          int     `json:"dream_walk_count"`
	DreamSimilarityThreshold float64 `json:"dream_similarity_threshold"`
	DreamMaxNewEdges        int     `json:"dream_max_new_edges"`
	DreamClusterMinSize     int     `json:"dream_cluster_min_size"`
	DreamEnabled            bool    `json:"dream_enabled"`

	// Policy logging (A4)
	PolicyLoggingEnabled             bool    `json:"policy_logging_enabled"`
	SaveOutcomeLookbackDays          int     `json:"save_outcome_lookback_days"`
	RetrievalOutcomeFollowupTurns    int     `json:"retrieval_outcome_followup_turns"`
	RetrievalOutcomeKeywordOverlap   float64 `json:"retrieval_outcome_keyword_overlap"`

	// Policy training (A5, v2)
	PolicyMinSaveExamples      int `json:"policy_min_save_examples"`
	PolicyMinRetrievalExamples int `json:"policy_min_retrieval_examples"`

	// System Prompts
	SystemPrompts Prompts `json:"prompts"`
}

// DefaultConfig returns a MemoryConfig with sensible defaults matching the Python implementation.
func DefaultConfig() MemoryConfig {
	dataDir := DefaultDataDir
	return MemoryConfig{
		DataDir:       dataDir,
		LogDir:        filepath.Join(dataDir, "logs", "sessions"),
		DBPath:        filepath.Join(dataDir, "memory.db"),
		GraphDir:      filepath.Join(dataDir, "graph"),
		VectorDir:     filepath.Join(dataDir, "vectors"),
		PolicyDataDir: filepath.Join(dataDir, "policy_data"),

		LLMModel:       "claude-sonnet-4-6",
		LLMTemperature: 0.2,

		SaveConfidenceThreshold: 0.5,
		FastPathArousal:         0.85,
		FastPathSurprise:        0.75,
		MaxKeywordsPerMemory:    10,

		GapLookbackTurns:      20,
		GapOverlapThreshold:   0.3,
		GapThresholdReduction: 0.7,

		RetrievalLayers:     []string{"grep", "keyword", "semantic"},
		GraphTraversalDepth: 2,
		MoodCongruentWeight: 0.2,
		TopKPerLayer:        5,

		HotTierThreshold:             500,
		CompactionCandidateThreshold: 0.7,
		KeywordOverlapMergeThreshold: 0.6,
		ValenceMergeExclusionDelta:   0.6,

		MergeValidationQueries:    5,
		MergeDegradationTolerance: 0.15,
		MaxGenerationGapForMerge:  1,

		DecayRecencyWeight:   0.6,
		DecayFrequencyWeight: 0.4,
		DecayHalflifeDays:    7,

		VisualSalienceThreshold: 0.7,
		CLIPModel:               "ViT-B-32",

		TextEmbeddingModel: "all-MiniLM-L6-v2",
		TextEmbeddingDim:   384,
		VisualEmbeddingDim: 512,

		QdrantURL: "http://localhost:6333",

		DreamWalkCount:           50,
		DreamSimilarityThreshold: 0.7,
		DreamMaxNewEdges:         20,
		DreamClusterMinSize:      3,
		DreamEnabled:             true,

		PolicyLoggingEnabled:           true,
		SaveOutcomeLookbackDays:        30,
		RetrievalOutcomeFollowupTurns:  3,
		RetrievalOutcomeKeywordOverlap: 0.5,

		PolicyMinSaveExamples:      1000,
		PolicyMinRetrievalExamples: 500,

		SystemPrompts: Prompts{
			SceneDescription:     SceneDescriptionSystem,
			SaveDecision:         SaveDecisionSystem,
			Merge:                MergeSystem,
			SyntheticQuery:       SyntheticQuerySystem,
			Emotion:              EmotionSystem,
			ClassifyRelationship: ClassifyRelationshipSystem,
		},
	}
}

// NowUTC returns the current UTC time. Used by models to stamp creation times.
func NowUTC() time.Time {
	return time.Now().UTC()
}

// NowISO returns the current UTC time as an ISO 8601 string.
func NowISO() string {
	return NowUTC().Format(time.RFC3339)
}
