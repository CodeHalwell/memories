package core

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/google/uuid"

	agentmemory "github.com/CodeHalwell/Memories/go"
	"github.com/CodeHalwell/Memories/go/embeddings"
	"github.com/CodeHalwell/Memories/go/llm"
	"github.com/CodeHalwell/Memories/go/policy"
	"github.com/CodeHalwell/Memories/go/storage"
)

// MemoryManager is the top-level orchestrator for the agent memory system.
type MemoryManager struct {
	Config         agentmemory.MemoryConfig
	JSONL          *storage.JSONLLogger
	SQLite         *storage.SQLiteStore
	Graph          *storage.GraphStore
	Vector         *storage.VectorStore
	TextEmbedder   *embeddings.TextEmbedder
	VisualEmbedder *embeddings.VisualEmbedder
	LLMClient      *llm.Client

	retrieval  *RetrievalEngine
	compaction *CompactionEngine
	firstTurns map[string]bool
}

// NewMemoryManager creates a MemoryManager with the given config.
func NewMemoryManager(cfg agentmemory.MemoryConfig) *MemoryManager {
	return &MemoryManager{
		Config:     cfg,
		firstTurns: make(map[string]bool),
	}
}

// Initialize sets up all storage backends and sub-engines.
func (mm *MemoryManager) Initialize(ctx context.Context) error {
	var err error

	mm.JSONL, err = storage.NewJSONLLogger(mm.Config.LogDir)
	if err != nil {
		return fmt.Errorf("initializing JSONL logger: %w", err)
	}

	mm.SQLite = storage.NewSQLiteStore(mm.Config.DBPath)
	if err := mm.SQLite.Initialize(ctx); err != nil {
		return fmt.Errorf("initializing SQLite: %w", err)
	}

	mm.Graph = storage.NewGraphStore(mm.Config.GraphDir)
	if err := mm.Graph.Initialize(ctx); err != nil {
		return fmt.Errorf("initializing graph store: %w", err)
	}

	mm.Vector = storage.NewVectorStore(mm.Config.QdrantURL)

	if mm.Config.EmbeddingBaseURL != "" {
		mm.TextEmbedder = embeddings.NewTextEmbedder(mm.Config.EmbeddingBaseURL, mm.Config.EmbeddingAPIKey, mm.Config.TextEmbeddingModel, mm.Config.TextEmbeddingDim)
		mm.VisualEmbedder = embeddings.NewVisualEmbedder(mm.Config.EmbeddingBaseURL, mm.Config.EmbeddingAPIKey, mm.Config.CLIPModel, mm.Config.VisualEmbeddingDim)
	}

	if mm.Config.LLMBaseURL != "" {
		mm.LLMClient = llm.NewClient(mm.Config.LLMBaseURL, mm.Config.LLMAPIKey, mm.Config.LLMModel, mm.Config.LLMTemperature)
	}

	if err := mm.Vector.Initialize(ctx, mm.Config.TextEmbeddingDim, mm.Config.VisualEmbeddingDim); err != nil {
		log.Printf("Warning: vector store initialization failed: %v", err)
	}

	mm.retrieval = &RetrievalEngine{
		SQLite:         mm.SQLite,
		Graph:          mm.Graph,
		Vector:         mm.Vector,
		TextEmbedder:   mm.TextEmbedder,
		VisualEmbedder: mm.VisualEmbedder,
		LLMClient:      mm.LLMClient,
		Config:         mm.Config,
		LogDir:         mm.Config.LogDir,
	}

	mm.compaction = &CompactionEngine{
		SQLite:         mm.SQLite,
		Graph:          mm.Graph,
		Vector:         mm.Vector,
		TextEmbedder:   mm.TextEmbedder,
		VisualEmbedder: mm.VisualEmbedder,
		LLMClient:      mm.LLMClient,
		Config:         mm.Config,
	}

	return nil
}

// InitializeLite initializes without embedding models (for testing).
func (mm *MemoryManager) InitializeLite(ctx context.Context) error {
	var err error
	mm.JSONL, err = storage.NewJSONLLogger(mm.Config.LogDir)
	if err != nil {
		return err
	}
	mm.SQLite = storage.NewSQLiteStore(mm.Config.DBPath)
	if err := mm.SQLite.Initialize(ctx); err != nil {
		return err
	}
	mm.Graph = storage.NewGraphStore(mm.Config.GraphDir)
	return mm.Graph.Initialize(ctx)
}

// Close closes all storage backends.
func (mm *MemoryManager) Close() error {
	var firstErr error
	if mm.SQLite != nil {
		if err := mm.SQLite.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if mm.Graph != nil {
		if err := mm.Graph.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if mm.Vector != nil {
		if err := mm.Vector.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// ProcessTurn logs an agent output and decides whether to save it as a memory.
func (mm *MemoryManager) ProcessTurn(ctx context.Context, sessionID string, turn int, content, role string, tokenCount int, model, provider string) (*agentmemory.Memory, error) {
	entry := agentmemory.NewRawLogEntry()
	entry.SessionID = sessionID
	entry.Turn = turn
	entry.Content = content
	entry.Role = role
	entry.TokenCount = tokenCount
	entry.Model = model
	entry.Provider = provider

	filePath, byteOffset, err := mm.JSONL.Append(entry)
	if err != nil {
		return nil, fmt.Errorf("appending raw log: %w", err)
	}

	if err := mm.SQLite.IndexRawLog(ctx, entry.ID, sessionID, turn, entry.Timestamp, filePath, byteOffset); err != nil {
		return nil, fmt.Errorf("indexing raw log: %w", err)
	}

	isFirst := !mm.firstTurns[sessionID]
	if isFirst {
		mm.firstTurns[sessionID] = true
	}

	decision, memory := MakeSaveDecision(ctx, mm.Config, mm.LLMClient, entry, isFirst, mm.SQLite)
	if err := mm.SQLite.LogSaveDecision(ctx, decision); err != nil {
		log.Printf("Failed to log save decision: %v", err)
	}

	if memory == nil {
		return nil, nil
	}

	if err := mm.SQLite.SaveMemory(ctx, *memory); err != nil {
		return nil, fmt.Errorf("saving memory: %w", err)
	}

	// Create graph node
	_ = mm.Graph.AddMemoryNode(ctx, memory.ID, ptrStr(memory.Summary), memory.Tier, memory.Salience, memory.Valence, memory.CompactionGen, memory.CreatedAt)
	memory.GraphNodeID = &memory.ID
	_ = mm.SQLite.UpdateMemoryGraphRef(ctx, memory.ID, memory.ID)

	// Create text embedding
	if mm.TextEmbedder != nil {
		textVector, err := mm.TextEmbedder.Embed(ctx, memory.Content)
		if err == nil {
			pointID, err := mm.Vector.UpsertTextVector(ctx, memory.ID, textVector, memory.Tier, memory.Valence, memory.Arousal, sessionID, memory.CreatedAt)
			if err == nil {
				memory.VectorID = &pointID
				_ = mm.SQLite.UpdateMemoryVectorRef(ctx, memory.ID, pointID)
			}
		}
	}

	// Visual layer for salient memories
	if memory.Salience > mm.Config.VisualSalienceThreshold {
		mm.generateVisualLayer(ctx, memory)
	}

	return memory, nil
}

// Retrieve runs multi-layer retrieval and returns ranked memories.
func (mm *MemoryManager) Retrieve(ctx context.Context, query string, sessionID *string, topK *int) ([]agentmemory.Memory, error) {
	if mm.retrieval == nil {
		return nil, agentmemory.ErrNotInitialized
	}

	memories, err := mm.retrieval.Retrieve(ctx, query, sessionID, topK, true, true)
	if err != nil {
		return nil, err
	}

	// A4: Log retrieval decision
	if mm.Config.PolicyLoggingEnabled {
		now := time.Now().UTC().Format(time.RFC3339)
		k := mm.Config.TopKPerLayer
		if topK != nil {
			k = *topK
		}
		memIDs := make([]string, len(memories))
		for i, m := range memories {
			memIDs[i] = m.ID
		}
		sid := ""
		if sessionID != nil {
			sid = *sessionID
		}
		_ = mm.SQLite.LogRetrievalDecision(ctx, uuid.New().String(), sid, nil, query, now,
			mm.Config.RetrievalLayers, mm.Config.GraphTraversalDepth, mm.Config.MoodCongruentWeight,
			k, memIDs, len(memories))
	}

	return memories, nil
}

// RunCompaction runs a compaction cycle with all phases.
func (mm *MemoryManager) RunCompaction(ctx context.Context, trigger string) (agentmemory.CompactionResult, error) {
	if mm.compaction == nil {
		return agentmemory.CompactionResult{}, agentmemory.ErrNotInitialized
	}

	// Phase 1: Standard compaction
	result, err := mm.compaction.Run(ctx, trigger)
	if err != nil {
		return result, err
	}

	// Phase 2: Keyword reweighting (A2.5)
	kwUpdated, err := ReweightKeywordsFromGraph(ctx, mm.SQLite, mm.Graph, 2, 50)
	if err != nil {
		log.Printf("Keyword reweighting failed: %v", err)
	} else {
		result.KeywordsUpdated = kwUpdated
	}

	// Phase 3: Dream exploration (A3)
	if (trigger == "scheduled" || trigger == "manual") && mm.Config.DreamEnabled {
		discoveries, err := ExploratoryWalk(ctx, mm.Config, mm.LLMClient, mm.SQLite, mm.Graph, mm.Vector, mm.TextEmbedder, nil, nil, nil)
		if err != nil {
			log.Printf("Dream exploration failed: %v", err)
		} else if len(discoveries) > 0 {
			committed, _ := CommitDiscoveries(ctx, discoveries, mm.Graph, mm.SQLite, nil)
			result.EdgesDiscovered = committed
		}
	}

	// Phase 4: Outcome assessment (A4)
	if mm.Config.PolicyLoggingEnabled {
		_, _ = policy.AssessSaveOutcomes(ctx, mm.SQLite, mm.Config)
		_, _ = policy.AssessRetrievalOutcomes(ctx, mm.SQLite, mm.Config)
	}

	return result, nil
}

// GetMemory fetches a single memory and logs the access.
func (mm *MemoryManager) GetMemory(ctx context.Context, memoryID string) (*agentmemory.Memory, error) {
	mem, err := mm.SQLite.GetMemory(ctx, memoryID)
	if err != nil {
		return nil, err
	}
	if mem == nil {
		return nil, nil
	}

	now := time.Now().UTC().Format(time.RFC3339)
	mem.AccessCount++
	mem.LastAccessed = &now
	lastAccessedTime, _ := time.Parse(time.RFC3339, now)
	mem.DecayScore = ComputeDecay(mm.Config, lastAccessedTime, mem.AccessCount, mem.Arousal, mem.Surprise, mem.IsSemantic)

	_ = mm.SQLite.LogAccess(ctx, uuid.New().String(), memoryID, now, "primary", nil, nil)
	_ = mm.SQLite.UpdateMemoryAccess(ctx, memoryID, mem.DecayScore, mem.AccessCount, now)

	return mem, nil
}

func (mm *MemoryManager) generateVisualLayer(ctx context.Context, memory *agentmemory.Memory) {
	if mm.LLMClient == nil || mm.VisualEmbedder == nil {
		return
	}
	scene, err := mm.LLMClient.Complete(ctx,
		fmt.Sprintf("Generate an abstract scene description for this memory:\n\n<memory_content>\n%s\n</memory_content>", memory.Content),
		&mm.Config.SystemPrompts.SceneDescription, nil, nil)
	if err != nil {
		return
	}
	scene = trimStr(scene)
	memory.SceneDescription = &scene

	spatialBytes, err := mm.VisualEmbedder.EmbedToBytes(ctx, scene)
	if err != nil {
		return
	}
	memory.SpatialEmbedding = spatialBytes

	visualVector, err := mm.VisualEmbedder.Embed(ctx, scene)
	if err != nil {
		return
	}
	_, _ = mm.Vector.UpsertVisualVector(ctx, memory.ID, visualVector, memory.SessionID, memory.CreatedAt)
	_ = mm.SQLite.UpdateMemoryVisual(ctx, memory.ID, scene, spatialBytes)
}

func ptrStr(s *string) string {
	if s != nil {
		return *s
	}
	return ""
}

func trimStr(s string) string {
	// Trim whitespace
	for len(s) > 0 && (s[0] == ' ' || s[0] == '\n' || s[0] == '\r' || s[0] == '\t') {
		s = s[1:]
	}
	for len(s) > 0 && (s[len(s)-1] == ' ' || s[len(s)-1] == '\n' || s[len(s)-1] == '\r' || s[len(s)-1] == '\t') {
		s = s[:len(s)-1]
	}
	return s
}
