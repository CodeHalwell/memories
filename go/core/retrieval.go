package core

import (
	"context"
	"fmt"
	"log"
	"math"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"

	agentmemory "github.com/CodeHalwell/Memories/go"
	"github.com/CodeHalwell/Memories/go/embeddings"
	"github.com/CodeHalwell/Memories/go/emotion"
	"github.com/CodeHalwell/Memories/go/llm"
	"github.com/CodeHalwell/Memories/go/storage"
)

// RetrievalEngine orchestrates multi-layer retrieval across all storage backends.
type RetrievalEngine struct {
	SQLite         *storage.SQLiteStore
	Graph          *storage.GraphStore
	Vector         *storage.VectorStore
	TextEmbedder   *embeddings.TextEmbedder
	VisualEmbedder *embeddings.VisualEmbedder
	LLMClient      *llm.Client
	Config         agentmemory.MemoryConfig
	LogDir         string
}

type candidate struct {
	MemoryID string
	Score    float64
	Layers   map[string]bool
}

// Retrieve runs all retrieval layers and returns ranked, deduplicated memories.
func (r *RetrievalEngine) Retrieve(ctx context.Context, query string, sessionID *string, topK *int, enableMoodCongruent, enableVisual bool) ([]agentmemory.Memory, error) {
	k := r.Config.TopKPerLayer
	if topK != nil {
		k = *topK
	}

	// Score context for mood-congruent weighting
	var contextEmotion *agentmemory.EmotionScores
	if enableMoodCongruent && r.LLMClient != nil {
		scores, err := emotion.ScoreEmotion(ctx, r.LLMClient, r.Config, query)
		if err == nil {
			contextEmotion = &scores
		}
	}

	// Run layers concurrently
	type layerResult struct {
		Name    string
		Results []idScore
		Err     error
	}

	var wg sync.WaitGroup
	ch := make(chan layerResult, 4)

	wg.Add(3)
	go func() { defer wg.Done(); res, err := r.grepLayer(ctx, query, k); ch <- layerResult{"grep", res, err} }()
	go func() { defer wg.Done(); res, err := r.keywordLayer(ctx, query, k); ch <- layerResult{"keyword", res, err} }()
	go func() { defer wg.Done(); res, err := r.semanticLayer(ctx, query, k); ch <- layerResult{"semantic", res, err} }()

	if enableVisual && r.VisualEmbedder != nil {
		wg.Add(1)
		go func() { defer wg.Done(); res, err := r.visualLayer(ctx, query, k); ch <- layerResult{"visual", res, err} }()
	}

	go func() { wg.Wait(); close(ch) }()

	candidates := make(map[string]*candidate)
	for lr := range ch {
		if lr.Err != nil {
			log.Printf("Retrieval layer %s failed: %v", lr.Name, lr.Err)
			continue
		}
		for _, is := range lr.Results {
			if c, ok := candidates[is.ID]; ok {
				c.Score += is.Score
				c.Layers[lr.Name] = true
			} else {
				candidates[is.ID] = &candidate{
					MemoryID: is.ID,
					Score:    is.Score,
					Layers:   map[string]bool{lr.Name: true},
				}
			}
		}
	}

	// Graph traversal expansion
	for memID := range candidates {
		related, err := r.Graph.GetRelatedMemories(ctx, memID, r.Config.GraphTraversalDepth, 0.0)
		if err != nil {
			continue
		}
		for _, rel := range related {
			if _, exists := candidates[rel.ID]; !exists {
				depthScore := 1.0 / float64(rel.Depth+1) * rel.Salience
				candidates[rel.ID] = &candidate{
					MemoryID: rel.ID,
					Score:    depthScore,
					Layers:   map[string]bool{"graph_traversal": true},
				}
			}
		}
	}

	// Load full memories and apply scoring
	type scoredMem struct {
		Mem   agentmemory.Memory
		Score float64
	}
	var memories []scoredMem

	for _, cand := range candidates {
		mem, err := r.SQLite.GetMemory(ctx, cand.MemoryID)
		if err != nil || mem == nil {
			continue
		}

		score := cand.Score

		// Mood-congruent boosting
		if contextEmotion != nil && enableMoodCongruent {
			moodWeight := r.Config.MoodCongruentWeight
			valenceSim := 1.0 - math.Abs(contextEmotion.Valence-mem.Valence)/2.0
			arousalSim := 1.0 - math.Abs(contextEmotion.Arousal-mem.Arousal)
			moodBonus := (valenceSim + arousalSim) / 2.0 * moodWeight
			score += moodBonus
		}

		score *= mem.DecayScore
		memories = append(memories, scoredMem{*mem, score})
	}

	// Sort by score descending
	sortScoredMems(memories)

	// Log access and update decay for returned memories
	now := time.Now().UTC().Format(time.RFC3339)
	limit := k * 2
	if limit > len(memories) {
		limit = len(memories)
	}

	var resultMemories []agentmemory.Memory
	for _, sm := range memories[:limit] {
		mem := sm.Mem
		cand := candidates[mem.ID]

		accessType := "primary"
		if cand != nil {
			if cand.Layers["graph_traversal"] {
				accessType = "graph_traversal"
			} else if cand.Layers["grep"] && len(cand.Layers) == 1 {
				accessType = "grep_entrypoint"
			} else if cand.Layers["semantic"] {
				accessType = "vector"
			}
		}

		mem.AccessCount++
		mem.LastAccessed = &now
		lastAccessedTime, _ := time.Parse(time.RFC3339, now)
		mem.DecayScore = ComputeDecay(r.Config, lastAccessedTime, mem.AccessCount, mem.Arousal, mem.Surprise, mem.IsSemantic)

		sid := sessionID
		q := &query
		_ = r.SQLite.LogAccess(ctx, uuid.New().String(), mem.ID, now, accessType, func() *string { if sid != nil { return sid }; return nil }(), q)
		_ = r.SQLite.UpdateMemoryAccess(ctx, mem.ID, mem.DecayScore, mem.AccessCount, now)

		resultMemories = append(resultMemories, mem)
	}

	return resultMemories, nil
}

type idScore struct {
	ID    string
	Score float64
}

func (r *RetrievalEngine) grepLayer(ctx context.Context, query string, limit int) ([]idScore, error) {
	words := strings.Fields(query)
	if len(words) > 5 {
		words = words[:5]
	}
	var results []idScore
	for _, word := range words {
		if len(word) <= 2 {
			continue
		}
		keyword := strings.ToLower(word)
		mems, err := r.SQLite.SearchByKeywords(ctx, []string{keyword}, limit)
		if err != nil {
			continue
		}
		for _, m := range mems {
			results = append(results, idScore{m.ID, 1.0})
		}
	}
	return dedup(results, limit), nil
}

func (r *RetrievalEngine) keywordLayer(ctx context.Context, query string, limit int) ([]idScore, error) {
	keywords := extractKeywords(query)
	if len(keywords) == 0 {
		return nil, nil
	}
	memories, err := r.SQLite.SearchByKeywords(ctx, keywords, limit)
	if err != nil {
		return nil, err
	}
	results := make([]idScore, len(memories))
	for i, m := range memories {
		results[i] = idScore{m.ID, m.DecayScore}
	}
	return results, nil
}

func (r *RetrievalEngine) semanticLayer(ctx context.Context, query string, limit int) ([]idScore, error) {
	if r.TextEmbedder == nil {
		return nil, nil
	}
	qv, err := r.TextEmbedder.Embed(ctx, query)
	if err != nil {
		return nil, err
	}
	searchResults, err := r.Vector.SearchText(ctx, qv, limit, nil)
	if err != nil {
		return nil, err
	}
	results := make([]idScore, len(searchResults))
	for i, sr := range searchResults {
		results[i] = idScore{sr.MemoryID, sr.Score}
	}
	return results, nil
}

func (r *RetrievalEngine) visualLayer(ctx context.Context, query string, limit int) ([]idScore, error) {
	if r.VisualEmbedder == nil {
		return nil, nil
	}
	qv, err := r.VisualEmbedder.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("visual embed failed: %w", err)
	}
	searchResults, err := r.Vector.SearchVisual(ctx, qv, limit)
	if err != nil {
		return nil, err
	}
	results := make([]idScore, len(searchResults))
	for i, sr := range searchResults {
		results[i] = idScore{sr.MemoryID, sr.Score}
	}
	return results, nil
}

func extractKeywords(query string) []string {
	var kws []string
	for _, w := range strings.Fields(query) {
		w = strings.ToLower(strings.TrimSpace(w))
		if len(w) > 2 {
			kws = append(kws, w)
		}
	}
	return kws
}

func dedup(items []idScore, limit int) []idScore {
	seen := make(map[string]bool)
	var result []idScore
	for _, it := range items {
		if !seen[it.ID] {
			seen[it.ID] = true
			result = append(result, it)
			if len(result) >= limit {
				break
			}
		}
	}
	return result
}

func sortScoredMems(items []struct {
	Mem   agentmemory.Memory
	Score float64
}) {
	// Use a simple sort
	for i := 0; i < len(items); i++ {
		for j := i + 1; j < len(items); j++ {
			if items[j].Score > items[i].Score {
				items[i], items[j] = items[j], items[i]
			}
		}
	}
}
