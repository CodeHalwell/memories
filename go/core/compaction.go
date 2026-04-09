package core

import (
	"context"
	"fmt"
	"log"
	"math"
	"strings"
	"time"

	"github.com/google/uuid"

	agentmemory "github.com/CodeHalwell/Memories/go"
	"github.com/CodeHalwell/Memories/go/embeddings"
	"github.com/CodeHalwell/Memories/go/llm"
	"github.com/CodeHalwell/Memories/go/storage"
)

// CompactionScore scores a memory for compaction candidacy.
// Low decay + low salience = good candidate.
func CompactionScore(mem agentmemory.Memory) float64 {
	return (1-mem.DecayScore)*0.6 + (1-mem.Salience)*0.4
}

// KeywordOverlap computes Jaccard similarity between two memories' keyword sets.
func KeywordOverlap(a, b agentmemory.Memory) float64 {
	kwA := make(map[string]bool)
	for _, kw := range a.Keywords {
		kwA[kw.Keyword] = true
	}
	kwB := make(map[string]bool)
	for _, kw := range b.Keywords {
		kwB[kw.Keyword] = true
	}
	if len(kwA) == 0 || len(kwB) == 0 {
		return 0.0
	}
	intersection := 0
	union := make(map[string]bool)
	for k := range kwA {
		union[k] = true
		if kwB[k] {
			intersection++
		}
	}
	for k := range kwB {
		union[k] = true
	}
	if len(union) == 0 {
		return 0.0
	}
	return float64(intersection) / float64(len(union))
}

// CanMerge checks if a group of memories can be merged (no exclusion conditions).
func CanMerge(group []agentmemory.Memory, cfg agentmemory.MemoryConfig) bool {
	for i := 0; i < len(group); i++ {
		for j := i + 1; j < len(group); j++ {
			a, b := group[i], group[j]
			// Opposite valence exclusion
			if a.Valence*b.Valence < 0 && math.Abs(a.Valence-b.Valence) > cfg.ValenceMergeExclusionDelta {
				return false
			}
			// Either is fast_pathed gen-0
			if (a.FastPathed && a.CompactionGen == 0) || (b.FastPathed && b.CompactionGen == 0) {
				return false
			}
			// A2.3: Generation gap guard
			if abs(a.CompactionGen-b.CompactionGen) > cfg.MaxGenerationGapForMerge {
				return false
			}
		}
	}
	return true
}

// GroupByKeywords groups memories by keyword overlap using greedy clustering.
func GroupByKeywords(candidates []agentmemory.Memory, threshold float64) [][]agentmemory.Memory {
	if len(candidates) == 0 {
		return nil
	}

	used := make(map[int]bool)
	var groups [][]agentmemory.Memory

	for i, memA := range candidates {
		if used[i] {
			continue
		}
		group := []agentmemory.Memory{memA}
		used[i] = true
		for j, memB := range candidates {
			if used[j] {
				continue
			}
			allAbove := true
			for _, g := range group {
				if KeywordOverlap(g, memB) < threshold {
					allAbove = false
					break
				}
			}
			if allAbove {
				group = append(group, memB)
				used[j] = true
			}
		}
		if len(group) > 1 {
			groups = append(groups, group)
		}
	}
	return groups
}

// CosineSimilarityVec computes cosine similarity between two vectors.
func CosineSimilarityVec(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0.0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	norm := math.Sqrt(normA) * math.Sqrt(normB)
	if norm == 0 {
		return 0.0
	}
	return dot / norm
}

// ValidateMerge validates a merge via generative replay (A2.4).
func ValidateMerge(
	ctx context.Context,
	cfg agentmemory.MemoryConfig,
	client *llm.Client,
	textEmbedder *embeddings.TextEmbedder,
	sourceMemories []agentmemory.Memory,
	candidateContent string,
	nQueries int,
	degradationTolerance float64,
) (agentmemory.MergeValidation, error) {
	var sourceTexts []string
	for _, m := range sourceMemories {
		sourceTexts = append(sourceTexts, m.Content)
	}
	sourceText := strings.Join(sourceTexts, "\n\n")
	prompt := fmt.Sprintf("Generate %d search queries for this content:\n\n<source_text>\n%s\n</source_text>", nQueries, sourceText)

	queriesRaw, err := client.CompleteJSONArray(ctx, prompt, &cfg.SystemPrompts.SyntheticQuery)
	if err != nil {
		log.Printf("Synthetic query generation failed: %v", err)
		return agentmemory.MergeValidation{Passed: false}, nil
	}

	var queries []string
	for _, q := range queriesRaw {
		if s, ok := q.(string); ok {
			queries = append(queries, s)
		}
	}
	if len(queries) == 0 {
		return agentmemory.MergeValidation{Passed: true, QueriesTested: queries}, nil
	}

	candidateEmb, err := textEmbedder.Embed(ctx, candidateContent)
	if err != nil {
		return agentmemory.MergeValidation{Passed: false}, err
	}

	sourceEmbs := make([][]float64, len(sourceMemories))
	for i, m := range sourceMemories {
		emb, err := textEmbedder.Embed(ctx, m.Content)
		if err != nil {
			return agentmemory.MergeValidation{Passed: false}, err
		}
		sourceEmbs[i] = emb
	}

	var mergedScores, sourceScores []float64
	for _, q := range queries {
		qEmb, err := textEmbedder.Embed(ctx, q)
		if err != nil {
			continue
		}
		mergedScores = append(mergedScores, CosineSimilarityVec(qEmb, candidateEmb))
		bestSource := 0.0
		for _, se := range sourceEmbs {
			sim := CosineSimilarityVec(qEmb, se)
			if sim > bestSource {
				bestSource = sim
			}
		}
		sourceScores = append(sourceScores, bestSource)
	}

	avgMerged := avg(mergedScores)
	avgSource := avg(sourceScores)
	degradation := avgSource - avgMerged

	return agentmemory.MergeValidation{
		Passed:         degradation < degradationTolerance,
		AvgSourceScore: avgSource,
		AvgMergedScore: avgMerged,
		Degradation:    degradation,
		QueriesTested:  queries,
	}, nil
}

// CompactionEngine runs compaction cycles.
type CompactionEngine struct {
	SQLite         *storage.SQLiteStore
	Graph          *storage.GraphStore
	Vector         *storage.VectorStore
	TextEmbedder   *embeddings.TextEmbedder
	VisualEmbedder *embeddings.VisualEmbedder
	LLMClient      *llm.Client
	Config         agentmemory.MemoryConfig
}

// Run executes a full compaction cycle.
func (c *CompactionEngine) Run(ctx context.Context, trigger string) (agentmemory.CompactionResult, error) {
	result := agentmemory.NewCompactionResult()
	result.Trigger = trigger

	candidates, err := c.SQLite.GetCompactionCandidates(ctx, c.Config.CompactionCandidateThreshold)
	if err != nil {
		return result, err
	}
	result.MemoriesReviewed = len(candidates)

	if len(candidates) == 0 {
		_ = c.SQLite.LogCompactionRun(ctx, result)
		return result, nil
	}

	// Filter by graph edge count
	var filtered []agentmemory.Memory
	for _, mem := range candidates {
		if mem.GraphNodeID != nil {
			edgeCount, _ := c.Graph.GetEdgeCount(ctx, mem.ID)
			if edgeCount > 3 {
				continue
			}
		}
		filtered = append(filtered, mem)
	}

	groups := GroupByKeywords(filtered, c.Config.KeywordOverlapMergeThreshold)

	mergedCount := 0
	for _, group := range groups {
		if !CanMerge(group, c.Config) {
			continue
		}
		newMem, err := c.mergeGroup(ctx, group, result.ID)
		if err != nil {
			log.Printf("Merge failed: %v", err)
			continue
		}
		if newMem != nil {
			mergedCount++
		}
	}
	result.MemoriesMerged = mergedCount

	// Tier promotion
	hot := "hot"
	hotCount, _ := c.SQLite.CountMemories(ctx, &hot)
	if hotCount > c.Config.HotTierThreshold {
		c.promoteTier(ctx, hotCount-c.Config.HotTierThreshold)
	}

	notes := fmt.Sprintf("Reviewed %d, merged into %d semantic memories", result.MemoriesReviewed, mergedCount)
	result.Notes = &notes
	_ = c.SQLite.LogCompactionRun(ctx, result)

	return result, nil
}

func (c *CompactionEngine) mergeGroup(ctx context.Context, group []agentmemory.Memory, compactionID string) (*agentmemory.Memory, error) {
	var sources []string
	for i, m := range group {
		sources = append(sources, fmt.Sprintf("Memory %d (salience=%.2f, valence=%.2f):\n<memory_%d>\n%s\n</memory_%d>",
			i+1, m.Salience, m.Valence, i+1, m.Content, i+1))
	}
	prompt := fmt.Sprintf("Merge these %d related memories into a single generalised memory:\n\n%s\n\nRespond with JSON only.",
		len(group), strings.Join(sources, "\n\n"))

	result, err := c.LLMClient.CompleteJSON(ctx, prompt, &c.Config.SystemPrompts.Merge, nil, nil)
	if err != nil {
		return nil, err
	}

	mergedContent := getString(result, "content", "")

	// A2.4: Validate merge
	if c.TextEmbedder != nil {
		validation, _ := ValidateMerge(ctx, c.Config, c.LLMClient, c.TextEmbedder,
			group, mergedContent, c.Config.MergeValidationQueries, c.Config.MergeDegradationTolerance)
		if !validation.Passed {
			log.Printf("Merge validation failed (degradation=%.3f)", validation.Degradation)
			return nil, nil
		}

		vp := validation.Passed
		_ = c.SQLite.LogCompactionMerge(ctx, compactionID,
			memoryIDs(group), "", &vp, &validation.AvgSourceScore, &validation.AvgMergedScore, &validation.Degradation)
	}

	now := time.Now().UTC().Format(time.RFC3339)
	newID := uuid.New().String()
	maxGen := 0
	for _, m := range group {
		if m.CompactionGen > maxGen {
			maxGen = m.CompactionGen
		}
	}

	var keywords []agentmemory.Keyword
	if kwList, ok := result["keywords"].([]interface{}); ok {
		for _, item := range kwList {
			if kwMap, ok := item.(map[string]interface{}); ok {
				keywords = append(keywords, agentmemory.Keyword{
					Keyword: strings.ToLower(getString(kwMap, "keyword", "")),
					Weight:  getFloat(kwMap, "weight", 1.0),
				})
			}
		}
	}
	if len(keywords) > c.Config.MaxKeywordsPerMemory {
		keywords = keywords[:c.Config.MaxKeywordsPerMemory]
	}

	summaryStr := getString(result, "summary", "")
	newMem := agentmemory.Memory{
		ID:            newID,
		CreatedAt:     now,
		UpdatedAt:     now,
		Content:       mergedContent,
		Summary:       &summaryStr,
		RawLogID:      group[0].RawLogID,
		SessionID:     group[0].SessionID,
		Turn:          group[0].Turn,
		Valence:       getFloat(result, "valence", 0.0),
		Arousal:       getFloat(result, "arousal", 0.0),
		Salience:      getFloat(result, "salience", 0.5),
		CompactionGen: maxGen + 1,
		Tier:          "warm",
		IsSemantic:    true,
		DecayScore:    1.0,
		Keywords:      keywords,
	}

	_ = c.SQLite.SaveMemory(ctx, newMem)
	_ = c.Graph.AddMemoryNode(ctx, newID, summaryStr, "warm", newMem.Salience, newMem.Valence, newMem.CompactionGen, now)
	_ = c.SQLite.UpdateMemoryGraphRef(ctx, newID, newID)

	sourceIDs := memoryIDs(group)
	for _, src := range group {
		_ = c.Graph.AddEvolvedFrom(ctx, newID, src.ID, compactionID, now)
	}
	_ = c.Graph.ReplicateEdgesToNewNode(ctx, sourceIDs, newID)

	for _, src := range group {
		_ = c.SQLite.UpdateMemoryTier(ctx, src.ID, "cold")
		if src.GraphNodeID != nil {
			_ = c.Graph.UpdateMemoryTier(ctx, src.ID, "cold")
		}
	}

	if c.TextEmbedder != nil {
		vector, err := c.TextEmbedder.Embed(ctx, newMem.Content)
		if err == nil {
			pointID, err := c.Vector.UpsertTextVector(ctx, newID, vector, "warm", newMem.Valence, newMem.Arousal, newMem.SessionID, now)
			if err == nil {
				_ = c.SQLite.UpdateMemoryVectorRef(ctx, newID, pointID)
			}
		}
	}

	return &newMem, nil
}

func (c *CompactionEngine) promoteTier(ctx context.Context, count int) {
	rows, err := c.SQLite.GetLowestDecayHotMemories(ctx, count)
	if err != nil {
		return
	}
	for _, row := range rows {
		_ = c.SQLite.UpdateMemoryTier(ctx, row.ID, "warm")
		if row.GraphNodeID != nil {
			_ = c.Graph.UpdateMemoryTier(ctx, row.ID, "warm")
		}
	}
}

func memoryIDs(mems []agentmemory.Memory) []string {
	ids := make([]string, len(mems))
	for i, m := range mems {
		ids[i] = m.ID
	}
	return ids
}

func avg(vals []float64) float64 {
	if len(vals) == 0 {
		return 0.0
	}
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
