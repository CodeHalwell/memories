package core

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"strings"
	"time"

	"github.com/google/uuid"

	agentmemory "github.com/CodeHalwell/Memories/go"
	"github.com/CodeHalwell/Memories/go/embeddings"
	"github.com/CodeHalwell/Memories/go/llm"
	"github.com/CodeHalwell/Memories/go/storage"
)

// ClassifyRelationship asks the LLM to classify the relationship between two memories.
func ClassifyRelationship(ctx context.Context, client *llm.Client, cfg agentmemory.MemoryConfig, memAContent, memBContent string) string {
	prompt := "Memory A:\n<memory_a>\n" + memAContent + "\n</memory_a>\n\n" +
		"Memory B:\n<memory_b>\n" + memBContent + "\n</memory_b>\n\n" +
		"What is the relationship between Memory A and Memory B?"

	temp := 0.1
	response, err := client.Complete(ctx, prompt, &cfg.SystemPrompts.ClassifyRelationship, nil, &temp)
	if err != nil {
		return "unrelated"
	}

	result := strings.ToLower(strings.TrimSpace(response))
	valid := map[string]bool{
		"caused": true, "supports": true, "contradicts": true,
		"precedes": true, "part_of": true, "analogous": true, "unrelated": true,
	}
	if valid[result] {
		return result
	}
	return "unrelated"
}

// ExploratoryWalk performs semi-random walks to discover non-obvious memory connections (A3).
func ExploratoryWalk(
	ctx context.Context,
	cfg agentmemory.MemoryConfig,
	client *llm.Client,
	sqlite *storage.SQLiteStore,
	graph *storage.GraphStore,
	vector *storage.VectorStore,
	textEmbedder *embeddings.TextEmbedder,
	nWalks *int,
	similarityThreshold *float64,
	maxNewEdges *int,
) ([]agentmemory.DiscoveredEdge, error) {
	walks := cfg.DreamWalkCount
	if nWalks != nil {
		walks = *nWalks
	}
	simThreshold := cfg.DreamSimilarityThreshold
	if similarityThreshold != nil {
		simThreshold = *similarityThreshold
	}
	maxEdges := cfg.DreamMaxNewEdges
	if maxNewEdges != nil {
		maxEdges = *maxNewEdges
	}

	var discovered []agentmemory.DiscoveredEdge

	allMemories, err := sqlite.GetMemoriesWithVectors(ctx, []string{"hot", "warm"})
	if err != nil || len(allMemories) < 2 {
		return discovered, err
	}

	if textEmbedder == nil {
		return discovered, nil
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	for i := 0; i < walks; i++ {
		if len(discovered) >= maxEdges {
			break
		}

		idxA := rng.Intn(len(allMemories))
		idxB := rng.Intn(len(allMemories))
		if idxA == idxB {
			continue
		}
		a, b := allMemories[idxA], allMemories[idxB]

		if a.SessionID == b.SessionID {
			continue
		}

		exists, _ := graph.PathExists(ctx, a.ID, b.ID, 1)
		if exists {
			continue
		}

		sim, err := vector.Similarity(ctx, a.VectorID, b.VectorID)
		if err != nil || sim == nil || *sim < simThreshold {
			continue
		}

		memA, _ := sqlite.GetMemory(ctx, a.ID)
		memB, _ := sqlite.GetMemory(ctx, b.ID)
		if memA == nil || memB == nil {
			continue
		}

		relType := ClassifyRelationship(ctx, client, cfg, memA.Content, memB.Content)
		if relType != "unrelated" {
			discovered = append(discovered, agentmemory.DiscoveredEdge{
				SourceID:         a.ID,
				TargetID:         b.ID,
				Similarity:       *sim,
				RelationshipType: relType,
				DiscoveryMethod:  "random_walk",
			})
		}
	}

	if len(discovered) > maxEdges {
		discovered = discovered[:maxEdges]
	}
	return discovered, nil
}

// CommitDiscoveries commits discovered edges to the graph and logs the exploration run (A3).
func CommitDiscoveries(
	ctx context.Context,
	discoveries []agentmemory.DiscoveredEdge,
	graph *storage.GraphStore,
	sqlite *storage.SQLiteStore,
	runID *string,
) (int, error) {
	now := time.Now().UTC().Format(time.RFC3339)
	rid := uuid.New().String()
	if runID != nil {
		rid = *runID
	}

	strategiesSet := make(map[string]bool)
	for _, d := range discoveries {
		strategiesSet[d.DiscoveryMethod] = true
	}
	strategies := make([]string, 0, len(strategiesSet))
	for s := range strategiesSet {
		strategies = append(strategies, s)
	}

	committed := 0
	for _, edge := range discoveries {
		edgeID := uuid.New().String()
		err := graph.AddRelatesTo(ctx, edge.SourceID, edge.TargetID, edge.Similarity, edge.RelationshipType, now)
		if err == nil {
			committed++
			_ = sqlite.LogDreamEdge(ctx, edgeID, rid, edge.SourceID, edge.TargetID, edge.Similarity, edge.RelationshipType, edge.DiscoveryMethod, true)
		} else {
			_ = sqlite.LogDreamEdge(ctx, edgeID, rid, edge.SourceID, edge.TargetID, edge.Similarity, edge.RelationshipType, edge.DiscoveryMethod, false)
		}
	}

	notes := fmt.Sprintf("Committed %d/%d edges", committed, len(discoveries))
	_ = sqlite.LogDreamRun(ctx, rid, now, len(discoveries), len(discoveries), committed, strategies, &notes)

	log.Printf("Dream exploration: committed %d/%d edges", committed, len(discoveries))
	return committed, nil
}
