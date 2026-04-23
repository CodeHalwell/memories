package core

import (
	"context"
	"log"
	"sort"

	"github.com/CodeHalwell/Memories/go/storage"
)

// ReweightKeywordsFromGraph adjusts keyword weights based on graph connectivity (A2.5).
// Keywords shared across well-connected memories get boosted.
// Returns the number of keyword weights updated.
func ReweightKeywordsFromGraph(ctx context.Context, sqlite *storage.SQLiteStore, graph *storage.GraphStore, maxHops, maxMemoriesPerKeyword int) (int, error) {
	rows, err := sqlite.GetAllKeywordsWithMemories(ctx, nil)
	if err != nil {
		return 0, err
	}
	if len(rows) == 0 {
		return 0, nil
	}

	// Group by keyword
	keywordIndex := make(map[string][]storage.KeywordMemoryAssociation)
	for _, r := range rows {
		keywordIndex[r.Keyword] = append(keywordIndex[r.Keyword], r)
	}

	var pendingUpdates []storage.KeywordWeightUpdate

	for keyword, entries := range keywordIndex {
		if len(entries) < 2 {
			continue
		}

		// Cap and sort by weight desc, then memory_id for stability
		sort.Slice(entries, func(i, j int) bool {
			if entries[i].Weight != entries[j].Weight {
				return entries[i].Weight > entries[j].Weight
			}
			return entries[i].MemoryID < entries[j].MemoryID
		})
		if len(entries) > maxMemoriesPerKeyword {
			entries = entries[:maxMemoriesPerKeyword]
		}

		memoryIDs := make([]string, len(entries))
		for i, e := range entries {
			memoryIDs[i] = e.MemoryID
		}

		connectedPairs := 0
		totalPairs := 0
		for i := 0; i < len(memoryIDs); i++ {
			for j := i + 1; j < len(memoryIDs); j++ {
				totalPairs++
				exists, err := graph.PathExists(ctx, memoryIDs[i], memoryIDs[j], maxHops)
				if err == nil && exists {
					connectedPairs++
				}
			}
		}

		if totalPairs == 0 {
			continue
		}

		connectivityRatio := float64(connectedPairs) / float64(totalPairs)
		if connectivityRatio <= 0.0 {
			continue
		}

		// Scale: 0.0 connectivity = no change, 1.0 = +50% weight
		boost := 1.0 + 0.5*connectivityRatio

		for _, entry := range entries {
			newWeight := entry.Weight * boost
			if newWeight > 1.0 {
				newWeight = 1.0
			}
			if newWeight != entry.Weight {
				pendingUpdates = append(pendingUpdates, storage.KeywordWeightUpdate{
					Weight:   newWeight,
					MemoryID: entry.MemoryID,
					Keyword:  keyword,
				})
			}
		}
	}

	if len(pendingUpdates) > 0 {
		if err := sqlite.BatchUpdateKeywordWeights(ctx, pendingUpdates); err != nil {
			return 0, err
		}
	}

	log.Printf("Keyword reweighting: updated %d weights", len(pendingUpdates))
	return len(pendingUpdates), nil
}
