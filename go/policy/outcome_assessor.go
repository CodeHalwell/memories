package policy

import (
	"context"
	"log"
	"strings"
	"time"

	agentmemory "github.com/CodeHalwell/Memories/go"
	"github.com/CodeHalwell/Memories/go/storage"
)

// AssessSaveOutcomes assesses whether saved memories turned out to be useful (A4).
// A memory is considered useful if it was retrieved at least once.
// Returns the number of decisions assessed.
func AssessSaveOutcomes(ctx context.Context, sqlite *storage.SQLiteStore, cfg agentmemory.MemoryConfig) (int, error) {
	lookbackDays := cfg.SaveOutcomeLookbackDays
	now := time.Now().UTC().Format(time.RFC3339)

	unassessed, err := sqlite.GetUnassessedSaveDecisions(ctx, lookbackDays)
	if err != nil {
		return 0, err
	}

	updated := 0
	for _, row := range unassessed {
		useful := row.AccessCount != nil && *row.AccessCount > 0
		if err := sqlite.UpdateSaveOutcome(ctx, row.ID, useful, now); err != nil {
			continue
		}
		updated++
	}

	log.Printf("Save outcome assessment: assessed %d decisions", updated)
	return updated, nil
}

// AssessRetrievalOutcomes assesses whether retrievals were helpful (A4).
// Heuristic: if the agent did not re-query the same topic within N turns,
// the retrieval was probably adequate.
func AssessRetrievalOutcomes(ctx context.Context, sqlite *storage.SQLiteStore, cfg agentmemory.MemoryConfig) (int, error) {
	followupTurns := cfg.RetrievalOutcomeFollowupTurns
	overlapThreshold := cfg.RetrievalOutcomeKeywordOverlap
	now := time.Now().UTC().Format(time.RFC3339)

	unassessed, err := sqlite.GetUnassessedRetrievalDecisions(ctx)
	if err != nil {
		return 0, err
	}

	updated := 0
	for _, row := range unassessed {
		if row.Turn == nil {
			continue
		}

		followups, err := sqlite.GetRetrievalFollowups(ctx, row.SessionID, *row.Turn, followupTurns)
		if err != nil {
			continue
		}

		originalKeywords := extractKeywordSet(row.Query)
		reQueried := false

		for _, fuQuery := range followups {
			fuKeywords := extractKeywordSet(fuQuery)
			if len(originalKeywords) > 0 && len(fuKeywords) > 0 {
				intersection := 0
				for k := range originalKeywords {
					if fuKeywords[k] {
						intersection++
					}
				}
				overlap := float64(intersection) / float64(max(len(originalKeywords), 1))
				if overlap > overlapThreshold {
					reQueried = true
					break
				}
			}
		}

		helpful := !reQueried
		if err := sqlite.UpdateRetrievalOutcome(ctx, row.ID, helpful, now); err != nil {
			continue
		}
		updated++
	}

	log.Printf("Retrieval outcome assessment: assessed %d decisions", updated)
	return updated, nil
}

func extractKeywordSet(query string) map[string]bool {
	kws := make(map[string]bool)
	for _, w := range strings.Fields(query) {
		w = strings.ToLower(strings.TrimSpace(w))
		if len(w) > 2 {
			kws[w] = true
		}
	}
	return kws
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
