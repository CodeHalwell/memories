package core

import (
	"context"
	"fmt"
	"log"
	"strings"

	agentmemory "github.com/CodeHalwell/Memories/go"
	"github.com/CodeHalwell/Memories/go/llm"
	"github.com/CodeHalwell/Memories/go/storage"
)

// IsFastPath checks if a memory should bypass the LLM save decision.
func IsFastPath(cfg agentmemory.MemoryConfig, arousal, surprise float64, content string) bool {
	if arousal > cfg.FastPathArousal && surprise > cfg.FastPathSurprise {
		return true
	}
	lower := strings.ToLower(content)
	markers := []string{"remember this", "don't forget", "save this", "keep in mind"}
	for _, phrase := range markers {
		if strings.Contains(lower, phrase) {
			return true
		}
	}
	return false
}

// GetRetrievalGaps identifies topic areas where recent retrievals returned poor results (A2.1).
func GetRetrievalGaps(ctx context.Context, sqlite *storage.SQLiteStore, cfg agentmemory.MemoryConfig, sessionID string) ([]string, error) {
	return sqlite.GetFailedRetrievalKeywords(ctx, sessionID, cfg.GapLookbackTurns)
}

// ComputeGapOverlap computes overlap between content keywords and retrieval gap keywords (A2.1).
func ComputeGapOverlap(contentKeywords, gapKeywords []string) float64 {
	if len(contentKeywords) == 0 || len(gapKeywords) == 0 {
		return 0.0
	}
	contentSet := make(map[string]bool)
	for _, k := range contentKeywords {
		contentSet[k] = true
	}
	gapSet := make(map[string]bool)
	for _, k := range gapKeywords {
		gapSet[k] = true
	}
	intersectionSize := 0
	for k := range contentSet {
		if gapSet[k] {
			intersectionSize++
		}
	}
	size := len(contentSet)
	if size == 0 {
		size = 1
	}
	return float64(intersectionSize) / float64(size)
}

// MakeSaveDecision decides whether to save an agent output as a memory.
// Returns (SaveDecision, *Memory). Memory is nil if the decision is to skip.
func MakeSaveDecision(
	ctx context.Context,
	cfg agentmemory.MemoryConfig,
	client *llm.Client,
	entry agentmemory.RawLogEntry,
	isFirstTurn bool,
	sqlite *storage.SQLiteStore,
) (agentmemory.SaveDecision, *agentmemory.Memory) {
	// First turn of a session is always saved via fast path
	if isFirstTurn {
		mem := agentmemory.NewMemory()
		mem.Content = entry.Content
		mem.RawLogID = entry.ID
		mem.SessionID = entry.SessionID
		mem.Turn = entry.Turn
		mem.Salience = 0.7
		mem.FastPathed = true

		reason := "First turn of session — always saved"
		dec := agentmemory.NewSaveDecision()
		dec.RawLogID = entry.ID
		dec.SessionID = entry.SessionID
		dec.Turn = entry.Turn
		dec.Decision = "fast_path"
		dec.Reason = &reason
		dec.Confidence = 1.0

		return dec, &mem
	}

	// Ask LLM for structured evaluation
	prompt := fmt.Sprintf(`Evaluate whether this agent output should be saved as a memory:

Session: %s
Turn: %d
Content:
<content>
%s
</content>

Respond with JSON only.`, entry.SessionID, entry.Turn, entry.Content)

	result, err := client.CompleteJSON(ctx, prompt, &cfg.SystemPrompts.SaveDecision, nil, nil)
	if err != nil {
		log.Printf("LLM save decision failed: %v, defaulting to skip", err)
		reason := "LLM evaluation failed"
		dec := agentmemory.NewSaveDecision()
		dec.RawLogID = entry.ID
		dec.SessionID = entry.SessionID
		dec.Turn = entry.Turn
		dec.Decision = "skip"
		dec.Reason = &reason
		dec.Confidence = 0.0
		return dec, nil
	}

	confidence := getFloat(result, "confidence", 0.0)
	shouldSave := getBool(result, "should_save", false)
	valence := getFloat(result, "valence", 0.0)
	arousal := getFloat(result, "arousal", 0.0)
	surprise := getFloat(result, "surprise", 0.0)
	salience := getFloat(result, "salience", 0.5)

	// Extract keywords
	var keywords []agentmemory.Keyword
	if kwList, ok := result["keywords"].([]interface{}); ok {
		for _, item := range kwList {
			if kwMap, ok := item.(map[string]interface{}); ok {
				kw := agentmemory.Keyword{
					Keyword: strings.ToLower(getString(kwMap, "keyword", "")),
					Weight:  getFloat(kwMap, "weight", 1.0),
				}
				if kw.Keyword != "" {
					keywords = append(keywords, kw)
				}
			}
		}
	}
	if len(keywords) > cfg.MaxKeywordsPerMemory {
		keywords = keywords[:cfg.MaxKeywordsPerMemory]
	}

	contentKWNames := make([]string, len(keywords))
	for i, kw := range keywords {
		contentKWNames[i] = kw.Keyword
	}

	// A2.1: Retrieval gap awareness — lower threshold if content fills a gap
	threshold := cfg.SaveConfidenceThreshold
	gapTriggered := false
	if sqlite != nil {
		gapKeywords, err := GetRetrievalGaps(ctx, sqlite, cfg, entry.SessionID)
		if err == nil {
			gapOverlap := ComputeGapOverlap(contentKWNames, gapKeywords)
			if gapOverlap > cfg.GapOverlapThreshold {
				threshold *= cfg.GapThresholdReduction
				gapTriggered = true
			}
		}
	}

	// Check fast path conditions
	fastPath := IsFastPath(cfg, arousal, surprise, entry.Content)

	var decision string
	if fastPath {
		decision = "fast_path"
		shouldSave = true
		if confidence < 0.9 {
			confidence = 0.9
		}
	} else if shouldSave && confidence >= threshold {
		decision = "save"
	} else {
		decision = "skip"
	}

	reasonStr := getString(result, "reason", "")
	dec := agentmemory.NewSaveDecision()
	dec.RawLogID = entry.ID
	dec.SessionID = entry.SessionID
	dec.Turn = entry.Turn
	dec.Decision = decision
	dec.Reason = &reasonStr
	dec.Confidence = confidence
	dec.GapTriggered = gapTriggered
	dec.ThresholdUsed = &threshold

	if decision == "save" || decision == "fast_path" {
		mem := agentmemory.NewMemory()
		mem.Content = entry.Content
		summaryStr := getString(result, "summary", "")
		if summaryStr != "" {
			mem.Summary = &summaryStr
		}
		mem.RawLogID = entry.ID
		mem.SessionID = entry.SessionID
		mem.Turn = entry.Turn
		mem.Valence = valence
		mem.Arousal = arousal
		mem.Surprise = surprise
		mem.Salience = salience
		mem.FastPathed = fastPath
		mem.Keywords = keywords
		return dec, &mem
	}

	return dec, nil
}

// Helper functions for extracting typed values from map[string]interface{}.
func getFloat(m map[string]interface{}, key string, def float64) float64 {
	if v, ok := m[key]; ok {
		switch n := v.(type) {
		case float64:
			return n
		case int:
			return float64(n)
		}
	}
	return def
}

func getBool(m map[string]interface{}, key string, def bool) bool {
	if v, ok := m[key]; ok {
		if b, ok := v.(bool); ok {
			return b
		}
	}
	return def
}

func getString(m map[string]interface{}, key, def string) string {
	if v, ok := m[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return def
}
