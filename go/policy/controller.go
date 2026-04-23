// Package policy provides the policy controller and outcome assessment for the memory system.
package policy

import (
	agentmemory "github.com/CodeHalwell/Memories/go"
)

// HardConstraints that cannot be overridden by any learned policy.
var HardConstraints = map[string]interface{}{
	"min_save_rate":               0.05,
	"max_save_rate":               0.50,
	"fast_path_override":          true,
	"min_layers":                  1,
	"max_graph_depth":             4,
	"max_top_k":                   20,
	"never_delete_raw_logs":       true,
	"never_compact_fast_path_gen0": true,
	"require_merge_validation":    true,
}

// PolicyState is the state vector for policy decisions (A5.2).
type PolicyState struct {
	TurnNumber               int     `json:"turn_number"`
	SessionLength            int     `json:"session_length"`
	TimeSinceLastSave        float64 `json:"time_since_last_save"`
	ContentLength            int     `json:"content_length"`
	EmotionalValence         float64 `json:"emotional_valence"`
	EmotionalArousal         float64 `json:"emotional_arousal"`
	EmotionalSurprise        float64 `json:"emotional_surprise"`
	HotTierCount             int     `json:"hot_tier_count"`
	RecentRetrievalHitRate   float64 `json:"recent_retrieval_hit_rate"`
	RetrievalGapScore        float64 `json:"retrieval_gap_score"`
	GraphNodeCount           int     `json:"graph_node_count"`
	AvgEdgeDegree            float64 `json:"avg_edge_degree"`
	OrphanMemoryCount        int     `json:"orphan_memory_count"`
	DaysSinceLastCompaction  float64 `json:"days_since_last_compaction"`
	PendingMergeCandidates   int     `json:"pending_merge_candidates"`
}

// Controller is the stub policy controller — uses heuristics, logs decisions for future training.
type Controller struct {
	Constraints map[string]interface{}
	Config      agentmemory.MemoryConfig
}

// NewController creates a PolicyController.
func NewController(cfg agentmemory.MemoryConfig) *Controller {
	return &Controller{
		Constraints: HardConstraints,
		Config:      cfg,
	}
}

// ShouldSave decides whether to save (heuristic — v1).
func (c *Controller) ShouldSave(state PolicyState, llmConfidence float64) bool {
	return llmConfidence >= c.Config.SaveConfidenceThreshold
}

// RetrievalConfig returns retrieval parameters (heuristic — v1).
func (c *Controller) RetrievalConfig(state PolicyState) map[string]interface{} {
	maxGraphDepth := c.Config.GraphTraversalDepth
	if maxD, ok := c.Constraints["max_graph_depth"].(int); ok && maxGraphDepth > maxD {
		maxGraphDepth = maxD
	}
	maxTopK := c.Config.TopKPerLayer
	if maxK, ok := c.Constraints["max_top_k"].(int); ok && maxTopK > maxK {
		maxTopK = maxK
	}
	return map[string]interface{}{
		"layers":      c.Config.RetrievalLayers,
		"graph_depth": maxGraphDepth,
		"mood_weight": c.Config.MoodCongruentWeight,
		"top_k":       maxTopK,
	}
}

// CompactionPriority returns compaction urgency score (heuristic — v1).
func (c *Controller) CompactionPriority(state PolicyState) float64 {
	if state.HotTierCount > c.Config.HotTierThreshold {
		return 1.0
	}
	threshold := c.Config.HotTierThreshold
	if threshold == 0 {
		threshold = 1
	}
	return float64(state.HotTierCount) / float64(threshold)
}
