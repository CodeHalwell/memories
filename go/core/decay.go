// Package core implements the core memory processing algorithms.
package core

import (
	"math"
	"time"

	agentmemory "github.com/CodeHalwell/Memories/go"
)

// ComputeDecay computes a decay score between 0.0 and ~1.0.
// Higher scores indicate more "alive" memories. Combines recency (exponential
// decay), frequency (log-scaled access count), emotional boost (A2.2), and
// semantic floor (A2.2).
func ComputeDecay(cfg agentmemory.MemoryConfig, lastAccessed time.Time, accessCount int, arousal, surprise float64, isSemantic bool) float64 {
	now := time.Now().UTC()
	if lastAccessed.IsZero() {
		lastAccessed = now
	}

	daysSince := math.Max(now.Sub(lastAccessed).Hours()/24.0, 0.0)

	halflife := cfg.DecayHalflifeDays
	lambda := math.Log(2) / halflife
	if halflife <= 0 {
		lambda = 0.1
	}

	// A2.2: Emotional memories decay more slowly
	// arousal + surprise in [0, 2], so boost is in [1.0, 2.0]
	emotionalBoost := 1.0 + 0.5*(arousal+surprise)
	recency := math.Exp(-lambda * daysSince / emotionalBoost)

	frequency := math.Log1p(float64(accessCount)) / 10.0

	// A2.2: Semantic (compacted) memories have a flatter decay curve
	if isSemantic {
		recency = math.Max(recency, 0.3)
	}

	score := cfg.DecayRecencyWeight*recency + cfg.DecayFrequencyWeight*frequency
	return math.Round(score*10000) / 10000
}
