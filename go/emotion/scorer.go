// Package emotion provides emotional scoring via LLM.
package emotion

import (
	"context"
	"fmt"
	"math"

	agentmemory "github.com/CodeHalwell/Memories/go"
	"github.com/CodeHalwell/Memories/go/llm"
)

// ScoreEmotion scores the emotional dimensions of a text.
// Returns EmotionScores with valence, arousal, surprise.
func ScoreEmotion(ctx context.Context, client *llm.Client, cfg agentmemory.MemoryConfig, text string) (agentmemory.EmotionScores, error) {
	prompt := fmt.Sprintf("Score the emotional tone of this text:\n\n<text>\n%s\n</text>", text)

	result, err := client.CompleteJSON(ctx, prompt, &cfg.SystemPrompts.Emotion, nil, nil)
	if err != nil {
		return agentmemory.EmotionScores{}, err
	}

	valence, _ := toFloat64(result["valence"])
	arousal, _ := toFloat64(result["arousal"])
	surprise, _ := toFloat64(result["surprise"])

	return agentmemory.EmotionScores{
		Valence:  clamp(valence, -1.0, 1.0),
		Arousal:  clamp(arousal, 0.0, 1.0),
		Surprise: clamp(surprise, 0.0, 1.0),
	}, nil
}

func clamp(value, lo, hi float64) float64 {
	return math.Max(lo, math.Min(hi, value))
}

func toFloat64(v interface{}) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	default:
		return 0, false
	}
}
