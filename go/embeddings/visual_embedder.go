package embeddings

import (
	"context"
	"encoding/binary"
	"math"
)

// VisualEmbedder generates visual/spatial embeddings via an HTTP API (CLIP-based).
type VisualEmbedder struct {
	BaseURL   string
	APIKey    string
	Model     string
	Dimension int
	embedder  *TextEmbedder
}

// NewVisualEmbedder creates a VisualEmbedder. It reuses the TextEmbedder HTTP logic
// pointed at a CLIP-compatible embedding endpoint.
func NewVisualEmbedder(baseURL, apiKey, model string, dimension int) *VisualEmbedder {
	return &VisualEmbedder{
		BaseURL:   baseURL,
		APIKey:    apiKey,
		Model:     model,
		Dimension: dimension,
		embedder:  NewTextEmbedder(baseURL, apiKey, model, dimension),
	}
}

// Embed embeds a scene description text using the visual embedding model.
func (v *VisualEmbedder) Embed(ctx context.Context, text string) ([]float64, error) {
	return v.embedder.Embed(ctx, text)
}

// EmbedToBytes embeds and returns raw bytes for storage in a SQLite BLOB column.
func (v *VisualEmbedder) EmbedToBytes(ctx context.Context, text string) ([]byte, error) {
	floats, err := v.Embed(ctx, text)
	if err != nil {
		return nil, err
	}
	buf := make([]byte, len(floats)*4)
	for i, f := range floats {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(float32(f)))
	}
	return buf, nil
}

// BytesToVector converts raw bytes back to a float64 slice.
func BytesToVector(data []byte) []float64 {
	count := len(data) / 4
	result := make([]float64, count)
	for i := 0; i < count; i++ {
		bits := binary.LittleEndian.Uint32(data[i*4:])
		result[i] = float64(math.Float32frombits(bits))
	}
	return result
}

// GetDimension returns the visual embedding dimension.
func (v *VisualEmbedder) GetDimension() int {
	return v.Dimension
}
