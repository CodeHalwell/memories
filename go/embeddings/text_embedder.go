// Package embeddings provides interfaces and implementations for text and visual embedding.
package embeddings

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// TextEmbedder generates text embeddings via an HTTP embedding API.
type TextEmbedder struct {
	BaseURL   string
	APIKey    string
	Model     string
	Dimension int
	client    *http.Client
}

// NewTextEmbedder creates a TextEmbedder targeting the given API.
func NewTextEmbedder(baseURL, apiKey, model string, dimension int) *TextEmbedder {
	return &TextEmbedder{
		BaseURL:   baseURL,
		APIKey:    apiKey,
		Model:     model,
		Dimension: dimension,
		client:    &http.Client{},
	}
}

// Embed embeds a single text string. Returns a slice of float64.
func (t *TextEmbedder) Embed(ctx context.Context, text string) ([]float64, error) {
	results, err := t.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, fmt.Errorf("empty embedding result")
	}
	return results[0], nil
}

// EmbedBatch embeds multiple texts.
func (t *TextEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	body := map[string]interface{}{
		"input": texts,
		"model": t.Model,
	}
	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", t.BaseURL+"/embeddings", bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if t.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+t.APIKey)
	}

	resp, err := t.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embedding request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embedding API error %d: %s", resp.StatusCode, string(respBody))
	}

	var result struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decoding embedding response: %w", err)
	}

	vectors := make([][]float64, len(result.Data))
	for i, d := range result.Data {
		vectors[i] = d.Embedding
	}
	return vectors, nil
}

// GetDimension returns the embedding dimension.
func (t *TextEmbedder) GetDimension() int {
	return t.Dimension
}
