package storage

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"

	"github.com/google/uuid"
)

const (
	// TextCollection is the Qdrant collection for text embeddings.
	TextCollection = "memory_text"
	// VisualCollection is the Qdrant collection for visual (CLIP) embeddings.
	VisualCollection = "memory_visual"
)

// VectorStore provides vector similarity search via Qdrant REST API.
type VectorStore struct {
	baseURL    string
	httpClient *http.Client
}

// NewVectorStore creates a new VectorStore targeting the given Qdrant REST API URL.
func NewVectorStore(baseURL string) *VectorStore {
	return &VectorStore{
		baseURL:    baseURL,
		httpClient: &http.Client{},
	}
}

// Initialize ensures the text and visual collections exist in Qdrant.
func (v *VectorStore) Initialize(ctx context.Context, textDim, visualDim int) error {
	if err := v.ensureCollection(ctx, TextCollection, textDim); err != nil {
		return fmt.Errorf("ensuring text collection: %w", err)
	}
	if err := v.ensureCollection(ctx, VisualCollection, visualDim); err != nil {
		return fmt.Errorf("ensuring visual collection: %w", err)
	}
	return nil
}

// Close is a no-op for the REST-based vector store.
func (v *VectorStore) Close() error {
	return nil
}

func (v *VectorStore) ensureCollection(ctx context.Context, name string, dim int) error {
	// Check if collection exists
	resp, err := v.doRequest(ctx, "GET", fmt.Sprintf("/collections/%s", name), nil)
	if err == nil && resp.StatusCode == http.StatusOK {
		resp.Body.Close()
		return nil
	}
	if resp != nil {
		resp.Body.Close()
	}

	// Create collection
	body := map[string]interface{}{
		"vectors": map[string]interface{}{
			"size":     dim,
			"distance": "Cosine",
		},
	}
	resp, err = v.doRequest(ctx, "PUT", fmt.Sprintf("/collections/%s", name), body)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("failed to create collection %s: %s", name, string(data))
	}
	return nil
}

// ---------- Text embeddings ----------

// UpsertTextVector inserts or updates a text embedding. Returns the point ID.
func (v *VectorStore) UpsertTextVector(
	ctx context.Context,
	memoryID string, vector []float64,
	tier string, valence, arousal float64,
	sessionID, createdAt string,
) (string, error) {
	pointID := uuid.New().String()

	body := map[string]interface{}{
		"points": []map[string]interface{}{
			{
				"id":     pointID,
				"vector": vector,
				"payload": map[string]interface{}{
					"memory_id":  memoryID,
					"tier":       tier,
					"valence":    valence,
					"arousal":    arousal,
					"session_id": sessionID,
					"created_at": createdAt,
				},
			},
		},
	}

	resp, err := v.doRequest(ctx, "PUT", fmt.Sprintf("/collections/%s/points", TextCollection), body)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("upsert text vector failed: %s", string(data))
	}
	return pointID, nil
}

// TextSearchResult holds a result from text vector search.
type TextSearchResult struct {
	MemoryID string  `json:"memory_id"`
	Score    float64 `json:"score"`
	Tier     string  `json:"tier"`
	Valence  float64 `json:"valence"`
	Arousal  float64 `json:"arousal"`
}

// SearchText searches for nearest text embeddings.
func (v *VectorStore) SearchText(ctx context.Context, queryVector []float64, limit int, tierFilter *string) ([]TextSearchResult, error) {
	body := map[string]interface{}{
		"vector":       queryVector,
		"limit":        limit,
		"with_payload": true,
	}

	if tierFilter != nil {
		body["filter"] = map[string]interface{}{
			"must": []map[string]interface{}{
				{
					"key": "tier",
					"match": map[string]interface{}{
						"value": *tierFilter,
					},
				},
			},
		}
	}

	resp, err := v.doRequest(ctx, "POST", fmt.Sprintf("/collections/%s/points/search", TextCollection), body)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		Result []struct {
			ID      interface{}            `json:"id"`
			Score   float64                `json:"score"`
			Payload map[string]interface{} `json:"payload"`
		} `json:"result"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decoding search response: %w", err)
	}

	var results []TextSearchResult
	for _, r := range result.Result {
		ts := TextSearchResult{
			Score: r.Score,
		}
		if p := r.Payload; p != nil {
			if v, ok := p["memory_id"].(string); ok {
				ts.MemoryID = v
			}
			if v, ok := p["tier"].(string); ok {
				ts.Tier = v
			}
			if v, ok := p["valence"].(float64); ok {
				ts.Valence = v
			}
			if v, ok := p["arousal"].(float64); ok {
				ts.Arousal = v
			}
		}
		results = append(results, ts)
	}
	return results, nil
}

// ---------- Visual embeddings ----------

// UpsertVisualVector inserts or updates a visual (CLIP) embedding. Returns the point ID.
func (v *VectorStore) UpsertVisualVector(
	ctx context.Context,
	memoryID string, vector []float64,
	sessionID, createdAt string,
) (string, error) {
	pointID := uuid.New().String()

	body := map[string]interface{}{
		"points": []map[string]interface{}{
			{
				"id":     pointID,
				"vector": vector,
				"payload": map[string]interface{}{
					"memory_id":  memoryID,
					"session_id": sessionID,
					"created_at": createdAt,
				},
			},
		},
	}

	resp, err := v.doRequest(ctx, "PUT", fmt.Sprintf("/collections/%s/points", VisualCollection), body)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("upsert visual vector failed: %s", string(data))
	}
	return pointID, nil
}

// VisualSearchResult holds a result from visual vector search.
type VisualSearchResult struct {
	MemoryID string  `json:"memory_id"`
	Score    float64 `json:"score"`
}

// SearchVisual searches for nearest visual embeddings.
func (v *VectorStore) SearchVisual(ctx context.Context, queryVector []float64, limit int) ([]VisualSearchResult, error) {
	body := map[string]interface{}{
		"vector":       queryVector,
		"limit":        limit,
		"with_payload": true,
	}

	resp, err := v.doRequest(ctx, "POST", fmt.Sprintf("/collections/%s/points/search", VisualCollection), body)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		Result []struct {
			ID      interface{}            `json:"id"`
			Score   float64                `json:"score"`
			Payload map[string]interface{} `json:"payload"`
		} `json:"result"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decoding visual search response: %w", err)
	}

	var results []VisualSearchResult
	for _, r := range result.Result {
		vs := VisualSearchResult{Score: r.Score}
		if p := r.Payload; p != nil {
			if mid, ok := p["memory_id"].(string); ok {
				vs.MemoryID = mid
			}
		}
		results = append(results, vs)
	}
	return results, nil
}

// Similarity computes cosine similarity between two points in the text collection.
// Returns nil if either point is not found.
func (v *VectorStore) Similarity(ctx context.Context, pointIDA, pointIDB string) (*float64, error) {
	body := map[string]interface{}{
		"ids":          []string{pointIDA, pointIDB},
		"with_vector":  true,
		"with_payload": false,
	}

	resp, err := v.doRequest(ctx, "POST", fmt.Sprintf("/collections/%s/points", TextCollection), body)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		Result []struct {
			ID     interface{} `json:"id"`
			Vector []float64   `json:"vector"`
		} `json:"result"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	if len(result.Result) < 2 {
		return nil, nil
	}

	sim := cosineSimilarity(result.Result[0].Vector, result.Result[1].Vector)
	return &sim, nil
}

// DeletePoint deletes all points for a given memory_id from a collection.
func (v *VectorStore) DeletePoint(ctx context.Context, collection, memoryID string) error {
	body := map[string]interface{}{
		"filter": map[string]interface{}{
			"must": []map[string]interface{}{
				{
					"key": "memory_id",
					"match": map[string]interface{}{
						"value": memoryID,
					},
				},
			},
		},
	}

	resp, err := v.doRequest(ctx, "POST", fmt.Sprintf("/collections/%s/points/delete", collection), body)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("delete point failed: %s", string(data))
	}
	return nil
}

// ---------- Helpers ----------

func (v *VectorStore) doRequest(ctx context.Context, method, path string, body interface{}) (*http.Response, error) {
	var bodyReader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("marshalling request body: %w", err)
		}
		bodyReader = bytes.NewReader(data)
	}

	req, err := http.NewRequestWithContext(ctx, method, v.baseURL+path, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	return v.httpClient.Do(req)
}

func cosineSimilarity(a, b []float64) float64 {
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
