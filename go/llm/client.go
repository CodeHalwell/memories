// Package llm provides an OpenAI-compatible HTTP client with retry logic.
package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"time"
)

func normalizeBaseURL(baseURL string) string {
	return strings.TrimSuffix(strings.TrimRight(baseURL, "/"), "/v1")
}

// Client calls an OpenAI-compatible chat completion API.
type Client struct {
	BaseURL     string
	APIKey      string
	Model       string
	Temperature float64
	MaxRetries  int
	httpClient  *http.Client
}

// NewClient creates an LLM Client.
func NewClient(baseURL, apiKey, model string, temperature float64) *Client {
	return &Client{
		BaseURL:     normalizeBaseURL(baseURL),
		APIKey:      apiKey,
		Model:       model,
		Temperature: temperature,
		MaxRetries:  3,
		httpClient:  &http.Client{Timeout: 120 * time.Second},
	}
}

// chatMessage is an OpenAI chat message.
type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Temperature float64       `json:"temperature"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

// Complete sends a completion request and returns the text response with retries.
func (c *Client) Complete(ctx context.Context, prompt string, system *string, model *string, temperature *float64) (string, error) {
	m := c.Model
	if model != nil {
		m = *model
	}
	t := c.Temperature
	if temperature != nil {
		t = *temperature
	}

	var messages []chatMessage
	if system != nil {
		messages = append(messages, chatMessage{Role: "system", Content: *system})
	}
	messages = append(messages, chatMessage{Role: "user", Content: prompt})

	reqBody := chatRequest{Model: m, Messages: messages, Temperature: t}

	var lastErr error
	for attempt := 0; attempt < c.MaxRetries; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(math.Pow(2, float64(attempt))) * time.Second
			select {
			case <-ctx.Done():
				return "", ctx.Err()
			case <-time.After(backoff):
			}
		}

		data, err := json.Marshal(reqBody)
		if err != nil {
			return "", fmt.Errorf("marshalling request: %w", err)
		}

		req, err := http.NewRequestWithContext(ctx, "POST", c.BaseURL+"/v1/chat/completions", bytes.NewReader(data))
		if err != nil {
			return "", fmt.Errorf("creating request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
		if c.APIKey != "" {
			req.Header.Set("Authorization", "Bearer "+c.APIKey)
		}

		resp, err := c.httpClient.Do(req)
		if err != nil {
			lastErr = err
			continue
		}

		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			lastErr = err
			continue
		}

		if resp.StatusCode != http.StatusOK {
			lastErr = fmt.Errorf("LLM API error %d: %s", resp.StatusCode, string(body))
			continue
		}

		var chatResp chatResponse
		if err := json.Unmarshal(body, &chatResp); err != nil {
			lastErr = fmt.Errorf("decoding response: %w", err)
			continue
		}

		if len(chatResp.Choices) > 0 {
			return chatResp.Choices[0].Message.Content, nil
		}
		lastErr = fmt.Errorf("no choices in LLM response")
	}

	return "", fmt.Errorf("LLM call failed after %d retries: %w", c.MaxRetries, lastErr)
}

// CompleteJSON sends a completion request and parses the response as JSON.
func (c *Client) CompleteJSON(ctx context.Context, prompt string, system *string, model *string, temperature *float64) (map[string]interface{}, error) {
	text, err := c.Complete(ctx, prompt, system, model, temperature)
	if err != nil {
		return nil, err
	}

	cleaned := strings.TrimSpace(text)
	if strings.HasPrefix(cleaned, "```") {
		lines := strings.Split(cleaned, "\n")
		var filtered []string
		for i, line := range lines {
			if i == 0 && strings.HasPrefix(strings.TrimSpace(line), "```") {
				continue
			}
			if strings.HasPrefix(strings.TrimSpace(line), "```") {
				continue
			}
			filtered = append(filtered, line)
		}
		cleaned = strings.Join(filtered, "\n")
	}

	var result map[string]interface{}
	if err := json.Unmarshal([]byte(cleaned), &result); err != nil {
		// Try parsing as array
		var arr []interface{}
		if arrErr := json.Unmarshal([]byte(cleaned), &arr); arrErr == nil {
			return map[string]interface{}{"items": arr}, nil
		}
		return nil, fmt.Errorf("parsing JSON response: %w", err)
	}
	return result, nil
}

// CompleteJSONArray sends a completion request and parses the response as a JSON array.
func (c *Client) CompleteJSONArray(ctx context.Context, prompt string, system *string) ([]interface{}, error) {
	text, err := c.Complete(ctx, prompt, system, nil, nil)
	if err != nil {
		return nil, err
	}

	cleaned := strings.TrimSpace(text)
	if strings.HasPrefix(cleaned, "```") {
		lines := strings.Split(cleaned, "\n")
		var filtered []string
		for i, line := range lines {
			if i == 0 && strings.HasPrefix(strings.TrimSpace(line), "```") {
				continue
			}
			if strings.HasPrefix(strings.TrimSpace(line), "```") {
				continue
			}
			filtered = append(filtered, line)
		}
		cleaned = strings.Join(filtered, "\n")
	}

	var result []interface{}
	if err := json.Unmarshal([]byte(cleaned), &result); err != nil {
		return nil, fmt.Errorf("parsing JSON array response: %w", err)
	}
	return result, nil
}
