package policy

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"

	agentmemory "github.com/CodeHalwell/Memories/go"
	"github.com/CodeHalwell/Memories/go/storage"
)

// ExportResult holds metadata about a policy training data export.
type ExportResult struct {
	SaveExamples      int    `json:"save_examples"`
	RetrievalExamples int    `json:"retrieval_examples"`
	SavePath          string `json:"save_path"`
	RetrievalPath     string `json:"retrieval_path"`
	ReadyForTraining  bool   `json:"ready_for_training"`
}

// ExportPolicyTrainingData exports decision-outcome pairs for offline policy model training (A4.4).
func ExportPolicyTrainingData(ctx context.Context, sqlite *storage.SQLiteStore, cfg agentmemory.MemoryConfig, outputDir *string) (ExportResult, error) {
	dir := cfg.PolicyDataDir
	if outputDir != nil {
		dir = *outputDir
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return ExportResult{}, fmt.Errorf("creating output dir: %w", err)
	}

	saveData, err := sqlite.ExportSavePolicyData(ctx)
	if err != nil {
		return ExportResult{}, fmt.Errorf("exporting save data: %w", err)
	}

	retrievalData, err := sqlite.ExportRetrievalPolicyData(ctx)
	if err != nil {
		return ExportResult{}, fmt.Errorf("exporting retrieval data: %w", err)
	}

	savePath := filepath.Join(dir, "save_policy_data.jsonl")
	retrievalPath := filepath.Join(dir, "retrieval_policy_data.jsonl")

	if err := writeJSONL(savePath, saveData); err != nil {
		return ExportResult{}, err
	}
	if err := writeJSONL(retrievalPath, retrievalData); err != nil {
		return ExportResult{}, err
	}

	result := ExportResult{
		SaveExamples:      len(saveData),
		RetrievalExamples: len(retrievalData),
		SavePath:          savePath,
		RetrievalPath:     retrievalPath,
		ReadyForTraining: len(saveData) >= cfg.PolicyMinSaveExamples &&
			len(retrievalData) >= cfg.PolicyMinRetrievalExamples,
	}

	log.Printf("Policy data export: %d save examples, %d retrieval examples (ready=%v)",
		result.SaveExamples, result.RetrievalExamples, result.ReadyForTraining)

	return result, nil
}

func writeJSONL(path string, rows []map[string]interface{}) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("creating %s: %w", path, err)
	}
	defer f.Close()

	encoder := json.NewEncoder(f)
	encoder.SetEscapeHTML(false)
	for _, row := range rows {
		if err := encoder.Encode(row); err != nil {
			return fmt.Errorf("writing to %s: %w", path, err)
		}
	}
	return nil
}
