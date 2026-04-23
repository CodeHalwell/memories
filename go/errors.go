// Package agentmemory provides an AI agent memory system with multi-layer
// retrieval, compaction, and policy-driven decisions.
package agentmemory

import "errors"

// Sentinel errors returned by the memory system.
var (
	ErrNotInitialized    = errors.New("agentmemory: store not initialized — call Initialize() first")
	ErrMemoryNotFound    = errors.New("agentmemory: memory not found")
	ErrRawLogNotFound    = errors.New("agentmemory: raw log entry not found")
	ErrLLMCallFailed     = errors.New("agentmemory: LLM completion call failed")
	ErrEmbeddingFailed   = errors.New("agentmemory: embedding generation failed")
	ErrVectorStoreError  = errors.New("agentmemory: vector store operation failed")
	ErrGraphStoreError   = errors.New("agentmemory: graph store operation failed")
	ErrInvalidConfig     = errors.New("agentmemory: invalid configuration")
	ErrMergeValidation   = errors.New("agentmemory: merge validation failed")
	ErrCompactionFailed  = errors.New("agentmemory: compaction cycle failed")
	ErrExportFailed      = errors.New("agentmemory: policy data export failed")
	ErrSessionNotFound   = errors.New("agentmemory: session log not found")
	ErrJSONParseFailed   = errors.New("agentmemory: failed to parse JSON response")
	ErrDreamWalkFailed   = errors.New("agentmemory: dream exploration walk failed")
)
