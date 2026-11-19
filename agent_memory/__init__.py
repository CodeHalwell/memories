"""Agent Memory System — local-first, cognitively-inspired memory for AI agents.

This module is the **stable public API**. Import from here rather than from
internal submodules (``agent_memory.core.*``, ``agent_memory.storage.*``); those
paths are implementation details and may change between minor versions.

Typical usage::

    from agent_memory import MemoryManager

    manager = MemoryManager()
    await manager.initialize()
    await manager.process_turn(session_id="s1", turn=1, content="...")
    results = await manager.retrieve("query")
    await manager.close()

The import surface is dependency-light: ``MemoryManager`` and the data models
import with only the core dependencies installed. Heavy backends (embeddings,
graph, vectors, LLM) are loaded lazily on first use, so unused capabilities
never need their extra installed. See ``pyproject.toml`` for the extras matrix.
"""

from agent_memory.config import MEMORY_CONFIG, make_config
from agent_memory.core.memory_manager import MemoryManager
from agent_memory.embeddings.base import (
    CallableTextEmbedder,
    HashingTextEmbedder,
    NullVisualEmbedder,
    TextEmbedderProtocol,
    VisualEmbedderProtocol,
)
from agent_memory.models import (
    CompactionResult,
    DiscoveredEdge,
    Memory,
    MergeValidation,
    RawLogEntry,
    SaveDecision,
)
from agent_memory.policy.controller import PolicyController, PolicyDecision
from agent_memory.policy.decision_model import LogisticDecisionModel
from agent_memory.rerank import CallableReranker, CrossEncoderReranker, RerankerProtocol

__version__ = "0.1.0"

__all__ = [
    "MemoryManager",
    "Memory",
    "RawLogEntry",
    "SaveDecision",
    "CompactionResult",
    "MergeValidation",
    "DiscoveredEdge",
    "MEMORY_CONFIG",
    "make_config",
    # Decision policy (save decisions) and reranking.
    "PolicyController",
    "PolicyDecision",
    "LogisticDecisionModel",
    "RerankerProtocol",
    "CrossEncoderReranker",
    "CallableReranker",
    # Pluggable embedder providers (Protocols + torch-free implementations).
    "TextEmbedderProtocol",
    "VisualEmbedderProtocol",
    "HashingTextEmbedder",
    "CallableTextEmbedder",
    "NullVisualEmbedder",
    "__version__",
]
