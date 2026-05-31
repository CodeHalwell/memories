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

from agent_memory.config import MEMORY_CONFIG
from agent_memory.core.memory_manager import MemoryManager
from agent_memory.models import (
    CompactionResult,
    DiscoveredEdge,
    Memory,
    MergeValidation,
    RawLogEntry,
    SaveDecision,
)

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
    "__version__",
]
