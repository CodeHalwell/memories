"""Transport-agnostic service facade over :class:`MemoryManager`.

This is the serialization boundary between the in-process memory library and any
out-of-process transport (MCP server, REST/WebSocket service, chatbot
connector). It exposes the four core verbs — save, retrieve, get, compact — as
``async`` methods that take and return plain JSON-serializable ``dict``s, so a
transport layer only has to map its own request/response shapes onto these.

Keeping this layer free of any transport dependency means the MCP server
(Track A of the integration roadmap) and a future REST service (Track C) share
exactly one implementation of the memory verbs and their serialization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent_memory.core.memory_manager import MemoryManager
from agent_memory.models import Memory

logger = logging.getLogger(__name__)


def memory_to_dict(memory: Memory) -> dict[str, Any]:
    """Serialize a :class:`Memory` to a compact, JSON-safe dict.

    Binary/internal fields (embeddings, cross-store reference ids) are omitted;
    this is the shape returned to external callers.
    """
    return {
        "id": memory.id,
        "content": memory.content,
        "summary": memory.summary,
        "namespace": memory.namespace,
        "session_id": memory.session_id,
        "turn": memory.turn,
        "created_at": memory.created_at,
        "tier": memory.tier,
        "salience": memory.salience,
        "decay_score": memory.decay_score,
        "valence": memory.valence,
        "arousal": memory.arousal,
        "surprise": memory.surprise,
        "access_count": memory.access_count,
        "keywords": [kw for kw, _weight in memory.keywords],
    }


class MemoryService:
    """Async facade exposing the memory verbs as dict-in / dict-out methods.

    Parameters mirror :class:`MemoryManager`. Set ``load_embeddings=False`` to
    run on the lite profile (no ``text``/``visual`` extras): retrieval falls
    back to the grep + keyword + graph layers.
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        load_embeddings: bool = True,
    ) -> None:
        self._manager = MemoryManager(data_dir=data_dir)
        self._load_embeddings = load_embeddings
        self._initialized = False

    @property
    def manager(self) -> MemoryManager:
        return self._manager

    async def initialize(self) -> None:
        if self._initialized:
            return
        await self._manager.initialize(load_embeddings=self._load_embeddings)
        self._initialized = True

    async def close(self) -> None:
        if self._initialized:
            await self._manager.close()
            self._initialized = False

    # ── Verbs ──

    async def save_turn(
        self,
        content: str,
        session_id: str,
        turn: int = 0,
        role: str = "assistant",
        namespace: str = "default",
    ) -> dict[str, Any]:
        """Log a conversation turn and let the system decide whether to save it.

        Returns ``{"saved": bool, "memory": {...} | None}``.
        """
        memory = await self._manager.process_turn(
            session_id=session_id, turn=turn, content=content, role=role,
            namespace=namespace,
        )
        return {
            "saved": memory is not None,
            "memory": memory_to_dict(memory) if memory else None,
        }

    async def retrieve(
        self,
        query: str,
        session_id: str | None = None,
        top_k: int | None = None,
        namespace: str = "default",
    ) -> dict[str, Any]:
        """Retrieve relevant memories for a query.

        Returns ``{"query": str, "count": int, "memories": [ {...}, ... ]}``.
        """
        memories = await self._manager.retrieve(
            query=query, session_id=session_id, top_k=top_k, namespace=namespace,
        )
        return {
            "query": query,
            "count": len(memories),
            "memories": [memory_to_dict(m) for m in memories],
        }

    async def get_memory(
        self, memory_id: str, namespace: str | None = None,
    ) -> dict[str, Any] | None:
        """Fetch a single memory by id (records an access). Returns None if absent
        or if it belongs to a different ``namespace`` (when one is given)."""
        memory = await self._manager.get_memory(memory_id, namespace=namespace)
        return memory_to_dict(memory) if memory else None

    async def compact(self, trigger: str = "manual") -> dict[str, Any]:
        """Run a compaction/maintenance cycle. Returns a summary of the result."""
        result = await self._manager.run_compaction(trigger=trigger)
        return {
            "trigger": result.trigger,
            "memories_reviewed": result.memories_reviewed,
            "memories_merged": result.memories_merged,
            "memories_pruned": result.memories_pruned,
            "keywords_updated": result.keywords_updated,
            "edges_discovered": result.edges_discovered,
        }
