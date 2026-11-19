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
        "role": memory.role,
        "event_time": memory.event_time,
        # "memory" for a curated memory, "raw" for an uncurated logged turn.
        "kind": memory.kind,
        "score": memory.score,
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
        text_embedder: Any = None,
        visual_embedder: Any = None,
    ) -> None:
        self._manager = MemoryManager(
            data_dir=data_dir,
            text_embedder=text_embedder,
            visual_embedder=visual_embedder,
        )
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
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """Log a conversation turn and let the policy decide whether to curate it.

        Returns ``{"saved": bool, "memory": {...} | None}``. Unsaved turns are
        still logged and retrievable as raw evidence.
        """
        memory = await self._manager.process_turn(
            session_id=session_id, turn=turn, content=content, role=role,
            namespace=namespace, timestamp=timestamp,
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
        turn: int | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        """Retrieve relevant memories (and raw turns) for a query.

        Returns ``{"query", "retrieval_id", "count", "memories": [...]}``. Pass
        ``retrieval_id`` to :meth:`record_response` / :meth:`mark_used`.
        """
        memories = await self._manager.retrieve(
            query=query, session_id=session_id, top_k=top_k, namespace=namespace,
            turn=turn, since=since, until=until,
        )
        return {
            "query": query,
            "retrieval_id": memories[0].retrieval_id if memories else None,
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

    async def delete_memory(
        self, memory_id: str, namespace: str | None = None, purge_raw: bool = False,
    ) -> dict[str, Any]:
        """Delete a memory (``purge_raw`` also removes its source turn from the index)."""
        deleted = await self._manager.delete_memory(
            memory_id, namespace=namespace, purge_raw=purge_raw,
        )
        return {"deleted": deleted, "memory_id": memory_id}

    async def purge_namespace(self, namespace: str) -> dict[str, Any]:
        """Delete everything stored for a tenant."""
        return await self._manager.purge_namespace(namespace)

    async def record_response(self, retrieval_id: str, response_text: str) -> dict[str, Any]:
        """Tell the system what was said using a retrieval's results (outcome signal)."""
        rows = await self._manager.record_response(retrieval_id, response_text)
        return {"retrieval_id": retrieval_id, "used": sum(r["used"] for r in rows), "assessed": len(rows)}

    async def mark_used(self, retrieval_id: str, memory_ids: list[str]) -> dict[str, Any]:
        rows = await self._manager.mark_used(retrieval_id, memory_ids)
        return {"retrieval_id": retrieval_id, "used": sum(r["used"] for r in rows)}

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
            "notes": result.notes,
        }

    async def train_policy(self, force: bool = False) -> dict[str, Any]:
        """Train the save-decision model now (promotion is still gated)."""
        return await self._manager.train_policy(force=force)
