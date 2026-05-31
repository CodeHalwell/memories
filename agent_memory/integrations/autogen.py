"""AutoGen integration for the Agent Memory System.

Implements AutoGen's ``Memory`` interface (``autogen_core.memory.Memory``)
backed by the memory system, so an AutoGen agent gains persistent, namespace-
isolated long-term memory: ``update_context`` injects relevant memories into the
model context before the model is called, ``add`` records new memories, and
``query`` exposes retrieval directly.

AutoGen's ``Memory`` methods are all ``async``, so this adapter maps 1:1 onto the
async :class:`MemoryService` with no event-loop gymnastics.

Like the other framework adapters, this module imports ``autogen-core`` at load
time (it subclasses ``Memory``), so it is not importable on the core profile and
is not re-exported from ``agent_memory.integrations``. Install the extra:

    pip install agent-memory[autogen]

Example::

    from agent_memory.service import MemoryService
    from agent_memory.integrations.autogen import AgentMemory

    service = MemoryService(load_embeddings=False)
    await service.initialize()
    memory = AgentMemory(service, namespace="user-42")
    # pass `memory` to an AutoGen AssistantAgent(memory=[memory])
"""

from __future__ import annotations

from typing import Any

try:
    from autogen_core.memory import (
        Memory,
        MemoryContent,
        MemoryMimeType,
        MemoryQueryResult,
        UpdateContextResult,
    )
    from autogen_core.model_context import ChatCompletionContext
    from autogen_core.models import SystemMessage
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "The AutoGen integration requires the 'autogen' extra. "
        "Install with: pip install agent-memory[autogen]"
    ) from exc

from agent_memory.service import MemoryService


def _content_text(content: Any) -> str:
    """Extract plain text from AutoGen message/memory content.

    Content may be a string or a list of parts (multi-modal). Concatenate the
    string parts and the text of any ``{"type": "text", "text": ...}`` blocks.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_val = block.get("text")
                if isinstance(text_val, str):
                    parts.append(text_val)
        return " ".join(p for p in parts if p)
    return str(content)


class AgentMemory(Memory):
    """AutoGen ``Memory`` backed by the Agent Memory System.

    Args:
        service: An initialized :class:`MemoryService`.
        namespace: Tenant namespace to scope all memory to.
        session_id: Session id passed through to save/retrieve (defaults to the
            namespace).
        top_k: Default number of memories to retrieve.
        inject_header: Header line prepended to the memories injected into the
            model context by :meth:`update_context`.
    """

    def __init__(
        self,
        service: MemoryService,
        namespace: str = "default",
        session_id: str | None = None,
        top_k: int | None = None,
        inject_header: str = "Relevant memory:",
    ) -> None:
        self._service = service
        self._namespace = namespace
        self._session_id = session_id or namespace
        self._top_k = top_k
        self._inject_header = inject_header

    async def add(
        self, content: MemoryContent, cancellation_token: Any = None,
    ) -> None:
        """Record a memory. Role/turn may be supplied via ``content.metadata``.

        Metadata values are extracted defensively so an explicit ``None`` (or a
        non-coercible ``turn``) falls back to the default rather than raising.
        """
        metadata = content.metadata or {}

        session_val = metadata.get("session_id")
        session_id = str(session_val) if session_val is not None else self._session_id

        turn_val = metadata.get("turn")
        try:
            turn = int(turn_val) if turn_val is not None else 0
        except (TypeError, ValueError):
            turn = 0

        role_val = metadata.get("role")
        role = str(role_val) if role_val is not None else "assistant"

        await self._service.save_turn(
            content=_content_text(content.content),
            session_id=session_id,
            turn=turn,
            role=role,
            namespace=self._namespace,
        )

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: Any = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Retrieve relevant memories as a ``MemoryQueryResult``."""
        query_text = _content_text(query.content if isinstance(query, MemoryContent) else query)
        result = await self._service.retrieve(
            query=query_text,
            session_id=self._session_id,
            top_k=kwargs.get("top_k", self._top_k),
            namespace=self._namespace,
        )
        contents = [
            MemoryContent(
                content=m["content"],
                mime_type=MemoryMimeType.TEXT,
                metadata={k: v for k, v in m.items() if k != "content"},
            )
            for m in result["memories"]
        ]
        return MemoryQueryResult(results=contents)

    async def update_context(
        self, model_context: ChatCompletionContext,
    ) -> UpdateContextResult:
        """Inject memories relevant to the latest message into ``model_context``."""
        messages = await model_context.get_messages()
        if not messages:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        query_text = _content_text(getattr(messages[-1], "content", ""))
        if not query_text.strip():
            # Nothing to query on (e.g. an empty or purely non-text message);
            # avoid a pointless retrieval/embedding round-trip.
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))
        result = await self.query(query_text)

        if result.results:
            lines = [self._inject_header]
            lines.extend(
                f"{i}. {_content_text(c.content)}"
                for i, c in enumerate(result.results, start=1)
            )
            await model_context.add_message(SystemMessage(content="\n".join(lines)))

        return UpdateContextResult(memories=result)

    async def clear(self) -> None:
        """No-op: the memory store is durable and append-oriented.

        Bulk deletion is intentionally not supported (the system never deletes
        raw logs); use retention/compaction policies instead. Implemented as a
        no-op so AutoGen reset flows don't fail.
        """
        return None

    async def close(self) -> None:
        """No-op: the shared MemoryService lifecycle is owned by the caller."""
        return None
