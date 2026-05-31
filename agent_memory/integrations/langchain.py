"""LangChain integration for the Agent Memory System.

Provides a LangChain ``BaseRetriever`` backed by the memory system's multi-layer
retrieval, plus helpers to persist LangChain messages as memories. This lets a
LangChain / LangGraph agent use the system as long-term, cross-session memory:
retrieve relevant context before a turn, record the turn after it.

Unlike the MCP and REST adapters, this module must import ``langchain-core`` at
load time (a retriever is defined by subclassing ``BaseRetriever``), so it is
*not* importable on the core profile and is intentionally not re-exported from
``agent_memory.integrations``. Install the extra:

    pip install agent-memory[langchain]

Example::

    from agent_memory.service import MemoryService
    from agent_memory.integrations.langchain import AgentMemoryRetriever, arecord_message

    service = MemoryService(load_embeddings=False)
    await service.initialize()
    retriever = AgentMemoryRetriever(service=service, namespace="user-42")

    docs = await retriever.ainvoke("what does the user like?")
    await arecord_message(service, ai_message, session_id="s1", turn=3, namespace="user-42")
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

try:
    from langchain_core.callbacks import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from pydantic import ConfigDict
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "The LangChain integration requires the 'langchain' extra. "
        "Install with: pip install agent-memory[langchain]"
    ) from exc

from agent_memory.service import MemoryService

# LangChain message type -> memory role.
_ROLE_BY_MESSAGE_TYPE = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
}


def _memory_dict_to_document(mem: dict[str, Any]) -> Document:
    """Map a serialized memory (from MemoryService) to a LangChain Document."""
    content = mem.get("content", "")
    metadata = {k: v for k, v in mem.items() if k != "content"}
    return Document(page_content=content, metadata=metadata)


class AgentMemoryRetriever(BaseRetriever):
    """A LangChain retriever backed by the Agent Memory System.

    Attributes:
        service: An initialized :class:`MemoryService`.
        namespace: Tenant namespace to scope retrieval to.
        session_id: Optional session id passed through to retrieval (used for
            access logging / policy signals).
        top_k: Optional override for the number of results.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    service: MemoryService
    namespace: str = "default"
    session_id: Optional[str] = None
    top_k: Optional[int] = None

    async def _aretrieve_documents(self, query: str) -> list[Document]:
        result = await self.service.retrieve(
            query=query,
            session_id=self.session_id,
            top_k=self.top_k,
            namespace=self.namespace,
        )
        return [_memory_dict_to_document(m) for m in result["memories"]]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return await self._aretrieve_documents(query)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        # The memory system is async-native. Synchronous retrieval is supported
        # only outside a running event loop; inside one, use ``ainvoke``.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._aretrieve_documents(query))
        raise RuntimeError(
            "AgentMemoryRetriever is async-native; call `await retriever.ainvoke(...)` "
            "in an async context instead of the synchronous API."
        )


async def arecord_message(
    service: MemoryService,
    message: Any,
    *,
    session_id: str,
    turn: int = 0,
    namespace: str = "default",
) -> dict[str, Any]:
    """Persist a LangChain message (or plain string) as a candidate memory.

    The memory system decides whether the content is worth saving. Returns the
    :meth:`MemoryService.save_turn` result.
    """
    content = getattr(message, "content", message)
    if not isinstance(content, str):
        content = str(content)
    msg_type = getattr(message, "type", None)
    role = _ROLE_BY_MESSAGE_TYPE.get(msg_type, "assistant")
    return await service.save_turn(
        content=content, session_id=session_id, turn=turn,
        role=role, namespace=namespace,
    )
