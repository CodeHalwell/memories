"""LlamaIndex integration for the Agent Memory System.

Provides a LlamaIndex ``BaseRetriever`` backed by the memory system's multi-layer
retrieval, plus a helper to persist LlamaIndex chat messages as memories. This
lets a LlamaIndex query engine or agent use the system as long-term,
cross-session memory.

Like the LangChain adapter, this module imports ``llama-index-core`` at load
time (a retriever is defined by subclassing ``BaseRetriever``), so it is *not*
importable on the core profile and is not re-exported from
``agent_memory.integrations``. Install the extra:

    pip install agent-memory[llamaindex]

Example::

    from agent_memory.service import MemoryService
    from agent_memory.integrations.llamaindex import AgentMemoryRetriever

    service = MemoryService(load_embeddings=False)
    await service.initialize()
    retriever = AgentMemoryRetriever(service, namespace="user-42")

    nodes = await retriever.aretrieve("what does the user like?")
"""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional

try:
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "The LlamaIndex integration requires the 'llamaindex' extra. "
        "Install with: pip install agent-memory[llamaindex]"
    ) from exc

from agent_memory.service import MemoryService

# LlamaIndex chat role -> memory role (roles already align; mapped explicitly
# so unexpected/likely-enum values fall back to "assistant").
_ROLE_MAP = {
    "user": "user",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
}


def _memory_dict_to_node(mem: dict[str, Any], score: float) -> NodeWithScore:
    """Map a serialized memory (from MemoryService) to a scored LlamaIndex node."""
    content = mem.get("content", "")
    metadata = {k: v for k, v in mem.items() if k != "content"}
    node = TextNode(text=content, id_=mem.get("id", ""), metadata=metadata)
    return NodeWithScore(node=node, score=score)


def _to_nodes(memories: list[dict[str, Any]]) -> List[NodeWithScore]:
    # Results arrive already ranked; assign a descending rank score in (0, 1]
    # so downstream rerankers/score-thresholds see the retrieval order.
    n = len(memories)
    return [
        _memory_dict_to_node(mem, score=(n - i) / n)
        for i, mem in enumerate(memories)
    ]


class AgentMemoryRetriever(BaseRetriever):
    """A LlamaIndex retriever backed by the Agent Memory System.

    Args:
        service: An initialized :class:`MemoryService`.
        namespace: Tenant namespace to scope retrieval to.
        session_id: Optional session id passed through to retrieval.
        top_k: Optional override for the number of results.
    """

    def __init__(
        self,
        service: MemoryService,
        namespace: str = "default",
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self._service = service
        self._namespace = namespace
        self._session_id = session_id
        self._top_k = top_k
        super().__init__(**kwargs)

    async def _aretrieve_query(self, query: str) -> List[NodeWithScore]:
        result = await self._service.retrieve(
            query=query,
            session_id=self._session_id,
            top_k=self._top_k,
            namespace=self._namespace,
        )
        return _to_nodes(result["memories"])

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return await self._aretrieve_query(query_bundle.query_str)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # The memory system is async-native. Synchronous retrieval is supported
        # only outside a running event loop; inside one, use ``aretrieve``.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._aretrieve_query(query_bundle.query_str))
        raise RuntimeError(
            "AgentMemoryRetriever is async-native; call `await retriever.aretrieve(...)` "
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
    """Persist a LlamaIndex ``ChatMessage`` (or plain string) as a candidate memory.

    The memory system decides whether the content is worth saving. Returns the
    :meth:`MemoryService.save_turn` result.
    """
    content = getattr(message, "content", message)
    if not isinstance(content, str):
        content = str(content)
    role_attr = getattr(message, "role", None)
    # role may be a MessageRole enum; normalise via its string value.
    role_value = getattr(role_attr, "value", role_attr)
    role = _ROLE_MAP.get(role_value, "assistant")
    return await service.save_turn(
        content=content, session_id=session_id, turn=turn,
        role=role, namespace=namespace,
    )
