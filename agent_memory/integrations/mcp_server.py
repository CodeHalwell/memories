"""Model Context Protocol (MCP) server for the Agent Memory System.

Exposes the memory verbs as MCP tools so any MCP-capable client — Claude
Desktop/Code, the Claude Agent SDK, or another agent framework — can give its
agent persistent, searchable long-term memory without bespoke per-framework
integration code.

Tools:
    - ``memory_save``     — log a turn; the system decides whether to persist it
    - ``memory_retrieve`` — semantic/keyword/graph retrieval for a query
    - ``memory_get``      — fetch a single memory by id
    - ``memory_compact``  — run a maintenance/compaction cycle

The server is a thin wrapper over :class:`agent_memory.service.MemoryService`;
all memory logic and serialization live there.

Run it (after ``pip install agent-memory[mcp]``)::

    python -m agent_memory.integrations.mcp_server

Configuration via environment variables:
    - ``AGENT_MEMORY_DATA_DIR`` — data directory (default: ``data``)
    - ``AGENT_MEMORY_PROFILE``  — ``full`` (default) loads embeddings; ``lite``
      runs without the text/visual extras (grep + keyword + graph retrieval).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_memory.service import MemoryService

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

SERVER_NAME = "agent-memory"


def build_service_from_env() -> MemoryService:
    """Construct a :class:`MemoryService` from environment configuration."""
    data_dir = os.environ.get("AGENT_MEMORY_DATA_DIR")
    profile = os.environ.get("AGENT_MEMORY_PROFILE", "full").strip().lower()
    return MemoryService(
        data_dir=Path(data_dir) if data_dir else None,
        load_embeddings=profile != "lite",
    )


def create_server(service: MemoryService | None = None) -> FastMCP:
    """Build a FastMCP server exposing the memory tools.

    Args:
        service: An existing :class:`MemoryService`. If omitted, one is built
            from environment configuration.

    The ``mcp`` package is imported here (lazily) so this module is importable
    on the core profile without the ``mcp`` extra installed.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "The MCP server requires the 'mcp' extra. "
            "Install with: pip install agent-memory[mcp]"
        ) from exc

    svc = service or build_service_from_env()
    server = FastMCP(SERVER_NAME)

    async def _ensure_ready() -> None:
        # Idempotent: MemoryService.initialize() is a no-op after first call.
        await svc.initialize()

    @server.tool()
    async def memory_save(
        content: str,
        session_id: str,
        turn: int = 0,
        role: str = "assistant",
        namespace: str = "default",
    ) -> dict[str, Any]:
        """Record a conversation turn as a candidate memory.

        The system decides whether the turn is worth persisting (salience,
        emotion, novelty). Returns whether a memory was saved and its details.
        ``namespace`` isolates data per tenant (e.g. a user or agent id).
        """
        await _ensure_ready()
        return await svc.save_turn(
            content=content, session_id=session_id, turn=turn, role=role,
            namespace=namespace,
        )

    @server.tool()
    async def memory_retrieve(
        query: str,
        session_id: str | None = None,
        top_k: int | None = None,
        namespace: str = "default",
    ) -> dict[str, Any]:
        """Retrieve memories relevant to a query, ranked by relevance and decay.

        Results are isolated to ``namespace``.
        """
        await _ensure_ready()
        return await svc.retrieve(
            query=query, session_id=session_id, top_k=top_k, namespace=namespace,
        )

    @server.tool()
    async def memory_get(
        memory_id: str, namespace: str = "default",
    ) -> dict[str, Any] | None:
        """Fetch a single memory by its id within ``namespace``. Null if absent."""
        await _ensure_ready()
        return await svc.get_memory(memory_id, namespace=namespace)

    @server.tool()
    async def memory_compact() -> dict[str, Any]:
        """Run a compaction/maintenance cycle (merge, reweight, prune)."""
        await _ensure_ready()
        return await svc.compact(trigger="manual")

    return server


def main() -> None:
    """Entry point: build the server and serve over stdio."""
    logging.basicConfig(level=logging.INFO)
    server = create_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
