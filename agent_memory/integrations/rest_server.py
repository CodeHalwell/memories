"""REST/HTTP service for the Agent Memory System.

A thin FastAPI app over :class:`agent_memory.service.MemoryService`, exposing the
same memory verbs as the MCP server but over HTTP — the surface a hosted,
multi-tenant deployment and chatbot connectors (Track C of the integration
roadmap) build on. All memory logic and serialization live in the shared
service facade; this module only maps HTTP requests onto it.

Run it (after ``pip install agent-memory[server]``)::

    uvicorn agent_memory.integrations.rest_server:app
    # or: python -m agent_memory.integrations.rest_server

Endpoints:
    GET  /health                      — liveness probe
    POST /memories                    — save a turn (body: content, session_id, ...)
    POST /retrieve                    — retrieve by query (body: query, namespace, ...)
    GET  /memories/{id}?namespace=... — fetch one memory
    POST /compact                     — run a maintenance cycle

Configuration mirrors the MCP server: ``AGENT_MEMORY_DATA_DIR`` and
``AGENT_MEMORY_PROFILE`` (``full``/``lite``).
"""

import logging
from typing import TYPE_CHECKING, Any

# NOTE: this module deliberately does NOT use ``from __future__ import
# annotations``. FastAPI resolves route handler type hints at runtime to tell
# request bodies from query params; stringized annotations of the locally
# defined Pydantic models would be misread as query parameters.

from agent_memory.integrations.mcp_server import build_service_from_env
from agent_memory.service import MemoryService

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)


def create_app(service: "MemoryService | None" = None) -> "FastAPI":
    """Build the FastAPI app exposing the memory verbs.

    Args:
        service: An existing :class:`MemoryService`. If omitted, one is built
            from environment configuration.

    ``fastapi`` is imported here (lazily) so this module is importable on the
    core profile without the ``server`` extra installed.
    """
    try:
        from contextlib import asynccontextmanager

        from fastapi import FastAPI, HTTPException, Query
        from pydantic import BaseModel
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "The REST server requires the 'server' extra. "
            "Install with: pip install agent-memory[server]"
        ) from exc

    svc = service or build_service_from_env()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await svc.initialize()
        try:
            yield
        finally:
            await svc.close()

    app = FastAPI(title="Agent Memory System", lifespan=lifespan)

    class SaveRequest(BaseModel):
        content: str
        session_id: str
        turn: int = 0
        role: str = "assistant"
        namespace: str = "default"

    class RetrieveRequest(BaseModel):
        query: str
        session_id: str | None = None
        top_k: int | None = None
        namespace: str = "default"

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok"}

    @app.post("/memories")
    async def save(req: SaveRequest) -> dict[str, Any]:
        return await svc.save_turn(
            content=req.content, session_id=req.session_id, turn=req.turn,
            role=req.role, namespace=req.namespace,
        )

    @app.post("/retrieve")
    async def retrieve(req: RetrieveRequest) -> dict[str, Any]:
        return await svc.retrieve(
            query=req.query, session_id=req.session_id, top_k=req.top_k,
            namespace=req.namespace,
        )

    @app.get("/memories/{memory_id}")
    async def get_memory(
        memory_id: str, namespace: str = Query("default"),
    ) -> dict[str, Any]:
        mem = await svc.get_memory(memory_id, namespace=namespace)
        if mem is None:
            raise HTTPException(status_code=404, detail="memory not found")
        return mem

    @app.post("/compact")
    async def compact() -> dict[str, Any]:
        return await svc.compact(trigger="manual")

    return app


# Module-level app for ``uvicorn agent_memory.integrations.rest_server:app``.
# Built lazily so importing the module doesn't require the 'server' extra.
def __getattr__(name: str) -> Any:  # pragma: no cover - thin lazy hook
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main() -> None:
    """Entry point: serve the app with uvicorn."""
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
