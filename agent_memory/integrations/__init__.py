"""Integration adapters that expose the Agent Memory System to external runtimes.

Each adapter is a thin wrapper over :class:`agent_memory.service.MemoryService`
and pulls in its own optional dependency only when used:

- ``mcp_server`` — a Model Context Protocol server (extra: ``mcp``) usable by
  Claude Desktop/Code, the Agent SDK, and any MCP-capable client.
- ``rest_server`` — a FastAPI HTTP service (extra: ``server``).
- ``langchain`` — a LangChain ``BaseRetriever`` + message helpers (extra:
  ``langchain``). Not re-exported here because importing it requires
  ``langchain-core`` (a retriever is defined by subclassing); import it directly
  from ``agent_memory.integrations.langchain``.

The ``mcp_server`` and ``rest_server`` modules import their transport deps
lazily, so they remain importable on the core profile.
"""
