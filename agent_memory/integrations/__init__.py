"""Integration adapters that expose the Agent Memory System to external runtimes.

Each adapter is a thin wrapper over :class:`agent_memory.service.MemoryService`
and pulls in its own optional dependency only when used:

- ``mcp_server`` — a Model Context Protocol server (extra: ``mcp``) usable by
  Claude Desktop/Code, the Agent SDK, and any MCP-capable client.
- ``rest_server`` — a FastAPI HTTP service (extra: ``server``).
- ``chat`` — a framework-agnostic :class:`ChatConnector` for wiring chatbots
  (Discord/Slack/Telegram/web) to the memory system. No extra required.
- ``langchain`` — a LangChain ``BaseRetriever`` + message helpers (extra:
  ``langchain``).
- ``llamaindex`` — a LlamaIndex ``BaseRetriever`` + message helpers (extra:
  ``llamaindex``).
- ``autogen`` — an AutoGen ``Memory`` implementation (extra: ``autogen``).

The framework adapters are not re-exported here because importing them requires
their framework package (they subclass a framework base class); import them
directly from ``agent_memory.integrations.langchain`` /
``agent_memory.integrations.llamaindex`` / ``agent_memory.integrations.autogen``.

The ``mcp_server`` and ``rest_server`` modules import their transport deps
lazily, so they remain importable on the core profile.
"""

from agent_memory.integrations.chat import ChatConnector

__all__ = ["ChatConnector"]
