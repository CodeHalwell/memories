"""Integration adapters that expose the Agent Memory System to external runtimes.

Each adapter is a thin wrapper over :class:`agent_memory.service.MemoryService`
and pulls in its own optional dependency only when used:

- ``mcp_server`` — a Model Context Protocol server (extra: ``mcp``) usable by
  Claude Desktop/Code, the Agent SDK, and any MCP-capable client.

Future adapters (REST service, framework bindings) live here too.
"""
