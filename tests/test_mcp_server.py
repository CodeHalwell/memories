"""Tests for the MCP server adapter.

The memory logic itself is covered by test_service.py; here we verify the
adapter wiring: env-based construction, the dependency guard, and that the
expected tools are registered (when the 'mcp' extra is installed).
"""

import importlib.util

import pytest

from agent_memory.integrations import mcp_server
from agent_memory.service import MemoryService

MCP_INSTALLED = importlib.util.find_spec("mcp") is not None

EXPECTED_TOOLS = {"memory_save", "memory_retrieve", "memory_get", "memory_compact"}


def test_module_imports_without_mcp_extra():
    # The adapter module must import on the core profile (mcp imported lazily).
    assert hasattr(mcp_server, "create_server")
    assert hasattr(mcp_server, "build_service_from_env")


def test_build_service_from_env_respects_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_MEMORY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_MEMORY_PROFILE", "lite")
    svc = mcp_server.build_service_from_env()
    assert isinstance(svc, MemoryService)
    assert svc._load_embeddings is False

    monkeypatch.setenv("AGENT_MEMORY_PROFILE", "full")
    svc_full = mcp_server.build_service_from_env()
    assert svc_full._load_embeddings is True


@pytest.mark.skipif(MCP_INSTALLED, reason="mcp extra is installed")
def test_create_server_without_mcp_raises_actionable_error():
    with pytest.raises(ImportError, match=r"agent-memory\[mcp\]"):
        mcp_server.create_server(MemoryService(load_embeddings=False))


@pytest.mark.skipif(not MCP_INSTALLED, reason="requires the 'mcp' extra")
async def test_create_server_registers_expected_tools(tmp_path):
    service = MemoryService(data_dir=tmp_path, load_embeddings=False)
    server = mcp_server.create_server(service)
    tools = await server.list_tools()
    names = {t.name for t in tools}
    assert EXPECTED_TOOLS.issubset(names)
