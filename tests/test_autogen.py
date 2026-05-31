"""Tests for the AutoGen Memory adapter (roadmap §3.2).

Verifies the adapter implements AutoGen's Memory interface over MemoryService:
add/query roundtrip, namespace isolation, and update_context injecting a
SystemMessage. Skipped unless the 'autogen' extra is installed.
"""

import importlib.util

import pytest
import pytest_asyncio

AUTOGEN_INSTALLED = importlib.util.find_spec("autogen_core") is not None

pytestmark = pytest.mark.skipif(
    not AUTOGEN_INSTALLED, reason="requires the 'autogen' extra"
)


@pytest_asyncio.fixture
async def service(tmp_path):
    from agent_memory.service import MemoryService

    svc = MemoryService(data_dir=tmp_path, load_embeddings=False)
    await svc.initialize()
    yield svc
    await svc.close()


async def test_add_and_query_roundtrip(service):
    from autogen_core.memory import MemoryContent, MemoryMimeType

    from agent_memory.integrations.autogen import AgentMemory

    mem = AgentMemory(service, namespace="u1")
    await mem.add(MemoryContent(
        content="The user manages a sourdough bakery in Leeds",
        mime_type=MemoryMimeType.TEXT,
        metadata={"role": "user"},
    ))

    result = await mem.query("sourdough bakery")
    assert any("sourdough" in c.content for c in result.results)
    # Results are TEXT MemoryContent carrying memory metadata (incl. namespace).
    assert all(c.mime_type == MemoryMimeType.TEXT for c in result.results)
    assert result.results[0].metadata.get("namespace") == "u1"


async def test_query_is_namespace_isolated(service):
    from autogen_core.memory import MemoryContent, MemoryMimeType

    from agent_memory.integrations.autogen import AgentMemory

    alice = AgentMemory(service, namespace="alice")
    bob = AgentMemory(service, namespace="bob")
    await alice.add(MemoryContent(content="alice secret token alpha", mime_type=MemoryMimeType.TEXT))

    bob_results = await bob.query("secret token alpha")
    assert bob_results.results == []


async def test_update_context_injects_system_message(service):
    from autogen_core.memory import MemoryContent, MemoryMimeType
    from autogen_core.model_context import UnboundedChatCompletionContext
    from autogen_core.models import SystemMessage, UserMessage

    from agent_memory.integrations.autogen import AgentMemory

    mem = AgentMemory(service, namespace="u1")
    await mem.add(MemoryContent(content="the user speaks fluent Welsh", mime_type=MemoryMimeType.TEXT))

    ctx = UnboundedChatCompletionContext()
    await ctx.add_message(UserMessage(content="what languages does the user speak Welsh", source="user"))

    result = await mem.update_context(ctx)

    # A SystemMessage with the recalled memory was appended to the context.
    messages = await ctx.get_messages()
    sys_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    assert sys_msgs
    assert any("Welsh" in m.content for m in sys_msgs)
    # And the result reports the injected memories.
    assert any("Welsh" in c.content for c in result.memories.results)


async def test_update_context_empty_when_no_messages(service):
    from autogen_core.model_context import UnboundedChatCompletionContext

    from agent_memory.integrations.autogen import AgentMemory

    mem = AgentMemory(service, namespace="u1")
    result = await mem.update_context(UnboundedChatCompletionContext())
    assert result.memories.results == []


async def test_clear_and_close_are_noops(service):
    from agent_memory.integrations.autogen import AgentMemory

    mem = AgentMemory(service, namespace="u1")
    # Should not raise; durable store does not support bulk delete.
    await mem.clear()
    await mem.close()
