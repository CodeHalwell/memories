"""Tests for the LlamaIndex adapter (roadmap §3.2).

Verifies the adapter wiring: retriever returns scored LlamaIndex nodes,
namespace isolation holds through the adapter, and message recording maps roles.
Skipped unless the 'llamaindex' extra is installed.
"""

import importlib.util

import pytest
import pytest_asyncio

LLAMAINDEX_INSTALLED = importlib.util.find_spec("llama_index") is not None

pytestmark = pytest.mark.skipif(
    not LLAMAINDEX_INSTALLED, reason="requires the 'llamaindex' extra"
)


@pytest_asyncio.fixture
async def service(tmp_path):
    from agent_memory.service import MemoryService

    svc = MemoryService(data_dir=tmp_path, load_embeddings=False)
    await svc.initialize()
    yield svc
    await svc.close()


async def test_retriever_returns_scored_nodes(service):
    from llama_index.core.schema import NodeWithScore

    from agent_memory.integrations.llamaindex import AgentMemoryRetriever

    await service.save_turn(content="The user is learning to play the violin", session_id="s1", turn=1, namespace="u1")
    retriever = AgentMemoryRetriever(service, namespace="u1")

    nodes = await retriever.aretrieve("violin practice")
    assert nodes
    assert all(isinstance(n, NodeWithScore) for n in nodes)
    assert any("violin" in n.node.text for n in nodes)
    # Scores are descending in (0, 1].
    scores = [n.score for n in nodes]
    assert scores == sorted(scores, reverse=True)
    assert all(0 < s <= 1 for s in scores)
    # Metadata carries memory fields but not the content itself.
    assert nodes[0].node.metadata.get("namespace") == "u1"
    assert "content" not in nodes[0].node.metadata


async def test_retriever_is_namespace_isolated(service):
    from agent_memory.integrations.llamaindex import AgentMemoryRetriever

    await service.save_turn(content="alice confidential note about merger", session_id="s", turn=1, namespace="alice")
    nodes = await AgentMemoryRetriever(service, namespace="bob").aretrieve("merger note")
    assert nodes == []


async def test_arecord_chat_message_maps_role(service):
    from llama_index.core.llms import ChatMessage, MessageRole

    from agent_memory.integrations.llamaindex import arecord_message

    result = await arecord_message(
        service,
        ChatMessage(role=MessageRole.USER, content="Remember I am allergic to peanuts"),
        session_id="s1", turn=1, namespace="u1",
    )
    assert result["saved"] is True

    from agent_memory.integrations.llamaindex import AgentMemoryRetriever

    nodes = await AgentMemoryRetriever(service, namespace="u1").aretrieve("allergic peanuts")
    assert any("peanuts" in n.node.text for n in nodes)


async def test_arecord_plain_string(service):
    from agent_memory.integrations.llamaindex import arecord_message

    result = await arecord_message(service, "a plain string fact", session_id="s1", turn=1, namespace="u1")
    assert result["saved"] is True
