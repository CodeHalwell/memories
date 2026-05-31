"""Tests for the LangChain adapter (roadmap §3.2).

Memory logic is covered by test_service.py; here we verify the adapter wiring:
retriever returns LangChain Documents, namespace isolation holds through the
adapter, and message recording maps roles. Skipped unless 'langchain' extra is
installed.
"""

import importlib.util

import pytest
import pytest_asyncio

LANGCHAIN_INSTALLED = importlib.util.find_spec("langchain_core") is not None

pytestmark = pytest.mark.skipif(
    not LANGCHAIN_INSTALLED, reason="requires the 'langchain' extra"
)


@pytest_asyncio.fixture
async def service(tmp_path):
    from agent_memory.service import MemoryService

    svc = MemoryService(data_dir=tmp_path, load_embeddings=False)
    await svc.initialize()
    yield svc
    await svc.close()


async def test_retriever_returns_documents(service):
    from langchain_core.documents import Document

    from agent_memory.integrations.langchain import AgentMemoryRetriever

    await service.save_turn(content="The user enjoys playing chess on weekends", session_id="s1", turn=1, namespace="u1")
    retriever = AgentMemoryRetriever(service=service, namespace="u1")

    docs = await retriever.ainvoke("chess hobby")
    assert docs
    assert all(isinstance(d, Document) for d in docs)
    assert any("chess" in d.page_content for d in docs)
    # Metadata carries memory fields (namespace, id, etc.) but not content.
    assert docs[0].metadata.get("namespace") == "u1"
    assert "content" not in docs[0].metadata


async def test_retriever_is_namespace_isolated(service):
    from agent_memory.integrations.langchain import AgentMemoryRetriever

    await service.save_turn(content="alice private memo about project zeta", session_id="s", turn=1, namespace="alice")

    bob_retriever = AgentMemoryRetriever(service=service, namespace="bob")
    docs = await bob_retriever.ainvoke("project zeta")
    assert docs == []


async def test_arecord_message_maps_role(service):
    from langchain_core.messages import HumanMessage

    from agent_memory.integrations.langchain import arecord_message

    result = await arecord_message(
        service, HumanMessage(content="Remember my favourite colour is teal"),
        session_id="s1", turn=1, namespace="u1",
    )
    assert result["saved"] is True
    # Round-trips through retrieval within the same namespace.
    from agent_memory.integrations.langchain import AgentMemoryRetriever

    docs = await AgentMemoryRetriever(service=service, namespace="u1").ainvoke("favourite colour")
    assert any("teal" in d.page_content for d in docs)


async def test_arecord_plain_string(service):
    from agent_memory.integrations.langchain import arecord_message

    result = await arecord_message(service, "a plain string note", session_id="s1", turn=1, namespace="u1")
    assert result["saved"] is True


async def test_sync_retrieval_raises(service):
    from agent_memory.integrations.langchain import AgentMemoryRetriever

    retriever = AgentMemoryRetriever(service=service, namespace="u1")
    with pytest.raises(RuntimeError, match="async-native"):
        retriever._get_relevant_documents("q", run_manager=None)


def test_generic_chat_message_role_preserved():
    from langchain_core.messages import ChatMessage

    from agent_memory.integrations.langchain import _message_role

    # Generic ChatMessage has type "chat"; its explicit role must be preserved.
    assert _message_role(ChatMessage(role="user", content="hi")) == "user"
    assert _message_role(ChatMessage(role="system", content="hi")) == "system"


def test_multimodal_content_extracts_text():
    from langchain_core.messages import HumanMessage

    from agent_memory.integrations.langchain import _message_role, _message_text

    msg = HumanMessage(content=[
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
    ])
    assert _message_text(msg) == "describe this"
    assert _message_role(msg) == "user"
