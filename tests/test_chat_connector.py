"""Tests for the framework-agnostic ChatConnector (roadmap §5.2).

Runs on the lite profile (no extras). Verifies per-user namespace isolation,
recall/record helpers, the prompt context block, and the one-call handle_turn.
"""

import pytest
import pytest_asyncio

from agent_memory.integrations.chat import ChatConnector
from agent_memory.service import MemoryService


@pytest_asyncio.fixture
async def connector(tmp_path):
    svc = MemoryService(data_dir=tmp_path, load_embeddings=False)
    await svc.initialize()
    yield ChatConnector(svc, namespace_prefix="chat")
    await svc.close()


def test_namespace_mapping():
    svc = MemoryService(load_embeddings=False)
    c = ChatConnector(svc, namespace_prefix="discord")
    assert c.namespace_for("U123") == "discord:U123"


async def test_record_and_recall_roundtrip(connector):
    await connector.record_user_message("alice", "I play the trumpet in a jazz band")
    memories = await connector.recall("alice", "jazz trumpet")
    assert any("trumpet" in m["content"] for m in memories)
    # Recall is namespace-scoped.
    assert all(m["namespace"] == "chat:alice" for m in memories)


async def test_users_are_isolated(connector):
    await connector.record_user_message("alice", "my favourite language is Python")
    await connector.record_user_message("bob", "my favourite language is Rust")

    alice = await connector.recall("alice", "favourite language")
    bob = await connector.recall("bob", "favourite language")

    assert any("Python" in m["content"] for m in alice)
    assert not any("Rust" in m["content"] for m in alice)
    assert any("Rust" in m["content"] for m in bob)
    assert not any("Python" in m["content"] for m in bob)


async def test_context_block_formats_and_is_empty_when_nothing(connector):
    # Nothing recorded yet -> empty string (safe to prepend unconditionally).
    assert await connector.context_block("alice", "anything") == ""

    await connector.record_user_message("alice", "I am a vegetarian")
    # Lite profile is lexical, so the query overlaps the stored content.
    block = await connector.context_block("alice", "vegetarian meals", header="Known:")
    assert block.startswith("Known:")
    assert "- I am a vegetarian" in block


async def test_handle_turn_records_new_user(connector):
    # First turn for a fresh user is always saved (fast path, no LLM needed).
    result = await connector.handle_turn("dave", "I drive a red car", turn=0)
    assert result["namespace"] == "chat:dave"
    assert result["session_id"] == "chat:dave"
    assert result["context"] == []  # nothing prior to recall
    assert result["recorded"]["saved"] is True


async def test_handle_turn_recalls_prior_context_before_recording(connector):
    # Seed a prior memory (first turn -> saved).
    await connector.record_user_message("erin", "I live in Manchester", turn=1)

    result = await connector.handle_turn(
        "erin", "what's the weather like here?", turn=2, recall_query="Manchester",
    )
    assert result["namespace"] == "chat:erin"
    # Context reflects prior turns only (the seeded memory).
    assert any("Manchester" in m["content"] for m in result["context"])
    assert "saved" in result["recorded"]


async def test_record_assistant_message(connector):
    res = await connector.record_assistant_message("alice", "The capital of France is Paris", turn=1)
    assert res["saved"] is True
    assert res["memory"]["namespace"] == "chat:alice"
