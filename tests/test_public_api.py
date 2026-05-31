"""Tests for the stable public API surface (agent_memory.__init__).

These lock in the contract that external code and integration adapters depend
on, and the guarantee that the public surface imports on the lightweight core
profile without the heavy embedding/graph/vector/LLM extras.
"""

import importlib

import pytest


def test_public_exports_are_importable():
    import agent_memory

    expected = {
        "MemoryManager",
        "Memory",
        "RawLogEntry",
        "SaveDecision",
        "CompactionResult",
        "MergeValidation",
        "DiscoveredEdge",
        "MEMORY_CONFIG",
        "__version__",
    }
    assert expected.issubset(set(agent_memory.__all__))
    for name in expected:
        assert hasattr(agent_memory, name), f"{name} missing from public API"


def test_version_present():
    import agent_memory

    assert isinstance(agent_memory.__version__, str)
    assert agent_memory.__version__


def test_memory_manager_constructs_without_heavy_backends():
    """Constructing MemoryManager must not load embeddings/graph/vector backends.

    Heavy backends are deferred until initialize()/first use, so this works on
    the lite/core profile.
    """
    from agent_memory import MemoryManager

    manager = MemoryManager()
    assert manager.config is not None


def test_core_logic_modules_import_on_core_profile():
    """The core pipeline modules must import without the heavy extras.

    Heavy third-party imports (kuzu, qdrant_client, sentence_transformers,
    open_clip, torch, litellm) are deferred to first use, so importing these
    modules must succeed with only the core dependencies installed.
    """
    for mod in [
        "agent_memory.core.memory_manager",
        "agent_memory.core.compaction",
        "agent_memory.core.retrieval",
        "agent_memory.core.save_decision",
        "agent_memory.core.decay",
        "agent_memory.core.keyword_reweight",
        "agent_memory.core.dream_explorer",
        "agent_memory.storage.graph_store",
        "agent_memory.storage.vector_store",
        "agent_memory.embeddings.text_embedder",
        "agent_memory.embeddings.visual_embedder",
        "agent_memory.llm.client",
        "agent_memory.policy.controller",
        # Service + integration adapters must also import on the core profile;
        # their transport deps (mcp, fastapi) are imported lazily.
        "agent_memory.service",
        "agent_memory.integrations.mcp_server",
        "agent_memory.integrations.rest_server",
    ]:
        assert importlib.import_module(mod) is not None


def test_missing_extra_raises_actionable_error(monkeypatch):
    """A backend whose extra is absent must raise a clear install hint, not a
    bare ModuleNotFoundError at import time."""
    import builtins

    from agent_memory.embeddings.text_embedder import TextEmbedder

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    embedder = TextEmbedder()
    with pytest.raises(ImportError, match=r"agent-memory\[text\]"):
        _ = embedder.model
