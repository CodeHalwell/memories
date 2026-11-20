"""Shared fixtures: managers on the lite and edge profiles, and an LLM stub."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import pytest_asyncio

from agent_memory import HashingTextEmbedder, MemoryManager, NullVisualEmbedder


def make_llm_stub(salience: float = 0.5, entities=None, answer_for=None):
    """A deterministic stand-in for every LLM call the system makes."""

    async def stub(prompt, system=None, **kwargs):
        if "Combine these" in prompt:
            parts = re.findall(r"<memory_\d+>\n(.*?)\n</memory_\d+>", prompt, flags=re.S)
            return {"content": " ".join(parts), "summary": "merged", "keywords": [], "salience": 0.5}
        if "questions with answers" in prompt:
            src = re.search(r"<source_text>\n(.*?)\n</source_text>", prompt, flags=re.S).group(1)
            answer = answer_for(src) if answer_for else src.split()[0]
            return [{"question": "what?", "answer": answer}]
        if "emotional tone" in prompt:
            return {"valence": 0.0, "arousal": 0.0, "surprise": 0.0}
        return {
            "should_save": True, "confidence": 0.9, "reason": "stub",
            "keywords": [{"keyword": "billing", "weight": 1.0}, {"keyword": "invoice", "weight": 1.0}],
            "valence": 0.0, "arousal": 0.1, "surprise": 0.1, "summary": "s",
            "salience": salience, "entities": entities or [],
        }

    return stub


@pytest.fixture
def stub_llm(monkeypatch):
    """Enable a stub LLM for save decisions and compaction."""
    import agent_memory.core.compaction as compaction_mod
    import agent_memory.core.memory_manager as manager_mod
    import agent_memory.core.save_decision as save_mod

    def install(**kwargs):
        stub = make_llm_stub(**kwargs)
        for mod in (save_mod, compaction_mod):
            monkeypatch.setattr(mod, "llm_complete_json", stub)
            monkeypatch.setattr(mod, "llm_available", lambda *a, **k: True)
        monkeypatch.setattr(manager_mod, "llm_available", lambda *a, **k: False)  # no visual/dream
        return stub

    return install


@pytest_asyncio.fixture
async def lite_manager(tmp_path: Path):
    """No embeddings, no LLM: BM25 (memories + raw turns) and graph only."""
    mgr = MemoryManager(data_dir=tmp_path, config={"llm_enabled": False, "policy_auto_train": False})
    await mgr.initialize(load_embeddings=False)
    yield mgr
    await mgr.close()


@pytest_asyncio.fixture
async def edge_manager(tmp_path: Path):
    """Torch-free semantic profile (hashing embedder + Qdrant), no LLM."""
    pytest.importorskip("qdrant_client")
    mgr = MemoryManager(
        data_dir=tmp_path, text_embedder=HashingTextEmbedder(256),
        visual_embedder=NullVisualEmbedder(),
        config={"llm_enabled": False, "policy_auto_train": False},
    )
    await mgr.initialize(load_embeddings=True)
    yield mgr
    await mgr.close()
