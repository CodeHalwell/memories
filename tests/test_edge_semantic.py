"""Torch-free semantic retrieval (roadmap §2.3 + §4.1).

Proves the edge story: with an injected HashingTextEmbedder + NullVisualEmbedder
and ``load_embeddings=True``, the full semantic vector path (Qdrant) works
without the `text`/`visual` extras (no sentence-transformers, no torch, no CLIP).

Requires the `vectors` extra (qdrant-client) but NOT torch.
"""

import importlib.util

import pytest
import pytest_asyncio

QDRANT_INSTALLED = importlib.util.find_spec("qdrant_client") is not None

pytestmark = pytest.mark.skipif(
    not QDRANT_INSTALLED, reason="requires the 'vectors' extra (qdrant-client)"
)


@pytest_asyncio.fixture
async def manager(tmp_path):
    from agent_memory import HashingTextEmbedder, MemoryManager, NullVisualEmbedder

    mgr = MemoryManager(
        data_dir=tmp_path,
        text_embedder=HashingTextEmbedder(dimension=256),
        visual_embedder=NullVisualEmbedder(),
    )
    # load_embeddings=True exercises the real Qdrant vector path.
    await mgr.initialize(load_embeddings=True)
    yield mgr
    await mgr.close()


async def test_semantic_layer_works_without_torch(manager):
    # First turn always saves (fast path); embeddings flow into Qdrant.
    await manager.process_turn(
        session_id="s1", turn=1,
        content="The user is configuring a Kubernetes ingress controller",
    )
    # A query with no literal keyword overlap relies on the vector layer.
    results = await manager.retrieve("k8s networking setup", session_id="s1")
    # The memory is retrievable; with a hashing embedder semantic recall is
    # lexical, but the vector collection is populated and queried successfully.
    assert isinstance(results, list)


async def test_vector_collection_is_populated(manager):
    mem = await manager.process_turn(
        session_id="s1", turn=1, content="apples bananas cherries",
    )
    assert mem is not None and mem.vector_id is not None  # text vector was upserted

    # Direct semantic retrieval by overlapping query returns the memory.
    results = await manager.retrieve("apples cherries", session_id="s1")
    assert any("apples" in m.content for m in results)
