"""Tests for pluggable embedder providers (roadmap §2.3)."""

import math

import pytest

from agent_memory.embeddings.base import (
    CallableTextEmbedder,
    HashingTextEmbedder,
    NullVisualEmbedder,
    TextEmbedderProtocol,
    VisualEmbedderProtocol,
)


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def test_implementations_satisfy_protocols():
    assert isinstance(HashingTextEmbedder(), TextEmbedderProtocol)
    assert isinstance(CallableTextEmbedder(lambda s: [0.0], 1), TextEmbedderProtocol)
    assert isinstance(NullVisualEmbedder(), VisualEmbedderProtocol)


def test_hashing_embedder_dimension_and_shape():
    emb = HashingTextEmbedder(dimension=128)
    assert emb.dimension == 128
    v = emb.embed("hello world")
    assert len(v) == 128


def test_hashing_embedder_is_deterministic():
    emb = HashingTextEmbedder()
    assert emb.embed("the quick brown fox") == emb.embed("the quick brown fox")


def test_hashing_embedder_is_l2_normalized():
    v = HashingTextEmbedder().embed("some tokens here")
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-9)


def test_hashing_embedder_empty_text_is_zero_vector():
    v = HashingTextEmbedder(dimension=32).embed("")
    assert v == [0.0] * 32


def test_hashing_embedder_captures_lexical_similarity():
    emb = HashingTextEmbedder(dimension=512)
    base = emb.embed("python async retrieval memory")
    similar = emb.embed("python async memory retrieval system")
    disjoint = emb.embed("banana orchard harvest weather")
    assert _cosine(base, similar) > _cosine(base, disjoint)


def test_callable_embedder_wraps_function():
    emb = CallableTextEmbedder(lambda s: [float(len(s)), 1.0], dimension=2)
    assert emb.dimension == 2
    assert emb.embed("abc") == [3.0, 1.0]


def test_invalid_dimension_rejected():
    with pytest.raises(ValueError):
        HashingTextEmbedder(dimension=0)
    with pytest.raises(ValueError):
        CallableTextEmbedder(lambda s: [], 0)


def test_null_visual_embedder_reports_dimension_but_refuses_embed():
    nve = NullVisualEmbedder(dimension=512)
    assert nve.dimension == 512
    assert nve.is_null is True
    with pytest.raises(NotImplementedError):
        nve.embed("anything")
