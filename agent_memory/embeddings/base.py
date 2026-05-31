"""Embedder provider interfaces and lightweight, dependency-free implementations.

The memory system depends on embedders only through the small structural
``Protocol``s defined here, so any conforming object can be injected into
``MemoryManager`` / ``MemoryService``. This is the seam that lets the system run
on the edge/serverless profile **without torch**: supply a hashing or remote
embedder instead of the default sentence-transformers / CLIP wrappers.

Provided implementations:
  - ``HashingTextEmbedder`` — offline, deterministic lexical embeddings.
  - ``CallableTextEmbedder`` — adapt any ``str -> sequence[float]`` function
    (e.g. an ONNX model or a remote embeddings API).
  - ``NullVisualEmbedder`` — disables the visual layer (text-only deployments).
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Callable, Protocol, Sequence, runtime_checkable


@runtime_checkable
class TextEmbedderProtocol(Protocol):
    """Structural type for text embedders consumed by the memory system."""

    @property
    def dimension(self) -> int:
        """Dimensionality of the produced vectors."""
        ...

    def embed(self, text: str) -> list[float]:
        """Embed a single string into a vector."""
        ...


@runtime_checkable
class VisualEmbedderProtocol(Protocol):
    """Structural type for visual/spatial embedders (optional layer)."""

    @property
    def dimension(self) -> int:
        ...

    def embed(self, text: str) -> list[float]:
        ...


class HashingTextEmbedder:
    """Deterministic, dependency-free text embedder using the hashing trick.

    Tokenizes text and accumulates signed token counts into a fixed-dimension
    vector (feature hashing), then L2-normalizes. It is fully offline and
    reproducible — suitable for edge / air-gapped deployments and CI where the
    ``text`` extra (sentence-transformers + torch) is unavailable. It captures
    lexical (token-overlap) similarity under cosine distance, not deep semantic
    similarity; for the latter, inject a model-backed or remote embedder.
    """

    _TOKEN_RE = re.compile(r"[a-z0-9]+")

    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _tokens(self, text: str) -> list[str]:
        return self._TOKEN_RE.findall(text.lower())

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dimension
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "big")
            index = h % self._dimension
            sign = 1.0 if (h // self._dimension) % 2 == 0 else -1.0
            vec[index] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class CallableTextEmbedder:
    """Adapt any ``str -> sequence[float]`` function into a text embedder.

    The escape hatch for plugging in a custom embedding source (a quantized
    local ONNX model, a remote embeddings API, etc.) without the system needing
    to know about it. Provide the output ``dimension`` explicitly.
    """

    def __init__(self, fn: Callable[[str], Sequence[float]], dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._fn = fn
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        return [float(x) for x in self._fn(text)]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class NullVisualEmbedder:
    """A no-op visual embedder for text-only / edge deployments.

    Reports a nominal dimension (so the vector store can allocate the visual
    collection) but does not produce embeddings. ``MemoryManager`` detects it
    via ``is_null`` and skips the visual layer entirely — importantly, reading
    ``dimension`` here never loads CLIP/torch.
    """

    is_null = True

    def __init__(self, dimension: int = 512) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:  # pragma: no cover - never called
        raise NotImplementedError(
            "visual embedding is disabled (NullVisualEmbedder); inject a real "
            "visual embedder to enable the visual layer"
        )
