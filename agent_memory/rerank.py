"""Optional second-stage rerankers.

Retrieval fuses several cheap first-stage layers; a reranker then rescores the
top ``rerank_depth`` candidates jointly with the query. A cross-encoder over a
wide dense candidate pool is the component the raw-RAG literature finds most
under-used in published memory baselines.

Rerankers are injected (``MemoryManager(reranker=...)``) or created from
``MEMORY_CONFIG["reranker_model"]``; none is used by default so the core
install stays light.
"""

from __future__ import annotations

import logging
from typing import Callable, Protocol, Sequence, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class RerankerProtocol(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Relevance score per document (higher is better; any scale)."""
        ...


class CrossEncoderReranker:
    """sentence-transformers ``CrossEncoder`` (requires the ``text`` extra)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ImportError(
                    "CrossEncoderReranker requires the 'text' extra. "
                    "Install with: pip install agent-memory[text]"
                ) from exc
            logger.info("Loading reranker model: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        return [float(s) for s in self.model.predict([(query, d) for d in documents])]


class CallableReranker:
    """Wrap any ``(query, documents) -> scores`` function (a hosted reranker API, …)."""

    def __init__(self, fn: Callable[[str, Sequence[str]], Sequence[float]]) -> None:
        self._fn = fn

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        return [float(s) for s in self._fn(query, documents)]


DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def reranker_from_config(
    config: dict, embeddings_loaded: bool = False, text_embedder=None,
) -> RerankerProtocol | None:
    """Build the configured reranker.

    ``reranker_model``:
      - ``None``   — no reranking.
      - ``"auto"`` — a MiniLM cross-encoder, but only on the full local-model
        profile: embeddings loaded with the sentence-transformers
        ``TextEmbedder``. Lite, edge and injected-embedder setups get none, so
        they never download a model they did not ask for.
      - any other string — that cross-encoder model.
    """
    name = config.get("reranker_model")
    if not name:
        return None
    if name == "auto":
        from agent_memory.embeddings.text_embedder import TextEmbedder

        if not embeddings_loaded or not isinstance(text_embedder, TextEmbedder):
            return None
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return None
        return CrossEncoderReranker(DEFAULT_CROSS_ENCODER)
    return CrossEncoderReranker(name)
