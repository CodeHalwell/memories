"""Sentence-transformers wrapper for text embeddings.

Used for semantic similarity search in the Qdrant vector store.
Runs locally on CPU or GPU.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agent_memory.config import MEMORY_CONFIG

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class TextEmbedder:
    """Lazy-loading wrapper around sentence-transformers.

    The ``sentence-transformers`` import is deferred until a model is first
    loaded, so importing this module does not require the ``text`` extra to be
    installed. Consumers that never touch local embeddings (e.g. the edge/lite
    profile or remote-embedding setups) pay no dependency cost.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or MEMORY_CONFIG["text_embedding_model"]
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ImportError(
                    "TextEmbedder requires the 'text' extra. "
                    "Install with: pip install agent-memory[text]"
                ) from exc
            logger.info("Loading text embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a list of floats."""
        vector = self.model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Returns a list of float lists."""
        vectors = self.model.encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in vectors]
