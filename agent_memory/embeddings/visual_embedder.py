"""Open-CLIP wrapper for visual/spatial embeddings.

Embeds scene descriptions (text) using CLIP to provide an independent
retrieval channel based on spatial/perceptual similarity.
"""

from __future__ import annotations

import logging

from agent_memory.config import MEMORY_CONFIG

logger = logging.getLogger(__name__)


class VisualEmbedder:
    """Lazy-loading wrapper around open_clip for scene description embeddings.

    ``open_clip`` and ``torch`` are imported only when a model is first loaded,
    so importing this module does not pull in the heavy ``visual`` extra. The
    visual layer is optional and disabled by default on the lite/edge profile.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or MEMORY_CONFIG["clip_model"]
        self._model = None
        self._tokenizer = None
        self._dimension: int | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import open_clip
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "VisualEmbedder requires the 'visual' extra. "
                "Install with: pip install agent-memory[visual]"
            ) from exc
        logger.info("Loading CLIP model: %s", self._model_name)
        self._model, _, _ = open_clip.create_model_and_transforms(
            self._model_name, pretrained="openai",
        )
        self._tokenizer = open_clip.get_tokenizer(self._model_name)
        self._model.eval()

    @property
    def dimension(self) -> int:
        self._load()
        if self._dimension is None:
            import torch

            # Infer from a dummy forward pass
            dummy = self._tokenizer(["test"])
            with torch.no_grad():
                out = self._model.encode_text(dummy)
            self._dimension = out.shape[-1]
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """Embed a scene description text using CLIP. Returns list of floats."""
        import torch

        self._load()
        tokens = self._tokenizer([text])
        with torch.no_grad():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().tolist()

    def embed_to_bytes(self, text: str) -> bytes:
        """Embed and return raw bytes for storage in SQLite BLOB column."""
        import struct
        floats = self.embed(text)
        return struct.pack(f"{len(floats)}f", *floats)

    def bytes_to_vector(self, data: bytes) -> list[float]:
        """Convert raw bytes back to float list."""
        import struct
        count = len(data) // 4
        return list(struct.unpack(f"{count}f", data))
