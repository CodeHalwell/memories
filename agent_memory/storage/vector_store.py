"""Qdrant vector store for semantic similarity search.

Runs in embedded (local) mode — no server required. Manages two collections:
  - memory_text: sentence-transformer embeddings of memory content
  - memory_visual: CLIP embeddings of scene descriptions
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from agent_memory.config import VECTOR_DIR

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

TEXT_COLLECTION = "memory_text"
VISUAL_COLLECTION = "memory_visual"


class VectorStore:
    """Qdrant-backed vector store for memory embeddings.

    ``qdrant_client`` is imported lazily so this module is importable without
    the ``vectors`` extra. Profiles that retrieve only via grep/keyword (e.g.
    the lite/edge profile) need not install Qdrant.
    """

    def __init__(self, vector_dir: Path | None = None) -> None:
        self.vector_dir = vector_dir or VECTOR_DIR
        self._client: QdrantClient | None = None

    def initialize(self, text_dim: int = 384, visual_dim: int = 512) -> None:
        """Initialize Qdrant in embedded mode and ensure collections exist."""
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "VectorStore requires the 'vectors' extra. "
                "Install with: pip install agent-memory[vectors]"
            ) from exc
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(self.vector_dir))
        self._ensure_collection(TEXT_COLLECTION, text_dim)
        self._ensure_collection(VISUAL_COLLECTION, visual_dim)

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    @property
    def client(self) -> QdrantClient:
        assert self._client is not None, "VectorStore not initialized — call initialize() first"
        return self._client

    def _ensure_collection(self, name: str, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        collections = [c.name for c in self.client.get_collections().collections]
        if name in collections:
            # Guard against reopening a data directory with an embedder whose
            # output dimension differs from the existing collection. Qdrant would
            # otherwise silently reject upserts/searches and semantic retrieval
            # would degrade with only swallowed errors. Fail clearly instead.
            try:
                existing_dim = self.client.get_collection(name).config.params.vectors.size
            except Exception:
                existing_dim = None
            if existing_dim is not None and existing_dim != dim:
                raise ValueError(
                    f"Vector collection '{name}' was created with dimension "
                    f"{existing_dim}, but the configured embedder produces dimension "
                    f"{dim}. The embedding provider/model changed for an existing "
                    f"data directory. Use a fresh data directory, or an embedder "
                    f"whose dimension is {existing_dim}."
                )
            return
        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    # ── Text embeddings ──

    def upsert_text_vector(
        self, memory_id: str, vector: list[float],
        tier: str = "hot", valence: float = 0.0, arousal: float = 0.0,
        session_id: str = "", created_at: str = "", namespace: str = "default",
    ) -> str:
        """Insert or update a text embedding. Returns the point ID."""
        from qdrant_client.models import PointStruct

        point_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=TEXT_COLLECTION,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "memory_id": memory_id,
                        "tier": tier,
                        "valence": valence,
                        "arousal": arousal,
                        "session_id": session_id,
                        "created_at": created_at,
                        "namespace": namespace,
                    },
                )
            ],
        )
        return point_id

    def search_text(
        self, query_vector: list[float], limit: int = 5,
        tier_filter: str | None = None, namespace: str | None = None,
    ) -> list[dict]:
        """Search for nearest text embeddings.

        Returns list of dicts: {memory_id, score, tier, valence, arousal}.
        Results are restricted to ``namespace`` when provided.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        must = []
        if tier_filter:
            must.append(FieldCondition(key="tier", match=MatchValue(value=tier_filter)))
        if namespace is not None:
            must.append(FieldCondition(key="namespace", match=MatchValue(value=namespace)))
        search_filter = Filter(must=must) if must else None

        results = self.client.query_points(
            collection_name=TEXT_COLLECTION,
            query=query_vector,
            limit=limit,
            query_filter=search_filter,
        )
        return [
            {
                "memory_id": r.payload["memory_id"],
                "score": r.score,
                "tier": r.payload.get("tier", "hot"),
                "valence": r.payload.get("valence", 0.0),
                "arousal": r.payload.get("arousal", 0.0),
            }
            for r in results.points
        ]

    # ── Visual embeddings ──

    def upsert_visual_vector(
        self, memory_id: str, vector: list[float],
        session_id: str = "", created_at: str = "", namespace: str = "default",
    ) -> str:
        """Insert or update a visual (CLIP) embedding. Returns the point ID."""
        from qdrant_client.models import PointStruct

        point_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=VISUAL_COLLECTION,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "memory_id": memory_id,
                        "session_id": session_id,
                        "created_at": created_at,
                        "namespace": namespace,
                    },
                )
            ],
        )
        return point_id

    def search_visual(
        self, query_vector: list[float], limit: int = 5,
        namespace: str | None = None,
    ) -> list[dict]:
        """Search for nearest visual embeddings.

        Returns list of dicts: {memory_id, score}. Restricted to ``namespace``
        when provided.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        search_filter = None
        if namespace is not None:
            search_filter = Filter(
                must=[FieldCondition(key="namespace", match=MatchValue(value=namespace))]
            )
        results = self.client.query_points(
            collection_name=VISUAL_COLLECTION,
            query=query_vector,
            limit=limit,
            query_filter=search_filter,
        )
        return [
            {"memory_id": r.payload["memory_id"], "score": r.score}
            for r in results.points
        ]

    def similarity(self, point_id_a: str, point_id_b: str) -> float | None:
        """Compute cosine similarity between two points in the text collection.

        Returns the similarity score or None if either point is not found.
        Used by dream explorer (A3) for cross-session similarity checks.
        """
        try:
            points = self.client.retrieve(
                collection_name=TEXT_COLLECTION,
                ids=[point_id_a, point_id_b],
                with_vectors=True,
            )
            if len(points) < 2:
                return None
            import numpy as np
            a = np.array(points[0].vector)
            b = np.array(points[1].vector)
            dot = np.dot(a, b)
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            return float(dot / norm) if norm > 0 else 0.0
        except Exception:
            return None

    def delete_point(self, collection: str, memory_id: str) -> None:
        """Delete all points for a given memory_id from a collection."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self.client.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[FieldCondition(key="memory_id", match=MatchValue(value=memory_id))]
            ),
        )
