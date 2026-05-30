"""Memory Manager — orchestrates save, retrieve, and compact operations.

This is the primary interface for the agent runtime. It coordinates all
storage backends, embedding models, and the LLM-driven save decision pipeline.

Addendum integrations:
  - A2.1: Gap-aware save decisions (save_decision.py)
  - A2.2: Emotional decay modulation (decay.py)
  - A2.3: Generation gap guard (compaction.py)
  - A2.4: Merge validation (compaction.py)
  - A2.5: Keyword reweighting during compaction
  - A3:   Dream exploration during sleep cycle
  - A4:   Policy logging for retrieval decisions and outcome assessment
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agent_memory.config import DATA_DIR, MEMORY_CONFIG
from agent_memory.core.compaction import CompactionEngine
from agent_memory.core.decay import compute_decay
from agent_memory.core.dream_explorer import commit_discoveries, exploratory_walk
from agent_memory.core.keyword_reweight import reweight_keywords_from_graph
from agent_memory.core.retrieval import RetrievalEngine
from agent_memory.core.save_decision import make_save_decision
from agent_memory.embeddings.text_embedder import TextEmbedder
from agent_memory.embeddings.visual_embedder import VisualEmbedder
from agent_memory.models import CompactionResult, Memory, RawLogEntry
from agent_memory.policy.outcome_assessor import (
    assess_retrieval_outcomes,
    assess_save_outcomes,
)
from agent_memory.storage.graph_store import GraphStore
from agent_memory.storage.jsonl_log import JSONLLogger
from agent_memory.storage.sqlite_store import SQLiteStore
from agent_memory.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class MemoryManager:
    """Top-level orchestrator for the agent memory system."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or DATA_DIR
        self.config = MEMORY_CONFIG

        # Storage backends
        self.jsonl = JSONLLogger(self.data_dir / "logs" / "sessions")
        self.sqlite = SQLiteStore(self.data_dir / "memory.db")
        self.graph = GraphStore(self.data_dir / "graph")
        self.vector = VectorStore(self.data_dir / "vectors")

        # Embedding models (lazy-loaded)
        self.text_embedder = TextEmbedder()
        self.visual_embedder = VisualEmbedder()

        # Sub-engines (initialized after storage is ready)
        self._retrieval: RetrievalEngine | None = None
        self._compaction: CompactionEngine | None = None
        self._embeddings_loaded: bool = False

        # Track first turns per session
        self._first_turns: set[str] = set()

    async def initialize(self, load_embeddings: bool = True) -> None:
        """Initialize storage backends and the retrieval/compaction engines.

        Args:
            load_embeddings: When True (default), load the text/visual embedding
                models and initialize the vector store — required for semantic
                and visual retrieval. When False, run on the **lite profile**:
                storage and engines are wired, but the embedding models and
                vector store are left unloaded. Retrieval then degrades
                gracefully to the grep + keyword + graph layers, and the
                semantic/visual layers are skipped. This lets the system run on
                edge/serverless targets without the ``text``/``visual`` extras.
        """
        await self.sqlite.initialize()
        self.graph.initialize()
        if load_embeddings:
            self.vector.initialize(
                text_dim=self.text_embedder.dimension,
                visual_dim=self.visual_embedder.dimension,
            )
        self._embeddings_loaded = load_embeddings

        self._retrieval = RetrievalEngine(
            sqlite=self.sqlite,
            graph=self.graph,
            vector=self.vector,
            text_embedder=self.text_embedder,
            # In lite mode, drop the visual embedder so the visual layer is
            # skipped cleanly instead of failing per-query.
            visual_embedder=self.visual_embedder if load_embeddings else None,
            embeddings_available=load_embeddings,
        )
        self._compaction = CompactionEngine(
            sqlite=self.sqlite,
            graph=self.graph,
            vector=self.vector,
            text_embedder=self.text_embedder,
            visual_embedder=self.visual_embedder,
        )

    async def initialize_lite(self) -> None:
        """Initialize without loading embedding models (for testing or lightweight use)."""
        await self.sqlite.initialize()
        self.graph.initialize()

    async def close(self) -> None:
        """Close all storage backends."""
        await self.sqlite.close()
        self.graph.close()
        self.vector.close()

    # ── Core operations ──

    async def process_turn(
        self, session_id: str, turn: int, content: str,
        role: str = "assistant", token_count: int = 0,
        model: str = "", provider: str = "",
    ) -> Memory | None:
        """Log an agent output and decide whether to save it as a memory.

        This is the main entry point called at the end of each turn.

        Returns the Memory if one was created, else None.
        """
        # 1. Create and persist raw log entry
        entry = RawLogEntry(
            session_id=session_id,
            turn=turn,
            content=content,
            role=role,
            token_count=token_count,
            model=model,
            provider=provider,
        )
        file_path, byte_offset = self.jsonl.append(entry)

        # 2. Index the raw log entry in SQLite
        await self.sqlite.index_raw_log(
            entry_id=entry.id,
            session_id=session_id,
            turn=turn,
            timestamp=entry.timestamp,
            file_path=file_path,
            byte_offset=byte_offset,
        )

        # 3. Run save decision (A2.1: pass sqlite for gap awareness)
        is_first = session_id not in self._first_turns
        if is_first:
            self._first_turns.add(session_id)

        decision, memory = await make_save_decision(
            entry, is_first_turn=is_first, sqlite=self.sqlite,
        )

        # 4. Log the decision
        await self.sqlite.log_save_decision(decision)

        if memory is None:
            return None

        # 5. Save the memory
        await self.sqlite.save_memory(memory)

        # 6. Create graph node
        self.graph.add_memory_node(
            memory_id=memory.id,
            summary=memory.summary or "",
            tier=memory.tier,
            salience=memory.salience,
            valence=memory.valence,
            compaction_gen=memory.compaction_gen,
            created_at=memory.created_at,
        )
        memory.graph_node_id = memory.id
        await self.sqlite.update_memory_graph_ref(memory.id, memory.id)

        # 7. Create text embedding and store in Qdrant (skipped on lite profile)
        if not self._embeddings_loaded:
            return memory
        try:
            text_vector = self.text_embedder.embed(memory.content)
            point_id = self.vector.upsert_text_vector(
                memory_id=memory.id,
                vector=text_vector,
                tier=memory.tier,
                valence=memory.valence,
                arousal=memory.arousal,
                session_id=session_id,
                created_at=memory.created_at,
            )
            memory.vector_id = point_id
            await self.sqlite.update_memory_vector_ref(memory.id, point_id)
        except Exception:
            logger.exception("Failed to create text embedding for memory %s", memory.id)

        # 8. Visual layer — generate scene description and CLIP embedding for salient memories
        if memory.salience > self.config["visual_salience_threshold"]:
            await self._generate_visual_layer(memory)

        return memory

    async def retrieve(
        self, query: str, session_id: str | None = None,
        top_k: int | None = None,
    ) -> list[Memory]:
        """Run three-layer retrieval and return ranked memories.

        Also logs the retrieval decision for policy training (A4).
        """
        if self._retrieval is None:
            raise RuntimeError("MemoryManager not initialized — call initialize() first")

        memories = await self._retrieval.retrieve(
            query=query, session_id=session_id, top_k=top_k,
            # Mood-congruent weighting needs the LLM; skip it on the lite
            # profile, which typically runs without the 'llm' extra.
            enable_mood_congruent=self._embeddings_loaded,
        )

        # A4: Log retrieval decision
        if self.config.get("policy_logging_enabled"):
            try:
                now = datetime.now(timezone.utc).isoformat()
                await self.sqlite.log_retrieval_decision(
                    decision_id=str(uuid.uuid4()),
                    session_id=session_id or "",
                    turn=None,
                    query=query,
                    decided_at=now,
                    layers_queried=self.config["retrieval_layers"],
                    graph_depth=self.config["graph_traversal_depth"],
                    mood_weight=self.config["mood_congruent_weight"],
                    top_k=top_k or self.config["top_k_per_layer"],
                    memory_ids=[m.id for m in memories],
                    return_count=len(memories),
                )
            except Exception:
                logger.debug("Failed to log retrieval decision")

        return memories

    async def run_compaction(self, trigger: str = "scheduled") -> CompactionResult:
        """Run a compaction cycle with optional exploration phase.

        Phases:
          1. Standard compaction (merge pass)
          2. Keyword reweighting from graph structure (A2.5)
          3. Exploratory walk / dream phase (A3, scheduled/manual only)
          4. Outcome assessment (A4)
        """
        if self._compaction is None:
            raise RuntimeError("MemoryManager not initialized — call initialize() first")

        # Phase 1: Standard compaction
        result = await self._compaction.run(trigger=trigger)

        # Phase 2: Keyword reweighting (A2.5)
        try:
            keywords_updated = await reweight_keywords_from_graph(self.sqlite, self.graph)
            result.keywords_updated = keywords_updated
        except Exception:
            logger.exception("Keyword reweighting failed")

        # Phase 3: Dream exploration (A3) — only for scheduled/manual triggers
        if trigger in ("scheduled", "manual") and self.config.get("dream_enabled"):
            try:
                discoveries = await exploratory_walk(
                    sqlite=self.sqlite,
                    graph=self.graph,
                    vector=self.vector,
                    text_embedder=self.text_embedder,
                )
                if discoveries:
                    committed = await commit_discoveries(
                        discoveries, self.graph, self.sqlite,
                    )
                    result.edges_discovered = committed
            except Exception:
                logger.exception("Dream exploration failed")

        # Phase 4: Outcome assessment (A4)
        if self.config.get("policy_logging_enabled"):
            try:
                await assess_save_outcomes(self.sqlite)
                await assess_retrieval_outcomes(self.sqlite)
            except Exception:
                logger.debug("Outcome assessment failed")

        return result

    async def get_memory(self, memory_id: str) -> Memory | None:
        """Fetch a single memory and log the access."""
        mem = await self.sqlite.get_memory(memory_id)
        if mem is None:
            return None

        now = datetime.now(timezone.utc).isoformat()
        mem.access_count += 1
        mem.last_accessed = now
        mem.decay_score = compute_decay(
            datetime.fromisoformat(now), mem.access_count,
            arousal=mem.arousal, surprise=mem.surprise,
            is_semantic=mem.is_semantic,
        )

        await self.sqlite.log_access(
            access_id=str(uuid.uuid4()),
            memory_id=memory_id,
            accessed_at=now,
            access_type="primary",
        )
        await self.sqlite.update_memory_access(
            memory_id, mem.decay_score, mem.access_count, now,
        )
        return mem

    # ── Visual layer ──

    async def _generate_visual_layer(self, memory: Memory) -> None:
        """Generate a scene description and CLIP embedding for a memory."""
        try:
            # Generate scene description via LLM
            from agent_memory.llm.client import llm_complete
            scene = await llm_complete(
                f"Generate an abstract scene description for this memory:\n\n<memory_content>\n{memory.content}\n</memory_content>",
                system=self.config["prompts"]["scene_description"],
            )
            memory.scene_description = scene.strip()

            # Create CLIP embedding
            spatial_bytes = self.visual_embedder.embed_to_bytes(memory.scene_description)
            memory.spatial_embedding = spatial_bytes

            # Store in Qdrant visual collection
            visual_vector = self.visual_embedder.embed(memory.scene_description)
            self.vector.upsert_visual_vector(
                memory_id=memory.id,
                vector=visual_vector,
                session_id=memory.session_id,
                created_at=memory.created_at,
            )

            # Update SQLite
            await self.sqlite.update_memory_visual(
                memory.id, memory.scene_description, spatial_bytes,
            )
        except Exception:
            logger.exception("Visual layer generation failed for memory %s", memory.id)
