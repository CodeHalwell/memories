"""Configuration for the Agent Memory System."""

from pathlib import Path

from agent_memory import prompts

# Base data directory — all runtime data stored here
DATA_DIR = Path("data")
LOG_DIR = DATA_DIR / "logs" / "sessions"
DB_PATH = DATA_DIR / "memory.db"
GRAPH_DIR = DATA_DIR / "graph"
VECTOR_DIR = DATA_DIR / "vectors"
POLICY_DATA_DIR = DATA_DIR / "policy_data"

MEMORY_CONFIG = {
    # LLM
    "llm_model": "claude-sonnet-4-6",
    "llm_temperature": 0.2,
    # Set False to run without any LLM calls (save decisions fall back to the
    # decision model's lexical features; merge/dream/mood phases are skipped).
    "llm_enabled": True,

    # Save thresholds
    "save_confidence_threshold": 0.5,
    "fast_path_arousal": 0.85,
    "fast_path_surprise": 0.75,
    "max_keywords_per_memory": 10,
    # Explicit user instructions that always save, checked *before* any LLM call.
    "explicit_save_markers": [
        "remember this", "remember that", "don't forget", "do not forget",
        "save this", "keep in mind", "make a note", "note that",
    ],
    # Kept for backward compatibility: the first turn of a session is saved.
    # Detection is persisted (SQLite), so it survives process restarts.
    "always_save_first_turn": True,

    # Save decision — retrieval gap awareness (A2.1)
    "gap_lookback_turns": 20,
    "gap_overlap_threshold": 0.3,
    "gap_threshold_reduction": 0.7,

    # Retrieval
    # Layers fused with reciprocal-rank fusion. "raw" is a BM25 index over every
    # logged turn (curated or not); "raw_semantic" is its dense counterpart.
    # "grep" (ripgrep over JSONL) is still available but superseded by "raw".
    "retrieval_layers": ["keyword", "raw", "semantic", "raw_semantic", "entity", "graph"],
    "retrieval_top_k": 10,           # results returned
    "layer_depth": 30,               # candidates taken from each layer
    "rrf_k": 60,
    "layer_weights": {
        "keyword": 1.0, "raw": 1.0, "semantic": 1.0, "raw_semantic": 1.0,
        "entity": 0.7, "grep": 0.5, "visual": 0.5, "graph": 0.5,
    },
    "graph_traversal_depth": 2,
    "graph_expansion_seeds": 5,      # top fused memories whose neighbours are added
    # Final score = relevance (fused rank, normalised to 1.0 at the top) + small
    # recency/importance/mood terms. Reciprocal-rank fusion has a flat top
    # (adjacent ranks differ by ~1.6% at k=60), so these are sized as
    # tie-breakers; at 0.05 importance alone was measured to override relevance
    # (LoCoMo evidence recall@1 halved). Decay never multiplies relevance away.
    "recency_weight": 0.02,
    "importance_weight": 0.01,
    "mood_congruent_weight": 0.01,
    "cold_tier_weight": 0.7,         # superseded (merged/deduplicated) memories rank lower
    "semantic_min_score": 0.15,      # below this a vector hit is treated as noise
    "rerank_depth": 50,              # candidates passed to an optional reranker
    "index_raw_vectors": True,       # embed every raw turn when embeddings are loaded
    "link_similarity_threshold": 0.8,  # write-time RELATES_TO links between memories
    "link_max_neighbours": 3,
    # Kept for compatibility with older callers/policy logging.
    "top_k_per_layer": 5,

    # Compaction
    "hot_tier_threshold": 500,
    "compaction_candidate_threshold": 0.7,
    "keyword_overlap_merge_threshold": 0.6,
    "valence_merge_exclusion_delta": 0.6,
    "dedupe_jaccard_threshold": 0.9,

    # Compaction — merge validation (A2.4)
    "merge_validation_queries": 5,
    "merge_degradation_tolerance": 0.15,
    # Fraction of generated (question, answer) pairs whose answer must survive
    # in the merged text. 1.0 = no observed fact loss is tolerated.
    "merge_min_answer_preservation": 1.0,

    # Compaction — generation gap guard (A2.3)
    "max_generation_gap_for_merge": 1,

    # Decay — recency of last access (or of the event, if never accessed).
    # Each access lengthens the effective half-life (spacing effect); arousal
    # and surprise lengthen it further (A2.2).
    "decay_halflife_days": 7,
    "decay_frequency_strength": 0.5,
    "decay_semantic_floor": 0.3,
    # Legacy keys (the previous additive formula); no longer used.
    "decay_recency_weight": 0.6,
    "decay_frequency_weight": 0.4,

    # Visual layer
    "visual_salience_threshold": 0.7,
    "clip_model": "ViT-B-32",

    # Embeddings
    "text_embedding_model": "all-MiniLM-L6-v2",
    # Cross-encoder reranker over the fused top `rerank_depth` candidates.
    # "auto" = MiniLM cross-encoder when the local sentence-transformers
    # embedder is in use (full profile), none otherwise; None disables; any
    # other value names a cross-encoder model. On LoCoMo it raised evidence
    # recall@1 from 0.30 (best first stage) to 0.46 (see docs/EVALUATION.md).
    "reranker_model": "auto",

    # Dream exploration (A3)
    "dream_walk_count": 50,
    "dream_similarity_threshold": 0.7,
    "dream_max_new_edges": 20,
    "dream_cluster_min_size": 3,
    "dream_enabled": True,

    # Policy logging (A4)
    "policy_logging_enabled": True,
    # A decision is labelled useful when its turn ranks in the top
    # `outcome_useful_rank` of the raw index for a later query in the same
    # namespace (reranked when a reranker is configured), or when usage
    # attribution marks it as used. Unused decisions are labelled not-useful
    # once older than this window. Rank 1 gave the most precise labels on
    # LoCoMo (precision 0.69 with reranking; see docs/DECISION_MODEL.md).
    "save_outcome_lookback_days": 14,
    "outcome_useful_rank": 1,
    "outcome_replay_max_queries": 2000,   # per namespace, per sleep cycle
    "retrieval_outcome_followup_turns": 3,
    "retrieval_outcome_followup_minutes": 10,
    "retrieval_outcome_keyword_overlap": 0.5,
    "usage_overlap_threshold": 0.25,

    # Policy training (A5) — learned save-decision model
    "policy_min_save_examples": 1000,
    "policy_min_retrieval_examples": 500,
    "policy_min_class_examples": 20,
    # Train during the sleep cycle when data allows. "auto" = only when a
    # reranker produces the replay labels: models trained on plain-BM25 replay
    # labels scored below the lexical prior on LoCoMo (AUC ~0.71 vs 0.77),
    # while reranked labels reached 0.80. True/False force it on/off;
    # train_policy() is always available manually.
    "policy_auto_train": "auto",
    "policy_promotion_margin": 0.01,  # learned model must beat heuristic F2 by this
    "policy_min_auc": 0.6,
    "policy_fbeta": 2.0,              # recall-weighted: a missed fact costs more
    "policy_validation_fraction": 0.25,

    # System Prompts
    "prompts": {
        "scene_description": prompts.SCENE_DESCRIPTION_SYSTEM,
        "save_decision": prompts.SAVE_DECISION_SYSTEM,
        "merge": prompts.MERGE_SYSTEM,
        "synthetic_query": prompts.SYNTHETIC_QUERY_SYSTEM,
        "emotion": prompts.EMOTION_SYSTEM,
        "classify_relationship": prompts.CLASSIFY_RELATIONSHIP_SYSTEM,
    },
}


def make_config(overrides: dict | None = None) -> dict:
    """Return a copy of :data:`MEMORY_CONFIG` with ``overrides`` applied.

    Nested dicts (``layer_weights``, ``prompts``) are merged one level deep so a
    caller can override a single weight without restating the rest.
    """
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in MEMORY_CONFIG.items()}
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value
    return cfg
