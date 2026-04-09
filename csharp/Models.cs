// Data models for the Agent Memory System.

namespace AgentMemory;

/// <summary>Raw agent output logging — immutable ground truth.</summary>
public sealed class RawLogEntry
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string SessionId { get; set; } = "";
    public int Turn { get; set; }
    public string Timestamp { get; set; } = DateTime.UtcNow.ToString("o");
    public string Role { get; set; } = "assistant";
    public string Content { get; set; } = "";
    public int TokenCount { get; set; }
    public string Model { get; set; } = "";
    public string Provider { get; set; } = "";
}

/// <summary>Core memory record with emotional metadata, decay, compaction state, and cross-store references.</summary>
public sealed class Memory
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string CreatedAt { get; set; } = DateTime.UtcNow.ToString("o");
    public string UpdatedAt { get; set; } = DateTime.UtcNow.ToString("o");
    public string Content { get; set; } = "";
    public string? Summary { get; set; }
    public string RawLogId { get; set; } = "";
    public string SessionId { get; set; } = "";
    public int Turn { get; set; }

    // Emotional metadata
    public double Valence { get; set; }
    public double Arousal { get; set; }
    public double Surprise { get; set; }

    // Salience and access
    public double Salience { get; set; } = 0.5;
    public int AccessCount { get; set; }
    public string? LastAccessed { get; set; }
    public double DecayScore { get; set; } = 1.0;

    // Compaction state
    public int CompactionGen { get; set; }
    public string Tier { get; set; } = "hot";
    public bool FastPathed { get; set; }
    public bool IsSemantic { get; set; }

    // Cross-store references
    public string? GraphNodeId { get; set; }
    public string? VectorId { get; set; }

    // Visual layer
    public byte[]? SpatialEmbedding { get; set; }
    public string? SceneDescription { get; set; }

    // Keywords (stored in separate table)
    public List<(string Keyword, double Weight)> Keywords { get; set; } = [];
}

/// <summary>Save/skip/fast_path decision tracking with gap awareness.</summary>
public sealed class SaveDecision
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string RawLogId { get; set; } = "";
    public string SessionId { get; set; } = "";
    public int Turn { get; set; }
    public string DecidedAt { get; set; } = DateTime.UtcNow.ToString("o");
    public string Decision { get; set; } = "skip"; // save | skip | fast_path
    public string? Reason { get; set; }
    public double Confidence { get; set; }
    // A2.1 — retrieval gap awareness
    public bool GapTriggered { get; set; }
    public double? ThresholdUsed { get; set; }
}

/// <summary>Compaction run metrics.</summary>
public sealed class CompactionResult
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string RanAt { get; set; } = DateTime.UtcNow.ToString("o");
    public string Trigger { get; set; } = "scheduled";
    public int MemoriesReviewed { get; set; }
    public int MemoriesMerged { get; set; }
    public int MemoriesPruned { get; set; }
    public string? Notes { get; set; }
    // A2.5 / A3 — addendum tracking
    public int KeywordsUpdated { get; set; }
    public int EdgesDiscovered { get; set; }
}

/// <summary>Result of generative replay validation for a compaction merge (A2.4).</summary>
public sealed class MergeValidation
{
    public bool Passed { get; set; } = true;
    public double AvgSourceScore { get; set; }
    public double AvgMergedScore { get; set; }
    public double Degradation { get; set; }
    public List<string> QueriesTested { get; set; } = [];
}

/// <summary>An edge discovered during exploratory graph walks (A3).</summary>
public sealed class DiscoveredEdge
{
    public string SourceId { get; set; } = "";
    public string TargetId { get; set; } = "";
    public double Similarity { get; set; }
    public string RelationshipType { get; set; } = "";
    public string DiscoveryMethod { get; set; } = ""; // random_walk | cluster_bridge | temporal_proximity
}
