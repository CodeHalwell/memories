// Policy controller interface (A5 — stub for v2).
//
// Defines the interface for a learned memory policy that will eventually
// replace fixed heuristics. Hard constraints cannot be overridden.

using Microsoft.Extensions.Logging;

namespace AgentMemory.Policy;

/// <summary>Immutable policy constraints that cannot be overridden by any learned policy.</summary>
public static class PolicyHardConstraints
{
    // Save constraints
    public const double MinSaveRate = 0.05;
    public const double MaxSaveRate = 0.50;
    public const bool FastPathOverride = true;

    // Retrieval constraints
    public const int MinLayers = 1;
    public const int MaxGraphDepth = 4;
    public const int MaxTopK = 20;

    // Compaction constraints
    public const bool NeverDeleteRawLogs = true;
    public const bool NeverCompactFastPathGen0 = true;
    public const bool RequireMergeValidation = true;
}

/// <summary>State vector for policy decisions (A5.2).</summary>
public sealed class PolicyState
{
    public int TurnNumber { get; set; }
    public int SessionLength { get; set; }
    public double TimeSinceLastSave { get; set; }
    public int ContentLength { get; set; }
    public double EmotionalValence { get; set; }
    public double EmotionalArousal { get; set; }
    public double EmotionalSurprise { get; set; }
    public int HotTierCount { get; set; }
    public double RecentRetrievalHitRate { get; set; }
    public double RetrievalGapScore { get; set; }
    public int GraphNodeCount { get; set; }
    public double AvgEdgeDegree { get; set; }
    public int OrphanMemoryCount { get; set; }
    public double DaysSinceLastCompaction { get; set; }
    public int PendingMergeCandidates { get; set; }
}

/// <summary>
/// Stub policy controller — uses heuristics, logs decisions for future training.
/// In v2, will be replaced with a learned model.
/// </summary>
public sealed class PolicyController
{
    private readonly MemoryConfig _config;

    public PolicyController(MemoryConfig? config = null)
    {
        _config = config ?? new MemoryConfig();
    }

    /// <summary>Decide whether to save (heuristic — v1).</summary>
    public bool ShouldSave(PolicyState state, double llmConfidence)
        => llmConfidence >= _config.SaveConfidenceThreshold;

    /// <summary>Return retrieval parameters (heuristic — v1).</summary>
    public Dictionary<string, object> RetrievalConfig(PolicyState state) => new()
    {
        ["layers"] = _config.RetrievalLayers,
        ["graph_depth"] = Math.Min(
            _config.GraphTraversalDepth,
            PolicyHardConstraints.MaxGraphDepth),
        ["mood_weight"] = _config.MoodCongruentWeight,
        ["top_k"] = Math.Min(
            _config.TopKPerLayer,
            PolicyHardConstraints.MaxTopK),
    };

    /// <summary>Return compaction urgency score (heuristic — v1).</summary>
    public double CompactionPriority(PolicyState state)
    {
        if (state.HotTierCount > _config.HotTierThreshold)
            return 1.0;
        return (double)state.HotTierCount / Math.Max(_config.HotTierThreshold, 1);
    }
}
