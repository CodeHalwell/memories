// Decay scoring for memory access patterns.
//
// Implements time-based exponential decay combined with frequency-based persistence.
// A2.2: emotional salience slows decay, and semantic memories have a floor.

namespace AgentMemory.Core;

/// <summary>Decay computation for memory access patterns.</summary>
public static class Decay
{
    /// <summary>
    /// Compute a decay score between 0.0 and ~1.0.
    ///
    /// Higher scores indicate more "alive" memories. The score combines:
    ///   - Recency: exponential decay based on days since last access
    ///   - Frequency: logarithmic scaling of access count
    ///   - Emotional boost: high arousal + surprise slows decay (A2.2)
    ///   - Semantic floor: compacted memories never fully decay (A2.2)
    /// </summary>
    public static double ComputeDecay(
        DateTime lastAccessed,
        int accessCount,
        double arousal = 0.0,
        double surprise = 0.0,
        bool isSemantic = false,
        MemoryConfig? config = null)
    {
        config ??= new MemoryConfig();

        var now = DateTime.UtcNow;

        // Ensure UTC
        if (lastAccessed.Kind == DateTimeKind.Unspecified)
            lastAccessed = DateTime.SpecifyKind(lastAccessed, DateTimeKind.Utc);
        else if (lastAccessed.Kind == DateTimeKind.Local)
            lastAccessed = lastAccessed.ToUniversalTime();

        var daysSince = Math.Max((now - lastAccessed).TotalDays, 0.0);

        var halflife = config.DecayHalflifeDays;
        var lambda = halflife > 0 ? Math.Log(2) / halflife : 0.1;

        // A2.2: Emotional memories decay more slowly
        // arousal + surprise in [0, 2], so boost is in [1.0, 2.0]
        var emotionalBoost = 1.0 + 0.5 * (arousal + surprise);
        var recency = Math.Exp(-lambda * daysSince / emotionalBoost);

        var frequency = Math.Log(1.0 + accessCount) / 10.0;

        // A2.2: Semantic (compacted) memories have a flatter decay curve
        if (isSemantic)
            recency = Math.Max(recency, 0.3);

        return Math.Round(
            config.DecayRecencyWeight * recency +
            config.DecayFrequencyWeight * frequency,
            4);
    }
}
