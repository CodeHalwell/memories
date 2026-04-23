// Decision-outcome pairing for policy training data (A4.3).
//
// Save decisions are assessed by checking whether the saved memory was ever
// retrieved. Retrieval decisions are assessed by checking whether the agent
// re-queried the same topic shortly afterward.

using AgentMemory.Storage;
using Microsoft.Extensions.Logging;

namespace AgentMemory.Policy;

/// <summary>Outcome assessment for save and retrieval decisions (A4.3).</summary>
public static class OutcomeAssessor
{
    /// <summary>
    /// Assess whether saved memories turned out to be useful.
    /// A memory is considered useful if it was retrieved at least once.
    /// Returns the number of decisions assessed.
    /// </summary>
    public static async Task<int> AssessSaveOutcomesAsync(
        SqliteStore sqlite,
        MemoryConfig? config = null,
        int? lookbackDays = null,
        ILogger? logger = null)
    {
        config ??= new MemoryConfig();
        lookbackDays ??= config.SaveOutcomeLookbackDays;
        var now = DateTime.UtcNow.ToString("o");

        var unassessed = await sqlite.GetUnassessedSaveDecisionsAsync(lookbackDays.Value);
        var updated = 0;

        foreach (var row in unassessed)
        {
            var accessCount = row.TryGetValue("access_count", out var ac) && ac is int count ? count : 0;
            var useful = accessCount > 0;
            var id = row["id"]?.ToString();
            if (id is not null)
            {
                await sqlite.UpdateSaveOutcomeAsync(id, useful, now);
                updated++;
            }
        }

        logger?.LogInformation("Save outcome assessment: assessed {Count} decisions", updated);
        return updated;
    }

    /// <summary>
    /// Assess whether retrievals were helpful.
    /// Heuristic: if the agent did not re-query the same topic within N turns,
    /// the retrieval was probably adequate. Returns the number assessed.
    /// </summary>
    public static async Task<int> AssessRetrievalOutcomesAsync(
        SqliteStore sqlite,
        MemoryConfig? config = null,
        int? followupTurns = null,
        double? keywordOverlapThreshold = null,
        ILogger? logger = null)
    {
        config ??= new MemoryConfig();
        followupTurns ??= config.RetrievalOutcomeFollowupTurns;
        var overlapThreshold = keywordOverlapThreshold ?? config.RetrievalOutcomeKeywordOverlap;
        var now = DateTime.UtcNow.ToString("o");

        var unassessed = await sqlite.GetUnassessedRetrievalDecisionsAsync();
        var updated = 0;

        foreach (var row in unassessed)
        {
            if (!row.TryGetValue("turn", out var turnVal) || turnVal is not int turn)
                continue;

            var sessionId = row["session_id"]?.ToString() ?? "";
            var query = row["query"]?.ToString() ?? "";

            var followups = await sqlite.GetRetrievalFollowupsAsync(
                sessionId, turn, window: followupTurns.Value);

            var originalKeywords = query.Split(' ', StringSplitOptions.RemoveEmptyEntries)
                .Where(w => w.Length > 2)
                .Select(w => w.ToLowerInvariant().Trim())
                .ToHashSet();

            var reQueried = false;

            foreach (var fuQuery in followups)
            {
                var fuKeywords = fuQuery.Split(' ', StringSplitOptions.RemoveEmptyEntries)
                    .Where(w => w.Length > 2)
                    .Select(w => w.ToLowerInvariant().Trim())
                    .ToHashSet();

                if (originalKeywords.Count > 0 && fuKeywords.Count > 0)
                {
                    var overlap = (double)originalKeywords.Intersect(fuKeywords).Count() /
                                  Math.Max(originalKeywords.Count, 1);
                    if (overlap > overlapThreshold)
                    {
                        reQueried = true;
                        break;
                    }
                }
            }

            var helpful = !reQueried;
            var id = row["id"]?.ToString();
            if (id is not null)
            {
                await sqlite.UpdateRetrievalOutcomeAsync(id, helpful, now);
                updated++;
            }
        }

        logger?.LogInformation("Retrieval outcome assessment: assessed {Count} decisions", updated);
        return updated;
    }
}
