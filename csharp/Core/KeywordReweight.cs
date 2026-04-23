// Graph-informed keyword reweighting (A2.5).
//
// Keywords that appear in memories connected by RELATES_TO edges are
// structurally more important. Runs during compaction to boost weights
// for keywords shared across well-connected memory clusters.

using AgentMemory.Storage;
using Microsoft.Extensions.Logging;

namespace AgentMemory.Core;

/// <summary>Graph-informed keyword reweighting (A2.5).</summary>
public static class KeywordReweight
{
    /// <summary>
    /// Adjust keyword weights based on graph connectivity.
    /// Keywords shared across well-connected memories get boosted.
    /// Returns the number of keyword weights updated.
    /// </summary>
    public static async Task<int> ReweightKeywordsFromGraphAsync(
        SqliteStore sqlite,
        GraphStore graph,
        int maxHops = 2,
        int maxMemoriesPerKeyword = 50,
        ILogger? logger = null)
    {
        var rows = await sqlite.GetAllKeywordsWithMemoriesAsync();
        if (rows.Count == 0)
            return 0;

        // Group by keyword
        var keywordIndex = new Dictionary<string, List<Dictionary<string, object>>>();
        foreach (var row in rows)
        {
            var keyword = (string)row["keyword"];
            if (!keywordIndex.TryGetValue(keyword, out var list))
            {
                list = [];
                keywordIndex[keyword] = list;
            }
            list.Add(row);
        }

        var pendingUpdates = new List<(double Weight, string MemoryId, string Keyword)>();

        foreach (var (keyword, entries) in keywordIndex)
        {
            if (entries.Count < 2)
                continue;

            // Cap memories per keyword to avoid O(n²) graph queries.
            var sortedEntries = entries
                .OrderByDescending(e => (double)e["weight"])
                .ThenBy(e => (string)e["memory_id"])
                .Take(maxMemoriesPerKeyword)
                .ToList();

            var memoryIds = sortedEntries.Select(e => (string)e["memory_id"]).ToList();

            // Check graph connectivity between memories sharing this keyword
            var connectedPairs = 0;
            var totalPairs = 0;
            for (var i = 0; i < memoryIds.Count; i++)
            {
                for (var j = i + 1; j < memoryIds.Count; j++)
                {
                    totalPairs++;
                    if (graph.PathExists(memoryIds[i], memoryIds[j], maxHops: maxHops))
                        connectedPairs++;
                }
            }

            if (totalPairs == 0)
                continue;

            var connectivityRatio = (double)connectedPairs / totalPairs;

            if (connectivityRatio <= 0.0)
                continue;

            // Boost: 0.0 connectivity = no change, 1.0 = +50% weight
            var boost = 1.0 + 0.5 * connectivityRatio;

            foreach (var entry in sortedEntries)
            {
                var oldWeight = (double)entry["weight"];
                var newWeight = Math.Min(oldWeight * boost, 1.0);
                if (Math.Abs(newWeight - oldWeight) > 1e-10)
                    pendingUpdates.Add((newWeight, (string)entry["memory_id"], keyword));
            }
        }

        if (pendingUpdates.Count > 0)
            await sqlite.BatchUpdateKeywordWeightsAsync(pendingUpdates);

        logger?.LogInformation("Keyword reweighting: updated {Count} weights", pendingUpdates.Count);
        return pendingUpdates.Count;
    }
}
