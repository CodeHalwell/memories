// Compaction scheduler and merge logic.
//
// Compaction runs between sessions (the "sleep cycle"). Collapses episodic
// detail into semantic generalisations with intentional forgetting.
//
// A2.3: Generation gap guard — only merge memories within 1 generation.
// A2.4: Merge validation via generative replay.

using System.Text.Json;
using AgentMemory.Embeddings;
using AgentMemory.Llm;
using AgentMemory.Storage;
using Microsoft.Extensions.Logging;

namespace AgentMemory.Core;

/// <summary>Compaction engine — merges low-value memories into generalisations.</summary>
public sealed class CompactionEngine
{
    private readonly SqliteStore _sqlite;
    private readonly GraphStore _graph;
    private readonly VectorStore _vector;
    private readonly ITextEmbedder? _textEmbedder;
    private readonly ILlmClient _llmClient;
    private readonly MemoryConfig _config;
    private readonly ILogger<CompactionEngine>? _logger;

    public CompactionEngine(
        SqliteStore sqlite,
        GraphStore graph,
        VectorStore vector,
        ILlmClient llmClient,
        MemoryConfig config,
        ITextEmbedder? textEmbedder = null,
        ILogger<CompactionEngine>? logger = null)
    {
        _sqlite = sqlite;
        _graph = graph;
        _vector = vector;
        _llmClient = llmClient;
        _config = config;
        _textEmbedder = textEmbedder;
        _logger = logger;
    }

    /// <summary>Score a memory for compaction candidacy. Low decay + low salience = good candidate.</summary>
    public static double CompactionScore(Memory memory)
        => (1.0 - memory.DecayScore) * 0.6 + (1.0 - memory.Salience) * 0.4;

    /// <summary>Compute keyword overlap ratio (Jaccard) between two memories.</summary>
    public static double KeywordOverlap(Memory memA, Memory memB)
    {
        var kwA = new HashSet<string>(memA.Keywords.Select(k => k.Keyword));
        var kwB = new HashSet<string>(memB.Keywords.Select(k => k.Keyword));
        if (kwA.Count == 0 || kwB.Count == 0)
            return 0.0;

        var intersection = kwA.Intersect(kwB).Count();
        var union = kwA.Union(kwB).Count();
        return union > 0 ? (double)intersection / union : 0.0;
    }

    /// <summary>Check if a group of memories can be merged (no exclusion conditions).</summary>
    public static bool CanMerge(List<Memory> group, MemoryConfig config)
    {
        for (var i = 0; i < group.Count; i++)
        {
            for (var j = i + 1; j < group.Count; j++)
            {
                var a = group[i];
                var b = group[j];

                // Opposite valence exclusion
                if (a.Valence * b.Valence < 0 &&
                    Math.Abs(a.Valence - b.Valence) > config.ValenceMergeExclusionDelta)
                    return false;

                // Either is fast_pathed gen-0
                if ((a.FastPathed && a.CompactionGen == 0) ||
                    (b.FastPathed && b.CompactionGen == 0))
                    return false;

                // A2.3: Generation gap guard
                if (Math.Abs(a.CompactionGen - b.CompactionGen) > config.MaxGenerationGapForMerge)
                    return false;
            }
        }
        return true;
    }

    /// <summary>Group memories by keyword overlap using greedy clustering.</summary>
    public static List<List<Memory>> GroupByKeywords(List<Memory> candidates, double threshold)
    {
        if (candidates.Count == 0)
            return [];

        var used = new HashSet<int>();
        var groups = new List<List<Memory>>();

        for (var i = 0; i < candidates.Count; i++)
        {
            if (used.Contains(i))
                continue;

            var group = new List<Memory> { candidates[i] };
            used.Add(i);

            for (var j = 0; j < candidates.Count; j++)
            {
                if (used.Contains(j))
                    continue;

                var overlaps = group.Select(g => KeywordOverlap(g, candidates[j])).ToList();
                if (overlaps.Count > 0 && overlaps.Min() >= threshold)
                {
                    group.Add(candidates[j]);
                    used.Add(j);
                }
            }

            if (group.Count > 1)
                groups.Add(group);
        }

        return groups;
    }

    /// <summary>Compute cosine similarity between two vectors.</summary>
    public static double CosineSimilarity(List<double> a, List<double> b)
    {
        double dot = 0, normA = 0, normB = 0;
        var len = Math.Min(a.Count, b.Count);
        for (var i = 0; i < len; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        var norm = Math.Sqrt(normA) * Math.Sqrt(normB);
        return norm > 0 ? dot / norm : 0.0;
    }

    /// <summary>
    /// Validate a merge via generative replay (A2.4).
    /// Generate synthetic queries from source memories, then test whether the
    /// candidate merge still retrieves well for those queries.
    /// </summary>
    public async Task<MergeValidation> ValidateMergeAsync(
        List<Memory> sourceMemories,
        string candidateContent,
        int nQueries = 5,
        double degradationTolerance = 0.15)
    {
        if (_textEmbedder is null)
            return new MergeValidation { Passed = true };

        var sourceText = string.Join("\n\n", sourceMemories.Select(m => m.Content));
        var prompt = $"Generate {nQueries} search queries for this content:\n\n<source_text>\n{sourceText}\n</source_text>";

        List<string> queries;
        try
        {
            var result = await _llmClient.CompleteJsonAsync(prompt, system: _config.Prompts.SyntheticQuery);

            // The result might be a list or dict
            if (result.Values.FirstOrDefault() is List<object?> listVal)
                queries = listVal.Where(v => v is not null).Select(v => v!.ToString()!).ToList();
            else
                queries = result.Values.Where(v => v is not null).Select(v => v!.ToString()!).ToList();
        }
        catch
        {
            _logger?.LogWarning("Synthetic query generation failed, failing validation to be safe");
            return new MergeValidation { Passed = false, QueriesTested = [] };
        }

        if (queries.Count == 0)
            return new MergeValidation { Passed = true, QueriesTested = [] };

        // Embed candidate and sources
        var candidateEmb = await _textEmbedder.EmbedAsync(candidateContent);
        var sourceEmbs = new List<List<double>>();
        foreach (var m in sourceMemories)
            sourceEmbs.Add(await _textEmbedder.EmbedAsync(m.Content));

        var mergedScores = new List<double>();
        var sourceScores = new List<double>();

        foreach (var query in queries)
        {
            var queryEmb = await _textEmbedder.EmbedAsync(query);
            mergedScores.Add(CosineSimilarity(queryEmb, candidateEmb));
            var bestSource = sourceEmbs.Max(se => CosineSimilarity(queryEmb, se));
            sourceScores.Add(bestSource);
        }

        var avgMerged = mergedScores.Count > 0 ? mergedScores.Average() : 0.0;
        var avgSource = sourceScores.Count > 0 ? sourceScores.Average() : 0.0;
        var degradation = avgSource - avgMerged;

        return new MergeValidation
        {
            Passed = degradation < degradationTolerance,
            AvgSourceScore = avgSource,
            AvgMergedScore = avgMerged,
            Degradation = degradation,
            QueriesTested = queries,
        };
    }

    /// <summary>Execute a full compaction cycle.</summary>
    public async Task<CompactionResult> RunAsync(string trigger = "scheduled")
    {
        var result = new CompactionResult { Trigger = trigger };

        // Get candidates from SQLite
        var candidates = await _sqlite.GetCompactionCandidatesAsync(
            threshold: _config.CompactionCandidateThreshold);
        result.MemoriesReviewed = candidates.Count;

        if (candidates.Count == 0)
        {
            _logger?.LogInformation("Compaction: no candidates found");
            await _sqlite.LogCompactionRunAsync(result);
            return result;
        }

        // Filter by graph edge count (structurally important anchors)
        var filtered = new List<Memory>();
        foreach (var mem in candidates)
        {
            if (mem.GraphNodeId is not null)
            {
                var edgeCount = _graph.GetEdgeCount(mem.Id);
                if (edgeCount > 3)
                    continue;
            }
            filtered.Add(mem);
        }

        // Group by keyword overlap
        var groups = GroupByKeywords(filtered, _config.KeywordOverlapMergeThreshold);

        var mergedCount = 0;
        foreach (var group in groups)
        {
            if (!CanMerge(group, _config))
                continue;

            var newMem = await MergeGroupAsync(group, result.Id);
            if (newMem is not null)
                mergedCount++;
        }

        result.MemoriesMerged = mergedCount;

        // Tier promotion: move hot memories exceeding threshold to warm
        var hotCount = await _sqlite.CountMemoriesAsync(tier: "hot");
        if (hotCount > _config.HotTierThreshold)
            await PromoteTierAsync(hotCount - _config.HotTierThreshold);

        result.Notes = $"Reviewed {result.MemoriesReviewed}, merged into {mergedCount} semantic memories";
        await _sqlite.LogCompactionRunAsync(result);

        _logger?.LogInformation(
            "Compaction complete: reviewed={Reviewed}, merged={Merged}",
            result.MemoriesReviewed, mergedCount);

        return result;
    }

    private async Task<Memory?> MergeGroupAsync(List<Memory> group, string compactionId)
    {
        // Build prompt with all source memories
        var sources = string.Join("\n\n", group.Select((m, i) =>
            $"Memory {i + 1} (salience={m.Salience}, valence={m.Valence}):\n<memory_{i + 1}>\n{m.Content}\n</memory_{i + 1}>"));

        var prompt = $"""
            Merge these {group.Count} related memories into a single generalised memory:

            {sources}

            Respond with JSON only.
            """;

        Dictionary<string, object?> result;
        try
        {
            result = await _llmClient.CompleteJsonAsync(prompt, system: _config.Prompts.Merge);
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "LLM merge failed for group of {Count} memories", group.Count);
            return null;
        }

        var mergedContent = GetString(result, "content", "");

        // A2.4: Validate merge via generative replay
        MergeValidation? validation = null;
        if (_textEmbedder is not null)
        {
            validation = await ValidateMergeAsync(
                sourceMemories: group,
                candidateContent: mergedContent,
                nQueries: _config.MergeValidationQueries,
                degradationTolerance: _config.MergeDegradationTolerance);

            if (!validation.Passed)
            {
                _logger?.LogInformation(
                    "Merge validation failed (degradation={Degradation:F3}), skipping group of {Count} memories",
                    validation.Degradation, group.Count);
                return null;
            }
        }

        var now = DateTime.UtcNow.ToString("o");
        var newId = Guid.NewGuid().ToString();
        var maxGen = group.Max(m => m.CompactionGen);

        var keywords = ExtractKeywords(result, _config.MaxKeywordsPerMemory);

        var newMem = new Memory
        {
            Id = newId,
            CreatedAt = now,
            UpdatedAt = now,
            Content = mergedContent,
            Summary = GetString(result, "summary", null),
            RawLogId = group[0].RawLogId,
            SessionId = group[0].SessionId,
            Turn = group[0].Turn,
            Valence = GetDouble(result, "valence", 0.0),
            Arousal = GetDouble(result, "arousal", 0.0),
            Salience = GetDouble(result, "salience", 0.5),
            CompactionGen = maxGen + 1,
            Tier = "warm",
            IsSemantic = true,
            Keywords = keywords,
        };

        // Save to SQLite
        await _sqlite.SaveMemoryAsync(newMem);

        // Create graph node
        _graph.AddMemoryNode(
            memoryId: newId,
            summary: newMem.Summary ?? "",
            tier: "warm",
            salience: newMem.Salience,
            valence: newMem.Valence,
            compactionGen: newMem.CompactionGen,
            createdAt: now);
        await _sqlite.UpdateMemoryGraphRefAsync(newId, newId);

        // Create EVOLVED_FROM edges and replicate RELATES_TO edges
        var sourceIds = group.Select(m => m.Id).ToList();
        foreach (var src in group)
            _graph.AddEvolvedFrom(newId, src.Id, compactionId: compactionId, createdAt: now);

        _graph.ReplicateEdgesToNewNode(sourceIds, newId);

        // Move source memories to cold tier
        foreach (var src in group)
        {
            await _sqlite.UpdateMemoryTierAsync(src.Id, "cold");
            if (src.GraphNodeId is not null)
                _graph.UpdateMemoryTier(src.Id, "cold");
        }

        // Create text embedding for new memory
        if (_textEmbedder is not null)
        {
            var vector = await _textEmbedder.EmbedAsync(newMem.Content);
            var pointId = await _vector.UpsertTextVectorAsync(
                memoryId: newId, vector: vector, tier: "warm",
                valence: newMem.Valence, arousal: newMem.Arousal,
                sessionId: newMem.SessionId, createdAt: now);
            await _sqlite.UpdateMemoryVectorRefAsync(newId, pointId);
        }

        // Log the merge with validation data
        await _sqlite.LogCompactionMergeAsync(
            compactionId, sourceIds, newId,
            validationPassed: validation?.Passed,
            avgSourceScore: validation?.AvgSourceScore,
            avgMergedScore: validation?.AvgMergedScore,
            degradation: validation?.Degradation);

        return newMem;
    }

    private async Task PromoteTierAsync(int count)
    {
        var memories = await _sqlite.ListMemoriesAsync(tier: "hot", limit: count);
        // Sort by decay ascending to promote lowest-decay first
        var sorted = memories.OrderBy(m => m.DecayScore).ToList();

        foreach (var mem in sorted.Take(count))
        {
            await _sqlite.UpdateMemoryTierAsync(mem.Id, "warm");
            if (mem.GraphNodeId is not null)
                _graph.UpdateMemoryTier(mem.Id, "warm");
        }
    }

    private static List<(string Keyword, double Weight)> ExtractKeywords(
        Dictionary<string, object?> result, int maxKeywords)
    {
        var keywords = new List<(string, double)>();

        if (!result.TryGetValue("keywords", out var kwVal) || kwVal is not List<object?> kwList)
            return keywords;

        foreach (var item in kwList)
        {
            if (item is Dictionary<string, object?> kwDict)
            {
                var kw = GetString(kwDict, "keyword", "")?.ToLowerInvariant() ?? "";
                var weight = GetDouble(kwDict, "weight", 1.0);
                if (!string.IsNullOrEmpty(kw))
                    keywords.Add((kw, weight));
            }

            if (keywords.Count >= maxKeywords)
                break;
        }

        return keywords;
    }

    private static double GetDouble(Dictionary<string, object?> dict, string key, double defaultValue)
    {
        if (dict.TryGetValue(key, out var val) && val is not null)
        {
            if (val is double d) return d;
            if (val is JsonElement je && je.ValueKind == JsonValueKind.Number) return je.GetDouble();
            if (double.TryParse(val.ToString(), out var parsed)) return parsed;
        }
        return defaultValue;
    }

    private static string? GetString(Dictionary<string, object?> dict, string key, string? defaultValue)
    {
        if (dict.TryGetValue(key, out var val) && val is not null)
        {
            if (val is string s) return s;
            if (val is JsonElement je && je.ValueKind == JsonValueKind.String) return je.GetString();
            return val.ToString();
        }
        return defaultValue;
    }
}
