// Three-layer retrieval stack with mood-congruent weighting.
//
// Layers:
//   1. Grep — raw log search via process
//   2. Keyword — SQLite keyword search weighted by decay score
//   3. Semantic — vector similarity search
//
// Results are merged, deduplicated, and re-ranked. Graph traversal expands
// along RELATES_TO edges. Visual channel provides independent retrieval via CLIP.

using System.Diagnostics;
using System.Text.Json;
using AgentMemory.Embeddings;
using AgentMemory.Emotion;
using AgentMemory.Llm;
using AgentMemory.Storage;
using Microsoft.Extensions.Logging;

namespace AgentMemory.Core;

/// <summary>Orchestrates multi-layer retrieval across all storage backends.</summary>
public sealed class RetrievalEngine
{
    private readonly SqliteStore _sqlite;
    private readonly GraphStore _graph;
    private readonly VectorStore _vector;
    private readonly ITextEmbedder _textEmbedder;
    private readonly IVisualEmbedder? _visualEmbedder;
    private readonly ILlmClient _llmClient;
    private readonly MemoryConfig _config;
    private readonly ILogger<RetrievalEngine>? _logger;

    public RetrievalEngine(
        SqliteStore sqlite,
        GraphStore graph,
        VectorStore vector,
        ITextEmbedder textEmbedder,
        ILlmClient llmClient,
        MemoryConfig config,
        IVisualEmbedder? visualEmbedder = null,
        ILogger<RetrievalEngine>? logger = null)
    {
        _sqlite = sqlite;
        _graph = graph;
        _vector = vector;
        _textEmbedder = textEmbedder;
        _visualEmbedder = visualEmbedder;
        _llmClient = llmClient;
        _config = config;
        _logger = logger;
    }

    /// <summary>Run all retrieval layers and return ranked, deduplicated memories.</summary>
    public async Task<List<Memory>> RetrieveAsync(
        string query,
        string? sessionId = null,
        int? topK = null,
        bool enableMoodCongruent = true,
        bool enableVisual = true)
    {
        topK ??= _config.TopKPerLayer;

        // Score current context for mood-congruent weighting
        Dictionary<string, double>? contextEmotion = null;
        if (enableMoodCongruent)
        {
            try
            {
                contextEmotion = await Scorer.ScoreEmotionAsync(query, _llmClient, _config);
            }
            catch
            {
                _logger?.LogDebug("Mood scoring failed, proceeding without");
            }
        }

        // Run layers concurrently, tolerating individual layer failures so a
        // degraded backend (e.g. vector store down) doesn't fail the whole call.
        var grepTask = RunLayerAsync("grep", () => GrepLayerAsync(query, topK.Value));
        var keywordTask = RunLayerAsync("keyword", () => KeywordLayerAsync(query, topK.Value));
        var semanticTask = RunLayerAsync("semantic", () => SemanticLayerAsync(query, topK.Value));

        var tasks = new List<Task<List<(string MemoryId, double Score)>>>
        {
            grepTask, keywordTask, semanticTask
        };

        if (enableVisual && _visualEmbedder is not null)
        {
            tasks.Add(RunLayerAsync("visual", () => VisualLayerAsync(query, topK.Value)));
        }

        var results = await Task.WhenAll(tasks);

        // Merge all candidates
        var candidates = new Dictionary<string, Candidate>();
        var layerNames = new[] { "grep", "keyword", "semantic", "visual" };

        for (var i = 0; i < results.Length; i++)
        {
            var layerName = layerNames[i];
            foreach (var (memId, score) in results[i])
            {
                if (candidates.TryGetValue(memId, out var existing))
                {
                    existing.Score += score;
                    existing.Layers.Add(layerName);
                }
                else
                {
                    candidates[memId] = new Candidate
                    {
                        MemoryId = memId,
                        Score = score,
                        Layers = new HashSet<string> { layerName },
                    };
                }
            }
        }

        // Graph traversal expansion
        var graphExpanded = new HashSet<string>();
        foreach (var memId in candidates.Keys.ToList())
        {
            var related = _graph.GetRelatedMemories(memId, maxDepth: _config.GraphTraversalDepth);
            foreach (var rel in related)
            {
                var rid = (string)rel["id"];
                if (!candidates.ContainsKey(rid) && !graphExpanded.Contains(rid))
                {
                    graphExpanded.Add(rid);
                    var depth = Convert.ToInt32(rel["depth"]);
                    var relSalience = rel.TryGetValue("salience", out var s) ? Convert.ToDouble(s) : 0.5;
                    var depthScore = 1.0 / (depth + 1) * relSalience;
                    candidates[rid] = new Candidate
                    {
                        MemoryId = rid,
                        Score = depthScore,
                        Layers = new HashSet<string> { "graph_traversal" },
                    };
                }
            }
        }

        // Load full memories from SQLite
        var memories = new List<(Memory Mem, double Score)>();
        foreach (var cand in candidates.Values)
        {
            var mem = await _sqlite.GetMemoryAsync(cand.MemoryId);
            if (mem is null)
                continue;

            var score = cand.Score;

            // Mood-congruent boosting
            if (contextEmotion is not null && enableMoodCongruent)
            {
                var moodWeight = _config.MoodCongruentWeight;
                var valenceSim = 1.0 - Math.Abs(contextEmotion["valence"] - mem.Valence) / 2.0;
                var arousalSim = 1.0 - Math.Abs(contextEmotion["arousal"] - mem.Arousal);
                var moodBonus = (valenceSim + arousalSim) / 2.0 * moodWeight;
                score += moodBonus;
            }

            // Factor in decay
            score *= mem.DecayScore;

            memories.Add((mem, score));
        }

        // Sort by score descending
        memories.Sort((a, b) => b.Score.CompareTo(a.Score));

        // Log access and update decay for returned memories
        var now = DateTime.UtcNow.ToString("o");
        var resultMemories = new List<Memory>();

        foreach (var (mem, _) in memories.Take(topK.Value * 2))
        {
            var cand = candidates.GetValueOrDefault(mem.Id);
            var accessType = "primary";
            if (cand is not null && cand.Layers.Contains("graph_traversal"))
                accessType = "graph_traversal";
            else if (cand is not null && cand.Layers.Contains("grep") && cand.Layers.Count == 1)
                accessType = "grep_entrypoint";
            else if (cand is not null && cand.Layers.Contains("semantic"))
                accessType = "vector";

            mem.AccessCount += 1;
            mem.LastAccessed = now;
            mem.DecayScore = Decay.ComputeDecay(
                DateTime.Parse(now).ToUniversalTime(),
                mem.AccessCount,
                arousal: mem.Arousal,
                surprise: mem.Surprise,
                isSemantic: mem.IsSemantic,
                config: _config);

            await _sqlite.LogAccessAsync(
                accessId: Guid.NewGuid().ToString(),
                memoryId: mem.Id,
                accessedAt: now,
                accessType: accessType,
                sessionId: sessionId,
                query: query);

            await _sqlite.UpdateMemoryAccessAsync(
                mem.Id, mem.DecayScore, mem.AccessCount, now);

            resultMemories.Add(mem);
        }

        return resultMemories;
    }

    // ── Layer implementations ──

    private async Task<List<(string MemoryId, double Score)>> RunLayerAsync(
        string layerName,
        Func<Task<List<(string MemoryId, double Score)>>> layer)
    {
        try
        {
            return await layer();
        }
        catch (Exception ex)
        {
            _logger?.LogWarning(ex, "Retrieval layer '{Layer}' failed, returning empty results", layerName);
            return [];
        }
    }

    private async Task<List<(string MemoryId, double Score)>> GrepLayerAsync(string query, int limit)
    {
        var logDir = _config.LogDir;
        var terms = query.Split(' ', StringSplitOptions.RemoveEmptyEntries).Take(5);
        var pattern = string.Join("|", terms);

        var results = new List<(string, double)>();

        try
        {
            var psi = new ProcessStartInfo("rg", $"--json -i -e \"{pattern}\" \"{logDir}\"")
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            };

            using var proc = Process.Start(psi);
            if (proc is null)
                return results;

            var stdout = await proc.StandardOutput.ReadToEndAsync();
            await proc.WaitForExitAsync();

            if (string.IsNullOrEmpty(stdout))
                return results;

            var hits = new Dictionary<string, int>();
            foreach (var line in stdout.Split('\n', StringSplitOptions.RemoveEmptyEntries))
            {
                try
                {
                    using var doc = JsonDocument.Parse(line);
                    if (doc.RootElement.TryGetProperty("type", out var typeProp) &&
                        typeProp.GetString() == "match" &&
                        doc.RootElement.TryGetProperty("data", out var data) &&
                        data.TryGetProperty("lines", out var lines) &&
                        lines.TryGetProperty("text", out var text))
                    {
                        try
                        {
                            using var entryDoc = JsonDocument.Parse(text.GetString() ?? "");
                            if (entryDoc.RootElement.TryGetProperty("Id", out var idProp))
                            {
                                var entryId = idProp.GetString() ?? "";
                                if (!string.IsNullOrEmpty(entryId))
                                    hits[entryId] = hits.GetValueOrDefault(entryId, 0) + 1;
                            }
                        }
                        catch { }
                    }
                }
                catch { }
            }

            // Map raw_log_ids to memory_ids
            foreach (var (rawId, count) in hits.OrderByDescending(kv => kv.Value).Take(limit))
            {
                var memId = await _sqlite.FindMemoryIdByRawLogIdAsync(rawId);
                if (memId is not null)
                    results.Add((memId, count));
            }
        }
        catch
        {
            _logger?.LogDebug("ripgrep not found or failed, skipping grep layer");
        }

        return results;
    }

    private async Task<List<(string MemoryId, double Score)>> KeywordLayerAsync(string query, int limit)
    {
        var keywords = query.Split(' ', StringSplitOptions.RemoveEmptyEntries)
            .Where(w => w.Length > 2)
            .Select(w => w.ToLowerInvariant().Trim())
            .ToList();

        if (keywords.Count == 0)
            return [];

        var memories = await _sqlite.SearchByKeywordsAsync(keywords, limit: limit);
        return memories.Select(m => (m.Id, m.DecayScore)).ToList();
    }

    private async Task<List<(string MemoryId, double Score)>> SemanticLayerAsync(string query, int limit)
    {
        var queryVector = await _textEmbedder.EmbedAsync(query);
        var results = await _vector.SearchTextAsync(queryVector, limit: limit);
        return results.Select(r => (r.MemoryId, r.Score)).ToList();
    }

    private async Task<List<(string MemoryId, double Score)>> VisualLayerAsync(string query, int limit)
    {
        if (_visualEmbedder is null)
            return [];

        var queryVector = await _visualEmbedder.EmbedAsync(query);
        var results = await _vector.SearchVisualAsync(queryVector, limit: limit);
        return results.Select(r => (r.MemoryId, r.Score)).ToList();
    }

    private sealed class Candidate
    {
        public string MemoryId { get; init; } = "";
        public double Score { get; set; }
        public HashSet<string> Layers { get; init; } = [];
    }
}
