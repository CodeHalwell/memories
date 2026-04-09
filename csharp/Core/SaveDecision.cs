// LLM-driven save decision and keyword extraction.
//
// A2.1: the save decision is informed by retrieval gaps — if recent queries
// failed to find relevant memories, the save threshold is lowered.

using System.Text.Json;
using AgentMemory.Llm;
using AgentMemory.Storage;
using Microsoft.Extensions.Logging;

namespace AgentMemory.Core;

/// <summary>LLM-driven save decision with gap awareness (A2.1).</summary>
public static class SaveDecisionEngine
{
    private static readonly string[] FastPathPhrases =
        ["remember this", "don't forget", "save this", "keep in mind"];

    /// <summary>Check if a memory should bypass the LLM save decision.</summary>
    public static bool IsFastPath(double arousal, double surprise, string content, MemoryConfig config)
    {
        if (arousal > config.FastPathArousal && surprise > config.FastPathSurprise)
            return true;

        var lower = content.ToLowerInvariant();
        return FastPathPhrases.Any(phrase => lower.Contains(phrase));
    }

    /// <summary>
    /// Identify topic areas where recent retrievals returned poor results (A2.1).
    /// </summary>
    public static async Task<List<string>> GetRetrievalGapsAsync(
        SqliteStore sqlite, string sessionId, MemoryConfig config)
    {
        return await sqlite.GetFailedRetrievalKeywordsAsync(
            sessionId, lookback: config.GapLookbackTurns);
    }

    /// <summary>Compute overlap between content keywords and retrieval gap keywords (A2.1).</summary>
    public static double ComputeGapOverlap(List<string> contentKeywords, List<string> gapKeywords)
    {
        if (contentKeywords.Count == 0 || gapKeywords.Count == 0)
            return 0.0;

        var contentSet = new HashSet<string>(contentKeywords);
        var gapSet = new HashSet<string>(gapKeywords);
        var intersectionSize = contentSet.Intersect(gapSet).Count();
        return (double)intersectionSize / Math.Max(contentSet.Count, 1);
    }

    /// <summary>
    /// Decide whether to save an agent output as a memory.
    /// Returns (SaveDecision, Memory or null).
    /// </summary>
    public static async Task<(SaveDecision Decision, Memory? Memory)> MakeSaveDecisionAsync(
        RawLogEntry entry,
        ILlmClient llmClient,
        MemoryConfig config,
        bool isFirstTurn = false,
        SqliteStore? sqlite = null,
        ILogger? logger = null)
    {
        // First turn of a session is always saved via fast path
        if (isFirstTurn)
        {
            var mem = new Memory
            {
                Content = entry.Content,
                RawLogId = entry.Id,
                SessionId = entry.SessionId,
                Turn = entry.Turn,
                Salience = 0.7,
                FastPathed = true,
            };
            var dec = new SaveDecision
            {
                RawLogId = entry.Id,
                SessionId = entry.SessionId,
                Turn = entry.Turn,
                Decision = "fast_path",
                Reason = "First turn of session — always saved",
                Confidence = 1.0,
            };
            return (dec, mem);
        }

        // Ask LLM for structured evaluation
        var prompt = $"""
            Evaluate whether this agent output should be saved as a memory:

            Session: {entry.SessionId}
            Turn: {entry.Turn}
            Content:
            <content>
            {entry.Content}
            </content>

            Respond with JSON only.
            """;

        Dictionary<string, object?> result;
        try
        {
            result = await llmClient.CompleteJsonAsync(prompt, system: config.Prompts.SaveDecision);
        }
        catch (Exception ex)
        {
            logger?.LogError(ex, "LLM save decision failed, defaulting to skip");
            var skipDec = new SaveDecision
            {
                RawLogId = entry.Id,
                SessionId = entry.SessionId,
                Turn = entry.Turn,
                Decision = "skip",
                Reason = "LLM evaluation failed",
                Confidence = 0.0,
            };
            return (skipDec, null);
        }

        var confidence = GetDouble(result, "confidence", 0.0);
        var shouldSave = GetBool(result, "should_save", false);
        var valence = GetDouble(result, "valence", 0.0);
        var arousal = GetDouble(result, "arousal", 0.0);
        var surprise = GetDouble(result, "surprise", 0.0);
        var salience = GetDouble(result, "salience", 0.5);

        // Extract keywords
        var keywords = ExtractKeywords(result, config.MaxKeywordsPerMemory);
        var contentKwNames = keywords.Select(kw => kw.Keyword).ToList();

        // A2.1: Retrieval gap awareness
        var threshold = config.SaveConfidenceThreshold;
        var gapTriggered = false;

        if (sqlite is not null)
        {
            try
            {
                var gapKeywords = await GetRetrievalGapsAsync(sqlite, entry.SessionId, config);
                var gapOverlap = ComputeGapOverlap(contentKwNames, gapKeywords);
                if (gapOverlap > config.GapOverlapThreshold)
                {
                    threshold *= config.GapThresholdReduction;
                    gapTriggered = true;
                }
            }
            catch
            {
                logger?.LogDebug("Gap detection failed, using default threshold");
            }
        }

        // Check fast path conditions
        var fastPath = IsFastPath(arousal, surprise, entry.Content, config);

        string decision;
        if (fastPath)
        {
            decision = "fast_path";
            shouldSave = true;
            confidence = Math.Max(confidence, 0.9);
        }
        else if (shouldSave && confidence >= threshold)
        {
            decision = "save";
        }
        else
        {
            decision = "skip";
        }

        var saveDec = new SaveDecision
        {
            RawLogId = entry.Id,
            SessionId = entry.SessionId,
            Turn = entry.Turn,
            Decision = decision,
            Reason = GetString(result, "reason", ""),
            Confidence = confidence,
            GapTriggered = gapTriggered,
            ThresholdUsed = threshold,
        };

        if (decision is "save" or "fast_path")
        {
            var memory = new Memory
            {
                Content = entry.Content,
                Summary = GetString(result, "summary", null),
                RawLogId = entry.Id,
                SessionId = entry.SessionId,
                Turn = entry.Turn,
                Valence = valence,
                Arousal = arousal,
                Surprise = surprise,
                Salience = salience,
                FastPathed = fastPath,
                Keywords = keywords,
            };
            return (saveDec, memory);
        }

        return (saveDec, null);
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
                var kw = GetString(kwDict, "keyword", "").ToLowerInvariant();
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

    private static bool GetBool(Dictionary<string, object?> dict, string key, bool defaultValue)
    {
        if (dict.TryGetValue(key, out var val) && val is not null)
        {
            if (val is bool b) return b;
            if (val is JsonElement je && je.ValueKind is JsonValueKind.True or JsonValueKind.False)
                return je.GetBoolean();
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
