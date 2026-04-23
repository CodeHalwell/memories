// Emotional scoring via LLM.
// Scores the current context for valence and arousal to support mood-congruent retrieval.

using AgentMemory.Llm;
using Microsoft.Extensions.Logging;

namespace AgentMemory.Emotion;

/// <summary>Emotional dimension scorer using LLM.</summary>
public static class Scorer
{
    /// <summary>
    /// Score the emotional dimensions of a text.
    /// Returns a dictionary with keys: valence, arousal, surprise.
    /// </summary>
    public static async Task<Dictionary<string, double>> ScoreEmotionAsync(
        string text, ILlmClient llmClient, MemoryConfig config,
        ILogger? logger = null)
    {
        var prompt = $"Score the emotional tone of this text:\n\n<text>\n{text}\n</text>";

        try
        {
            var result = await llmClient.CompleteJsonAsync(prompt, system: config.Prompts.Emotion);
            return new Dictionary<string, double>
            {
                ["valence"] = Clamp(GetDouble(result, "valence", 0.0), -1.0, 1.0),
                ["arousal"] = Clamp(GetDouble(result, "arousal", 0.0), 0.0, 1.0),
                ["surprise"] = Clamp(GetDouble(result, "surprise", 0.0), 0.0, 1.0),
            };
        }
        catch (Exception ex)
        {
            logger?.LogError(ex, "Emotion scoring failed, returning neutral");
            return new Dictionary<string, double>
            {
                ["valence"] = 0.0,
                ["arousal"] = 0.0,
                ["surprise"] = 0.0,
            };
        }
    }

    public static double Clamp(double value, double lo, double hi)
        => Math.Max(lo, Math.Min(hi, value));

    private static double GetDouble(Dictionary<string, object?> dict, string key, double defaultValue)
    {
        if (dict.TryGetValue(key, out var val) && val is not null)
        {
            if (val is double d) return d;
            if (val is System.Text.Json.JsonElement je) return je.GetDouble();
            if (double.TryParse(val.ToString(), out var parsed)) return parsed;
        }
        return defaultValue;
    }
}
