// LLM client using HttpClient to call OpenAI-compatible HTTP APIs.
// Includes retry logic with exponential backoff.

using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace AgentMemory.Llm;

/// <summary>
/// LLM client using HttpClient to call OpenAI-compatible chat completion APIs.
/// </summary>
public sealed class LlmClient : ILlmClient
{
    private readonly HttpClient _client;
    private readonly string _baseUrl;
    private readonly MemoryConfig _config;
    private readonly ILogger<LlmClient>? _logger;

    public LlmClient(
        string? baseUrl = null,
        MemoryConfig? config = null,
        HttpClient? client = null,
        ILogger<LlmClient>? logger = null)
    {
        _baseUrl = (baseUrl ?? "https://api.openai.com").TrimEnd('/');
        _config = config ?? new MemoryConfig();
        _client = client ?? new HttpClient();
        _logger = logger;
    }

    public async Task<string> CompleteAsync(
        string prompt,
        string? system = null,
        string? model = null,
        double? temperature = null,
        int maxRetries = 3)
    {
        model ??= _config.LlmModel;
        temperature ??= _config.LlmTemperature;

        var messages = new List<object>();
        if (system is not null)
            messages.Add(new { role = "system", content = system });
        messages.Add(new { role = "user", content = prompt });

        for (var attempt = 0; attempt < maxRetries; attempt++)
        {
            try
            {
                var body = JsonSerializer.Serialize(new
                {
                    model,
                    messages,
                    temperature,
                });

                var content = new StringContent(body, System.Text.Encoding.UTF8, "application/json");
                var resp = await _client.PostAsync($"{_baseUrl}/v1/chat/completions", content);
                resp.EnsureSuccessStatusCode();

                var json = await resp.Content.ReadAsStringAsync();
                using var doc = JsonDocument.Parse(json);

                return doc.RootElement
                    .GetProperty("choices")[0]
                    .GetProperty("message")
                    .GetProperty("content")
                    .GetString() ?? "";
            }
            catch (Exception ex)
            {
                if (attempt == maxRetries - 1)
                    throw;

                _logger?.LogWarning(
                    "LLM call failed (attempt {Attempt}/{Max}), retrying...",
                    attempt + 1, maxRetries);

                // Exponential backoff
                await Task.Delay(TimeSpan.FromMilliseconds(Math.Pow(2, attempt) * 500));
            }
        }

        return ""; // unreachable but satisfies type checker
    }

    public async Task<Dictionary<string, object?>> CompleteJsonAsync(
        string prompt,
        string? system = null,
        string? model = null,
        double? temperature = null)
    {
        var text = await CompleteAsync(prompt, system, model, temperature);

        // Strip markdown code fences if present
        var cleaned = text.Trim();
        if (cleaned.StartsWith("```"))
        {
            var lines = cleaned.Split('\n');
            var filteredLines = lines
                .Skip(1)
                .Where(l => !l.Trim().StartsWith("```"))
                .ToArray();
            cleaned = string.Join('\n', filteredLines);
        }

        using var doc = JsonDocument.Parse(cleaned);
        return DeserializeElement(doc.RootElement);
    }

    private static Dictionary<string, object?> DeserializeElement(JsonElement element)
    {
        var dict = new Dictionary<string, object?>();

        foreach (var prop in element.EnumerateObject())
        {
            dict[prop.Name] = prop.Value.ValueKind switch
            {
                JsonValueKind.String => prop.Value.GetString(),
                JsonValueKind.Number => prop.Value.GetDouble(),
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                JsonValueKind.Null => null,
                JsonValueKind.Array => DeserializeArray(prop.Value),
                JsonValueKind.Object => DeserializeElement(prop.Value),
                _ => prop.Value.GetRawText(),
            };
        }

        return dict;
    }

    private static List<object?> DeserializeArray(JsonElement element)
    {
        var list = new List<object?>();
        foreach (var item in element.EnumerateArray())
        {
            list.Add(item.ValueKind switch
            {
                JsonValueKind.String => item.GetString(),
                JsonValueKind.Number => item.GetDouble(),
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                JsonValueKind.Null => null,
                JsonValueKind.Array => DeserializeArray(item),
                JsonValueKind.Object => DeserializeElement(item),
                _ => item.GetRawText(),
            });
        }
        return list;
    }
}
