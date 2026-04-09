// HTTP-based text embedder using an embedding API endpoint.

using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace AgentMemory.Embeddings;

/// <summary>
/// Text embedder that calls an HTTP embedding API endpoint.
/// Compatible with OpenAI-style /v1/embeddings API.
/// </summary>
public sealed class TextEmbedder : ITextEmbedder
{
    private readonly HttpClient _client;
    private readonly string _baseUrl;
    private readonly string _model;
    private readonly ILogger<TextEmbedder>? _logger;
    private int? _dimension;

    public TextEmbedder(
        string? baseUrl = null,
        string? model = null,
        HttpClient? client = null,
        ILogger<TextEmbedder>? logger = null)
    {
        _baseUrl = (baseUrl ?? "http://localhost:11434").TrimEnd('/');
        _model = model ?? new MemoryConfig().TextEmbeddingModel;
        _client = client ?? new HttpClient();
        _logger = logger;
    }

    public int Dimension
    {
        get
        {
            if (_dimension is null)
            {
                // Infer dimension from a dummy embedding
                var task = EmbedAsync("test");
                task.Wait();
                _dimension = task.Result.Count;
            }
            return _dimension.Value;
        }
    }

    public async Task<List<double>> EmbedAsync(string text)
    {
        var body = JsonSerializer.Serialize(new
        {
            input = text,
            model = _model,
        });

        var content = new StringContent(body, System.Text.Encoding.UTF8, "application/json");
        var resp = await _client.PostAsync($"{_baseUrl}/v1/embeddings", content);
        var json = await resp.Content.ReadAsStringAsync();

        using var doc = JsonDocument.Parse(json);
        var embedding = doc.RootElement
            .GetProperty("data")[0]
            .GetProperty("embedding")
            .EnumerateArray()
            .Select(e => e.GetDouble())
            .ToList();

        _dimension ??= embedding.Count;
        return embedding;
    }

    public async Task<List<List<double>>> EmbedBatchAsync(List<string> texts)
    {
        var results = new List<List<double>>();
        foreach (var text in texts)
            results.Add(await EmbedAsync(text));
        return results;
    }
}
