// HTTP-based visual embedder using an embedding API endpoint.

using System.Buffers.Binary;
using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace AgentMemory.Embeddings;

/// <summary>
/// Visual embedder that calls an HTTP embedding API endpoint for CLIP-style embeddings.
/// </summary>
public sealed class VisualEmbedder : IVisualEmbedder
{
    private readonly HttpClient _client;
    private readonly string _baseUrl;
    private readonly string _model;
    private readonly ILogger<VisualEmbedder>? _logger;
    private int? _dimension;

    public VisualEmbedder(
        string? baseUrl = null,
        string? model = null,
        HttpClient? client = null,
        ILogger<VisualEmbedder>? logger = null)
    {
        _baseUrl = (baseUrl ?? "http://localhost:11434").TrimEnd('/');
        _model = model ?? new MemoryConfig().ClipModel;
        _client = client ?? new HttpClient();
        _logger = logger;
    }

    public int Dimension
    {
        get
        {
            if (_dimension is null)
            {
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

    public async Task<byte[]> EmbedToBytesAsync(string text)
    {
        var floats = await EmbedAsync(text);
        var bytes = new byte[floats.Count * sizeof(float)];
        for (var i = 0; i < floats.Count; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(
                bytes.AsSpan(i * sizeof(float)), (float)floats[i]);
        }
        return bytes;
    }

    public List<double> BytesToVector(byte[] data)
    {
        var count = data.Length / sizeof(float);
        var result = new List<double>(count);
        for (var i = 0; i < count; i++)
        {
            var f = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(i * sizeof(float)));
            result.Add(f);
        }
        return result;
    }
}
