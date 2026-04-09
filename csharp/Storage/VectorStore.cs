// Qdrant vector store client using REST API via HttpClient.
// Manages two collections: memory_text and memory_visual.

using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Logging;

namespace AgentMemory.Storage;

/// <summary>Qdrant REST API vector store for memory embeddings.</summary>
public sealed class VectorStore : IDisposable
{
    public const string TextCollection = "memory_text";
    public const string VisualCollection = "memory_visual";

    private readonly HttpClient _client;
    private readonly string _baseUrl;
    private readonly ILogger<VectorStore>? _logger;
    private readonly bool _ownsClient;

    public VectorStore(string? baseUrl = null, HttpClient? client = null, ILogger<VectorStore>? logger = null)
    {
        _baseUrl = (baseUrl ?? "http://localhost:6333").TrimEnd('/');
        _ownsClient = client is null;
        _client = client ?? new HttpClient();
        _logger = logger;
    }

    public async Task InitializeAsync(int textDim = 384, int visualDim = 512)
    {
        await EnsureCollectionAsync(TextCollection, textDim);
        await EnsureCollectionAsync(VisualCollection, visualDim);
    }

    public void Dispose()
    {
        if (_ownsClient)
            _client.Dispose();
    }

    private async Task EnsureCollectionAsync(string name, int dim)
    {
        try
        {
            var resp = await _client.GetAsync($"{_baseUrl}/collections/{name}");
            if (resp.IsSuccessStatusCode)
                return;
        }
        catch { /* collection doesn't exist */ }

        var body = JsonSerializer.Serialize(new
        {
            vectors = new { size = dim, distance = "Cosine" }
        });

        var content = new StringContent(body, System.Text.Encoding.UTF8, "application/json");
        await _client.PutAsync($"{_baseUrl}/collections/{name}", content);
    }

    // ── Text embeddings ──

    /// <summary>Insert or update a text embedding. Returns the point ID.</summary>
    public async Task<string> UpsertTextVectorAsync(
        string memoryId, List<double> vector,
        string tier = "hot", double valence = 0.0, double arousal = 0.0,
        string sessionId = "", string createdAt = "")
    {
        var pointId = Guid.NewGuid().ToString();

        var body = JsonSerializer.Serialize(new
        {
            points = new[]
            {
                new
                {
                    id = pointId,
                    vector = vector,
                    payload = new Dictionary<string, object>
                    {
                        ["memory_id"] = memoryId,
                        ["tier"] = tier,
                        ["valence"] = valence,
                        ["arousal"] = arousal,
                        ["session_id"] = sessionId,
                        ["created_at"] = createdAt,
                    }
                }
            }
        });

        var content = new StringContent(body, System.Text.Encoding.UTF8, "application/json");
        await _client.PutAsync($"{_baseUrl}/collections/{TextCollection}/points", content);

        return pointId;
    }

    /// <summary>Search for nearest text embeddings.</summary>
    public async Task<List<VectorSearchResult>> SearchTextAsync(
        List<double> queryVector, int limit = 5, string? tierFilter = null)
    {
        var request = new Dictionary<string, object>
        {
            ["vector"] = queryVector,
            ["limit"] = limit,
            ["with_payload"] = true,
        };

        if (tierFilter is not null)
        {
            request["filter"] = new
            {
                must = new[]
                {
                    new { key = "tier", match = new { value = tierFilter } }
                }
            };
        }

        var body = JsonSerializer.Serialize(request);
        var content = new StringContent(body, System.Text.Encoding.UTF8, "application/json");
        var resp = await _client.PostAsync($"{_baseUrl}/collections/{TextCollection}/points/search", content);
        var json = await resp.Content.ReadAsStringAsync();

        return ParseSearchResults(json);
    }

    // ── Visual embeddings ──

    /// <summary>Insert or update a visual (CLIP) embedding. Returns the point ID.</summary>
    public async Task<string> UpsertVisualVectorAsync(
        string memoryId, List<double> vector,
        string sessionId = "", string createdAt = "")
    {
        var pointId = Guid.NewGuid().ToString();

        var body = JsonSerializer.Serialize(new
        {
            points = new[]
            {
                new
                {
                    id = pointId,
                    vector = vector,
                    payload = new Dictionary<string, object>
                    {
                        ["memory_id"] = memoryId,
                        ["session_id"] = sessionId,
                        ["created_at"] = createdAt,
                    }
                }
            }
        });

        var content = new StringContent(body, System.Text.Encoding.UTF8, "application/json");
        await _client.PutAsync($"{_baseUrl}/collections/{VisualCollection}/points", content);

        return pointId;
    }

    /// <summary>Search for nearest visual embeddings.</summary>
    public async Task<List<VectorSearchResult>> SearchVisualAsync(
        List<double> queryVector, int limit = 5)
    {
        var body = JsonSerializer.Serialize(new
        {
            vector = queryVector,
            limit,
            with_payload = true,
        });

        var content = new StringContent(body, System.Text.Encoding.UTF8, "application/json");
        var resp = await _client.PostAsync($"{_baseUrl}/collections/{VisualCollection}/points/search", content);
        var json = await resp.Content.ReadAsStringAsync();

        return ParseSearchResults(json);
    }

    /// <summary>
    /// Compute cosine similarity between two points in the text collection.
    /// Used by dream explorer (A3) for cross-session similarity checks.
    /// </summary>
    public async Task<double?> SimilarityAsync(string pointIdA, string pointIdB)
    {
        try
        {
            var body = JsonSerializer.Serialize(new
            {
                ids = new[] { pointIdA, pointIdB },
                with_vector = true,
            });

            var content = new StringContent(body, System.Text.Encoding.UTF8, "application/json");
            var resp = await _client.PostAsync(
                $"{_baseUrl}/collections/{TextCollection}/points", content);
            var json = await resp.Content.ReadAsStringAsync();

            using var doc = JsonDocument.Parse(json);
            var result = doc.RootElement.GetProperty("result");

            if (result.GetArrayLength() < 2)
                return null;

            var vecA = result[0].GetProperty("vector").EnumerateArray().Select(e => e.GetDouble()).ToArray();
            var vecB = result[1].GetProperty("vector").EnumerateArray().Select(e => e.GetDouble()).ToArray();

            return CosineSimilarity(vecA, vecB);
        }
        catch
        {
            return null;
        }
    }

    /// <summary>Delete all points for a given memory_id from a collection.</summary>
    public async Task DeletePointAsync(string collection, string memoryId)
    {
        var body = JsonSerializer.Serialize(new
        {
            filter = new
            {
                must = new[]
                {
                    new { key = "memory_id", match = new { value = memoryId } }
                }
            }
        });

        var content = new StringContent(body, System.Text.Encoding.UTF8, "application/json");
        await _client.PostAsync($"{_baseUrl}/collections/{collection}/points/delete", content);
    }

    // ── Helpers ──

    private static double CosineSimilarity(double[] a, double[] b)
    {
        double dot = 0, normA = 0, normB = 0;
        for (var i = 0; i < a.Length && i < b.Length; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        var norm = Math.Sqrt(normA) * Math.Sqrt(normB);
        return norm > 0 ? dot / norm : 0.0;
    }

    private static List<VectorSearchResult> ParseSearchResults(string json)
    {
        var results = new List<VectorSearchResult>();
        try
        {
            using var doc = JsonDocument.Parse(json);
            if (!doc.RootElement.TryGetProperty("result", out var resultArray))
                return results;

            foreach (var item in resultArray.EnumerateArray())
            {
                var payload = item.GetProperty("payload");
                var result = new VectorSearchResult
                {
                    MemoryId = payload.TryGetProperty("memory_id", out var mid) ? mid.GetString() ?? "" : "",
                    Score = item.TryGetProperty("score", out var s) ? s.GetDouble() : 0.0,
                    Tier = payload.TryGetProperty("tier", out var t) ? t.GetString() ?? "hot" : "hot",
                    Valence = payload.TryGetProperty("valence", out var v) ? v.GetDouble() : 0.0,
                    Arousal = payload.TryGetProperty("arousal", out var a) ? a.GetDouble() : 0.0,
                };
                results.Add(result);
            }
        }
        catch { /* parse failure — return empty */ }

        return results;
    }
}

/// <summary>Search result from the vector store.</summary>
public sealed class VectorSearchResult
{
    public string MemoryId { get; set; } = "";
    public double Score { get; set; }
    public string Tier { get; set; } = "hot";
    public double Valence { get; set; }
    public double Arousal { get; set; }
}
