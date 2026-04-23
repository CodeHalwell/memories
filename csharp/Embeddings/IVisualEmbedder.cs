// Interface for visual/spatial embedding providers (CLIP-style).

namespace AgentMemory.Embeddings;

/// <summary>Interface for visual/CLIP embedding providers.</summary>
public interface IVisualEmbedder
{
    /// <summary>The dimensionality of the embedding vectors.</summary>
    int Dimension { get; }

    /// <summary>Embed a scene description text. Returns list of doubles.</summary>
    Task<List<double>> EmbedAsync(string text);

    /// <summary>Embed and return raw bytes for storage in SQLite BLOB column.</summary>
    Task<byte[]> EmbedToBytesAsync(string text);

    /// <summary>Convert raw bytes back to double list.</summary>
    List<double> BytesToVector(byte[] data);
}
