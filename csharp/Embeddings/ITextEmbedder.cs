// Interface for text embedding providers.

namespace AgentMemory.Embeddings;

/// <summary>Interface for text embedding providers.</summary>
public interface ITextEmbedder
{
    /// <summary>The dimensionality of the embedding vectors.</summary>
    int Dimension { get; }

    /// <summary>Embed a single text string. Returns a list of doubles.</summary>
    Task<List<double>> EmbedAsync(string text);

    /// <summary>Embed multiple texts. Returns a list of double lists.</summary>
    Task<List<List<double>>> EmbedBatchAsync(List<string> texts);
}
