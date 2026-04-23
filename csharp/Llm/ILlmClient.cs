// Interface for LLM completion clients.

namespace AgentMemory.Llm;

/// <summary>Interface for LLM completion providers.</summary>
public interface ILlmClient
{
    /// <summary>Send a completion request and return the text response.</summary>
    Task<string> CompleteAsync(
        string prompt,
        string? system = null,
        string? model = null,
        double? temperature = null,
        int maxRetries = 3);

    /// <summary>Send a completion request and parse the response as JSON.</summary>
    Task<Dictionary<string, object?>> CompleteJsonAsync(
        string prompt,
        string? system = null,
        string? model = null,
        double? temperature = null);
}
