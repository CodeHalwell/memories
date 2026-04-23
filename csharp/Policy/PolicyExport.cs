// Policy training data export (A4.4).
// Exports decision-outcome pairs as JSONL files for offline policy model training.

using System.Text.Json;
using AgentMemory.Storage;
using Microsoft.Extensions.Logging;

namespace AgentMemory.Policy;

/// <summary>Exports decision-outcome pairs for offline policy model training (A4.4).</summary>
public static class PolicyExport
{
    /// <summary>
    /// Export decision-outcome pairs for offline policy model training.
    /// Returns dict with export metadata: example counts and file paths.
    /// </summary>
    public static async Task<Dictionary<string, object>> ExportPolicyTrainingDataAsync(
        SqliteStore sqlite,
        MemoryConfig? config = null,
        string? outputDir = null,
        ILogger? logger = null)
    {
        config ??= new MemoryConfig();
        outputDir ??= config.PolicyDataDir;
        Directory.CreateDirectory(outputDir);

        var saveData = await sqlite.ExportSavePolicyDataAsync();
        var retrievalData = await sqlite.ExportRetrievalPolicyDataAsync();

        var savePath = Path.Combine(outputDir, "save_policy_data.jsonl");
        var retrievalPath = Path.Combine(outputDir, "retrieval_policy_data.jsonl");

        await using (var writer = new StreamWriter(savePath, false, System.Text.Encoding.UTF8))
        {
            foreach (var row in saveData)
                await writer.WriteLineAsync(JsonSerializer.Serialize(row));
        }

        await using (var writer = new StreamWriter(retrievalPath, false, System.Text.Encoding.UTF8))
        {
            foreach (var row in retrievalData)
                await writer.WriteLineAsync(JsonSerializer.Serialize(row));
        }

        var result = new Dictionary<string, object>
        {
            ["save_examples"] = saveData.Count,
            ["retrieval_examples"] = retrievalData.Count,
            ["save_path"] = savePath,
            ["retrieval_path"] = retrievalPath,
            ["ready_for_training"] =
                saveData.Count >= config.PolicyMinSaveExamples &&
                retrievalData.Count >= config.PolicyMinRetrievalExamples,
        };

        logger?.LogInformation(
            "Policy data export: {SaveCount} save examples, {RetrievalCount} retrieval examples (ready={Ready})",
            result["save_examples"], result["retrieval_examples"], result["ready_for_training"]);

        return result;
    }
}
