// Exploratory graph walks during sleep (A3).
//
// Two strategies:
//   1. Random anchor pairs — sample pairs from different sessions, check
//      semantic similarity, classify relationship via LLM.
//   2. Cluster bridges — find memories close in vector space but disconnected
//      in the graph (latent connections).

using AgentMemory.Embeddings;
using AgentMemory.Llm;
using AgentMemory.Storage;
using Microsoft.Extensions.Logging;

namespace AgentMemory.Core;

/// <summary>Exploratory graph walks for discovering non-obvious memory connections (A3).</summary>
public static class DreamExplorer
{
    private static readonly HashSet<string> ValidRelationships = new(StringComparer.OrdinalIgnoreCase)
    {
        "caused", "supports", "contradicts", "precedes", "part_of", "analogous", "unrelated"
    };

    /// <summary>Ask the LLM to classify the relationship between two memories.</summary>
    public static async Task<string> ClassifyRelationshipAsync(
        string memAContent, string memBContent,
        ILlmClient llmClient, MemoryConfig config,
        ILogger? logger = null)
    {
        var prompt =
            $"Memory A:\n<memory_a>\n{memAContent}\n</memory_a>\n\n" +
            $"Memory B:\n<memory_b>\n{memBContent}\n</memory_b>\n\n" +
            "What is the relationship between Memory A and Memory B?";

        try
        {
            var response = await llmClient.CompleteAsync(
                prompt, system: config.Prompts.ClassifyRelationship, temperature: 0.1);
            var result = response.Trim().ToLowerInvariant();
            return ValidRelationships.Contains(result) ? result : "unrelated";
        }
        catch
        {
            logger?.LogDebug("Relationship classification failed");
            return "unrelated";
        }
    }

    /// <summary>
    /// Perform semi-random walks to discover non-obvious memory connections.
    /// Returns a list of discovered edges. Caller is responsible for committing them.
    /// </summary>
    public static async Task<List<DiscoveredEdge>> ExploratoryWalkAsync(
        SqliteStore sqlite,
        GraphStore graph,
        VectorStore vector,
        ILlmClient llmClient,
        MemoryConfig config,
        ITextEmbedder? textEmbedder = null,
        int? nWalks = null,
        double? similarityThreshold = null,
        int? maxNewEdges = null,
        ILogger? logger = null)
    {
        nWalks ??= config.DreamWalkCount;
        similarityThreshold ??= config.DreamSimilarityThreshold;
        maxNewEdges ??= config.DreamMaxNewEdges;

        var discovered = new List<DiscoveredEdge>();

        // Get all memories with vector embeddings
        var allMemories = await sqlite.GetMemoriesWithVectorsAsync(tiers: ["hot", "warm"]);
        if (allMemories.Count < 2)
            return discovered;

        if (textEmbedder is null)
            return discovered;

        var rng = new Random();

        for (var i = 0; i < nWalks.Value; i++)
        {
            if (discovered.Count >= maxNewEdges.Value)
                break;

            // Pick two random memories
            var idxA = rng.Next(allMemories.Count);
            int idxB;
            do { idxB = rng.Next(allMemories.Count); } while (idxB == idxA);

            var a = allMemories[idxA];
            var b = allMemories[idxB];

            // Skip if same session (likely already connected)
            if ((string)a["session_id"] == (string)b["session_id"])
                continue;

            // Skip if already connected in graph
            if (graph.PathExists((string)a["id"], (string)b["id"], maxHops: 1))
                continue;

            // Check semantic similarity via vector store
            var sim = await vector.SimilarityAsync(
                (string)a["vector_id"], (string)b["vector_id"]);
            if (sim is null || sim.Value < similarityThreshold.Value)
                continue;

            // Load full memories for classification
            var memA = await sqlite.GetMemoryAsync((string)a["id"]);
            var memB = await sqlite.GetMemoryAsync((string)b["id"]);
            if (memA is null || memB is null)
                continue;

            var relType = await ClassifyRelationshipAsync(
                memA.Content, memB.Content, llmClient, config, logger);

            if (relType != "unrelated")
            {
                discovered.Add(new DiscoveredEdge
                {
                    SourceId = (string)a["id"],
                    TargetId = (string)b["id"],
                    Similarity = sim.Value,
                    RelationshipType = relType,
                    DiscoveryMethod = "random_walk",
                });
            }
        }

        return discovered.Take(maxNewEdges.Value).ToList();
    }

    /// <summary>
    /// Commit discovered edges to the graph and log the exploration run.
    /// Returns the number of edges committed.
    /// </summary>
    public static async Task<int> CommitDiscoveriesAsync(
        List<DiscoveredEdge> discoveries,
        GraphStore graph,
        SqliteStore sqlite,
        string? runId = null,
        ILogger? logger = null)
    {
        var now = DateTime.UtcNow.ToString("o");
        runId ??= Guid.NewGuid().ToString();
        var strategies = discoveries.Select(d => d.DiscoveryMethod).Distinct().ToList();
        var committed = 0;

        foreach (var edge in discoveries)
        {
            var edgeId = Guid.NewGuid().ToString();
            try
            {
                graph.AddRelatesTo(
                    fromId: edge.SourceId,
                    toId: edge.TargetId,
                    weight: edge.Similarity,
                    relationshipType: edge.RelationshipType,
                    createdAt: now);
                committed++;

                await sqlite.LogDreamEdgeAsync(
                    edgeId: edgeId,
                    runId: runId,
                    sourceId: edge.SourceId,
                    targetId: edge.TargetId,
                    similarity: edge.Similarity,
                    relationshipType: edge.RelationshipType,
                    discoveryMethod: edge.DiscoveryMethod,
                    committed: true);
            }
            catch
            {
                logger?.LogDebug("Failed to commit edge {Source} -> {Target}", edge.SourceId, edge.TargetId);

                await sqlite.LogDreamEdgeAsync(
                    edgeId: edgeId,
                    runId: runId,
                    sourceId: edge.SourceId,
                    targetId: edge.TargetId,
                    similarity: edge.Similarity,
                    relationshipType: edge.RelationshipType,
                    discoveryMethod: edge.DiscoveryMethod,
                    committed: false);
            }
        }

        await sqlite.LogDreamRunAsync(
            runId: runId,
            ranAt: now,
            nWalks: discoveries.Count,
            edgesDiscovered: discoveries.Count,
            edgesCommitted: committed,
            strategies: strategies,
            notes: $"Committed {committed}/{discoveries.Count} edges");

        logger?.LogInformation("Dream exploration: committed {Committed}/{Total} edges", committed, discoveries.Count);
        return committed;
    }
}
