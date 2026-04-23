// Memory Manager — orchestrates save, retrieve, and compact operations.
//
// Primary interface for the agent runtime. Coordinates all storage backends,
// embedding models, and the LLM-driven save decision pipeline.
//
// Addendum integrations:
//   A2.1: Gap-aware save decisions
//   A2.2: Emotional decay modulation
//   A2.3: Generation gap guard
//   A2.4: Merge validation
//   A2.5: Keyword reweighting during compaction
//   A3:   Dream exploration during sleep cycle
//   A4:   Policy logging for retrieval decisions and outcome assessment

using AgentMemory.Embeddings;
using AgentMemory.Llm;
using AgentMemory.Policy;
using AgentMemory.Storage;
using Microsoft.Extensions.Logging;

namespace AgentMemory.Core;

/// <summary>Top-level orchestrator for the agent memory system.</summary>
public sealed class MemoryManager : IAsyncDisposable
{
    private readonly MemoryConfig _config;
    private readonly ILogger<MemoryManager>? _logger;

    // Storage backends
    public JsonlLogger Jsonl { get; }
    public SqliteStore Sqlite { get; }
    public GraphStore Graph { get; }
    public VectorStore Vector { get; }

    // Embedding models
    public ITextEmbedder TextEmbedder { get; }
    public IVisualEmbedder? VisualEmbedder { get; }

    // LLM client
    public ILlmClient LlmClient { get; }

    // Sub-engines
    private RetrievalEngine? _retrieval;
    private CompactionEngine? _compaction;

    // Track first turns per session
    private readonly HashSet<string> _firstTurns = [];

    public MemoryManager(
        MemoryConfig? config = null,
        ITextEmbedder? textEmbedder = null,
        IVisualEmbedder? visualEmbedder = null,
        ILlmClient? llmClient = null,
        string? dataDir = null,
        ILogger<MemoryManager>? logger = null)
    {
        _config = config ?? new MemoryConfig();
        if (dataDir is not null)
            _config.DataDir = dataDir;

        _logger = logger;

        // Storage backends
        Jsonl = new JsonlLogger(_config.LogDir);
        Sqlite = new SqliteStore(_config.DbPath);
        Graph = new GraphStore(_config.GraphDir);
        Vector = new VectorStore();

        // Embedding models
        TextEmbedder = textEmbedder ?? new TextEmbedder();
        VisualEmbedder = visualEmbedder;

        // LLM client
        LlmClient = llmClient ?? new LlmClient(config: _config);
    }

    /// <summary>Initialize all storage backends. Must be called before any operations.</summary>
    public async Task InitializeAsync()
    {
        await Sqlite.InitializeAsync();
        Graph.Initialize();
        await Vector.InitializeAsync(
            textDim: TextEmbedder.Dimension,
            visualDim: VisualEmbedder?.Dimension ?? 512);

        _retrieval = new RetrievalEngine(
            sqlite: Sqlite,
            graph: Graph,
            vector: Vector,
            textEmbedder: TextEmbedder,
            llmClient: LlmClient,
            config: _config,
            visualEmbedder: VisualEmbedder);

        _compaction = new CompactionEngine(
            sqlite: Sqlite,
            graph: Graph,
            vector: Vector,
            llmClient: LlmClient,
            config: _config,
            textEmbedder: TextEmbedder);
    }

    /// <summary>Initialize without loading embedding models (for testing or lightweight use).</summary>
    public async Task InitializeLiteAsync()
    {
        await Sqlite.InitializeAsync();
        Graph.Initialize();
    }

    public async ValueTask DisposeAsync()
    {
        await Sqlite.DisposeAsync();
        Graph.Close();
        Vector.Dispose();
    }

    // ── Core operations ──

    /// <summary>
    /// Log an agent output and decide whether to save it as a memory.
    /// This is the main entry point called at the end of each turn.
    /// Returns the Memory if one was created, else null.
    /// </summary>
    public async Task<Memory?> ProcessTurnAsync(
        string sessionId, int turn, string content,
        string role = "assistant", int tokenCount = 0,
        string model = "", string provider = "")
    {
        // 1. Create and persist raw log entry
        var entry = new RawLogEntry
        {
            SessionId = sessionId,
            Turn = turn,
            Content = content,
            Role = role,
            TokenCount = tokenCount,
            Model = model,
            Provider = provider,
        };
        var (filePath, byteOffset) = Jsonl.Append(entry);

        // 2. Index the raw log entry in SQLite
        await Sqlite.IndexRawLogAsync(
            entryId: entry.Id,
            sessionId: sessionId,
            turn: turn,
            timestamp: entry.Timestamp,
            filePath: filePath,
            byteOffset: byteOffset);

        // 3. Run save decision (A2.1: pass sqlite for gap awareness)
        var isFirst = _firstTurns.Add(sessionId);

        var (decision, memory) = await SaveDecisionEngine.MakeSaveDecisionAsync(
            entry, LlmClient, _config,
            isFirstTurn: isFirst,
            sqlite: Sqlite,
            logger: _logger);

        // 4. Log the decision
        await Sqlite.LogSaveDecisionAsync(decision);

        if (memory is null)
            return null;

        // 5. Save the memory
        await Sqlite.SaveMemoryAsync(memory);

        // 6. Create graph node
        Graph.AddMemoryNode(
            memoryId: memory.Id,
            summary: memory.Summary ?? "",
            tier: memory.Tier,
            salience: memory.Salience,
            valence: memory.Valence,
            compactionGen: memory.CompactionGen,
            createdAt: memory.CreatedAt);
        memory.GraphNodeId = memory.Id;
        await Sqlite.UpdateMemoryGraphRefAsync(memory.Id, memory.Id);

        // 7. Create text embedding and store in vector store
        try
        {
            var textVector = await TextEmbedder.EmbedAsync(memory.Content);
            var pointId = await Vector.UpsertTextVectorAsync(
                memoryId: memory.Id,
                vector: textVector,
                tier: memory.Tier,
                valence: memory.Valence,
                arousal: memory.Arousal,
                sessionId: sessionId,
                createdAt: memory.CreatedAt);
            memory.VectorId = pointId;
            await Sqlite.UpdateMemoryVectorRefAsync(memory.Id, pointId);
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Failed to create text embedding for memory {MemoryId}", memory.Id);
        }

        // 8. Visual layer — generate scene description and CLIP embedding for salient memories
        if (memory.Salience > _config.VisualSalienceThreshold)
            await GenerateVisualLayerAsync(memory);

        return memory;
    }

    /// <summary>
    /// Run three-layer retrieval and return ranked memories.
    /// Also logs the retrieval decision for policy training (A4).
    /// </summary>
    public async Task<List<Memory>> RetrieveAsync(
        string query, string? sessionId = null, int? topK = null)
    {
        if (_retrieval is null)
            throw new InvalidOperationException("MemoryManager not initialized — call InitializeAsync() first");

        var memories = await _retrieval.RetrieveAsync(
            query: query, sessionId: sessionId, topK: topK);

        // A4: Log retrieval decision
        if (_config.PolicyLoggingEnabled)
        {
            try
            {
                var now = DateTime.UtcNow.ToString("o");
                await Sqlite.LogRetrievalDecisionAsync(
                    decisionId: Guid.NewGuid().ToString(),
                    sessionId: sessionId ?? "",
                    turn: null,
                    query: query,
                    decidedAt: now,
                    layersQueried: _config.RetrievalLayers,
                    graphDepth: _config.GraphTraversalDepth,
                    moodWeight: _config.MoodCongruentWeight,
                    topK: topK ?? _config.TopKPerLayer,
                    memoryIds: memories.Select(m => m.Id).ToList(),
                    returnCount: memories.Count);
            }
            catch
            {
                _logger?.LogDebug("Failed to log retrieval decision");
            }
        }

        return memories;
    }

    /// <summary>
    /// Run a compaction cycle with optional exploration phase.
    /// Phases:
    ///   1. Standard compaction (merge pass)
    ///   2. Keyword reweighting from graph structure (A2.5)
    ///   3. Exploratory walk / dream phase (A3, scheduled/manual only)
    ///   4. Outcome assessment (A4)
    /// </summary>
    public async Task<CompactionResult> RunCompactionAsync(string trigger = "scheduled")
    {
        if (_compaction is null)
            throw new InvalidOperationException("MemoryManager not initialized — call InitializeAsync() first");

        // Phase 1: Standard compaction
        var result = await _compaction.RunAsync(trigger: trigger);

        // Phase 2: Keyword reweighting (A2.5)
        try
        {
            var keywordsUpdated = await KeywordReweight.ReweightKeywordsFromGraphAsync(Sqlite, Graph);
            result.KeywordsUpdated = keywordsUpdated;
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Keyword reweighting failed");
        }

        // Phase 3: Dream exploration (A3) — only for scheduled/manual triggers
        if (trigger is "scheduled" or "manual" && _config.DreamEnabled)
        {
            try
            {
                var discoveries = await DreamExplorer.ExploratoryWalkAsync(
                    sqlite: Sqlite,
                    graph: Graph,
                    vector: Vector,
                    llmClient: LlmClient,
                    config: _config,
                    textEmbedder: TextEmbedder);

                if (discoveries.Count > 0)
                {
                    var committed = await DreamExplorer.CommitDiscoveriesAsync(
                        discoveries, Graph, Sqlite);
                    result.EdgesDiscovered = committed;
                }
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Dream exploration failed");
            }
        }

        // Phase 4: Outcome assessment (A4)
        if (_config.PolicyLoggingEnabled)
        {
            try
            {
                await OutcomeAssessor.AssessSaveOutcomesAsync(Sqlite, _config);
                await OutcomeAssessor.AssessRetrievalOutcomesAsync(Sqlite, _config);
            }
            catch
            {
                _logger?.LogDebug("Outcome assessment failed");
            }
        }

        return result;
    }

    /// <summary>Fetch a single memory and log the access.</summary>
    public async Task<Memory?> GetMemoryAsync(string memoryId)
    {
        var mem = await Sqlite.GetMemoryAsync(memoryId);
        if (mem is null)
            return null;

        var now = DateTime.UtcNow.ToString("o");
        mem.AccessCount += 1;
        mem.LastAccessed = now;
        mem.DecayScore = Decay.ComputeDecay(
            DateTime.Parse(now).ToUniversalTime(),
            mem.AccessCount,
            arousal: mem.Arousal,
            surprise: mem.Surprise,
            isSemantic: mem.IsSemantic,
            config: _config);

        await Sqlite.LogAccessAsync(
            accessId: Guid.NewGuid().ToString(),
            memoryId: memoryId,
            accessedAt: now,
            accessType: "primary");

        await Sqlite.UpdateMemoryAccessAsync(
            memoryId, mem.DecayScore, mem.AccessCount, now);

        return mem;
    }

    // ── Visual layer ──

    private async Task GenerateVisualLayerAsync(Memory memory)
    {
        try
        {
            // Generate scene description via LLM
            var scene = await LlmClient.CompleteAsync(
                $"Generate an abstract scene description for this memory:\n\n<memory_content>\n{memory.Content}\n</memory_content>",
                system: _config.Prompts.SceneDescription);

            memory.SceneDescription = scene.Trim();

            if (VisualEmbedder is not null)
            {
                // Create CLIP embedding
                var spatialBytes = await VisualEmbedder.EmbedToBytesAsync(memory.SceneDescription);
                memory.SpatialEmbedding = spatialBytes;

                // Store in vector store visual collection
                var visualVector = await VisualEmbedder.EmbedAsync(memory.SceneDescription);
                await Vector.UpsertVisualVectorAsync(
                    memoryId: memory.Id,
                    vector: visualVector,
                    sessionId: memory.SessionId,
                    createdAt: memory.CreatedAt);
            }

            // Update SQLite
            if (memory.SpatialEmbedding is not null)
            {
                await Sqlite.UpdateMemoryVisualAsync(
                    memory.Id, memory.SceneDescription, memory.SpatialEmbedding);
            }
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Visual layer generation failed for memory {MemoryId}", memory.Id);
        }
    }
}
