// Configuration for the Agent Memory System.

using Microsoft.Extensions.Configuration;

namespace AgentMemory;

/// <summary>
/// Central configuration for the Agent Memory System.
/// All runtime data paths and tuning parameters are defined here.
/// </summary>
public sealed class MemoryConfig
{
    // ── Data paths ──
    public string DataDir { get; set; } = "data";
    public string LogDir => Path.Combine(DataDir, "logs", "sessions");
    public string DbPath => Path.Combine(DataDir, "memory.db");
    public string GraphDir => Path.Combine(DataDir, "graph");
    public string VectorDir => Path.Combine(DataDir, "vectors");
    public string PolicyDataDir => Path.Combine(DataDir, "policy_data");

    // ── LLM ──
    public string? ApiKey { get; set; }
    public string LlmModel { get; set; } = "claude-sonnet-4-6";
    public double LlmTemperature { get; set; } = 0.2;

    // ── Save thresholds ──
    public double SaveConfidenceThreshold { get; set; } = 0.5;
    public double FastPathArousal { get; set; } = 0.85;
    public double FastPathSurprise { get; set; } = 0.75;
    public int MaxKeywordsPerMemory { get; set; } = 10;

    // ── Save decision — retrieval gap awareness (A2.1) ──
    public int GapLookbackTurns { get; set; } = 20;
    public double GapOverlapThreshold { get; set; } = 0.3;
    public double GapThresholdReduction { get; set; } = 0.7;

    // ── Retrieval ──
    public List<string> RetrievalLayers { get; set; } = ["grep", "keyword", "semantic"];
    public int GraphTraversalDepth { get; set; } = 2;
    public double MoodCongruentWeight { get; set; } = 0.2;
    public int TopKPerLayer { get; set; } = 5;

    // ── Compaction ──
    public int HotTierThreshold { get; set; } = 500;
    public double CompactionCandidateThreshold { get; set; } = 0.7;
    public double KeywordOverlapMergeThreshold { get; set; } = 0.6;
    public double ValenceMergeExclusionDelta { get; set; } = 0.6;

    // ── Compaction — merge validation (A2.4) ──
    public int MergeValidationQueries { get; set; } = 5;
    public double MergeDegradationTolerance { get; set; } = 0.15;

    // ── Compaction — generation gap guard (A2.3) ──
    public int MaxGenerationGapForMerge { get; set; } = 1;

    // ── Decay ──
    public double DecayRecencyWeight { get; set; } = 0.6;
    public double DecayFrequencyWeight { get; set; } = 0.4;
    public double DecayHalflifeDays { get; set; } = 7;

    // ── Visual layer ──
    public double VisualSalienceThreshold { get; set; } = 0.7;
    public string ClipModel { get; set; } = "ViT-B-32";

    // ── Embeddings ──
    public string TextEmbeddingModel { get; set; } = "all-MiniLM-L6-v2";

    // ── Dream exploration (A3) ──
    public int DreamWalkCount { get; set; } = 50;
    public double DreamSimilarityThreshold { get; set; } = 0.7;
    public int DreamMaxNewEdges { get; set; } = 20;
    public int DreamClusterMinSize { get; set; } = 3;
    public bool DreamEnabled { get; set; } = true;

    // ── Policy logging (A4) ──
    public bool PolicyLoggingEnabled { get; set; } = true;
    public int SaveOutcomeLookbackDays { get; set; } = 30;
    public int RetrievalOutcomeFollowupTurns { get; set; } = 3;
    public double RetrievalOutcomeKeywordOverlap { get; set; } = 0.5;

    // ── Policy training (A5, v2) ──
    public int PolicyMinSaveExamples { get; set; } = 1000;
    public int PolicyMinRetrievalExamples { get; set; } = 500;

    // ── System Prompts ──
    public PromptConfig Prompts { get; set; } = new();

    /// <summary>
    /// Load configuration from an appsettings.json file, falling back to defaults.
    /// </summary>
    public static MemoryConfig Load(string? path = null)
    {
        var config = new MemoryConfig();

        var filePath = path ?? "appsettings.json";
        if (!File.Exists(filePath))
            return config;

        var builder = new ConfigurationBuilder()
            .AddJsonFile(filePath, optional: true, reloadOnChange: false);

        var root = builder.Build();
        var section = root.GetSection("MemoryConfig");
        if (section.Exists())
        {
            section.Bind(config);
        }

        return config;
    }
}

/// <summary>Prompt templates used by LLM operations.</summary>
public sealed class PromptConfig
{
    public string SceneDescription { get; set; } = Prompts.SceneDescriptionSystem;
    public string SaveDecision { get; set; } = Prompts.SaveDecisionSystem;
    public string Merge { get; set; } = Prompts.MergeSystem;
    public string SyntheticQuery { get; set; } = Prompts.SyntheticQuerySystem;
    public string Emotion { get; set; } = Prompts.EmotionSystem;
    public string ClassifyRelationship { get; set; } = Prompts.ClassifyRelationshipSystem;
}
