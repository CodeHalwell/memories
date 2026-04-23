// SQLite storage for memory metadata, access tracking, and compaction history.

using System.Text.Json;
using Microsoft.Data.Sqlite;

namespace AgentMemory.Storage;

/// <summary>Async SQLite store for memory metadata.</summary>
public sealed class SqliteStore : IAsyncDisposable
{
    private readonly string _dbPath;
    private SqliteConnection? _db;

    public SqliteStore(string? dbPath = null)
    {
        _dbPath = dbPath ?? new MemoryConfig().DbPath;
    }

    public async Task InitializeAsync()
    {
        var dir = Path.GetDirectoryName(_dbPath);
        if (!string.IsNullOrEmpty(dir))
            Directory.CreateDirectory(dir);

        _db = new SqliteConnection($"Data Source={_dbPath}");
        await _db.OpenAsync();

        await using var cmd = _db.CreateCommand();
        cmd.CommandText = Schema;
        await cmd.ExecuteNonQueryAsync();
    }

    public async ValueTask DisposeAsync()
    {
        if (_db is not null)
        {
            await _db.DisposeAsync();
            _db = null;
        }
    }

    private SqliteConnection Db =>
        _db ?? throw new InvalidOperationException("SqliteStore not initialized — call InitializeAsync() first");

    // ── Raw log index ──

    public async Task IndexRawLogAsync(
        string entryId, string sessionId, int turn,
        string timestamp, string filePath, long byteOffset)
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText =
            "INSERT OR IGNORE INTO raw_log_index (id, session_id, turn, timestamp, file_path, byte_offset) " +
            "VALUES ($id, $session_id, $turn, $timestamp, $file_path, $byte_offset)";
        cmd.Parameters.AddWithValue("$id", entryId);
        cmd.Parameters.AddWithValue("$session_id", sessionId);
        cmd.Parameters.AddWithValue("$turn", turn);
        cmd.Parameters.AddWithValue("$timestamp", timestamp);
        cmd.Parameters.AddWithValue("$file_path", filePath);
        cmd.Parameters.AddWithValue("$byte_offset", byteOffset);
        await cmd.ExecuteNonQueryAsync();
    }

    public async Task<Dictionary<string, object>?> GetRawLogRefAsync(string entryId)
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText = "SELECT * FROM raw_log_index WHERE id = $id";
        cmd.Parameters.AddWithValue("$id", entryId);
        await using var reader = await cmd.ExecuteReaderAsync();
        if (!await reader.ReadAsync())
            return null;
        return ReadRow(reader);
    }

    // ── Memories ──

    public async Task SaveMemoryAsync(Memory mem)
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText =
            """
            INSERT OR REPLACE INTO memories
            (id, created_at, updated_at, content, summary, raw_log_id, session_id, turn,
             valence, arousal, surprise, salience, access_count, last_accessed, decay_score,
             compaction_gen, tier, fast_pathed, is_semantic, graph_node_id, vector_id,
             spatial_embedding, scene_description)
            VALUES ($id,$created_at,$updated_at,$content,$summary,$raw_log_id,$session_id,$turn,
                    $valence,$arousal,$surprise,$salience,$access_count,$last_accessed,$decay_score,
                    $compaction_gen,$tier,$fast_pathed,$is_semantic,$graph_node_id,$vector_id,
                    $spatial_embedding,$scene_description)
            """;
        cmd.Parameters.AddWithValue("$id", mem.Id);
        cmd.Parameters.AddWithValue("$created_at", mem.CreatedAt);
        cmd.Parameters.AddWithValue("$updated_at", mem.UpdatedAt);
        cmd.Parameters.AddWithValue("$content", mem.Content);
        cmd.Parameters.AddWithValue("$summary", (object?)mem.Summary ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$raw_log_id", mem.RawLogId);
        cmd.Parameters.AddWithValue("$session_id", mem.SessionId);
        cmd.Parameters.AddWithValue("$turn", mem.Turn);
        cmd.Parameters.AddWithValue("$valence", mem.Valence);
        cmd.Parameters.AddWithValue("$arousal", mem.Arousal);
        cmd.Parameters.AddWithValue("$surprise", mem.Surprise);
        cmd.Parameters.AddWithValue("$salience", mem.Salience);
        cmd.Parameters.AddWithValue("$access_count", mem.AccessCount);
        cmd.Parameters.AddWithValue("$last_accessed", (object?)mem.LastAccessed ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$decay_score", mem.DecayScore);
        cmd.Parameters.AddWithValue("$compaction_gen", mem.CompactionGen);
        cmd.Parameters.AddWithValue("$tier", mem.Tier);
        cmd.Parameters.AddWithValue("$fast_pathed", mem.FastPathed ? 1 : 0);
        cmd.Parameters.AddWithValue("$is_semantic", mem.IsSemantic ? 1 : 0);
        cmd.Parameters.AddWithValue("$graph_node_id", (object?)mem.GraphNodeId ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$vector_id", (object?)mem.VectorId ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$spatial_embedding", (object?)mem.SpatialEmbedding ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$scene_description", (object?)mem.SceneDescription ?? DBNull.Value);
        await cmd.ExecuteNonQueryAsync();

        // Save keywords
        await using var delCmd = Db.CreateCommand();
        delCmd.CommandText = "DELETE FROM memory_keywords WHERE memory_id = $id";
        delCmd.Parameters.AddWithValue("$id", mem.Id);
        await delCmd.ExecuteNonQueryAsync();

        foreach (var (kw, weight) in mem.Keywords)
        {
            await using var kwCmd = Db.CreateCommand();
            kwCmd.CommandText =
                "INSERT INTO memory_keywords (memory_id, keyword, weight) VALUES ($mid, $kw, $w)";
            kwCmd.Parameters.AddWithValue("$mid", mem.Id);
            kwCmd.Parameters.AddWithValue("$kw", kw);
            kwCmd.Parameters.AddWithValue("$w", weight);
            await kwCmd.ExecuteNonQueryAsync();
        }
    }

    public async Task<Memory?> GetMemoryAsync(string memoryId)
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText = "SELECT * FROM memories WHERE id = $id";
        cmd.Parameters.AddWithValue("$id", memoryId);
        await using var reader = await cmd.ExecuteReaderAsync();
        if (!await reader.ReadAsync())
            return null;

        var mem = RowToMemory(reader);

        // Load keywords
        mem.Keywords = await LoadKeywordsAsync(memoryId);
        return mem;
    }

    public async Task<List<Memory>> ListMemoriesAsync(
        string? tier = null, int limit = 100, int offset = 0)
    {
        await using var cmd = Db.CreateCommand();
        if (tier is not null)
        {
            cmd.CommandText = "SELECT * FROM memories WHERE tier = $tier ORDER BY created_at DESC LIMIT $limit OFFSET $offset";
            cmd.Parameters.AddWithValue("$tier", tier);
        }
        else
        {
            cmd.CommandText = "SELECT * FROM memories ORDER BY created_at DESC LIMIT $limit OFFSET $offset";
        }
        cmd.Parameters.AddWithValue("$limit", limit);
        cmd.Parameters.AddWithValue("$offset", offset);

        var memories = new List<Memory>();
        await using var reader = await cmd.ExecuteReaderAsync();
        while (await reader.ReadAsync())
            memories.Add(RowToMemory(reader));

        // Batch-load keywords
        if (memories.Count > 0)
        {
            var kwMap = await BatchLoadKeywordsAsync(memories.Select(m => m.Id).ToList());
            foreach (var m in memories)
                m.Keywords = kwMap.GetValueOrDefault(m.Id, []);
        }

        return memories;
    }

    public async Task<int> CountMemoriesAsync(string? tier = null)
    {
        await using var cmd = Db.CreateCommand();
        if (tier is not null)
        {
            cmd.CommandText = "SELECT COUNT(*) FROM memories WHERE tier = $tier";
            cmd.Parameters.AddWithValue("$tier", tier);
        }
        else
        {
            cmd.CommandText = "SELECT COUNT(*) FROM memories";
        }

        var result = await cmd.ExecuteScalarAsync();
        return Convert.ToInt32(result);
    }

    public async Task UpdateMemoryAccessAsync(
        string memoryId, double decayScore, int accessCount, string lastAccessed)
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText =
            "UPDATE memories SET decay_score = $ds, access_count = $ac, last_accessed = $la, updated_at = $la WHERE id = $id";
        cmd.Parameters.AddWithValue("$ds", decayScore);
        cmd.Parameters.AddWithValue("$ac", accessCount);
        cmd.Parameters.AddWithValue("$la", lastAccessed);
        cmd.Parameters.AddWithValue("$id", memoryId);
        await cmd.ExecuteNonQueryAsync();
    }

    public async Task UpdateMemoryTierAsync(string memoryId, string tier)
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText = "UPDATE memories SET tier = $tier WHERE id = $id";
        cmd.Parameters.AddWithValue("$tier", tier);
        cmd.Parameters.AddWithValue("$id", memoryId);
        await cmd.ExecuteNonQueryAsync();
    }

    public async Task UpdateMemoryGraphRefAsync(string memoryId, string graphNodeId)
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText = "UPDATE memories SET graph_node_id = $gid WHERE id = $id";
        cmd.Parameters.AddWithValue("$gid", graphNodeId);
        cmd.Parameters.AddWithValue("$id", memoryId);
        await cmd.ExecuteNonQueryAsync();
    }

    public async Task UpdateMemoryVectorRefAsync(string memoryId, string vectorId)
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText = "UPDATE memories SET vector_id = $vid WHERE id = $id";
        cmd.Parameters.AddWithValue("$vid", vectorId);
        cmd.Parameters.AddWithValue("$id", memoryId);
        await cmd.ExecuteNonQueryAsync();
    }

    public async Task UpdateMemoryVisualAsync(
        string memoryId, string sceneDescription, byte[] spatialEmbedding)
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText =
            "UPDATE memories SET scene_description = $sd, spatial_embedding = $se WHERE id = $id";
        cmd.Parameters.AddWithValue("$sd", sceneDescription);
        cmd.Parameters.AddWithValue("$se", spatialEmbedding);
        cmd.Parameters.AddWithValue("$id", memoryId);
        await cmd.ExecuteNonQueryAsync();
    }

    // ── Keyword search ──

    public async Task<List<Memory>> SearchByKeywordsAsync(List<string> keywords, int limit = 10)
    {
        if (keywords.Count == 0)
            return [];

        var placeholders = string.Join(",", keywords.Select((_, i) => $"$kw{i}"));
        var sql = $"""
            SELECT m.*, SUM(mk.weight) as match_score
            FROM memories m
            JOIN memory_keywords mk ON m.id = mk.memory_id
            WHERE mk.keyword IN ({placeholders})
            GROUP BY m.id
            ORDER BY match_score * m.decay_score DESC
            LIMIT $limit
            """;

        await using var cmd = Db.CreateCommand();
        cmd.CommandText = sql;
        for (var i = 0; i < keywords.Count; i++)
            cmd.Parameters.AddWithValue($"$kw{i}", keywords[i]);
        cmd.Parameters.AddWithValue("$limit", limit);

        var memories = new List<Memory>();
        await using var reader = await cmd.ExecuteReaderAsync();
        while (await reader.ReadAsync())
            memories.Add(RowToMemory(reader));

        // Load keywords for results
        if (memories.Count > 0)
        {
            var kwMap = await BatchLoadKeywordsAsync(memories.Select(m => m.Id).ToList());
            foreach (var m in memories)
                m.Keywords = kwMap.GetValueOrDefault(m.Id, []);
        }

        return memories;
    }

    public async Task UpdateKeywordWeightAsync(string memoryId, string keyword, double weight)
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText =
            "UPDATE memory_keywords SET weight = MIN($w, 1.0) WHERE memory_id = $mid AND keyword = $kw";
        cmd.Parameters.AddWithValue("$w", weight);
        cmd.Parameters.AddWithValue("$mid", memoryId);
        cmd.Parameters.AddWithValue("$kw", keyword);
        await cmd.ExecuteNonQueryAsync();
    }

    public async Task BatchUpdateKeywordWeightsAsync(List<(double Weight, string MemoryId, string Keyword)> updates)
    {
        foreach (var (weight, memoryId, keyword) in updates)
        {
            await using var cmd = Db.CreateCommand();
            cmd.CommandText =
                "UPDATE memory_keywords SET weight = MIN($w, 1.0) WHERE memory_id = $mid AND keyword = $kw";
            cmd.Parameters.AddWithValue("$w", weight);
            cmd.Parameters.AddWithValue("$mid", memoryId);
            cmd.Parameters.AddWithValue("$kw", keyword);
            await cmd.ExecuteNonQueryAsync();
        }
    }

    public async Task<List<Dictionary<string, object>>> GetAllKeywordsWithMemoriesAsync(
        List<string>? tiers = null)
    {
        var tiersToUse = tiers ?? ["hot", "warm"];
        var placeholders = string.Join(",", tiersToUse.Select((_, i) => $"$t{i}"));
        var sql = $"""
            SELECT mk.keyword, mk.memory_id, mk.weight
            FROM memory_keywords mk
            JOIN memories m ON mk.memory_id = m.id
            WHERE m.tier IN ({placeholders})
            """;

        await using var cmd = Db.CreateCommand();
        cmd.CommandText = sql;
        for (var i = 0; i < tiersToUse.Count; i++)
            cmd.Parameters.AddWithValue($"$t{i}", tiersToUse[i]);

        var rows = new List<Dictionary<string, object>>();
        await using var reader = await cmd.ExecuteReaderAsync();
        while (await reader.ReadAsync())
        {
            rows.Add(new Dictionary<string, object>
            {
                ["keyword"] = reader.GetString(reader.GetOrdinal("keyword")),
                ["memory_id"] = reader.GetString(reader.GetOrdinal("memory_id")),
                ["weight"] = reader.GetDouble(reader.GetOrdinal("weight")),
            });
        }

        return rows;
    }

    // ── Access log ──

    public async Task LogAccessAsync(
        string accessId, string memoryId, string accessedAt,
        string accessType, string? sessionId = null, string? query = null)
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText =
            "INSERT INTO memory_access_log (id, memory_id, accessed_at, access_type, session_id, query) " +
            "VALUES ($id, $mid, $at, $atype, $sid, $q)";
        cmd.Parameters.AddWithValue("$id", accessId);
        cmd.Parameters.AddWithValue("$mid", memoryId);
        cmd.Parameters.AddWithValue("$at", accessedAt);
        cmd.Parameters.AddWithValue("$atype", accessType);
        cmd.Parameters.AddWithValue("$sid", (object?)sessionId ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$q", (object?)query ?? DBNull.Value);
        await cmd.ExecuteNonQueryAsync();
    }

    public async Task<List<string>> GetRecentAccessQueriesAsync(string sessionId, int limit = 20)
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText =
            "SELECT query FROM memory_access_log " +
            "WHERE session_id = $sid AND query IS NOT NULL " +
            "ORDER BY accessed_at DESC LIMIT $limit";
        cmd.Parameters.AddWithValue("$sid", sessionId);
        cmd.Parameters.AddWithValue("$limit", limit);

        var queries = new List<string>();
        await using var reader = await cmd.ExecuteReaderAsync();
        while (await reader.ReadAsync())
            queries.Add(reader.GetString(0));
        return queries;
    }

    public async Task<List<string>> GetFailedRetrievalKeywordsAsync(string sessionId, int lookback = 20)
    {
        var queries = await GetRecentAccessQueriesAsync(sessionId, limit: lookback);
        if (queries.Count == 0)
            return [];

        var gapKeywords = new HashSet<string>();

        foreach (var query in queries)
        {
            await using var cmd = Db.CreateCommand();
            cmd.CommandText =
                """
                SELECT m.tier FROM memory_access_log mal
                JOIN memories m ON mal.memory_id = m.id
                WHERE mal.query = $q AND mal.session_id = $sid
                """;
            cmd.Parameters.AddWithValue("$q", query);
            cmd.Parameters.AddWithValue("$sid", sessionId);

            var tiers = new List<string>();
            await using var reader = await cmd.ExecuteReaderAsync();
            while (await reader.ReadAsync())
                tiers.Add(reader.GetString(0));

            // If no results or only cold-tier, this is a gap
            if (tiers.Count == 0 || tiers.All(t => t == "cold"))
            {
                var words = query.Split(' ', StringSplitOptions.RemoveEmptyEntries)
                    .Where(w => w.Length > 2)
                    .Select(w => w.ToLowerInvariant().Trim());
                foreach (var w in words)
                    gapKeywords.Add(w);
            }
        }

        return gapKeywords.ToList();
    }

    // ── Save decisions ──

    public async Task LogSaveDecisionAsync(SaveDecision dec)
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText =
            "INSERT INTO save_decisions " +
            "(id, raw_log_id, session_id, turn, decided_at, decision, reason, confidence, gap_triggered, threshold_used) " +
            "VALUES ($id, $rlid, $sid, $turn, $dat, $dec, $reason, $conf, $gap, $thresh)";
        cmd.Parameters.AddWithValue("$id", dec.Id);
        cmd.Parameters.AddWithValue("$rlid", dec.RawLogId);
        cmd.Parameters.AddWithValue("$sid", dec.SessionId);
        cmd.Parameters.AddWithValue("$turn", dec.Turn);
        cmd.Parameters.AddWithValue("$dat", dec.DecidedAt);
        cmd.Parameters.AddWithValue("$dec", dec.Decision);
        cmd.Parameters.AddWithValue("$reason", (object?)dec.Reason ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$conf", dec.Confidence);
        cmd.Parameters.AddWithValue("$gap", dec.GapTriggered ? 1 : 0);
        cmd.Parameters.AddWithValue("$thresh", (object?)dec.ThresholdUsed ?? DBNull.Value);
        await cmd.ExecuteNonQueryAsync();
    }

    public async Task UpdateSaveOutcomeAsync(string decisionId, bool useful, string assessedAt)
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText =
            "UPDATE save_decisions SET outcome_useful = $u, outcome_assessed_at = $at WHERE id = $id";
        cmd.Parameters.AddWithValue("$u", useful ? 1 : 0);
        cmd.Parameters.AddWithValue("$at", assessedAt);
        cmd.Parameters.AddWithValue("$id", decisionId);
        await cmd.ExecuteNonQueryAsync();
    }

    public async Task<List<Dictionary<string, object?>>> GetUnassessedSaveDecisionsAsync(int lookbackDays = 30)
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText =
            """
            SELECT sd.id, sd.raw_log_id, m.id as memory_id, m.access_count
            FROM save_decisions sd
            LEFT JOIN memories m ON m.raw_log_id = sd.raw_log_id
            WHERE sd.decision IN ('save', 'fast_path')
              AND sd.id NOT IN (
                  SELECT id FROM save_decisions WHERE outcome_useful IS NOT NULL
              )
              AND sd.decided_at < datetime('now', $lookback)
            """;
        cmd.Parameters.AddWithValue("$lookback", $"-{lookbackDays} days");

        var rows = new List<Dictionary<string, object?>>();
        await using var reader = await cmd.ExecuteReaderAsync();
        while (await reader.ReadAsync())
        {
            rows.Add(new Dictionary<string, object?>
            {
                ["id"] = reader.IsDBNull(reader.GetOrdinal("id")) ? null : reader.GetString(reader.GetOrdinal("id")),
                ["raw_log_id"] = reader.IsDBNull(reader.GetOrdinal("raw_log_id")) ? null : reader.GetString(reader.GetOrdinal("raw_log_id")),
                ["memory_id"] = reader.IsDBNull(reader.GetOrdinal("memory_id")) ? null : reader.GetString(reader.GetOrdinal("memory_id")),
                ["access_count"] = reader.IsDBNull(reader.GetOrdinal("access_count")) ? null : (object)reader.GetInt32(reader.GetOrdinal("access_count")),
            });
        }
        return rows;
    }

    // ── Retrieval decisions (A4) ──

    public async Task LogRetrievalDecisionAsync(
        string decisionId, string sessionId, int? turn,
        string query, string decidedAt, List<string> layersQueried,
        int graphDepth, double moodWeight, int topK,
        List<string> memoryIds, int returnCount)
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText =
            "INSERT INTO retrieval_decisions " +
            "(id, session_id, turn, query, decided_at, layers_queried, graph_depth, " +
            "mood_weight, top_k, memories_returned, return_count) " +
            "VALUES ($id, $sid, $turn, $q, $dat, $layers, $gd, $mw, $tk, $mids, $rc)";
        cmd.Parameters.AddWithValue("$id", decisionId);
        cmd.Parameters.AddWithValue("$sid", sessionId);
        cmd.Parameters.AddWithValue("$turn", (object?)turn ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$q", query);
        cmd.Parameters.AddWithValue("$dat", decidedAt);
        cmd.Parameters.AddWithValue("$layers", JsonSerializer.Serialize(layersQueried));
        cmd.Parameters.AddWithValue("$gd", graphDepth);
        cmd.Parameters.AddWithValue("$mw", moodWeight);
        cmd.Parameters.AddWithValue("$tk", topK);
        cmd.Parameters.AddWithValue("$mids", JsonSerializer.Serialize(memoryIds));
        cmd.Parameters.AddWithValue("$rc", returnCount);
        await cmd.ExecuteNonQueryAsync();
    }

    public async Task<List<Dictionary<string, object?>>> GetUnassessedRetrievalDecisionsAsync()
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText =
            """
            SELECT id, session_id, turn, query
            FROM retrieval_decisions
            WHERE outcome_helpful IS NULL
              AND decided_at < datetime('now', '-1 hour')
            """;

        var rows = new List<Dictionary<string, object?>>();
        await using var reader = await cmd.ExecuteReaderAsync();
        while (await reader.ReadAsync())
        {
            rows.Add(new Dictionary<string, object?>
            {
                ["id"] = reader.GetString(reader.GetOrdinal("id")),
                ["session_id"] = reader.GetString(reader.GetOrdinal("session_id")),
                ["turn"] = reader.IsDBNull(reader.GetOrdinal("turn")) ? null : (object)reader.GetInt32(reader.GetOrdinal("turn")),
                ["query"] = reader.GetString(reader.GetOrdinal("query")),
            });
        }
        return rows;
    }

    public async Task UpdateRetrievalOutcomeAsync(string decisionId, bool helpful, string assessedAt)
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText =
            "UPDATE retrieval_decisions SET outcome_helpful = $h, outcome_assessed_at = $at WHERE id = $id";
        cmd.Parameters.AddWithValue("$h", helpful ? 1 : 0);
        cmd.Parameters.AddWithValue("$at", assessedAt);
        cmd.Parameters.AddWithValue("$id", decisionId);
        await cmd.ExecuteNonQueryAsync();
    }

    public async Task<List<string>> GetRetrievalFollowupsAsync(
        string sessionId, int turn, int window = 3)
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText =
            "SELECT query FROM retrieval_decisions WHERE session_id = $sid AND turn > $t1 AND turn <= $t2";
        cmd.Parameters.AddWithValue("$sid", sessionId);
        cmd.Parameters.AddWithValue("$t1", turn);
        cmd.Parameters.AddWithValue("$t2", turn + window);

        var queries = new List<string>();
        await using var reader = await cmd.ExecuteReaderAsync();
        while (await reader.ReadAsync())
            queries.Add(reader.GetString(0));
        return queries;
    }

    // ── Dream exploration logging (A3) ──

    public async Task LogDreamRunAsync(
        string runId, string ranAt, int nWalks,
        int edgesDiscovered, int edgesCommitted,
        List<string> strategies, string? notes = null)
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText =
            "INSERT INTO dream_exploration_runs " +
            "(id, ran_at, n_walks, edges_discovered, edges_committed, strategies_used, notes) " +
            "VALUES ($id, $rat, $nw, $ed, $ec, $strat, $notes)";
        cmd.Parameters.AddWithValue("$id", runId);
        cmd.Parameters.AddWithValue("$rat", ranAt);
        cmd.Parameters.AddWithValue("$nw", nWalks);
        cmd.Parameters.AddWithValue("$ed", edgesDiscovered);
        cmd.Parameters.AddWithValue("$ec", edgesCommitted);
        cmd.Parameters.AddWithValue("$strat", JsonSerializer.Serialize(strategies));
        cmd.Parameters.AddWithValue("$notes", (object?)notes ?? DBNull.Value);
        await cmd.ExecuteNonQueryAsync();
    }

    public async Task LogDreamEdgeAsync(
        string edgeId, string runId, string sourceId, string targetId,
        double similarity, string relationshipType, string discoveryMethod,
        bool committed = false)
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText =
            "INSERT INTO dream_discovered_edges " +
            "(id, exploration_run_id, source_memory_id, target_memory_id, " +
            "similarity, relationship_type, discovery_method, committed) " +
            "VALUES ($id, $rid, $src, $tgt, $sim, $rtype, $dm, $c)";
        cmd.Parameters.AddWithValue("$id", edgeId);
        cmd.Parameters.AddWithValue("$rid", runId);
        cmd.Parameters.AddWithValue("$src", sourceId);
        cmd.Parameters.AddWithValue("$tgt", targetId);
        cmd.Parameters.AddWithValue("$sim", similarity);
        cmd.Parameters.AddWithValue("$rtype", relationshipType);
        cmd.Parameters.AddWithValue("$dm", discoveryMethod);
        cmd.Parameters.AddWithValue("$c", committed ? 1 : 0);
        await cmd.ExecuteNonQueryAsync();
    }

    // ── Compaction ──

    public async Task LogCompactionRunAsync(CompactionResult result)
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText =
            "INSERT INTO compaction_runs " +
            "(id, ran_at, trigger, memories_reviewed, memories_merged, memories_pruned, " +
            "notes, keywords_updated, edges_discovered) " +
            "VALUES ($id, $rat, $trigger, $mr, $mm, $mp, $notes, $ku, $ed)";
        cmd.Parameters.AddWithValue("$id", result.Id);
        cmd.Parameters.AddWithValue("$rat", result.RanAt);
        cmd.Parameters.AddWithValue("$trigger", result.Trigger);
        cmd.Parameters.AddWithValue("$mr", result.MemoriesReviewed);
        cmd.Parameters.AddWithValue("$mm", result.MemoriesMerged);
        cmd.Parameters.AddWithValue("$mp", result.MemoriesPruned);
        cmd.Parameters.AddWithValue("$notes", (object?)result.Notes ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$ku", result.KeywordsUpdated);
        cmd.Parameters.AddWithValue("$ed", result.EdgesDiscovered);
        await cmd.ExecuteNonQueryAsync();
    }

    public async Task LogCompactionMergeAsync(
        string compactionId, List<string> sourceIds, string resultingId,
        bool? validationPassed = null, double? avgSourceScore = null,
        double? avgMergedScore = null, double? degradation = null)
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText =
            "INSERT INTO compaction_merges " +
            "(compaction_id, source_memory_ids, resulting_memory_id, " +
            "validation_passed, avg_source_score, avg_merged_score, degradation) " +
            "VALUES ($cid, $src, $res, $vp, $ass, $ams, $deg)";
        cmd.Parameters.AddWithValue("$cid", compactionId);
        cmd.Parameters.AddWithValue("$src", JsonSerializer.Serialize(sourceIds));
        cmd.Parameters.AddWithValue("$res", resultingId);
        cmd.Parameters.AddWithValue("$vp", validationPassed.HasValue ? (object)(validationPassed.Value ? 1 : 0) : DBNull.Value);
        cmd.Parameters.AddWithValue("$ass", (object?)avgSourceScore ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$ams", (object?)avgMergedScore ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$deg", (object?)degradation ?? DBNull.Value);
        await cmd.ExecuteNonQueryAsync();
    }

    public async Task<List<Memory>> GetCompactionCandidatesAsync(double threshold = 0.7)
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText =
            """
            SELECT m.* FROM memories m
            LEFT JOIN (
                SELECT memory_id, COUNT(*) as edge_count
                FROM memory_keywords
                GROUP BY memory_id
            ) kc ON m.id = kc.memory_id
            WHERE m.tier = 'hot'
              AND m.fast_pathed = 0
              AND NOT (m.compaction_gen = 0 AND m.access_count > 5)
            ORDER BY ((1 - m.decay_score) * 0.6 + (1 - m.salience) * 0.4) DESC
            """;

        var candidates = new List<Memory>();
        await using var reader = await cmd.ExecuteReaderAsync();
        while (await reader.ReadAsync())
        {
            var mem = RowToMemory(reader);
            var score = (1.0 - mem.DecayScore) * 0.6 + (1.0 - mem.Salience) * 0.4;
            if (score > threshold)
                candidates.Add(mem);
        }

        // Load keywords
        if (candidates.Count > 0)
        {
            var kwMap = await BatchLoadKeywordsAsync(candidates.Select(m => m.Id).ToList());
            foreach (var m in candidates)
                m.Keywords = kwMap.GetValueOrDefault(m.Id, []);
        }

        return candidates;
    }

    // ── Policy data export (A4.4) ──

    public async Task<List<Dictionary<string, object?>>> ExportSavePolicyDataAsync()
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText =
            """
            SELECT sd.confidence, sd.decision, sd.gap_triggered,
                   m.valence, m.arousal, m.surprise, m.salience,
                   sd.outcome_useful
            FROM save_decisions sd
            LEFT JOIN memories m ON m.raw_log_id = sd.raw_log_id
            WHERE sd.outcome_useful IS NOT NULL
            """;

        var rows = new List<Dictionary<string, object?>>();
        await using var reader = await cmd.ExecuteReaderAsync();
        while (await reader.ReadAsync())
            rows.Add(ReadNullableRow(reader));
        return rows;
    }

    public async Task<List<Dictionary<string, object?>>> ExportRetrievalPolicyDataAsync()
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText =
            """
            SELECT layers_queried, graph_depth, mood_weight, top_k,
                   return_count, outcome_helpful
            FROM retrieval_decisions
            WHERE outcome_helpful IS NOT NULL
            """;

        var rows = new List<Dictionary<string, object?>>();
        await using var reader = await cmd.ExecuteReaderAsync();
        while (await reader.ReadAsync())
            rows.Add(ReadNullableRow(reader));
        return rows;
    }

    public async Task<List<Dictionary<string, object>>> GetMemoriesWithVectorsAsync(
        List<string>? tiers = null)
    {
        var tierClause = "";
        var parameters = new List<SqliteParameter>();
        if (tiers is not null && tiers.Count > 0)
        {
            var placeholders = string.Join(",", tiers.Select((_, i) => $"$t{i}"));
            tierClause = $"AND tier IN ({placeholders})";
            for (var i = 0; i < tiers.Count; i++)
                parameters.Add(new SqliteParameter($"$t{i}", tiers[i]));
        }

        await using var cmd = Db.CreateCommand();
        cmd.CommandText = $"SELECT id, session_id, vector_id FROM memories WHERE vector_id IS NOT NULL {tierClause}";
        foreach (var p in parameters)
            cmd.Parameters.Add(p);

        var rows = new List<Dictionary<string, object>>();
        await using var reader = await cmd.ExecuteReaderAsync();
        while (await reader.ReadAsync())
        {
            rows.Add(new Dictionary<string, object>
            {
                ["id"] = reader.GetString(reader.GetOrdinal("id")),
                ["session_id"] = reader.GetString(reader.GetOrdinal("session_id")),
                ["vector_id"] = reader.GetString(reader.GetOrdinal("vector_id")),
            });
        }
        return rows;
    }

    /// <summary>Find memory ID by raw_log_id.</summary>
    public async Task<string?> FindMemoryIdByRawLogIdAsync(string rawLogId)
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText = "SELECT id FROM memories WHERE raw_log_id = $rlid";
        cmd.Parameters.AddWithValue("$rlid", rawLogId);
        await using var reader = await cmd.ExecuteReaderAsync();
        if (await reader.ReadAsync())
            return reader.GetString(0);
        return null;
    }

    // ── Helpers ──

    private async Task<List<(string Keyword, double Weight)>> LoadKeywordsAsync(string memoryId)
    {
        await using var cmd = Db.CreateCommand();
        cmd.CommandText = "SELECT keyword, weight FROM memory_keywords WHERE memory_id = $mid";
        cmd.Parameters.AddWithValue("$mid", memoryId);

        var kws = new List<(string, double)>();
        await using var reader = await cmd.ExecuteReaderAsync();
        while (await reader.ReadAsync())
            kws.Add((reader.GetString(0), reader.GetDouble(1)));
        return kws;
    }

    private async Task<Dictionary<string, List<(string Keyword, double Weight)>>> BatchLoadKeywordsAsync(
        List<string> ids)
    {
        var kwMap = ids.ToDictionary(id => id, _ => new List<(string, double)>());
        if (ids.Count == 0)
            return kwMap;

        var placeholders = string.Join(",", ids.Select((_, i) => $"$id{i}"));
        await using var cmd = Db.CreateCommand();
        cmd.CommandText =
            $"SELECT memory_id, keyword, weight FROM memory_keywords WHERE memory_id IN ({placeholders})";
        for (var i = 0; i < ids.Count; i++)
            cmd.Parameters.AddWithValue($"$id{i}", ids[i]);

        await using var reader = await cmd.ExecuteReaderAsync();
        while (await reader.ReadAsync())
        {
            var mid = reader.GetString(0);
            if (kwMap.TryGetValue(mid, out var list))
                list.Add((reader.GetString(1), reader.GetDouble(2)));
        }

        return kwMap;
    }

    private static Memory RowToMemory(SqliteDataReader reader)
    {
        return new Memory
        {
            Id = reader.GetString(reader.GetOrdinal("id")),
            CreatedAt = reader.GetString(reader.GetOrdinal("created_at")),
            UpdatedAt = reader.GetString(reader.GetOrdinal("updated_at")),
            Content = reader.GetString(reader.GetOrdinal("content")),
            Summary = reader.IsDBNull(reader.GetOrdinal("summary")) ? null : reader.GetString(reader.GetOrdinal("summary")),
            RawLogId = reader.GetString(reader.GetOrdinal("raw_log_id")),
            SessionId = reader.GetString(reader.GetOrdinal("session_id")),
            Turn = reader.GetInt32(reader.GetOrdinal("turn")),
            Valence = reader.IsDBNull(reader.GetOrdinal("valence")) ? 0.0 : reader.GetDouble(reader.GetOrdinal("valence")),
            Arousal = reader.IsDBNull(reader.GetOrdinal("arousal")) ? 0.0 : reader.GetDouble(reader.GetOrdinal("arousal")),
            Surprise = reader.IsDBNull(reader.GetOrdinal("surprise")) ? 0.0 : reader.GetDouble(reader.GetOrdinal("surprise")),
            Salience = reader.IsDBNull(reader.GetOrdinal("salience")) ? 0.5 : reader.GetDouble(reader.GetOrdinal("salience")),
            AccessCount = reader.IsDBNull(reader.GetOrdinal("access_count")) ? 0 : reader.GetInt32(reader.GetOrdinal("access_count")),
            LastAccessed = reader.IsDBNull(reader.GetOrdinal("last_accessed")) ? null : reader.GetString(reader.GetOrdinal("last_accessed")),
            DecayScore = reader.IsDBNull(reader.GetOrdinal("decay_score")) ? 1.0 : reader.GetDouble(reader.GetOrdinal("decay_score")),
            CompactionGen = reader.IsDBNull(reader.GetOrdinal("compaction_gen")) ? 0 : reader.GetInt32(reader.GetOrdinal("compaction_gen")),
            Tier = reader.IsDBNull(reader.GetOrdinal("tier")) ? "hot" : reader.GetString(reader.GetOrdinal("tier")),
            FastPathed = !reader.IsDBNull(reader.GetOrdinal("fast_pathed")) && reader.GetInt32(reader.GetOrdinal("fast_pathed")) != 0,
            IsSemantic = !reader.IsDBNull(reader.GetOrdinal("is_semantic")) && reader.GetInt32(reader.GetOrdinal("is_semantic")) != 0,
            GraphNodeId = reader.IsDBNull(reader.GetOrdinal("graph_node_id")) ? null : reader.GetString(reader.GetOrdinal("graph_node_id")),
            VectorId = reader.IsDBNull(reader.GetOrdinal("vector_id")) ? null : reader.GetString(reader.GetOrdinal("vector_id")),
            SpatialEmbedding = reader.IsDBNull(reader.GetOrdinal("spatial_embedding")) ? null : (byte[])reader.GetValue(reader.GetOrdinal("spatial_embedding")),
            SceneDescription = reader.IsDBNull(reader.GetOrdinal("scene_description")) ? null : reader.GetString(reader.GetOrdinal("scene_description")),
        };
    }

    private static Dictionary<string, object> ReadRow(SqliteDataReader reader)
    {
        var dict = new Dictionary<string, object>();
        for (var i = 0; i < reader.FieldCount; i++)
        {
            dict[reader.GetName(i)] = reader.IsDBNull(i) ? null! : reader.GetValue(i);
        }
        return dict;
    }

    private static Dictionary<string, object?> ReadNullableRow(SqliteDataReader reader)
    {
        var dict = new Dictionary<string, object?>();
        for (var i = 0; i < reader.FieldCount; i++)
        {
            dict[reader.GetName(i)] = reader.IsDBNull(i) ? null : reader.GetValue(i);
        }
        return dict;
    }

    // ── Schema ──

    private const string Schema =
        """
        CREATE TABLE IF NOT EXISTS raw_log_index (
            id          TEXT PRIMARY KEY,
            session_id  TEXT NOT NULL,
            turn        INTEGER NOT NULL,
            timestamp   TEXT NOT NULL,
            file_path   TEXT NOT NULL,
            byte_offset INTEGER
        );

        CREATE TABLE IF NOT EXISTS memories (
            id                TEXT PRIMARY KEY,
            created_at        TEXT NOT NULL,
            updated_at        TEXT NOT NULL,
            content           TEXT NOT NULL,
            summary           TEXT,
            raw_log_id        TEXT NOT NULL,
            session_id        TEXT NOT NULL,
            turn              INTEGER NOT NULL,
            valence           REAL,
            arousal           REAL,
            surprise          REAL,
            salience          REAL DEFAULT 0.5,
            access_count      INTEGER DEFAULT 0,
            last_accessed     TEXT,
            decay_score       REAL DEFAULT 1.0,
            compaction_gen    INTEGER DEFAULT 0,
            tier              TEXT DEFAULT 'hot',
            fast_pathed       INTEGER DEFAULT 0,
            is_semantic       INTEGER DEFAULT 0,
            graph_node_id     TEXT,
            vector_id         TEXT,
            spatial_embedding BLOB,
            scene_description TEXT,
            FOREIGN KEY (raw_log_id) REFERENCES raw_log_index(id)
        );

        CREATE TABLE IF NOT EXISTS memory_keywords (
            memory_id   TEXT NOT NULL,
            keyword     TEXT NOT NULL,
            weight      REAL DEFAULT 1.0,
            PRIMARY KEY (memory_id, keyword),
            FOREIGN KEY (memory_id) REFERENCES memories(id)
        );

        CREATE INDEX IF NOT EXISTS idx_keyword ON memory_keywords(keyword);

        CREATE TABLE IF NOT EXISTS memory_access_log (
            id          TEXT PRIMARY KEY,
            memory_id   TEXT NOT NULL,
            accessed_at TEXT NOT NULL,
            access_type TEXT NOT NULL,
            session_id  TEXT,
            query       TEXT,
            FOREIGN KEY (memory_id) REFERENCES memories(id)
        );

        CREATE TABLE IF NOT EXISTS compaction_runs (
            id                  TEXT PRIMARY KEY,
            ran_at              TEXT NOT NULL,
            trigger             TEXT,
            memories_reviewed   INTEGER,
            memories_merged     INTEGER,
            memories_pruned     INTEGER,
            notes               TEXT,
            keywords_updated    INTEGER DEFAULT 0,
            edges_discovered    INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS compaction_merges (
            compaction_id         TEXT NOT NULL,
            source_memory_ids     TEXT NOT NULL,
            resulting_memory_id   TEXT NOT NULL,
            validation_passed     INTEGER,
            avg_source_score      REAL,
            avg_merged_score      REAL,
            degradation           REAL,
            FOREIGN KEY (compaction_id) REFERENCES compaction_runs(id)
        );

        CREATE TABLE IF NOT EXISTS save_decisions (
            id                  TEXT PRIMARY KEY,
            raw_log_id          TEXT NOT NULL,
            session_id          TEXT NOT NULL,
            turn                INTEGER NOT NULL,
            decided_at          TEXT NOT NULL,
            decision            TEXT NOT NULL,
            reason              TEXT,
            confidence          REAL,
            gap_triggered       INTEGER DEFAULT 0,
            threshold_used      REAL,
            outcome_useful      INTEGER,
            outcome_assessed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS retrieval_decisions (
            id                  TEXT PRIMARY KEY,
            session_id          TEXT NOT NULL,
            turn                INTEGER,
            query               TEXT NOT NULL,
            decided_at          TEXT NOT NULL,
            layers_queried      TEXT NOT NULL,
            graph_depth         INTEGER,
            mood_weight         REAL,
            top_k               INTEGER,
            memories_returned   TEXT NOT NULL,
            return_count        INTEGER NOT NULL,
            outcome_helpful     INTEGER,
            outcome_assessed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS dream_exploration_runs (
            id                TEXT PRIMARY KEY,
            ran_at            TEXT NOT NULL,
            n_walks           INTEGER,
            edges_discovered  INTEGER,
            edges_committed   INTEGER,
            strategies_used   TEXT,
            notes             TEXT
        );

        CREATE TABLE IF NOT EXISTS dream_discovered_edges (
            id                  TEXT PRIMARY KEY,
            exploration_run_id  TEXT NOT NULL,
            source_memory_id    TEXT NOT NULL,
            target_memory_id    TEXT NOT NULL,
            similarity          REAL,
            relationship_type   TEXT,
            discovery_method    TEXT,
            committed           INTEGER DEFAULT 0,
            FOREIGN KEY (exploration_run_id) REFERENCES dream_exploration_runs(id)
        );
        """;
}
