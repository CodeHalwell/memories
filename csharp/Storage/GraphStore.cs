// Graph store using SQLite with recursive CTEs for memory relationships.
// Encodes RELATES_TO, MENTIONS, and EVOLVED_FROM edges.

using Microsoft.Data.Sqlite;
using Microsoft.Extensions.Logging;

namespace AgentMemory.Storage;

/// <summary>
/// SQLite-backed graph store for memory relationships.
/// Uses separate tables for Memory nodes, Entity nodes, and edge types,
/// with recursive CTEs for traversal.
/// </summary>
public sealed class GraphStore : IDisposable
{
    private readonly string _dbPath;
    private SqliteConnection? _conn;
    private readonly ILogger<GraphStore>? _logger;

    public GraphStore(string? graphDir = null, ILogger<GraphStore>? logger = null)
    {
        var dir = graphDir ?? new MemoryConfig().GraphDir;
        _dbPath = Path.Combine(dir, "graph.db");
        _logger = logger;
    }

    public void Initialize()
    {
        var dir = Path.GetDirectoryName(_dbPath);
        if (!string.IsNullOrEmpty(dir))
            Directory.CreateDirectory(dir);

        _conn = new SqliteConnection($"Data Source={_dbPath}");
        _conn.Open();
        CreateSchema();
    }

    public void Close()
    {
        _conn?.Dispose();
        _conn = null;
    }

    public void Dispose() => Close();

    private SqliteConnection Conn =>
        _conn ?? throw new InvalidOperationException("GraphStore not initialized — call Initialize() first");

    private void CreateSchema()
    {
        using var cmd = Conn.CreateCommand();
        cmd.CommandText =
            """
            CREATE TABLE IF NOT EXISTS memory_nodes (
                id TEXT PRIMARY KEY,
                summary TEXT,
                tier TEXT,
                salience REAL,
                valence REAL,
                compaction_gen INTEGER,
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS entity_nodes (
                id TEXT PRIMARY KEY,
                name TEXT,
                type TEXT
            );

            CREATE TABLE IF NOT EXISTS relates_to (
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                weight REAL,
                relationship_type TEXT,
                created_at TEXT,
                FOREIGN KEY (from_id) REFERENCES memory_nodes(id),
                FOREIGN KEY (to_id) REFERENCES memory_nodes(id)
            );

            CREATE TABLE IF NOT EXISTS mentions (
                memory_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                weight REAL,
                FOREIGN KEY (memory_id) REFERENCES memory_nodes(id),
                FOREIGN KEY (entity_id) REFERENCES entity_nodes(id)
            );

            CREATE TABLE IF NOT EXISTS evolved_from (
                new_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                compaction_id TEXT,
                created_at TEXT,
                FOREIGN KEY (new_id) REFERENCES memory_nodes(id),
                FOREIGN KEY (source_id) REFERENCES memory_nodes(id)
            );

            CREATE INDEX IF NOT EXISTS idx_relates_to_from ON relates_to(from_id);
            CREATE INDEX IF NOT EXISTS idx_relates_to_to ON relates_to(to_id);
            CREATE INDEX IF NOT EXISTS idx_evolved_from_new ON evolved_from(new_id);
            CREATE INDEX IF NOT EXISTS idx_evolved_from_src ON evolved_from(source_id);
            CREATE INDEX IF NOT EXISTS idx_mentions_mem ON mentions(memory_id);
            """;
        cmd.ExecuteNonQuery();
    }

    // ── Node operations ──

    public void AddMemoryNode(
        string memoryId, string summary, string tier = "hot",
        double salience = 0.5, double valence = 0.0,
        int compactionGen = 0, string createdAt = "")
    {
        using var cmd = Conn.CreateCommand();
        cmd.CommandText =
            """
            INSERT INTO memory_nodes (id, summary, tier, salience, valence, compaction_gen, created_at)
            VALUES ($id, $summary, $tier, $salience, $valence, $cg, $cat)
            ON CONFLICT(id) DO UPDATE SET
                summary = excluded.summary,
                tier = excluded.tier,
                salience = excluded.salience,
                valence = excluded.valence,
                compaction_gen = excluded.compaction_gen,
                created_at = excluded.created_at
            """;
        cmd.Parameters.AddWithValue("$id", memoryId);
        cmd.Parameters.AddWithValue("$summary", summary);
        cmd.Parameters.AddWithValue("$tier", tier);
        cmd.Parameters.AddWithValue("$salience", salience);
        cmd.Parameters.AddWithValue("$valence", valence);
        cmd.Parameters.AddWithValue("$cg", compactionGen);
        cmd.Parameters.AddWithValue("$cat", createdAt);
        cmd.ExecuteNonQuery();
    }

    public void AddEntityNode(string entityId, string name, string entityType)
    {
        using var cmd = Conn.CreateCommand();
        cmd.CommandText =
            """
            INSERT INTO entity_nodes (id, name, type)
            VALUES ($id, $name, $type)
            ON CONFLICT(id) DO UPDATE SET name = excluded.name, type = excluded.type
            """;
        cmd.Parameters.AddWithValue("$id", entityId);
        cmd.Parameters.AddWithValue("$name", name);
        cmd.Parameters.AddWithValue("$type", entityType);
        cmd.ExecuteNonQuery();
    }

    // ── Edge operations ──

    public void AddRelatesTo(
        string fromId, string toId, double weight = 1.0,
        string relationshipType = "supports", string createdAt = "")
    {
        using var cmd = Conn.CreateCommand();
        cmd.CommandText =
            "INSERT INTO relates_to (from_id, to_id, weight, relationship_type, created_at) " +
            "VALUES ($from, $to, $w, $rtype, $cat)";
        cmd.Parameters.AddWithValue("$from", fromId);
        cmd.Parameters.AddWithValue("$to", toId);
        cmd.Parameters.AddWithValue("$w", weight);
        cmd.Parameters.AddWithValue("$rtype", relationshipType);
        cmd.Parameters.AddWithValue("$cat", createdAt);
        cmd.ExecuteNonQuery();
    }

    public void AddMentions(string memoryId, string entityId, double weight = 1.0)
    {
        using var cmd = Conn.CreateCommand();
        cmd.CommandText =
            "INSERT INTO mentions (memory_id, entity_id, weight) VALUES ($mid, $eid, $w)";
        cmd.Parameters.AddWithValue("$mid", memoryId);
        cmd.Parameters.AddWithValue("$eid", entityId);
        cmd.Parameters.AddWithValue("$w", weight);
        cmd.ExecuteNonQuery();
    }

    public void AddEvolvedFrom(
        string newId, string sourceId, string compactionId = "", string createdAt = "")
    {
        using var cmd = Conn.CreateCommand();
        cmd.CommandText =
            "INSERT INTO evolved_from (new_id, source_id, compaction_id, created_at) " +
            "VALUES ($nid, $sid, $cid, $cat)";
        cmd.Parameters.AddWithValue("$nid", newId);
        cmd.Parameters.AddWithValue("$sid", sourceId);
        cmd.Parameters.AddWithValue("$cid", compactionId);
        cmd.Parameters.AddWithValue("$cat", createdAt);
        cmd.ExecuteNonQuery();
    }

    // ── Queries ──

    /// <summary>
    /// Traverse RELATES_TO edges up to maxDepth hops from a memory node using recursive CTE.
    /// </summary>
    public List<Dictionary<string, object>> GetRelatedMemories(
        string memoryId, int maxDepth = 2, double minWeight = 0.0)
    {
        using var cmd = Conn.CreateCommand();
        cmd.CommandText = $"""
            WITH RECURSIVE traversal(id, depth) AS (
                SELECT to_id, 1 FROM relates_to WHERE from_id = $id AND weight >= $mw
                UNION
                SELECT rt.to_id, t.depth + 1
                FROM relates_to rt
                JOIN traversal t ON rt.from_id = t.id
                WHERE t.depth < $maxd AND rt.weight >= $mw
            )
            SELECT DISTINCT mn.id, mn.summary, mn.tier, mn.salience, t.depth
            FROM traversal t
            JOIN memory_nodes mn ON mn.id = t.id
            WHERE mn.id != $id
            """;
        cmd.Parameters.AddWithValue("$id", memoryId);
        cmd.Parameters.AddWithValue("$mw", minWeight);
        cmd.Parameters.AddWithValue("$maxd", maxDepth);

        var rows = new List<Dictionary<string, object>>();
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            rows.Add(new Dictionary<string, object>
            {
                ["id"] = reader.GetString(0),
                ["summary"] = reader.IsDBNull(1) ? "" : reader.GetString(1),
                ["tier"] = reader.IsDBNull(2) ? "hot" : reader.GetString(2),
                ["salience"] = reader.IsDBNull(3) ? 0.5 : reader.GetDouble(3),
                ["depth"] = reader.GetInt32(4),
            });
        }
        return rows;
    }

    /// <summary>Get all entities mentioned by a memory.</summary>
    public List<Dictionary<string, object>> GetMemoryEntities(string memoryId)
    {
        using var cmd = Conn.CreateCommand();
        cmd.CommandText =
            "SELECT en.id, en.name, en.type, m.weight " +
            "FROM mentions m " +
            "JOIN entity_nodes en ON en.id = m.entity_id " +
            "WHERE m.memory_id = $id";
        cmd.Parameters.AddWithValue("$id", memoryId);

        var rows = new List<Dictionary<string, object>>();
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            rows.Add(new Dictionary<string, object>
            {
                ["id"] = reader.GetString(0),
                ["name"] = reader.IsDBNull(1) ? "" : reader.GetString(1),
                ["type"] = reader.IsDBNull(2) ? "" : reader.GetString(2),
                ["weight"] = reader.IsDBNull(3) ? 1.0 : reader.GetDouble(3),
            });
        }
        return rows;
    }

    /// <summary>Trace the full lineage of a compacted memory back to originals using recursive CTE.</summary>
    public List<Dictionary<string, object>> GetEvolutionLineage(string memoryId)
    {
        using var cmd = Conn.CreateCommand();
        cmd.CommandText =
            """
            WITH RECURSIVE lineage(id, depth) AS (
                SELECT source_id, 1 FROM evolved_from WHERE new_id = $id
                UNION
                SELECT ef.source_id, l.depth + 1
                FROM evolved_from ef
                JOIN lineage l ON ef.new_id = l.id
                WHERE l.depth < 10
            )
            SELECT mn.id, mn.summary, mn.compaction_gen, l.depth
            FROM lineage l
            JOIN memory_nodes mn ON mn.id = l.id
            ORDER BY l.depth
            """;
        cmd.Parameters.AddWithValue("$id", memoryId);

        var rows = new List<Dictionary<string, object>>();
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            rows.Add(new Dictionary<string, object>
            {
                ["id"] = reader.GetString(0),
                ["summary"] = reader.IsDBNull(1) ? "" : reader.GetString(1),
                ["gen"] = reader.IsDBNull(2) ? 0 : reader.GetInt32(2),
                ["depth"] = reader.GetInt32(3),
            });
        }
        return rows;
    }

    /// <summary>Count RELATES_TO edges connected to a memory (both directions).</summary>
    public int GetEdgeCount(string memoryId)
    {
        using var cmd = Conn.CreateCommand();
        cmd.CommandText =
            "SELECT COUNT(*) FROM relates_to WHERE from_id = $id OR to_id = $id";
        cmd.Parameters.AddWithValue("$id", memoryId);
        return Convert.ToInt32(cmd.ExecuteScalar());
    }

    /// <summary>
    /// Copy all RELATES_TO edges from source memories to a new compacted memory node.
    /// Skips edges between source nodes (they are being merged).
    /// </summary>
    public void ReplicateEdgesToNewNode(List<string> sourceIds, string newId)
    {
        var srcSet = new HashSet<string>(sourceIds);

        foreach (var srcId in sourceIds)
        {
            // Outgoing edges
            using (var cmd = Conn.CreateCommand())
            {
                cmd.CommandText =
                    "SELECT to_id, weight, relationship_type, created_at " +
                    "FROM relates_to WHERE from_id = $id";
                cmd.Parameters.AddWithValue("$id", srcId);

                var outgoing = new List<(string TargetId, double Weight, string Rtype, string Cat)>();
                using var reader = cmd.ExecuteReader();
                while (reader.Read())
                {
                    outgoing.Add((
                        reader.GetString(0),
                        reader.IsDBNull(1) ? 1.0 : reader.GetDouble(1),
                        reader.IsDBNull(2) ? "supports" : reader.GetString(2),
                        reader.IsDBNull(3) ? "" : reader.GetString(3)
                    ));
                }

                foreach (var (targetId, weight, rtype, cat) in outgoing)
                {
                    if (!srcSet.Contains(targetId) && targetId != newId)
                    {
                        try { AddRelatesTo(newId, targetId, weight, rtype, cat); }
                        catch { /* dedup — edge may already exist */ }
                    }
                }
            }

            // Incoming edges
            using (var cmd = Conn.CreateCommand())
            {
                cmd.CommandText =
                    "SELECT from_id, weight, relationship_type, created_at " +
                    "FROM relates_to WHERE to_id = $id";
                cmd.Parameters.AddWithValue("$id", srcId);

                var incoming = new List<(string SourceId, double Weight, string Rtype, string Cat)>();
                using var reader = cmd.ExecuteReader();
                while (reader.Read())
                {
                    incoming.Add((
                        reader.GetString(0),
                        reader.IsDBNull(1) ? 1.0 : reader.GetDouble(1),
                        reader.IsDBNull(2) ? "supports" : reader.GetString(2),
                        reader.IsDBNull(3) ? "" : reader.GetString(3)
                    ));
                }

                foreach (var (sourceId, weight, rtype, cat) in incoming)
                {
                    if (!srcSet.Contains(sourceId) && sourceId != newId)
                    {
                        try { AddRelatesTo(sourceId, newId, weight, rtype, cat); }
                        catch { /* dedup */ }
                    }
                }
            }
        }
    }

    /// <summary>Check if a path exists between two memory nodes via RELATES_TO edges.</summary>
    public bool PathExists(string fromId, string toId, int maxHops = 2)
    {
        try
        {
            using var cmd = Conn.CreateCommand();
            cmd.CommandText = $"""
                WITH RECURSIVE traversal(id, depth) AS (
                    SELECT to_id, 1 FROM relates_to WHERE from_id = $from
                    UNION
                    SELECT rt.to_id, t.depth + 1
                    FROM relates_to rt
                    JOIN traversal t ON rt.from_id = t.id
                    WHERE t.depth < $maxd
                )
                SELECT COUNT(*) FROM traversal WHERE id = $to LIMIT 1
                """;
            cmd.Parameters.AddWithValue("$from", fromId);
            cmd.Parameters.AddWithValue("$to", toId);
            cmd.Parameters.AddWithValue("$maxd", maxHops);

            var count = Convert.ToInt32(cmd.ExecuteScalar());
            return count > 0;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>Update the tier of a memory node.</summary>
    public void UpdateMemoryTier(string memoryId, string tier)
    {
        using var cmd = Conn.CreateCommand();
        cmd.CommandText = "UPDATE memory_nodes SET tier = $tier WHERE id = $id";
        cmd.Parameters.AddWithValue("$tier", tier);
        cmd.Parameters.AddWithValue("$id", memoryId);
        cmd.ExecuteNonQuery();
    }
}
