// Raw JSONL log writer and indexer.
// Every agent output is appended to a session-specific JSONL file.
// This is the immutable ground truth — entries are never modified or deleted.

using System.Text.Json;

namespace AgentMemory.Storage;

/// <summary>Append-only JSONL logger for raw agent outputs.</summary>
public sealed class JsonlLogger
{
    private readonly string _logDir;

    public JsonlLogger(string? logDir = null)
    {
        _logDir = logDir ?? new MemoryConfig().LogDir;
        Directory.CreateDirectory(_logDir);
    }

    private string SessionPath(string sessionId)
    {
        var safeId = Path.GetFileName(sessionId);
        return Path.Combine(_logDir, $"{safeId}.jsonl");
    }

    /// <summary>Append an entry and return (filePath, byteOffset).</summary>
    public (string FilePath, long ByteOffset) Append(RawLogEntry entry)
    {
        var path = SessionPath(entry.SessionId);
        var line = JsonSerializer.Serialize(entry) + "\n";

        long byteOffset = 0;
        if (File.Exists(path))
        {
            var fi = new FileInfo(path);
            byteOffset = fi.Length;
        }

        File.AppendAllText(path, line, System.Text.Encoding.UTF8);
        return (path, byteOffset);
    }

    /// <summary>Read a single entry at the given byte offset.</summary>
    public RawLogEntry ReadEntry(string filePath, long byteOffset)
    {
        using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        fs.Seek(byteOffset, SeekOrigin.Begin);
        using var reader = new StreamReader(fs, System.Text.Encoding.UTF8);
        var line = reader.ReadLine()
            ?? throw new InvalidOperationException("No data at offset");
        return JsonSerializer.Deserialize<RawLogEntry>(line)
            ?? throw new InvalidOperationException("Failed to deserialize entry");
    }

    /// <summary>Yield all entries for a session in order.</summary>
    public IEnumerable<RawLogEntry> IterSession(string sessionId)
    {
        var path = SessionPath(sessionId);
        if (!File.Exists(path))
            yield break;

        foreach (var line in File.ReadLines(path))
        {
            var trimmed = line.Trim();
            if (string.IsNullOrEmpty(trimmed))
                continue;

            var entry = JsonSerializer.Deserialize<RawLogEntry>(trimmed);
            if (entry is not null)
                yield return entry;
        }
    }

    /// <summary>Simple text search within a session log.</summary>
    public List<RawLogEntry> Search(string sessionId, string text)
    {
        var results = new List<RawLogEntry>();
        var lowerText = text.ToLowerInvariant();

        foreach (var entry in IterSession(sessionId))
        {
            if (entry.Content.ToLowerInvariant().Contains(lowerText))
                results.Add(entry);
        }

        return results;
    }

    /// <summary>Return all session IDs that have log files.</summary>
    public List<string> ListSessions()
    {
        return Directory.GetFiles(_logDir, "*.jsonl")
            .OrderBy(f => f)
            .Select(f => Path.GetFileNameWithoutExtension(f))
            .ToList();
    }

    /// <summary>Return the file size in bytes for a session log.</summary>
    public long SessionSize(string sessionId)
    {
        var path = SessionPath(sessionId);
        if (!File.Exists(path))
            return 0;
        return new FileInfo(path).Length;
    }
}
