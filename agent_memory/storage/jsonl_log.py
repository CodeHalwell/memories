"""Raw JSONL log writer and indexer.

Every agent output is appended to a session-specific JSONL file. This is the
immutable ground truth — entries are never modified or deleted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict
from pathlib import Path

from agent_memory.config import LOG_DIR
from agent_memory.models import RawLogEntry


class JSONLLogger:
    """Append-only JSONL logger for raw agent outputs."""

    def __init__(self, log_dir: Path | None = None) -> None:
        self.log_dir = log_dir or LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str, namespace: str = "default") -> Path:
        """Path of a session's log file.

        The default namespace keeps the original flat layout; other namespaces
        get their own sub-directory so two tenants that reuse a session id never
        share a file (which also makes per-tenant purges possible).
        """
        # Sanitize ids to prevent path traversal
        safe_id = os.path.basename(session_id) or "_"
        if namespace == "default":
            return self.log_dir / f"{safe_id}.jsonl"
        ns_dir = self.namespace_dir(namespace)
        ns_dir.mkdir(parents=True, exist_ok=True)
        return ns_dir / f"{safe_id}.jsonl"

    def namespace_dir(self, namespace: str) -> Path:
        """Directory for a namespace's logs.

        A readable slug plus a hash of the exact namespace, so namespaces that
        slug identically ("tenant/a" and "tenant_a") never share a directory —
        a shared directory would let one tenant's purge delete another's logs.
        """
        if namespace == "default":
            return self.log_dir
        slug = re.sub(r"[^A-Za-z0-9._-]", "_", namespace)[:40] or "_"
        digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:12]
        return self.log_dir / f"ns-{slug}-{digest}"

    def append(self, entry: RawLogEntry) -> tuple[str, int]:
        """Append an entry and return (file_path, byte_offset)."""
        path = self._session_path(entry.session_id, entry.namespace)
        line = json.dumps(asdict(entry), ensure_ascii=False) + "\n"
        byte_offset = path.stat().st_size if path.exists() else 0
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
        return str(path), byte_offset

    def read_entry(self, file_path: str, byte_offset: int) -> RawLogEntry:
        """Read a single entry at the given byte offset."""
        with open(file_path, "r", encoding="utf-8") as f:
            f.seek(byte_offset)
            line = f.readline()
        data = json.loads(line)
        return RawLogEntry(**data)

    def iter_session(self, session_id: str, namespace: str = "default"):
        """Yield all entries for a session in order."""
        path = self._session_path(session_id, namespace)
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield RawLogEntry(**json.loads(line))

    def search(self, session_id: str, text: str, namespace: str = "default") -> list[RawLogEntry]:
        """Simple text search within a session log."""
        results = []
        for entry in self.iter_session(session_id, namespace):
            if text.lower() in entry.content.lower():
                results.append(entry)
        return results

    def list_sessions(self) -> list[str]:
        """Return all session IDs that have log files."""
        return [
            p.stem for p in sorted(self.log_dir.glob("*.jsonl"))
        ]

    def session_size(self, session_id: str, namespace: str = "default") -> int:
        """Return the file size in bytes for a session log."""
        path = self._session_path(session_id, namespace)
        return path.stat().st_size if path.exists() else 0

    def delete_files(self, paths: list[str]) -> int:
        """Delete log files (used only by an explicit namespace purge)."""
        removed = 0
        root = self.log_dir.resolve()
        for p in paths:
            path = Path(p).resolve()
            if root in path.parents and path.exists():
                path.unlink()
                removed += 1
        return removed
