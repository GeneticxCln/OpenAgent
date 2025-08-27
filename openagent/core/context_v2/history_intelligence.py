from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import json


@dataclass
class HistoryEntry:
    ts: float
    cwd: str
    cmd: str
    decision: str | None = None
    reason: str | None = None
    approved: bool | None = None


class HistoryIntelligence:
    def __init__(self, path: Path | None = None):
        if path is None:
            path = Path.home() / ".config" / "openagent" / "history.jsonl"
        self.path = path

    def recent_commands(self, limit: int = 50) -> List[str]:
        cmds: List[str] = []
        try:
            if not self.path.exists():
                return []
            with self.path.open("r", encoding="utf-8") as fp:
                lines = fp.readlines()
            # Read from the end for most recent
            for line in reversed(lines):
                try:
                    obj = json.loads(line)
                    cmd = obj.get("cmd") or obj.get("command")
                    if not cmd:
                        continue
                    cmds.append(str(cmd))
                    if len(cmds) >= limit:
                        break
                except Exception:
                    continue
        except Exception:
            return []
        # Reverse back to chronological order (oldest->newest) for display consistency
        return list(reversed(cmds))

