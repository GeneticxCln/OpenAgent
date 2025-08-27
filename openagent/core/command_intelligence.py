from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from pathlib import Path
import os

from .context_v2.project_analyzer import ProjectContextEngine
from .context_v2.history_intelligence import HistoryIntelligence


@dataclass
class CompletionContext:
    current_directory: Path
    project_type: Optional[str] = None
    git_repo: bool = False
    git_branch: Optional[str] = None
    recent_commands: List[str] = None
    environment_vars: dict[str, str] = None


@dataclass
class Suggestion:
    text: str
    confidence: float = 0.5


class CommandCompletionEngine:
    """Heuristic command completion/suggestion engine.

    Lightweight, no-LLM, fast suggestions based on project context
    and recent command history.
    """

    def __init__(self):
        self.common_commands = [
            "git status",
            "git diff",
            "git log --oneline -n 20",
            "ls -la",
            "pwd",
            "grep -r --line-number --color=never PATTERN .",
            "find . -name '*.py'",
            "python -m pytest -q",
            "pytest -q",
            "pip install -r requirements.txt",
            "npm test",
            "npm run build",
        ]

    def suggest_commands(
        self, partial: str, ctx: CompletionContext, max_suggestions: int = 10
    ) -> List[Suggestion]:
        p = (partial or "").strip()
        suggestions: List[Suggestion] = []

        # 1) History-based suggestions first (startswith match)
        hist = (ctx.recent_commands or [])[:100]
        seen = set()
        for cmd in hist:
            if p and not cmd.startswith(p):
                continue
            if cmd in seen:
                continue
            suggestions.append(Suggestion(text=cmd, confidence=0.9))
            seen.add(cmd)
            if len(suggestions) >= max_suggestions:
                return suggestions

        # 2) Project-aware templates
        project_cmds = self._project_commands(ctx)
        for cmd in project_cmds:
            if p and not cmd.startswith(p):
                continue
            if cmd in seen:
                continue
            suggestions.append(Suggestion(text=cmd, confidence=0.8))
            seen.add(cmd)
            if len(suggestions) >= max_suggestions:
                return suggestions

        # 3) Common commands fallback
        for cmd in self.common_commands:
            if p and not cmd.startswith(p):
                continue
            if cmd in seen:
                continue
            suggestions.append(Suggestion(text=cmd, confidence=0.6))
            seen.add(cmd)
            if len(suggestions) >= max_suggestions:
                break

        return suggestions

    def _project_commands(self, ctx: CompletionContext) -> List[str]:
        out: List[str] = []
        pt = (ctx.project_type or "").lower()
        if pt == "python":
            out += [
                "pytest -q",
                "pytest -m unit -q",
                "pytest -m integration -q",
                "black openagent tests",
                "isort openagent tests",
                "mypy openagent",
            ]
        if pt == "node":
            out += [
                "npm test",
                "npm run build",
                "npm run lint",
            ]
        if pt == "go":
            out += [
                "go test ./...",
                "go build ./...",
            ]
        # Git repo generic
        if ctx.git_repo:
            out += [
                "git status",
                "git diff --name-only",
                "git branch --all",
            ]
        return out

    def auto_correct_command(self, command: str) -> Optional[str]:
        """Very simple auto-correction for common typos and flags.
        Returns corrected command or None if not applicable.
        """
        c = (command or "").strip()
        if not c:
            return None
        # Common typos
        fixes = {
            "git staus": "git status",
            "git statsu": "git status",
            "pytes": "pytest",
            "pip isntall": "pip install",
            "npn": "npm",
            "gti ": "git ",
        }
        for bad, good in fixes.items():
            if c.startswith(bad):
                return good + c[len(bad) :]
        # Add '--color=never' for grep suggestions if missing
        if c.startswith("grep ") and "--color" not in c:
            return c + " --color=never"
        return None


def create_command_completion_engine() -> CommandCompletionEngine:
    return CommandCompletionEngine()

