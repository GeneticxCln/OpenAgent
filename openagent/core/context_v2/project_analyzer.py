from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import subprocess


class ProjectType:
    PYTHON = "python"
    NODE = "node"
    GO = "go"
    UNKNOWN = "unknown"


@dataclass
class GitContext:
    is_repo: bool = False
    current_branch: Optional[str] = None


@dataclass
class Workspace:
    root: Path
    project_type: str
    git_context: GitContext


class ProjectContextEngine:
    def analyze_workspace(self, current_dir: Path) -> Workspace:
        root = Path(current_dir).resolve()
        pt = self._detect_project_type(root)
        git = self._git_context(root)
        return Workspace(root=root, project_type=pt, git_context=git)

    def _detect_project_type(self, root: Path) -> str:
        # Straightforward file heuristics
        if (root / "pyproject.toml").exists() or (root / "setup.py").exists():
            return ProjectType.PYTHON
        if (root / "package.json").exists():
            return ProjectType.NODE
        if (root / "go.mod").exists():
            return ProjectType.GO
        return ProjectType.UNKNOWN

    def _git_context(self, root: Path) -> GitContext:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=str(root),
                capture_output=True,
                text=True,
                check=False,
            )
            is_repo = result.returncode == 0 and result.stdout.strip() == "true"
            branch = None
            if is_repo:
                r2 = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if r2.returncode == 0:
                    branch = r2.stdout.strip()
            return GitContext(is_repo=is_repo, current_branch=branch)
        except Exception:
            return GitContext(is_repo=False, current_branch=None)

