"""
Workflow and Snippet management for OpenAgent.

Workflows are YAML files in ~/.config/openagent/workflows/*.yaml
Each defines: name, description, params, and steps (natural language or tool calls).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

WORKFLOWS_DIR = Path.home() / ".config" / "openagent" / "workflows"
WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Workflow:
    name: str
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    steps: List[Dict[str, Any]] = field(default_factory=list)


class WorkflowManager:
    def __init__(self, base_dir: Path = WORKFLOWS_DIR) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def list(self) -> List[Workflow]:
        out: List[Workflow] = []
        for p in sorted(self.base_dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(p.read_text()) or {}
                wf = Workflow(
                    name=data.get("name") or p.stem,
                    description=data.get("description", ""),
                    params=data.get("params", {}) or {},
                    steps=data.get("steps", []) or [],
                )
                out.append(wf)
            except Exception:
                continue
        return out

    def get(self, name: str) -> Optional[Workflow]:
        path = self.base_dir / f"{name}.yaml"
        if not path.exists():
            return None
        try:
            data = yaml.safe_load(path.read_text()) or {}
            return Workflow(
                name=data.get("name") or name,
                description=data.get("description", ""),
                params=data.get("params", {}) or {},
                steps=data.get("steps", []) or [],
            )
        except Exception:
            return None

    def create(self, name: str, description: str = "") -> Workflow:
        wf = Workflow(name=name, description=description, params={}, steps=[])
        path = self.base_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump({
            "name": wf.name,
            "description": wf.description,
            "params": wf.params,
            "steps": wf.steps,
        }, sort_keys=False))
        return wf

    def save(self, wf: Workflow) -> None:
        path = self.base_dir / f"{wf.name}.yaml"
        path.write_text(yaml.safe_dump({
            "name": wf.name,
            "description": wf.description,
            "params": wf.params,
            "steps": wf.steps,
        }, sort_keys=False))

    def sync(self, repo_url: str, branch: str = "main", dest: Optional[Path] = None) -> Path:
        """Sync workflows from a Git repo to the workflows directory or dest."""
        import subprocess as sp
        target = dest or self.base_dir
        target.mkdir(parents=True, exist_ok=True)
        git_dir = target / ".git"
        try:
            if git_dir.exists():
                sp.run(["git", "-C", str(target), "fetch", "--all", "--quiet"], check=False)
                sp.run(["git", "-C", str(target), "checkout", branch], check=False)
                sp.run(["git", "-C", str(target), "pull", "--ff-only", "origin", branch], check=False)
            else:
                sp.run(["git", "clone", "--depth", "1", "--branch", branch, repo_url, str(target)], check=True)
        except Exception:
            # Best-effort; leave files as-is
            pass
        return target
