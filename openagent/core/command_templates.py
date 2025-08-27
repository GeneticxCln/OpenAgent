from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pathlib import Path

from .context_v2.project_analyzer import ProjectContextEngine


@dataclass
class Template:
    name: str


class CommandTemplates:
    """Suggest command templates based on workspace context."""

    def __init__(self):
        pass

    def suggest_templates(self, workspace) -> List[Template]:
        out: List[Template] = []
        pt = (getattr(workspace, "project_type", None) or "").lower()
        if pt == "python":
            out += [
                Template(name="run_tests"),
                Template(name="format_python"),
                Template(name="type_check"),
            ]
        if pt == "node":
            out += [
                Template(name="npm_test"),
                Template(name="npm_build"),
                Template(name="npm_lint"),
            ]
        if getattr(workspace, "git_context", None) and workspace.git_context.is_repo:
            out += [Template(name="git_status"), Template(name="git_diff_summary")]
        return out


def create_command_templates() -> CommandTemplates:
    return CommandTemplates()

