"""
Gather runtime context for prompts and planning.
"""
from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional


@dataclass
class SystemContext:
    cwd: str
    shell: str
    os: str
    os_version: str
    machine: str
    git_branch: Optional[str]
    git_short_log: Optional[str]

    def to_prompt_block(self) -> str:
        lines = [
            f"CWD: {self.cwd}",
            f"Shell: {self.shell}",
            f"OS: {self.os} {self.os_version} ({self.machine})",
        ]
        if self.git_branch:
            lines.append(f"Git Branch: {self.git_branch}")
        if self.git_short_log:
            lines.append("Recent Git: " + self.git_short_log)
        return "\n".join(lines)


def _run(cmd: str, cwd: Optional[str] = None, timeout: float = 1.5) -> Optional[str]:
    try:
        out = subprocess.run(
            cmd, shell=True, cwd=cwd or os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout
        )
        if out.returncode == 0:
            return out.stdout.decode("utf-8", errors="replace").strip()
    except Exception:
        pass
    return None


def gather_context(start_dir: Optional[str] = None) -> SystemContext:
    cwd = start_dir or os.getcwd()
    shell = os.environ.get("SHELL", "zsh")
    uname = platform.uname()

    # Git info (best-effort)
    branch = _run("git rev-parse --abbrev-ref HEAD", cwd=cwd)
    short_log = _run("git --no-pager log --oneline -n 3", cwd=cwd)

    return SystemContext(
        cwd=cwd,
        shell=Path(shell).name,
        os=uname.system,
        os_version=uname.release,
        machine=uname.machine,
        git_branch=branch,
        git_short_log=short_log,
    )

