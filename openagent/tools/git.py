"""
Git and repository tools for OpenAgent.

Provides safe, non-interactive wrappers around common git operations and
repo-wide text search/grep.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from openagent.core.base import BaseTool, ToolResult


@dataclass
class GitCommand:
    subcommand: str
    args: List[str]

    def to_argv(self) -> List[str]:
        base = ["git", "--no-pager", self.subcommand]
        return base + self.args


SAFE_GIT_SUBCOMMANDS = {
    "status": [],
    "log": ["--oneline", "-n", "--stat"],
    "diff": ["--", "--name-only"],
    "branch": ["--all"],
    "show": [],
}


class GitTool(BaseTool):
    """Safe git queries (status, log, diff, branch, show)."""

    def __init__(self, **kwargs):
        caps = set(kwargs.get("capabilities", []))
        caps.add("terminal")
        kwargs["capabilities"] = list(caps)
        super().__init__(
            name="git_tool",
            description="Safe git operations (status, log, diff, branch, show)",
            **kwargs,
        )

    async def execute(self, input_data: Union[str, Dict[str, Any]]) -> ToolResult:
        try:
            if isinstance(input_data, dict):
                sub = str(input_data.get("subcommand", "status")).strip()
                args = input_data.get("args", [])
            else:
                parts = str(input_data).strip().split()
                sub = parts[0] if parts else "status"
                args = parts[1:]

            if sub not in SAFE_GIT_SUBCOMMANDS:
                return ToolResult(
                    success=False,
                    content="",
                    error=f"Unsupported git subcommand: {sub}",
                )

            # Filter args to safe allowlist patterns
            allowed_prefixes = SAFE_GIT_SUBCOMMANDS[sub]
            safe_args: List[str] = []
            for a in args:
                if a.startswith("-"):
                    if any(a.startswith(p) for p in allowed_prefixes):
                        safe_args.append(a)
                    else:
                        # skip unsafe flags silently
                        continue
                else:
                    safe_args.append(a)

            cmd = GitCommand(sub, safe_args)
            result = await self._run_exec(cmd.to_argv())
            return ToolResult(
                success=result["success"],
                content=result["output"],
                error=result.get("error"),
                metadata={
                    "command": " ".join(cmd.to_argv()),
                    "exit_code": result.get("exit_code"),
                },
            )
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))

    async def _run_exec(self, argv: List[str]) -> Dict[str, Any]:
        import time

        start = time.time()
        try:
            from openagent.utils.subprocess_utils import run_exec

            res = await run_exec(argv, timeout=15.0)
            out = (res.get("stdout") or "").strip()
            err = (res.get("stderr") or "").strip()
            success = bool(res.get("success"))
            exit_code = int(res.get("exit_code") or 0)
            return {
                "success": success,
                "output": out if out else err,
                "error": None if success else err,
                "exit_code": exit_code,
                "duration": time.time() - start,
            }
        except Exception as e:
            return {"success": False, "output": "", "error": str(e), "exit_code": -1}


class RepoGrep(BaseTool):
    """Non-interactive repo-wide search/grep utility."""

    def __init__(self, **kwargs):
        caps = set(kwargs.get("capabilities", []))
        caps.add("terminal")
        kwargs["capabilities"] = list(caps)
        super().__init__(
            name="repo_grep",
            description="Search repository files using ripgrep or grep",
            **kwargs,
        )

    async def execute(self, input_data: Union[str, Dict[str, Any]]) -> ToolResult:
        try:
            if isinstance(input_data, dict):
                pattern = input_data.get("pattern", "")
                path = input_data.get("path", ".")
                flags = input_data.get("flags", ["-n", "-H", "-I"])
            else:
                parts = str(input_data).strip().split()
                pattern = parts[0] if parts else ""
                path = parts[1] if len(parts) > 1 else "."
                flags = ["-n", "-H", "-I"]

            if not pattern:
                return ToolResult(
                    success=False, content="", error="pattern is required"
                )

            # Prefer ripgrep if available for speed
            cmd = await self._pick_search_cmd(pattern, path, flags)
            result = await self._run_exec(cmd)
            return ToolResult(
                success=result["success"],
                content=result["output"],
                error=result.get("error"),
                metadata={"command": " ".join(cmd)},
            )
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))

    async def _pick_search_cmd(
        self, pattern: str, path: str, flags: List[str]
    ) -> List[str]:
        rg_check = await self._which("rg")
        if rg_check:
            # Use smart defaults, ignore binary, no colors, no pager
            safe_flags = ["--no-messages", "--hidden", "--line-number", "--color=never"]
            return ["rg", *safe_flags, pattern, path]
        # Fallback to grep -r
        return [
            "grep",
            "-r",
            "--binary-files=without-match",
            "--line-number",
            "--color=never",
            pattern,
            path,
        ]

    async def _which(self, prog: str) -> bool:
        proc = await asyncio.create_subprocess_shell(
            f"command -v {prog}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0

    async def _run(self, command: str) -> Dict[str, Any]:
        # Deprecated: prefer _run_exec with argv
        return await self._run_exec(command.split())
