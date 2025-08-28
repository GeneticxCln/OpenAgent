"""
Patch Editor Tool for OpenAgent.

Provides uniquely-scoped, atomic search/replace editing with dry-run previews
and rollback on failure. Enforces safe_paths policy via PolicyEngine.
"""

from __future__ import annotations

import asyncio
import difflib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from openagent.core.base import BaseTool, ToolResult
from openagent.core.policy import get_policy_engine


@dataclass
class PatchHunk:
    search: str
    replace: str


class PatchEditor(BaseTool):
    """
    Apply search/replace patches to one or more files with safety checks.

    Input schema (dict):
      {
        "files": [
          { "path": "path/to/file", "hunks": [ {"search": "...", "replace": "..."}, ... ] }
        ],
        "require_unique": true,
        "dry_run": false,
        "rollback_on_failure": true,
        "encoding": "utf-8"
      }
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="patch_editor",
            description="Apply uniquely-scoped patches to files with dry-run and rollback",
            **kwargs,
        )

    async def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        try:
            files = input_data.get("files") or []
            require_unique = bool(input_data.get("require_unique", True))
            dry_run = bool(input_data.get("dry_run", False))
            rollback = bool(input_data.get("rollback_on_failure", True))
            encoding = input_data.get("encoding", "utf-8")

            if not isinstance(files, list) or not files:
                return ToolResult(
                    success=False, content="", error="'files' must be a non-empty list"
                )

            # Enforce policy for each file (writes within safe_paths only)
            policy_engine = get_policy_engine()

            # Keep original contents for rollback
            originals: Dict[Path, str] = {}
            changed: Dict[Path, str] = {}
            previews: Dict[str, str] = {}

            for f in files:
                path_str = (f or {}).get("path")
                hunks = (f or {}).get("hunks") or []
                if not path_str or not hunks:
                    return ToolResult(
                        success=False,
                        content="",
                        error="Each file entry must have 'path' and non-empty 'hunks'",
                    )

                p = Path(path_str).expanduser().resolve()

                # Safe path policy (reuse file ops constraints): require path in safe_paths and not in restricted_paths
                policy_err = await self._enforce_file_policy_write(p)
                if policy_err:
                    return ToolResult(success=False, content="", error=policy_err)

                if not p.exists():
                    return ToolResult(
                        success=False, content="", error=f"File does not exist: {p}"
                    )
                if not p.is_file():
                    return ToolResult(
                        success=False, content="", error=f"Path is not a file: {p}"
                    )

                original = p.read_text(encoding=encoding)
                originals[p] = original

                # Apply hunks in order
                new_content = original
                for h in hunks:
                    search = (h or {}).get("search")
                    replace = (h or {}).get("replace", "")
                    if search is None:
                        return ToolResult(
                            success=False, content="", error="Hunk missing 'search'"
                        )

                    occurrences = new_content.count(search)
                    if require_unique and occurrences != 1:
                        return ToolResult(
                            success=False,
                            content="",
                            error=f"Search text must appear exactly once in {p}, found {occurrences} occurrences",
                        )
                    if occurrences == 0:
                        return ToolResult(
                            success=False,
                            content="",
                            error=f"Search text not found in {p}",
                        )

                    # Replace only first occurrence to preserve intent
                    new_content = new_content.replace(search, replace, 1)

                # Prepare unified diff preview
                diff_lines = list(
                    difflib.unified_diff(
                        original.splitlines(keepends=True),
                        new_content.splitlines(keepends=True),
                        fromfile=str(p),
                        tofile=str(p),
                    )
                )
                preview = "".join(diff_lines)
                previews[str(p)] = preview

                if dry_run:
                    # Do not mutate files, just record preview
                    continue

                # Atomic write with temporary file
                tmp_path = p.with_suffix(p.suffix + ".openagent.tmp")
                try:
                    tmp_path.write_text(new_content, encoding=encoding)
                    tmp_path.replace(p)
                    changed[p] = new_content
                except Exception as e:
                    # Rollback previously written files if requested
                    if rollback:
                        for rp, content in originals.items():
                            if rp in changed:
                                try:
                                    rp.write_text(content, encoding=encoding)
                                except Exception:
                                    pass
                    return ToolResult(
                        success=False, content="", error=f"Failed to write {p}: {e}"
                    )

            # Success: build content summary
            if dry_run:
                summary = "Dry-run successful. No changes written."
                return ToolResult(
                    success=True,
                    content=summary,
                    metadata={"diff_preview": previews, "dry_run": True},
                )
            else:
                summary_lines = ["Applied patches to files:"]
                for p in originals.keys():
                    if p in changed:
                        summary_lines.append(f"- {p}")
                return ToolResult(
                    success=True,
                    content="\n".join(summary_lines),
                    metadata={"changed_files": [str(p) for p in changed.keys()]},
                )
        except Exception as e:
            return ToolResult(success=False, content="", error=f"Patch failed: {e}")

    async def _enforce_file_policy_write(self, path: Path) -> Optional[str]:
        """Return error string if policy forbids writing to path; else None."""
        try:
            engine = get_policy_engine()

            # Check restricted paths first
            for restricted in engine.policy.restricted_paths:
                try:
                    restricted_path = Path(restricted).expanduser().resolve()
                    if str(path.resolve()).startswith(str(restricted_path)):
                        return f"Path not in safe paths: {path}"
                except Exception:
                    continue

            # Then require safe paths
            in_safe = False
            for safe in engine.policy.safe_paths:
                try:
                    safe_path = Path(safe).expanduser().resolve()
                    if str(path.resolve()).startswith(str(safe_path)):
                        in_safe = True
                        break
                except Exception:
                    continue
            if not in_safe:
                return f"Path not in safe paths: {path}"

            # Evaluate pseudo-command for risk bookkeeping (optional)
            _ = await engine.evaluate_command(f"file:write {path}")
        except Exception:
            # Be conservative if policy engine fails
            return "Policy engine unavailable"
        return None
