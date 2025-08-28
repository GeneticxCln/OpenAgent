"""
Tools module for OpenAgent.

This module contains built-in tools for various operations including
system management, file operations, and command execution.
"""

from .git import GitTool, RepoGrep
from .patch import PatchEditor
from .system import CommandExecutor, FileManager, SystemInfo

# Optional: GitHubTool requires httpx; make import optional to avoid hard dependency
try:  # pragma: no cover - environment without httpx
    from .github import GitHubTool  # type: ignore
except Exception:  # ImportError or other optional dep issues
    GitHubTool = None  # type: ignore

__all__ = [
    "CommandExecutor",
    "FileManager",
    "SystemInfo",
    "GitTool",
    "RepoGrep",
    "PatchEditor",
    "GitHubTool",
]
