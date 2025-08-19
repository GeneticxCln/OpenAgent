"""
Enhanced Context Engine for OpenAgent.

This module extends the base context engine with advanced features:
- Git context integration with branch, commits, and changes
- File relevance scoring and automatic inclusion
- Error context extraction from recent terminal commands
- Terminal history analysis for better assistance
- Enhanced project type detection with framework-specific contexts
"""

import asyncio
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from openagent.core.context_engine import (
    ContextDetectionResult,
    ContextDetector,
    FileInfo,
    ProjectType,
)
from openagent.core.exceptions import AgentError


@dataclass
class GitContext:
    """Git repository context information."""

    is_repo: bool = False
    branch: Optional[str] = None
    remote_url: Optional[str] = None
    recent_commits: List[Dict[str, Any]] = field(default_factory=list)
    staged_files: List[str] = field(default_factory=list)
    modified_files: List[str] = field(default_factory=list)
    untracked_files: List[str] = field(default_factory=list)
    status_summary: Optional[str] = None
    last_commit_author: Optional[str] = None
    last_commit_date: Optional[str] = None
    total_commits: int = 0


@dataclass
class ErrorContext:
    """Context about recent terminal errors and issues."""

    recent_errors: List[Dict[str, Any]] = field(default_factory=list)
    failed_commands: List[str] = field(default_factory=list)
    error_patterns: Dict[str, int] = field(default_factory=dict)
    suggested_fixes: Dict[str, str] = field(default_factory=dict)
    last_error_time: Optional[float] = None


@dataclass
class FileRelevanceData:
    """File relevance scoring data."""

    path: Path
    score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    last_modified: float = 0.0
    size_bytes: int = 0
    language: Optional[str] = None
    is_config: bool = False
    is_recent: bool = False
    has_errors: bool = False


@dataclass
class EnhancedContext:
    """Complete enhanced context for intelligent assistance."""

    project: ContextDetectionResult = field(default_factory=ContextDetectionResult)
    git: GitContext = field(default_factory=GitContext)
    errors: ErrorContext = field(default_factory=ErrorContext)
    relevant_files: List[FileRelevanceData] = field(default_factory=list)
    terminal_history: List[str] = field(default_factory=list)
    current_task_context: Optional[str] = None
    environment_vars: Dict[str, str] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class EnhancedContextDetector(ContextDetector):
    """
    Enhanced context detector with git integration, error tracking,
    and intelligent file relevance scoring.
    """

    def __init__(self, start_path: Path, max_depth: int = 4, history_limit: int = 50):
        super().__init__(start_path, max_depth)
        self.history_limit = history_limit
        self.error_patterns = {
            r"command not found": "missing_command",
            r"No such file or directory": "file_not_found",
            r"Permission denied": "permission_error",
            r"syntax error": "syntax_error",
            r"ModuleNotFoundError": "python_import_error",
            r"npm ERR!": "npm_error",
            r"fatal: not a git repository": "git_not_repo",
            r"error: failed to push": "git_push_error",
            r"ENOENT": "node_file_error",
        }

    async def detect_enhanced_context(self) -> EnhancedContext:
        """Perform comprehensive enhanced context detection."""
        start_time = time.time()

        try:
            context = EnhancedContext()

            # Gather basic project context
            context.project = await self.detect_context()

            # Gather git context
            if context.project.project_root:
                context.git = await self._detect_git_context(
                    context.project.project_root
                )

            # Analyze file relevance
            context.relevant_files = await self._analyze_file_relevance(
                context.project, context.git
            )

            # Extract error context from recent commands
            context.errors = await self._extract_error_context()

            # Get terminal history for pattern analysis
            context.terminal_history = await self._get_terminal_history()

            # Gather environment context
            context.environment_vars = self._get_relevant_env_vars()

            # Record performance metrics
            context.performance_metrics = {
                "detection_time_ms": (time.time() - start_time) * 1000,
                "files_analyzed": len(context.relevant_files),
                "git_repo_detected": context.git.is_repo,
                "recent_errors": len(context.errors.recent_errors),
            }

            return context

        except Exception as e:
            raise AgentError(f"Enhanced context detection failed: {e}")

    async def _detect_git_context(self, project_root: Path) -> GitContext:
        """Detect git repository context and recent activity."""
        git_context = GitContext()

        try:
            # Check if this is a git repository
            result = await self._run_git_command("rev-parse --git-dir", project_root)
            if not result["success"]:
                return git_context

            git_context.is_repo = True

            # Get current branch
            branch_result = await self._run_git_command(
                "rev-parse --abbrev-ref HEAD", project_root
            )
            if branch_result["success"]:
                git_context.branch = branch_result["output"].strip()

            # Get remote URL
            remote_result = await self._run_git_command(
                "config --get remote.origin.url", project_root
            )
            if remote_result["success"]:
                git_context.remote_url = remote_result["output"].strip()

            # Get recent commits (last 10)
            log_result = await self._run_git_command(
                "log --oneline --no-merges -10 --pretty=format:'%h|%an|%ad|%s'",
                project_root,
            )
            if log_result["success"]:
                for line in log_result["output"].split("\n"):
                    if "|" in line:
                        parts = line.split("|", 3)
                        if len(parts) == 4:
                            git_context.recent_commits.append(
                                {
                                    "hash": parts[0],
                                    "author": parts[1],
                                    "date": parts[2],
                                    "message": parts[3],
                                }
                            )

            # Get repository status
            status_result = await self._run_git_command(
                "status --porcelain", project_root
            )
            if status_result["success"]:
                for line in status_result["output"].split("\n"):
                    if line.strip():
                        status_code = line[:2]
                        filename = line[3:].strip()

                        if status_code.startswith("A") or status_code.startswith("M"):
                            git_context.staged_files.append(filename)
                        elif status_code.endswith("M") or status_code.endswith("T"):
                            git_context.modified_files.append(filename)
                        elif status_code.startswith("??"):
                            git_context.untracked_files.append(filename)

            # Get status summary
            status_summary_result = await self._run_git_command(
                "status --short", project_root
            )
            if status_summary_result["success"]:
                git_context.status_summary = status_summary_result["output"]

            # Get last commit info
            if git_context.recent_commits:
                last_commit = git_context.recent_commits[0]
                git_context.last_commit_author = last_commit["author"]
                git_context.last_commit_date = last_commit["date"]

            # Get total commit count
            count_result = await self._run_git_command(
                "rev-list --count HEAD", project_root
            )
            if count_result["success"]:
                try:
                    git_context.total_commits = int(count_result["output"].strip())
                except ValueError:
                    pass

        except Exception as e:
            # If git operations fail, return what we have
            pass

        return git_context

    async def _run_git_command(
        self, command: str, cwd: Path, timeout: float = 10.0
    ) -> Dict[str, Any]:
        """Run a git command safely with timeout."""
        full_command = f"git {command}"

        try:
            proc = await asyncio.create_subprocess_shell(
                full_command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.terminate()
                await proc.wait()
                return {"success": False, "output": "", "error": "Command timeout"}

            output = stdout.decode("utf-8", errors="replace").strip()
            error = stderr.decode("utf-8", errors="replace").strip()

            return {
                "success": proc.returncode == 0,
                "output": output,
                "error": error if proc.returncode != 0 else None,
                "exit_code": proc.returncode,
            }

        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}

    async def _analyze_file_relevance(
        self, project_context: ContextDetectionResult, git_context: GitContext
    ) -> List[FileRelevanceData]:
        """Analyze and score file relevance for the current context."""
        relevant_files = []

        if not project_context.project_root:
            return relevant_files

        try:
            # Get all files in the project
            all_files = await self._list_project_files(project_context.project_root)
            current_time = time.time()

            for file_path in all_files:
                try:
                    stat = file_path.stat()
                    relevance_data = FileRelevanceData(
                        path=file_path,
                        last_modified=stat.st_mtime,
                        size_bytes=stat.st_size,
                        language=self._detect_file_language(file_path),
                    )

                    # Score the file based on various factors
                    relevance_data.score = await self._calculate_file_relevance_score(
                        relevance_data, project_context, git_context, current_time
                    )

                    # Only include files with significant relevance
                    if relevance_data.score > 0.1:
                        relevant_files.append(relevance_data)

                except (OSError, PermissionError):
                    continue

            # Sort by relevance score (descending)
            relevant_files.sort(key=lambda f: f.score, reverse=True)

            # Limit to top N most relevant files
            return relevant_files[:50]

        except Exception as e:
            return relevant_files

    async def _calculate_file_relevance_score(
        self,
        file_data: FileRelevanceData,
        project_context: ContextDetectionResult,
        git_context: GitContext,
        current_time: float,
    ) -> float:
        """Calculate relevance score for a file based on multiple factors."""
        score = 0.0
        reasons = []

        file_path = file_data.path
        relative_path = str(
            file_path.relative_to(project_context.project_root or Path.cwd())
        )

        # Base score for project type relevance
        if file_path in project_context.key_files:
            score += 1.0
            reasons.append("key_project_file")

        # Configuration files are highly relevant
        config_patterns = [
            r".*\.json$",
            r".*\.ya?ml$",
            r".*\.toml$",
            r".*\.ini$",
            r".*\.cfg$",
            r".*\.env.*$",
            r"Dockerfile.*",
            r".*\.conf$",
            r"Makefile.*",
            r".*requirements.*\.txt$",
            r"package\.json$",
            r"tsconfig\.json$",
        ]

        for pattern in config_patterns:
            if re.match(pattern, file_path.name, re.IGNORECASE):
                score += 0.8
                reasons.append("config_file")
                file_data.is_config = True
                break

        # Recently modified files are more relevant
        hours_since_modified = (current_time - file_data.last_modified) / 3600
        if hours_since_modified < 24:  # Last 24 hours
            score += 0.6
            reasons.append("recently_modified")
            file_data.is_recent = True
        elif hours_since_modified < 168:  # Last week
            score += 0.3
            reasons.append("modified_this_week")

        # Git status relevance
        if git_context.is_repo:
            if relative_path in git_context.staged_files:
                score += 0.7
                reasons.append("git_staged")
            if relative_path in git_context.modified_files:
                score += 0.6
                reasons.append("git_modified")
            if relative_path in git_context.untracked_files:
                score += 0.4
                reasons.append("git_untracked")

        # File size penalty for very large files
        if file_data.size_bytes > 1024 * 1024:  # > 1MB
            score *= 0.5
            reasons.append("large_file_penalty")

        # Language-specific relevance
        if file_data.language:
            if file_data.language in project_context.language_map:
                score += 0.3
                reasons.append(f"matches_project_language_{file_data.language}")

        # Main/entry files boost
        entry_patterns = [
            r"main\.(py|js|ts|go|rs|java)$",
            r"index\.(js|ts|html)$",
            r"app\.(py|js|ts)$",
            r"server\.(py|js|ts)$",
            r"__init__\.py$",
            r"setup\.(py|cfg)$",
        ]

        for pattern in entry_patterns:
            if re.match(pattern, file_path.name, re.IGNORECASE):
                score += 0.5
                reasons.append("entry_point")
                break

        # Documentation files
        if any(
            ext in file_path.suffix.lower() for ext in [".md", ".rst", ".txt"]
        ) and any(
            name in file_path.name.lower()
            for name in ["readme", "changelog", "todo", "notes"]
        ):
            score += 0.4
            reasons.append("documentation")

        # Test files get lower priority
        if "test" in relative_path.lower() or file_path.name.startswith("test_"):
            score *= 0.3
            reasons.append("test_file_penalty")

        # Hidden files and directories get lower priority
        if file_path.name.startswith(".") and file_path.name not in [
            ".env",
            ".gitignore",
        ]:
            score *= 0.4
            reasons.append("hidden_file_penalty")

        file_data.reasons = reasons
        return max(0.0, score)

    def _detect_file_language(self, file_path: Path) -> Optional[str]:
        """Detect programming language from file extension."""
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".kt": "kotlin",
            ".cs": "csharp",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".r": "r",
            ".scala": "scala",
            ".sh": "shell",
            ".bash": "shell",
            ".zsh": "shell",
            ".fish": "shell",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".toml": "toml",
            ".xml": "xml",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".less": "less",
            ".sql": "sql",
        }

        return extension_map.get(file_path.suffix.lower())

    async def _extract_error_context(self) -> ErrorContext:
        """Extract error context from recent terminal commands and logs."""
        error_context = ErrorContext()

        try:
            # Get recent command history with errors
            history_with_errors = await self._get_error_history()

            for entry in history_with_errors:
                # Parse error information
                if entry.get("exit_code", 0) != 0:
                    error_info = {
                        "command": entry.get("command", ""),
                        "error_output": entry.get("stderr", ""),
                        "exit_code": entry.get("exit_code"),
                        "timestamp": entry.get("timestamp"),
                    }

                    error_context.recent_errors.append(error_info)
                    error_context.failed_commands.append(entry.get("command", ""))

                    # Analyze error patterns
                    error_text = entry.get("stderr", "") + entry.get("stdout", "")
                    for pattern, error_type in self.error_patterns.items():
                        if re.search(pattern, error_text, re.IGNORECASE):
                            error_context.error_patterns[error_type] = (
                                error_context.error_patterns.get(error_type, 0) + 1
                            )

            # Generate suggested fixes based on error patterns
            error_context.suggested_fixes = self._generate_error_fixes(
                error_context.error_patterns
            )

            if error_context.recent_errors:
                error_context.last_error_time = max(
                    entry.get("timestamp", 0) for entry in error_context.recent_errors
                )

        except Exception as e:
            # If we can't extract error context, return empty context
            pass

        return error_context

    async def _get_error_history(self) -> List[Dict[str, Any]]:
        """Get recent command history with error information."""
        # This is a simplified version - in a real implementation,
        # you would integrate with shell history, logs, or a command tracker
        history = []

        try:
            # Try to read from shell history files
            shell_history_files = [
                Path.home() / ".zsh_history",
                Path.home() / ".bash_history",
                Path.home() / ".history",
            ]

            for history_file in shell_history_files:
                if history_file.exists():
                    # This is simplified - real implementation would be more sophisticated
                    with open(
                        history_file, "r", encoding="utf-8", errors="ignore"
                    ) as f:
                        lines = f.readlines()[-100:]  # Get last 100 commands

                    for line in lines:
                        if line.strip():
                            history.append(
                                {
                                    "command": line.strip(),
                                    "timestamp": time.time(),  # Placeholder
                                    "exit_code": 0,  # Would need real tracking
                                    "stderr": "",  # Would need real tracking
                                }
                            )
                    break

        except Exception:
            pass

        return history

    def _generate_error_fixes(self, error_patterns: Dict[str, int]) -> Dict[str, str]:
        """Generate suggested fixes for common error patterns."""
        fixes = {}

        fix_suggestions = {
            "missing_command": "Install the missing command or check if it's in your PATH",
            "file_not_found": "Check the file path and ensure the file exists",
            "permission_error": "Check file permissions or run with appropriate privileges",
            "syntax_error": "Review the syntax of your command or script",
            "python_import_error": "Install the missing Python module with pip",
            "npm_error": 'Try clearing npm cache with "npm cache clean --force"',
            "git_not_repo": 'Initialize a git repository with "git init" or navigate to a git repo',
            "git_push_error": "Check your git remote configuration and authentication",
            "node_file_error": "Check if the Node.js file exists and npm dependencies are installed",
        }

        for error_type, count in error_patterns.items():
            if count > 0 and error_type in fix_suggestions:
                fixes[error_type] = fix_suggestions[error_type]

        return fixes

    async def _get_terminal_history(self) -> List[str]:
        """Get recent terminal command history for pattern analysis."""
        history = []

        try:
            # Get history from various shell history files
            shell_history_files = [
                Path.home() / ".zsh_history",
                Path.home() / ".bash_history",
            ]

            for history_file in shell_history_files:
                if history_file.exists():
                    with open(
                        history_file, "r", encoding="utf-8", errors="ignore"
                    ) as f:
                        lines = f.readlines()
                        # Get recent commands, clean them up
                        recent_commands = [
                            line.strip().split(";")[-1] if ";" in line else line.strip()
                            for line in lines[-self.history_limit :]
                        ]
                        history.extend(recent_commands)
                    break

        except Exception:
            pass

        return [cmd for cmd in history if cmd and not cmd.startswith("#")]

    def _get_relevant_env_vars(self) -> Dict[str, str]:
        """Get relevant environment variables for context."""
        relevant_vars = {}

        # Important environment variables for development context
        important_vars = [
            "PATH",
            "SHELL",
            "TERM",
            "USER",
            "HOME",
            "PWD",
            "PYTHON_PATH",
            "PYTHONPATH",
            "VIRTUAL_ENV",
            "CONDA_DEFAULT_ENV",
            "NODE_ENV",
            "NPM_CONFIG_PREFIX",
            "JAVA_HOME",
            "GIT_AUTHOR_NAME",
            "GIT_AUTHOR_EMAIL",
            "EDITOR",
            "VISUAL",
            "PAGER",
            "LANG",
            "LC_ALL",
            "LANGUAGE",
            "OPENAGENT_MODEL",
            "OPENAGENT_EXPLAIN",
            "HF_TOKEN",
            "HUGGINGFACE_TOKEN",
        ]

        for var in important_vars:
            value = os.environ.get(var)
            if value:
                relevant_vars[var] = value

        return relevant_vars


# Utility function for easy usage
async def get_enhanced_context(start_path: Optional[Path] = None) -> EnhancedContext:
    """Get enhanced context for the current or specified directory."""
    detector = EnhancedContextDetector(start_path or Path.cwd())
    return await detector.detect_enhanced_context()
