"""
Intelligent Context Engine for OpenAgent.

This module provides the core components for the Intelligent Context Engine,
including context detection, smart gathering, and dynamic prompt engineering.
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openagent.core.exceptions import AgentError


class ProjectType(Enum):
    """Detected project types."""

    PYTHON = "Python"
    JAVASCRIPT = "JavaScript"
    TYPESCRIPT = "TypeScript"
    RUST = "Rust"
    GO = "Go"
    JAVA = "Java"
    DOTNET = "C#/.NET"
    DOCKER = "Docker"
    TERRAFORM = "Terraform"
    GENERIC = "Generic"
    UNKNOWN = "Unknown"


@dataclass
class FileInfo:
    """Information about a specific file."""

    path: Path
    size: int
    last_modified: float
    relevance_score: float = 0.0
    language: Optional[str] = None


@dataclass
class ContextDetectionResult:
    """Result of context detection."""

    project_type: ProjectType = ProjectType.UNKNOWN
    project_name: Optional[str] = None
    project_root: Optional[Path] = None
    confidence: float = 0.0
    key_files: List[Path] = field(default_factory=list)
    language_map: Dict[str, float] = field(default_factory=dict)
    frameworks: List[str] = field(default_factory=list)
    has_docker: bool = False
    has_terraform: bool = False


class ContextDetector:
    """
    Detects project context from the file system.

    Features:
    - Project type detection (Python, JS, Rust, etc.)
    - Framework identification (React, FastAPI, etc.)
    - Language composition analysis
    - Docker and Terraform integration detection
    """

    def __init__(self, start_path: Path, max_depth: int = 4):
        """Initialize context detector."""
        self.start_path = start_path
        self.max_depth = max_depth
        self.detection_rules = {
            ProjectType.PYTHON: {
                "files": [
                    "requirements.txt",
                    "setup.py",
                    "pyproject.toml",
                    "manage.py",
                ],
                "extensions": [".py"],
                "frameworks": {
                    "django": ["manage.py"],
                    "fastapi": ["main.py"],  # Needs content check
                    "flask": ["app.py"],  # Needs content check
                },
            },
            ProjectType.JAVASCRIPT: {
                "files": ["package.json", "yarn.lock", "pnpm-lock.yaml"],
                "extensions": [".js", ".jsx"],
                "frameworks": {
                    "react": ["react"],  # Dependency check
                    "vue": ["vue"],  # Dependency check
                    "angular": ["@angular/core"],  # Dependency check
                    "express": ["express"],  # Dependency check
                },
            },
            ProjectType.TYPESCRIPT: {
                "files": ["tsconfig.json"],
                "extensions": [".ts", ".tsx"],
            },
            ProjectType.RUST: {
                "files": ["Cargo.toml"],
                "extensions": [".rs"],
            },
            ProjectType.GO: {
                "files": ["go.mod"],
                "extensions": [".go"],
            },
            ProjectType.JAVA: {
                "files": ["pom.xml", "build.gradle"],
                "extensions": [".java", ".kt"],
                "frameworks": {
                    "spring": ["spring-boot"],  # Dependency check
                    "maven": ["pom.xml"],
                    "gradle": ["build.gradle"],
                },
            },
            ProjectType.DOTNET: {
                "files": [".sln", ".csproj"],
                "extensions": [".cs", ".fs"],
            },
            ProjectType.DOCKER: {
                "files": ["Dockerfile", "docker-compose.yml"],
                "extensions": [],
            },
            ProjectType.TERRAFORM: {
                "files": [],
                "extensions": [".tf", ".tfvars"],
            },
        }

    async def detect_context(self) -> ContextDetectionResult:
        """Perform comprehensive context detection."""
        result = ContextDetectionResult()

        # Find project root
        project_root = await self._find_project_root()
        if not project_root:
            result.project_root = self.start_path
            result.project_type = ProjectType.GENERIC
            return result

        result.project_root = project_root
        result.project_name = project_root.name

        # Analyze files in project
        files = await self._list_project_files(project_root)

        # Perform detection based on rules
        scores: Dict[ProjectType, float] = {pt: 0.0 for pt in ProjectType}

        for file_path in files:
            for proj_type, rules in self.detection_rules.items():
                # Check key files
                if file_path.name in rules["files"]:
                    scores[proj_type] += 1.0
                    result.key_files.append(file_path)

                # Check extensions
                if file_path.suffix in rules["extensions"]:
                    scores[proj_type] += 0.2
                    lang = rules["extensions"][0][1:]  # e.g. .py -> py
                    result.language_map[lang] = result.language_map.get(lang, 0) + 1

        # Determine primary project type
        if any(s > 0 for s in scores.values()):
            primary_type, max_score = max(scores.items(), key=lambda item: item[1])
            result.project_type = primary_type
            result.confidence = min(1.0, max_score / 2.0)

        # Detect frameworks and dependencies
        if result.project_type == ProjectType.JAVASCRIPT:
            await self._detect_js_frameworks(result, files)
        elif result.project_type == ProjectType.PYTHON:
            await self._detect_python_frameworks(result, files)

        # Detect Docker/Terraform presence
        result.has_docker = any(
            f.name in ["Dockerfile", "docker-compose.yml"] for f in files
        )
        result.has_terraform = any(f.suffix == ".tf" for f in files)

        return result

    async def _find_project_root(self) -> Optional[Path]:
        """Find the project root directory."""
        current_path = self.start_path
        for _ in range(self.max_depth):
            # Check for common root indicators
            if any(
                (current_path / f).exists()
                for f in [".git", "pyproject.toml", "package.json"]
            ):
                return current_path

            # Move up one level
            if current_path.parent == current_path:
                return None
            current_path = current_path.parent

        return None

    async def _list_project_files(self, project_root: Path) -> List[Path]:
        """List relevant files in the project."""
        files = []
        ignore_dirs = {".git", ".vscode", "node_modules", "__pycache__", "target"}

        for file_path in project_root.glob("**/*"):
            if file_path.is_file():
                # Check if any parent is in ignore_dirs
                if not any(d in file_path.parts for d in ignore_dirs):
                    files.append(file_path)

        return files

    async def _detect_js_frameworks(
        self, result: ContextDetectionResult, files: List[Path]
    ):
        """Detect JavaScript frameworks from package.json."""
        package_json_path = next((f for f in files if f.name == "package.json"), None)

        if package_json_path:
            try:
                with open(package_json_path, "r") as f:
                    data = json.load(f)

                dependencies = {
                    **data.get("dependencies", {}),
                    **data.get("devDependencies", {}),
                }

                rules = self.detection_rules[ProjectType.JAVASCRIPT]["frameworks"]
                for framework, keywords in rules.items():
                    if any(k in dependencies for k in keywords):
                        result.frameworks.append(framework)

            except (IOError, json.JSONDecodeError) as e:
                # Handle error if package.json is unreadable
                pass

    async def _detect_python_frameworks(
        self, result: ContextDetectionResult, files: List[Path]
    ):
        """Detect Python frameworks."""
        rules = self.detection_rules[ProjectType.PYTHON]["frameworks"]

        # File-based detection
        for framework, file_names in rules.items():
            if any(f.name in file_names for f in files):
                result.frameworks.append(framework)

        # Content-based detection (if needed)
        if "fastapi" not in result.frameworks:
            main_py = next((f for f in files if f.name == "main.py"), None)
            if main_py:
                content = await self._read_file_content(main_py)
                if "from fastapi import FastAPI" in content:
                    result.frameworks.append("fastapi")

    async def _read_file_content(self, file_path: Path, max_lines: int = 50) -> str:
        """Read file content for analysis."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [next(f) for _ in range(max_lines)]
            return "".join(lines)
        except (IOError, StopIteration):
            return ""


# Example usage:
async def main():
    detector = ContextDetector(Path.cwd())
    context = await detector.detect_context()
    print(f"Project Type: {context.project_type.value}")
    print(f"Project Name: {context.project_name}")
    print(f"Project Root: {context.project_root}")
    print(f"Confidence: {context.confidence:.2f}")
    print(f"Key Files: {[f.name for f in context.key_files]}")
    print(f"Languages: {context.language_map}")
    print(f"Frameworks: {context.frameworks}")
    print(f"Docker Detected: {context.has_docker}")
    print(f"Terraform Detected: {context.has_terraform}")


if __name__ == "__main__":
    asyncio.run(main())
