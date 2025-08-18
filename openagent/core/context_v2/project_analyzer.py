"""
Project Context Analyzer for OpenAgent.

Provides Warp-style workspace understanding and project intelligence.
"""

import json
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml


class ProjectType(Enum):
    """Supported project types for context-aware assistance."""
    
    PYTHON = "python"
    JAVASCRIPT = "javascript" 
    TYPESCRIPT = "typescript"
    RUST = "rust"
    GO = "go"
    JAVA = "java"
    CPP = "cpp"
    BASH = "bash"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    TERRAFORM = "terraform"
    UNKNOWN = "unknown"


@dataclass
class GitContext:
    """Git repository context information."""
    
    is_repo: bool = False
    current_branch: Optional[str] = None
    has_uncommitted_changes: bool = False
    ahead_commits: int = 0
    behind_commits: int = 0
    remote_url: Optional[str] = None
    last_commit_hash: Optional[str] = None
    merge_conflict: bool = False
    stash_count: int = 0


@dataclass
class DependencyInfo:
    """Project dependency information."""
    
    package_manager: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    dev_dependencies: List[str] = field(default_factory=list)
    outdated_packages: List[str] = field(default_factory=list)
    security_vulnerabilities: List[str] = field(default_factory=list)


@dataclass 
class WorkspaceContext:
    """Complete workspace context information."""
    
    project_type: ProjectType
    root_path: Path
    project_name: str
    git_context: GitContext
    dependencies: DependencyInfo
    relevant_files: List[Path] = field(default_factory=list)
    build_files: List[Path] = field(default_factory=list)
    config_files: List[Path] = field(default_factory=list)
    test_files: List[Path] = field(default_factory=list)
    entry_points: List[Path] = field(default_factory=list)
    virtual_env: Optional[Path] = None
    container_info: Dict[str, Any] = field(default_factory=dict)


class ProjectContextEngine:
    """
    Enhanced project context analyzer similar to Warp AI.
    
    Provides deep understanding of workspace structure, project type,
    dependencies, and development environment.
    """
    
    def __init__(self, cache_duration: int = 300):
        """Initialize the project context engine.
        
        Args:
            cache_duration: Context cache duration in seconds
        """
        self.cache_duration = cache_duration
        self._context_cache: Dict[str, Tuple[float, WorkspaceContext]] = {}
        
        # Project detection patterns
        self.project_indicators = {
            ProjectType.PYTHON: {
                "files": ["setup.py", "pyproject.toml", "requirements.txt", "Pipfile"],
                "dirs": ["venv", ".venv", "__pycache__"],
                "patterns": ["*.py"]
            },
            ProjectType.JAVASCRIPT: {
                "files": ["package.json", "yarn.lock", "npm-shrinkwrap.json"],
                "dirs": ["node_modules", "dist", "build"],
                "patterns": ["*.js", "*.jsx"]
            },
            ProjectType.TYPESCRIPT: {
                "files": ["tsconfig.json", "package.json"],
                "dirs": ["node_modules", "dist", "build"],
                "patterns": ["*.ts", "*.tsx"]
            },
            ProjectType.RUST: {
                "files": ["Cargo.toml", "Cargo.lock"],
                "dirs": ["target", "src"],
                "patterns": ["*.rs"]
            },
            ProjectType.GO: {
                "files": ["go.mod", "go.sum"],
                "dirs": ["vendor", "bin"],
                "patterns": ["*.go"]
            },
            ProjectType.DOCKER: {
                "files": ["Dockerfile", "docker-compose.yml", "docker-compose.yaml"],
                "dirs": [".docker"],
                "patterns": ["Dockerfile*"]
            },
            ProjectType.KUBERNETES: {
                "files": ["kubernetes.yaml", "kustomization.yaml"],
                "dirs": ["k8s", "kubernetes"],
                "patterns": ["*.yaml", "*.yml"]
            }
        }
    
    async def analyze_workspace(self, path: Optional[Path] = None) -> WorkspaceContext:
        """
        Analyze current workspace for comprehensive context.
        
        Args:
            path: Path to analyze (defaults to current directory)
            
        Returns:
            WorkspaceContext with complete project understanding
        """
        if path is None:
            path = Path.cwd()
        
        cache_key = str(path.resolve())
        
        # Check cache first
        if cache_key in self._context_cache:
            timestamp, context = self._context_cache[cache_key]
            if time.time() - timestamp < self.cache_duration:
                return context
        
        # Analyze the workspace
        context = await self._analyze_workspace_deep(path)
        
        # Cache the result
        import time
        self._context_cache[cache_key] = (time.time(), context)
        
        return context
    
    async def _analyze_workspace_deep(self, path: Path) -> WorkspaceContext:
        """Perform deep workspace analysis."""
        # Detect project type
        project_type = await self.detect_project_type(path)
        
        # Get project name
        project_name = self._get_project_name(path, project_type)
        
        # Analyze git context
        git_context = await self._analyze_git_context(path)
        
        # Analyze dependencies
        dependencies = await self._analyze_dependencies(path, project_type)
        
        # Find relevant files
        relevant_files = await self._find_relevant_files(path, project_type)
        build_files = await self._find_build_files(path, project_type)
        config_files = await self._find_config_files(path, project_type)
        test_files = await self._find_test_files(path, project_type)
        entry_points = await self._find_entry_points(path, project_type)
        
        # Detect virtual environment
        virtual_env = await self._detect_virtual_env(path, project_type)
        
        # Container information
        container_info = await self._analyze_container_setup(path)
        
        return WorkspaceContext(
            project_type=project_type,
            root_path=path,
            project_name=project_name,
            git_context=git_context,
            dependencies=dependencies,
            relevant_files=relevant_files,
            build_files=build_files,
            config_files=config_files,
            test_files=test_files,
            entry_points=entry_points,
            virtual_env=virtual_env,
            container_info=container_info
        )
    
    async def detect_project_type(self, path: Path) -> ProjectType:
        """
        Detect project type based on files and structure.
        
        Args:
            path: Project root path
            
        Returns:
            Detected project type
        """
        scores: Dict[ProjectType, int] = {}
        
        for project_type, indicators in self.project_indicators.items():
            score = 0
            
            # Check for indicator files
            for file_name in indicators["files"]:
                if (path / file_name).exists():
                    score += 3
            
            # Check for indicator directories
            for dir_name in indicators["dirs"]:
                if (path / dir_name).exists():
                    score += 2
            
            # Check for pattern files
            for pattern in indicators["patterns"]:
                if list(path.rglob(pattern)):
                    score += 1
            
            scores[project_type] = score
        
        # Return the project type with highest score
        if scores:
            best_type = max(scores.keys(), key=lambda x: scores[x])
            if scores[best_type] > 0:
                return best_type
        
        return ProjectType.UNKNOWN
    
    def _get_project_name(self, path: Path, project_type: ProjectType) -> str:
        """Extract project name from various sources."""
        # Try package.json for JS/TS projects
        if project_type in [ProjectType.JAVASCRIPT, ProjectType.TYPESCRIPT]:
            package_json = path / "package.json"
            if package_json.exists():
                try:
                    data = json.loads(package_json.read_text())
                    if "name" in data:
                        return data["name"]
                except Exception:
                    pass
        
        # Try pyproject.toml for Python projects
        if project_type == ProjectType.PYTHON:
            pyproject = path / "pyproject.toml"
            if pyproject.exists():
                try:
                    import tomllib
                    data = tomllib.loads(pyproject.read_text())
                    if "project" in data and "name" in data["project"]:
                        return data["project"]["name"]
                except Exception:
                    pass
        
        # Try Cargo.toml for Rust projects
        if project_type == ProjectType.RUST:
            cargo_toml = path / "Cargo.toml"
            if cargo_toml.exists():
                try:
                    import tomllib
                    data = tomllib.loads(cargo_toml.read_text())
                    if "package" in data and "name" in data["package"]:
                        return data["package"]["name"]
                except Exception:
                    pass
        
        # Try go.mod for Go projects
        if project_type == ProjectType.GO:
            go_mod = path / "go.mod"
            if go_mod.exists():
                try:
                    content = go_mod.read_text()
                    lines = content.split('\n')
                    for line in lines:
                        if line.startswith('module '):
                            return line.split(' ', 1)[1].strip()
                except Exception:
                    pass
        
        # Fallback to directory name
        return path.name
    
    async def _analyze_git_context(self, path: Path) -> GitContext:
        """Analyze git repository context."""
        git_context = GitContext()
        
        try:
            # Check if it's a git repo
            result = await self._run_git_command("rev-parse --git-dir", path)
            if result.returncode != 0:
                return git_context
            
            git_context.is_repo = True
            
            # Get current branch
            result = await self._run_git_command("branch --show-current", path)
            if result.returncode == 0:
                git_context.current_branch = result.stdout.strip()
            
            # Check for uncommitted changes
            result = await self._run_git_command("status --porcelain", path)
            if result.returncode == 0:
                git_context.has_uncommitted_changes = bool(result.stdout.strip())
            
            # Get ahead/behind status
            result = await self._run_git_command("status --porcelain=v1 --branch", path)
            if result.returncode == 0:
                status_line = result.stdout.split('\n')[0]
                if '[ahead ' in status_line:
                    git_context.ahead_commits = int(status_line.split('[ahead ')[1].split(']')[0])
                if '[behind ' in status_line:
                    git_context.behind_commits = int(status_line.split('[behind ')[1].split(']')[0])
            
            # Get remote URL
            result = await self._run_git_command("remote get-url origin", path)
            if result.returncode == 0:
                git_context.remote_url = result.stdout.strip()
            
            # Get last commit hash
            result = await self._run_git_command("rev-parse HEAD", path)
            if result.returncode == 0:
                git_context.last_commit_hash = result.stdout.strip()[:8]
            
            # Check for merge conflicts
            result = await self._run_git_command("diff --name-only --diff-filter=U", path)
            if result.returncode == 0:
                git_context.merge_conflict = bool(result.stdout.strip())
            
            # Count stashes
            result = await self._run_git_command("stash list", path)
            if result.returncode == 0:
                git_context.stash_count = len([l for l in result.stdout.split('\n') if l.strip()])
                
        except Exception:
            # Return basic context if git analysis fails
            pass
        
        return git_context
    
    async def _run_git_command(self, command: str, cwd: Path) -> subprocess.CompletedProcess:
        """Run git command safely."""
        try:
            result = await asyncio.create_subprocess_shell(
                f"git {command}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )
            stdout, stderr = await result.communicate()
            return subprocess.CompletedProcess(
                args=command,
                returncode=result.returncode,
                stdout=stdout.decode('utf-8', errors='replace'),
                stderr=stderr.decode('utf-8', errors='replace')
            )
        except Exception:
            return subprocess.CompletedProcess(
                args=command,
                returncode=1,
                stdout="",
                stderr="Command failed"
            )
    
    async def _analyze_dependencies(self, path: Path, project_type: ProjectType) -> DependencyInfo:
        """Analyze project dependencies."""
        deps = DependencyInfo()
        
        if project_type == ProjectType.PYTHON:
            await self._analyze_python_dependencies(path, deps)
        elif project_type in [ProjectType.JAVASCRIPT, ProjectType.TYPESCRIPT]:
            await self._analyze_node_dependencies(path, deps)
        elif project_type == ProjectType.RUST:
            await self._analyze_rust_dependencies(path, deps)
        elif project_type == ProjectType.GO:
            await self._analyze_go_dependencies(path, deps)
        
        return deps
    
    async def _analyze_python_dependencies(self, path: Path, deps: DependencyInfo):
        """Analyze Python project dependencies."""
        deps.package_manager = "pip"
        
        # Check requirements.txt
        req_file = path / "requirements.txt"
        if req_file.exists():
            try:
                content = req_file.read_text()
                deps.dependencies = [line.strip() for line in content.split('\n') 
                                   if line.strip() and not line.startswith('#')]
            except Exception:
                pass
        
        # Check pyproject.toml
        pyproject = path / "pyproject.toml"
        if pyproject.exists():
            try:
                import tomllib
                data = tomllib.loads(pyproject.read_text())
                if "project" in data and "dependencies" in data["project"]:
                    deps.dependencies.extend(data["project"]["dependencies"])
                
                # Check optional dependencies
                if "project" in data and "optional-dependencies" in data["project"]:
                    for group, group_deps in data["project"]["optional-dependencies"].items():
                        if group == "dev":
                            deps.dev_dependencies.extend(group_deps)
                        else:
                            deps.dependencies.extend(group_deps)
            except Exception:
                pass
        
        # Check for virtual environment
        venv_indicators = ["venv", ".venv", "env", ".env"]
        for venv_name in venv_indicators:
            venv_path = path / venv_name
            if venv_path.exists() and (venv_path / "pyvenv.cfg").exists():
                # Found virtual environment
                break
    
    async def _analyze_node_dependencies(self, path: Path, deps: DependencyInfo):
        """Analyze Node.js project dependencies."""
        package_json = path / "package.json"
        if not package_json.exists():
            return
        
        try:
            data = json.loads(package_json.read_text())
            
            # Determine package manager
            if (path / "yarn.lock").exists():
                deps.package_manager = "yarn"
            elif (path / "pnpm-lock.yaml").exists():
                deps.package_manager = "pnpm"
            else:
                deps.package_manager = "npm"
            
            # Extract dependencies
            if "dependencies" in data:
                deps.dependencies = list(data["dependencies"].keys())
            
            if "devDependencies" in data:
                deps.dev_dependencies = list(data["devDependencies"].keys())
                
        except Exception:
            pass
    
    async def _analyze_rust_dependencies(self, path: Path, deps: DependencyInfo):
        """Analyze Rust project dependencies."""
        cargo_toml = path / "Cargo.toml"
        if not cargo_toml.exists():
            return
        
        deps.package_manager = "cargo"
        
        try:
            import tomllib
            data = tomllib.loads(cargo_toml.read_text())
            
            if "dependencies" in data:
                deps.dependencies = list(data["dependencies"].keys())
            
            if "dev-dependencies" in data:
                deps.dev_dependencies = list(data["dev-dependencies"].keys())
                
        except Exception:
            pass
    
    async def _analyze_go_dependencies(self, path: Path, deps: DependencyInfo):
        """Analyze Go project dependencies."""
        go_mod = path / "go.mod"
        if not go_mod.exists():
            return
        
        deps.package_manager = "go"
        
        try:
            content = go_mod.read_text()
            lines = content.split('\n')
            in_require = False
            
            for line in lines:
                line = line.strip()
                if line.startswith('require ('):
                    in_require = True
                    continue
                elif line == ')' and in_require:
                    in_require = False
                    continue
                elif in_require and line:
                    # Extract module name
                    parts = line.split()
                    if parts:
                        deps.dependencies.append(parts[0])
                        
        except Exception:
            pass
    
    async def _find_relevant_files(self, path: Path, project_type: ProjectType) -> List[Path]:
        """Find files relevant to the project type."""
        relevant_files = []
        
        try:
            if project_type in self.project_indicators:
                patterns = self.project_indicators[project_type]["patterns"]
                for pattern in patterns:
                    relevant_files.extend(path.rglob(pattern))
            
            # Limit to reasonable number of files
            return relevant_files[:100]
        except Exception:
            return []
    
    async def _find_build_files(self, path: Path, project_type: ProjectType) -> List[Path]:
        """Find build-related files."""
        build_files = []
        build_patterns = {
            ProjectType.PYTHON: ["setup.py", "pyproject.toml", "setup.cfg", "Makefile"],
            ProjectType.JAVASCRIPT: ["webpack.config.js", "rollup.config.js", "package.json"],
            ProjectType.TYPESCRIPT: ["tsconfig.json", "webpack.config.ts", "package.json"],
            ProjectType.RUST: ["Cargo.toml", "build.rs"],
            ProjectType.GO: ["go.mod", "Makefile"],
            ProjectType.DOCKER: ["Dockerfile", "docker-compose.yml", ".dockerignore"],
        }
        
        if project_type in build_patterns:
            for pattern in build_patterns[project_type]:
                file_path = path / pattern
                if file_path.exists():
                    build_files.append(file_path)
        
        return build_files
    
    async def _find_config_files(self, path: Path, project_type: ProjectType) -> List[Path]:
        """Find configuration files."""
        config_files = []
        config_patterns = [".env*", "*.conf", "*.config.*", "*.yaml", "*.yml", "*.json"]
        
        for pattern in config_patterns:
            config_files.extend(path.glob(pattern))
        
        return config_files[:20]  # Limit to avoid too many files
    
    async def _find_test_files(self, path: Path, project_type: ProjectType) -> List[Path]:
        """Find test files."""
        test_files = []
        test_patterns = {
            ProjectType.PYTHON: ["test_*.py", "*_test.py", "tests/**/*.py"],
            ProjectType.JAVASCRIPT: ["*.test.js", "*.spec.js", "tests/**/*.js"],
            ProjectType.TYPESCRIPT: ["*.test.ts", "*.spec.ts", "tests/**/*.ts"],
            ProjectType.RUST: ["tests/**/*.rs"],
            ProjectType.GO: ["*_test.go"],
        }
        
        if project_type in test_patterns:
            for pattern in test_patterns[project_type]:
                test_files.extend(path.rglob(pattern))
        
        return test_files[:50]  # Limit test files
    
    async def _find_entry_points(self, path: Path, project_type: ProjectType) -> List[Path]:
        """Find main entry points."""
        entry_points = []
        
        if project_type == ProjectType.PYTHON:
            candidates = ["main.py", "app.py", "run.py", "__main__.py"]
            for candidate in candidates:
                file_path = path / candidate
                if file_path.exists():
                    entry_points.append(file_path)
        
        elif project_type in [ProjectType.JAVASCRIPT, ProjectType.TYPESCRIPT]:
            package_json = path / "package.json"
            if package_json.exists():
                try:
                    data = json.loads(package_json.read_text())
                    if "main" in data:
                        main_file = path / data["main"]
                        if main_file.exists():
                            entry_points.append(main_file)
                except Exception:
                    pass
        
        elif project_type == ProjectType.RUST:
            src_main = path / "src" / "main.rs"
            if src_main.exists():
                entry_points.append(src_main)
        
        elif project_type == ProjectType.GO:
            main_go = path / "main.go"
            if main_go.exists():
                entry_points.append(main_go)
        
        return entry_points
    
    async def _detect_virtual_env(self, path: Path, project_type: ProjectType) -> Optional[Path]:
        """Detect virtual environment."""
        if project_type == ProjectType.PYTHON:
            venv_candidates = ["venv", ".venv", "env", ".env"]
            for candidate in venv_candidates:
                venv_path = path / candidate
                if venv_path.exists() and (venv_path / "pyvenv.cfg").exists():
                    return venv_path
        
        return None
    
    async def _analyze_container_setup(self, path: Path) -> Dict[str, Any]:
        """Analyze Docker/container setup."""
        container_info = {}
        
        # Check for Dockerfile
        dockerfile = path / "Dockerfile"
        if dockerfile.exists():
            container_info["has_dockerfile"] = True
            try:
                content = dockerfile.read_text()
                # Extract base image
                for line in content.split('\n'):
                    if line.strip().startswith('FROM '):
                        container_info["base_image"] = line.split()[1]
                        break
            except Exception:
                pass
        
        # Check for docker-compose
        compose_files = ["docker-compose.yml", "docker-compose.yaml"]
        for compose_file in compose_files:
            compose_path = path / compose_file
            if compose_path.exists():
                container_info["has_compose"] = True
                try:
                    data = yaml.safe_load(compose_path.read_text())
                    if "services" in data:
                        container_info["services"] = list(data["services"].keys())
                except Exception:
                    pass
                break
        
        return container_info
    
    def get_relevant_commands(self, context: WorkspaceContext) -> List[str]:
        """Get commands relevant to the current project context."""
        commands = []
        
        # Base commands
        commands.extend(["ls", "pwd", "cd", "cat", "grep", "find"])
        
        # Git commands if in repo
        if context.git_context.is_repo:
            commands.extend([
                "git status", "git add", "git commit", "git push", 
                "git pull", "git branch", "git checkout", "git log"
            ])
        
        # Project-specific commands
        if context.project_type == ProjectType.PYTHON:
            commands.extend([
                "python", "pip install", "pytest", "black", "flake8", 
                "mypy", "python -m venv", "pip freeze"
            ])
            if context.virtual_env:
                commands.append(f"source {context.virtual_env}/bin/activate")
        
        elif context.project_type in [ProjectType.JAVASCRIPT, ProjectType.TYPESCRIPT]:
            if context.dependencies.package_manager == "npm":
                commands.extend(["npm install", "npm run", "npm test", "npm start"])
            elif context.dependencies.package_manager == "yarn":
                commands.extend(["yarn install", "yarn run", "yarn test", "yarn start"])
        
        elif context.project_type == ProjectType.RUST:
            commands.extend([
                "cargo build", "cargo run", "cargo test", "cargo check",
                "cargo fmt", "cargo clippy"
            ])
        
        elif context.project_type == ProjectType.GO:
            commands.extend([
                "go build", "go run", "go test", "go mod tidy",
                "go fmt", "go vet"
            ])
        
        # Docker commands if containerized
        if context.container_info.get("has_dockerfile"):
            commands.extend([
                "docker build", "docker run", "docker ps", "docker logs"
            ])
        
        if context.container_info.get("has_compose"):
            commands.extend([
                "docker-compose up", "docker-compose down", 
                "docker-compose build", "docker-compose logs"
            ])
        
        return commands
    
    def invalidate_cache(self, path: Optional[Path] = None):
        """Invalidate context cache."""
        if path:
            cache_key = str(path.resolve())
            self._context_cache.pop(cache_key, None)
        else:
            self._context_cache.clear()
