"""
Environment Detector for OpenAgent.

Provides comprehensive environment context awareness for intelligent assistance.
"""

import asyncio
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import psutil


@dataclass
class ProcessContext:
    """Information about running processes."""

    active_processes: List[Dict[str, Any]] = field(default_factory=list)
    development_servers: List[str] = field(default_factory=list)
    databases: List[str] = field(default_factory=list)
    editors: List[str] = field(default_factory=list)
    terminals: List[str] = field(default_factory=list)


@dataclass
class SystemContext:
    """System resource and configuration context."""

    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    load_average: Optional[Tuple[float, float, float]] = None
    network_active: bool = False

    # System info
    platform: str = ""
    architecture: str = ""
    kernel_version: str = ""
    shell: str = ""

    # Resource limits
    memory_available_gb: float = 0.0
    disk_available_gb: float = 0.0


@dataclass
class DevEnvironment:
    """Development environment context."""

    containers: Dict[str, Any] = field(default_factory=dict)
    virtual_environments: List[Path] = field(default_factory=list)
    kubernetes_context: Optional[str] = None
    cloud_context: Dict[str, Any] = field(default_factory=dict)
    development_tools: Set[str] = field(default_factory=set)


@dataclass
class EnvironmentContext:
    """Complete environment context information."""

    system: SystemContext
    processes: ProcessContext
    development: DevEnvironment
    timestamp: float = field(default_factory=lambda: time.time())


class EnvironmentDetector:
    """
    Advanced environment detection and analysis.

    Provides comprehensive understanding of the system state,
    running processes, and development environment.
    """

    def __init__(self):
        """Initialize environment detector."""
        self.development_process_patterns = {
            "webpack": ["webpack", "webpack-dev-server"],
            "vite": ["vite"],
            "react": ["react-scripts"],
            "node": ["node", "npm", "yarn"],
            "python": ["python", "python3", "uvicorn", "gunicorn", "flask", "django"],
            "rust": ["cargo"],
            "go": ["go"],
            "docker": ["docker", "dockerd", "docker-compose"],
            "kubernetes": ["kubectl", "minikube", "k3s"],
            "databases": ["postgres", "mysql", "mongodb", "redis", "sqlite"],
            "editors": ["code", "vim", "nvim", "emacs", "sublime"],
            "terminals": ["zsh", "bash", "fish", "tmux", "screen"],
        }

    async def detect_environment(self) -> EnvironmentContext:
        """
        Detect complete environment context.

        Returns:
            EnvironmentContext with comprehensive environment information
        """
        import time

        # Analyze system context
        system_context = await self._analyze_system_context()

        # Analyze running processes
        process_context = await self._analyze_process_context()

        # Analyze development environment
        dev_environment = await self._analyze_dev_environment()

        return EnvironmentContext(
            system=system_context,
            processes=process_context,
            development=dev_environment,
            timestamp=time.time(),
        )

    async def _analyze_system_context(self) -> SystemContext:
        """Analyze system resource context."""
        # Get system information
        uname = platform.uname()

        # Get resource usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Get load average (Unix only)
        load_avg = None
        try:
            load_avg = os.getloadavg()
        except (OSError, AttributeError):
            pass

        # Detect shell
        shell = os.environ.get("SHELL", "").split("/")[-1]

        # Check network connectivity
        network_active = await self._check_network_connectivity()

        return SystemContext(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=(disk.used / disk.total) * 100,
            load_average=load_avg,
            network_active=network_active,
            platform=uname.system,
            architecture=uname.machine,
            kernel_version=uname.release,
            shell=shell,
            memory_available_gb=memory.available / (1024**3),
            disk_available_gb=disk.free / (1024**3),
        )

    async def _analyze_process_context(self) -> ProcessContext:
        """Analyze running processes for development context."""
        context = ProcessContext()

        try:
            # Get all running processes
            processes = []
            for proc in psutil.process_iter(["pid", "name", "cmdline", "cpu_percent"]):
                try:
                    pinfo = proc.info
                    if pinfo["name"]:
                        processes.append(pinfo)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            context.active_processes = processes[:20]  # Limit to top 20

            # Categorize development-related processes
            for proc in processes:
                proc_name = proc["name"].lower()
                cmdline = " ".join(proc.get("cmdline", [])).lower()

                # Check against development patterns
                for category, patterns in self.development_process_patterns.items():
                    for pattern in patterns:
                        if pattern in proc_name or pattern in cmdline:
                            if category == "databases":
                                context.databases.append(proc_name)
                            elif category == "editors":
                                context.editors.append(proc_name)
                            elif category == "terminals":
                                context.terminals.append(proc_name)
                            else:
                                context.development_servers.append(
                                    f"{proc_name} (pid: {proc['pid']})"
                                )
                            break

        except Exception:
            # Return empty context if analysis fails
            pass

        return context

    async def _analyze_dev_environment(self) -> DevEnvironment:
        """Analyze development environment setup."""
        dev_env = DevEnvironment()

        # Check for Docker
        await self._detect_docker_environment(dev_env)

        # Check for virtual environments
        await self._detect_virtual_environments(dev_env)

        # Check for Kubernetes
        await self._detect_kubernetes_context(dev_env)

        # Check for cloud contexts
        await self._detect_cloud_context(dev_env)

        # Detect development tools
        await self._detect_development_tools(dev_env)

        return dev_env

    async def _detect_docker_environment(self, dev_env: DevEnvironment):
        """Detect Docker containers and environment."""
        try:
            # Check if Docker is running
            result = await self._run_command("docker ps --format json")
            if result.returncode == 0 and result.stdout.strip():
                containers = []
                for line in result.stdout.strip().split("\n"):
                    try:
                        container = json.loads(line)
                        containers.append(
                            {
                                "name": container.get("Names", ""),
                                "image": container.get("Image", ""),
                                "status": container.get("Status", ""),
                                "ports": container.get("Ports", ""),
                            }
                        )
                    except json.JSONDecodeError:
                        pass

                dev_env.containers["docker"] = containers

            # Check for docker-compose
            if (
                Path("docker-compose.yml").exists()
                or Path("docker-compose.yaml").exists()
            ):
                result = await self._run_command("docker-compose ps --format json")
                if result.returncode == 0:
                    dev_env.containers["compose"] = True

        except Exception:
            pass

    async def _detect_virtual_environments(self, dev_env: DevEnvironment):
        """Detect Python virtual environments."""
        venv_paths = []

        # Check common virtual environment locations
        candidates = [
            Path.cwd() / "venv",
            Path.cwd() / ".venv",
            Path.cwd() / "env",
            Path.home() / ".virtualenvs",
            Path("/opt/conda/envs") if Path("/opt/conda").exists() else None,
        ]

        for candidate in candidates:
            if candidate and candidate.exists():
                if (candidate / "pyvenv.cfg").exists():
                    venv_paths.append(candidate)
                elif candidate.name == ".virtualenvs":
                    # Check for virtualenvwrapper environments
                    for subdir in candidate.iterdir():
                        if subdir.is_dir() and (subdir / "pyvenv.cfg").exists():
                            venv_paths.append(subdir)

        dev_env.virtual_environments = venv_paths

    async def _detect_kubernetes_context(self, dev_env: DevEnvironment):
        """Detect Kubernetes context."""
        try:
            result = await self._run_command("kubectl config current-context")
            if result.returncode == 0:
                dev_env.kubernetes_context = result.stdout.strip()
        except Exception:
            pass

    async def _detect_cloud_context(self, dev_env: DevEnvironment):
        """Detect cloud provider contexts."""
        cloud_contexts = {}

        # AWS
        try:
            result = await self._run_command(
                "aws sts get-caller-identity --output json"
            )
            if result.returncode == 0:
                identity = json.loads(result.stdout)
                cloud_contexts["aws"] = {
                    "account": identity.get("Account"),
                    "user": (
                        identity.get("Arn", "").split("/")[-1]
                        if identity.get("Arn")
                        else None
                    ),
                }
        except Exception:
            pass

        # GCP
        try:
            result = await self._run_command("gcloud config get-value project")
            if result.returncode == 0:
                cloud_contexts["gcp"] = {"project": result.stdout.strip()}
        except Exception:
            pass

        # Azure
        try:
            result = await self._run_command("az account show --output json")
            if result.returncode == 0:
                account = json.loads(result.stdout)
                cloud_contexts["azure"] = {
                    "subscription": account.get("name"),
                    "tenant": account.get("tenantId"),
                }
        except Exception:
            pass

        dev_env.cloud_context = cloud_contexts

    async def _detect_development_tools(self, dev_env: DevEnvironment):
        """Detect installed development tools."""
        tools = set()

        # Common development tools to check
        tool_commands = [
            "git",
            "docker",
            "kubectl",
            "helm",
            "terraform",
            "python",
            "python3",
            "node",
            "npm",
            "yarn",
            "pnpm",
            "cargo",
            "rustc",
            "go",
            "java",
            "javac",
            "vim",
            "nvim",
            "emacs",
            "code",
            "make",
            "cmake",
            "gcc",
            "clang",
        ]

        for tool in tool_commands:
            try:
                result = await self._run_command(f"command -v {tool}")
                if result.returncode == 0:
                    tools.add(tool)
            except Exception:
                pass

        dev_env.development_tools = tools

    async def _check_network_connectivity(self) -> bool:
        """Check if network is available."""
        try:
            result = await self._run_command("ping -c 1 -W 1 8.8.8.8")
            return result.returncode == 0
        except Exception:
            return False

    async def _run_command(self, command: str) -> subprocess.CompletedProcess:
        """Run a system command safely."""
        try:
            process = await asyncio.create_subprocess_shell(
                command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            return subprocess.CompletedProcess(
                args=command,
                returncode=process.returncode,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
            )
        except Exception:
            return subprocess.CompletedProcess(
                args=command, returncode=1, stdout="", stderr="Command failed"
            )

    def get_context_summary(self, env_context: EnvironmentContext) -> str:
        """Get a human-readable summary of the environment context."""
        summary = []

        # System summary
        system = env_context.system
        summary.append(f"System: {system.platform} {system.architecture}")
        summary.append(
            f"Resources: CPU {system.cpu_percent:.1f}%, RAM {system.memory_percent:.1f}%, Disk {system.disk_percent:.1f}%"
        )

        # Development environment
        dev = env_context.development
        if dev.containers:
            summary.append(
                f"Containers: {len(dev.containers.get('docker', []))} running"
            )

        if dev.virtual_environments:
            summary.append(f"Python venvs: {len(dev.virtual_environments)} found")

        if dev.kubernetes_context:
            summary.append(f"K8s context: {dev.kubernetes_context}")

        # Development tools
        if dev.development_tools:
            tools = list(dev.development_tools)[:5]  # Show top 5
            summary.append(f"Dev tools: {', '.join(tools)}")

        # Active processes
        if env_context.processes.development_servers:
            servers = env_context.processes.development_servers[:3]  # Top 3
            summary.append(f"Dev servers: {', '.join(servers)}")

        return "; ".join(summary)

    def suggest_optimization_commands(
        self, env_context: EnvironmentContext
    ) -> List[str]:
        """Suggest commands based on current environment state."""
        suggestions = []

        system = env_context.system

        # High CPU usage suggestions
        if system.cpu_percent > 80:
            suggestions.extend(["top -o cpu", "ps aux --sort=-%cpu | head -10", "htop"])

        # High memory usage suggestions
        if system.memory_percent > 80:
            suggestions.extend(
                [
                    "free -h",
                    "ps aux --sort=-%mem | head -10",
                    (
                        "docker system prune"
                        if "docker" in env_context.development.development_tools
                        else None
                    ),
                ]
            )

        # Low disk space suggestions
        if system.disk_percent > 90:
            suggestions.extend(
                [
                    "df -h",
                    "du -sh * | sort -hr | head -10",
                    (
                        "docker system prune -a"
                        if "docker" in env_context.development.development_tools
                        else None
                    ),
                    (
                        "npm cache clean --force"
                        if "npm" in env_context.development.development_tools
                        else None
                    ),
                ]
            )

        # Filter out None values
        return [cmd for cmd in suggestions if cmd is not None]
