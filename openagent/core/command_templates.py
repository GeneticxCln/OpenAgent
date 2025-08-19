"""
Command Templates System for OpenAgent.

Provides pre-built command templates and workflows for common development tasks.
Similar to Warp's workflow automation but more extensible and context-aware.
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from openagent.core.context_v2.project_analyzer import ProjectType, WorkspaceContext


class TemplateCategory(Enum):
    """Categories of command templates."""

    SETUP = "setup"
    DEVELOPMENT = "development"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MAINTENANCE = "maintenance"
    GIT = "git"
    DOCKER = "docker"
    DATABASE = "database"
    DEBUGGING = "debugging"
    PERFORMANCE = "performance"


@dataclass
class CommandStep:
    """A single command step in a template."""

    command: str
    description: str
    optional: bool = False
    confirmation_required: bool = False
    expected_output: Optional[str] = None
    error_recovery: Optional[str] = None
    environment_vars: Dict[str, str] = field(default_factory=dict)


@dataclass
class CommandTemplate:
    """A command template with metadata and steps."""

    name: str
    description: str
    category: TemplateCategory
    steps: List[CommandStep]
    project_types: List[ProjectType] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    estimated_time: int = 60  # seconds
    danger_level: int = 1  # 1-5 scale, 5 being most dangerous
    author: str = "OpenAgent"
    version: str = "1.0.0"
    usage_count: int = 0
    success_rate: float = 1.0


class CommandTemplates:
    """
    Command Templates system for managing and executing development workflows.

    Features:
    - Pre-built templates for common tasks
    - Context-aware template suggestions
    - Template customization and parameterization
    - Success tracking and optimization
    - Safety checks and confirmations
    """

    def __init__(self, templates_dir: Optional[Path] = None):
        """Initialize command templates system."""
        self.templates_dir = (
            templates_dir or Path.home() / ".config" / "openagent" / "templates"
        )
        self.templates_dir.mkdir(parents=True, exist_ok=True)

        # Built-in templates
        self.builtin_templates = self._create_builtin_templates()

        # User templates loaded from disk
        self.user_templates: Dict[str, CommandTemplate] = {}
        self._load_user_templates()

    def _create_builtin_templates(self) -> Dict[str, CommandTemplate]:
        """Create built-in command templates."""
        templates = {}

        # Python Project Setup
        templates["python-setup"] = CommandTemplate(
            name="Python Project Setup",
            description="Set up a new Python project with virtual environment and basic structure",
            category=TemplateCategory.SETUP,
            project_types=[ProjectType.PYTHON],
            steps=[
                CommandStep(
                    command="python -m venv venv",
                    description="Create virtual environment",
                ),
                CommandStep(
                    command="source venv/bin/activate",
                    description="Activate virtual environment",
                ),
                CommandStep(
                    command="pip install --upgrade pip",
                    description="Upgrade pip to latest version",
                ),
                CommandStep(
                    command="pip install pytest black flake8 mypy",
                    description="Install development tools",
                    optional=True,
                ),
                CommandStep(
                    command="mkdir -p src tests docs",
                    description="Create project directories",
                ),
                CommandStep(
                    command="touch src/__init__.py tests/__init__.py",
                    description="Create Python package files",
                ),
                CommandStep(
                    command="echo '# {project_name}\n\nA Python project.\n\n## Setup\n\n```bash\npython -m venv venv\nsource venv/bin/activate\npip install -r requirements.txt\n```' > README.md",
                    description="Create README file",
                ),
                CommandStep(
                    command="touch requirements.txt requirements-dev.txt",
                    description="Create requirements files",
                ),
            ],
            estimated_time=120,
            tags=["python", "setup", "venv"],
        )

        # Git Workflow Templates
        templates["git-feature-branch"] = CommandTemplate(
            name="Git Feature Branch Workflow",
            description="Create a feature branch, make changes, and prepare for merge",
            category=TemplateCategory.GIT,
            steps=[
                CommandStep(
                    command="git checkout main", description="Switch to main branch"
                ),
                CommandStep(
                    command="git pull origin main", description="Update main branch"
                ),
                CommandStep(
                    command="git checkout -b feature/{feature_name}",
                    description="Create and switch to feature branch",
                ),
                CommandStep(
                    command="git push -u origin feature/{feature_name}",
                    description="Push feature branch to remote",
                    optional=True,
                ),
            ],
            estimated_time=60,
            tags=["git", "workflow", "branch"],
        )

        templates["git-commit-push"] = CommandTemplate(
            name="Git Add, Commit, and Push",
            description="Stage changes, commit with message, and push to remote",
            category=TemplateCategory.GIT,
            steps=[
                CommandStep(
                    command="git add .",
                    description="Stage all changes",
                    confirmation_required=True,
                ),
                CommandStep(command="git status", description="Review staged changes"),
                CommandStep(
                    command="git commit -m '{commit_message}'",
                    description="Commit changes with message",
                ),
                CommandStep(
                    command="git push", description="Push to remote repository"
                ),
            ],
            estimated_time=30,
            tags=["git", "commit", "push"],
        )

        # Docker Templates
        templates["docker-app-deploy"] = CommandTemplate(
            name="Docker Application Deployment",
            description="Build Docker image and run container for web application",
            category=TemplateCategory.DEPLOYMENT,
            project_types=[
                ProjectType.DOCKER,
                ProjectType.PYTHON,
                ProjectType.JAVASCRIPT,
            ],
            steps=[
                CommandStep(
                    command="docker build -t {app_name}:latest .",
                    description="Build Docker image",
                ),
                CommandStep(
                    command="docker stop {app_name} 2>/dev/null || true",
                    description="Stop existing container if running",
                    optional=True,
                ),
                CommandStep(
                    command="docker rm {app_name} 2>/dev/null || true",
                    description="Remove existing container if exists",
                    optional=True,
                ),
                CommandStep(
                    command="docker run -d --name {app_name} -p {port}:{port} {app_name}:latest",
                    description="Run container in background",
                ),
                CommandStep(
                    command="docker ps | grep {app_name}",
                    description="Verify container is running",
                ),
            ],
            estimated_time=180,
            danger_level=2,
            tags=["docker", "deploy", "container"],
        )

        # Testing Templates
        templates["python-test-suite"] = CommandTemplate(
            name="Python Test Suite",
            description="Run comprehensive Python test suite with coverage",
            category=TemplateCategory.TESTING,
            project_types=[ProjectType.PYTHON],
            steps=[
                CommandStep(
                    command="python -m pytest tests/ -v",
                    description="Run tests with verbose output",
                ),
                CommandStep(
                    command="python -m pytest tests/ --cov=src --cov-report=html",
                    description="Run tests with coverage report",
                    optional=True,
                ),
                CommandStep(
                    command="python -m flake8 src/",
                    description="Check code style with flake8",
                    optional=True,
                ),
                CommandStep(
                    command="python -m mypy src/",
                    description="Run type checking with mypy",
                    optional=True,
                ),
            ],
            estimated_time=120,
            tags=["python", "testing", "coverage", "linting"],
        )

        # Node.js Templates
        templates["node-setup"] = CommandTemplate(
            name="Node.js Project Setup",
            description="Initialize Node.js project with common tools and structure",
            category=TemplateCategory.SETUP,
            project_types=[ProjectType.JAVASCRIPT, ProjectType.TYPESCRIPT],
            steps=[
                CommandStep(
                    command="npm init -y", description="Initialize package.json"
                ),
                CommandStep(
                    command="npm install --save-dev prettier eslint jest",
                    description="Install development tools",
                ),
                CommandStep(
                    command="mkdir -p src tests docs",
                    description="Create project directories",
                ),
                CommandStep(
                    command="echo 'node_modules/\ndist/\n.env\n*.log' > .gitignore",
                    description="Create .gitignore file",
                ),
                CommandStep(
                    command='echo \'{\n  "semi": true,\n  "trailingComma": "es5",\n  "singleQuote": true,\n  "printWidth": 100\n}\' > .prettierrc',
                    description="Create Prettier configuration",
                ),
            ],
            estimated_time=90,
            tags=["nodejs", "javascript", "setup"],
        )

        # Database Templates
        templates["postgres-backup"] = CommandTemplate(
            name="PostgreSQL Database Backup",
            description="Create a backup of PostgreSQL database",
            category=TemplateCategory.DATABASE,
            steps=[
                CommandStep(
                    command="pg_dump -h {host} -U {username} -d {database} > backup_{database}_{timestamp}.sql",
                    description="Create database dump",
                    confirmation_required=True,
                ),
                CommandStep(
                    command="gzip backup_{database}_{timestamp}.sql",
                    description="Compress backup file",
                ),
                CommandStep(
                    command="ls -lh backup_{database}_{timestamp}.sql.gz",
                    description="Show backup file size",
                ),
            ],
            estimated_time=300,  # Depends on database size
            danger_level=1,
            tags=["database", "postgres", "backup"],
        )

        # Performance Templates
        templates["system-monitor"] = CommandTemplate(
            name="System Performance Monitoring",
            description="Monitor system performance and resource usage",
            category=TemplateCategory.PERFORMANCE,
            steps=[
                CommandStep(
                    command="top -n 1 -b",
                    description="Show current process information",
                ),
                CommandStep(command="free -h", description="Show memory usage"),
                CommandStep(command="df -h", description="Show disk usage"),
                CommandStep(
                    command="iostat -x 1 3",
                    description="Show I/O statistics",
                    optional=True,
                ),
            ],
            estimated_time=30,
            tags=["monitoring", "performance", "system"],
        )

        # Debugging Templates
        templates["python-debug"] = CommandTemplate(
            name="Python Application Debug",
            description="Debug Python application with common diagnostic commands",
            category=TemplateCategory.DEBUGGING,
            project_types=[ProjectType.PYTHON],
            steps=[
                CommandStep(
                    command="python -c 'import sys; print(sys.version)'",
                    description="Check Python version",
                ),
                CommandStep(command="pip list", description="List installed packages"),
                CommandStep(
                    command="python -m py_compile {script_name}",
                    description="Check for syntax errors",
                ),
                CommandStep(
                    command="python -m trace --trace {script_name}",
                    description="Trace script execution",
                    optional=True,
                ),
            ],
            estimated_time=60,
            tags=["python", "debugging", "trace"],
        )

        return templates

    def get_template(self, template_name: str) -> Optional[CommandTemplate]:
        """Get a specific template by name."""
        # Check built-in templates first
        if template_name in self.builtin_templates:
            return self.builtin_templates[template_name]

        # Check user templates
        if template_name in self.user_templates:
            return self.user_templates[template_name]

        return None

    def list_templates(
        self,
        category: Optional[TemplateCategory] = None,
        project_type: Optional[ProjectType] = None,
        tags: Optional[List[str]] = None,
    ) -> List[CommandTemplate]:
        """List templates with optional filtering."""
        all_templates = {**self.builtin_templates, **self.user_templates}
        filtered_templates = []

        for template in all_templates.values():
            # Filter by category
            if category and template.category != category:
                continue

            # Filter by project type
            if (
                project_type
                and template.project_types
                and project_type not in template.project_types
            ):
                continue

            # Filter by tags
            if tags and not any(tag in template.tags for tag in tags):
                continue

            filtered_templates.append(template)

        # Sort by usage count and success rate
        return sorted(
            filtered_templates,
            key=lambda t: (t.usage_count * t.success_rate, t.name),
            reverse=True,
        )

    def suggest_templates(
        self, context: WorkspaceContext, recent_commands: List[str] = None
    ) -> List[CommandTemplate]:
        """Suggest templates based on workspace context."""
        suggestions = []
        recent_commands = recent_commands or []

        # Get templates for current project type
        project_templates = self.list_templates(project_type=context.project_type)
        suggestions.extend(project_templates[:3])

        # Git-specific suggestions if in git repo
        if context.git_context.is_repo:
            git_templates = self.list_templates(category=TemplateCategory.GIT)
            suggestions.extend(git_templates[:2])

        # Docker suggestions if Docker files present
        if context.container_info.get("has_dockerfile") or context.container_info.get(
            "has_compose"
        ):
            docker_templates = self.list_templates(category=TemplateCategory.DOCKER)
            suggestions.extend(docker_templates[:2])

        # Suggest based on recent commands
        if recent_commands:
            for cmd in recent_commands[-5:]:  # Last 5 commands
                if cmd.startswith("git"):
                    git_templates = self.list_templates(category=TemplateCategory.GIT)
                    suggestions.extend(git_templates[:1])
                elif cmd.startswith("docker"):
                    docker_templates = self.list_templates(
                        category=TemplateCategory.DOCKER
                    )
                    suggestions.extend(docker_templates[:1])
                elif "test" in cmd:
                    test_templates = self.list_templates(
                        category=TemplateCategory.TESTING
                    )
                    suggestions.extend(test_templates[:1])

        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for template in suggestions:
            if template.name not in seen:
                seen.add(template.name)
                unique_suggestions.append(template)

        return unique_suggestions[:10]

    def execute_template(
        self,
        template_name: str,
        parameters: Dict[str, str] = None,
        workspace: Optional[WorkspaceContext] = None,
        dry_run: bool = False,
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Execute a command template.

        Args:
            template_name: Name of template to execute
            parameters: Template parameter substitutions
            workspace: Current workspace context
            dry_run: If True, only show commands without executing

        Returns:
            Tuple of (success, executed_commands, error_messages)
        """
        template = self.get_template(template_name)
        if not template:
            return False, [], [f"Template '{template_name}' not found"]

        parameters = parameters or {}
        executed_commands = []
        error_messages = []

        # Add default parameters
        default_params = self._get_default_parameters(workspace)
        parameters = {**default_params, **parameters}

        # Safety check for dangerous templates
        if template.danger_level >= 3 and not dry_run:
            confirmation = input(
                f"Template '{template.name}' has danger level {template.danger_level}. Continue? (y/N): "
            )
            if confirmation.lower() != "y":
                return False, [], ["Template execution cancelled by user"]

        for step in template.steps:
            # Skip optional steps if they fail requirements
            if step.optional and not self._check_step_requirements(step, workspace):
                continue

            # Substitute parameters in command
            command = self._substitute_parameters(step.command, parameters)

            if dry_run:
                executed_commands.append(f"[DRY RUN] {command}")
                continue

            # Confirmation for dangerous steps
            if step.confirmation_required:
                print(f"About to execute: {command}")
                print(f"Description: {step.description}")
                confirmation = input("Continue? (y/N): ")
                if confirmation.lower() != "y":
                    error_messages.append(f"Step cancelled: {step.description}")
                    continue

            # Execute command (this is a simplified version - in practice you'd use proper command execution)
            try:
                # Here you would integrate with the actual command executor
                executed_commands.append(command)
                print(f"Executing: {command}")

                # For now, just simulate execution
                # result = subprocess.run(command, shell=True, capture_output=True, text=True)
                # if result.returncode != 0 and not step.optional:
                #     error_messages.append(f"Command failed: {command}\nError: {result.stderr}")
                #     break

            except Exception as e:
                error_messages.append(f"Failed to execute: {command}\nError: {str(e)}")
                if not step.optional:
                    break

        # Update usage statistics
        if not dry_run:
            self._update_template_stats(template_name, len(error_messages) == 0)

        success = len(error_messages) == 0
        return success, executed_commands, error_messages

    def create_custom_template(
        self,
        name: str,
        description: str,
        steps: List[Dict[str, Any]],
        category: TemplateCategory = TemplateCategory.DEVELOPMENT,
        **kwargs,
    ) -> bool:
        """Create a custom user template."""
        try:
            command_steps = []
            for step_data in steps:
                command_steps.append(
                    CommandStep(
                        command=step_data["command"],
                        description=step_data.get("description", ""),
                        optional=step_data.get("optional", False),
                        confirmation_required=step_data.get(
                            "confirmation_required", False
                        ),
                    )
                )

            template = CommandTemplate(
                name=name,
                description=description,
                category=category,
                steps=command_steps,
                **kwargs,
            )

            self.user_templates[name] = template
            self._save_user_template(template)
            return True

        except Exception as e:
            print(f"Failed to create template: {e}")
            return False

    def _substitute_parameters(self, command: str, parameters: Dict[str, str]) -> str:
        """Substitute template parameters in command."""
        result = command
        for key, value in parameters.items():
            result = result.replace(f"{{{key}}}", value)
        return result

    def _get_default_parameters(
        self, workspace: Optional[WorkspaceContext]
    ) -> Dict[str, str]:
        """Get default template parameters from workspace context."""
        params = {
            "timestamp": str(int(time.time())),
            "date": time.strftime("%Y-%m-%d"),
            "time": time.strftime("%H-%M-%S"),
        }

        if workspace:
            params.update(
                {
                    "project_name": workspace.project_name,
                    "project_type": workspace.project_type.value,
                    "port": "8000",  # Default port
                    "host": "localhost",
                    "username": "user",
                    "database": workspace.project_name.lower().replace("-", "_"),
                }
            )

            if workspace.git_context.current_branch:
                params["branch"] = workspace.git_context.current_branch

        return params

    def _check_step_requirements(
        self, step: CommandStep, workspace: Optional[WorkspaceContext]
    ) -> bool:
        """Check if step requirements are met."""
        # For now, assume all requirements are met
        # In practice, this would check for required tools, files, etc.
        return True

    def _update_template_stats(self, template_name: str, success: bool):
        """Update template usage statistics."""
        template = self.get_template(template_name)
        if template:
            template.usage_count += 1
            if success:
                # Update success rate with weighted average
                old_success_rate = template.success_rate
                new_success_rate = (
                    old_success_rate * (template.usage_count - 1) + 1
                ) / template.usage_count
                template.success_rate = new_success_rate
            else:
                new_success_rate = (
                    template.success_rate * (template.usage_count - 1)
                ) / template.usage_count
                template.success_rate = new_success_rate

            # Save updated template if it's a user template
            if template_name in self.user_templates:
                self._save_user_template(template)

    def _load_user_templates(self):
        """Load user templates from disk."""
        templates_file = self.templates_dir / "user_templates.json"
        if templates_file.exists():
            try:
                with open(templates_file, "r") as f:
                    data = json.load(f)

                for template_data in data:
                    steps = []
                    for step_data in template_data.get("steps", []):
                        steps.append(
                            CommandStep(
                                command=step_data["command"],
                                description=step_data.get("description", ""),
                                optional=step_data.get("optional", False),
                                confirmation_required=step_data.get(
                                    "confirmation_required", False
                                ),
                            )
                        )

                    template = CommandTemplate(
                        name=template_data["name"],
                        description=template_data["description"],
                        category=TemplateCategory(
                            template_data.get("category", "development")
                        ),
                        steps=steps,
                        project_types=[
                            ProjectType(pt)
                            for pt in template_data.get("project_types", [])
                        ],
                        requirements=template_data.get("requirements", []),
                        tags=template_data.get("tags", []),
                        estimated_time=template_data.get("estimated_time", 60),
                        danger_level=template_data.get("danger_level", 1),
                        usage_count=template_data.get("usage_count", 0),
                        success_rate=template_data.get("success_rate", 1.0),
                    )

                    self.user_templates[template.name] = template

            except Exception as e:
                print(f"Failed to load user templates: {e}")

    def _save_user_template(self, template: CommandTemplate):
        """Save a user template to disk."""
        templates_file = self.templates_dir / "user_templates.json"

        # Load existing templates
        existing_templates = []
        if templates_file.exists():
            try:
                with open(templates_file, "r") as f:
                    existing_templates = json.load(f)
            except Exception:
                existing_templates = []

        # Update or add template
        template_data = {
            "name": template.name,
            "description": template.description,
            "category": template.category.value,
            "steps": [
                {
                    "command": step.command,
                    "description": step.description,
                    "optional": step.optional,
                    "confirmation_required": step.confirmation_required,
                }
                for step in template.steps
            ],
            "project_types": [pt.value for pt in template.project_types],
            "requirements": template.requirements,
            "tags": template.tags,
            "estimated_time": template.estimated_time,
            "danger_level": template.danger_level,
            "usage_count": template.usage_count,
            "success_rate": template.success_rate,
        }

        # Replace existing or add new
        found = False
        for i, existing in enumerate(existing_templates):
            if existing["name"] == template.name:
                existing_templates[i] = template_data
                found = True
                break

        if not found:
            existing_templates.append(template_data)

        # Save to file
        try:
            with open(templates_file, "w") as f:
                json.dump(existing_templates, f, indent=2)
        except Exception as e:
            print(f"Failed to save template: {e}")


# Factory function for easy creation
def create_command_templates(templates_dir: Optional[Path] = None) -> CommandTemplates:
    """Create a command templates system with default configuration."""
    return CommandTemplates(templates_dir)
