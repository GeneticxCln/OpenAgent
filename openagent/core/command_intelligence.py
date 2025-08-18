"""
Command Intelligence System for OpenAgent.

Provides Warp-style intelligent command completion, auto-correction,
and context-aware suggestions for terminal commands.
"""

import asyncio
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from openagent.core.context_v2.history_intelligence import HistoryIntelligence, PredictionContext
from openagent.core.context_v2.project_analyzer import ProjectContextEngine, ProjectType


class SuggestionType(Enum):
    """Types of command suggestions."""
    
    COMPLETION = "completion"
    CORRECTION = "correction" 
    FLAG = "flag"
    TEMPLATE = "template"
    HISTORY = "history"
    CONTEXT = "context"


@dataclass
class CommandSuggestion:
    """A command suggestion with metadata."""
    
    text: str
    type: SuggestionType
    confidence: float
    description: str = ""
    context_hint: str = ""
    executable: bool = True


@dataclass
class CompletionContext:
    """Context for command completion."""
    
    current_directory: Path
    project_type: Optional[ProjectType] = None
    git_repo: bool = False
    git_branch: Optional[str] = None
    recent_commands: List[str] = field(default_factory=list)
    environment_vars: Dict[str, str] = field(default_factory=dict)


class CommandCompletionEngine:
    """
    Intelligent command completion system with context awareness.
    
    Features:
    - Context-aware command suggestions
    - Auto-correction of common typos
    - Smart flag and argument completion
    - Integration with command history
    - Project-type specific suggestions
    """
    
    def __init__(
        self,
        project_engine: Optional[ProjectContextEngine] = None,
        history_engine: Optional[HistoryIntelligence] = None
    ):
        """Initialize command completion engine."""
        self.project_engine = project_engine or ProjectContextEngine()
        self.history_engine = history_engine or HistoryIntelligence()
        
        # Common command corrections
        self.corrections = {
            'gi': 'git',
            'gti': 'git',
            'got': 'git',
            'cd..': 'cd ..',
            'lls': 'ls',
            'l': 'ls',
            'll': 'ls -la',
            'la': 'ls -la',
            'pythno': 'python',
            'python3': 'python',
            'mkdri': 'mkdir',
            'mkdie': 'mkdir',
            'toch': 'touch',
            'nano': 'vim',  # Controversial but configurable
            'claer': 'clear',
            'clea': 'clear',
            'exti': 'exit',
            'q': 'exit',
            'quit': 'exit',
        }
        
        # Command aliases and shortcuts
        self.aliases = {
            'status': 'git status',
            'log': 'git log --oneline -10',
            'diff': 'git diff',
            'add': 'git add',
            'commit': 'git commit -m',
            'push': 'git push',
            'pull': 'git pull',
            'branch': 'git branch',
            'checkout': 'git checkout',
            'serve': 'python -m http.server',
            'json': 'python -m json.tool',
            'ports': 'netstat -tulpn',
            'processes': 'ps aux',
        }
        
        # Command flags database
        self.command_flags = {
            'git': {
                'status': ['--short', '--branch', '--porcelain'],
                'log': ['--oneline', '--graph', '--decorate', '--all', '--since=', '--author='],
                'diff': ['--staged', '--cached', '--name-only', '--stat'],
                'add': ['--all', '--interactive', '--patch'],
                'commit': ['--message=', '--amend', '--no-edit', '--all'],
                'push': ['--force-with-lease', '--set-upstream', '--tags'],
                'pull': ['--rebase', '--no-edit', '--strategy='],
                'branch': ['--list', '--delete', '--move', '--remote'],
                'checkout': ['--branch', '--track', '--force'],
            },
            'docker': {
                'run': ['--interactive', '--tty', '--detach', '--rm', '--publish=', '--volume='],
                'build': ['--tag=', '--file=', '--no-cache', '--pull'],
                'ps': ['--all', '--quiet', '--filter='],
                'logs': ['--follow', '--tail=', '--since='],
                'exec': ['--interactive', '--tty'],
            },
            'ls': ['--long', '--all', '--human-readable', '--time-style=', '--color='],
            'grep': ['--recursive', '--ignore-case', '--line-number', '--color=', '--include='],
            'find': ['-name', '-type', '-size', '-mtime', '-exec', '-print0'],
            'curl': ['--location', '--silent', '--header=', '--data=', '--output='],
        }
        
        # Project-specific command suggestions
        self.project_commands = {
            ProjectType.PYTHON: [
                'python -m pip install',
                'python -m pytest',
                'python -m black .',
                'python -m flake8',
                'python -m mypy',
                'python -m venv venv',
                'source venv/bin/activate',
                'pip freeze > requirements.txt',
                'python setup.py sdist bdist_wheel',
            ],
            ProjectType.JAVASCRIPT: [
                'npm install',
                'npm run start',
                'npm run build',
                'npm run test',
                'npm run lint',
                'yarn install',
                'yarn start',
                'yarn build',
                'npx create-react-app',
            ],
            ProjectType.TYPESCRIPT: [
                'tsc --init',
                'tsc --watch',
                'npm run type-check',
                'npx tsc --noEmit',
            ],
            ProjectType.RUST: [
                'cargo build',
                'cargo run',
                'cargo test',
                'cargo check',
                'cargo fmt',
                'cargo clippy',
                'cargo doc --open',
                'cargo update',
            ],
            ProjectType.GO: [
                'go build',
                'go run .',
                'go test ./...',
                'go mod tidy',
                'go mod init',
                'go fmt ./...',
                'go vet ./...',
                'go get -u',
            ],
            ProjectType.DOCKER: [
                'docker build -t app .',
                'docker run -p 8000:8000 app',
                'docker ps',
                'docker images',
                'docker-compose up',
                'docker-compose down',
                'docker system prune',
            ],
        }
    
    async def suggest_commands(
        self,
        partial: str,
        context: CompletionContext,
        max_suggestions: int = 10
    ) -> List[CommandSuggestion]:
        """
        Generate intelligent command suggestions based on partial input.
        
        Args:
            partial: Partial command input
            context: Current completion context
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of command suggestions ordered by relevance
        """
        suggestions = []
        partial = partial.strip()
        
        if not partial:
            # If no input, suggest common commands based on context
            suggestions.extend(await self._suggest_context_commands(context))
        else:
            # Auto-correction suggestions
            corrections = self._get_corrections(partial)
            suggestions.extend(corrections)
            
            # Completion suggestions
            completions = await self._get_completions(partial, context)
            suggestions.extend(completions)
            
            # History-based suggestions
            history_suggestions = await self._get_history_suggestions(partial, context)
            suggestions.extend(history_suggestions)
            
            # Alias expansion
            alias_suggestions = self._get_alias_suggestions(partial)
            suggestions.extend(alias_suggestions)
        
        # Rank and filter suggestions
        ranked_suggestions = self._rank_suggestions(suggestions, partial, context)
        
        return ranked_suggestions[:max_suggestions]
    
    async def suggest_flags(
        self,
        command: str,
        partial_flag: str = "",
        context: Optional[CompletionContext] = None
    ) -> List[CommandSuggestion]:
        """
        Suggest appropriate flags for a command.
        
        Args:
            command: Base command (e.g., 'git status')
            partial_flag: Partial flag input (e.g., '--st')
            context: Current completion context
            
        Returns:
            List of flag suggestions
        """
        suggestions = []
        
        # Parse command into parts
        cmd_parts = command.split()
        if not cmd_parts:
            return suggestions
        
        base_cmd = cmd_parts[0]
        subcmd = cmd_parts[1] if len(cmd_parts) > 1 else None
        
        # Get flags from database
        if base_cmd in self.command_flags:
            flag_data = self.command_flags[base_cmd]
            
            if isinstance(flag_data, dict) and subcmd and subcmd in flag_data:
                flags = flag_data[subcmd]
            elif isinstance(flag_data, list):
                flags = flag_data
            else:
                flags = []
            
            for flag in flags:
                if not partial_flag or flag.startswith(partial_flag):
                    suggestions.append(CommandSuggestion(
                        text=flag,
                        type=SuggestionType.FLAG,
                        confidence=0.8,
                        description=f"Flag for {command}",
                        context_hint=f"Common flag for {base_cmd}"
                    ))
        
        return suggestions
    
    def auto_correct_command(self, command: str) -> Optional[str]:
        """
        Auto-correct common command typos.
        
        Args:
            command: Command to check for corrections
            
        Returns:
            Corrected command if correction found, None otherwise
        """
        command = command.strip()
        
        # Direct corrections
        if command in self.corrections:
            return self.corrections[command]
        
        # Fuzzy matching for more complex corrections
        words = command.split()
        if words and words[0] in self.corrections:
            corrected_words = [self.corrections[words[0]]] + words[1:]
            return ' '.join(corrected_words)
        
        return None
    
    async def complete_arguments(
        self,
        command: str,
        partial_arg: str,
        context: CompletionContext
    ) -> List[CommandSuggestion]:
        """
        Complete command arguments intelligently.
        
        Args:
            command: Base command
            partial_arg: Partial argument to complete
            context: Current completion context
            
        Returns:
            List of argument completion suggestions
        """
        suggestions = []
        
        # File path completion
        if self._expects_file_path(command):
            file_suggestions = await self._complete_file_paths(partial_arg, context)
            suggestions.extend(file_suggestions)
        
        # Git-specific completions
        if command.startswith('git'):
            git_suggestions = await self._complete_git_arguments(command, partial_arg, context)
            suggestions.extend(git_suggestions)
        
        # Docker-specific completions
        if command.startswith('docker'):
            docker_suggestions = await self._complete_docker_arguments(command, partial_arg, context)
            suggestions.extend(docker_suggestions)
        
        # Environment variable completion
        if partial_arg.startswith('$'):
            env_suggestions = self._complete_environment_vars(partial_arg, context)
            suggestions.extend(env_suggestions)
        
        return suggestions
    
    def _get_corrections(self, partial: str) -> List[CommandSuggestion]:
        """Get auto-correction suggestions."""
        suggestions = []
        correction = self.auto_correct_command(partial)
        
        if correction:
            suggestions.append(CommandSuggestion(
                text=correction,
                type=SuggestionType.CORRECTION,
                confidence=0.9,
                description=f"Auto-correction for '{partial}'",
                context_hint="Common typo correction"
            ))
        
        return suggestions
    
    async def _get_completions(
        self,
        partial: str,
        context: CompletionContext
    ) -> List[CommandSuggestion]:
        """Get command completion suggestions."""
        suggestions = []
        
        # Common commands that start with the partial
        common_commands = [
            'git', 'docker', 'python', 'npm', 'yarn', 'cargo', 'go',
            'ls', 'cd', 'mkdir', 'touch', 'rm', 'cp', 'mv', 'grep',
            'find', 'curl', 'wget', 'ssh', 'scp', 'rsync',
        ]
        
        for cmd in common_commands:
            if cmd.startswith(partial) and cmd != partial:
                suggestions.append(CommandSuggestion(
                    text=cmd,
                    type=SuggestionType.COMPLETION,
                    confidence=0.7,
                    description=f"Complete '{partial}' to '{cmd}'",
                    context_hint="Common command"
                ))
        
        # Project-specific command completions
        if context.project_type and context.project_type in self.project_commands:
            project_cmds = self.project_commands[context.project_type]
            for cmd in project_cmds:
                if cmd.startswith(partial):
                    suggestions.append(CommandSuggestion(
                        text=cmd,
                        type=SuggestionType.CONTEXT,
                        confidence=0.8,
                        description=f"{context.project_type.value} project command",
                        context_hint=f"Common for {context.project_type.value} projects"
                    ))
        
        return suggestions
    
    async def _get_history_suggestions(
        self,
        partial: str,
        context: CompletionContext
    ) -> List[CommandSuggestion]:
        """Get suggestions from command history."""
        suggestions = []
        
        try:
            # Create prediction context for history intelligence
            prediction_context = PredictionContext(
                current_directory=str(context.current_directory),
                recent_commands=context.recent_commands,
                project_type=context.project_type.value if context.project_type else "unknown",
                git_branch=context.git_branch
            )
            
            # Get history-based suggestions
            history_suggestions = await self.history_engine.get_command_suggestions(
                partial, prediction_context
            )
            
            for cmd in history_suggestions:
                if cmd != partial:
                    suggestions.append(CommandSuggestion(
                        text=cmd,
                        type=SuggestionType.HISTORY,
                        confidence=0.6,
                        description=f"From command history",
                        context_hint="Previously used command"
                    ))
        
        except Exception:
            # If history intelligence fails, continue without it
            pass
        
        return suggestions
    
    def _get_alias_suggestions(self, partial: str) -> List[CommandSuggestion]:
        """Get alias expansion suggestions."""
        suggestions = []
        
        if partial in self.aliases:
            expanded = self.aliases[partial]
            suggestions.append(CommandSuggestion(
                text=expanded,
                type=SuggestionType.TEMPLATE,
                confidence=0.8,
                description=f"Expand alias '{partial}' to '{expanded}'",
                context_hint="Command alias"
            ))
        
        return suggestions
    
    async def _suggest_context_commands(
        self,
        context: CompletionContext
    ) -> List[CommandSuggestion]:
        """Suggest commands based on current context."""
        suggestions = []
        
        # Git repository commands
        if context.git_repo:
            git_commands = ['git status', 'git log --oneline', 'git diff']
            for cmd in git_commands:
                suggestions.append(CommandSuggestion(
                    text=cmd,
                    type=SuggestionType.CONTEXT,
                    confidence=0.7,
                    description="Common git command",
                    context_hint="You're in a git repository"
                ))
        
        # Project-specific commands
        if context.project_type and context.project_type in self.project_commands:
            project_cmds = self.project_commands[context.project_type][:3]
            for cmd in project_cmds:
                suggestions.append(CommandSuggestion(
                    text=cmd,
                    type=SuggestionType.CONTEXT,
                    confidence=0.6,
                    description=f"{context.project_type.value} project command",
                    context_hint=f"Common for {context.project_type.value} projects"
                ))
        
        return suggestions
    
    def _rank_suggestions(
        self,
        suggestions: List[CommandSuggestion],
        partial: str,
        context: CompletionContext
    ) -> List[CommandSuggestion]:
        """Rank suggestions by relevance."""
        
        def score_suggestion(suggestion: CommandSuggestion) -> float:
            score = suggestion.confidence
            
            # Boost exact prefix matches
            if suggestion.text.startswith(partial):
                score += 0.2
            
            # Boost context-appropriate suggestions
            if suggestion.type == SuggestionType.CONTEXT:
                score += 0.1
            
            # Boost corrections for very short partials
            if suggestion.type == SuggestionType.CORRECTION and len(partial) <= 3:
                score += 0.15
            
            # Boost history suggestions
            if suggestion.type == SuggestionType.HISTORY:
                score += 0.05
            
            return score
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion.text not in seen:
                seen.add(suggestion.text)
                unique_suggestions.append(suggestion)
        
        # Sort by score
        return sorted(unique_suggestions, key=score_suggestion, reverse=True)
    
    def _expects_file_path(self, command: str) -> bool:
        """Check if command expects file path arguments."""
        file_commands = {
            'cat', 'less', 'more', 'head', 'tail', 'vim', 'nano', 'emacs',
            'cp', 'mv', 'rm', 'chmod', 'chown', 'ln', 'file', 'stat',
            'python', 'node', 'ruby', 'php'
        }
        
        cmd_parts = command.split()
        return cmd_parts and cmd_parts[0] in file_commands
    
    async def _complete_file_paths(
        self,
        partial: str,
        context: CompletionContext
    ) -> List[CommandSuggestion]:
        """Complete file paths intelligently."""
        suggestions = []
        
        try:
            # Handle relative and absolute paths
            if partial.startswith('/'):
                search_dir = Path(partial).parent
                prefix = Path(partial).name
            elif '/' in partial:
                search_dir = context.current_directory / Path(partial).parent
                prefix = Path(partial).name
            else:
                search_dir = context.current_directory
                prefix = partial
            
            if not search_dir.exists():
                return suggestions
            
            # Find matching files/directories
            matches = []
            for item in search_dir.iterdir():
                if item.name.startswith(prefix):
                    matches.append(item)
            
            # Sort: directories first, then files
            matches.sort(key=lambda x: (not x.is_dir(), x.name))
            
            for match in matches[:10]:  # Limit to 10 suggestions
                display_path = str(match.relative_to(context.current_directory))
                if match.is_dir():
                    display_path += '/'
                
                suggestions.append(CommandSuggestion(
                    text=display_path,
                    type=SuggestionType.COMPLETION,
                    confidence=0.8,
                    description=f"{'Directory' if match.is_dir() else 'File'}: {match.name}",
                    context_hint="File system completion"
                ))
        
        except Exception:
            # If file completion fails, continue without it
            pass
        
        return suggestions
    
    async def _complete_git_arguments(
        self,
        command: str,
        partial_arg: str,
        context: CompletionContext
    ) -> List[CommandSuggestion]:
        """Complete git-specific arguments."""
        suggestions = []
        
        if not context.git_repo:
            return suggestions
        
        cmd_parts = command.split()
        if len(cmd_parts) < 2:
            return suggestions
        
        git_subcmd = cmd_parts[1]
        
        # Branch name completion for checkout, merge, etc.
        if git_subcmd in ['checkout', 'merge', 'branch', 'rebase'] and partial_arg:
            branches = await self._get_git_branches(context.current_directory)
            for branch in branches:
                if branch.startswith(partial_arg):
                    suggestions.append(CommandSuggestion(
                        text=branch,
                        type=SuggestionType.COMPLETION,
                        confidence=0.8,
                        description=f"Git branch: {branch}",
                        context_hint="Available git branch"
                    ))
        
        return suggestions
    
    async def _complete_docker_arguments(
        self,
        command: str,
        partial_arg: str,
        context: CompletionContext
    ) -> List[CommandSuggestion]:
        """Complete docker-specific arguments."""
        suggestions = []
        
        cmd_parts = command.split()
        if len(cmd_parts) < 2:
            return suggestions
        
        docker_subcmd = cmd_parts[1]
        
        # Container name completion
        if docker_subcmd in ['exec', 'logs', 'stop', 'rm'] and partial_arg:
            containers = await self._get_docker_containers()
            for container in containers:
                if container.startswith(partial_arg):
                    suggestions.append(CommandSuggestion(
                        text=container,
                        type=SuggestionType.COMPLETION,
                        confidence=0.8,
                        description=f"Docker container: {container}",
                        context_hint="Running container"
                    ))
        
        return suggestions
    
    def _complete_environment_vars(
        self,
        partial: str,
        context: CompletionContext
    ) -> List[CommandSuggestion]:
        """Complete environment variable names."""
        suggestions = []
        
        # Remove the $ prefix for matching
        var_prefix = partial[1:] if partial.startswith('$') else partial
        
        for var_name, var_value in context.environment_vars.items():
            if var_name.startswith(var_prefix):
                suggestions.append(CommandSuggestion(
                    text=f"${var_name}",
                    type=SuggestionType.COMPLETION,
                    confidence=0.7,
                    description=f"Environment variable: {var_value[:50]}",
                    context_hint="Environment variable"
                ))
        
        return suggestions
    
    async def _get_git_branches(self, repo_path: Path) -> List[str]:
        """Get list of git branches."""
        try:
            result = await asyncio.create_subprocess_shell(
                "git branch --all --format='%(refname:short)'",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=repo_path
            )
            stdout, _ = await result.communicate()
            
            if result.returncode == 0:
                branches = []
                for line in stdout.decode('utf-8').strip().split('\n'):
                    line = line.strip()
                    if line and not line.startswith('origin/'):
                        branches.append(line)
                return branches
        
        except Exception:
            pass
        
        return []
    
    async def _get_docker_containers(self) -> List[str]:
        """Get list of running docker containers."""
        try:
            result = await asyncio.create_subprocess_shell(
                "docker ps --format '{{.Names}}'",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await result.communicate()
            
            if result.returncode == 0:
                containers = []
                for line in stdout.decode('utf-8').strip().split('\n'):
                    line = line.strip()
                    if line:
                        containers.append(line)
                return containers
        
        except Exception:
            pass
        
        return []


# Factory function for easy creation
async def create_command_completion_engine(
    project_engine: Optional[ProjectContextEngine] = None,
    history_engine: Optional[HistoryIntelligence] = None
) -> CommandCompletionEngine:
    """Create a command completion engine with default configuration."""
    
    if project_engine is None:
        project_engine = ProjectContextEngine()
    
    if history_engine is None:
        from openagent.core.history import HistoryManager
        history_manager = HistoryManager()
        history_engine = HistoryIntelligence(history_manager)
    
    return CommandCompletionEngine(project_engine, history_engine)
