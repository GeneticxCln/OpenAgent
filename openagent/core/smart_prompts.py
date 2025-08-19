"""
Smart Prompt Engineering System for OpenAgent.

This module provides dynamic prompt templates, intelligent model selection,
context window optimization, and adaptive response formatting based on
detected context and user intent.
"""

import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from openagent.core.enhanced_context import EnhancedContext, ProjectType
from openagent.core.exceptions import AgentError


class TaskType(Enum):
    """Types of tasks the agent can handle."""

    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    DEBUGGING = "debugging"
    SYSTEM_ADMIN = "system_admin"
    EXPLANATION = "explanation"
    DOCUMENTATION = "documentation"
    GENERAL_CHAT = "general_chat"
    TERMINAL_HELP = "terminal_help"
    PROJECT_ANALYSIS = "project_analysis"
    FILE_OPERATIONS = "file_operations"


class ModelType(Enum):
    """Types of models optimized for different tasks."""

    CODE_MODEL = "code_model"  # CodeLlama, DeepSeek-Coder, etc.
    CHAT_MODEL = "chat_model"  # Mistral, Zephyr, etc.
    LIGHTWEIGHT_MODEL = "lightweight_model"  # TinyLlama, Phi-2
    SYSTEM_MODEL = "system_model"  # Models good for system tasks


@dataclass
class PromptTemplate:
    """Dynamic prompt template with context integration."""

    name: str
    task_type: TaskType
    system_prompt_template: str
    user_prompt_template: str = "{input}"
    context_sections: List[str] = field(default_factory=list)
    max_tokens: int = 2048
    temperature: float = 0.7
    requires_tools: bool = False
    priority_context: List[str] = field(default_factory=list)


@dataclass
class ContextWindow:
    """Context window management for efficient token usage."""

    total_tokens: int = 4096
    reserved_tokens: int = 512  # For response
    system_tokens: int = 0
    context_tokens: int = 0
    available_tokens: int = 0
    prioritized_sections: List[str] = field(default_factory=list)


@dataclass
class ModelSelection:
    """Model selection result with reasoning."""

    model_name: str
    model_type: ModelType
    confidence: float
    reasoning: List[str] = field(default_factory=list)
    fallback_models: List[str] = field(default_factory=list)


class SmartPromptEngineer:
    """
    Smart prompt engineering system that adapts prompts based on context,
    task type, and available resources.
    """

    def __init__(self):
        self.prompt_templates = self._initialize_templates()
        self.model_configs = self._initialize_model_configs()
        self.task_classifiers = self._initialize_task_classifiers()
        self.context_prioritizers = self._initialize_context_prioritizers()

    def _initialize_templates(self) -> Dict[TaskType, PromptTemplate]:
        """Initialize dynamic prompt templates for different task types."""

        templates = {
            TaskType.CODE_GENERATION: PromptTemplate(
                name="code_generation",
                task_type=TaskType.CODE_GENERATION,
                system_prompt_template=(
                    "You are an expert {project_language} programmer working on a {project_type} project. "
                    "Generate clean, efficient, and well-documented code following best practices.\n"
                    "{project_context}\n{git_context}\n{file_context}\n"
                    "Focus on:\n"
                    "- Writing maintainable and readable code\n"
                    "- Following {project_type} conventions and patterns\n"
                    "- Including proper error handling\n"
                    "- Adding relevant comments and docstrings\n"
                    "{framework_specific_guidance}"
                ),
                context_sections=[
                    "project_context",
                    "git_context",
                    "file_context",
                    "error_context",
                ],
                max_tokens=1024,
                temperature=0.3,
                requires_tools=True,
                priority_context=["file_context", "project_context"],
            ),
            TaskType.CODE_REVIEW: PromptTemplate(
                name="code_review",
                task_type=TaskType.CODE_REVIEW,
                system_prompt_template=(
                    "You are an expert code reviewer specializing in {project_language} and {project_type} projects. "
                    "Provide constructive, actionable feedback on code quality, security, and maintainability.\n"
                    "{project_context}\n{git_context}\n{recent_changes}\n"
                    "Review criteria:\n"
                    "- Code quality and readability\n"
                    "- Security vulnerabilities\n"
                    "- Performance implications\n"
                    "- Adherence to project standards\n"
                    "- Testing considerations\n"
                    "{framework_specific_patterns}"
                ),
                context_sections=[
                    "project_context",
                    "git_context",
                    "recent_changes",
                    "file_context",
                ],
                max_tokens=1536,
                temperature=0.4,
                priority_context=["recent_changes", "file_context"],
            ),
            TaskType.DEBUGGING: PromptTemplate(
                name="debugging",
                task_type=TaskType.DEBUGGING,
                system_prompt_template=(
                    "You are an expert debugging assistant for {project_language} applications. "
                    "Analyze errors, identify root causes, and provide step-by-step solutions.\n"
                    "{project_context}\n{error_context}\n{recent_commands}\n{git_context}\n"
                    "Debugging approach:\n"
                    "1. Analyze the error message and stack trace\n"
                    "2. Identify potential root causes\n"
                    "3. Suggest debugging steps and tools\n"
                    "4. Provide fixes with explanations\n"
                    "5. Recommend prevention strategies\n"
                    "{debugging_tools_available}"
                ),
                context_sections=[
                    "error_context",
                    "recent_commands",
                    "project_context",
                    "git_context",
                ],
                max_tokens=1536,
                temperature=0.2,
                requires_tools=True,
                priority_context=["error_context", "recent_commands"],
            ),
            TaskType.SYSTEM_ADMIN: PromptTemplate(
                name="system_admin",
                task_type=TaskType.SYSTEM_ADMIN,
                system_prompt_template=(
                    "You are an expert system administrator and DevOps engineer. "
                    "Provide safe, efficient solutions for system management and automation.\n"
                    "{system_context}\n{project_context}\n{recent_commands}\n"
                    "System administration principles:\n"
                    "- Safety first - explain risks and alternatives\n"
                    "- Provide step-by-step instructions\n"
                    "- Include verification steps\n"
                    "- Consider security implications\n"
                    "- Suggest automation opportunities\n"
                    "{available_tools}"
                ),
                context_sections=[
                    "system_context",
                    "recent_commands",
                    "project_context",
                ],
                max_tokens=1024,
                temperature=0.3,
                requires_tools=True,
                priority_context=["system_context", "recent_commands"],
            ),
            TaskType.TERMINAL_HELP: PromptTemplate(
                name="terminal_help",
                task_type=TaskType.TERMINAL_HELP,
                system_prompt_template=(
                    "You are an expert terminal and shell assistant. Help users with command-line operations, "
                    "shell scripting, and terminal productivity.\n"
                    "{shell_context}\n{recent_commands}\n{system_context}\n"
                    "Terminal assistance approach:\n"
                    "- Provide exact commands with explanations\n"
                    "- Explain command options and flags\n"
                    "- Warn about potentially dangerous operations\n"
                    "- Suggest safer alternatives when appropriate\n"
                    "- Include examples and use cases\n"
                    "Shell: {shell_type}, OS: {operating_system}"
                ),
                context_sections=["shell_context", "recent_commands", "system_context"],
                max_tokens=1024,
                temperature=0.4,
                requires_tools=True,
                priority_context=["shell_context", "recent_commands"],
            ),
            TaskType.EXPLANATION: PromptTemplate(
                name="explanation",
                task_type=TaskType.EXPLANATION,
                system_prompt_template=(
                    "You are an expert technical educator. Provide clear, comprehensive explanations "
                    "tailored to the user's context and experience level.\n"
                    "{project_context}\n{topic_context}\n"
                    "Explanation principles:\n"
                    "- Start with high-level concepts\n"
                    "- Use relevant examples from user's project\n"
                    "- Break down complex topics step-by-step\n"
                    "- Provide practical applications\n"
                    "- Include references for deeper learning\n"
                ),
                context_sections=["project_context", "topic_context", "file_context"],
                max_tokens=2048,
                temperature=0.5,
                priority_context=["topic_context", "project_context"],
            ),
            TaskType.GENERAL_CHAT: PromptTemplate(
                name="general_chat",
                task_type=TaskType.GENERAL_CHAT,
                system_prompt_template=(
                    "You are a helpful and knowledgeable AI assistant. Provide accurate, helpful responses "
                    "while being conversational and engaging.\n"
                    "{minimal_context}\n"
                    "Response guidelines:\n"
                    "- Be concise but thorough\n"
                    "- Ask clarifying questions when needed\n"
                    "- Provide practical advice when applicable\n"
                    "- Acknowledge uncertainty when appropriate\n"
                ),
                context_sections=["minimal_context"],
                max_tokens=1024,
                temperature=0.7,
                priority_context=["minimal_context"],
            ),
        }

        return templates

    def _initialize_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize model configurations for different types of tasks."""

        return {
            # Code-optimized models
            "codellama-7b": {
                "type": ModelType.CODE_MODEL,
                "max_context": 4096,
                "strengths": ["code_generation", "debugging", "code_review"],
                "weaknesses": ["general_chat"],
                "optimal_temperature": 0.3,
                "supports_tools": True,
            },
            "deepseek-coder": {
                "type": ModelType.CODE_MODEL,
                "max_context": 8192,
                "strengths": ["code_generation", "code_review", "debugging"],
                "weaknesses": ["general_chat"],
                "optimal_temperature": 0.2,
                "supports_tools": True,
            },
            "starcoder": {
                "type": ModelType.CODE_MODEL,
                "max_context": 8192,
                "strengths": ["code_generation", "code_review"],
                "weaknesses": ["debugging", "system_admin"],
                "optimal_temperature": 0.3,
                "supports_tools": True,
            },
            # Chat-optimized models
            "mistral-7b": {
                "type": ModelType.CHAT_MODEL,
                "max_context": 4096,
                "strengths": ["general_chat", "explanation", "system_admin"],
                "weaknesses": ["code_generation"],
                "optimal_temperature": 0.7,
                "supports_tools": True,
            },
            "zephyr-7b": {
                "type": ModelType.CHAT_MODEL,
                "max_context": 4096,
                "strengths": ["general_chat", "explanation", "documentation"],
                "weaknesses": ["code_generation", "debugging"],
                "optimal_temperature": 0.6,
                "supports_tools": True,
            },
            # Lightweight models
            "tiny-llama": {
                "type": ModelType.LIGHTWEIGHT_MODEL,
                "max_context": 2048,
                "strengths": ["simple_tasks", "terminal_help"],
                "weaknesses": ["complex_reasoning", "code_generation"],
                "optimal_temperature": 0.8,
                "supports_tools": False,
            },
            "phi-2": {
                "type": ModelType.LIGHTWEIGHT_MODEL,
                "max_context": 2048,
                "strengths": ["explanation", "general_chat"],
                "weaknesses": ["code_generation", "system_admin"],
                "optimal_temperature": 0.7,
                "supports_tools": False,
            },
            # Ollama models (local inference)
            "ollama:llama3.2": {
                "type": ModelType.CHAT_MODEL,
                "max_context": 8192,
                "strengths": ["general_chat", "explanation", "code_generation"],
                "weaknesses": [],
                "optimal_temperature": 0.7,
                "supports_tools": True,
            },
            "ollama:codellama": {
                "type": ModelType.CODE_MODEL,
                "max_context": 4096,
                "strengths": ["code_generation", "debugging", "code_review"],
                "weaknesses": ["general_chat"],
                "optimal_temperature": 0.3,
                "supports_tools": True,
            },
        }

    def _initialize_task_classifiers(self) -> Dict[str, Tuple[TaskType, float]]:
        """Initialize task classification patterns and confidence scores."""

        return {
            # Code generation patterns
            r"(write|create|generate|implement|build).*?(function|class|script|program|code)": (
                TaskType.CODE_GENERATION,
                0.9,
            ),
            r"(create|make|write).*?(app|application|service|module)": (
                TaskType.CODE_GENERATION,
                0.8,
            ),
            r"how to.*?(implement|code|program|write)": (TaskType.CODE_GENERATION, 0.7),
            # Code review patterns
            r"(review|check|analyze|examine).*?(code|implementation)": (
                TaskType.CODE_REVIEW,
                0.9,
            ),
            r"(what's wrong|problems|issues).*?(code|implementation)": (
                TaskType.CODE_REVIEW,
                0.8,
            ),
            r"(improve|optimize|refactor).*?code": (TaskType.CODE_REVIEW, 0.7),
            # Debugging patterns
            r"(error|exception|bug|issue|problem|fail|broken)": (
                TaskType.DEBUGGING,
                0.8,
            ),
            r"(debug|troubleshoot|fix|resolve)": (TaskType.DEBUGGING, 0.9),
            r"(not working|doesn't work|won't run)": (TaskType.DEBUGGING, 0.7),
            r"(stack trace|traceback|exception)": (TaskType.DEBUGGING, 0.9),
            # System administration patterns
            r"(install|setup|configure|deploy)": (TaskType.SYSTEM_ADMIN, 0.7),
            r"(server|system|network|infrastructure)": (TaskType.SYSTEM_ADMIN, 0.6),
            r"(docker|kubernetes|nginx|apache)": (TaskType.SYSTEM_ADMIN, 0.8),
            r"(permission|user|group|sudo|root)": (TaskType.SYSTEM_ADMIN, 0.7),
            # Terminal help patterns
            r"(command|terminal|shell|bash|zsh)": (TaskType.TERMINAL_HELP, 0.8),
            r"how to.*?(run|execute|use).*?command": (TaskType.TERMINAL_HELP, 0.9),
            r"(grep|find|sed|awk|pipe|redirect)": (TaskType.TERMINAL_HELP, 0.8),
            # Explanation patterns
            r"(explain|what is|how does|why|what's the difference)": (
                TaskType.EXPLANATION,
                0.8,
            ),
            r"(understand|learn|tutorial|guide)": (TaskType.EXPLANATION, 0.7),
            r"(concept|theory|principle|how.*?works)": (TaskType.EXPLANATION, 0.8),
            # File operations patterns
            r"(read|open|view|show|display).*?file": (TaskType.FILE_OPERATIONS, 0.8),
            r"(edit|modify|change|update).*?file": (TaskType.FILE_OPERATIONS, 0.8),
            r"(create|delete|move|copy).*?file": (TaskType.FILE_OPERATIONS, 0.7),
        }

    def _initialize_context_prioritizers(self) -> Dict[TaskType, List[str]]:
        """Initialize context prioritization rules for different task types."""

        return {
            TaskType.CODE_GENERATION: [
                "project_context",
                "file_context",
                "git_context",
                "framework_context",
            ],
            TaskType.CODE_REVIEW: [
                "file_context",
                "git_context",
                "recent_changes",
                "project_context",
            ],
            TaskType.DEBUGGING: [
                "error_context",
                "recent_commands",
                "file_context",
                "git_context",
            ],
            TaskType.SYSTEM_ADMIN: [
                "system_context",
                "recent_commands",
                "environment_context",
            ],
            TaskType.TERMINAL_HELP: [
                "shell_context",
                "recent_commands",
                "system_context",
            ],
            TaskType.EXPLANATION: ["topic_context", "project_context", "file_context"],
            TaskType.GENERAL_CHAT: ["minimal_context"],
        }

    async def engineer_prompt(
        self,
        user_input: str,
        context: EnhancedContext,
        task_type: Optional[TaskType] = None,
        model_hint: Optional[str] = None,
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Engineer an optimal prompt based on user input and context.

        Returns:
            Tuple of (system_prompt, user_prompt, metadata)
        """

        # Classify task type if not provided
        if task_type is None:
            task_type = self.classify_task(user_input)

        # Get appropriate template
        template = self.prompt_templates.get(
            task_type, self.prompt_templates[TaskType.GENERAL_CHAT]
        )

        # Build context sections
        context_data = self._build_context_data(context, template)

        # Optimize context window
        context_window = self._optimize_context_window(
            context_data, template, user_input
        )

        # Generate system prompt
        system_prompt = self._generate_system_prompt(
            template, context_data, context_window
        )

        # Generate user prompt (usually just the input, but can be enhanced)
        user_prompt = template.user_prompt_template.format(input=user_input)

        # Prepare metadata
        metadata = {
            "task_type": task_type.value,
            "template_name": template.name,
            "context_sections_used": list(context_window.prioritized_sections),
            "tokens_estimated": context_window.system_tokens + len(user_prompt.split()),
            "context_window": {
                "total_tokens": context_window.total_tokens,
                "available_tokens": context_window.available_tokens,
                "context_tokens": context_window.context_tokens,
            },
            "requires_tools": template.requires_tools,
            "temperature": template.temperature,
            "max_tokens": template.max_tokens,
        }

        return system_prompt, user_prompt, metadata

    def classify_task(self, user_input: str) -> TaskType:
        """Classify the task type based on user input patterns."""

        user_input_lower = user_input.lower()
        best_match = TaskType.GENERAL_CHAT
        best_confidence = 0.0

        for pattern, (task_type, confidence) in self.task_classifiers.items():
            if re.search(pattern, user_input_lower):
                if confidence > best_confidence:
                    best_match = task_type
                    best_confidence = confidence

        return best_match

    def select_optimal_model(
        self,
        task_type: TaskType,
        context: EnhancedContext,
        available_models: List[str],
        resource_constraints: Optional[Dict[str, Any]] = None,
    ) -> ModelSelection:
        """Select the optimal model for the given task and constraints."""

        resource_constraints = resource_constraints or {}
        scored_models = []

        for model_name in available_models:
            if model_name not in self.model_configs:
                continue

            config = self.model_configs[model_name]
            score = self._calculate_model_score(
                model_name, config, task_type, context, resource_constraints
            )
            scored_models.append((model_name, score, config))

        if not scored_models:
            # Fallback to first available model
            return ModelSelection(
                model_name=available_models[0] if available_models else "codellama-7b",
                model_type=ModelType.CODE_MODEL,
                confidence=0.1,
                reasoning=["No models found in configuration, using default"],
                fallback_models=(
                    available_models[1:5] if len(available_models) > 1 else []
                ),
            )

        # Sort by score (descending)
        scored_models.sort(key=lambda x: x[1], reverse=True)

        best_model, best_score, best_config = scored_models[0]

        reasoning = self._generate_model_selection_reasoning(
            best_model, best_config, task_type, context, best_score
        )

        return ModelSelection(
            model_name=best_model,
            model_type=best_config["type"],
            confidence=best_score,
            reasoning=reasoning,
            fallback_models=[
                model[0] for model in scored_models[1:4]
            ],  # Top 3 alternatives
        )

    def _calculate_model_score(
        self,
        model_name: str,
        config: Dict[str, Any],
        task_type: TaskType,
        context: EnhancedContext,
        resource_constraints: Dict[str, Any],
    ) -> float:
        """Calculate a score for model suitability."""

        score = 0.0

        # Base score for task type alignment
        strengths = config.get("strengths", [])
        weaknesses = config.get("weaknesses", [])

        task_name = task_type.value.replace("_", "")

        if task_name in strengths:
            score += 1.0
        if any(strength in task_type.value for strength in strengths):
            score += 0.5
        if task_name in weaknesses:
            score -= 0.5

        # Context window consideration
        estimated_context_tokens = self._estimate_context_tokens(context)
        max_context = config.get("max_context", 4096)

        if estimated_context_tokens <= max_context:
            score += 0.3
        else:
            score -= 0.3  # Penalty for insufficient context window

        # Tool support consideration
        requires_tools = task_type in [
            TaskType.CODE_GENERATION,
            TaskType.DEBUGGING,
            TaskType.SYSTEM_ADMIN,
        ]
        supports_tools = config.get("supports_tools", False)

        if requires_tools and supports_tools:
            score += 0.4
        elif requires_tools and not supports_tools:
            score -= 0.4

        # Resource constraints
        memory_limit = resource_constraints.get("max_memory_mb", float("inf"))
        if "lightweight" in model_name and memory_limit < 8192:  # 8GB limit
            score += 0.3

        # Project type specific bonuses
        if context.project.project_type != ProjectType.UNKNOWN:
            project_type_lower = context.project.project_type.value.lower()
            if "code" in strengths and "python" in project_type_lower:
                score += 0.2
            elif "chat" in strengths and project_type_lower in [
                "generic",
                "documentation",
            ]:
                score += 0.2

        return max(0.0, min(2.0, score))  # Clamp between 0 and 2

    def _estimate_context_tokens(self, context: EnhancedContext) -> int:
        """Estimate the number of tokens needed for context."""

        # Rough estimation: 1 token ≈ 4 characters for English text
        # This is a simplification - real implementation would use tokenizer

        tokens = 0

        # Project context
        if context.project.project_name:
            tokens += 50  # Basic project info
        if context.project.frameworks:
            tokens += len(context.project.frameworks) * 10

        # Git context
        if context.git.is_repo:
            tokens += 30  # Branch, status info
            tokens += len(context.git.recent_commits) * 15
            tokens += len(context.git.modified_files) * 5

        # File context
        tokens += len(context.relevant_files) * 20

        # Error context
        tokens += len(context.errors.recent_errors) * 25

        # Terminal history
        tokens += len(context.terminal_history) * 8

        # Environment variables
        tokens += len(context.environment_vars) * 5

        return tokens

    def _build_context_data(
        self, context: EnhancedContext, template: PromptTemplate
    ) -> Dict[str, str]:
        """Build context data for prompt template substitution."""

        context_data = {}

        # Project context
        if "project_context" in template.context_sections:
            project_info = []
            if context.project.project_type != ProjectType.UNKNOWN:
                project_info.append(
                    f"Project Type: {context.project.project_type.value}"
                )
            if context.project.project_name:
                project_info.append(f"Project Name: {context.project.project_name}")
            if context.project.frameworks:
                project_info.append(
                    f"Frameworks: {', '.join(context.project.frameworks)}"
                )
            if context.project.language_map:
                languages = [
                    f"{lang} ({count} files)"
                    for lang, count in context.project.language_map.items()
                ]
                project_info.append(f"Languages: {', '.join(languages)}")

            context_data["project_context"] = (
                "Project Context:\n" + "\n".join(f"- {info}" for info in project_info)
                if project_info
                else ""
            )
            context_data["project_type"] = context.project.project_type.value
            context_data["project_language"] = (
                max(
                    context.project.language_map.keys(),
                    key=context.project.language_map.get,
                )
                if context.project.language_map
                else "Unknown"
            )

        # Git context
        if "git_context" in template.context_sections and context.git.is_repo:
            git_info = []
            if context.git.branch:
                git_info.append(f"Current branch: {context.git.branch}")
            if context.git.modified_files:
                git_info.append(
                    f"Modified files: {', '.join(context.git.modified_files[:5])}"
                )
            if context.git.recent_commits:
                recent = context.git.recent_commits[0]
                git_info.append(f"Last commit: {recent['message'][:50]}...")

            context_data["git_context"] = (
                "Git Context:\n" + "\n".join(f"- {info}" for info in git_info)
                if git_info
                else ""
            )

        # File context
        if "file_context" in template.context_sections:
            relevant_files = context.relevant_files[:10]  # Top 10 most relevant
            if relevant_files:
                file_info = []
                for file_data in relevant_files:
                    relative_path = str(
                        file_data.path.relative_to(
                            context.project.project_root or Path.cwd()
                        )
                    )
                    reasons = ", ".join(file_data.reasons[:2])  # Top 2 reasons
                    file_info.append(f"{relative_path} ({reasons})")

                context_data["file_context"] = "Relevant Files:\n" + "\n".join(
                    f"- {info}" for info in file_info
                )
            else:
                context_data["file_context"] = ""

        # Error context
        if (
            "error_context" in template.context_sections
            and context.errors.recent_errors
        ):
            error_info = []
            for error in context.errors.recent_errors[:3]:  # Last 3 errors
                cmd = error.get("command", "")[:30]
                error_text = error.get("error_output", "")[:100]
                error_info.append(f"Command: {cmd}, Error: {error_text}")

            if context.errors.suggested_fixes:
                fixes = [
                    f"{error_type}: {fix}"
                    for error_type, fix in list(context.errors.suggested_fixes.items())[
                        :2
                    ]
                ]
                error_info.extend(fixes)

            context_data["error_context"] = "Recent Errors:\n" + "\n".join(
                f"- {info}" for info in error_info
            )

        # Recent commands
        if "recent_commands" in template.context_sections and context.terminal_history:
            recent_cmds = context.terminal_history[-10:]  # Last 10 commands
            context_data["recent_commands"] = "Recent Commands:\n" + "\n".join(
                f"- {cmd}" for cmd in recent_cmds
            )

        # System context
        if "system_context" in template.context_sections:
            sys_info = []
            if "SHELL" in context.environment_vars:
                sys_info.append(
                    f"Shell: {Path(context.environment_vars['SHELL']).name}"
                )
            if "USER" in context.environment_vars:
                sys_info.append(f"User: {context.environment_vars['USER']}")
            if "PWD" in context.environment_vars:
                sys_info.append(
                    f"Working Directory: {Path(context.environment_vars['PWD']).name}"
                )

            context_data["system_context"] = (
                "System Context:\n" + "\n".join(f"- {info}" for info in sys_info)
                if sys_info
                else ""
            )
            context_data["shell_type"] = Path(
                context.environment_vars.get("SHELL", "bash")
            ).name
            context_data["operating_system"] = "Linux"  # Could be detected from context

        # Framework specific guidance
        if "framework_specific_guidance" in template.context_sections:
            guidance = []
            for framework in context.project.frameworks:
                if framework == "fastapi":
                    guidance.append(
                        "- Use FastAPI best practices: dependency injection, proper HTTP status codes"
                    )
                elif framework == "django":
                    guidance.append(
                        "- Follow Django conventions: models, views, templates pattern"
                    )
                elif framework == "react":
                    guidance.append("- Use React hooks and functional components")

            context_data["framework_specific_guidance"] = (
                "\n".join(guidance) if guidance else ""
            )

        # Minimal context for general chat
        if "minimal_context" in template.context_sections:
            context_data["minimal_context"] = (
                f"Working in: {context.project.project_name or 'current directory'}"
            )

        return context_data

    def _optimize_context_window(
        self, context_data: Dict[str, str], template: PromptTemplate, user_input: str
    ) -> ContextWindow:
        """Optimize context window usage based on token limits."""

        # Estimate token counts (simplified)
        system_template_tokens = len(template.system_prompt_template.split())
        user_input_tokens = len(user_input.split())

        context_window = ContextWindow(
            total_tokens=4096,  # Default, should be based on model config
            reserved_tokens=template.max_tokens,
        )

        # Calculate available tokens for context
        available_for_context = (
            context_window.total_tokens
            - context_window.reserved_tokens
            - system_template_tokens
            - user_input_tokens
        )

        context_window.available_tokens = max(0, available_for_context)

        # Prioritize context sections
        priority_sections = template.priority_context or template.context_sections

        used_tokens = 0
        included_sections = []

        for section in priority_sections:
            if section in context_data:
                section_tokens = len(context_data[section].split())
                if used_tokens + section_tokens <= context_window.available_tokens:
                    used_tokens += section_tokens
                    included_sections.append(section)
                else:
                    # Truncate section to fit
                    remaining_tokens = context_window.available_tokens - used_tokens
                    if remaining_tokens > 50:  # Minimum useful size
                        truncated_content = " ".join(
                            context_data[section].split()[:remaining_tokens]
                        )
                        context_data[section] = truncated_content + "..."
                        used_tokens += remaining_tokens
                        included_sections.append(section)
                    break

        context_window.context_tokens = used_tokens
        context_window.prioritized_sections = included_sections

        return context_window

    def _generate_system_prompt(
        self,
        template: PromptTemplate,
        context_data: Dict[str, str],
        context_window: ContextWindow,
    ) -> str:
        """Generate the final system prompt with context substitution."""

        # Filter context data to only include sections that fit in context window
        filtered_context = {
            key: value
            for key, value in context_data.items()
            if key in context_window.prioritized_sections
            or key
            in ["project_type", "project_language", "shell_type", "operating_system"]
        }

        try:
            return template.system_prompt_template.format(**filtered_context)
        except KeyError as e:
            # Handle missing template variables gracefully
            missing_key = str(e).strip("'\"")
            filtered_context[missing_key] = f"[{missing_key}_not_available]"
            return template.system_prompt_template.format(**filtered_context)

    def _generate_model_selection_reasoning(
        self,
        model_name: str,
        config: Dict[str, Any],
        task_type: TaskType,
        context: EnhancedContext,
        score: float,
    ) -> List[str]:
        """Generate human-readable reasoning for model selection."""

        reasoning = []

        reasoning.append(f"Selected {model_name} for {task_type.value} task")
        reasoning.append(f"Model type: {config['type'].value}")
        reasoning.append(f"Confidence score: {score:.2f}/2.0")

        strengths = config.get("strengths", [])
        if any(strength in task_type.value for strength in strengths):
            reasoning.append(f"Model optimized for: {', '.join(strengths)}")

        if config.get("supports_tools", False) and task_type in [
            TaskType.CODE_GENERATION,
            TaskType.DEBUGGING,
        ]:
            reasoning.append("Tool support available for enhanced capabilities")

        if context.project.project_type != ProjectType.UNKNOWN:
            reasoning.append(f"Context: {context.project.project_type.value} project")

        return reasoning


# Utility function for easy usage
async def engineer_smart_prompt(
    user_input: str,
    context: EnhancedContext,
    task_type: Optional[TaskType] = None,
    model_hint: Optional[str] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """Engineer a smart prompt with context optimization."""
    engineer = SmartPromptEngineer()
    return await engineer.engineer_prompt(user_input, context, task_type, model_hint)
