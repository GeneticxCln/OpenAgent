"""
Intelligent Context-Aware Agent for OpenAgent.

This module integrates enhanced context detection, smart prompt engineering,
and memory/learning systems to provide truly intelligent, adaptive assistance.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from openagent.core.base import BaseAgent, BaseMessage, BaseTool, ToolResult
from openagent.core.enhanced_context import EnhancedContext, get_enhanced_context
from openagent.core.exceptions import AgentError
from openagent.core.llm import get_llm
from openagent.core.memory_learning import (
    ConversationSession,
    MemoryManager,
    PersonalizedSuggestionEngine,
    create_learning_session,
)
from openagent.core.smart_prompts import (
    ModelSelection,
    SmartPromptEngineer,
    TaskType,
    engineer_smart_prompt,
)

logger = logging.getLogger(__name__)


class IntelligentAgent(BaseAgent):
    """
    Intelligent context-aware agent that combines enhanced context detection,
    smart prompt engineering, and memory/learning for adaptive assistance.

    Features:
    - Automatic context detection and relevance scoring
    - Dynamic prompt templates based on task type and context
    - Intelligent model selection based on requirements
    - Conversation continuity and memory across sessions
    - User preference learning and personalized suggestions
    - Command pattern recognition and workflow optimization
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        tools: Optional[List[BaseTool]] = None,
        working_directory: Optional[Path] = None,
        memory_db_path: Optional[Path] = None,
        available_models: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        super().__init__(name, description, tools, **kwargs)

        self.working_directory = working_directory or Path.cwd()
        self.available_models = available_models or [
            "codellama-7b",
            "mistral-7b",
            "ollama:llama3.2",
        ]

        # Initialize core components
        self.prompt_engineer = SmartPromptEngineer()
        self.memory_manager: Optional[MemoryManager] = None
        self.suggestion_engine: Optional[PersonalizedSuggestionEngine] = None
        self.current_session: Optional[ConversationSession] = None

        # Initialize asynchronously (will be called in process_message)
        self._initialization_task = None

        # Context caching for performance
        self._context_cache: Optional[EnhancedContext] = None
        self._context_cache_time: float = 0
        self._context_cache_ttl: float = 60  # 1 minute TTL

        logger.info(
            f"Initialized IntelligentAgent '{name}' with {len(self.tools)} tools"
        )

    async def _ensure_initialized(self):
        """Ensure all async components are initialized."""
        if self.memory_manager is None:
            self.memory_manager = MemoryManager()
            self.suggestion_engine = PersonalizedSuggestionEngine(self.memory_manager)
            logger.info("Initialized memory and learning systems")

    async def process_message(self, message: Union[str, BaseMessage]) -> BaseMessage:
        """
        Process an incoming message with full intelligent context awareness.

        This method orchestrates the entire intelligent processing pipeline:
        1. Initialize systems if needed
        2. Detect and analyze context
        3. Classify task and engineer optimal prompt
        4. Select best model for the task
        5. Generate response using tools if needed
        6. Learn from the interaction
        7. Provide personalized suggestions
        """
        start_time = time.time()

        try:
            await self._ensure_initialized()

            # Convert string to BaseMessage if needed
            if isinstance(message, str):
                input_message = BaseMessage(content=message, role="user")
            else:
                input_message = message

            logger.info(f"Processing message: {input_message.content[:100]}...")

            # Phase 1: Enhanced Context Detection
            context = await self._get_enhanced_context()

            # Phase 2: Smart Prompt Engineering
            task_type = self.prompt_engineer.classify_task(input_message.content)
            system_prompt, user_prompt, prompt_metadata = await engineer_smart_prompt(
                input_message.content, context, task_type
            )

            # Model Selection
            model_selection = await self._select_optimal_model(
                task_type, context, prompt_metadata
            )

            # Phase 3: Memory & Learning - Get or create session
            if not self.current_session:
                self.current_session = await self.memory_manager.get_or_create_session(
                    context=context
                )

            # Get conversation history for continuity
            conversation_history = self.memory_manager.get_conversation_context(
                self.current_session, max_turns=3
            )

            # Generate response with intelligent context
            response_content = await self._generate_intelligent_response(
                user_prompt,
                system_prompt,
                conversation_history,
                context,
                task_type,
                model_selection,
                prompt_metadata,
            )

            # Create response message with rich metadata
            response_message = BaseMessage(
                content=response_content,
                role="assistant",
                metadata={
                    "agent_name": self.name,
                    "task_type": task_type.value,
                    "model_used": model_selection.model_name,
                    "model_confidence": model_selection.confidence,
                    "context_sections_used": prompt_metadata.get(
                        "context_sections_used", []
                    ),
                    "tools_used": getattr(self, "_last_tools_used", []),
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "session_id": self.current_session.session_id,
                    "enhanced_context_metrics": context.performance_metrics,
                },
            )

            # Record interaction for learning
            await self._record_interaction(
                input_message, response_message, task_type, context, prompt_metadata
            )

            # Generate personalized suggestions
            suggestions = await self._get_personalized_suggestions(
                context, task_type, input_message.content
            )

            if suggestions:
                response_message.metadata["suggestions"] = [
                    {
                        "id": s.suggestion_id,
                        "text": s.suggestion_text,
                        "type": s.suggestion_type,
                        "confidence": s.confidence,
                    }
                    for s in suggestions[:3]  # Top 3 suggestions
                ]

            logger.info(
                f"Generated response using {model_selection.model_name} "
                f"for {task_type.value} task in {(time.time() - start_time) * 1000:.1f}ms"
            )

            return response_message

        except Exception as e:
            logger.error(f"Error in intelligent processing: {e}")
            error_response = BaseMessage(
                content=f"I encountered an error while processing your request: {str(e)}",
                role="assistant",
                metadata={
                    "error": True,
                    "agent_name": self.name,
                    "processing_time_ms": (time.time() - start_time) * 1000,
                },
            )
            return error_response

    async def _get_enhanced_context(
        self, force_refresh: bool = False
    ) -> EnhancedContext:
        """Get enhanced context with caching for performance."""
        current_time = time.time()

        if (
            not force_refresh
            and self._context_cache
            and current_time - self._context_cache_time < self._context_cache_ttl
        ):
            return self._context_cache

        logger.debug("Refreshing enhanced context...")
        context = await get_enhanced_context(self.working_directory)

        self._context_cache = context
        self._context_cache_time = current_time

        return context

    async def _select_optimal_model(
        self,
        task_type: TaskType,
        context: EnhancedContext,
        prompt_metadata: Dict[str, Any],
    ) -> ModelSelection:
        """Select the optimal model for the current task and context."""

        # Consider resource constraints based on context size
        estimated_tokens = prompt_metadata.get("tokens_estimated", 1000)
        resource_constraints = {}

        if estimated_tokens > 6000:
            resource_constraints["requires_large_context"] = True
        if context.performance_metrics.get("files_analyzed", 0) > 100:
            resource_constraints["high_context_complexity"] = True

        model_selection = self.prompt_engineer.select_optimal_model(
            task_type, context, self.available_models, resource_constraints
        )

        logger.debug(
            f"Selected model: {model_selection.model_name} "
            f"(confidence: {model_selection.confidence:.2f})"
        )

        return model_selection

    async def _generate_intelligent_response(
        self,
        user_prompt: str,
        system_prompt: str,
        conversation_history: List[Dict[str, str]],
        context: EnhancedContext,
        task_type: TaskType,
        model_selection: ModelSelection,
        prompt_metadata: Dict[str, Any],
    ) -> str:
        """Generate an intelligent response using the selected model and context."""

        # Initialize LLM with selected model
        llm_config = {
            "temperature": prompt_metadata.get("temperature", 0.7),
            "max_new_tokens": prompt_metadata.get("max_tokens", 1024),
        }

        # Apply user preferences if available
        if self.memory_manager:
            preferred_verbosity = self.memory_manager.get_user_preference(
                "response_verbosity"
            )
            if preferred_verbosity == "concise":
                llm_config["max_new_tokens"] = min(llm_config["max_new_tokens"], 512)
                system_prompt += "\nBe concise and to the point in your response."
            elif preferred_verbosity == "detailed":
                llm_config["max_new_tokens"] = max(llm_config["max_new_tokens"], 1536)
                system_prompt += "\nProvide detailed explanations and examples."

        llm = get_llm(model_name=model_selection.model_name, **llm_config)

        # Use tools if required by the task
        if prompt_metadata.get("requires_tools", False) and self.tools:
            response = await self._generate_response_with_tools(
                user_prompt,
                system_prompt,
                conversation_history,
                context,
                task_type,
                llm,
            )
        else:
            response = await llm.generate_response(
                prompt=user_prompt,
                system_prompt=system_prompt,
                context=conversation_history,
                max_new_tokens=llm_config["max_new_tokens"],
            )

        return response.strip()

    async def _generate_response_with_tools(
        self,
        user_prompt: str,
        system_prompt: str,
        conversation_history: List[Dict[str, str]],
        context: EnhancedContext,
        task_type: TaskType,
        llm: Any,
    ) -> str:
        """Generate response with intelligent tool usage."""

        # Analyze if tools are needed for this specific request
        tools_needed = await self._analyze_tool_requirements(
            user_prompt, context, task_type
        )

        if not tools_needed:
            return await llm.generate_response(
                prompt=user_prompt,
                system_prompt=system_prompt,
                context=conversation_history,
            )

        # Execute relevant tools
        tool_results = []
        self._last_tools_used = []

        for tool_name, tool_args in tools_needed:
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if tool:
                try:
                    result = await tool.execute(tool_args)
                    tool_results.append((tool_name, result))
                    self._last_tools_used.append(tool_name)
                    logger.debug(
                        f"Executed tool {tool_name} with success: {result.success}"
                    )
                except Exception as e:
                    logger.warning(f"Tool {tool_name} failed: {e}")

        # Incorporate tool results into the prompt
        if tool_results:
            tool_context = "\n\nTool Results:\n"
            for tool_name, result in tool_results:
                if result.success:
                    tool_context += f"- {tool_name}: {str(result.content)[:500]}\n"
                else:
                    tool_context += f"- {tool_name}: Error - {result.error}\n"

            enhanced_prompt = (
                user_prompt
                + tool_context
                + "\n\nPlease provide a comprehensive response using the tool results above."
            )
        else:
            enhanced_prompt = user_prompt

        return await llm.generate_response(
            prompt=enhanced_prompt,
            system_prompt=system_prompt,
            context=conversation_history,
        )

    async def _analyze_tool_requirements(
        self, user_prompt: str, context: EnhancedContext, task_type: TaskType
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Analyze what tools are needed for the current request."""

        tools_needed = []
        user_prompt_lower = user_prompt.lower()

        # File operations
        if any(
            keyword in user_prompt_lower
            for keyword in ["read file", "show file", "view file", "cat "]
        ):
            # Would need a file reading tool
            pass

        # Git operations
        if context.git.is_repo and any(
            keyword in user_prompt_lower
            for keyword in ["git status", "git log", "recent changes", "commits"]
        ):
            if any(t.name == "git_tool" for t in self.tools):
                if "status" in user_prompt_lower:
                    tools_needed.append(("git_tool", {"subcommand": "status"}))
                elif "log" in user_prompt_lower:
                    tools_needed.append(
                        ("git_tool", {"subcommand": "log", "args": ["-10"]})
                    )

        # System commands
        if task_type in [TaskType.SYSTEM_ADMIN, TaskType.TERMINAL_HELP] and any(
            keyword in user_prompt_lower
            for keyword in ["run", "execute", "command", "ps", "ls", "find"]
        ):
            if any(t.name == "command_executor" for t in self.tools):
                # Extract potential command from the prompt
                import re

                cmd_match = re.search(
                    r'(?:run|execute)\s+["`]?([^"`\n]+)["`]?', user_prompt_lower
                )
                if cmd_match:
                    command = cmd_match.group(1)
                    tools_needed.append(
                        ("command_executor", {"command": command, "explain_only": True})
                    )

        # Search operations
        if any(keyword in user_prompt_lower for keyword in ["search", "find", "grep"]):
            if any(t.name == "repo_grep" for t in self.tools):
                # Extract search term
                import re

                search_match = re.search(
                    r'(?:search|find|grep)\s+(?:for\s+)?["`]?([^"`\n]+)["`]?',
                    user_prompt_lower,
                )
                if search_match:
                    search_term = search_match.group(1)
                    tools_needed.append(("repo_grep", {"pattern": search_term}))

        return tools_needed

    async def _record_interaction(
        self,
        input_message: BaseMessage,
        response_message: BaseMessage,
        task_type: TaskType,
        context: EnhancedContext,
        prompt_metadata: Dict[str, Any],
    ):
        """Record interaction for learning and improvement."""

        if not self.memory_manager or not self.current_session:
            return

        # Record conversation turn
        await self.memory_manager.add_conversation_turn(
            self.current_session,
            input_message.content,
            response_message.content,
            task_type,
            prompt_metadata,
            self._last_tools_used if hasattr(self, "_last_tools_used") else [],
        )

        # Learn preferences from interaction patterns
        model_used = response_message.metadata.get("model_used", "")
        if model_used:
            await self.memory_manager.learn_user_preference(
                "preferred_model", model_used, context=task_type.value
            )

        response_length = len(response_message.content.split())
        if response_length < 50:
            verbosity = "concise"
        elif response_length > 200:
            verbosity = "detailed"
        else:
            verbosity = "balanced"

        await self.memory_manager.learn_user_preference(
            "response_verbosity", verbosity, context=task_type.value
        )

        # Analyze command patterns if relevant
        if task_type in [
            TaskType.TERMINAL_HELP,
            TaskType.SYSTEM_ADMIN,
            TaskType.DEBUGGING,
        ]:
            recent_commands = (
                context.terminal_history[-10:] if context.terminal_history else []
            )
            if recent_commands:
                context_tags = [
                    task_type.value,
                    (
                        context.project.project_type.value
                        if context.project.project_type
                        else ""
                    ),
                ]
                await self.memory_manager.analyze_command_patterns(
                    recent_commands, context_tags
                )

    async def _get_personalized_suggestions(
        self, context: EnhancedContext, task_type: TaskType, user_input: str
    ) -> List[Any]:
        """Get personalized suggestions based on current context and user patterns."""

        if not self.suggestion_engine:
            return []

        recent_commands = (
            context.terminal_history[-5:] if context.terminal_history else []
        )

        try:
            suggestions = await self.suggestion_engine.get_suggestions(
                context, task_type, recent_commands
            )
            return suggestions
        except Exception as e:
            logger.warning(f"Failed to generate suggestions: {e}")
            return []

    async def provide_suggestion_feedback(
        self, suggestion_id: str, accepted: bool, feedback: Optional[str] = None
    ):
        """Record feedback on a suggestion for learning."""

        if self.suggestion_engine:
            await self.suggestion_engine.record_suggestion_interaction(
                suggestion_id, accepted, feedback
            )

    async def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session and learned patterns."""

        if not self.current_session or not self.memory_manager:
            return {"error": "No active session"}

        preferences = {
            pref_type: pref.preference_value
            for pref_type, pref in self.memory_manager.user_preferences.items()
            if pref.confidence > 0.6
        }

        return {
            "session_id": self.current_session.session_id,
            "turns_count": len(self.current_session.turns),
            "session_duration_minutes": (time.time() - self.current_session.start_time)
            / 60,
            "learned_preferences": preferences,
            "patterns_recognized": len(self.memory_manager.command_patterns),
            "working_context": {
                "project_type": self.current_session.project_context,
                "directory": self.current_session.working_directory,
            },
        }

    async def refresh_context(self) -> Dict[str, Any]:
        """Manually refresh the enhanced context."""

        context = await self._get_enhanced_context(force_refresh=True)
        return {
            "project_type": context.project.project_type.value,
            "project_name": context.project.project_name,
            "git_repo": context.git.is_repo,
            "relevant_files_count": len(context.relevant_files),
            "recent_errors_count": len(context.errors.recent_errors),
            "detection_time_ms": context.performance_metrics.get(
                "detection_time_ms", 0
            ),
        }


# Factory function for easy creation
async def create_intelligent_agent(
    name: str = "IntelligentAgent",
    working_directory: Optional[Path] = None,
    tools: Optional[List[BaseTool]] = None,
    **kwargs,
) -> IntelligentAgent:
    """Create an intelligent agent with default configuration."""

    # Import and create default tools if none provided
    if tools is None:
        from openagent.tools.git import GitTool, RepoGrep
        from openagent.tools.system import CommandExecutor

        tools = [
            CommandExecutor(),
            GitTool(),
            RepoGrep(),
        ]

    agent = IntelligentAgent(
        name=name,
        description="An intelligent, context-aware AI assistant for programming and system tasks",
        tools=tools,
        working_directory=working_directory,
        **kwargs,
    )

    return agent
