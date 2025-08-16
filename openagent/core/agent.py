"""
Main Agent implementation for OpenAgent framework.

This module provides the concrete Agent class that implements the BaseAgent
interface with practical functionality for message processing and tool usage.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from openagent.core.base import BaseAgent, BaseMessage, BaseTool, ToolResult
from openagent.core.exceptions import AgentError, ToolError
from openagent.core.llm import HuggingFaceLLM, get_llm


logger = logging.getLogger(__name__)


class Agent(BaseAgent):
    """
    Main Agent implementation with tool integration and message processing.
    
    This class provides a practical implementation of an AI agent that can
    process messages, use tools, and maintain conversation history.
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        tools: Optional[List[BaseTool]] = None,
        max_iterations: int = 10,
        model_name: str = "codellama-7b",
        llm_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the Agent.
        
        Args:
            name: Unique name for the agent
            description: Description of the agent's role/purpose
            tools: List of tools available to the agent
            max_iterations: Maximum iterations for tool usage loops
            model_name: Hugging Face model to use for responses
            llm_config: Configuration for the LLM
            **kwargs: Additional agent-specific parameters
        """
        super().__init__(name, description, tools, **kwargs)
        self.max_iterations = max_iterations
        self.iteration_count = 0
        self.model_name = model_name
        
        # Initialize LLM
        llm_config = llm_config or {}
        self.llm = HuggingFaceLLM(model_name=model_name, **llm_config)
        
        # Agent state
        self.is_processing = False
        self.current_task = None
        
        logger.info(f"Initialized agent '{name}' with {len(self.tools)} tools and model '{model_name}'")
    
    async def process_message(self, message: Union[str, BaseMessage]) -> BaseMessage:
        """
        Process an incoming message and generate a response.
        
        This method handles the main agent workflow:
        1. Parse and validate the input message
        2. Analyze if tools are needed
        3. Execute tools if necessary
        4. Generate and return a response
        
        Args:
            message: Input message to process
            
        Returns:
            BaseMessage containing the agent's response
            
        Raises:
            AgentError: If message processing fails
        """
        if self.is_processing:
            raise AgentError("Agent is already processing a message")
        
        try:
            self.is_processing = True
            self.iteration_count = 0
            
            # Convert string to BaseMessage if needed
            if isinstance(message, str):
                input_message = BaseMessage(content=message, role="user")
            else:
                input_message = message
            
            # Add to conversation history
            self.add_message(input_message)
            
            logger.info(f"Agent '{self.name}' processing message: {input_message.content[:100]}...")
            
            # Process the message and generate response
            response_content = await self._generate_response(input_message.content)
            
            # Create response message
            response = BaseMessage(
                content=response_content,
                role="assistant",
                metadata={
                    "agent_name": self.name,
                    "tools_used": getattr(self, '_tools_used', []),
                    "iterations": self.iteration_count
                }
            )
            
            # Add response to history
            self.add_message(response)
            
            logger.info(f"Agent '{self.name}' generated response")
            return response
            
        except Exception as e:
            logger.error(f"Error processing message in agent '{self.name}': {e}")
            error_response = BaseMessage(
                content=f"I encountered an error while processing your request: {str(e)}",
                role="assistant",
                metadata={"error": True, "agent_name": self.name}
            )
            self.add_message(error_response)
            return error_response
        finally:
            self.is_processing = False
            self.current_task = None
            self.iteration_count = 0
    
    async def _generate_response(self, input_text: str) -> str:
        """
        Generate a response using the integrated Hugging Face LLM.
        Attempts fast fallback first for common queries.
        
        Args:
            input_text: Input text to respond to
            
        Returns:
            Generated response text
        """
        # Import fallback functions locally to avoid circular imports
        from ..core.fallback import can_handle_fast, handle_fast
        
        # Try fast fallback first for common queries
        if can_handle_fast(input_text):
            fallback_response = handle_fast(input_text)
            if fallback_response:
                logger.info(f"Using fast fallback response for query: {input_text[:50]}...")
                return fallback_response
        
        # Reset tools used tracking
        self._tools_used = []
        
        # Get conversation context
        context = []
        for msg in self.message_history[-5:]:  # Last 5 messages for context
            context.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Determine if this is a code/terminal related query
        code_keywords = ["code", "script", "function", "class", "command", "terminal", "bash", "python", "programming"]
        is_code_query = any(keyword in input_text.lower() for keyword in code_keywords)
        
        # Build context block for prompt injection (cwd, shell, OS, recent git)
        try:
            from openagent.core.context import gather_context
            sysctx = gather_context()
            ctx_block = sysctx.to_prompt_block()
        except Exception:
            ctx_block = None

        # Create system prompt based on agent role and query type, with context
        base_system = (
            f"You are {self.name}, an expert programming and terminal assistant.\n"
            if is_code_query
            else f"You are {self.name}, an intelligent AI assistant.\n"
        )
        guidance = (
            "Provide clear, accurate, and actionable responses. When providing code, include explanations.\n"
            "If asked about terminal commands, explain what they do and any potential risks.\n"
            if is_code_query
            else "Provide helpful, accurate, and engaging responses. Be concise and professional. If unsure, say so.\n"
        )
        system_prompt = base_system + (self.description or "") + "\n" + guidance
        if ctx_block:
            system_prompt += "\nContext:\n" + ctx_block
        
        # Check if we need to use tools
        tool_context = ""
        planned_calls = None
        if self.tools and await self._should_use_tools(input_text):
            # Ask model to propose structured tool calls
            planned_calls = await self._plan_tool_calls(input_text)
            tool_results = await self._execute_tools(input_text, planned_calls)
            if tool_results:
                tool_context = "\n\nTool Results:\n"
                for tool_name, result in tool_results.items():
                    if result.success:
                        tool_context += f"- {tool_name}: {str(result.content)[:800]}\n"
                    else:
                        tool_context += f"- {tool_name}: Error - {result.error}\n"
        
        # Prepare the final prompt with tool context if available
        final_input = input_text
        if tool_context:
            final_input += tool_context + "\n\nPlease provide a comprehensive response using the tool results above."
        
        try:
            # Generate response using the LLM
            response = await self.llm.generate_response(
                prompt=final_input,
                system_prompt=system_prompt,
                context=context[-3:] if context else None,  # Last 3 exchanges
                max_new_tokens=1024
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Propagate to outer handler so error metadata is set
            raise AgentError(f"LLM generation failed: {e}")
    
    async def _should_use_tools(self, input_text: str) -> bool:
        """
        Determine if tools should be used for this input.
        
        Args:
            input_text: Input text to analyze
            
        Returns:
            True if tools should be used
        """
        # Prefer using LLM to decide if tools are needed for non-trivial tasks
        input_lower = input_text.lower()
        heuristics = ["run", "execute", "search", "grep", "status", "log", "diff", "list files", "read file", "show system"]
        return any(k in input_lower for k in heuristics)
    
    async def _execute_tools(self, input_text: str, planned_calls: Optional[List[Dict[str, Any]]] = None) -> Dict[str, ToolResult]:
        """
        Execute relevant tools based on input text.
        
        Args:
            input_text: Input text to process with tools
            
        Returns:
            Dictionary mapping tool names to their results
        """
        results: Dict[str, ToolResult] = {}
        calls: List[Dict[str, Any]]
        if planned_calls is not None:
            calls = planned_calls
        else:
            # Fallback: naive selection over all tools
            calls = [{"name": t.name, "args": input_text} for t in self.tools if await self._tool_is_relevant(t, input_text)]

        for call in calls:
            try:
                tool = self.get_tool(call.get("name", ""))
                if not tool:
                    continue
                logger.info(f"Executing tool: {tool.name}")
                args = call.get("args", input_text)
                result = await tool.execute(args)
                results[tool.name] = result
                self._tools_used.append(tool.name)
                self.iteration_count += 1
                if self.iteration_count >= self.max_iterations:
                    logger.warning(f"Maximum iterations ({self.max_iterations}) reached")
                    break
            except Exception as e:
                logger.error(f"Error executing tool {call}: {e}")
                results[call.get("name", "unknown")] = ToolResult(success=False, content="", error=str(e))
        return results
    
    def _tool_is_relevant_sync(self, tool: BaseTool, input_text: str) -> bool:
        """
        Synchronous logic to check if a tool is relevant for the given input.
        Used for both runtime and tests (via __wrapped__).
        """
        # Simple keyword-based relevance (replace with LLM analysis)
        input_lower = input_text.lower()
        tool_name_lower = tool.name.lower()
        
        # Basic keyword matching
        if "calculator" in tool_name_lower or "math" in tool_name_lower:
            return any(word in input_lower for word in ["calculate", "compute", "math", "+", "-", "*", "/"])
        
        if "search" in tool_name_lower or "web" in tool_name_lower:
            return any(word in input_lower for word in ["search", "find", "look up", "web", "internet"])
        
        if "weather" in tool_name_lower:
            return any(word in input_lower for word in ["weather", "temperature", "forecast", "rain", "sunny"])
        
        # Default: always consider relevant if no specific matching logic
        return True

    async def _tool_is_relevant(self, tool: BaseTool, input_text: str) -> bool:
        """
        Check if a tool is relevant for the given input.
        
        Args:
            tool: Tool to check
            input_text: Input text to analyze
            
        Returns:
            True if tool is relevant
        """
        return self._tool_is_relevant_sync(tool, input_text)

    async def _plan_tool_calls(self, input_text: str) -> Optional[List[Dict[str, Any]]]:
        """Ask the model to propose structured tool calls in JSON.
        Schema: {"tool_calls": [{"name": str, "args": object}]}
        """
        try:
            tool_schemas = self.list_tools()
            schema_text = str(tool_schemas)
            prompt = (
                "Given the user's request, choose zero or more tools from the following list and "
                "return a JSON object with a 'tool_calls' array. Each item must include 'name' (one of the tool names) "
                "and 'args' (object or string appropriate for that tool). If no tools are needed, return {\"tool_calls\": []}.\n\n"
                f"Tools: {schema_text}\n\nRequest: {input_text}\nOnly return JSON."
            )
            raw = await self.llm.generate_response(prompt, system_prompt="Tool Planner")
            import json
            # Extract JSON if wrapped
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1:
                return None
            obj = json.loads(raw[start:end+1])
            calls = obj.get("tool_calls")
            if isinstance(calls, list):
                # keep only known tools
                valid_names = {t.name for t in self.tools}
                filtered: List[Dict[str, Any]] = []
                for c in calls:
                    if isinstance(c, dict) and c.get("name") in valid_names:
                        filtered.append({"name": c["name"], "args": c.get("args")})
                return filtered
            return None
        except Exception:
            return None
    
    async def _generate_main_response(self, input_text: str, tool_results: List[str]) -> str:
        """
        Generate the main response text.
        
        Args:
            input_text: Original input text
            tool_results: Results from tool execution
            
        Returns:
            Generated response text
        """
        # Simple response generation (replace with LLM)
        responses = [
            f"Hello! I'm {self.name}, an AI agent here to help you.",
            "I understand you're asking about something. Let me help you with that.",
            "Based on your message, I'll do my best to provide a helpful response.",
            f"As an AI agent named {self.name}, I'm designed to assist with various tasks.",
        ]
        
        # Select response based on input length or content
        if len(input_text) > 100:
            response = "Thank you for your detailed message. I've processed your request and here's my response:"
        elif any(word in input_text.lower() for word in ["hello", "hi", "hey"]):
            response = f"Hello! I'm {self.name}. How can I help you today?"
        elif "?" in input_text:
            response = "That's a great question! Let me help you find an answer."
        else:
            response = responses[hash(input_text) % len(responses)]
        
        return response
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status.
        
        Returns:
            Dictionary containing agent status information
        """
        return {
            "name": self.name,
            "description": self.description,
            "is_processing": self.is_processing,
            "current_task": self.current_task,
            "tools_count": len(self.tools),
            "tools": [tool.name for tool in self.tools],
            "message_history_length": len(self.message_history),
            "iteration_count": self.iteration_count,
            "max_iterations": self.max_iterations,
        }
    
    def reset(self) -> None:
        """Reset agent state and clear history."""
        self.clear_history()
        self.is_processing = False
        self.current_task = None
        self.iteration_count = 0
        logger.info(f"Agent '{self.name}' has been reset")

# Expose __wrapped__ for testing convenience
Agent._tool_is_relevant.__wrapped__ = Agent._tool_is_relevant_sync
