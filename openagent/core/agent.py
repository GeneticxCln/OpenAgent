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
        
        Args:
            input_text: Input text to respond to
            
        Returns:
            Generated response text
        """
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
        
        # Create system prompt based on agent role and query type
        if is_code_query:
            system_prompt = f"""
You are {self.name}, an expert programming and terminal assistant. You help users with:
- Writing and debugging code
- Explaining commands and scripts
- Code optimization and best practices
- Terminal operations and automation
- Software development workflows

{self.description}

Provide clear, accurate, and actionable responses. When providing code, include explanations.
If asked about terminal commands, explain what they do and any potential risks.
"""
        else:
            system_prompt = f"""
You are {self.name}, an intelligent AI assistant. 

{self.description}

You help users with various tasks and questions. Provide helpful, accurate, and engaging responses.
Be conversational but professional. If you're unsure about something, say so.
"""
        
        # Check if we need to use tools
        tool_context = ""
        if self.tools and await self._should_use_tools(input_text):
            tool_results = await self._execute_tools(input_text)
            if tool_results:
                tool_context = "\n\nTool Results:\n"
                for tool_name, result in tool_results.items():
                    if result.success:
                        tool_context += f"- {tool_name}: {result.content}\n"
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
        # Simple keyword-based detection (replace with LLM analysis)
        tool_keywords = {
            "calculate", "compute", "math", "addition", "subtraction",
            "search", "find", "look up", "web", "internet",
            "weather", "temperature", "forecast",
            "time", "date", "clock"
        }
        
        input_lower = input_text.lower()
        return any(keyword in input_lower for keyword in tool_keywords)
    
    async def _execute_tools(self, input_text: str) -> Dict[str, ToolResult]:
        """
        Execute relevant tools based on input text.
        
        Args:
            input_text: Input text to process with tools
            
        Returns:
            Dictionary mapping tool names to their results
        """
        results = {}
        
        for tool in self.tools:
            try:
                # Simple tool selection logic (replace with LLM-based selection)
                if await self._tool_is_relevant(tool, input_text):
                    logger.info(f"Executing tool: {tool.name}")
                    
                    # Execute the tool
                    result = await tool.execute(input_text)
                    results[tool.name] = result
                    self._tools_used.append(tool.name)
                    
                    self.iteration_count += 1
                    if self.iteration_count >= self.max_iterations:
                        logger.warning(f"Maximum iterations ({self.max_iterations}) reached")
                        break
                        
            except Exception as e:
                logger.error(f"Error executing tool {tool.name}: {e}")
                results[tool.name] = ToolResult(
                    success=False,
                    content="",
                    error=str(e)
                )
        
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
