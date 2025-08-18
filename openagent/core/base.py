"""
Base classes for OpenAgent framework.

This module defines the foundational abstract base classes that all agents
and tools must inherit from, providing a consistent interface and behavior.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from openagent.core.exceptions import AgentError, ToolError


class BaseMessage(BaseModel):
    """Base class for messages exchanged between agents and tools."""

    content: str = Field(..., description="Message content")
    role: str = Field("user", description="Role of the message sender")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ToolResult(BaseModel):
    """Result returned by tool execution."""

    success: bool = Field(..., description="Whether the tool execution was successful")
    content: Union[str, Dict[str, Any]] = Field(..., description="Tool output content")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    Tools are discrete functionalities that agents can use to perform
    specific tasks like web searches, calculations, API calls, etc.
    """

    def __init__(self, name: str, description: str, **kwargs: Any) -> None:
        """
        Initialize the tool.

        Args:
            name: Unique name for the tool
            description: Description of what the tool does
            **kwargs: Additional tool-specific parameters
        """
        self.name = name
        self.description = description
        self.config = kwargs

    @abstractmethod
    async def execute(self, input_data: Union[str, Dict[str, Any]]) -> ToolResult:
        """
        Execute the tool with given input.

        Args:
            input_data: Input data for the tool

        Returns:
            ToolResult containing the execution result

        Raises:
            ToolError: If tool execution fails
        """
        pass

    def validate_input(self, input_data: Union[str, Dict[str, Any]]) -> bool:
        """
        Validate input data before execution.

        Args:
            input_data: Input data to validate

        Returns:
            True if input is valid, False otherwise
        """
        return True

    def get_schema(self) -> Dict[str, Any]:
        """
        Get the tool's input/output schema.

        Returns:
            Dictionary describing the tool's schema
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {},
            "returns": {
                "success": "boolean",
                "content": "string or object",
                "error": "optional string",
                "metadata": "object",
            },
        }


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Agents are autonomous entities that can process messages, use tools,
    and interact with other agents to accomplish tasks.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the agent.

        Args:
            name: Unique name for the agent
            description: Description of the agent's role/purpose
            tools: List of tools available to the agent
            **kwargs: Additional agent-specific parameters
        """
        self.name = name
        self.description = description
        self.tools = tools or []
        self.config = kwargs
        self.message_history: List[BaseMessage] = []

    @abstractmethod
    async def process_message(self, message: Union[str, BaseMessage]) -> BaseMessage:
        """
        Process an incoming message and generate a response.

        Args:
            message: Input message to process

        Returns:
            BaseMessage containing the agent's response

        Raises:
            AgentError: If message processing fails
        """
        pass

    def add_tool(self, tool: BaseTool) -> None:
        """
        Add a tool to the agent's toolkit.

        Args:
            tool: Tool to add

        Raises:
            AgentError: If tool cannot be added
        """
        if not isinstance(tool, BaseTool):
            raise AgentError(f"Expected BaseTool instance, got {type(tool)}")

        # Check for duplicate tool names
        if any(t.name == tool.name for t in self.tools):
            raise AgentError(f"Tool with name '{tool.name}' already exists")

        self.tools.append(tool)

    def remove_tool(self, tool_name: str) -> bool:
        """
        Remove a tool from the agent's toolkit.

        Args:
            tool_name: Name of the tool to remove

        Returns:
            True if tool was removed, False if not found
        """
        for i, tool in enumerate(self.tools):
            if tool.name == tool_name:
                self.tools.pop(i)
                return True
        return False

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.

        Args:
            tool_name: Name of the tool to retrieve

        Returns:
            BaseTool instance or None if not found
        """
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Get a list of all available tools with their schemas.

        Returns:
            List of tool schemas
        """
        return [tool.get_schema() for tool in self.tools]

    def add_message(self, message: BaseMessage) -> None:
        """
        Add a message to the conversation history.

        Args:
            message: Message to add to history
        """
        self.message_history.append(message)

    def get_conversation_history(
        self, limit: Optional[int] = None
    ) -> List[BaseMessage]:
        """
        Get conversation history.

        Args:
            limit: Maximum number of messages to return

        Returns:
            List of messages from conversation history
        """
        if limit is None:
            return self.message_history.copy()
        return self.message_history[-limit:] if limit > 0 else []

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.message_history.clear()
