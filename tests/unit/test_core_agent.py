"""
Unit tests for the core Agent class.

Tests agent initialization, message processing, tool management,
and error handling scenarios.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from openagent.core.agent import Agent
from openagent.core.base import BaseMessage, BaseTool, ToolResult
from openagent.core.exceptions import AgentError


class MockTool(BaseTool):
    """Mock tool for testing."""
    
    def __init__(self, name="mock_tool", should_fail=False):
        super().__init__(name=name, description="A mock tool for testing")
        self.should_fail = should_fail
        self.execution_count = 0
    
    async def execute(self, input_data):
        self.execution_count += 1
        if self.should_fail:
            return ToolResult(
                success=False,
                content="",
                error="Mock tool failure"
            )
        return ToolResult(
            success=True,
            content=f"Mock tool executed with: {input_data}",
            metadata={"execution_count": self.execution_count}
        )


class TestAgent:
    """Test suite for Agent class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent_name = "TestAgent"
        self.agent_description = "Test agent for unit testing"
        
    def test_agent_initialization(self):
        """Test agent initialization with various parameters."""
        # Basic initialization
        agent = Agent(
            name=self.agent_name,
            description=self.agent_description,
            model_name="tiny-llama"
        )
        
        assert agent.name == self.agent_name
        assert agent.description == self.agent_description
        assert agent.model_name == "tiny-llama"
        assert len(agent.tools) == 0
        assert len(agent.message_history) == 0
        assert not agent.is_processing
        
    def test_agent_with_tools(self):
        """Test agent initialization with tools."""
        tools = [MockTool("tool1"), MockTool("tool2")]
        agent = Agent(
            name=self.agent_name,
            description=self.agent_description,
            tools=tools,
            model_name="tiny-llama"
        )
        
        assert len(agent.tools) == 2
        assert agent.tools[0].name == "tool1"
        assert agent.tools[1].name == "tool2"
        
    def test_add_tool(self):
        """Test adding tools to agent."""
        agent = Agent(
            name=self.agent_name,
            description=self.agent_description,
            model_name="tiny-llama"
        )
        
        tool = MockTool("new_tool")
        agent.add_tool(tool)
        
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "new_tool"
        
    def test_add_duplicate_tool_raises_error(self):
        """Test that adding duplicate tools raises an error."""
        agent = Agent(
            name=self.agent_name,
            description=self.agent_description,
            model_name="tiny-llama"
        )
        
        tool1 = MockTool("duplicate")
        tool2 = MockTool("duplicate")
        
        agent.add_tool(tool1)
        
        with pytest.raises(AgentError, match="already exists"):
            agent.add_tool(tool2)
            
    def test_remove_tool(self):
        """Test removing tools from agent."""
        tool = MockTool("removable_tool")
        agent = Agent(
            name=self.agent_name,
            description=self.agent_description,
            tools=[tool],
            model_name="tiny-llama"
        )
        
        assert len(agent.tools) == 1
        
        # Remove existing tool
        result = agent.remove_tool("removable_tool")
        assert result is True
        assert len(agent.tools) == 0
        
        # Try to remove non-existent tool
        result = agent.remove_tool("nonexistent")
        assert result is False
        
    def test_get_tool(self):
        """Test getting tools by name."""
        tool = MockTool("findable_tool")
        agent = Agent(
            name=self.agent_name,
            description=self.agent_description,
            tools=[tool],
            model_name="tiny-llama"
        )
        
        # Find existing tool
        found_tool = agent.get_tool("findable_tool")
        assert found_tool is not None
        assert found_tool.name == "findable_tool"
        
        # Try to find non-existent tool
        not_found = agent.get_tool("nonexistent")
        assert not_found is None
        
    @pytest.mark.asyncio
    async def test_process_message_string(self):
        """Test processing string messages."""
        with patch('openagent.core.agent.HuggingFaceLLM') as mock_llm_class:
            mock_llm = AsyncMock()
            mock_llm.generate_response.return_value = "Test response"
            mock_llm_class.return_value = mock_llm
            
            agent = Agent(
                name=self.agent_name,
                description=self.agent_description,
                model_name="tiny-llama"
            )
            
            response = await agent.process_message("Hello, agent!")
            
            assert isinstance(response, BaseMessage)
            assert response.content == "Test response"
            assert response.role == "assistant"
            assert response.metadata["agent_name"] == self.agent_name
            assert len(agent.message_history) == 2  # Input + output
            
    @pytest.mark.asyncio
    async def test_process_message_object(self):
        """Test processing BaseMessage objects."""
        with patch('openagent.core.agent.HuggingFaceLLM') as mock_llm_class:
            mock_llm = AsyncMock()
            mock_llm.generate_response.return_value = "Test response"
            mock_llm_class.return_value = mock_llm
            
            agent = Agent(
                name=self.agent_name,
                description=self.agent_description,
                model_name="tiny-llama"
            )
            
            message = BaseMessage(content="Hello, agent!", role="user")
            response = await agent.process_message(message)
            
            assert isinstance(response, BaseMessage)
            assert response.content == "Test response"
            assert len(agent.message_history) == 2
            
    @pytest.mark.asyncio
    async def test_concurrent_processing_blocked(self):
        """Test that concurrent message processing is blocked."""
        with patch('openagent.core.agent.HuggingFaceLLM') as mock_llm_class:
            mock_llm = AsyncMock()
            mock_llm.generate_response.return_value = "Test response"
            mock_llm_class.return_value = mock_llm
            
            agent = Agent(
                name=self.agent_name,
                description=self.agent_description,
                model_name="tiny-llama"
            )
            
            # Set agent as processing
            agent.is_processing = True
            
            with pytest.raises(AgentError, match="already processing"):
                await agent.process_message("Test message")
                
    def test_get_status(self):
        """Test agent status reporting."""
        tools = [MockTool("tool1"), MockTool("tool2")]
        agent = Agent(
            name=self.agent_name,
            description=self.agent_description,
            tools=tools,
            model_name="tiny-llama",
            max_iterations=5
        )
        
        # Add some message history
        agent.add_message(BaseMessage(content="Test", role="user"))
        agent.add_message(BaseMessage(content="Response", role="assistant"))
        
        status = agent.get_status()
        
        assert status["name"] == self.agent_name
        assert status["description"] == self.agent_description
        assert status["tools_count"] == 2
        assert status["tools"] == ["tool1", "tool2"]
        assert status["message_history_length"] == 2
        assert status["max_iterations"] == 5
        assert not status["is_processing"]
        
    def test_reset(self):
        """Test agent reset functionality."""
        agent = Agent(
            name=self.agent_name,
            description=self.agent_description,
            model_name="tiny-llama"
        )
        
        # Add some state
        agent.add_message(BaseMessage(content="Test", role="user"))
        agent.is_processing = True
        agent.current_task = "test_task"
        agent.iteration_count = 3
        
        # Reset agent
        agent.reset()
        
        assert len(agent.message_history) == 0
        assert not agent.is_processing
        assert agent.current_task is None
        assert agent.iteration_count == 0
        
    @pytest.mark.asyncio
    async def test_tool_execution_in_message_processing(self):
        """Test that tools are executed during message processing."""
        mock_tool = MockTool("test_tool")
        
        with patch('openagent.core.agent.HuggingFaceLLM') as mock_llm_class:
            mock_llm = AsyncMock()
            mock_llm.generate_response.return_value = "Response with tool results"
            mock_llm_class.return_value = mock_llm
            
            agent = Agent(
                name=self.agent_name,
                description=self.agent_description,
                tools=[mock_tool],
                model_name="tiny-llama"
            )
            
            # Mock the tool relevance check to return True
            with patch.object(agent, '_should_use_tools', return_value=True), \
                 patch.object(agent, '_tool_is_relevant', return_value=True):
                
                response = await agent.process_message("calculate something")
                
                assert mock_tool.execution_count == 1
                assert "test_tool" in agent._tools_used
                
    @pytest.mark.asyncio 
    async def test_error_handling_in_message_processing(self):
        """Test error handling during message processing."""
        with patch('openagent.core.agent.HuggingFaceLLM') as mock_llm_class:
            mock_llm = AsyncMock()
            mock_llm.generate_response.side_effect = Exception("LLM Error")
            mock_llm_class.return_value = mock_llm
            
            agent = Agent(
                name=self.agent_name,
                description=self.agent_description,
                model_name="tiny-llama"
            )
            
            response = await agent.process_message("Test message")
            
            assert isinstance(response, BaseMessage)
            assert "error" in response.content.lower()
            assert response.metadata.get("error") is True
            
    def test_tool_is_relevant_basic_matching(self):
        """Test basic tool relevance matching."""
        agent = Agent(
            name=self.agent_name,
            description=self.agent_description,
            model_name="tiny-llama"
        )
        
        # Test calculator tool relevance
        calc_tool = MockTool("calculator")
        assert agent._tool_is_relevant.__wrapped__(agent, calc_tool, "calculate 2 + 2")
        assert agent._tool_is_relevant.__wrapped__(agent, calc_tool, "do some math")
        assert not agent._tool_is_relevant.__wrapped__(agent, calc_tool, "send an email")
        
        # Test search tool relevance
        search_tool = MockTool("search")
        assert agent._tool_is_relevant.__wrapped__(agent, search_tool, "search for information")
        assert agent._tool_is_relevant.__wrapped__(agent, search_tool, "find something")
        assert not agent._tool_is_relevant.__wrapped__(agent, search_tool, "calculate numbers")


if __name__ == "__main__":
    pytest.main([__file__])
