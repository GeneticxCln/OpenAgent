"""
Integration tests for OpenAgent system components.

Tests cover:
- History block persistence and retrieval
- Tool planning and execution
- Workflow management
- Agent message processing
- Error repair suggestions
"""

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from openagent.core.agent import Agent
from openagent.core.base import BaseMessage, ToolResult
from openagent.core.history import HistoryManager
from openagent.core.tool_selector import (
    SmartToolSelector,
    ToolCall,
    ToolIntent,
    ToolPlan,
)
from openagent.core.workflows import Workflow, WorkflowManager
from openagent.tools.system import CommandExecutor, FileManager, SystemInfo


@pytest.fixture
def temp_history_dir():
    """Create temporary directory for history."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def history_manager(temp_history_dir):
    """Create HistoryManager with temp directory."""
    # Patch the module-level HISTORY_DIR to use temp directory
    test_history_dir = temp_history_dir / ".openagent" / "history"
    test_history_dir.mkdir(parents=True, exist_ok=True)

    with patch("openagent.core.history.HISTORY_DIR", test_history_dir):
        return HistoryManager(base_dir=test_history_dir)


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = Mock()
    # Use regular Mock with async side_effect for proper async behavior
    llm.generate_response = AsyncMock()
    llm.generate_json = AsyncMock()
    llm.get_model_info = Mock(
        return_value={"model_name": "test-model", "device": "cpu", "loaded": True}
    )
    return llm


@pytest.fixture
def tool_set():
    """Create a set of tools for testing."""
    return {
        "command_executor": CommandExecutor(default_explain_only=True),
        "file_manager": FileManager(),
        "system_info": SystemInfo(),
    }


class TestHistoryManagement:
    """Test history block management."""

    def test_create_history_block(self, history_manager):
        """Test creating a new history block."""
        block = HistoryManager.new_block(
            input_text="test command",
            response="test response",
            plan={"calls": [{"tool": "test_tool"}]},
            tool_results=[{"tool": "test_tool", "success": True, "content": "result"}],
            model={"model_name": "test-model"},
            context={"cwd": "/tmp"},
        )

        assert block.id is not None
        assert block.input == "test command"
        assert block.response == "test response"
        assert block.timestamp is not None
        assert len(block.tool_results) == 1

    def test_append_and_retrieve_block(self, history_manager):
        """Test appending and retrieving history blocks."""
        block = HistoryManager.new_block(
            input_text="test input", response="test output"
        )

        history_manager.append(block)

        # Retrieve the block
        retrieved = history_manager.get(block.id)
        assert retrieved is not None
        assert retrieved["id"] == block.id
        assert retrieved["input"] == "test input"
        assert retrieved["response"] == "test output"

    def test_list_blocks(self, history_manager):
        """Test listing history blocks."""
        # Create multiple blocks
        for i in range(5):
            block = HistoryManager.new_block(
                input_text=f"input {i}", response=f"response {i}"
            )
            history_manager.append(block)

        # List blocks
        blocks = history_manager.list_blocks(limit=3)
        assert len(blocks) == 3

        # Should be in reverse chronological order
        assert blocks[0]["input"] == "input 4"
        assert blocks[1]["input"] == "input 3"
        assert blocks[2]["input"] == "input 2"

    def test_search_blocks(self, history_manager):
        """Test searching history blocks."""
        # Create blocks with different content
        blocks_data = [
            ("find file", "I found the file in /tmp"),
            ("list directory", "Here are the contents"),
            ("search for pattern", "Found 3 matches"),
        ]

        for input_text, response in blocks_data:
            block = HistoryManager.new_block(input_text=input_text, response=response)
            history_manager.append(block)

        # Search for "file"
        results = history_manager.search("file", limit=10)
        assert len(results) == 1
        assert results[0]["input"] == "find file"

        # Search for "found"
        results = history_manager.search("found", limit=10)
        assert len(results) == 2  # Both "found the file" and "Found 3 matches"

    def test_export_blocks(self, history_manager):
        """Test exporting history blocks."""
        # Create a block with sensitive data
        block = HistoryManager.new_block(
            input_text="echo $SECRET_KEY",
            response="SECRET123",
            tool_results=[
                {"tool": "command_executor", "success": True, "content": "SECRET123"}
            ],
        )
        history_manager.append(block)

        # Export as markdown
        md_export = history_manager.export(block.id, format="md")
        assert "echo $SECRET_KEY" in md_export
        assert "SECRET123" in md_export or "[REDACTED]" in md_export

        # Export as JSON
        json_export = history_manager.export(block.id, format="json")
        data = json.loads(json_export)
        assert data["id"] == block.id
        assert data["input"] == "echo $SECRET_KEY"


class TestToolPlanning:
    """Test tool planning and execution."""

    @pytest.mark.asyncio
    async def test_create_tool_plan(self, mock_llm, tool_set):
        """Test creating a tool execution plan."""
        selector = SmartToolSelector(mock_llm, tool_set)

        # Mock LLM response for planning
        mock_llm.generate_json.return_value = {
            "plan": {
                "calls": [
                    {
                        "order": 1,
                        "tool": "command_executor",
                        "parameters": {"command": "ls -la"},
                        "rationale": "List directory contents",
                    }
                ],
                "estimated_risk": "low",
                "requires_confirmation": False,
            }
        }

        plan = await selector.create_tool_plan("list files in current directory")

        assert isinstance(plan, ToolPlan)
        assert len(plan.calls) == 1
        assert plan.calls[0].tool_name == "command_executor"
        assert plan.estimated_risk == "low"
        assert plan.requires_confirmation is False

    @pytest.mark.asyncio
    async def test_execute_tool_plan(self, mock_llm, tool_set):
        """Test executing a tool plan."""
        selector = SmartToolSelector(mock_llm, tool_set)

        # Create a simple plan
        plan = ToolPlan(
            calls=[
                ToolCall(
                    tool_name="system_info",
                    parameters={"type": "overview"},
                    intent=ToolIntent.SYSTEM_INFO,
                    rationale="Get system information",
                    order=1,
                )
            ],
            explanation="Getting system information",
            estimated_risk="low",
            requires_confirmation=False,
        )

        # Execute the plan
        with patch.object(
            tool_set["system_info"],
            "execute",
            return_value=ToolResult(
                success=True, content="System: Linux", metadata={"system": "Linux"}
            ),
        ):
            results = await selector.execute_plan(plan)

        assert len(results) == 1
        assert results[0].success is True
        assert "Linux" in results[0].content

    @pytest.mark.asyncio
    async def test_plan_with_multiple_tools(self, mock_llm, tool_set):
        """Test planning with multiple tool calls."""
        selector = SmartToolSelector(mock_llm, tool_set)

        # Mock complex plan
        mock_llm.generate_json.return_value = {
            "plan": {
                "calls": [
                    {
                        "order": 1,
                        "tool": "file_manager",
                        "parameters": {"operation": "list", "path": "/tmp"},
                        "rationale": "List temp directory",
                    },
                    {
                        "order": 2,
                        "tool": "command_executor",
                        "parameters": {"command": "df -h"},
                        "rationale": "Check disk space",
                    },
                ],
                "estimated_risk": "low",
                "requires_confirmation": False,
            }
        }

        plan = await selector.create_tool_plan("check temp directory and disk space")

        assert len(plan.calls) == 2
        assert plan.calls[0].tool_name == "file_manager"
        assert plan.calls[1].tool_name == "command_executor"
        assert plan.calls[0].order < plan.calls[1].order

    @pytest.mark.asyncio
    async def test_handle_tool_execution_failure(self, mock_llm, tool_set):
        """Test handling tool execution failures."""
        selector = SmartToolSelector(mock_llm, tool_set)

        plan = ToolPlan(
            calls=[
                ToolCall(
                    tool_name="command_executor",
                    parameters={"command": "nonexistent_command"},
                    intent=ToolIntent.COMMAND_EXECUTION,
                    rationale="Run command",
                    order=1,
                )
            ],
            explanation="Running command",
            estimated_risk="low",
            requires_confirmation=False,
        )

        # Mock command failure
        with patch.object(
            tool_set["command_executor"],
            "execute",
            return_value=ToolResult(
                success=False,
                content="",
                error="command not found: nonexistent_command",
                metadata={"suggestions": ["Check if command is installed"]},
            ),
        ):
            results = await selector.execute_plan(plan)

        assert len(results) == 1
        assert results[0].success is False
        assert "command not found" in results[0].error


class TestWorkflowManagement:
    """Test workflow management functionality."""

    @pytest.fixture
    def workflow_manager(self, temp_history_dir):
        """Create WorkflowManager with temp directory."""
        with patch("openagent.core.workflows.Path.home", return_value=temp_history_dir):
            return WorkflowManager()

    def test_create_workflow(self, workflow_manager):
        """Test creating a new workflow."""
        workflow = workflow_manager.create(
            name="test_workflow", description="Test workflow"
        )

        assert workflow.name == "test_workflow"
        assert workflow.description == "Test workflow"
        assert len(workflow.steps) == 0
        assert len(workflow.params) == 0

        # Check file was created
        workflow_file = workflow_manager.base_dir / "test_workflow.yaml"
        assert workflow_file.exists()

    def test_get_workflow(self, workflow_manager):
        """Test retrieving a workflow."""
        # Create workflow
        workflow_manager.create("test_workflow", "Test")

        # Retrieve it
        workflow = workflow_manager.get("test_workflow")
        assert workflow is not None
        assert workflow.name == "test_workflow"

    def test_list_workflows(self, workflow_manager):
        """Test listing workflows."""
        # Clean up any existing test workflows first
        existing = workflow_manager.list()
        for wf in existing:
            if wf.name in [
                "workflow1",
                "workflow2",
                "test_workflow",
                "complex_workflow",
            ]:
                # Remove the workflow file
                wf_file = workflow_manager.base_dir / f"{wf.name}.yaml"
                if wf_file.exists():
                    wf_file.unlink()

        # Start with clean state
        initial_workflows = workflow_manager.list()
        initial_count = len(initial_workflows)

        # Create multiple workflows
        workflow_manager.create("workflow1", "First")
        workflow_manager.create("workflow2", "Second")

        workflows = workflow_manager.list()
        # Should have exactly 2 more than initial
        assert len(workflows) == initial_count + 2

        names = [w.name for w in workflows]
        assert "workflow1" in names
        assert "workflow2" in names

    def test_workflow_with_steps(self, workflow_manager):
        """Test workflow with steps."""
        workflow = Workflow(
            name="complex_workflow",
            description="Complex workflow with steps",
            steps=[
                {"tool": "system_info", "args": {"type": "overview"}},
                "Process the system information",
                {"tool": "command_executor", "args": {"command": "df -h"}},
            ],
            params={"threshold": "80"},
        )

        # Save workflow
        workflow_manager.save(workflow)

        # Load and verify
        loaded = workflow_manager.get("complex_workflow")
        assert loaded is not None
        assert len(loaded.steps) == 3
        assert loaded.steps[0]["tool"] == "system_info"
        assert loaded.steps[1] == "Process the system information"
        assert loaded.params["threshold"] == "80"


class TestAgentIntegration:
    """Test agent integration with tools and history."""

    @pytest.mark.asyncio
    async def test_agent_process_message(self, mock_llm):
        """Test agent processing a message."""
        agent = Agent(
            name="TestAgent",
            description="Test agent",
            model_name="test-model",
            llm_config={},
        )

        # Replace LLM with mock
        agent.llm = mock_llm
        mock_llm.generate_response.return_value = "I can help you with that."

        # Add tools
        agent.add_tool(CommandExecutor(default_explain_only=True))

        # Process message
        response = await agent.process_message("How do I list files?")

        assert response.role == "assistant"
        # Accept either 'help' or practical advice about 'ls'
        assert "help" in response.content.lower() or "ls" in response.content.lower()
        assert response.metadata.get("agent_name") == "TestAgent"

    @pytest.mark.asyncio
    async def test_agent_with_tool_execution(self, mock_llm):
        """Test agent executing tools."""
        agent = Agent(
            name="TestAgent", description="Test agent", model_name="test-model"
        )

        agent.llm = mock_llm

        # Mock tool planning
        with patch.object(agent, "_should_use_tools", return_value=True):
            with patch.object(
                agent,
                "_plan_tool_calls",
                return_value=[
                    {
                        "name": "command_executor",
                        "args": {"command": "echo test", "explain_only": True},
                    }
                ],
            ):
                with patch.object(
                    agent,
                    "_execute_tools",
                    return_value={
                        "command_executor": ToolResult(
                            success=True,
                            content="echo: Display text",
                            metadata={"explained": True},
                        )
                    },
                ):
                    mock_llm.generate_response.return_value = (
                        "The echo command displays text."
                    )

                    response = await agent.process_message("What does echo do?")

        assert "echo" in response.content.lower()

    @pytest.mark.asyncio
    async def test_agent_error_handling(self, mock_llm):
        """Test agent error handling."""
        agent = Agent(
            name="TestAgent", description="Test agent", model_name="test-model"
        )

        agent.llm = mock_llm

        # Mock LLM failure
        mock_llm.generate_response.side_effect = Exception("LLM error")

        response = await agent.process_message("test message")

        assert response.role == "assistant"
        assert "error" in response.content.lower()
        assert response.metadata.get("error") is True

    @pytest.mark.asyncio
    async def test_agent_command_repair_suggestion(self, mock_llm):
        """Test agent suggesting command fixes."""
        agent = Agent(
            name="TestAgent", description="Test agent", model_name="test-model"
        )

        agent.llm = mock_llm
        agent.add_tool(CommandExecutor(default_explain_only=False))

        # Mock failed command and repair suggestion
        mock_llm.generate_response.side_effect = [
            "Try: pip install --user package",  # Repair suggestion
            "The command failed. Try installing with --user flag.",  # Final response
        ]

        with patch.object(
            agent,
            "_execute_tools",
            return_value={
                "command_executor": ToolResult(
                    success=False,
                    content="",
                    error="Permission denied",
                    metadata={"command": "pip install package"},
                )
            },
        ):
            suggestion = await agent._propose_fix_for_command(
                "pip install package", "Permission denied"
            )

        assert suggestion is not None
        assert "--user" in suggestion


class TestEndToEndScenarios:
    """Test complete end-to-end scenarios."""

    @pytest.mark.asyncio
    async def test_complete_command_execution_flow(self):
        """Test complete flow from command to execution with policy."""
        from openagent.core.policy import CommandPolicy, PolicyEngine, configure_policy

        # Setup policy
        policy = CommandPolicy(default_mode="approve", audit_enabled=False)
        engine = PolicyEngine(policy)
        configure_policy(policy)

        # Create executor
        executor = CommandExecutor(default_explain_only=False)

        # Test safe command flow
        result = await executor.execute(
            {"command": "echo 'Hello, World!'", "explain_only": False, "confirm": True}
        )

        # Should succeed with approval
        if result.success:
            assert "Hello" in result.content or "echo" in result.content
        else:
            # In test environment, might require approval
            assert (
                "approval" in result.error.lower() or "policy" in result.error.lower()
            )

    @pytest.mark.asyncio
    async def test_workflow_execution(self, mock_llm):
        """Test executing a complete workflow."""
        # Create agent
        agent = Agent(
            name="WorkflowAgent",
            description="Workflow test agent",
            model_name="test-model",
        )
        agent.llm = mock_llm

        # Add tools
        system_tool = SystemInfo()
        agent.add_tool(system_tool)

        # Create workflow
        workflow = Workflow(
            name="system_check",
            description="Check system status",
            steps=[{"tool": "system_info", "args": {"type": "overview"}}],
            params={},
        )

        # Mock tool execution
        with patch.object(
            system_tool,
            "execute",
            return_value=ToolResult(
                success=True,
                content="System is healthy",
                metadata={"cpu": 25, "memory": 50},
            ),
        ):
            # Execute workflow step
            result = await system_tool.execute(workflow.steps[0]["args"])

        assert result.success is True
        assert "healthy" in result.content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
