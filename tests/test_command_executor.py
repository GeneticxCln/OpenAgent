"""
Comprehensive tests for CommandExecutor with policy integration.

Tests cover:
- Command execution success/failure scenarios
- Policy enforcement (explain-only, approval, denial)
- Risk assessment accuracy
- Timeout handling
- Dangerous command detection
- Suggestion generation
- Sandbox execution
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path

from openagent.tools.system import CommandExecutor
from openagent.core.base import ToolResult
from openagent.core.policy import (
    PolicyEngine, CommandPolicy, PolicyDecision, RiskLevel,
    get_policy_engine, configure_policy
)


@pytest.fixture
def executor():
    """Create a CommandExecutor instance for testing."""
    return CommandExecutor(default_explain_only=True)


@pytest.fixture
def unsafe_executor():
    """Create a CommandExecutor that allows execution."""
    return CommandExecutor(default_explain_only=False)


@pytest.fixture
def mock_policy_engine():
    """Create a mock policy engine for testing."""
    engine = Mock(spec=PolicyEngine)
    engine.policy = CommandPolicy(
        default_mode="explain_only",
        audit_enabled=False  # Disable audit for tests
    )
    engine.evaluate_command = AsyncMock()
    engine.audit_command = AsyncMock()
    engine.execute_sandboxed = AsyncMock()
    return engine


class TestCommandExecutorBasics:
    """Test basic CommandExecutor functionality."""
    
    @pytest.mark.asyncio
    async def test_explain_only_mode(self, executor):
        """Test that explain-only mode doesn't execute."""
        result = await executor.execute({
            "command": "echo 'test'",
            "explain_only": True
        })
        
        assert result.success is True
        assert "echo: Display" in result.content or "Execute 'echo'" in result.content
        assert result.metadata.get("explained") is True
        assert result.metadata.get("risk_level") in ["low", "medium", "high", "blocked"]
    
    @pytest.mark.asyncio
    async def test_missing_command(self, executor):
        """Test handling of missing command."""
        result = await executor.execute("")
        
        assert result.success is False
        assert "No command provided" in result.error
    
    @pytest.mark.asyncio
    async def test_command_risk_assessment(self, executor):
        """Test risk assessment for various commands."""
        # Test low risk
        result = await executor.execute({
            "command": "ls -la",
            "explain_only": True
        })
        assert result.metadata.get("risk_level") == "low"
        
        # Test high risk - rm command
        result = await executor.execute({
            "command": "rm -rf /tmp/test",
            "explain_only": True
        })
        assert result.metadata.get("risk_level") in ["high", "blocked"]
        
        # Test high risk - sudo
        result = await executor.execute({
            "command": "sudo apt update",
            "explain_only": True
        })
        assert result.metadata.get("risk_level") in ["high", "blocked"]
    
    @pytest.mark.asyncio
    async def test_suggestion_generation(self, executor):
        """Test that suggestions are generated for common errors."""
        with patch.object(executor, '_execute_command', return_value={
            "success": False,
            "output": "",
            "error": "command not found: foobar",
            "exit_code": 127,
            "execution_time": 0.1
        }):
            # Mock policy to allow execution
            with patch('openagent.tools.system.get_policy_engine') as mock_get_policy:
                mock_engine = Mock()
                mock_engine.evaluate_command = AsyncMock(return_value=(
                    PolicyDecision.ALLOW, RiskLevel.LOW, ["safe command"]
                ))
                mock_engine.audit_command = AsyncMock()
                mock_get_policy.return_value = mock_engine
                
                result = await executor.execute({
                    "command": "foobar",
                    "explain_only": False,
                    "confirm": True
                })
                
                assert result.success is False
                suggestions = result.metadata.get("suggestions", [])
                assert any("Install" in s or "PATH" in s for s in suggestions)


class TestPolicyIntegration:
    """Test integration with policy engine."""
    
    @pytest.mark.asyncio
    async def test_policy_denial(self, unsafe_executor):
        """Test that dangerous commands are denied by policy."""
        with patch('openagent.tools.system.get_policy_engine') as mock_get_policy:
            mock_engine = Mock()
            mock_engine.evaluate_command = AsyncMock(return_value=(
                PolicyDecision.DENY, RiskLevel.BLOCKED, ["Matched denylist: fork bomb"]
            ))
            mock_engine.audit_command = AsyncMock()
            mock_get_policy.return_value = mock_engine
            
            result = await unsafe_executor.execute({
                "command": ":(){ :|:& };:",
                "explain_only": False
            })
            
            assert result.success is False
            assert "denied by policy" in result.error.lower()
            assert result.metadata.get("policy_decision") == PolicyDecision.DENY.value
            
            # Verify audit was called
            mock_engine.audit_command.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_approval_requirement(self, unsafe_executor):
        """Test that medium risk commands require approval."""
        with patch('openagent.tools.system.get_policy_engine') as mock_get_policy:
            mock_engine = Mock()
            mock_engine.evaluate_command = AsyncMock(return_value=(
                PolicyDecision.REQUIRE_APPROVAL, RiskLevel.MEDIUM, ["Modifies permissions"]
            ))
            mock_engine.audit_command = AsyncMock()
            mock_get_policy.return_value = mock_engine
            
            # Without confirmation
            result = await unsafe_executor.execute({
                "command": "chmod 755 test.sh",
                "explain_only": False,
                "confirm": False
            })
            
            assert result.success is False
            assert "requires approval" in result.error.lower()
            
            # With confirmation - should proceed
            mock_engine.evaluate_command.return_value = (
                PolicyDecision.ALLOW, RiskLevel.MEDIUM, ["Approved by user"]
            )
            
            with patch.object(unsafe_executor, '_execute_command', return_value={
                "success": True,
                "output": "Command executed",
                "error": None,
                "exit_code": 0,
                "execution_time": 0.1
            }):
                result = await unsafe_executor.execute({
                    "command": "chmod 755 test.sh",
                    "explain_only": False,
                    "confirm": True
                })
                
                assert result.success is True
    
    @pytest.mark.asyncio
    async def test_sandbox_execution(self, unsafe_executor):
        """Test sandbox execution when enabled."""
        with patch('openagent.tools.system.get_policy_engine') as mock_get_policy:
            mock_engine = Mock()
            mock_engine.policy = CommandPolicy(sandbox_mode=True, audit_enabled=False)
            mock_engine.evaluate_command = AsyncMock(return_value=(
                PolicyDecision.ALLOW, RiskLevel.LOW, ["Safe command"]
            ))
            mock_engine.audit_command = AsyncMock()
            mock_engine.execute_sandboxed = AsyncMock(return_value={
                "success": True,
                "stdout": "Sandboxed output",
                "stderr": "",
                "exit_code": 0,
                "sandboxed": True
            })
            mock_get_policy.return_value = mock_engine
            
            result = await unsafe_executor.execute({
                "command": "echo 'test'",
                "explain_only": False,
                "sandbox": True
            })
            
            assert result.success is True
            assert result.content == "Sandboxed output"
            assert result.metadata.get("sandboxed") is True
            
            # Verify sandbox was used
            mock_engine.execute_sandboxed.assert_called_once()


class TestCommandExecution:
    """Test actual command execution."""
    
    @pytest.mark.asyncio
    async def test_successful_command(self, unsafe_executor):
        """Test successful command execution."""
        with patch('openagent.tools.system.get_policy_engine') as mock_get_policy:
            mock_engine = Mock()
            mock_engine.evaluate_command = AsyncMock(return_value=(
                PolicyDecision.ALLOW, RiskLevel.LOW, ["Safe command"]
            ))
            mock_engine.audit_command = AsyncMock()
            mock_get_policy.return_value = mock_engine
            
            with patch.object(unsafe_executor, '_execute_command', return_value={
                "success": True,
                "output": "Hello, World!",
                "error": None,
                "exit_code": 0,
                "execution_time": 0.05
            }):
                result = await unsafe_executor.execute({
                    "command": "echo 'Hello, World!'",
                    "explain_only": False
                })
                
                assert result.success is True
                assert result.content == "Hello, World!"
                assert result.metadata.get("exit_code") == 0
    
    @pytest.mark.asyncio
    async def test_failed_command(self, unsafe_executor):
        """Test failed command execution."""
        with patch('openagent.tools.system.get_policy_engine') as mock_get_policy:
            mock_engine = Mock()
            mock_engine.evaluate_command = AsyncMock(return_value=(
                PolicyDecision.ALLOW, RiskLevel.LOW, ["Safe command"]
            ))
            mock_engine.audit_command = AsyncMock()
            mock_get_policy.return_value = mock_engine
            
            with patch.object(unsafe_executor, '_execute_command', return_value={
                "success": False,
                "output": "",
                "error": "No such file or directory",
                "exit_code": 1,
                "execution_time": 0.01
            }):
                result = await unsafe_executor.execute({
                    "command": "cat /nonexistent/file",
                    "explain_only": False
                })
                
                assert result.success is False
                assert "No such file or directory" in result.error
                assert result.metadata.get("exit_code") == 1
                
                # Check suggestions
                suggestions = result.metadata.get("suggestions", [])
                assert any("Verify the path" in s for s in suggestions)
    
    @pytest.mark.asyncio
    async def test_command_timeout(self, unsafe_executor):
        """Test command timeout handling."""
        with patch('openagent.tools.system.get_policy_engine') as mock_get_policy:
            mock_engine = Mock()
            mock_engine.evaluate_command = AsyncMock(return_value=(
                PolicyDecision.ALLOW, RiskLevel.LOW, ["Safe command"]
            ))
            mock_engine.audit_command = AsyncMock()
            mock_get_policy.return_value = mock_engine
            
            # Mock timeout scenario
            async def mock_timeout_command(cmd):
                await asyncio.sleep(0.1)  # Simulate delay
                return {
                    "success": False,
                    "output": "",
                    "error": "Command timeout after 30 seconds",
                    "exit_code": -1,
                    "execution_time": 30.0
                }
            
            with patch.object(unsafe_executor, '_execute_command', side_effect=mock_timeout_command):
                result = await unsafe_executor.execute({
                    "command": "sleep 60",
                    "explain_only": False
                })
                
                assert result.success is False
                assert "timeout" in result.error.lower()
                assert result.metadata.get("exit_code") == -1


class TestExplanations:
    """Test command explanation functionality."""
    
    @pytest.mark.asyncio
    async def test_basic_command_explanations(self, executor):
        """Test explanations for common commands."""
        commands_and_keywords = [
            ("ls -la", ["List", "directory"]),
            ("grep pattern file.txt", ["Search", "pattern"]),
            ("git status", ["Git", "version control"]),
            ("pip install requests", ["Python", "package"]),
            ("rm file.txt", ["Remove", "delete"]),
        ]
        
        for command, keywords in commands_and_keywords:
            result = await executor.execute({
                "command": command,
                "explain_only": True
            })
            
            assert result.success is True
            explanation = result.content.lower()
            assert any(keyword.lower() in explanation for keyword in keywords)
    
    @pytest.mark.asyncio
    async def test_complex_command_explanation(self, executor):
        """Test explanation includes flags."""
        result = await executor.execute({
            "command": "ls -la --color=auto /tmp",
            "explain_only": True
        })
        
        assert result.success is True
        assert "-la" in result.content or "-l" in result.content or "flags" in result.content


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_malformed_input(self, executor):
        """Test handling of malformed input."""
        result = await executor.execute(None)
        assert result.success is False
        assert "No command provided" in result.error
        
        result = await executor.execute({})
        assert result.success is False
        assert "No command provided" in result.error
    
    @pytest.mark.asyncio
    async def test_exception_handling(self, executor):
        """Test exception handling during execution."""
        with patch('openagent.tools.system.get_policy_engine') as mock_get_policy:
            mock_engine = Mock()
            mock_engine.evaluate_command = AsyncMock(side_effect=Exception("Policy error"))
            mock_get_policy.return_value = mock_engine
            
            result = await executor.execute({
                "command": "echo test",
                "explain_only": False
            })
            
            assert result.success is False
            assert "Command execution failed" in result.error


@pytest.mark.asyncio
async def test_audit_logging():
    """Test that audit logging works correctly."""
    from openagent.core.policy import PolicyEngine, CommandPolicy
    import tempfile
    import json
    
    with tempfile.TemporaryDirectory() as tmpdir:
        audit_path = Path(tmpdir) / "audit"
        policy = CommandPolicy(audit_enabled=True, default_mode="execute")
        engine = PolicyEngine(policy, audit_path)
        
        # Configure global policy for test
        configure_policy(policy, audit_path)
        
        executor = CommandExecutor(default_explain_only=False)
        
        # Execute a command
        with patch.object(executor, '_execute_command', return_value={
            "success": True,
            "output": "test output",
            "error": None,
            "exit_code": 0,
            "execution_time": 0.1
        }):
            result = await executor.execute({
                "command": "echo 'audit test'",
                "explain_only": False,
                "user_id": "test_user",
                "block_id": "test_block"
            })
        
        # Check audit log was created
        import time
        audit_file = audit_path / f"audit_{time.strftime('%Y%m')}.jsonl"
        assert audit_file.exists()
        
        # Read and verify audit entry
        with open(audit_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) > 0
            
            entry = json.loads(lines[-1])
            assert entry["command"] == "echo 'audit test'"
            assert entry["user_id"] == "test_user"
            assert entry["block_id"] == "test_block"
            assert entry["executed"] is True
            assert entry["exit_code"] == 0
            assert "hash" in entry  # Verify hash was calculated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
