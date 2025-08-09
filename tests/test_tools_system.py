import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

from openagent.tools.system import CommandExecutor, FileManager, SystemInfo
from tests.fixtures.test_data import SAMPLE_COMMANDS, SAMPLE_POLICIES


pytestmark = pytest.mark.asyncio


async def test_command_executor_explain_only_default():
    tool = CommandExecutor()  # default explain-only True
    result = await tool.execute({"command": "ls -la"})
    assert result.success is True
    assert "ls" in result.content.lower()
    assert result.metadata.get("explained") is True


async def test_command_executor_rejects_dangerous():
    tool = CommandExecutor(default_explain_only=False)
    # Still should reject dangerous commands when execution requested
    result = await tool.execute({"command": "rm -rf /", "explain_only": False})
    assert result.success is False
    assert "rejected" in (result.error or "").lower()


async def test_file_manager_write_and_read(tmp_path):
    fm = FileManager()
    file_path = tmp_path / "example.txt"

    write_res = await fm.execute({
        "operation": "write",
        "path": str(file_path),
        "content": "hello"
    })
    assert write_res.success is True

    read_res = await fm.execute({
        "operation": "read",
        "path": str(file_path),
    })
    assert read_res.success is True
    assert read_res.content == "hello"


# Additional comprehensive tests

class TestCommandExecutorAdvanced:
    """Advanced tests for CommandExecutor."""
    
    async def test_safe_commands_execution(self):
        """Test execution of safe commands."""
        tool = CommandExecutor(default_explain_only=False)
        
        for cmd in SAMPLE_COMMANDS["safe"][:3]:  # Test first 3
            result = await tool.execute({"command": cmd, "explain_only": False})
            # Should not be rejected for security reasons
            if not result.success:
                assert "security" not in (result.error or "").lower()
    
    async def test_risky_commands_blocked(self):
        """Test that risky commands are blocked."""
        tool = CommandExecutor(default_explain_only=False)
        
        for cmd in SAMPLE_COMMANDS["risky"][:3]:  # Test first 3
            result = await tool.execute({"command": cmd, "explain_only": False})
            assert result.success is False
            assert any(word in (result.error or "").lower() 
                      for word in ["rejected", "security", "dangerous"])
    
    async def test_command_timeout(self):
        """Test command execution timeout."""
        tool = CommandExecutor(default_explain_only=False)
        
        # Mock long-running command
        with patch.object(tool, '_execute_command') as mock_execute:
            mock_execute.return_value = {
                "success": False,
                "output": "",
                "error": "Command timed out after 30 seconds",
                "exit_code": -1,
                "execution_time": 30.0
            }
            
            result = await tool.execute({"command": "sleep 60", "explain_only": False})
            assert result.success is False
            assert "timeout" in result.error.lower()
    
    async def test_command_explanation_generation(self):
        """Test command explanation generation."""
        tool = CommandExecutor()
        
        result = await tool.execute({"command": "ls -la"})
        assert result.success is True
        assert "ls" in result.content.lower()
        assert "list" in result.content.lower() or "directory" in result.content.lower()
        assert result.metadata.get("explained") is True


class TestFileManagerAdvanced:
    """Advanced tests for FileManager."""
    
    async def test_list_directory(self, tmp_path):
        """Test directory listing."""
        fm = FileManager()
        
        # Create test files
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.py").write_text("print('hello')")
        (tmp_path / "subdir").mkdir()
        
        result = await fm.execute({
            "operation": "list",
            "path": str(tmp_path)
        })
        
        assert result.success is True
        assert "file1.txt" in result.content
        assert "file2.py" in result.content
        assert "subdir" in result.content
        assert len(result.metadata["items"]) == 3
    
    async def test_file_operations_error_handling(self, tmp_path):
        """Test error handling in file operations."""
        fm = FileManager()
        
        # Test reading non-existent file
        result = await fm.execute({
            "operation": "read",
            "path": str(tmp_path / "nonexistent.txt")
        })
        assert result.success is False
        assert "does not exist" in result.error
        
        # Test listing non-existent directory
        result = await fm.execute({
            "operation": "list",
            "path": str(tmp_path / "nonexistent")
        })
        assert result.success is False
        assert "does not exist" in result.error
    
    async def test_file_info_retrieval(self, tmp_path):
        """Test file information retrieval."""
        fm = FileManager()
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        result = await fm.execute({
            "operation": "info",
            "path": str(test_file)
        })
        
        assert result.success is True
        assert "test.txt" in result.content
        assert "size" in result.content.lower()
        assert "type" in result.content.lower()
        assert result.metadata["name"] == "test.txt"
        assert result.metadata["type"] == "file"
    
    async def test_binary_file_handling(self, tmp_path):
        """Test handling of binary files."""
        fm = FileManager()
        
        # Create a "binary" file (with non-UTF8 content simulation)
        binary_file = tmp_path / "binary.dat"
        
        # Mock reading as binary to trigger UnicodeDecodeError
        with patch('pathlib.Path.read_text') as mock_read:
            mock_read.side_effect = UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid start byte')
            
            result = await fm.execute({
                "operation": "read",
                "path": str(binary_file)
            })
            
            assert result.success is False
            assert "binary" in result.error.lower()


class TestSystemInfo:
    """Tests for SystemInfo tool."""
    
    async def test_system_overview(self):
        """Test system overview retrieval."""
        tool = SystemInfo()
        result = await tool.execute("overview")
        
        assert result.success is True
        assert "system" in result.content.lower() or "operating" in result.content.lower()
        assert "cpu" in result.content.lower()
        assert "memory" in result.content.lower()
    
    async def test_specific_info_types(self):
        """Test specific system information types."""
        tool = SystemInfo()
        
        info_types = ["cpu", "memory", "disk", "processes"]
        
        for info_type in info_types:
            result = await tool.execute(info_type)
            assert result.success is True
            assert len(result.content) > 0
            assert info_type.lower() in result.content.lower()
    
    async def test_invalid_info_type(self):
        """Test handling of invalid info types."""
        tool = SystemInfo()
        result = await tool.execute("invalid_type")
        
        assert result.success is True  # Should still succeed
        assert "available info types" in result.content.lower()
    
    @patch('psutil.process_iter')
    async def test_process_info_error_handling(self, mock_process_iter):
        """Test process information with access errors."""
        # Mock process that raises AccessDenied
        mock_proc = Mock()
        mock_proc.info = {"pid": 1234, "name": "test", "cpu_percent": 5.0, "memory_percent": 2.0}
        mock_proc.configure_mock(**{"info.side_effect": [mock_proc.info, 
                                                        pytest.importorskip('psutil').AccessDenied()]})
        
        mock_process_iter.return_value = [mock_proc, mock_proc]
        
        tool = SystemInfo()
        result = await tool.execute("processes")
        
        # Should handle errors gracefully
        assert result.success is True
        assert "processes" in result.content.lower()
