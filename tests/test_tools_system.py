import asyncio
import pytest

from openagent.tools.system import CommandExecutor, FileManager


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
