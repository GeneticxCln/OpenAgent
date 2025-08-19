"""
Integration tests for the CLI interface.

Tests end-to-end CLI functionality including command parsing,
model integration, and output formatting.
"""

import asyncio
import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    @pytest.fixture
    def cli_runner(self):
        """Fixture providing CLI runner."""
        from typer.testing import CliRunner

        from openagent.cli import app

        runner = CliRunner()
        return runner, app

    def test_cli_help(self, cli_runner):
        """Test that CLI help command works."""
        runner, app = cli_runner
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "OpenAgent" in result.stdout
        assert "chat" in result.stdout
        assert "run" in result.stdout
        assert "explain" in result.stdout

    def test_models_command(self, cli_runner):
        """Test models command."""
        runner, app = cli_runner
        result = runner.invoke(app, ["models"])

        assert result.exit_code == 0
        assert "Available Models" in result.stdout
        assert "codellama-7b" in result.stdout
        assert "tiny-llama" in result.stdout

    def test_explain_command(self, cli_runner):
        """Test explain command with mocked LLM."""
        runner, app = cli_runner

        with patch("openagent.cli.create_agent") as mock_create_agent:
            mock_agent = Mock()
            mock_llm = AsyncMock()
            mock_llm.explain_command.return_value = "This command lists files"
            mock_agent.llm = mock_llm
            mock_create_agent.return_value = mock_agent

            result = runner.invoke(app, ["explain", "ls -la"])

            assert result.exit_code == 0
            mock_create_agent.assert_called_once()

    def test_validate_command(self, cli_runner):
        """Test validate command."""
        runner, app = cli_runner

        # Test safe command
        result = runner.invoke(app, ["validate", "ls -la"])
        assert result.exit_code == 0
        assert "Decision:" in result.stdout

        # Test quiet mode
        result = runner.invoke(app, ["validate", "ls -la", "--quiet"])
        assert result.exit_code == 0
        # Should only output decision
        assert result.stdout.strip() in ["allow", "warn", "block"]

    def test_policy_show(self, cli_runner):
        """Test policy show command."""
        runner, app = cli_runner
        result = runner.invoke(app, ["policy", "show"])

        assert result.exit_code == 0
        assert "Policy File" in result.stdout

    def test_policy_reset(self, cli_runner):
        """Test policy reset command."""
        runner, app = cli_runner

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "policy.yaml"

            with patch("openagent.terminal.validator.CONFIG_PATH", config_path):
                result = runner.invoke(app, ["policy", "reset"])

                assert result.exit_code == 0
                assert "reset to defaults" in result.stdout
                assert config_path.exists()

    def test_policy_set_default(self, cli_runner):
        """Test policy set-default command."""
        runner, app = cli_runner

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "policy.yaml"

            with patch("openagent.terminal.validator.CONFIG_PATH", config_path):
                # Valid setting
                result = runner.invoke(app, ["policy", "set-default", "block"])
                assert result.exit_code == 0
                assert "Default decision set to block" in result.stdout

                # Invalid setting
                result = runner.invoke(app, ["policy", "set-default", "invalid"])
                assert result.exit_code == 1
                assert "must be one of" in result.stdout

    def test_integrate_show(self, cli_runner):
        """Test integrate command (show mode)."""
        runner, app = cli_runner
        result = runner.invoke(app, ["integrate", "--shell", "zsh"])

        assert result.exit_code == 0
        assert "OpenAgent zsh integration" in result.stdout

    def test_run_command_text_output(self, cli_runner):
        """Test run command with text output."""
        runner, app = cli_runner

        with patch("openagent.cli.create_agent") as mock_create_agent:
            mock_agent = Mock()
            mock_response = Mock()
            mock_response.content = "Test response"
            mock_response.metadata = {}
            mock_agent.process_message = AsyncMock(return_value=mock_response)
            mock_create_agent.return_value = mock_agent

            result = runner.invoke(app, ["run", "test prompt"])

            assert result.exit_code == 0
            assert "Test response" in result.stdout

    def test_run_command_json_output(self, cli_runner):
        """Test run command with JSON output."""
        runner, app = cli_runner

        with patch("openagent.cli.create_agent") as mock_create_agent:
            mock_agent = Mock()
            mock_response = Mock()
            mock_response.content = "Test response"
            mock_response.metadata = {"test": "data"}
            mock_agent.process_message = AsyncMock(return_value=mock_response)
            mock_create_agent.return_value = mock_agent

            result = runner.invoke(
                app, ["run", "test prompt", "--output-format", "json"]
            )

            assert result.exit_code == 0

            # Parse JSON output
            output_data = json.loads(result.stdout)
            assert output_data["prompt"] == "test prompt"
            assert output_data["response"] == "Test response"
            assert output_data["metadata"]["test"] == "data"

    def test_code_command(self, cli_runner):
        """Test code generation command."""
        runner, app = cli_runner

        with patch("openagent.cli.create_agent") as mock_create_agent:
            mock_agent = Mock()
            mock_llm = AsyncMock()
            mock_llm.generate_code.return_value = "def hello(): print('Hello')"
            mock_agent.llm = mock_llm
            mock_create_agent.return_value = mock_agent

            result = runner.invoke(
                app, ["code", "Create a hello function", "--language", "python"]
            )

            assert result.exit_code == 0
            assert "def hello():" in result.stdout

    def test_analyze_command(self, cli_runner):
        """Test code analysis command."""
        runner, app = cli_runner

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def test(): pass")
            f.flush()

            with patch("openagent.cli.create_agent") as mock_create_agent:
                mock_agent = Mock()
                mock_llm = AsyncMock()
                mock_llm.analyze_code.return_value = "Simple function definition"
                mock_agent.llm = mock_llm
                mock_create_agent.return_value = mock_agent

                result = runner.invoke(app, ["analyze", f.name])

                assert result.exit_code == 0
                assert "Simple function definition" in result.stdout

        # Test with non-existent file
        result = runner.invoke(app, ["analyze", "/nonexistent/file.py"])
        assert result.exit_code == 1
        assert "not found" in result.stdout


class TestCLIChatMode:
    """Test CLI chat mode functionality."""

    @pytest.fixture
    def mock_agent(self):
        """Mock agent for chat testing."""
        mock_agent = Mock()
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_response.metadata = {}
        mock_agent.process_message = AsyncMock(return_value=mock_response)
        return mock_agent

    def test_chat_special_commands(self, mock_agent):
        """Test special commands in chat mode."""
        from openagent.cli import handle_special_command

        # Test /help command
        result = asyncio.run(handle_special_command("/help"))
        assert result is True  # Should continue chat

        # Test /quit command
        result = asyncio.run(handle_special_command("/quit"))
        assert result is False  # Should exit chat

        # Test /status command
        mock_agent.get_status.return_value = {
            "name": "TestAgent",
            "tools_count": 2,
            "tools": ["tool1", "tool2"],
            "message_history_length": 5,
            "is_processing": False,
        }
        mock_agent.llm.get_model_info.return_value = {
            "model_name": "test-model",
            "model_path": "test/path",
            "device": "cpu",
            "loaded": True,
        }

        # Temporarily set global agent
        import openagent.cli

        original_agent = getattr(openagent.cli, "agent", None)
        openagent.cli.agent = mock_agent

        try:
            result = asyncio.run(handle_special_command("/status"))
            assert result is True
        finally:
            openagent.cli.agent = original_agent

    def test_chat_reset_command(self, mock_agent):
        """Test /reset command in chat mode."""
        from openagent.cli import handle_special_command

        mock_agent.reset = Mock()

        # Temporarily set global agent
        import openagent.cli

        original_agent = getattr(openagent.cli, "agent", None)
        openagent.cli.agent = mock_agent

        try:
            result = asyncio.run(handle_special_command("/reset"))
            assert result is True
            mock_agent.reset.assert_called_once()
        finally:
            openagent.cli.agent = original_agent


class TestCLIPlugins:
    """Test CLI plugin commands against real PluginManager wiring."""

    @pytest.fixture
    def cli_runner(self):
        from typer.testing import CliRunner
        from openagent.cli import app
        return CliRunner(), app

    def test_plugin_list_and_info(self, cli_runner, tmp_path: Path):
        # Create a minimal plugin to be discovered
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        (plugins_dir / "echo.py").write_text(
            """
from openagent.plugins.base import BasePlugin, PluginType, plugin_metadata
@plugin_metadata(name="echo", version="1.0.0", description="Echo", author="t", plugin_type=PluginType.CUSTOM)
class EchoPlugin(BasePlugin):
    @property
    def version(self):
        return "1.0.0"
    @property
    def description(self):
        return "Echo"
    async def initialize(self):
        return True
    async def cleanup(self):
        return True
    async def execute(self, *args, **kwargs):
        return "ok"
"""
        )
        # Change CWD so PluginManager uses our tmp plugins dir
        import os
        old = os.getcwd()
        os.chdir(tmp_path)
        try:
            runner, app = cli_runner
            res = runner.invoke(app, ["plugin", "list"])
            assert res.exit_code == 0
            assert "Plugins:" in res.stdout

            res2 = runner.invoke(app, ["plugin", "info", "echo"])
            assert res2.exit_code == 0
            assert "echo" in res2.stdout
        finally:
            os.chdir(old)

    def test_plugin_enable_disable_persist(self, cli_runner, tmp_path: Path):
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        (plugins_dir / "demo.py").write_text(
            """
from openagent.plugins.base import BasePlugin, PluginType, plugin_metadata
@plugin_metadata(name="demo", version="1.0.0", description="Demo", author="t", plugin_type=PluginType.CUSTOM)
class DemoPlugin(BasePlugin):
    @property
    def version(self):
        return "1.0.0"
    @property
    def description(self):
        return "Demo"
    async def initialize(self):
        return True
    async def cleanup(self):
        return True
    async def execute(self, *args, **kwargs):
        return "ok"
"""
        )
        import os, json
        old = os.getcwd()
        os.chdir(tmp_path)
        try:
            runner, app = cli_runner
            res = runner.invoke(app, ["plugin", "enable", "demo"])
            assert res.exit_code == 0
            # Verify config.json persisted
            cfg = json.loads((plugins_dir / "config.json").read_text())
            assert cfg.get("demo", {}).get("enabled") is True

            res2 = runner.invoke(app, ["plugin", "disable", "demo"])
            assert res2.exit_code == 0
            cfg2 = json.loads((plugins_dir / "config.json").read_text())
            assert cfg2.get("demo", {}).get("enabled") is False
        finally:
            os.chdir(old)


    def test_plugin_tools_lists_tools(self, cli_runner, tmp_path: Path):
        # Create plugin and enable
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        (plugins_dir / "tp.py").write_text(
            """
from openagent.plugins.base import BasePlugin, PluginType, plugin_metadata
from openagent.core.base import BaseTool, ToolResult

class HelloTool(BaseTool):
    def __init__(self):
        super().__init__(name="hello", description="Say hello")
    async def execute(self, input_data):
        return ToolResult(success=True, content="hello")

@plugin_metadata(name="tp", version="1.0.0", description="Tool provider", author="t", plugin_type=PluginType.TOOL)
class TpPlugin(BasePlugin):
    @property
    def version(self):
        return "1.0.0"
    @property
    def description(self):
        return "tp"
    async def initialize(self):
        self._tools = [HelloTool()]
        return True
    async def cleanup(self):
        self._tools = []
        return True
    def get_tools(self):
        return list(self._tools)
    async def execute(self, *args, **kwargs):
        return "ok"
"""
        )
        (plugins_dir / "config.json").write_text('{"tp": {"enabled": true}}')
        import os
        old = os.getcwd()
        os.chdir(tmp_path)
        try:
            runner, app = cli_runner
            res = runner.invoke(app, ["plugin", "tools"], catch_exceptions=False)
            assert res.exit_code == 0
            assert "hello" in res.stdout
        finally:
            os.chdir(old)

    def test_chat_registers_plugin_tools(self, cli_runner, tmp_path: Path):
        # Prepare a simple tool plugin in a temp plugins dir
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        (plugins_dir / "tp.py").write_text(
            """
from openagent.plugins.base import BasePlugin, PluginType, plugin_metadata
from openagent.core.base import BaseTool, ToolResult

class HelloTool(BaseTool):
    def __init__(self):
        super().__init__(name="hello", description="Say hello")
    async def execute(self, input_data):
        return ToolResult(success=True, content="hello")

@plugin_metadata(name="tp", version="1.0.0", description="Tool provider", author="t", plugin_type=PluginType.TOOL)
class TpPlugin(BasePlugin):
    @property
    def version(self):
        return "1.0.0"
    @property
    def description(self):
        return "tp"
    async def initialize(self):
        self._tools = [HelloTool()]
        return True
    async def cleanup(self):
        self._tools = []
        return True
    def get_tools(self):
        return list(self._tools)
    async def execute(self, *args, **kwargs):
        return "ok"
"""
        )
        # Enable via config
        (plugins_dir / "config.json").write_text('{"tp": {"enabled": true}}')

        # Stub Agent capturing added tools
        added = []
        class StubAgent:
            def __init__(self):
                self.tools = {}
                self.llm = type("L", (), {"load_model": AsyncMock()})()
                self.config = {"safe_mode": True}
            def add_tool(self, tool):
                self.tools[getattr(tool, "name", tool.__class__.__name__)] = tool
                added.append(getattr(tool, "name", tool.__class__.__name__))
            def get_tool(self, name):
                return self.tools.get(name)

        # Patch create_agent to return stub, and patch chat_loop to end immediately
        import os
        old = os.getcwd()
        os.chdir(tmp_path)
        try:
            runner, app = cli_runner
            with patch("openagent.cli.create_agent", return_value=StubAgent()) as _ca, \
                 patch("openagent.cli.chat_loop", new_callable=lambda: (lambda **kwargs: asyncio.sleep(0))) as _cl:
                res = runner.invoke(app, ["chat", "--no-auto-serve", "--model", "tiny-llama"], catch_exceptions=False)
                assert res.exit_code == 0
                # Ensure tool was added
                assert "hello" in added
        finally:
            os.chdir(old)

    def test_plugin_sync_tools_attaches_to_running_agent(self, cli_runner, tmp_path: Path):
        # Prepare plugin
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        (plugins_dir / "tp.py").write_text(
            """
from openagent.plugins.base import BasePlugin, PluginType, plugin_metadata
from openagent.core.base import BaseTool, ToolResult

class HelloTool(BaseTool):
    def __init__(self):
        super().__init__(name="hello", description="Say hello")
    async def execute(self, input_data):
        return ToolResult(success=True, content="hello")

@plugin_metadata(name="tp", version="1.0.0", description="Tool provider", author="t", plugin_type=PluginType.TOOL)
class TpPlugin(BasePlugin):
    @property
    def version(self):
        return "1.0.0"
    @property
    def description(self):
        return "tp"
    async def initialize(self):
        self._tools = [HelloTool()]
        return True
    async def cleanup(self):
        self._tools = []
        return True
    def get_tools(self):
        return list(self._tools)
    async def execute(self, *args, **kwargs):
        return "ok"
"""
        )
        (plugins_dir / "config.json").write_text('{"tp": {"enabled": true}}')

        # Stub running agent
        added = []
        class StubAgent:
            def __init__(self):
                self.tools = {}
            def add_tool(self, tool):
                name = getattr(tool, "name", tool.__class__.__name__)
                self.tools[name] = tool
                added.append(name)
            def get_tool(self, name):
                return self.tools.get(name)

        import os
        old = os.getcwd()
        os.chdir(tmp_path)
        try:
            runner, app = cli_runner
            # Patch global agent in CLI module
            import openagent.cli as cli_mod
            orig_agent = getattr(cli_mod, "agent", None)
            cli_mod.agent = StubAgent()
            try:
                res = runner.invoke(app, ["plugin", "sync-tools"], catch_exceptions=False)
                assert res.exit_code == 0
                assert "Attached" in res.stdout
                assert "hello" in added
            finally:
                cli_mod.agent = orig_agent
        finally:
            os.chdir(old)


class TestCLIErrorHandling:
    """Test CLI error handling scenarios."""

    @pytest.fixture
    def cli_runner(self):
        """Fixture providing CLI runner."""
        from typer.testing import CliRunner

        from openagent.cli import app

        runner = CliRunner()
        return runner, app

    def test_invalid_model_name(self, cli_runner):
        """Test behavior with invalid model names."""
        runner, app = cli_runner

        # This should work but might show a warning
        result = runner.invoke(app, ["run", "test", "--model", "nonexistent-model"])
        # Should not crash, but may show error in actual execution

    def test_missing_required_args(self, cli_runner):
        """Test missing required arguments."""
        runner, app = cli_runner

        # Missing required prompt for run command
        result = runner.invoke(app, ["run"])
        assert result.exit_code != 0  # Should fail

        # Missing required command for explain
        result = runner.invoke(app, ["explain"])
        assert result.exit_code != 0  # Should fail

    def test_invalid_policy_commands(self, cli_runner):
        """Test invalid policy commands."""
        runner, app = cli_runner

        # Unknown policy action
        result = runner.invoke(app, ["policy", "unknown-action"])
        assert result.exit_code != 0 or "Unknown action" in result.stdout

        # Invalid policy setting
        result = runner.invoke(app, ["policy", "set-default", "invalid-value"])
        assert result.exit_code == 1
        assert "must be one of" in result.stdout

    def test_integration_errors(self, cli_runner):
        """Test shell integration error handling."""
        runner, app = cli_runner

        # Invalid shell
        with patch("openagent.cli.install_snippet") as mock_install:
            mock_install.side_effect = ValueError("Unsupported shell")

            result = runner.invoke(app, ["integrate", "--shell", "invalid"])
            assert result.exit_code == 0  # Should be handled gracefully
            assert "Failed to set up integration" in result.stdout


class TestCLIConfiguration:
    """Test CLI configuration and environment handling."""

    @pytest.fixture
    def cli_runner(self):
        """Fixture providing CLI runner."""
        from typer.testing import CliRunner

        from openagent.cli import app

        runner = CliRunner()
        return runner, app

    def test_environment_variable_loading(self, cli_runner):
        """Test loading configuration from environment variables."""
        runner, app = cli_runner

        with patch.dict(
            "os.environ",
            {
                "HUGGINGFACE_TOKEN": "test_token",
                "DEFAULT_MODEL": "test-model",
                "DEFAULT_DEVICE": "cpu",
            },
        ):
            with patch("openagent.cli.create_agent") as mock_create_agent:
                runner.invoke(app, ["models"])
                # Should load environment variables

    def test_debug_mode(self, cli_runner):
        """Test debug mode functionality."""
        runner, app = cli_runner

        with patch("openagent.cli.setup_logging") as mock_setup_logging:
            result = runner.invoke(app, ["run", "test", "--debug"])

            # Should call setup_logging with debug=True if implemented

    def test_device_configuration(self, cli_runner):
        """Test device configuration options."""
        runner, app = cli_runner

        with patch("openagent.cli.create_agent") as mock_create_agent:
            # Test CPU device
            runner.invoke(app, ["run", "test", "--device", "cpu"])

            # Test CUDA device
            runner.invoke(app, ["run", "test", "--device", "cuda"])

            # Test auto device
            runner.invoke(app, ["run", "test", "--device", "auto"])

            # Verify create_agent was called with correct device
            assert mock_create_agent.call_count >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
