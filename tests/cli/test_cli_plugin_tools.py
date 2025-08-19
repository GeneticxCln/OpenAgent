import textwrap
from pathlib import Path

import pytest

from openagent import cli as cli_mod


class DummyAgent:
    def __init__(self):
        self._tools = {}
    def add_tool(self, tool):
        name = getattr(tool, "name", tool.__class__.__name__)
        self._tools[name] = tool
    def get_tool(self, name):
        return self._tools.get(name)


def _write_tool_plugin(tmp_plugins: Path, name: str, version: str, tool_name: str, desc: str = ""):
    # Python file
    py = tmp_plugins / f"{name}.py"
    py.write_text(
        textwrap.dedent(
            f'''
            from openagent.plugins.base import BasePlugin, PluginType, plugin_metadata
            from openagent.core.base import BaseTool, ToolResult

            class _T(BaseTool):
                def __init__(self):
                    super().__init__(name="{tool_name}", description="{desc}")
                async def execute(self, input_data):
                    return ToolResult(success=True, content="ok")

            @plugin_metadata(
                name="{name}",
                version="{version}",
                description="d",
                author="t",
                plugin_type=PluginType.TOOL,
            )
            class P(BasePlugin):
                async def initialize(self):
                    self._tools = [_T()]
                    return True
                async def cleanup(self):
                    self._tools = []
                    return True
                def get_tools(self):
                    return list(self._tools)
                async def execute(self, *args, **kwargs):
                    return "ok"
            '''
        )
    )
    # Metadata file
    (tmp_plugins / f"{name}_metadata.json").write_text(
        textwrap.dedent(
            f"""
            {{
              "name": "{name}",
              "version": "{version}",
              "description": "d",
              "author": "t",
              "plugin_type": "tool"
            }}
            """
        )
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("monkeypatch")
def test_cli_plugin_tools_shows_version_and_filter(tmp_path, monkeypatch):
    # Arrange: temp CWD with plugins
    monkeypatch.chdir(tmp_path)
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    _write_tool_plugin(plugins_dir, "plugA", "1.2.3", tool_name="greet", desc="hello")
    _write_tool_plugin(plugins_dir, "plugB", "2.0.0", tool_name="greet", desc="hola")

    # Capture console output
    from rich.console import Console
    test_console = Console(record=True)
    monkeypatch.setattr(cli_mod, "console", test_console, raising=True)

    # Act: list all tools
    cli_mod.plugin_tools(plugin=None)
    out_all = test_console.export_text()

    # Assert: Version column appears and both versions present
    assert "Version" in out_all
    assert "1.2.3" in out_all
    assert "2.0.0" in out_all
    # Assert: plugin names are shown next to tools
    assert "plugA" in out_all and "plugB" in out_all

    # Act: filter by plugin A
    test_console.clear()
    cli_mod.plugin_tools(plugin="plugA")
    out_a = test_console.export_text()

    # Assert: Only plugA entries present
    assert "plugA" in out_a
    assert "plugB" not in out_a
    assert "1.2.3" in out_a


@pytest.mark.usefixtures("monkeypatch")
def test_cli_plugin_sync_tools_filtering(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.chdir(tmp_path)
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    _write_tool_plugin(plugins_dir, "plugA", "1.0.0", tool_name="hello")
    _write_tool_plugin(plugins_dir, "plugB", "1.0.0", tool_name="world")

    # Use dummy agent in the CLI module's global state
    agent = DummyAgent()
    monkeypatch.setattr(cli_mod, "agent", agent, raising=True)

    # Capture console
    from rich.console import Console
    test_console = Console(record=True)
    monkeypatch.setattr(cli_mod, "console", test_console, raising=True)

    # Act: sync only plugA
    cli_mod.plugin_sync_tools(plugin=["plugA"])

    # Assert: only tool from plugA is registered
    assert agent.get_tool("hello") is not None
    assert agent.get_tool("world") is None

    out = test_console.export_text()
    assert "Attached" in out and "plugin tool" in out

