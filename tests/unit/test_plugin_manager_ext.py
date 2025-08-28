import json
import textwrap
from pathlib import Path

import pytest

from openagent.plugins.manager import PluginManager


@pytest.mark.asyncio
async def test_yaml_config_and_schema_validation(tmp_path: Path):
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()

    # Plugin source
    plugin_py = plugins_dir / "cfgplug.py"
    plugin_py.write_text(
        textwrap.dedent(
            """
            from openagent.plugins.base import BasePlugin, PluginType, plugin_metadata

            @plugin_metadata(
                name="cfgplug",
                version="1.0.0",
                description="Config test plugin",
                author="tester",
                plugin_type=PluginType.CUSTOM,
            )
            class CfgplugPlugin(BasePlugin):
                @property
                def version(self):
                    return "1.0.0"
                @property
                def description(self):
                    return "cfg"
                async def initialize(self):
                    return True
                async def cleanup(self):
                    return True
                async def execute(self, *args, **kwargs):
                    return "ok"
            """
        )
    )

    # Metadata JSON including config_schema
    meta = {
        "name": "cfgplug",
        "version": "1.0.0",
        "description": "Config test plugin",
        "author": "tester",
        "plugin_type": "custom",
        "config_schema": {
            "type": "object",
            "required": ["enabled", "config"],
            "properties": {
                "enabled": {"type": "boolean"},
                "config": {
                    "type": "object",
                    "required": ["mode"],
                    "properties": {
                        "mode": {"type": "string", "enum": ["a", "b"]},
                        "params": {
                            "type": "object",
                            "properties": {"threshold": {"type": "number"}},
                        },
                    },
                },
            },
        },
    }
    (plugins_dir / "cfgplug_metadata.json").write_text(json.dumps(meta))

    # Central YAML config
    (plugins_dir / "config.yaml").write_text(
        textwrap.dedent(
            """
            cfgplug:
              enabled: true
              config:
                mode: a
                params:
                  threshold: 0.7
            """
        )
    )

    pm = PluginManager(plugin_dir=plugins_dir)
    await pm.initialize()
    discovered = await pm.discover_plugins()
    assert "cfgplug" in discovered
    ok = await pm.load_plugin("cfgplug")
    assert ok is True


@pytest.mark.asyncio
async def test_permission_gating_execute_commands(tmp_path: Path):
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()

    plugin_py = plugins_dir / "psec.py"
    plugin_py.write_text(
        textwrap.dedent(
            """
            from openagent.plugins.base import BasePlugin, PluginType, plugin_metadata

            @plugin_metadata(
                name="psec",
                version="1.0.0",
                description="Perm test",
                author="tester",
                plugin_type=PluginType.CUSTOM,
                permissions=["execute_commands"]
            )
            class PsecPlugin(BasePlugin):
                @property
                def version(self):
                    return "1.0.0"
                @property
                def description(self):
                    return "p"
                async def initialize(self):
                    return True
                async def cleanup(self):
                    return True
                async def execute(self, *args, **kwargs):
                    return "ok"
            """
        )
    )
    (plugins_dir / "psec_metadata.json").write_text(
        json.dumps(
            {
                "name": "psec",
                "version": "1.0.0",
                "description": "Perm test",
                "author": "tester",
                "plugin_type": "custom",
                "permissions": ["execute_commands"],
            }
        )
    )

    # Default manager config should deny execute_commands
    pm = PluginManager(plugin_dir=plugins_dir)
    await pm.initialize()
    await pm.discover_plugins()
    assert await pm.load_plugin("psec")
    ok = await pm.enable_plugin("psec")
    assert ok is False

    # Allow via manager config
    pm2 = PluginManager(plugin_dir=plugins_dir, config={"allow_execute_commands": True})
    await pm2.initialize()
    await pm2.discover_plugins()
    assert await pm2.load_plugin("psec")
    ok2 = await pm2.enable_plugin("psec")
    assert ok2 is True


@pytest.mark.asyncio
async def test_tool_auto_registration(tmp_path: Path):
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()

    plugin_py = plugins_dir / "tp.py"
    plugin_py.write_text(
        textwrap.dedent(
            """
            from openagent.plugins.base import BasePlugin, PluginType, plugin_metadata
            from openagent.core.base import BaseTool, ToolResult

            class HelloTool(BaseTool):
                def __init__(self):
                    super().__init__(name="hello", description="Say hello")
                async def execute(self, input_data):
                    return ToolResult(success=True, content="hello")

            @plugin_metadata(
                name="tp",
                version="1.0.0",
                description="Tool provider",
                author="tester",
                plugin_type=PluginType.TOOL,
            )
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
    )
    (plugins_dir / "tp_metadata.json").write_text(
        json.dumps(
            {
                "name": "tp",
                "version": "1.0.0",
                "description": "Tool provider",
                "author": "tester",
                "plugin_type": "tool",
            }
        )
    )

    pm = PluginManager(plugin_dir=plugins_dir)
    await pm.initialize()
    await pm.discover_plugins()
    assert await pm.load_plugin("tp")
    assert await pm.enable_plugin("tp")
    catalog = pm.get_tool_catalog()
    assert "hello" in catalog
