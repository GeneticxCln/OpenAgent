from openagent.plugins.base import BasePlugin, PluginType, plugin_metadata
from openagent.core.base import BaseTool, ToolResult


class GreetTool(BaseTool):
    def __init__(self):
        super().__init__(name="greet", description="Greet a user by name")

    async def execute(self, input_data):
        name = input_data.get("name") if isinstance(input_data, dict) else str(input_data)
        if not name:
            return ToolResult(success=False, error="name is required", content="")
        return ToolResult(success=True, content=f"Hello, {name}!", metadata={"source": "tool_provider"})


@plugin_metadata(
    name="tool_provider",
    version="1.0.0",
    description="Provides a simple 'greet' tool",
    author="OpenAgent",
    plugin_type=PluginType.TOOL,
)
class ToolProviderPlugin(BasePlugin):
    @property
    def version(self):
        return "1.0.0"

    @property
    def description(self):
        return "Tool provider sample plugin"

    async def initialize(self):
        self._tools = [GreetTool()]
        return True

    async def cleanup(self):
        self._tools = []
        return True

    def get_tools(self):
        return list(self._tools)

