from openagent.plugins.base import BasePlugin, PluginType, plugin_metadata

@plugin_metadata(
    name="echo",
    version="1.0.0",
    description="Echoes back input text with optional prefix",
    author="OpenAgent",
    plugin_type=PluginType.CUSTOM,
)
class EchoPlugin(BasePlugin):
    @property
    def version(self):
        return "1.0.0"

    @property
    def description(self):
        return "Top-level echo plugin"

    async def initialize(self):
        # Could perform setup here
        return True

    async def cleanup(self):
        return True

    async def execute(self, text: str, **kwargs) -> str:
        prefix = self.config.get("prefix", "")
        return f"{prefix}{text}"

