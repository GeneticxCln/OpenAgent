from datetime import datetime, timezone
from ..base import BasePlugin, PluginType, plugin_metadata


@plugin_metadata(
    name="time",
    version="1.0.0",
    description="Returns the current UTC time",
    author="OpenAgent",
    plugin_type=PluginType.CUSTOM,
)
class TimePlugin(BasePlugin):
    @property
    def version(self):
        return "1.0.0"

    @property
    def description(self):
        return "Time plugin for demonstration"

    async def initialize(self):
        return True

    async def cleanup(self):
        return True

    async def execute(self) -> str:
        return datetime.now(timezone.utc).isoformat()

