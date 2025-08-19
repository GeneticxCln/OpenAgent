from typing import Any, Dict, Optional
from datetime import datetime, timezone

from ..base import BasePlugin, PluginType, plugin_metadata, plugin_config_schema


@plugin_metadata(
    name="echo",
    version="1.0.0",
    description="Echo back the provided input",
    author="OpenAgent",
    plugin_type=PluginType.CUSTOM,
)
@plugin_config_schema({
    "type": "object",
    "properties": {
        "enabled": {"type": "boolean"},
        "prefix": {"type": "string"}
    },
    "additionalProperties": True
})
class EchoPlugin(BasePlugin):
    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Echo plugin for demonstration"

    async def initialize(self):
        self.initialized_at = datetime.now(timezone.utc)
        return True

    async def cleanup(self):
        return True

    async def execute(self, text: str, **kwargs) -> str:
        prefix = self.config.get("prefix", "")
        return f"{prefix}{text}"

