"""
Marketplace stubs for OpenAgent plugins.

This module provides a minimal placeholder implementation to satisfy imports
without requiring a full marketplace backend. It can be extended later.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PluginMarketplace:
    """Plugin marketplace for discovering and installing plugins (stub)."""

    def __init__(self, base_url: str | None = None):
        self.marketplace_url = base_url or "https://plugins.openagent.dev"
        logger.info("PluginMarketplace initialized (stub)")

    async def search_plugins(self, query: str) -> List[Dict[str, Any]]:
        """Search for plugins in the marketplace (stub returns empty list)."""
        logger.info(f"Searching marketplace for: {query}")
        return []

    async def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed plugin information from marketplace (stub returns None)."""
        logger.info(f"Getting plugin info for: {plugin_name}")
        return None

    async def download_plugin(self, plugin_name: str, version: str | None = None) -> bool:
        """Download a plugin from the marketplace (stub returns False)."""
        logger.info(f"Downloading plugin: {plugin_name} v{version}")
        return False

    async def publish_plugin(self, plugin_path: str, metadata: Dict[str, Any]) -> bool:
        """Publish a plugin to the marketplace (stub returns False)."""
        logger.info(f"Publishing plugin from: {plugin_path}")
        return False

