"""
Plugin Manager for OpenAgent

Handles the complete lifecycle of plugins including loading, validation,
execution, monitoring, and cleanup with proper security and error handling.
"""

import asyncio
import importlib
import importlib.util
import json
import logging
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type

from .base import (
    AgentPlugin,
    IntegrationPlugin,
    PluginBase,
    PluginMetadata,
    PluginStatus,
    PluginType,
    ToolPlugin,
)
from .loader import PluginLoader
from .registry import PluginRegistry
from .validator import PluginValidator

logger = logging.getLogger(__name__)


class PluginManager:
    """
    Central manager for all plugin operations in OpenAgent.

    Provides a unified interface for loading, managing, and executing plugins
    with proper security, monitoring, and error handling.
    """

    def __init__(
        self,
        plugin_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
        max_workers: int = 4,
    ):
        """Initialize the plugin manager."""
        self.plugin_dir = plugin_dir or Path("plugins")
        self.config = config or {}
        self.max_workers = max_workers

        # Core components
        self.registry = PluginRegistry()
        self.loader = PluginLoader(self.plugin_dir)
        self.validator = PluginValidator()

        # Plugin storage
        self._plugins: Dict[str, PluginBase] = {}
        self._plugin_classes: Dict[str, Type[PluginBase]] = {}
        self._plugin_configs: Dict[str, Dict[str, Any]] = {}

        # Event system
        self._event_handlers: Dict[str, List[Callable]] = {}

        # Thread pool for plugin operations
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Manager state
        self._is_initialized = False
        self._startup_time = datetime.now(timezone.utc)

        logger.info(
            f"PluginManager initialized with plugin directory: {self.plugin_dir}"
        )

    async def initialize(self) -> None:
        """Initialize the plugin manager and discover plugins."""
        if self._is_initialized:
            logger.warning("PluginManager already initialized")
            return

        try:
            logger.info("Initializing plugin manager...")

            # Ensure plugin directory exists
            self.plugin_dir.mkdir(parents=True, exist_ok=True)

            # Initialize registry
            await self.registry.initialize()

            # Discover and load plugins
            await self.discover_plugins()

            # Load configuration
            await self._load_plugin_configs()

            self._is_initialized = True
            logger.info("PluginManager initialized successfully")

            # Emit initialization event
            await self._emit_event("manager_initialized")

        except Exception as e:
            logger.error(f"Failed to initialize PluginManager: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the plugin manager and cleanup resources."""
        logger.info("Shutting down plugin manager...")

        try:
            # Emit shutdown event
            await self._emit_event("manager_shutting_down")

            # Unload all plugins
            await self.unload_all_plugins()

            # Shutdown executor
            self._executor.shutdown(wait=True)

            # Cleanup registry
            await self.registry.cleanup()

            self._is_initialized = False
            logger.info("PluginManager shutdown complete")

        except Exception as e:
            logger.error(f"Error during PluginManager shutdown: {e}")
            raise

    async def discover_plugins(self) -> List[str]:
        """Discover all available plugins in the plugin directory."""
        logger.info(f"Discovering plugins in {self.plugin_dir}")

        discovered_plugins = []

        try:
            # Scan for plugin files
            plugin_files = await self.loader.discover_plugin_files()

            for plugin_file in plugin_files:
                try:
                    # Load plugin metadata
                    metadata = await self.loader.load_plugin_metadata(plugin_file)

                    if metadata:
                        # Validate plugin
                        validation_result = await self.validator.validate_metadata(
                            metadata
                        )

                        if validation_result.is_valid:
                            # Register plugin
                            await self.registry.register_plugin(metadata)
                            discovered_plugins.append(metadata.name)

                            logger.info(
                                f"Discovered plugin: {metadata.name} v{metadata.version}"
                            )
                        else:
                            logger.warning(
                                f"Plugin validation failed for {plugin_file}: {validation_result.errors}"
                            )

                except Exception as e:
                    logger.error(f"Error discovering plugin {plugin_file}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error during plugin discovery: {e}")
            raise

        logger.info(f"Discovered {len(discovered_plugins)} valid plugins")
        return discovered_plugins

    async def load_plugin(
        self, plugin_name: str, config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Load a specific plugin by name."""
        if plugin_name in self._plugins:
            logger.warning(f"Plugin {plugin_name} is already loaded")
            return True

        try:
            logger.info(f"Loading plugin: {plugin_name}")

            # Get plugin metadata from registry
            metadata = await self.registry.get_plugin(plugin_name)
            if not metadata:
                logger.error(f"Plugin {plugin_name} not found in registry")
                return False

            # Load plugin class
            plugin_class = await self.loader.load_plugin_class(metadata)
            if not plugin_class:
                logger.error(f"Failed to load plugin class for {plugin_name}")
                return False

            # Validate plugin class
            validation_result = await self.validator.validate_plugin_class(plugin_class)
            if not validation_result.is_valid:
                logger.error(
                    f"Plugin class validation failed for {plugin_name}: {validation_result.errors}"
                )
                return False

            # Create plugin instance
            plugin_config = config or self._plugin_configs.get(plugin_name, {})
            plugin_instance = plugin_class(plugin_config)

            # Set plugin status
            plugin_instance._set_status(PluginStatus.LOADING)

            # Initialize plugin
            if not await plugin_instance.initialize():
                logger.error(f"Plugin initialization failed for {plugin_name}")
                plugin_instance._set_status(
                    PluginStatus.ERROR, Exception("Initialization failed")
                )
                return False

            # Store plugin
            self._plugins[plugin_name] = plugin_instance
            self._plugin_classes[plugin_name] = plugin_class

            # Set status to loaded
            plugin_instance._set_status(PluginStatus.LOADED)

            # Call on_load callback
            await plugin_instance.on_load()

            # Enable plugin if configured
            if plugin_config.get("enabled", True):
                await self.enable_plugin(plugin_name)

            logger.info(f"Plugin {plugin_name} loaded successfully")

            # Emit load event
            await self._emit_event("plugin_loaded", plugin_name=plugin_name)

            return True

        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}")
            logger.debug(traceback.format_exc())

            # Update plugin status if instance exists
            if plugin_name in self._plugins:
                self._plugins[plugin_name]._set_status(PluginStatus.ERROR, e)

            return False

    async def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a specific plugin."""
        if plugin_name not in self._plugins:
            logger.warning(f"Plugin {plugin_name} is not loaded")
            return True

        try:
            logger.info(f"Unloading plugin: {plugin_name}")

            plugin = self._plugins[plugin_name]

            # Disable plugin if active
            if plugin.is_active:
                await self.disable_plugin(plugin_name)

            # Call on_unload callback
            await plugin.on_unload()

            # Cleanup plugin
            if not await plugin.cleanup():
                logger.warning(f"Plugin cleanup failed for {plugin_name}")

            # Remove plugin
            del self._plugins[plugin_name]
            if plugin_name in self._plugin_classes:
                del self._plugin_classes[plugin_name]

            logger.info(f"Plugin {plugin_name} unloaded successfully")

            # Emit unload event
            await self._emit_event("plugin_unloaded", plugin_name=plugin_name)

            return True

        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False

    async def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a loaded plugin."""
        if plugin_name not in self._plugins:
            logger.error(f"Plugin {plugin_name} is not loaded")
            return False

        plugin = self._plugins[plugin_name]

        if plugin.is_active:
            logger.warning(f"Plugin {plugin_name} is already active")
            return True

        try:
            logger.info(f"Enabling plugin: {plugin_name}")

            # Call on_enable callback
            await plugin.on_enable()

            # Set status to active
            plugin._set_status(PluginStatus.ACTIVE)

            logger.info(f"Plugin {plugin_name} enabled successfully")

            # Emit enable event
            await self._emit_event("plugin_enabled", plugin_name=plugin_name)

            return True

        except Exception as e:
            logger.error(f"Error enabling plugin {plugin_name}: {e}")
            plugin._set_status(PluginStatus.ERROR, e)
            return False

    async def disable_plugin(self, plugin_name: str) -> bool:
        """Disable an active plugin."""
        if plugin_name not in self._plugins:
            logger.error(f"Plugin {plugin_name} is not loaded")
            return False

        plugin = self._plugins[plugin_name]

        if not plugin.is_active:
            logger.warning(f"Plugin {plugin_name} is not active")
            return True

        try:
            logger.info(f"Disabling plugin: {plugin_name}")

            # Call on_disable callback
            await plugin.on_disable()

            # Set status to inactive
            plugin._set_status(PluginStatus.INACTIVE)

            logger.info(f"Plugin {plugin_name} disabled successfully")

            # Emit disable event
            await self._emit_event("plugin_disabled", plugin_name=plugin_name)

            return True

        except Exception as e:
            logger.error(f"Error disabling plugin {plugin_name}: {e}")
            plugin._set_status(PluginStatus.ERROR, e)
            return False

    async def execute_plugin(
        self, plugin_name: str, *args, timeout: Optional[float] = None, **kwargs
    ) -> Any:
        """Execute a plugin with timeout and error handling."""
        if plugin_name not in self._plugins:
            raise ValueError(f"Plugin {plugin_name} is not loaded")

        plugin = self._plugins[plugin_name]

        if not plugin.is_active:
            raise RuntimeError(f"Plugin {plugin_name} is not active")

        try:
            # Increment execution metric
            plugin._increment_metric("executions")

            start_time = datetime.utcnow()

            # Execute with timeout
            if timeout:
                result = await asyncio.wait_for(
                    plugin.execute(*args, **kwargs), timeout=timeout
                )
            else:
                result = await plugin.execute(*args, **kwargs)

            # Update metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            plugin._update_metric("last_execution_time", execution_time)
            plugin._increment_metric("successful_executions")

            return result

        except asyncio.TimeoutError:
            plugin._increment_metric("timeout_errors")
            logger.error(f"Plugin {plugin_name} execution timed out after {timeout}s")
            raise
        except Exception as e:
            plugin._increment_metric("execution_errors")
            logger.error(f"Plugin {plugin_name} execution failed: {e}")
            raise

    async def get_plugin(self, plugin_name: str) -> Optional[PluginBase]:
        """Get a loaded plugin instance."""
        return self._plugins.get(plugin_name)

    async def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about a plugin."""
        plugin = self._plugins.get(plugin_name)
        if not plugin:
            return None

        return plugin.get_info()

    async def list_plugins(
        self,
        plugin_type: Optional[PluginType] = None,
        status: Optional[PluginStatus] = None,
    ) -> List[Dict[str, Any]]:
        """List all plugins with optional filtering."""
        plugins = []

        for plugin_name, plugin in self._plugins.items():
            plugin_info = plugin.get_info()

            # Apply filters
            if plugin_type and plugin.metadata.plugin_type != plugin_type:
                continue

            if status and plugin.status != status:
                continue

            plugins.append(plugin_info)

        return plugins

    async def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginBase]:
        """Get all loaded plugins of a specific type."""
        return [
            plugin
            for plugin in self._plugins.values()
            if plugin.metadata.plugin_type == plugin_type
        ]

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all plugins."""
        health_status = {
            "manager_status": "healthy" if self._is_initialized else "unhealthy",
            "total_plugins": len(self._plugins),
            "healthy_plugins": 0,
            "unhealthy_plugins": 0,
            "plugin_statuses": {},
        }

        for plugin_name, plugin in self._plugins.items():
            try:
                is_healthy = await plugin.health_check()
                health_status["plugin_statuses"][plugin_name] = {
                    "healthy": is_healthy,
                    "status": plugin.status.value,
                    "last_error": str(plugin.error) if plugin.error else None,
                }

                if is_healthy:
                    health_status["healthy_plugins"] += 1
                else:
                    health_status["unhealthy_plugins"] += 1

            except Exception as e:
                health_status["plugin_statuses"][plugin_name] = {
                    "healthy": False,
                    "status": "error",
                    "last_error": str(e),
                }
                health_status["unhealthy_plugins"] += 1

        return health_status

    async def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin (unload and load again)."""
        logger.info(f"Reloading plugin: {plugin_name}")

        # Store current config
        current_config = None
        if plugin_name in self._plugins:
            current_config = self._plugins[plugin_name].config.dict()

        # Unload plugin
        if not await self.unload_plugin(plugin_name):
            return False

        # Load plugin again
        return await self.load_plugin(plugin_name, current_config)

    async def load_all_plugins(self) -> Dict[str, bool]:
        """Load all discovered plugins."""
        logger.info("Loading all discovered plugins")

        results = {}
        plugins = await self.registry.list_plugins()

        for plugin_name in plugins:
            results[plugin_name] = await self.load_plugin(plugin_name)

        loaded_count = sum(1 for success in results.values() if success)
        logger.info(f"Loaded {loaded_count}/{len(results)} plugins")

        return results

    async def unload_all_plugins(self) -> Dict[str, bool]:
        """Unload all loaded plugins."""
        logger.info("Unloading all plugins")

        results = {}
        plugin_names = list(self._plugins.keys())

        for plugin_name in plugin_names:
            results[plugin_name] = await self.unload_plugin(plugin_name)

        return results

    # Event system
    def on(self, event: str, handler: Callable) -> None:
        """Register an event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    def off(self, event: str, handler: Callable) -> None:
        """Unregister an event handler."""
        if event in self._event_handlers:
            try:
                self._event_handlers[event].remove(handler)
            except ValueError:
                pass

    async def _emit_event(self, event: str, **kwargs) -> None:
        """Emit an event to all registered handlers."""
        if event in self._event_handlers:
            for handler in self._event_handlers[event]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event, **kwargs)
                    else:
                        handler(event, **kwargs)
                except Exception as e:
                    logger.error(f"Error in event handler for {event}: {e}")

    # Private methods
    async def _load_plugin_configs(self) -> None:
        """Load plugin configurations from files."""
        config_file = self.plugin_dir / "config.json"

        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    self._plugin_configs = json.load(f)
                logger.info(f"Loaded plugin configurations from {config_file}")
            except Exception as e:
                logger.error(f"Error loading plugin configurations: {e}")

    async def save_plugin_configs(self) -> None:
        """Save current plugin configurations to file."""
        config_file = self.plugin_dir / "config.json"

        try:
            # Collect current configs
            current_configs = {}
            for plugin_name, plugin in self._plugins.items():
                current_configs[plugin_name] = plugin.config.dict()

            with open(config_file, "w") as f:
                json.dump(current_configs, f, indent=2)

            logger.info(f"Saved plugin configurations to {config_file}")

        except Exception as e:
            logger.error(f"Error saving plugin configurations: {e}")

    # Context manager support
    @asynccontextmanager
    async def plugin_context(self, plugin_name: str):
        """Context manager for plugin operations."""
        plugin = await self.get_plugin(plugin_name)
        if not plugin:
            raise ValueError(f"Plugin {plugin_name} not found")

        try:
            yield plugin
        except Exception as e:
            logger.error(f"Error in plugin context for {plugin_name}: {e}")
            plugin._set_status(PluginStatus.ERROR, e)
            raise

    # Statistics and metrics
    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            "initialized": self._is_initialized,
            "startup_time": self._startup_time.isoformat(),
            "uptime": (datetime.now(timezone.utc) - self._startup_time).total_seconds(),
            "total_plugins": len(self._plugins),
            "active_plugins": len([p for p in self._plugins.values() if p.is_active]),
            "loaded_plugins": len([p for p in self._plugins.values() if p.is_loaded]),
            "plugin_types": {
                ptype.value: len(
                    [
                        p
                        for p in self._plugins.values()
                        if p.metadata.plugin_type == ptype
                    ]
                )
                for ptype in PluginType
            },
        }
