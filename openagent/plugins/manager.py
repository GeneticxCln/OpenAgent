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

        # Global tool catalog auto-populated from enabled plugins providing tools
        self._tool_catalog: Dict[str, Any] = {}

        # Event system
        self._event_handlers: Dict[str, List[Callable]] = {}

        # Simple in-process message bus (topic -> handlers)
        self._message_subscribers: Dict[str, List[Callable[[str, Dict[str, Any]], Any]]] = {}

        # Thread pool for plugin operations
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Manager state
        self._is_initialized = False
        self._startup_time = datetime.now(timezone.utc)

        logger.info(
            f"PluginManager initialized with plugin directory: {self.plugin_dir}"
        )

    def on(self, event: str, handler: Callable) -> None:
        """Register an event handler for a given event name."""
        self._event_handlers.setdefault(event, []).append(handler)

    async def _emit_event(self, event: str, **kwargs: Any) -> None:
        """Emit an event to all registered handlers. Best-effort, never throws."""
        handlers = list(self._event_handlers.get(event, []))
        for h in handlers:
            try:
                if asyncio.iscoroutinefunction(h):
                    await h(**kwargs)
                else:
                    # Execute sync handler in default loop executor to avoid blocking
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, lambda: h(**kwargs))
            except Exception:
                # Never fail manager operations due to handler errors
                continue

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

            # Validate config against metadata schema (if provided)
            try:
                if metadata.config_schema and isinstance(plugin_config, dict):
                    if not self._validate_config_schema(plugin_config, metadata.config_schema):
                        logger.error(f"Config schema validation failed for {plugin_name}")
                        return False
            except Exception as e:
                logger.error(f"Config validation error for {plugin_name}: {e}")
                return False

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

            # Permission gating
            metadata = plugin.metadata
            if metadata and metadata.permissions:
                if ("execute_commands" in metadata.permissions) and not self.config.get(
                    "allow_execute_commands", False
                ):
                    logger.error(
                        f"Permission denied: execute_commands not allowed for {plugin_name}"
                    )
                    return False
                if ("network_access" in metadata.permissions) and not self.config.get(
                    "allow_network", True
                ):
                    logger.error(
                        f"Permission denied: network_access not allowed for {plugin_name}"
                    )
                    return False

            # Call on_enable callback
            await plugin.on_enable()

            # Set status to active and mark enabled in config
            plugin._set_status(PluginStatus.ACTIVE)
            try:
                # Keep PluginConfig in sync for persistence
                if hasattr(plugin, "_config"):
                    plugin._config.enabled = True
                # Some legacy plugins mirror enabled attr
                if hasattr(plugin, "enabled"):
                    setattr(plugin, "enabled", True)
            except Exception:
                pass

            # Auto-register tools if provided
            tools_getter = getattr(plugin, "get_tools", None)
            if callable(tools_getter):
                try:
                    tools = tools_getter()
                    for t in tools or []:
                        tool_name = getattr(t, "name", t.__class__.__name__)
                        # Namespace key by plugin to avoid collisions in catalog
                        key = f"{plugin_name}:{tool_name}"
                        self._tool_catalog[key] = t
                except Exception as e:
                    logger.warning(f"Failed to register tools from {plugin_name}: {e}")

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

            # Set status to inactive and mark disabled in config
            plugin._set_status(PluginStatus.INACTIVE)
            try:
                if hasattr(plugin, "_config"):
                    plugin._config.enabled = False
                if hasattr(plugin, "enabled"):
                    setattr(plugin, "enabled", False)
            except Exception:
                pass

            # Remove tools from catalog that belong to this plugin (best-effort)
            try:
                # Remove all catalog entries for this plugin by namespaced prefix
                prefix = f"{plugin_name}:"
                for n in list(self._tool_catalog.keys()):
                    if n.startswith(prefix):
                        del self._tool_catalog[n]
            except Exception:
                pass

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
            "tool_catalog_size": len(self._tool_catalog),
        }

    async def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin (unload and load again)."""
        logger.info(f"Reloading plugin: {plugin_name}")
        current_config = None
        if plugin_name in self._plugins:
            current_config = self._plugins[plugin_name].config.dict()
        if not await self.unload_plugin(plugin_name):
            return False
        return await self.load_plugin(plugin_name, current_config)

    async def load_all_plugins(self) -> Dict[str, bool]:
        """Load all discovered plugins."""
        logger.info("Loading all discovered plugins")
        results: Dict[str, bool] = {}
        plugins = await self.registry.list_plugins()
        for plugin_name in plugins:
            results[plugin_name] = await self.load_plugin(plugin_name)
        loaded_count = sum(1 for success in results.values() if success)
        logger.info(f"Loaded {loaded_count}/{len(results)} plugins")
        return results

    async def unload_all_plugins(self) -> Dict[str, bool]:
        """Unload all loaded plugins."""
        logger.info("Unloading all plugins")
        results: Dict[str, bool] = {}
        plugin_names = list(self._plugins.keys())
        for plugin_name in plugin_names:
            results[plugin_name] = await self.unload_plugin(plugin_name)
        return results

    def get_tool_catalog(self) -> Dict[str, Any]:
        """Return the current tool catalog (name -> tool instance)."""
        return self._tool_catalog.copy()

    def get_tool_entries(self, plugins: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """Return enriched tool entries for listing/registration.
        Each entry: { 'plugin': str, 'version': str|None, 'tool_name': str, 'key': str, 'tool': Any }
        Optionally filter by a set of plugin names.
        """
        entries: List[Dict[str, Any]] = []
        for key, tool in self._tool_catalog.items():
            try:
                plug, tool_name = key.split(":", 1)
            except ValueError:
                plug, tool_name = "", key
            if plugins and plug not in plugins:
                continue
            version = None
            try:
                pinst = self._plugins.get(plug)
                if pinst and pinst.metadata:
                    version = getattr(pinst.metadata, "version", None)
            except Exception:
                version = None
            entries.append({
                "plugin": plug,
                "version": version,
                "tool_name": tool_name,
                "key": key,
                "tool": tool,
            })
        return entries

    def register_tools_with_agent(self, agent: Any, plugins: Optional[Set[str]] = None) -> int:
        """Attach tools from the catalog to the given agent, avoiding duplicates.
        If 'plugins' is provided, only tools from those plugins are registered.
        Returns the number of tools added.
        """
        added = 0
        try:
            for key, tool in self._tool_catalog.items():
                if plugins:
                    try:
                        plug_prefix = key.split(":", 1)[0]
                        if plug_prefix not in plugins:
                            continue
                    except Exception:
                        continue
                tool_name = getattr(tool, "name", tool.__class__.__name__)
                try:
                    existing = getattr(agent, "get_tool", None)
                    if callable(existing) and existing(tool_name):
                        continue
                except Exception:
                    pass
                try:
                    agent.add_tool(tool)
                    added += 1
                except Exception:
                    # continue on individual tool failures
                    continue
        except Exception:
            return added
        return added

    async def _load_plugin_configs(self) -> None:
        """Load plugin configurations from files."""
        config_json = self.plugin_dir / "config.json"
        config_yaml = self.plugin_dir / "config.yaml"
        if config_json.exists():
            try:
                with open(config_json, "r") as f:
                    self._plugin_configs = json.load(f)
                logger.info(f"Loaded plugin configurations from {config_json}")
            except Exception as e:
                logger.error(f"Error loading plugin configurations: {e}")
        elif config_yaml.exists():
            try:
                import yaml  # type: ignore
                with open(config_yaml, "r") as f:
                    self._plugin_configs = yaml.safe_load(f) or {}
                logger.info(f"Loaded plugin configurations from {config_yaml}")
            except Exception as e:
                logger.error(f"Error loading YAML plugin configurations: {e}")
        try:
            for item in self.plugin_dir.iterdir():
                plugin_name = None
                plugin_conf: Optional[Dict[str, Any]] = None
                if item.is_dir():
                    y = item / "plugin.yaml"
                    j = item / "plugin.json"
                    if y.exists():
                        try:
                            import yaml  # type: ignore
                            with open(y, "r") as f:
                                plugin_conf = yaml.safe_load(f) or {}
                        except Exception as e:
                            logger.warning(f"Failed to load {y}: {e}")
                    elif j.exists():
                        try:
                            with open(j, "r") as f:
                                plugin_conf = json.load(f)
                        except Exception as e:
                            logger.warning(f"Failed to load {j}: {e}")
                    if plugin_conf and isinstance(plugin_conf, dict):
                        plugin_name = plugin_conf.get("name") or item.name
                        self._plugin_configs.setdefault(plugin_name, {}).update(plugin_conf)
                elif item.is_file() and item.suffix in {".py"}:
                    y = item.with_suffix(".yaml")
                    j = item.with_suffix(".json")
                    if y.exists():
                        try:
                            import yaml  # type: ignore
                            with open(y, "r") as f:
                                plugin_conf = yaml.safe_load(f) or {}
                        except Exception as e:
                            logger.warning(f"Failed to load {y}: {e}")
                    elif j.exists():
                        try:
                            with open(j, "r") as f:
                                plugin_conf = json.load(f)
                        except Exception as e:
                            logger.warning(f"Failed to load {j}: {e}")
                    if plugin_conf and isinstance(plugin_conf, dict):
                        plugin_name = plugin_conf.get("name") or item.stem
                        self._plugin_configs.setdefault(plugin_name, {}).update(plugin_conf)
        except Exception as e:
            logger.error(f"Error scanning per-plugin configs: {e}")

    # Enhanced JSON-schema-like validator with required, enum, nested objects
    def _validate_config_schema(self, cfg: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        def _validate(value, sch) -> bool:
            stype = sch.get("type")
            if stype == "object" or (stype is None and isinstance(value, dict)):
                if not isinstance(value, dict):
                    return False
                # required keys
                required = sch.get("required", [])
                for req in required:
                    if req not in value:
                        return False
                # properties
                props = sch.get("properties", {})
                for k, sub in props.items():
                    if k in value and not _validate(value[k], sub):
                        return False
                return True
            if stype == "array":
                if not isinstance(value, list):
                    return False
                item_sch = sch.get("items")
                if item_sch:
                    for item in value:
                        if not _validate(item, item_sch):
                            return False
                enum_vals = sch.get("enum")
                if enum_vals is not None and value not in enum_vals:
                    # Arrays compared as whole when enum provided
                    return False
                return True
            # primitives
            type_map = {
                "string": str,
                "boolean": bool,
                "number": (int, float),
                "integer": int,
            }
            if stype in type_map and not isinstance(value, type_map[stype]):
                return False
            enum_vals = sch.get("enum")
            if enum_vals is not None and value not in enum_vals:
                return False
            return True

        # Ensure object at top level
        if not isinstance(cfg, dict):
            return False
        if schema.get("type") and schema["type"] != "object":
            return False
        # Validate each declared property if present
        return _validate(cfg, schema)
