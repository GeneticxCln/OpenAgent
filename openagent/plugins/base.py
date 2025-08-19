"""
Base classes and interfaces for the OpenAgent plugin system.

Provides the foundation for creating, loading, and managing plugins
with proper validation, security, and lifecycle management.

Also provides backwards-compatible shims for older plugin APIs used in
some unit tests (BasePlugin, PluginError).
"""

import abc
import asyncio
import hashlib
import inspect
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field, validator


class PluginType(Enum):
    """Plugin types supported by OpenAgent."""

    TOOL = "tool"
    AGENT = "agent"
    MODEL = "model"
    MIDDLEWARE = "middleware"
    INTEGRATION = "integration"
    UI_EXTENSION = "ui_extension"
    CUSTOM = "custom"


class PluginStatus(Enum):
    """Plugin status states."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DEPRECATED = "deprecated"


@dataclass
class PluginMetadata:
    """Metadata for plugin description and management."""

    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType

    # Dependencies and compatibility
    dependencies: List[str] = field(default_factory=list)
    openagent_version: str = ">=1.0.0"
    python_version: str = ">=3.9"

    # Plugin configuration
    config_schema: Optional[Dict[str, Any]] = None
    permissions: List[str] = field(default_factory=list)

    # Marketplace information
    homepage: Optional[str] = None
    repository: Optional[str] = None
    license: str = "MIT"
    keywords: List[str] = field(default_factory=list)

    # Internal metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    checksum: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "plugin_type": self.plugin_type.value,
            "dependencies": self.dependencies,
            "openagent_version": self.openagent_version,
            "python_version": self.python_version,
            "config_schema": self.config_schema,
            "permissions": self.permissions,
            "homepage": self.homepage,
            "repository": self.repository,
            "license": self.license,
            "keywords": self.keywords,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginMetadata":
        """Create metadata from dictionary."""
        return cls(
            name=data["name"],
            version=data["version"],
            description=data["description"],
            author=data["author"],
            plugin_type=PluginType(data["plugin_type"]),
            dependencies=data.get("dependencies", []),
            openagent_version=data.get("openagent_version", ">=1.0.0"),
            python_version=data.get("python_version", ">=3.9"),
            config_schema=data.get("config_schema"),
            permissions=data.get("permissions", []),
            homepage=data.get("homepage"),
            repository=data.get("repository"),
            license=data.get("license", "MIT"),
            keywords=data.get("keywords", []),
            created_at=datetime.fromisoformat(
                data.get("created_at", datetime.now(timezone.utc).isoformat())
            ),
            updated_at=datetime.fromisoformat(
                data.get("updated_at", datetime.now(timezone.utc).isoformat())
            ),
            checksum=data.get("checksum"),
        )


class PluginConfig(BaseModel):
    """Base configuration model for plugins."""

    enabled: bool = True
    debug: bool = False
    config: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class PluginError(Exception):
    """Generic plugin error (backwards-compat shim)."""


class PluginMessage(BaseModel):
    """Message used for plugin-to-plugin communication."""

    source: str
    target: Optional[str] = None  # None means broadcast
    type: str = Field(..., description="Message type identifier")
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PluginBase(abc.ABC):
    """
    Abstract base class for all OpenAgent plugins.

    This class defines the interface that all plugins must implement
    and provides common functionality for plugin lifecycle management.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the plugin with configuration."""
        self._metadata: Optional[PluginMetadata] = None
        self._config = PluginConfig(**(config or {}))
        self._status = PluginStatus.UNLOADED
        self._error: Optional[Exception] = None
        self._loaded_at: Optional[datetime] = None
        self._metrics: Dict[str, Any] = {}

    # Abstract methods that must be implemented
    @abc.abstractmethod
    async def initialize(self) -> bool:
        """Initialize the plugin. Return True if successful."""
        pass

    @abc.abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """Execute the plugin's main functionality."""
        pass

    @abc.abstractmethod
    async def cleanup(self) -> bool:
        """Clean up plugin resources. Return True if successful."""
        pass

    @abc.abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass

    # Optional methods with default implementations
    async def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration. Override for custom validation."""
        return True

    async def on_load(self) -> None:
        """Called when plugin is loaded."""
        pass

    async def on_unload(self) -> None:
        """Called when plugin is unloaded."""
        pass

    async def on_enable(self) -> None:
        """Called when plugin is enabled."""
        pass

    async def on_disable(self) -> None:
        """Called when plugin is disabled."""
        pass

    # Optional extended lifecycle hooks
    async def on_suspend(self) -> None:
        """Called when plugin is suspended (resources may be released)."""
        pass

    async def on_resume(self) -> None:
        """Called when plugin is resumed after suspension."""
        pass

    async def health_check(self) -> bool:
        """Check if plugin is healthy. Override for custom health checks."""
        return self._status == PluginStatus.ACTIVE

    # Property accessors
    @property
    def metadata(self) -> Optional[PluginMetadata]:
        """Get plugin metadata."""
        if not self._metadata:
            self._metadata = self.get_metadata()
        return self._metadata

    @property
    def config(self) -> PluginConfig:
        """Get plugin configuration."""
        return self._config

    @property
    def status(self) -> PluginStatus:
        """Get plugin status."""
        return self._status

    @property
    def is_loaded(self) -> bool:
        """Check if plugin is loaded."""
        return self._status in [PluginStatus.LOADED, PluginStatus.ACTIVE]

    @property
    def is_active(self) -> bool:
        """Check if plugin is active."""
        return self._status == PluginStatus.ACTIVE

    @property
    def error(self) -> Optional[Exception]:
        """Get last error if any."""
        return self._error

    @property
    def loaded_at(self) -> Optional[datetime]:
        """Get when plugin was loaded."""
        return self._loaded_at

    @property
    def metrics(self) -> Dict[str, Any]:
        """Get plugin metrics."""
        return self._metrics.copy()

    # Internal methods
    def _set_status(
        self, status: PluginStatus, error: Optional[Exception] = None
    ) -> None:
        """Set plugin status."""
        self._status = status
        self._error = error
        if status == PluginStatus.LOADED:
            self._loaded_at = datetime.now(timezone.utc)

    def _update_metric(self, key: str, value: Any) -> None:
        """Update a plugin metric."""
        self._metrics[key] = value

    def _increment_metric(self, key: str, amount: int = 1) -> None:
        """Increment a plugin metric."""
        self._metrics[key] = self._metrics.get(key, 0) + amount

    # Utility methods
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive plugin information."""
        metadata = self.metadata
        return {
            "metadata": metadata.to_dict() if metadata else None,
            "status": self._status.value,
            "config": self._config.dict(),
            "is_loaded": self.is_loaded,
            "is_active": self.is_active,
            "loaded_at": self._loaded_at.isoformat() if self._loaded_at else None,
            "error": str(self._error) if self._error else None,
            "metrics": self._metrics,
        }

    def calculate_checksum(self) -> str:
        """Calculate checksum for the plugin."""
        source = inspect.getsource(self.__class__)
        return hashlib.sha256(source.encode()).hexdigest()


class ToolPlugin(PluginBase):
    """Base class for tool plugins."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._tool_name: Optional[str] = None

    @abc.abstractmethod
    async def execute_tool(self, command: str, *args, **kwargs) -> Any:
        """Execute a tool command."""
        pass

    @abc.abstractmethod
    def get_available_commands(self) -> List[str]:
        """Get list of available tool commands."""
        pass

    async def execute(self, command: str, *args, **kwargs) -> Any:
        """Execute tool command (implements PluginBase.execute)."""
        return await self.execute_tool(command, *args, **kwargs)

    @property
    def tool_name(self) -> str:
        """Get tool name."""
        if not self._tool_name:
            metadata = self.metadata
            self._tool_name = metadata.name if metadata else self.__class__.__name__
        return self._tool_name


class AgentPlugin(PluginBase):
    """Base class for agent plugins."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._agent_name: Optional[str] = None

    @abc.abstractmethod
    async def process_message(
        self, message: str, context: Dict[str, Any] = None
    ) -> str:
        """Process a message and return response."""
        pass

    async def execute(self, message: str, context: Dict[str, Any] = None) -> str:
        """Process message (implements PluginBase.execute)."""
        return await self.process_message(message, context or {})

    @property
    def agent_name(self) -> str:
        """Get agent name."""
        if not self._agent_name:
            metadata = self.metadata
            self._agent_name = metadata.name if metadata else self.__class__.__name__
        return self._agent_name


class IntegrationPlugin(PluginBase):
    """Base class for integration plugins."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._service_name: Optional[str] = None

    @abc.abstractmethod
    async def connect(self) -> bool:
        """Connect to external service."""
        pass

    @abc.abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from external service."""
        pass

    @abc.abstractmethod
    async def call_service(self, method: str, *args, **kwargs) -> Any:
        """Call external service method."""
        pass

    async def execute(self, method: str, *args, **kwargs) -> Any:
        """Call service method (implements PluginBase.execute)."""
        return await self.call_service(method, *args, **kwargs)

    @property
    def service_name(self) -> str:
        """Get service name."""
        if not self._service_name:
            metadata = self.metadata
            self._service_name = metadata.name if metadata else self.__class__.__name__
        return self._service_name


# Backwards-compatible BasePlugin shim expected by some tests
class BasePlugin(PluginBase):
    """
    Compatibility layer that maps legacy BasePlugin API onto PluginBase.
    Expected legacy properties: name (from config), version, description.
    Expected methods: initialize(), shutdown().
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Legacy attributes
        self.name: str = self._config.config.get("name") or self._config.config.get(
            "plugin_name", self.__class__.__name__.lower()
        )
        self.enabled: bool = self._config.enabled
        self.config: Dict[str, Any] = self._config.config or {}

    # Legacy abstract properties expected by tests
    @property
    @abc.abstractmethod
    def version(self) -> str:  # pragma: no cover - abstract
        pass

    @property
    @abc.abstractmethod
    def description(self) -> str:  # pragma: no cover - abstract
        pass

    # Legacy lifecycle method name mapping
    async def shutdown(self) -> None:  # pragma: no cover - usually overridden
        await self.cleanup()

    # Provide default metadata implementation so tests need not implement it
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name,
            version=self.version,
            description=self.description,
            author=self.config.get("author", "unknown"),
            plugin_type=PluginType.CUSTOM,
        )


# Plugin factory function
def create_plugin(
    plugin_class: Type[PluginBase], config: Optional[Dict[str, Any]] = None
) -> PluginBase:
    """Create a plugin instance with proper validation."""
    if not issubclass(plugin_class, PluginBase):
        raise TypeError(
            f"Plugin class must inherit from PluginBase, got {plugin_class}"
        )

    return plugin_class(config)


# Plugin decorators
def plugin_metadata(
    name: str,
    version: str,
    description: str,
    author: str,
    plugin_type: PluginType,
    **kwargs,
):
    """Decorator to add metadata to a plugin class."""

    def decorator(cls):
        cls._plugin_metadata = PluginMetadata(
            name=name,
            version=version,
            description=description,
            author=author,
            plugin_type=plugin_type,
            **kwargs,
        )
        return cls

    return decorator


def requires_permissions(*permissions: str):
    """Decorator to specify required permissions for a plugin."""

    def decorator(cls):
        if not hasattr(cls, "_plugin_metadata"):
            raise AttributeError(
                "Plugin must have metadata before specifying permissions"
            )
        cls._plugin_metadata.permissions.extend(permissions)
        return cls

    return decorator


def plugin_config_schema(schema: Dict[str, Any]):
    """Decorator to specify configuration schema for a plugin."""

    def decorator(cls):
        if not hasattr(cls, "_plugin_metadata"):
            raise AttributeError(
                "Plugin must have metadata before specifying config schema"
            )
        cls._plugin_metadata.config_schema = schema
        return cls

    return decorator
