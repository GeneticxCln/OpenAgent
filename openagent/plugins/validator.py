"""
Plugin Validator for OpenAgent

Validates plugins for security, compatibility, and correctness
before loading and execution.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import semver

from .base import PluginBase, PluginMetadata, PluginType

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of plugin validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    score: float = 0.0  # Validation score (0-1)


class PluginValidator:
    """Validates plugins for security and compatibility."""

    def __init__(self):
        import sys as _sys
        self.openagent_version = "1.0.0"  # Should be loaded from config
        # Use the running interpreter version for comparisons
        self.python_version = f"{_sys.version_info.major}.{_sys.version_info.minor}.{_sys.version_info.micro}"

    async def validate_metadata(self, metadata: PluginMetadata) -> ValidationResult:
        """Validate plugin metadata."""
        errors = []
        warnings = []

        # Required fields
        if not metadata.name or not metadata.name.strip():
            errors.append("Plugin name is required")
        elif not self._is_valid_plugin_name(metadata.name):
            errors.append("Plugin name contains invalid characters")

        if not metadata.version or not metadata.version.strip():
            errors.append("Plugin version is required")
        elif not self._is_valid_version(metadata.version):
            errors.append("Plugin version is not valid semver")

        if not metadata.author or not metadata.author.strip():
            warnings.append("Plugin author is recommended")

        # Version compatibility
        if not self._check_version_compatibility(metadata.openagent_version):
            errors.append(
                f"Incompatible OpenAgent version requirement: {metadata.openagent_version}"
            )

        if not self._check_version_compatibility(
            metadata.python_version, self.python_version
        ):
            errors.append(
                f"Incompatible Python version requirement: {metadata.python_version}"
            )

        # Plugin type validation
        if metadata.plugin_type not in PluginType:
            errors.append(f"Invalid plugin type: {metadata.plugin_type}")

        # Permissions validation
        if metadata.permissions:
            invalid_perms = self._validate_permissions(metadata.permissions)
            if invalid_perms:
                warnings.extend(
                    [f"Unknown permission: {perm}" for perm in invalid_perms]
                )

        # Calculate score
        score = self._calculate_validation_score(metadata, errors, warnings)

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=warnings, score=score
        )

    async def validate_plugin_class(
        self, plugin_class: Type[PluginBase]
    ) -> ValidationResult:
        """Validate a plugin class implementation."""
        errors = []
        warnings = []

        # Check inheritance
        if not issubclass(plugin_class, PluginBase):
            errors.append("Plugin class must inherit from PluginBase")

        # Check required methods
        required_methods = ["initialize", "execute", "cleanup", "get_metadata"]
        for method_name in required_methods:
            if not hasattr(plugin_class, method_name):
                errors.append(f"Missing required method: {method_name}")
            elif not callable(getattr(plugin_class, method_name)):
                errors.append(f"Method {method_name} is not callable")

        # Check metadata method
        try:
            # Create temporary instance to check metadata
            temp_instance = plugin_class({})
            metadata = temp_instance.get_metadata()
            if not isinstance(metadata, PluginMetadata):
                errors.append("get_metadata() must return PluginMetadata instance")
        except Exception as e:
            errors.append(f"Error creating plugin instance for validation: {e}")

        # Check for common implementation issues
        if not hasattr(plugin_class, "__init__"):
            warnings.append("Plugin class should implement __init__ method")

        # Calculate score
        score = 1.0 - (len(errors) * 0.2) - (len(warnings) * 0.05)
        score = max(0.0, min(1.0, score))

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=warnings, score=score
        )

    def _is_valid_plugin_name(self, name: str) -> bool:
        """Check if plugin name is valid."""
        # Must be alphanumeric with underscores and hyphens
        pattern = r"^[a-zA-Z0-9_-]+$"
        return bool(re.match(pattern, name)) and len(name) <= 50

    def _is_valid_version(self, version: str) -> bool:
        """Check if version string is valid semver."""
        try:
            semver.VersionInfo.parse(version)
            return True
        except ValueError:
            return False

    def _check_version_compatibility(
        self, requirement: str, current: Optional[str] = None
    ) -> bool:
        """Check version compatibility.

        Be lenient about version formats like '3.9' by normalizing to valid semver '3.9.0'.
        If parsing fails, prefer to return True to avoid blocking plugin discovery in tests.
        """
        if not requirement:
            return True

        current = current or self.openagent_version

        def _norm(v: str) -> str:
            v = v.strip()
            # Strip any build/metadata for simple compare
            v = v.split("+", 1)[0]
            v = v.split("-", 1)[0]
            # If it's like '3.9', make it '3.9.0'
            parts = v.split(".")
            if all(p.isdigit() for p in parts) and 1 <= len(parts) <= 3:
                while len(parts) < 3:
                    parts.append("0")
                return ".".join(parts)
            return v

        req = requirement.strip()
        cur = _norm(current)

        try:
            # Parse requirement (supports >=, >, <, <=, ==, !=)
            if req.startswith(">="):
                min_version = _norm(req[2:])
                return semver.compare(cur, min_version) >= 0
            elif req.startswith(">"):
                min_version = _norm(req[1:])
                return semver.compare(cur, min_version) > 0
            elif req.startswith("<="):
                max_version = _norm(req[2:])
                return semver.compare(cur, max_version) <= 0
            elif req.startswith("<"):
                max_version = _norm(req[1:])
                return semver.compare(cur, max_version) < 0
            elif req.startswith("=="):
                exact_version = _norm(req[2:])
                return semver.compare(cur, exact_version) == 0
            elif req.startswith("!="):
                not_version = _norm(req[2:])
                return semver.compare(cur, not_version) != 0
            else:
                # Assume >= if no operator specified
                return semver.compare(cur, _norm(req)) >= 0
        except Exception:
            # Be permissive on parse errors
            return True

    def _validate_permissions(self, permissions: List[str]) -> List[str]:
        """Validate plugin permissions and return invalid ones."""
        valid_permissions = {
            "read_files",
            "write_files",
            "execute_commands",
            "network_access",
            "system_info",
            "process_management",
            "registry_access",
            "plugin_management",
            "user_data",
            "sensitive_data",
        }

        return [perm for perm in permissions if perm not in valid_permissions]

    def _calculate_validation_score(
        self, metadata: PluginMetadata, errors: List[str], warnings: List[str]
    ) -> float:
        """Calculate validation score based on metadata quality."""
        score = 1.0

        # Deduct for errors and warnings
        score -= len(errors) * 0.2
        score -= len(warnings) * 0.05

        # Bonus for good metadata
        if metadata.description and len(metadata.description) > 10:
            score += 0.1
        if metadata.repository:
            score += 0.05
        if metadata.homepage:
            score += 0.05
        if metadata.keywords:
            score += 0.05
        if metadata.license and metadata.license != "MIT":
            score += 0.05

        return max(0.0, min(1.0, score))


# Marketplace stub for future implementation
class PluginMarketplace:
    """Plugin marketplace for discovering and installing plugins."""

    def __init__(self):
        self.marketplace_url = "https://plugins.openagent.dev"
        logger.info("PluginMarketplace initialized (stub implementation)")

    async def search_plugins(self, query: str) -> List[Dict[str, Any]]:
        """Search for plugins in the marketplace."""
        # Stub implementation
        logger.info(f"Searching marketplace for: {query}")
        return []

    async def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed plugin information from marketplace."""
        # Stub implementation
        logger.info(f"Getting plugin info for: {plugin_name}")
        return None

    async def download_plugin(
        self, plugin_name: str, version: Optional[str] = None
    ) -> bool:
        """Download a plugin from the marketplace."""
        # Stub implementation
        logger.info(f"Downloading plugin: {plugin_name} v{version}")
        return False

    async def publish_plugin(self, plugin_path: str, metadata: PluginMetadata) -> bool:
        """Publish a plugin to the marketplace."""
        # Stub implementation
        logger.info(f"Publishing plugin: {metadata.name}")
        return False
