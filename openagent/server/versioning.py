"""
API Versioning support for OpenAgent server.

This module provides version detection, compatibility checks, and deprecation
warnings for maintaining backward compatibility across API versions.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class APIVersion(Enum):
    """Supported API versions."""

    V1_0 = "1.0"
    V1_1 = "1.1"  # Future version
    V1_2 = "1.2"  # Future version

    # Current stable version
    CURRENT = V1_0


# Deprecated versions (none yet)
DEPRECATED_VERSIONS: List[str] = []


@dataclass
class VersionInfo:
    """Version information response."""

    version: str
    api_version: str
    supported_versions: List[str]
    deprecated_versions: List[str]
    deprecation_warnings: List[str]
    changelog_url: Optional[str] = None
    migration_guide_url: Optional[str] = None


class APIVersioning:
    """API versioning manager."""

    def __init__(self):
        self.current_version = APIVersion.CURRENT
        self.supported_versions = [v.value for v in APIVersion if v.value != "CURRENT"]
        self.deprecated_versions = DEPRECATED_VERSIONS.copy()

    def extract_version_from_request(self, request: Request) -> str:
        """Extract API version from request headers or path."""

        # Check Accept header first (preferred)
        accept_header = request.headers.get("Accept", "")
        if "version=" in accept_header:
            for part in accept_header.split(";"):
                if part.strip().startswith("version="):
                    return part.strip().split("=")[1]

        # Check custom header
        api_version = request.headers.get("X-API-Version")
        if api_version:
            return api_version

        # Check path prefix (e.g., /v1/chat)
        path = request.url.path
        if path.startswith("/v"):
            parts = path.split("/")
            if len(parts) >= 2:
                version_part = parts[1]
                if version_part.startswith("v") and len(version_part) > 1:
                    return version_part[1:]  # Remove 'v' prefix

        # Default to current version
        return self.current_version.value

    def validate_version(self, version: str) -> bool:
        """Check if version is supported."""
        return version in self.supported_versions

    def is_deprecated(self, version: str) -> bool:
        """Check if version is deprecated."""
        return version in self.deprecated_versions

    def get_deprecation_warning(self, version: str) -> Optional[str]:
        """Get deprecation warning for version."""
        if self.is_deprecated(version):
            return (
                f"API version {version} is deprecated and will be removed in a future release. "
                f"Please upgrade to version {self.current_version.value}."
            )
        return None

    def get_version_info(self) -> VersionInfo:
        """Get comprehensive version information."""
        return VersionInfo(
            version="0.1.3",  # Application version
            api_version=self.current_version.value,
            supported_versions=self.supported_versions,
            deprecated_versions=self.deprecated_versions,
            deprecation_warnings=[
                self.get_deprecation_warning(v) for v in self.deprecated_versions
            ],
            changelog_url="https://github.com/yourusername/OpenAgent/blob/main/CHANGELOG.md",
            migration_guide_url="https://github.com/yourusername/OpenAgent/blob/main/docs/api-migration.md",
        )

    async def version_middleware(self, request: Request, call_next):
        """Middleware to handle API versioning."""

        # Skip version handling for certain paths
        path = request.url.path
        skip_paths = [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/healthz",
            "/readyz",
        ]
        if any(path.startswith(skip) for skip in skip_paths):
            return await call_next(request)

        # Extract and validate version
        requested_version = self.extract_version_from_request(request)

        if not self.validate_version(requested_version):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "unsupported_api_version",
                    "message": f"API version '{requested_version}' is not supported",
                    "supported_versions": self.supported_versions,
                    "current_version": self.current_version.value,
                },
            )

        # Add version info to request state
        request.state.api_version = requested_version

        # Process request
        response = await call_next(request)

        # Add version headers to response
        response.headers["X-API-Version"] = requested_version
        response.headers["X-Current-Version"] = self.current_version.value

        # Add deprecation warning if needed
        warning = self.get_deprecation_warning(requested_version)
        if warning:
            response.headers["Warning"] = f'299 - "{warning}"'
            logger.warning(f"Deprecated API version used: {requested_version}")

        return response


# Global versioning instance
versioning = APIVersioning()


def get_requested_version(request: Request) -> str:
    """Get the API version for the current request."""
    return getattr(request.state, "api_version", APIVersion.CURRENT.value)


def require_version(min_version: str):
    """Decorator to require minimum API version."""

    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            current_version = get_requested_version(request)

            # Simple version comparison (assumes semantic versioning)
            if current_version < min_version:
                raise HTTPException(
                    status_code=400,
                    detail=f"This endpoint requires API version {min_version} or higher. "
                    f"Current version: {current_version}",
                )

            return await func(request, *args, **kwargs)

        return wrapper

    return decorator


# Compatibility layer for common request/response transformations
class CompatibilityLayer:
    """Handle backward compatibility transformations."""

    @staticmethod
    def transform_chat_request_v1_0(data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform chat request for v1.0 compatibility."""
        # v1.0 expects 'message', newer versions might use 'content'
        if "content" in data and "message" not in data:
            data["message"] = data["content"]

        return data

    @staticmethod
    def transform_chat_response_v1_0(data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform chat response for v1.0 compatibility."""
        # Ensure required fields are present for v1.0
        if "timestamp" not in data:
            from datetime import datetime, timezone

            data["timestamp"] = datetime.now(timezone.utc).isoformat()

        return data


compatibility = CompatibilityLayer()
