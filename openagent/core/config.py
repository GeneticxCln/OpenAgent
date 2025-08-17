"""
Configuration management for OpenAgent framework.

This module provides configuration loading, validation, and management
functionality for the OpenAgent framework.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator

from openagent.core.exceptions import ConfigError


class DatabaseConfig(BaseModel):
    """Database configuration settings."""

    url: str = Field("sqlite:///openagent.db", description="Database connection URL")
    echo: bool = Field(False, description="Enable SQL query logging")
    pool_size: int = Field(5, description="Connection pool size")
    max_overflow: int = Field(10, description="Maximum pool overflow")


class LoggingConfig(BaseModel):
    """Logging configuration settings."""

    level: str = Field("INFO", description="Logging level")
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format",
    )
    file: Optional[str] = Field(None, description="Log file path")
    max_file_size: str = Field("10MB", description="Maximum log file size")
    backup_count: int = Field(5, description="Number of backup log files")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(
                f"Invalid logging level: {v}. Must be one of {valid_levels}"
            )
        return v.upper()


class SecurityConfig(BaseModel):
    """Security configuration settings."""

    api_key_header: str = Field("X-API-Key", description="API key header name")
    cors_origins: list = Field(["*"], description="Allowed CORS origins")
    cors_methods: list = Field(["GET", "POST"], description="Allowed CORS methods")
    rate_limit_per_minute: int = Field(60, description="Rate limit per minute")
    max_request_size: str = Field("10MB", description="Maximum request size")


class ServerConfig(BaseModel):
    """Server configuration settings."""

    host: str = Field("localhost", description="Server host")
    port: int = Field(8000, description="Server port")
    workers: int = Field(1, description="Number of worker processes")
    reload: bool = Field(False, description="Enable auto-reload")
    access_log: bool = Field(True, description="Enable access logging")

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port number."""
        if not 1 <= v <= 65535:
            raise ValueError(f"Invalid port number: {v}. Must be between 1 and 65535")
        return v


class Config(BaseModel):
    """Main configuration class for OpenAgent."""

    # Basic settings
    app_name: str = Field("OpenAgent", description="Application name")
    version: str = Field("0.1.0", description="Application version")
    debug: bool = Field(False, description="Enable debug mode")
    environment: str = Field(
        "development", description="Environment (development/staging/production)"
    )

    # Component configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    # Custom settings
    custom: Dict[str, Any] = Field(
        default_factory=dict, description="Custom configuration values"
    )

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        valid_envs = ["development", "staging", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Invalid environment: {v}. Must be one of {valid_envs}")
        return v.lower()

    @classmethod
    def load_from_file(cls, config_path: Union[str, Path]) -> "Config":
        """
        Load configuration from a YAML or JSON file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Config instance

        Raises:
            ConfigError: If configuration file cannot be loaded or parsed
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                if config_path.suffix.lower() in [".yaml", ".yml"]:
                    data = yaml.safe_load(f)
                elif config_path.suffix.lower() == ".json":
                    import json

                    data = json.load(f)
                else:
                    raise ConfigError(
                        f"Unsupported configuration file format: {config_path.suffix}"
                    )

            return cls(**data)

        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse YAML configuration: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to load configuration: {e}")

    @classmethod
    def load_from_env(cls) -> "Config":
        """
        Load configuration from environment variables.

        Returns:
            Config instance with values from environment variables
        """
        config_data = {}

        # Basic settings
        if os.getenv("OPENAGENT_APP_NAME"):
            config_data["app_name"] = os.getenv("OPENAGENT_APP_NAME")
        if os.getenv("OPENAGENT_DEBUG"):
            config_data["debug"] = os.getenv("OPENAGENT_DEBUG").lower() == "true"
        if os.getenv("OPENAGENT_ENVIRONMENT"):
            config_data["environment"] = os.getenv("OPENAGENT_ENVIRONMENT")

        # Database settings
        database_config = {}
        if os.getenv("DATABASE_URL"):
            database_config["url"] = os.getenv("DATABASE_URL")
        if database_config:
            config_data["database"] = database_config

        # Server settings
        server_config = {}
        if os.getenv("HOST"):
            server_config["host"] = os.getenv("HOST")
        if os.getenv("PORT"):
            server_config["port"] = int(os.getenv("PORT"))
        if server_config:
            config_data["server"] = server_config

        # Logging settings
        logging_config = {}
        if os.getenv("LOG_LEVEL"):
            logging_config["level"] = os.getenv("LOG_LEVEL")
        if os.getenv("LOG_FILE"):
            logging_config["file"] = os.getenv("LOG_FILE")
        if logging_config:
            config_data["logging"] = logging_config

        return cls(**config_data)

    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """
        Save configuration to a YAML file.

        Args:
            config_path: Path where to save the configuration

        Raises:
            ConfigError: If configuration cannot be saved
        """
        config_path = Path(config_path)

        try:
            # Create directory if it doesn't exist
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, "w") as f:
                yaml.dump(self.dict(), f, default_flow_style=False, indent=2)

        except Exception as e:
            raise ConfigError(f"Failed to save configuration: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.

        Args:
            key: Configuration key (supports dot notation, e.g., 'database.url')
            default: Default value if key is not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self.dict()

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by key.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set

        Raises:
            ConfigError: If key cannot be set
        """
        keys = key.split(".")

        if len(keys) == 1:
            if hasattr(self, keys[0]):
                setattr(self, keys[0], value)
            else:
                self.custom[keys[0]] = value
        else:
            # Handle nested keys
            current = self
            for k in keys[:-1]:
                if hasattr(current, k):
                    current = getattr(current, k)
                else:
                    raise ConfigError(f"Cannot set nested key: {key}")

            if hasattr(current, keys[-1]):
                setattr(current, keys[-1], value)
            else:
                raise ConfigError(f"Invalid configuration key: {key}")


# Global configuration instance
config = Config()
