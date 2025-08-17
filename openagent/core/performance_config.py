"""
Performance configuration integration for OpenAgent.

This module extends the existing configuration system to include
performance optimization settings.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from openagent.core.config import Config
from openagent.core.performance.memory_manager import MemoryThresholds
from openagent.core.performance.resource_monitor import ResourceThresholds
from openagent.core.performance.work_queue import RequestPriority, UserLimits


class ModelCacheConfig(BaseModel):
    """Model cache configuration."""
    enabled: bool = Field(True, description="Enable model caching")
    max_size: int = Field(5, description="Maximum number of models to cache")
    default_ttl: int = Field(1800, description="Default cache TTL in seconds (30 minutes)")
    memory_threshold: float = Field(0.8, description="Memory threshold for cache eviction")
    preload_models: list = Field(
        default_factory=lambda: ["tiny-llama"], 
        description="Models to preload on startup"
    )
    background_warming: bool = Field(True, description="Enable background model warming")


class WorkQueueConfig(BaseModel):
    """Work queue configuration."""
    enabled: bool = Field(True, description="Enable concurrent work queue")
    max_workers: int = Field(10, description="Maximum number of concurrent workers")
    max_queue_size: int = Field(1000, description="Maximum total queue size")
    default_timeout: float = Field(60.0, description="Default request timeout in seconds")
    enable_retries: bool = Field(True, description="Enable automatic retries")
    cleanup_interval: float = Field(300.0, description="Cleanup interval in seconds")
    
    # Per-user limits
    user_max_concurrent: int = Field(5, description="Max concurrent requests per user")
    user_max_queue_size: int = Field(20, description="Max queue size per user")
    user_rate_limit: int = Field(100, description="Rate limit per user per minute")
    user_max_processing_time: float = Field(300.0, description="Max processing time per user")


class MemoryManagementConfig(BaseModel):
    """Memory management configuration."""
    enabled: bool = Field(True, description="Enable memory management")
    gc_frequency: float = Field(300.0, description="Garbage collection frequency in seconds")
    enable_gpu_monitoring: bool = Field(True, description="Enable GPU memory monitoring")
    auto_cleanup: bool = Field(True, description="Enable automatic cleanup")
    
    # Memory thresholds
    system_warning: float = Field(80.0, description="System memory warning threshold (%)")
    system_critical: float = Field(90.0, description="System memory critical threshold (%)")
    process_warning: int = Field(2 * 1024 * 1024 * 1024, description="Process memory warning (bytes)")
    process_critical: int = Field(4 * 1024 * 1024 * 1024, description="Process memory critical (bytes)")
    gpu_warning: float = Field(80.0, description="GPU memory warning threshold (%)")
    gpu_critical: float = Field(90.0, description="GPU memory critical threshold (%)")


class ResourceMonitoringConfig(BaseModel):
    """Resource monitoring configuration."""
    enabled: bool = Field(True, description="Enable resource monitoring")
    monitoring_interval: float = Field(5.0, description="Monitoring interval in seconds")
    history_size: int = Field(1440, description="Number of historical metrics to keep")
    enable_gpu_monitoring: bool = Field(True, description="Enable GPU monitoring")
    enable_alerting: bool = Field(True, description="Enable alert generation")
    
    # Alert thresholds
    cpu_warning: float = Field(70.0, description="CPU warning threshold (%)")
    cpu_critical: float = Field(85.0, description="CPU critical threshold (%)")
    memory_warning: float = Field(75.0, description="Memory warning threshold (%)")
    memory_critical: float = Field(90.0, description="Memory critical threshold (%)")
    gpu_memory_warning: float = Field(75.0, description="GPU memory warning threshold (%)")
    gpu_memory_critical: float = Field(90.0, description="GPU memory critical threshold (%)")
    gpu_util_warning: float = Field(80.0, description="GPU utilization warning threshold (%)")
    gpu_util_critical: float = Field(95.0, description="GPU utilization critical threshold (%)")
    disk_warning: float = Field(80.0, description="Disk warning threshold (%)")
    disk_critical: float = Field(90.0, description="Disk critical threshold (%)")
    temperature_warning: float = Field(80.0, description="Temperature warning threshold (°C)")
    temperature_critical: float = Field(85.0, description="Temperature critical threshold (°C)")


class PerformanceConfig(BaseModel):
    """Complete performance configuration."""
    model_cache: ModelCacheConfig = Field(default_factory=ModelCacheConfig)
    work_queue: WorkQueueConfig = Field(default_factory=WorkQueueConfig)
    memory_management: MemoryManagementConfig = Field(default_factory=MemoryManagementConfig)
    resource_monitoring: ResourceMonitoringConfig = Field(default_factory=ResourceMonitoringConfig)


# Extend the main Config class
class ExtendedConfig(Config):
    """Extended configuration with performance settings."""
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)


class PerformanceConfigurationManager:
    """Manager for performance configuration integration."""
    
    def __init__(self, config: Optional[ExtendedConfig] = None):
        """Initialize performance configuration manager."""
        self.config = config or ExtendedConfig()
    
    def get_model_cache_config(self) -> Dict[str, Any]:
        """Get model cache configuration parameters."""
        cache_config = self.config.performance.model_cache
        return {
            "max_size": cache_config.max_size,
            "default_ttl": cache_config.default_ttl,
            "memory_threshold": cache_config.memory_threshold,
            "preload_models": cache_config.preload_models,
            "background_warming": cache_config.background_warming,
        }
    
    def get_work_queue_config(self) -> Dict[str, Any]:
        """Get work queue configuration parameters."""
        queue_config = self.config.performance.work_queue
        return {
            "max_workers": queue_config.max_workers,
            "max_queue_size": queue_config.max_queue_size,
            "default_timeout": queue_config.default_timeout,
            "enable_retries": queue_config.enable_retries,
            "cleanup_interval": queue_config.cleanup_interval,
        }
    
    def get_user_limits_config(self) -> UserLimits:
        """Get user limits configuration."""
        queue_config = self.config.performance.work_queue
        return UserLimits(
            max_concurrent=queue_config.user_max_concurrent,
            max_queue_size=queue_config.user_max_queue_size,
            rate_limit_per_minute=queue_config.user_rate_limit,
            max_processing_time=queue_config.user_max_processing_time,
        )
    
    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory management configuration parameters."""
        memory_config = self.config.performance.memory_management
        return {
            "gc_frequency": memory_config.gc_frequency,
            "enable_gpu_monitoring": memory_config.enable_gpu_monitoring,
            "auto_cleanup": memory_config.auto_cleanup,
            "memory_thresholds": MemoryThresholds(
                system_warning=memory_config.system_warning,
                system_critical=memory_config.system_critical,
                process_warning=memory_config.process_warning,
                process_critical=memory_config.process_critical,
                gpu_warning=memory_config.gpu_warning,
                gpu_critical=memory_config.gpu_critical,
            )
        }
    
    def get_resource_monitor_config(self) -> Dict[str, Any]:
        """Get resource monitoring configuration parameters."""
        monitor_config = self.config.performance.resource_monitoring
        return {
            "monitoring_interval": monitor_config.monitoring_interval,
            "history_size": monitor_config.history_size,
            "enable_gpu_monitoring": monitor_config.enable_gpu_monitoring,
            "enable_alerting": monitor_config.enable_alerting,
            "thresholds": ResourceThresholds(
                cpu_warning=monitor_config.cpu_warning,
                cpu_critical=monitor_config.cpu_critical,
                memory_warning=monitor_config.memory_warning,
                memory_critical=monitor_config.memory_critical,
                gpu_memory_warning=monitor_config.gpu_memory_warning,
                gpu_memory_critical=monitor_config.gpu_memory_critical,
                gpu_util_warning=monitor_config.gpu_util_warning,
                gpu_util_critical=monitor_config.gpu_util_critical,
                disk_warning=monitor_config.disk_warning,
                disk_critical=monitor_config.disk_critical,
                temperature_warning=monitor_config.temperature_warning,
                temperature_critical=monitor_config.temperature_critical,
            )
        }
    
    def is_model_cache_enabled(self) -> bool:
        """Check if model cache is enabled."""
        return self.config.performance.model_cache.enabled
    
    def is_work_queue_enabled(self) -> bool:
        """Check if work queue is enabled."""
        return self.config.performance.work_queue.enabled
    
    def is_memory_management_enabled(self) -> bool:
        """Check if memory management is enabled."""
        return self.config.performance.memory_management.enabled
    
    def is_resource_monitoring_enabled(self) -> bool:
        """Check if resource monitoring is enabled."""
        return self.config.performance.resource_monitoring.enabled
    
    def update_from_env(self) -> None:
        """Update performance configuration from environment variables."""
        import os
        
        # Model cache settings
        if os.getenv("OPENAGENT_MODEL_CACHE_ENABLED"):
            self.config.performance.model_cache.enabled = os.getenv("OPENAGENT_MODEL_CACHE_ENABLED").lower() == "true"
        
        if os.getenv("OPENAGENT_MODEL_CACHE_SIZE"):
            self.config.performance.model_cache.max_size = int(os.getenv("OPENAGENT_MODEL_CACHE_SIZE"))
        
        if os.getenv("OPENAGENT_MODEL_CACHE_TTL"):
            self.config.performance.model_cache.default_ttl = int(os.getenv("OPENAGENT_MODEL_CACHE_TTL"))
        
        # Work queue settings
        if os.getenv("OPENAGENT_WORK_QUEUE_ENABLED"):
            self.config.performance.work_queue.enabled = os.getenv("OPENAGENT_WORK_QUEUE_ENABLED").lower() == "true"
        
        if os.getenv("OPENAGENT_WORK_QUEUE_WORKERS"):
            self.config.performance.work_queue.max_workers = int(os.getenv("OPENAGENT_WORK_QUEUE_WORKERS"))
        
        if os.getenv("OPENAGENT_WORK_QUEUE_SIZE"):
            self.config.performance.work_queue.max_queue_size = int(os.getenv("OPENAGENT_WORK_QUEUE_SIZE"))
        
        # Memory management settings
        if os.getenv("OPENAGENT_MEMORY_MANAGEMENT_ENABLED"):
            self.config.performance.memory_management.enabled = os.getenv("OPENAGENT_MEMORY_MANAGEMENT_ENABLED").lower() == "true"
        
        if os.getenv("OPENAGENT_MEMORY_GC_FREQUENCY"):
            self.config.performance.memory_management.gc_frequency = float(os.getenv("OPENAGENT_MEMORY_GC_FREQUENCY"))
        
        if os.getenv("OPENAGENT_MEMORY_GPU_MONITORING"):
            self.config.performance.memory_management.enable_gpu_monitoring = os.getenv("OPENAGENT_MEMORY_GPU_MONITORING").lower() == "true"
        
        # Resource monitoring settings  
        if os.getenv("OPENAGENT_RESOURCE_MONITORING_ENABLED"):
            self.config.performance.resource_monitoring.enabled = os.getenv("OPENAGENT_RESOURCE_MONITORING_ENABLED").lower() == "true"
        
        if os.getenv("OPENAGENT_RESOURCE_MONITORING_INTERVAL"):
            self.config.performance.resource_monitoring.monitoring_interval = float(os.getenv("OPENAGENT_RESOURCE_MONITORING_INTERVAL"))
        
        if os.getenv("OPENAGENT_RESOURCE_GPU_MONITORING"):
            self.config.performance.resource_monitoring.enable_gpu_monitoring = os.getenv("OPENAGENT_RESOURCE_GPU_MONITORING").lower() == "true"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.config.dict()
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to file."""
        self.config.save_to_file(config_path)


# Global performance configuration manager
_performance_config_manager = None


def get_performance_config(config: Optional[ExtendedConfig] = None) -> PerformanceConfigurationManager:
    """Get or create the global performance configuration manager."""
    global _performance_config_manager
    
    if _performance_config_manager is None:
        _performance_config_manager = PerformanceConfigurationManager(config)
        _performance_config_manager.update_from_env()
    
    return _performance_config_manager


def initialize_performance_config(config: Optional[ExtendedConfig] = None) -> PerformanceConfigurationManager:
    """Initialize the performance configuration manager."""
    global _performance_config_manager
    
    _performance_config_manager = PerformanceConfigurationManager(config)
    _performance_config_manager.update_from_env()
    
    return _performance_config_manager
