"""
Performance optimization module for OpenAgent.

This module provides comprehensive performance improvements including:
- Model preloading and caching
- Concurrent work queue management
- Memory optimization and monitoring
- Resource usage tracking and scaling decisions
"""

from .memory_manager import (
    MemoryManager,
    MemoryOptimizer,
    MemoryStats,
    get_memory_manager,
)
from .model_cache import ModelCache, ModelCacheManager, get_model_cache
from .resource_monitor import (
    ResourceAlert,
    ResourceMetrics,
    ResourceMonitor,
    ResourceThresholds,
    ScalingRecommendation,
    get_resource_monitor,
)
from .work_queue import (
    QueueManager,
    QueueMetrics,
    RequestPriority,
    WorkQueue,
    get_work_queue,
)

__all__ = [
    # Model caching
    "ModelCache",
    "ModelCacheManager",
    "get_model_cache",
    # Work queue
    "WorkQueue",
    "RequestPriority",
    "QueueManager",
    "QueueMetrics",
    "get_work_queue",
    # Memory management
    "MemoryManager",
    "MemoryOptimizer",
    "MemoryStats",
    "get_memory_manager",
    # Resource monitoring
    "ResourceMonitor",
    "ResourceMetrics",
    "ResourceAlert",
    "ScalingRecommendation",
    "ResourceThresholds",
    "get_resource_monitor",
]
