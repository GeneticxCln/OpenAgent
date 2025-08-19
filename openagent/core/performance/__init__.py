"""
Performance optimization module for OpenAgent.

This module provides comprehensive performance improvements including:
- Model preloading and caching
- Concurrent work queue management
- Memory optimization and monitoring
- Resource usage tracking and scaling decisions
"""

from .memory_manager import MemoryManager, MemoryOptimizer, get_memory_manager
from .model_cache import ModelCache, ModelCacheManager, get_model_cache
from .resource_monitor import ResourceMetrics, ResourceMonitor, get_resource_monitor
from .work_queue import QueueManager, RequestPriority, WorkQueue, get_work_queue

__all__ = [
    # Model caching
    "ModelCache",
    "ModelCacheManager",
    "get_model_cache",
    # Work queue
    "WorkQueue",
    "RequestPriority",
    "QueueManager",
    "get_work_queue",
    # Memory management
    "MemoryManager",
    "MemoryOptimizer",
    "get_memory_manager",
    # Resource monitoring
    "ResourceMonitor",
    "ResourceMetrics",
    "get_resource_monitor",
]
