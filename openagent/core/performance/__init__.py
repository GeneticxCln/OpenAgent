"""
Performance optimization module for OpenAgent.

This module provides comprehensive performance improvements including:
- Model preloading and caching
- Concurrent work queue management
- Memory optimization and monitoring
- Resource usage tracking and scaling decisions
"""

from .model_cache import ModelCache, ModelCacheManager, get_model_cache
from .work_queue import WorkQueue, RequestPriority, QueueManager, get_work_queue
from .memory_manager import MemoryManager, MemoryOptimizer, get_memory_manager
from .resource_monitor import ResourceMonitor, ResourceMetrics, get_resource_monitor

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
