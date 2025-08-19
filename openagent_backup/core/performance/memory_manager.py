"""
Memory management system for OpenAgent.

This module provides:
- Memory usage monitoring and optimization
- GPU memory tracking and management
- Automatic cleanup of unused resources
- Model quantization and memory reduction
"""

import asyncio
import gc
import logging
import time
import weakref
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple

import psutil

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import nvidia_ml_py as nvml

    nvml.nvmlInit()
    HAS_NVIDIA = True
except (ImportError, Exception):
    HAS_NVIDIA = False

from openagent.core.exceptions import AgentError

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    system_total: int = 0  # bytes
    system_available: int = 0  # bytes
    system_used: int = 0  # bytes
    system_percent: float = 0.0
    process_memory: int = 0  # bytes
    process_percent: float = 0.0
    gpu_memory_total: int = 0  # bytes
    gpu_memory_used: int = 0  # bytes
    gpu_memory_free: int = 0  # bytes
    gpu_count: int = 0

    @property
    def system_used_mb(self) -> float:
        """System memory used in MB."""
        return self.system_used / (1024 * 1024)

    @property
    def process_memory_mb(self) -> float:
        """Process memory in MB."""
        return self.process_memory / (1024 * 1024)

    @property
    def gpu_memory_used_mb(self) -> float:
        """GPU memory used in MB."""
        return self.gpu_memory_used / (1024 * 1024)


@dataclass
class MemoryThresholds:
    """Memory usage thresholds for optimization."""

    system_warning: float = 80.0  # %
    system_critical: float = 90.0  # %
    process_warning: int = 2 * 1024 * 1024 * 1024  # 2GB
    process_critical: int = 4 * 1024 * 1024 * 1024  # 4GB
    gpu_warning: float = 80.0  # %
    gpu_critical: float = 90.0  # %


class MemoryOptimizer:
    """
    Advanced memory optimization and cleanup system.

    Features:
    - Automatic garbage collection
    - Model quantization and compression
    - GPU memory management
    - Cache cleanup based on usage patterns
    - Memory leak detection
    """

    def __init__(
        self,
        gc_frequency: float = 300.0,  # 5 minutes
        enable_gpu_monitoring: bool = True,
        auto_cleanup: bool = True,
        memory_thresholds: Optional[MemoryThresholds] = None,
    ):
        """
        Initialize memory optimizer.

        Args:
            gc_frequency: Frequency of garbage collection in seconds
            enable_gpu_monitoring: Enable GPU memory monitoring
            auto_cleanup: Enable automatic cleanup when thresholds exceeded
            memory_thresholds: Custom memory thresholds
        """
        self.gc_frequency = gc_frequency
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.auto_cleanup = auto_cleanup
        self.thresholds = memory_thresholds or MemoryThresholds()

        # Tracking
        self._tracked_objects: Dict[str, weakref.ref] = {}
        self._memory_history: List[Tuple[float, MemoryStats]] = []
        self._cleanup_callbacks: List[Callable] = []
        self._lock = Lock()

        # Background tasks
        self._gc_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        logger.info(
            f"MemoryOptimizer initialized: gpu_monitoring={enable_gpu_monitoring}"
        )
        self._start_background_tasks()

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        # System memory
        system_memory = psutil.virtual_memory()

        # Process memory
        process = psutil.Process()
        process_info = process.memory_info()

        stats = MemoryStats(
            system_total=system_memory.total,
            system_available=system_memory.available,
            system_used=system_memory.used,
            system_percent=system_memory.percent,
            process_memory=process_info.rss,
            process_percent=process.memory_percent(),
        )

        # GPU memory (if available)
        if self.enable_gpu_monitoring and HAS_NVIDIA:
            try:
                device_count = nvml.nvmlDeviceGetCount()
                stats.gpu_count = device_count

                total_gpu_memory = 0
                used_gpu_memory = 0

                for i in range(device_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)

                    total_gpu_memory += mem_info.total
                    used_gpu_memory += mem_info.used

                stats.gpu_memory_total = total_gpu_memory
                stats.gpu_memory_used = used_gpu_memory
                stats.gpu_memory_free = total_gpu_memory - used_gpu_memory

            except Exception as e:
                logger.warning(f"Failed to get GPU memory stats: {e}")

        return stats

    async def optimize_memory(self, force: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive memory optimization.

        Args:
            force: Force optimization regardless of thresholds

        Returns:
            Optimization results and statistics
        """
        start_time = time.time()
        stats_before = self.get_memory_stats()

        optimization_actions = []

        # Check if optimization is needed
        needs_optimization = force or self._needs_optimization(stats_before)

        if not needs_optimization:
            return {
                "optimization_performed": False,
                "reason": "Memory usage within acceptable thresholds",
                "stats": stats_before,
            }

        logger.info("Starting memory optimization...")

        # 1. Cleanup tracked objects
        cleaned_objects = await self._cleanup_tracked_objects()
        if cleaned_objects > 0:
            optimization_actions.append(f"Cleaned {cleaned_objects} tracked objects")

        # 2. Force garbage collection
        collected = gc.collect()
        if collected > 0:
            optimization_actions.append(f"Collected {collected} objects via GC")

        # 3. Clear PyTorch cache (if available)
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
            optimization_actions.append("Cleared PyTorch CUDA cache")

        # 4. Run cleanup callbacks
        callback_results = await self._run_cleanup_callbacks()
        optimization_actions.extend(callback_results)

        # 5. Additional system-specific optimizations
        await self._system_specific_cleanup()
        optimization_actions.append("Performed system-specific cleanup")

        stats_after = self.get_memory_stats()
        optimization_time = time.time() - start_time

        # Calculate memory freed
        memory_freed = stats_before.process_memory - stats_after.process_memory
        gpu_memory_freed = stats_before.gpu_memory_used - stats_after.gpu_memory_used

        result = {
            "optimization_performed": True,
            "optimization_time": optimization_time,
            "actions": optimization_actions,
            "memory_freed_mb": memory_freed / (1024 * 1024),
            "gpu_memory_freed_mb": gpu_memory_freed / (1024 * 1024),
            "stats_before": stats_before,
            "stats_after": stats_after,
        }

        logger.info(
            f"Memory optimization completed in {optimization_time:.2f}s. "
            f"Freed {memory_freed / (1024 * 1024):.1f}MB RAM, "
            f"{gpu_memory_freed / (1024 * 1024):.1f}MB GPU"
        )

        return result

    def track_object(self, name: str, obj: Any) -> None:
        """Track an object for memory management."""
        with self._lock:

            def cleanup_callback(ref):
                with self._lock:
                    if name in self._tracked_objects:
                        del self._tracked_objects[name]
                logger.debug(f"Tracked object '{name}' was garbage collected")

            self._tracked_objects[name] = weakref.ref(obj, cleanup_callback)
            logger.debug(f"Now tracking object: {name}")

    def untrack_object(self, name: str) -> bool:
        """Stop tracking an object."""
        with self._lock:
            if name in self._tracked_objects:
                del self._tracked_objects[name]
                logger.debug(f"Stopped tracking object: {name}")
                return True
            return False

    def get_tracked_objects(self) -> List[str]:
        """Get list of currently tracked object names."""
        with self._lock:
            # Filter out dead references
            alive_objects = []
            dead_refs = []

            for name, ref in self._tracked_objects.items():
                if ref() is not None:
                    alive_objects.append(name)
                else:
                    dead_refs.append(name)

            # Clean up dead references
            for name in dead_refs:
                del self._tracked_objects[name]

            return alive_objects

    def add_cleanup_callback(self, callback: Callable) -> None:
        """Add a cleanup callback to be called during optimization."""
        if asyncio.iscoroutinefunction(callback):
            self._cleanup_callbacks.append(callback)
        else:
            # Wrap sync function in async
            async def async_wrapper():
                return callback()

            self._cleanup_callbacks.append(async_wrapper)

    def quantize_model(self, model: Any, method: str = "dynamic") -> Any:
        """
        Quantize a model to reduce memory usage.

        Args:
            model: Model to quantize (typically PyTorch)
            method: Quantization method ('dynamic', 'static', 'qat')

        Returns:
            Quantized model
        """
        if not HAS_TORCH:
            logger.warning("PyTorch not available, cannot quantize model")
            return model

        try:
            if method == "dynamic":
                # Dynamic quantization
                quantized_model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("Applied dynamic quantization to model")
                return quantized_model
            elif method == "static":
                # Static quantization (requires calibration data)
                logger.warning(
                    "Static quantization requires calibration - using dynamic instead"
                )
                return self.quantize_model(model, "dynamic")
            else:
                logger.warning(f"Unknown quantization method: {method}")
                return model

        except Exception as e:
            logger.error(f"Failed to quantize model: {e}")
            return model

    async def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report."""
        stats = self.get_memory_stats()
        tracked_objects = self.get_tracked_objects()

        # Memory history analysis
        history_summary = None
        if self._memory_history:
            recent_history = self._memory_history[-10:]  # Last 10 measurements
            history_summary = {
                "measurements": len(recent_history),
                "avg_system_percent": sum(s.system_percent for _, s in recent_history)
                / len(recent_history),
                "avg_process_mb": sum(s.process_memory_mb for _, s in recent_history)
                / len(recent_history),
                "trend": (
                    "increasing"
                    if recent_history[-1][1].process_memory
                    > recent_history[0][1].process_memory
                    else "stable/decreasing"
                ),
            }

        # Check for memory issues
        issues = []
        if stats.system_percent > self.thresholds.system_warning:
            issues.append(f"High system memory usage: {stats.system_percent:.1f}%")
        if stats.process_memory > self.thresholds.process_warning:
            issues.append(f"High process memory usage: {stats.process_memory_mb:.1f}MB")
        if stats.gpu_memory_total > 0:
            gpu_percent = (stats.gpu_memory_used / stats.gpu_memory_total) * 100
            if gpu_percent > self.thresholds.gpu_warning:
                issues.append(f"High GPU memory usage: {gpu_percent:.1f}%")

        return {
            "current_stats": stats,
            "tracked_objects": {
                "count": len(tracked_objects),
                "names": tracked_objects,
            },
            "history": history_summary,
            "issues": issues,
            "thresholds": self.thresholds,
        }

    async def shutdown(self) -> None:
        """Shutdown the memory optimizer."""
        logger.info("Shutting down memory optimizer...")

        self._shutdown_event.set()

        # Cancel background tasks
        if self._gc_task:
            self._gc_task.cancel()
        if self._monitoring_task:
            self._monitoring_task.cancel()

        # Final cleanup
        await self.optimize_memory(force=True)

        logger.info("Memory optimizer shutdown complete")

    def _needs_optimization(self, stats: MemoryStats) -> bool:
        """Check if memory optimization is needed."""
        # System memory check
        if stats.system_percent > self.thresholds.system_warning:
            return True

        # Process memory check
        if stats.process_memory > self.thresholds.process_warning:
            return True

        # GPU memory check
        if stats.gpu_memory_total > 0:
            gpu_percent = (stats.gpu_memory_used / stats.gpu_memory_total) * 100
            if gpu_percent > self.thresholds.gpu_warning:
                return True

        return False

    async def _cleanup_tracked_objects(self) -> int:
        """Clean up tracked objects that are no longer referenced."""
        with self._lock:
            dead_refs = []
            for name, ref in self._tracked_objects.items():
                if ref() is None:
                    dead_refs.append(name)

            for name in dead_refs:
                del self._tracked_objects[name]

            return len(dead_refs)

    async def _run_cleanup_callbacks(self) -> List[str]:
        """Run all cleanup callbacks."""
        results = []

        for callback in self._cleanup_callbacks:
            try:
                result = await callback()
                if result:
                    results.append(str(result))
            except Exception as e:
                logger.error(f"Cleanup callback failed: {e}")
                results.append(f"Callback failed: {e}")

        return results

    async def _system_specific_cleanup(self) -> None:
        """Perform system-specific memory cleanup."""
        try:
            # Force Python garbage collection with all generations
            for generation in range(3):
                collected = gc.collect(generation)
                logger.debug(
                    f"GC generation {generation}: collected {collected} objects"
                )

            # Clear weak references
            gc.collect()

        except Exception as e:
            logger.error(f"System cleanup failed: {e}")

    def _start_background_tasks(self) -> None:
        """Start background monitoring and cleanup tasks."""
        self._gc_task = asyncio.create_task(self._gc_loop())
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def _gc_loop(self) -> None:
        """Background garbage collection loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.gc_frequency)

                if self.auto_cleanup:
                    stats = self.get_memory_stats()
                    if self._needs_optimization(stats):
                        await self.optimize_memory()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in GC loop: {e}")

    async def _monitoring_loop(self) -> None:
        """Background memory monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Monitor every minute

                stats = self.get_memory_stats()
                timestamp = time.time()

                # Store in history (keep last 100 measurements)
                self._memory_history.append((timestamp, stats))
                if len(self._memory_history) > 100:
                    self._memory_history.pop(0)

                # Log warnings if thresholds exceeded
                if stats.system_percent > self.thresholds.system_critical:
                    logger.warning(
                        f"Critical system memory usage: {stats.system_percent:.1f}%"
                    )
                elif stats.system_percent > self.thresholds.system_warning:
                    logger.info(
                        f"High system memory usage: {stats.system_percent:.1f}%"
                    )

                if stats.process_memory > self.thresholds.process_critical:
                    logger.warning(
                        f"Critical process memory usage: {stats.process_memory_mb:.1f}MB"
                    )
                elif stats.process_memory > self.thresholds.process_warning:
                    logger.info(
                        f"High process memory usage: {stats.process_memory_mb:.1f}MB"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")


class MemoryManager:
    """High-level memory management interface."""

    def __init__(self, optimizer: Optional[MemoryOptimizer] = None):
        """Initialize memory manager."""
        self.optimizer = optimizer or MemoryOptimizer()

    async def monitor_memory(self) -> MemoryStats:
        """Get current memory statistics."""
        return self.optimizer.get_memory_stats()

    async def optimize(self, force: bool = False) -> Dict[str, Any]:
        """Optimize memory usage."""
        return await self.optimizer.optimize_memory(force=force)

    def track_object(self, name: str, obj: Any) -> None:
        """Track an object for memory management."""
        self.optimizer.track_object(name, obj)

    def add_cleanup_callback(self, callback: Callable) -> None:
        """Add a cleanup callback."""
        self.optimizer.add_cleanup_callback(callback)

    async def get_report(self) -> Dict[str, Any]:
        """Get memory usage report."""
        return await self.optimizer.get_memory_report()

    async def shutdown(self) -> None:
        """Shutdown memory manager."""
        await self.optimizer.shutdown()


class MemoryManagerSingleton:
    """Global memory manager."""

    def __init__(self):
        self._manager: Optional[MemoryManager] = None

    def initialize(self, **kwargs) -> MemoryManager:
        """Initialize the global memory manager."""
        if self._manager is None:
            optimizer = MemoryOptimizer(**kwargs)
            self._manager = MemoryManager(optimizer)
        return self._manager

    def get_manager(self) -> Optional[MemoryManager]:
        """Get the global memory manager."""
        return self._manager

    async def shutdown(self) -> None:
        """Shutdown the memory manager."""
        if self._manager:
            await self._manager.shutdown()
            self._manager = None


# Global memory manager
_memory_manager = MemoryManagerSingleton()


def get_memory_manager(**kwargs) -> MemoryManager:
    """Get or create the global memory manager."""
    manager = _memory_manager.get_manager()
    if manager is None:
        manager = _memory_manager.initialize(**kwargs)
    return manager
