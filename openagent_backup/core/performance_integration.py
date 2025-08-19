"""
Performance integration module for OpenAgent server.

This module provides the main integration layer that coordinates all
performance optimization components.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional

from openagent.core.performance import (
    MemoryManager,
    MemoryStats,
    ModelCache,
    QueueMetrics,
    RequestPriority,
    ResourceAlert,
    ResourceMetrics,
    ResourceMonitor,
    ScalingRecommendation,
    WorkQueue,
    get_memory_manager,
    get_model_cache,
    get_resource_monitor,
    get_work_queue,
)
from openagent.core.performance_config import (
    PerformanceConfigurationManager,
    get_performance_config,
)

logger = logging.getLogger(__name__)


class PerformanceManager:
    """
    Central performance management coordinator.

    This class manages all performance optimization components and provides
    a unified interface for performance monitoring and optimization.
    """

    def __init__(self, config: Optional[PerformanceConfigurationManager] = None):
        """Initialize performance manager."""
        self.config = config or get_performance_config()

        # Performance components
        self._model_cache: Optional[ModelCache] = None
        self._work_queue: Optional[WorkQueue] = None
        self._memory_manager: Optional[MemoryManager] = None
        self._resource_monitor: Optional[ResourceMonitor] = None

        # Callbacks
        self._alert_callbacks: List[Callable] = []
        self._scaling_callbacks: List[Callable] = []

        # State
        self._initialized = False
        self._background_tasks: List[asyncio.Task] = []

        logger.info("Performance manager initialized")

    async def initialize(self) -> None:
        """Initialize all performance components."""
        if self._initialized:
            logger.warning("Performance manager already initialized")
            return

        logger.info("Initializing performance components...")

        try:
            # Initialize model cache
            if self.config.is_model_cache_enabled():
                cache_config = self.config.get_model_cache_config()
                self._model_cache = get_model_cache(**cache_config)
                logger.info("Model cache initialized")

            # Initialize work queue
            if self.config.is_work_queue_enabled():
                queue_config = self.config.get_work_queue_config()
                self._work_queue = get_work_queue(**queue_config)
                logger.info("Work queue initialized")

            # Initialize memory manager
            if self.config.is_memory_management_enabled():
                memory_config = self.config.get_memory_config()
                self._memory_manager = get_memory_manager(**memory_config)
                logger.info("Memory manager initialized")

            # Initialize resource monitor
            if self.config.is_resource_monitoring_enabled():
                monitor_config = self.config.get_resource_monitor_config()
                self._resource_monitor = get_resource_monitor(**monitor_config)

                # Register callbacks for alerts and scaling recommendations
                self._resource_monitor.add_alert_callback(self._handle_resource_alert)
                self._resource_monitor.add_scaling_callback(
                    self._handle_scaling_recommendation
                )

                logger.info("Resource monitor initialized")

            self._initialized = True
            logger.info("All performance components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize performance components: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown all performance components."""
        if not self._initialized:
            return

        logger.info("Shutting down performance components...")

        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()

        try:
            # Shutdown components
            if self._resource_monitor:
                await self._resource_monitor.shutdown()
                logger.info("Resource monitor shut down")

            if self._memory_manager:
                await self._memory_manager.shutdown()
                logger.info("Memory manager shut down")

            if self._work_queue:
                await self._work_queue.shutdown()
                logger.info("Work queue shut down")

            if self._model_cache:
                await self._model_cache.shutdown()
                logger.info("Model cache shut down")

            self._initialized = False
            logger.info("All performance components shut down successfully")

        except Exception as e:
            logger.error(f"Error during performance components shutdown: {e}")

    @asynccontextmanager
    async def lifespan(self):
        """Context manager for performance component lifespan."""
        try:
            await self.initialize()
            yield self
        finally:
            await self.shutdown()

    def add_alert_callback(self, callback: Callable[[ResourceAlert], None]) -> None:
        """Add callback for resource alerts."""
        self._alert_callbacks.append(callback)

    def add_scaling_callback(
        self, callback: Callable[[ScalingRecommendation], None]
    ) -> None:
        """Add callback for scaling recommendations."""
        self._scaling_callbacks.append(callback)

    async def _handle_resource_alert(self, alert: ResourceAlert) -> None:
        """Handle resource alerts."""
        logger.log(
            logging.CRITICAL if alert.level.value == "critical" else logging.WARNING,
            f"Resource alert: {alert.message}",
        )

        # Call registered callbacks
        for callback in self._alert_callbacks:
            try:
                (
                    await callback(alert)
                    if asyncio.iscoroutinefunction(callback)
                    else callback(alert)
                )
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    async def _handle_scaling_recommendation(
        self, recommendation: ScalingRecommendation
    ) -> None:
        """Handle scaling recommendations."""
        logger.info(
            f"Scaling recommendation: {recommendation.action.value} {recommendation.component} - {recommendation.reason}"
        )

        # Call registered callbacks
        for callback in self._scaling_callbacks:
            try:
                (
                    await callback(recommendation)
                    if asyncio.iscoroutinefunction(callback)
                    else callback(recommendation)
                )
            except Exception as e:
                logger.error(f"Scaling callback failed: {e}")

    # Model cache methods
    def get_model_cache(self) -> Optional[ModelCache]:
        """Get model cache instance."""
        return self._model_cache

    async def cache_model(
        self, model_name: str, model: Any, ttl: Optional[int] = None
    ) -> bool:
        """Cache a model."""
        if not self._model_cache:
            return False
        return await self._model_cache.put(model_name, model, ttl)

    async def get_cached_model(self, model_name: str) -> Optional[Any]:
        """Get cached model."""
        if not self._model_cache:
            return None
        return await self._model_cache.get(model_name)

    async def preload_models(self, model_names: List[str]) -> None:
        """Preload models into cache."""
        if not self._model_cache:
            return
        await self._model_cache.preload_models(model_names)

    # Work queue methods
    def get_work_queue(self) -> Optional[WorkQueue]:
        """Get work queue instance."""
        return self._work_queue

    async def submit_task(
        self,
        task_func: Callable,
        *args,
        priority: RequestPriority = RequestPriority.NORMAL,
        user_id: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Any:
        """Submit a task to the work queue."""
        if not self._work_queue:
            # Fallback to direct execution
            if asyncio.iscoroutinefunction(task_func):
                return await task_func(*args, **kwargs)
            else:
                return task_func(*args, **kwargs)

        return await self._work_queue.submit(
            task_func,
            *args,
            priority=priority,
            user_id=user_id,
            timeout=timeout,
            **kwargs,
        )

    async def submit_task_nowait(
        self,
        task_func: Callable,
        *args,
        priority: RequestPriority = RequestPriority.NORMAL,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Submit a task without waiting for result."""
        if not self._work_queue:
            # Create background task
            task = asyncio.create_task(task_func(*args, **kwargs))
            self._background_tasks.append(task)
            return str(id(task))

        return await self._work_queue.submit_nowait(
            task_func, *args, priority=priority, user_id=user_id, **kwargs
        )

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get work queue status."""
        if not self._work_queue:
            return {"error": "Work queue not available"}
        return await self._work_queue.get_queue_status()

    def get_queue_metrics(self) -> Optional[QueueMetrics]:
        """Get work queue metrics."""
        if not self._work_queue:
            return None
        return self._work_queue.get_metrics()

    # Memory management methods
    def get_memory_manager(self) -> Optional[MemoryManager]:
        """Get memory manager instance."""
        return self._memory_manager

    async def optimize_memory(self, force: bool = False) -> Dict[str, Any]:
        """Optimize memory usage."""
        if not self._memory_manager:
            return {"error": "Memory manager not available"}
        return await self._memory_manager.optimize(force=force)

    async def get_memory_stats(self) -> Optional[MemoryStats]:
        """Get memory usage statistics."""
        if not self._memory_manager:
            return None
        return await self._memory_manager.monitor_memory()

    async def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report."""
        if not self._memory_manager:
            return {"error": "Memory manager not available"}
        return await self._memory_manager.get_report()

    def track_object(self, name: str, obj: Any) -> None:
        """Track an object for memory management."""
        if self._memory_manager:
            self._memory_manager.track_object(name, obj)

    # Resource monitoring methods
    def get_resource_monitor(self) -> Optional[ResourceMonitor]:
        """Get resource monitor instance."""
        return self._resource_monitor

    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get current system metrics."""
        if not self._resource_monitor:
            return None
        return self._resource_monitor.get_current_metrics()

    def get_metrics_history(
        self, duration: Optional[float] = None
    ) -> List[ResourceMetrics]:
        """Get historical metrics."""
        if not self._resource_monitor:
            return []
        return self._resource_monitor.get_metrics_history(duration)

    def get_recent_alerts(self, duration: float = 3600) -> List[ResourceAlert]:
        """Get recent alerts."""
        if not self._resource_monitor:
            return []
        return self._resource_monitor.get_recent_alerts(duration)

    def get_scaling_recommendations(
        self, duration: float = 3600
    ) -> List[ScalingRecommendation]:
        """Get recent scaling recommendations."""
        if not self._resource_monitor:
            return []
        return self._resource_monitor.get_scaling_recommendations(duration)

    async def analyze_performance_trends(
        self, duration: float = 3600
    ) -> Dict[str, Any]:
        """Analyze performance trends."""
        if not self._resource_monitor:
            return {"error": "Resource monitor not available"}
        return await self._resource_monitor.analyze_performance_trends(duration)

    async def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report."""
        if not self._resource_monitor:
            return {"error": "Resource monitor not available"}
        return await self._resource_monitor.get_system_health_report()

    # Combined performance status
    async def get_performance_status(self) -> Dict[str, Any]:
        """Get comprehensive performance status."""
        status = {
            "initialized": self._initialized,
            "components": {
                "model_cache": {"enabled": self._model_cache is not None},
                "work_queue": {"enabled": self._work_queue is not None},
                "memory_manager": {"enabled": self._memory_manager is not None},
                "resource_monitor": {"enabled": self._resource_monitor is not None},
            },
        }

        # Add detailed status for each component
        if self._model_cache:
            try:
                cache_stats = await self._model_cache.get_stats()
                status["components"]["model_cache"]["stats"] = cache_stats
            except Exception as e:
                status["components"]["model_cache"]["error"] = str(e)

        if self._work_queue:
            try:
                queue_status = await self.get_queue_status()
                status["components"]["work_queue"]["status"] = queue_status
                queue_metrics = self.get_queue_metrics()
                if queue_metrics:
                    status["components"]["work_queue"]["metrics"] = {
                        "total_requests": queue_metrics.total_requests,
                        "completed_requests": queue_metrics.completed_requests,
                        "failed_requests": queue_metrics.failed_requests,
                        "success_rate": queue_metrics.success_rate,
                        "avg_processing_time": queue_metrics.avg_processing_time,
                        "throughput_per_minute": queue_metrics.throughput_per_minute,
                    }
            except Exception as e:
                status["components"]["work_queue"]["error"] = str(e)

        if self._memory_manager:
            try:
                memory_stats = await self.get_memory_stats()
                if memory_stats:
                    status["components"]["memory_manager"]["stats"] = {
                        "system_percent": memory_stats.system_percent,
                        "process_memory_mb": memory_stats.process_memory_mb,
                        "gpu_memory_used_mb": memory_stats.gpu_memory_used_mb,
                    }
            except Exception as e:
                status["components"]["memory_manager"]["error"] = str(e)

        if self._resource_monitor:
            try:
                health_report = await self.get_system_health_report()
                status["components"]["resource_monitor"]["health"] = health_report
            except Exception as e:
                status["components"]["resource_monitor"]["error"] = str(e)

        return status

    async def run_performance_optimization(self) -> Dict[str, Any]:
        """Run comprehensive performance optimization."""
        logger.info("Running performance optimization...")

        results = {"timestamp": asyncio.get_event_loop().time(), "optimizations": []}

        # Memory optimization
        if self._memory_manager:
            try:
                memory_result = await self.optimize_memory()
                results["optimizations"].append(
                    {"component": "memory", "result": memory_result}
                )
            except Exception as e:
                results["optimizations"].append(
                    {"component": "memory", "error": str(e)}
                )

        # Model cache cleanup
        if self._model_cache:
            try:
                await self._model_cache.cleanup_expired()
                results["optimizations"].append(
                    {
                        "component": "model_cache",
                        "result": "Cache cleaned up successfully",
                    }
                )
            except Exception as e:
                results["optimizations"].append(
                    {"component": "model_cache", "error": str(e)}
                )

        logger.info(
            f"Performance optimization completed: {len(results['optimizations'])} operations"
        )
        return results


# Global performance manager
_performance_manager: Optional[PerformanceManager] = None


def get_performance_manager(
    config: Optional[PerformanceConfigurationManager] = None,
) -> PerformanceManager:
    """Get or create the global performance manager."""
    global _performance_manager

    if _performance_manager is None:
        _performance_manager = PerformanceManager(config)

    return _performance_manager


async def initialize_performance_manager(
    config: Optional[PerformanceConfigurationManager] = None,
) -> PerformanceManager:
    """Initialize the global performance manager."""
    global _performance_manager

    _performance_manager = PerformanceManager(config)
    await _performance_manager.initialize()

    return _performance_manager


async def shutdown_performance_manager() -> None:
    """Shutdown the global performance manager."""
    global _performance_manager

    if _performance_manager:
        await _performance_manager.shutdown()
        _performance_manager = None
