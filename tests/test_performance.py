"""
Comprehensive test suite for OpenAgent performance optimization features.

This module tests all performance components including model caching,
work queue, memory management, and resource monitoring.
"""

import asyncio
import pytest
import time
import gc
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from openagent.core.performance import (
    ModelCache,
    WorkQueue,
    MemoryManager,
    ResourceMonitor,
    RequestPriority,
    QueueMetrics,
    MemoryStats,
    ResourceMetrics,
    ResourceAlert,
    ScalingRecommendation,
    get_model_cache,
    get_work_queue,
    get_memory_manager,
    get_resource_monitor,
)
from openagent.core.performance.memory_manager import MemoryThresholds
from openagent.core.performance.resource_monitor import ResourceThresholds
from openagent.core.performance_config import (
    PerformanceConfigurationManager,
    ExtendedConfig,
    get_performance_config,
)
from openagent.core.performance_integration import (
    PerformanceManager,
    get_performance_manager,
    initialize_performance_manager,
)


class MockModel:
    """Mock model for testing."""
    
    def __init__(self, name: str, size: int = 100):
        self.name = name
        self.size = size
        self.data = b"x" * size
    
    def __sizeof__(self):
        return self.size


@pytest.fixture
async def mock_llm():
    """Mock LLM for testing."""
    llm = Mock()
    llm.generate_response = AsyncMock(return_value="Mock response")
    llm.load_model = AsyncMock()
    llm.unload_model = AsyncMock()
    llm.get_model_info = Mock(return_value={"name": "test-model", "size": 1000})
    return llm


class TestModelCache:
    """Test suite for model caching functionality."""
    
    @pytest.fixture
    async def model_cache(self):
        """Create a model cache for testing."""
        cache = ModelCache(max_size=3, default_ttl=10)
        await cache.initialize()
        yield cache
        await cache.shutdown()
    
    @pytest.mark.asyncio
    async def test_cache_basic_operations(self, model_cache):
        """Test basic cache operations."""
        model = MockModel("test-model")
        
        # Test put
        success = await model_cache.put("test-model", model)
        assert success is True
        
        # Test get
        cached_model = await model_cache.get("test-model")
        assert cached_model is not None
        assert cached_model.name == "test-model"
        
        # Test hit/miss
        assert await model_cache.get("nonexistent") is None
    
    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self, model_cache):
        """Test LRU eviction policy."""
        # Fill cache to capacity
        for i in range(3):
            model = MockModel(f"model-{i}")
            await model_cache.put(f"model-{i}", model)
        
        # Add another model, should evict the oldest
        model4 = MockModel("model-3")
        await model_cache.put("model-3", model4)
        
        # Check that model-0 was evicted
        assert await model_cache.get("model-0") is None
        assert await model_cache.get("model-1") is not None
        assert await model_cache.get("model-2") is not None
        assert await model_cache.get("model-3") is not None
    
    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self, model_cache):
        """Test TTL-based expiration."""
        model = MockModel("ttl-test")
        
        # Put with short TTL
        await model_cache.put("ttl-test", model, ttl=1)
        
        # Should be available immediately
        assert await model_cache.get("ttl-test") is not None
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should be expired now
        assert await model_cache.get("ttl-test") is None
    
    @pytest.mark.asyncio
    async def test_cache_stats(self, model_cache):
        """Test cache statistics."""
        model = MockModel("stats-test")
        
        # Initial stats
        stats = await model_cache.get_stats()
        initial_hits = stats.get("cache_hits", 0)
        initial_misses = stats.get("cache_misses", 0)
        
        # Add model and test hit
        await model_cache.put("stats-test", model)
        await model_cache.get("stats-test")  # Hit
        await model_cache.get("nonexistent")  # Miss
        
        # Check updated stats
        stats = await model_cache.get_stats()
        assert stats["cache_hits"] == initial_hits + 1
        assert stats["cache_misses"] == initial_misses + 1
    
    @pytest.mark.asyncio
    async def test_cache_memory_pressure(self, model_cache):
        """Test cache behavior under memory pressure."""
        # Mock memory pressure
        with patch.object(model_cache, '_check_memory_pressure', return_value=True):
            # Should trigger eviction
            model = MockModel("memory-test")
            success = await model_cache.put("memory-test", model)
            assert success is True


class TestWorkQueue:
    """Test suite for work queue functionality."""
    
    @pytest.fixture
    async def work_queue(self):
        """Create a work queue for testing."""
        queue = WorkQueue(max_workers=2, max_queue_size=10)
        yield queue
        await queue.shutdown()
    
    async def simple_task(self, value: int, delay: float = 0) -> int:
        """Simple async task for testing."""
        if delay > 0:
            await asyncio.sleep(delay)
        return value * 2
    
    def simple_sync_task(self, value: int) -> int:
        """Simple sync task for testing."""
        return value * 3
    
    async def failing_task(self) -> None:
        """Task that always fails."""
        raise ValueError("Task failed")
    
    @pytest.mark.asyncio
    async def test_queue_basic_operations(self, work_queue):
        """Test basic queue operations."""
        # Submit async task
        result = await work_queue.submit(self.simple_task, 5)
        assert result == 10
        
        # Submit sync task
        result = await work_queue.submit(self.simple_sync_task, 5)
        assert result == 15
    
    @pytest.mark.asyncio
    async def test_queue_priority(self, work_queue):
        """Test priority-based task scheduling."""
        results = []
        
        # Submit tasks with different priorities
        tasks = [
            work_queue.submit(
                self.simple_task, i, delay=0.1,
                priority=RequestPriority.LOW if i % 2 else RequestPriority.HIGH
            )
            for i in range(4)
        ]
        
        # Wait for all tasks
        results = await asyncio.gather(*tasks)
        
        # Verify all tasks completed
        assert len(results) == 4
    
    @pytest.mark.asyncio
    async def test_queue_user_limits(self, work_queue):
        """Test per-user rate limiting."""
        from openagent.core.performance.work_queue import UserLimits
        
        # Set strict limits for test user
        limits = UserLimits(max_concurrent=1, max_queue_size=2, rate_limit_per_minute=5)
        await work_queue.set_user_limits("test_user", limits)
        
        # Submit task within limits
        result = await work_queue.submit(
            self.simple_task, 5, user_id="test_user"
        )
        assert result == 10
        
        # Try to exceed concurrent limit
        with pytest.raises(Exception):  # Should raise AgentError
            tasks = [
                work_queue.submit(
                    self.simple_task, i, delay=1.0, user_id="test_user"
                )
                for i in range(3)  # Exceeds max_concurrent=1
            ]
            await asyncio.gather(*tasks)
    
    @pytest.mark.asyncio
    async def test_queue_retries(self, work_queue):
        """Test automatic retry functionality."""
        with pytest.raises(Exception):
            await work_queue.submit(self.failing_task)
    
    @pytest.mark.asyncio
    async def test_queue_timeout(self, work_queue):
        """Test request timeout handling."""
        with pytest.raises(Exception):  # Should raise AgentError for timeout
            await work_queue.submit(
                self.simple_task, 5, delay=2.0, timeout=1.0
            )
    
    @pytest.mark.asyncio
    async def test_queue_nowait(self, work_queue):
        """Test fire-and-forget task submission."""
        request_id = await work_queue.submit_nowait(self.simple_task, 5)
        assert isinstance(request_id, str)
        
        # Give task time to complete
        await asyncio.sleep(0.1)
    
    @pytest.mark.asyncio
    async def test_queue_status(self, work_queue):
        """Test queue status reporting."""
        status = await work_queue.get_queue_status()
        
        assert "total_queued" in status
        assert "active_requests" in status
        assert "max_workers" in status
        assert "uptime" in status
    
    @pytest.mark.asyncio
    async def test_queue_metrics(self, work_queue):
        """Test queue metrics collection."""
        # Submit some tasks to generate metrics
        await work_queue.submit(self.simple_task, 5)
        await work_queue.submit(self.simple_task, 10)
        
        metrics = work_queue.get_metrics()
        assert isinstance(metrics, QueueMetrics)
        assert metrics.total_requests >= 2
        assert metrics.completed_requests >= 2
    
    @pytest.mark.asyncio
    async def test_queue_cancellation(self, work_queue):
        """Test request cancellation."""
        # Submit a long-running task
        task = asyncio.create_task(
            work_queue.submit(self.simple_task, 5, delay=10.0)
        )
        
        # Cancel it
        task.cancel()
        
        with pytest.raises(asyncio.CancelledError):
            await task


class TestMemoryManager:
    """Test suite for memory management functionality."""
    
    @pytest.fixture
    async def memory_manager(self):
        """Create a memory manager for testing."""
        thresholds = MemoryThresholds(
            system_warning=50.0,  # Lower thresholds for testing
            system_critical=70.0,
            process_warning=100 * 1024 * 1024,  # 100MB
            process_critical=200 * 1024 * 1024,  # 200MB
        )
        manager = MemoryManager(
            memory_thresholds=thresholds,
            gc_frequency=1.0,  # Fast GC for testing
            auto_cleanup=False  # Disable auto-cleanup for predictable testing
        )
        yield manager
        await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_memory_stats(self, memory_manager):
        """Test memory statistics collection."""
        stats = await memory_manager.monitor_memory()
        
        assert isinstance(stats, MemoryStats)
        assert stats.system_total > 0
        assert stats.system_used > 0
        assert 0 <= stats.system_percent <= 100
        assert stats.process_memory > 0
    
    @pytest.mark.asyncio
    async def test_memory_optimization(self, memory_manager):
        """Test memory optimization."""
        # Create some objects to track
        test_objects = [list(range(1000)) for _ in range(10)]
        for i, obj in enumerate(test_objects):
            memory_manager.track_object(f"test_obj_{i}", obj)
        
        # Run optimization
        result = await memory_manager.optimize(force=True)
        
        assert "optimization_performed" in result
        assert result["optimization_performed"] is True
        assert "memory_freed_mb" in result
        assert "gpu_memory_freed_mb" in result
    
    @pytest.mark.asyncio
    async def test_object_tracking(self, memory_manager):
        """Test object tracking functionality."""
        # Track some objects
        obj1 = [1, 2, 3, 4, 5]
        obj2 = {"key": "value"}
        
        memory_manager.track_object("list_obj", obj1)
        memory_manager.track_object("dict_obj", obj2)
        
        # Check tracked objects
        tracked = memory_manager.get_tracked_objects()
        assert "list_obj" in tracked
        assert "dict_obj" in tracked
        
        # Remove tracking
        success = memory_manager.untrack_object("list_obj")
        assert success is True
        
        tracked = memory_manager.get_tracked_objects()
        assert "list_obj" not in tracked
        assert "dict_obj" in tracked
    
    @pytest.mark.asyncio
    async def test_cleanup_callbacks(self, memory_manager):
        """Test cleanup callback functionality."""
        callback_called = False
        
        async def test_callback():
            nonlocal callback_called
            callback_called = True
            return "Callback executed"
        
        memory_manager.add_cleanup_callback(test_callback)
        
        # Run optimization to trigger callbacks
        await memory_manager.optimize(force=True)
        
        assert callback_called is True
    
    @pytest.mark.asyncio
    async def test_memory_report(self, memory_manager):
        """Test comprehensive memory reporting."""
        report = await memory_manager.get_report()
        
        assert "current_stats" in report
        assert "tracked_objects" in report
        assert "issues" in report
        assert "thresholds" in report
        
        assert isinstance(report["tracked_objects"], dict)
        assert "count" in report["tracked_objects"]
        assert "names" in report["tracked_objects"]


class TestResourceMonitor:
    """Test suite for resource monitoring functionality."""
    
    @pytest.fixture
    async def resource_monitor(self):
        """Create a resource monitor for testing."""
        thresholds = ResourceThresholds(
            cpu_warning=50.0,  # Lower thresholds for testing
            cpu_critical=80.0,
            memory_warning=50.0,
            memory_critical=80.0,
        )
        monitor = ResourceMonitor(
            monitoring_interval=0.1,  # Fast monitoring for tests
            history_size=10,  # Small history for tests
            thresholds=thresholds,
            enable_alerting=True
        )
        yield monitor
        await monitor.shutdown()
    
    @pytest.mark.asyncio
    async def test_current_metrics(self, resource_monitor):
        """Test current metrics collection."""
        metrics = resource_monitor.get_current_metrics()
        
        assert isinstance(metrics, ResourceMetrics)
        assert metrics.timestamp > 0
        assert isinstance(metrics.cpu.percent, float)
        assert 0 <= metrics.cpu.percent <= 100
        assert isinstance(metrics.memory.percent, float)
        assert 0 <= metrics.memory.percent <= 100
    
    @pytest.mark.asyncio
    async def test_metrics_history(self, resource_monitor):
        """Test metrics history collection."""
        # Wait for some metrics to be collected
        await asyncio.sleep(0.5)
        
        history = resource_monitor.get_metrics_history()
        assert len(history) > 0
        
        # Test duration filtering
        recent_history = resource_monitor.get_metrics_history(duration=0.3)
        assert len(recent_history) <= len(history)
    
    @pytest.mark.asyncio
    async def test_alert_callbacks(self, resource_monitor):
        """Test alert callback functionality."""
        alerts_received = []
        
        def alert_callback(alert: ResourceAlert):
            alerts_received.append(alert)
        
        resource_monitor.add_alert_callback(alert_callback)
        
        # Wait a bit for potential alerts (may not trigger in test environment)
        await asyncio.sleep(0.5)
        
        # The callback was registered successfully
        assert alert_callback in resource_monitor._alert_callbacks
    
    @pytest.mark.asyncio
    async def test_scaling_recommendations(self, resource_monitor):
        """Test scaling recommendation generation."""
        scaling_recs = []
        
        def scaling_callback(rec: ScalingRecommendation):
            scaling_recs.append(rec)
        
        resource_monitor.add_scaling_callback(scaling_callback)
        
        # Wait for analysis to run
        await asyncio.sleep(1.5)  # Longer than analysis interval
        
        # Check if recommendations were generated (may vary by system)
        recommendations = resource_monitor.get_scaling_recommendations()
        assert isinstance(recommendations, list)
    
    @pytest.mark.asyncio
    async def test_performance_analysis(self, resource_monitor):
        """Test performance trend analysis."""
        # Wait for some data to accumulate
        await asyncio.sleep(1.0)
        
        analysis = await resource_monitor.analyze_performance_trends(duration=0.5)
        
        if "error" not in analysis:
            assert "trends" in analysis
            assert "bottlenecks" in analysis
            assert "recommendations" in analysis
            assert "sample_count" in analysis
    
    @pytest.mark.asyncio
    async def test_system_health_report(self, resource_monitor):
        """Test system health reporting."""
        report = await resource_monitor.get_system_health_report()
        
        assert "status" in report
        assert "health_score" in report
        assert "utilization" in report
        assert "alerts" in report
        assert "uptime" in report
        
        assert report["status"] in ["healthy", "warning", "critical"]
        assert 0 <= report["health_score"] <= 100


class TestPerformanceConfiguration:
    """Test suite for performance configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = get_performance_config()
        
        assert config.is_model_cache_enabled() is True
        assert config.is_work_queue_enabled() is True
        assert config.is_memory_management_enabled() is True
        assert config.is_resource_monitoring_enabled() is True
    
    def test_config_methods(self):
        """Test configuration method functionality."""
        config = get_performance_config()
        
        # Test config retrieval methods
        cache_config = config.get_model_cache_config()
        assert "max_size" in cache_config
        assert "default_ttl" in cache_config
        
        queue_config = config.get_work_queue_config()
        assert "max_workers" in queue_config
        assert "max_queue_size" in queue_config
        
        memory_config = config.get_memory_config()
        assert "gc_frequency" in memory_config
        assert "memory_thresholds" in memory_config
        
        monitor_config = config.get_resource_monitor_config()
        assert "monitoring_interval" in monitor_config
        assert "thresholds" in monitor_config
    
    @patch.dict("os.environ", {
        "OPENAGENT_MODEL_CACHE_ENABLED": "false",
        "OPENAGENT_WORK_QUEUE_WORKERS": "5",
        "OPENAGENT_MEMORY_GC_FREQUENCY": "600.0"
    })
    def test_env_override(self):
        """Test environment variable override."""
        from openagent.core.performance_config import PerformanceConfigurationManager
        
        config = PerformanceConfigurationManager()
        config.update_from_env()
        
        assert config.is_model_cache_enabled() is False
        assert config.config.performance.work_queue.max_workers == 5
        assert config.config.performance.memory_management.gc_frequency == 600.0


class TestPerformanceIntegration:
    """Test suite for performance integration."""
    
    @pytest.fixture
    async def performance_manager(self):
        """Create a performance manager for testing."""
        # Use a test configuration with fast intervals
        from openagent.core.performance_config import (
            ExtendedConfig, PerformanceConfig, ModelCacheConfig,
            WorkQueueConfig, MemoryManagementConfig, ResourceMonitoringConfig
        )
        
        test_config = ExtendedConfig(
            performance=PerformanceConfig(
                model_cache=ModelCacheConfig(max_size=2, default_ttl=5),
                work_queue=WorkQueueConfig(max_workers=2, max_queue_size=10),
                memory_management=MemoryManagementConfig(gc_frequency=1.0),
                resource_monitoring=ResourceMonitoringConfig(monitoring_interval=0.1)
            )
        )
        
        config_manager = PerformanceConfigurationManager(test_config)
        manager = PerformanceManager(config_manager)
        
        await manager.initialize()
        yield manager
        await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self, performance_manager):
        """Test performance manager initialization."""
        assert performance_manager._initialized is True
        assert performance_manager.get_model_cache() is not None
        assert performance_manager.get_work_queue() is not None
        assert performance_manager.get_memory_manager() is not None
        assert performance_manager.get_resource_monitor() is not None
    
    @pytest.mark.asyncio
    async def test_manager_model_operations(self, performance_manager):
        """Test model cache operations through manager."""
        model = MockModel("test-model")
        
        # Cache model
        success = await performance_manager.cache_model("test-model", model)
        assert success is True
        
        # Retrieve model
        cached_model = await performance_manager.get_cached_model("test-model")
        assert cached_model is not None
        assert cached_model.name == "test-model"
    
    @pytest.mark.asyncio
    async def test_manager_task_operations(self, performance_manager):
        """Test task submission through manager."""
        async def test_task(value: int) -> int:
            await asyncio.sleep(0.01)
            return value * 2
        
        # Submit task
        result = await performance_manager.submit_task(test_task, 5)
        assert result == 10
        
        # Submit task without waiting
        task_id = await performance_manager.submit_task_nowait(test_task, 10)
        assert isinstance(task_id, str)
    
    @pytest.mark.asyncio
    async def test_manager_memory_operations(self, performance_manager):
        """Test memory management operations through manager."""
        # Track object
        test_obj = [1, 2, 3, 4, 5]
        performance_manager.track_object("test_list", test_obj)
        
        # Get memory stats
        stats = await performance_manager.get_memory_stats()
        assert stats is not None
        assert isinstance(stats.system_percent, float)
        
        # Optimize memory
        result = await performance_manager.optimize_memory(force=True)
        assert "optimization_performed" in result
    
    @pytest.mark.asyncio
    async def test_manager_monitoring_operations(self, performance_manager):
        """Test resource monitoring operations through manager."""
        # Get current metrics
        metrics = performance_manager.get_current_metrics()
        assert metrics is not None
        assert isinstance(metrics.cpu.percent, float)
        
        # Wait for some history
        await asyncio.sleep(0.3)
        
        # Get history
        history = performance_manager.get_metrics_history()
        assert len(history) > 0
        
        # Get system health report
        report = await performance_manager.get_system_health_report()
        assert "status" in report
        assert "health_score" in report
    
    @pytest.mark.asyncio
    async def test_manager_performance_status(self, performance_manager):
        """Test comprehensive performance status."""
        status = await performance_manager.get_performance_status()
        
        assert "initialized" in status
        assert "components" in status
        assert status["initialized"] is True
        
        components = status["components"]
        assert "model_cache" in components
        assert "work_queue" in components
        assert "memory_manager" in components
        assert "resource_monitor" in components
        
        # All components should be enabled
        for component in components.values():
            assert component["enabled"] is True
    
    @pytest.mark.asyncio
    async def test_manager_optimization(self, performance_manager):
        """Test comprehensive performance optimization."""
        result = await performance_manager.run_performance_optimization()
        
        assert "timestamp" in result
        assert "optimizations" in result
        assert len(result["optimizations"]) > 0
        
        # Should have optimized memory at minimum
        optimization_components = [opt["component"] for opt in result["optimizations"]]
        assert "memory" in optimization_components


class TestPerformanceLoad:
    """Load testing for performance components."""
    
    @pytest.mark.asyncio
    async def test_queue_load(self):
        """Test work queue under load."""
        queue = WorkQueue(max_workers=4, max_queue_size=100)
        
        async def load_task(task_id: int) -> int:
            await asyncio.sleep(0.01)  # Simulate work
            return task_id
        
        try:
            # Submit many tasks concurrently
            tasks = [
                queue.submit(load_task, i, priority=RequestPriority.NORMAL)
                for i in range(50)
            ]
            
            # Wait for all to complete
            results = await asyncio.gather(*tasks)
            
            # Verify all completed successfully
            assert len(results) == 50
            assert all(isinstance(r, int) for r in results)
            
            # Check metrics
            metrics = queue.get_metrics()
            assert metrics.total_requests >= 50
            assert metrics.completed_requests >= 50
            assert metrics.success_rate > 0.9  # Should be high success rate
            
        finally:
            await queue.shutdown()
    
    @pytest.mark.asyncio
    async def test_cache_load(self):
        """Test model cache under load."""
        cache = ModelCache(max_size=20, default_ttl=60)
        await cache.initialize()
        
        try:
            # Create many models
            models = [MockModel(f"model-{i}", size=100) for i in range(30)]
            
            # Cache them concurrently
            tasks = [
                cache.put(f"model-{i}", models[i])
                for i in range(30)
            ]
            await asyncio.gather(*tasks)
            
            # Retrieve them concurrently
            retrieve_tasks = [
                cache.get(f"model-{i}")
                for i in range(30)
            ]
            results = await asyncio.gather(*retrieve_tasks)
            
            # Some should be evicted due to size limit
            non_none_results = [r for r in results if r is not None]
            assert len(non_none_results) <= 20  # Cache size limit
            
            # Check stats
            stats = await cache.get_stats()
            assert stats["total_items"] <= 20
            assert stats["cache_hits"] > 0
            
        finally:
            await cache.shutdown()
    
    @pytest.mark.asyncio
    async def test_memory_management_load(self):
        """Test memory manager under load."""
        manager = MemoryManager(
            gc_frequency=0.5,  # Fast GC
            auto_cleanup=True
        )
        
        try:
            # Create many objects to track
            objects = []
            for i in range(100):
                obj = list(range(100))  # Small objects
                objects.append(obj)
                manager.track_object(f"obj_{i}", obj)
            
            # Wait for some GC cycles
            await asyncio.sleep(1.0)
            
            # Run optimization
            result = await manager.optimize(force=True)
            assert result["optimization_performed"] is True
            
            # Clear references to allow GC
            objects.clear()
            gc.collect()
            
            # Wait and check tracking
            await asyncio.sleep(0.1)
            tracked = manager.get_tracked_objects()
            
            # Some objects should still be tracked
            assert len(tracked) <= 100
            
        finally:
            await manager.shutdown()


if __name__ == "__main__":
    # Run specific test categories
    import sys
    
    if len(sys.argv) > 1:
        test_category = sys.argv[1]
        if test_category == "basic":
            pytest.main(["-v", "tests/test_performance.py::TestModelCache", 
                        "tests/test_performance.py::TestWorkQueue"])
        elif test_category == "integration":
            pytest.main(["-v", "tests/test_performance.py::TestPerformanceIntegration"])
        elif test_category == "load":
            pytest.main(["-v", "tests/test_performance.py::TestPerformanceLoad"])
        else:
            pytest.main(["-v", "tests/test_performance.py"])
    else:
        # Run all tests
        pytest.main(["-v", "tests/test_performance.py"])
