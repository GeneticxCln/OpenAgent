"""
Performance benchmarks and integration tests for OpenAgent.

This module contains benchmark tests to measure performance improvements
and integration tests with the actual OpenAgent server components.
"""

import asyncio
import time
import pytest
import statistics
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from openagent.core.performance_integration import PerformanceManager
from openagent.core.performance_config import PerformanceConfigurationManager, ExtendedConfig
from openagent.core.performance import RequestPriority


class PerformanceBenchmarks:
    """Performance benchmark utilities."""
    
    @staticmethod
    async def measure_throughput(func, *args, iterations: int = 100, **kwargs) -> Dict[str, float]:
        """Measure function throughput."""
        start_time = time.time()
        
        tasks = [func(*args, **kwargs) for _ in range(iterations)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        return {
            "iterations": iterations,
            "total_time": total_time,
            "avg_time": total_time / iterations,
            "throughput_per_second": iterations / total_time,
            "success_count": len([r for r in results if r is not None])
        }
    
    @staticmethod
    async def measure_latency(func, *args, iterations: int = 10, **kwargs) -> Dict[str, float]:
        """Measure function latency statistics."""
        latencies = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            await func(*args, **kwargs)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        return {
            "iterations": iterations,
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "avg_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "std_dev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            "p95_ms": sorted(latencies)[int(0.95 * len(latencies))] if len(latencies) > 1 else latencies[0],
            "p99_ms": sorted(latencies)[int(0.99 * len(latencies))] if len(latencies) > 1 else latencies[0]
        }


@pytest.mark.benchmark
class TestModelCacheBenchmarks:
    """Benchmarks for model cache performance."""
    
    @pytest.fixture
    async def performance_manager(self):
        """Performance manager with optimized cache settings."""
        from openagent.core.performance_config import (
            ExtendedConfig, PerformanceConfig, ModelCacheConfig
        )
        
        config = ExtendedConfig(
            performance=PerformanceConfig(
                model_cache=ModelCacheConfig(
                    max_size=100,
                    default_ttl=3600,
                    background_warming=True
                )
            )
        )
        
        config_manager = PerformanceConfigurationManager(config)
        manager = PerformanceManager(config_manager)
        await manager.initialize()
        
        yield manager
        await manager.shutdown()
    
    class MockModel:
        def __init__(self, name: str, size: int = 1000):
            self.name = name
            self.data = b"x" * size
    
    @pytest.mark.asyncio
    async def test_cache_put_throughput(self, performance_manager):
        """Benchmark cache put operations."""
        cache = performance_manager.get_model_cache()
        
        async def put_model(i):
            model = self.MockModel(f"model-{i}")
            return await cache.put(f"model-{i}", model)
        
        results = await PerformanceBenchmarks.measure_throughput(
            put_model, iterations=100
        )
        
        print(f"\nCache PUT throughput: {results['throughput_per_second']:.2f} ops/sec")
        print(f"Average latency: {results['avg_time']*1000:.2f}ms")
        
        # Performance assertions
        assert results["throughput_per_second"] > 50  # At least 50 puts/sec
        assert results["avg_time"] < 0.1  # Less than 100ms average
    
    @pytest.mark.asyncio
    async def test_cache_get_throughput(self, performance_manager):
        """Benchmark cache get operations."""
        cache = performance_manager.get_model_cache()
        
        # Pre-populate cache
        for i in range(50):
            model = self.MockModel(f"model-{i}")
            await cache.put(f"model-{i}", model)
        
        async def get_model(i):
            return await cache.get(f"model-{i % 50}")
        
        results = await PerformanceBenchmarks.measure_throughput(
            get_model, iterations=500
        )
        
        print(f"\nCache GET throughput: {results['throughput_per_second']:.2f} ops/sec")
        print(f"Average latency: {results['avg_time']*1000:.2f}ms")
        
        # Performance assertions
        assert results["throughput_per_second"] > 200  # At least 200 gets/sec
        assert results["avg_time"] < 0.01  # Less than 10ms average
    
    @pytest.mark.asyncio
    async def test_cache_mixed_workload(self, performance_manager):
        """Benchmark mixed cache workload (80% reads, 20% writes)."""
        cache = performance_manager.get_model_cache()
        
        # Pre-populate cache
        for i in range(20):
            model = self.MockModel(f"model-{i}")
            await cache.put(f"model-{i}", model)
        
        async def mixed_operation(i):
            if i % 5 == 0:  # 20% writes
                model = self.MockModel(f"new-model-{i}")
                return await cache.put(f"new-model-{i}", model)
            else:  # 80% reads
                return await cache.get(f"model-{i % 20}")
        
        results = await PerformanceBenchmarks.measure_throughput(
            mixed_operation, iterations=100
        )
        
        print(f"\nMixed workload throughput: {results['throughput_per_second']:.2f} ops/sec")
        
        assert results["throughput_per_second"] > 100


@pytest.mark.benchmark  
class TestWorkQueueBenchmarks:
    """Benchmarks for work queue performance."""
    
    @pytest.fixture
    async def performance_manager(self):
        """Performance manager with optimized queue settings."""
        from openagent.core.performance_config import (
            ExtendedConfig, PerformanceConfig, WorkQueueConfig
        )
        
        config = ExtendedConfig(
            performance=PerformanceConfig(
                work_queue=WorkQueueConfig(
                    max_workers=10,
                    max_queue_size=1000,
                    enable_retries=True
                )
            )
        )
        
        config_manager = PerformanceConfigurationManager(config)
        manager = PerformanceManager(config_manager)
        await manager.initialize()
        
        yield manager
        await manager.shutdown()
    
    async def cpu_bound_task(self, n: int) -> int:
        """CPU-bound task for testing."""
        # Simulate CPU work
        result = sum(i * i for i in range(n))
        return result
    
    async def io_bound_task(self, delay: float) -> str:
        """I/O-bound task for testing."""
        await asyncio.sleep(delay)
        return f"Completed after {delay}s"
    
    @pytest.mark.asyncio
    async def test_queue_cpu_throughput(self, performance_manager):
        """Benchmark CPU-bound task throughput."""
        
        async def submit_cpu_task(i):
            return await performance_manager.submit_task(
                self.cpu_bound_task, 100,
                priority=RequestPriority.NORMAL
            )
        
        results = await PerformanceBenchmarks.measure_throughput(
            submit_cpu_task, iterations=50
        )
        
        print(f"\nCPU task throughput: {results['throughput_per_second']:.2f} ops/sec")
        print(f"Success rate: {results['success_count']}/{results['iterations']}")
        
        assert results["success_count"] == results["iterations"]
        assert results["throughput_per_second"] > 5  # At least 5 CPU tasks/sec
    
    @pytest.mark.asyncio
    async def test_queue_io_throughput(self, performance_manager):
        """Benchmark I/O-bound task throughput."""
        
        async def submit_io_task(i):
            return await performance_manager.submit_task(
                self.io_bound_task, 0.01,  # 10ms delay
                priority=RequestPriority.NORMAL
            )
        
        results = await PerformanceBenchmarks.measure_throughput(
            submit_io_task, iterations=100
        )
        
        print(f"\nI/O task throughput: {results['throughput_per_second']:.2f} ops/sec")
        
        assert results["success_count"] == results["iterations"]
        assert results["throughput_per_second"] > 50  # At least 50 I/O tasks/sec
    
    @pytest.mark.asyncio
    async def test_queue_priority_performance(self, performance_manager):
        """Benchmark priority queue performance."""
        
        async def submit_priority_task(i):
            priority = RequestPriority.HIGH if i % 3 == 0 else RequestPriority.NORMAL
            return await performance_manager.submit_task(
                self.io_bound_task, 0.005,
                priority=priority
            )
        
        results = await PerformanceBenchmarks.measure_throughput(
            submit_priority_task, iterations=60
        )
        
        print(f"\nPriority queue throughput: {results['throughput_per_second']:.2f} ops/sec")
        
        assert results["success_count"] == results["iterations"]


@pytest.mark.benchmark
class TestMemoryManagerBenchmarks:
    """Benchmarks for memory management performance."""
    
    @pytest.fixture
    async def performance_manager(self):
        """Performance manager with optimized memory settings."""
        from openagent.core.performance_config import (
            ExtendedConfig, PerformanceConfig, MemoryManagementConfig
        )
        
        config = ExtendedConfig(
            performance=PerformanceConfig(
                memory_management=MemoryManagementConfig(
                    gc_frequency=1.0,  # Fast GC for benchmarking
                    auto_cleanup=True
                )
            )
        )
        
        config_manager = PerformanceConfigurationManager(config)
        manager = PerformanceManager(config_manager)
        await manager.initialize()
        
        yield manager
        await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_memory_optimization_latency(self, performance_manager):
        """Benchmark memory optimization latency."""
        
        # Create objects to optimize
        test_objects = []
        for i in range(100):
            obj = list(range(100))
            test_objects.append(obj)
            performance_manager.track_object(f"bench_obj_{i}", obj)
        
        async def run_optimization():
            return await performance_manager.optimize_memory(force=True)
        
        latency_stats = await PerformanceBenchmarks.measure_latency(
            run_optimization, iterations=5
        )
        
        print(f"\nMemory optimization latency:")
        print(f"  Average: {latency_stats['avg_ms']:.2f}ms")
        print(f"  P95: {latency_stats['p95_ms']:.2f}ms")
        print(f"  P99: {latency_stats['p99_ms']:.2f}ms")
        
        # Performance assertions
        assert latency_stats["avg_ms"] < 1000  # Less than 1 second average
        assert latency_stats["p99_ms"] < 5000   # Less than 5 seconds p99
    
    @pytest.mark.asyncio
    async def test_object_tracking_throughput(self, performance_manager):
        """Benchmark object tracking throughput."""
        
        objects = [list(range(50)) for _ in range(100)]
        
        async def track_object(i):
            performance_manager.track_object(f"throughput_obj_{i}", objects[i])
            return True
        
        results = await PerformanceBenchmarks.measure_throughput(
            track_object, iterations=100
        )
        
        print(f"\nObject tracking throughput: {results['throughput_per_second']:.2f} ops/sec")
        
        assert results["throughput_per_second"] > 1000  # At least 1000 tracks/sec


@pytest.mark.integration
class TestPerformanceIntegrationBenchmarks:
    """End-to-end performance integration benchmarks."""
    
    @pytest.fixture
    async def full_performance_manager(self):
        """Fully configured performance manager."""
        from openagent.core.performance_config import (
            ExtendedConfig, PerformanceConfig, ModelCacheConfig,
            WorkQueueConfig, MemoryManagementConfig, ResourceMonitoringConfig
        )
        
        config = ExtendedConfig(
            performance=PerformanceConfig(
                model_cache=ModelCacheConfig(max_size=10, default_ttl=300),
                work_queue=WorkQueueConfig(max_workers=5, max_queue_size=100),
                memory_management=MemoryManagementConfig(gc_frequency=5.0),
                resource_monitoring=ResourceMonitoringConfig(monitoring_interval=1.0)
            )
        )
        
        config_manager = PerformanceConfigurationManager(config)
        manager = PerformanceManager(config_manager)
        await manager.initialize()
        
        yield manager
        await manager.shutdown()
    
    class MockModel:
        def __init__(self, name: str):
            self.name = name
            self.data = [i for i in range(1000)]  # Some data
    
    @pytest.mark.asyncio
    async def test_full_workflow_performance(self, full_performance_manager):
        """Benchmark complete workflow with all components."""
        
        async def full_workflow(i):
            # 1. Try to get model from cache
            model = await full_performance_manager.get_cached_model(f"workflow_model_{i % 5}")
            
            # 2. If not cached, create and cache it
            if model is None:
                model = self.MockModel(f"workflow_model_{i % 5}")
                await full_performance_manager.cache_model(f"workflow_model_{i % 5}", model)
            
            # 3. Submit a task using the model
            async def process_with_model(model_name):
                # Simulate model processing
                await asyncio.sleep(0.01)
                return f"Processed with {model_name}"
            
            result = await full_performance_manager.submit_task(
                process_with_model, model.name,
                priority=RequestPriority.NORMAL
            )
            
            # 4. Track the model for memory management
            full_performance_manager.track_object(f"workflow_model_{i}", model)
            
            return result
        
        # Benchmark the full workflow
        results = await PerformanceBenchmarks.measure_throughput(
            full_workflow, iterations=50
        )
        
        print(f"\nFull workflow throughput: {results['throughput_per_second']:.2f} ops/sec")
        print(f"Success rate: {results['success_count']}/{results['iterations']}")
        
        # Get performance status
        status = await full_performance_manager.get_performance_status()
        print(f"\nPerformance status:")
        for component, info in status["components"].items():
            print(f"  {component}: {'✓' if info['enabled'] else '✗'}")
        
        # Performance assertions
        assert results["success_count"] == results["iterations"]
        assert results["throughput_per_second"] > 20  # At least 20 workflows/sec
        assert status["initialized"] is True
    
    @pytest.mark.asyncio
    async def test_concurrent_load_performance(self, full_performance_manager):
        """Test performance under concurrent load."""
        
        async def concurrent_task(task_id: int):
            # Mix different operations
            if task_id % 4 == 0:
                # Cache operation
                model = self.MockModel(f"load_model_{task_id}")
                return await full_performance_manager.cache_model(f"load_model_{task_id}", model)
            elif task_id % 4 == 1:
                # Queue operation
                async def work(x):
                    return x * x
                return await full_performance_manager.submit_task(work, task_id)
            elif task_id % 4 == 2:
                # Memory operation
                obj = [i for i in range(100)]
                full_performance_manager.track_object(f"load_obj_{task_id}", obj)
                return True
            else:
                # Mixed operation
                stats = await full_performance_manager.get_memory_stats()
                return stats is not None
        
        # Run many concurrent tasks
        start_time = time.time()
        tasks = [concurrent_task(i) for i in range(200)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Analyze results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        total_time = end_time - start_time
        throughput = len(successful_results) / total_time
        
        print(f"\nConcurrent load performance:")
        print(f"  Total tasks: {len(tasks)}")
        print(f"  Successful: {len(successful_results)}")
        print(f"  Failed: {len(failed_results)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.2f} ops/sec")
        
        # Performance assertions
        assert len(successful_results) > 180  # At least 90% success rate
        assert throughput > 50  # At least 50 ops/sec under load
        
        # Print any failures for debugging
        if failed_results:
            print(f"\nFailures: {failed_results[:5]}")  # First 5 failures


if __name__ == "__main__":
    # Run benchmark tests
    pytest.main(["-v", "-m", "benchmark", __file__])
