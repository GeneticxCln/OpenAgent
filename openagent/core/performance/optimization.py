"""
Performance Optimization System for OpenAgent.

This module provides advanced performance optimizations including
background model loading, caching, memory management, and startup acceleration.
"""

import asyncio
import gc
import os
import pickle
import threading
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""

    startup_time: float = 0.0
    model_load_time: float = 0.0
    memory_usage_mb: float = 0.0
    peak_memory_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    cache_hit_rate: float = 0.0
    avg_response_time: float = 0.0
    total_requests: int = 0

    def update_memory_stats(self):
        """Update memory statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        self.memory_usage_mb = memory_info.rss / 1024 / 1024
        self.peak_memory_mb = max(self.peak_memory_mb, self.memory_usage_mb)

        # CPU usage
        self.cpu_usage_percent = psutil.cpu_percent()

        # GPU memory if available
        if torch.cuda.is_available():
            self.gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024


class ModelCache:
    """Advanced model caching system with background loading."""

    def __init__(self, cache_dir: Optional[Path] = None, max_memory_gb: float = 4.0):
        """Initialize model cache."""
        self.cache_dir = cache_dir or Path.home() / ".cache" / "openagent" / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_memory_gb = max_memory_gb

        # Cache storage
        self.model_cache: Dict[str, Any] = {}
        self.tokenizer_cache: Dict[str, Any] = {}
        self.loading_tasks: Dict[str, asyncio.Task] = {}
        self.access_times: Dict[str, float] = {}

        # Background loading
        self.background_loader_active = False
        self.preload_queue: List[Tuple[str, Dict[str, Any]]] = []
        self.cache_lock = threading.RLock()

        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_memory_saved_gb = 0.0

    async def get_model(
        self, model_name: str, load_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Any]:
        """Get model and tokenizer from cache or load them."""
        cache_key = self._get_cache_key(model_name, load_config or {})

        with self.cache_lock:
            # Check if model is already loaded
            if cache_key in self.model_cache:
                self.cache_hits += 1
                self.access_times[cache_key] = time.time()
                return self.model_cache[cache_key], self.tokenizer_cache.get(cache_key)

            # Check if model is currently being loaded
            if cache_key in self.loading_tasks:
                model, tokenizer = await self.loading_tasks[cache_key]
                return model, tokenizer

        # Load model asynchronously
        self.cache_misses += 1
        loading_task = asyncio.create_task(
            self._load_model_async(model_name, load_config or {})
        )

        with self.cache_lock:
            self.loading_tasks[cache_key] = loading_task

        try:
            model, tokenizer = await loading_task

            with self.cache_lock:
                # Cache the loaded model
                self.model_cache[cache_key] = model
                self.tokenizer_cache[cache_key] = tokenizer
                self.access_times[cache_key] = time.time()

                # Clean up loading tasks
                if cache_key in self.loading_tasks:
                    del self.loading_tasks[cache_key]

                # Manage memory usage
                await self._manage_cache_memory()

            return model, tokenizer

        except Exception as e:
            with self.cache_lock:
                if cache_key in self.loading_tasks:
                    del self.loading_tasks[cache_key]
            raise e

    async def preload_model(
        self,
        model_name: str,
        load_config: Optional[Dict[str, Any]] = None,
        priority: int = 1,
    ):
        """Preload a model in the background."""
        cache_key = self._get_cache_key(model_name, load_config or {})

        with self.cache_lock:
            if (
                cache_key not in self.model_cache
                and cache_key not in self.loading_tasks
            ):
                self.preload_queue.append((model_name, load_config or {}))
                self.preload_queue.sort(key=lambda x: priority, reverse=True)

        if not self.background_loader_active:
            asyncio.create_task(self._background_loader())

    async def _background_loader(self):
        """Background model loader."""
        self.background_loader_active = True

        try:
            while self.preload_queue:
                with self.cache_lock:
                    if not self.preload_queue:
                        break
                    model_name, load_config = self.preload_queue.pop(0)

                # Check if model is still needed
                cache_key = self._get_cache_key(model_name, load_config)
                if cache_key in self.model_cache:
                    continue

                try:
                    # Load model in background
                    await self.get_model(model_name, load_config)

                    # Yield control to allow other tasks
                    await asyncio.sleep(0.1)

                except Exception as e:
                    print(f"Background model loading failed for {model_name}: {e}")

        finally:
            self.background_loader_active = False

    async def _load_model_async(
        self, model_name: str, load_config: Dict[str, Any]
    ) -> Tuple[Any, Any]:
        """Load model and tokenizer asynchronously."""

        def _load_sync():
            # Default configuration
            config = {
                "device_map": "auto",
                "torch_dtype": torch.float16,
                "load_in_4bit": True,
                "trust_remote_code": True,
                **load_config,
            }

            # Load tokenizer first (usually faster)
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=config.get("trust_remote_code", True)
            )

            # Load model
            model = AutoModelForCausalLM.from_pretrained(model_name, **config)

            return model, tokenizer

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _load_sync)

    async def _manage_cache_memory(self):
        """Manage cache memory usage by evicting old models."""
        current_memory = self._get_cache_memory_usage()

        if current_memory > self.max_memory_gb:
            # Sort by access time (LRU eviction)
            sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])

            for cache_key, _ in sorted_keys:
                if current_memory <= self.max_memory_gb * 0.8:  # Keep 20% buffer
                    break

                # Evict model
                if cache_key in self.model_cache:
                    model = self.model_cache.pop(cache_key)
                    tokenizer = self.tokenizer_cache.pop(cache_key, None)
                    del self.access_times[cache_key]

                    # Force cleanup
                    del model
                    if tokenizer:
                        del tokenizer

                    # Calculate memory saved
                    old_memory = current_memory
                    current_memory = self._get_cache_memory_usage()
                    self.total_memory_saved_gb += old_memory - current_memory

                    # Force garbage collection
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    def _get_cache_memory_usage(self) -> float:
        """Get current cache memory usage in GB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024 / 1024

    def _get_cache_key(self, model_name: str, load_config: Dict[str, Any]) -> str:
        """Generate cache key for model and config."""
        import hashlib

        config_str = str(sorted(load_config.items()))
        key_string = f"{model_name}_{config_str}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cached_models": len(self.model_cache),
            "loading_models": len(self.loading_tasks),
            "memory_usage_gb": self._get_cache_memory_usage(),
            "memory_saved_gb": self.total_memory_saved_gb,
            "preload_queue_size": len(self.preload_queue),
        }


class StartupOptimizer:
    """Startup optimization system."""

    def __init__(self):
        """Initialize startup optimizer."""
        self.optimization_enabled = True
        self.lazy_imports = {}
        self.deferred_initializations = []
        self.warmup_tasks = []

    def enable_lazy_imports(self):
        """Enable lazy importing of heavy modules."""
        # Store original imports and replace with lazy versions
        self.lazy_imports = {
            "transformers": self._create_lazy_import("transformers"),
            "torch": self._create_lazy_import("torch"),
            "numpy": self._create_lazy_import("numpy"),
        }

    def _create_lazy_import(self, module_name: str):
        """Create a lazy import wrapper."""

        class LazyModule:
            def __init__(self, name):
                self._module_name = name
                self._module = None

            def __getattr__(self, attr):
                if self._module is None:
                    import importlib

                    self._module = importlib.import_module(self._module_name)
                return getattr(self._module, attr)

        return LazyModule(module_name)

    async def optimize_startup(self, model_name: Optional[str] = None):
        """Optimize startup sequence."""
        start_time = time.time()

        # Phase 1: Essential components only
        essential_tasks = [
            self._init_essential_components(),
            self._setup_basic_config(),
        ]

        await asyncio.gather(*essential_tasks)

        # Phase 2: Background initialization
        if model_name:
            background_tasks = [
                self._preload_model_async(model_name),
                self._warm_cache(),
                self._optimize_memory(),
            ]

            # Don't wait for these - let them run in background
            for task in background_tasks:
                asyncio.create_task(task)

        startup_time = time.time() - start_time
        return startup_time

    async def _init_essential_components(self):
        """Initialize only essential components."""
        # Minimal initialization for immediate availability
        pass

    async def _setup_basic_config(self):
        """Setup basic configuration."""
        # Fast config loading
        pass

    async def _preload_model_async(self, model_name: str):
        """Preload model in background."""
        # This runs after startup, so UI is responsive immediately
        try:
            from ..llm import get_llm

            llm = get_llm()
            await llm.load_model(model_name)
        except Exception as e:
            print(f"Background model preload failed: {e}")

    async def _warm_cache(self):
        """Warm up caches."""
        # Warm up various caches in background
        pass

    async def _optimize_memory(self):
        """Optimize memory usage."""
        # Memory optimization in background
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class MemoryOptimizer:
    """Advanced memory optimization system."""

    def __init__(self):
        """Initialize memory optimizer."""
        self.memory_threshold_mb = 1024  # 1GB threshold
        self.monitoring_active = False
        self.optimization_callbacks = []
        self.weak_refs = {}

    def start_monitoring(self, interval: float = 30.0):
        """Start memory monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            asyncio.create_task(self._memory_monitor(interval))

    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring_active = False

    async def _memory_monitor(self, interval: float):
        """Monitor memory usage and optimize when needed."""
        while self.monitoring_active:
            try:
                current_memory = self._get_memory_usage_mb()

                if current_memory > self.memory_threshold_mb:
                    await self.optimize_memory()

                await asyncio.sleep(interval)

            except Exception as e:
                print(f"Memory monitor error: {e}")
                await asyncio.sleep(interval)

    async def optimize_memory(self):
        """Perform memory optimization."""
        print("🧹 Optimizing memory usage...")

        # 1. Garbage collection
        collected = gc.collect()

        # 2. Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3. Run optimization callbacks
        for callback in self.optimization_callbacks:
            try:
                await callback()
            except Exception as e:
                print(f"Memory optimization callback failed: {e}")

        # 4. Clean up weak references
        self._cleanup_weak_refs()

        print(f"✅ Memory optimization complete. Collected {collected} objects.")

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def _cleanup_weak_refs(self):
        """Clean up dead weak references."""
        dead_refs = []
        for key, ref in self.weak_refs.items():
            if ref() is None:
                dead_refs.append(key)

        for key in dead_refs:
            del self.weak_refs[key]

    def add_optimization_callback(self, callback: Callable):
        """Add a memory optimization callback."""
        self.optimization_callbacks.append(callback)

    def register_weak_ref(self, key: str, obj: Any):
        """Register an object with weak reference."""
        self.weak_refs[key] = weakref.ref(obj)


class ResponseCache:
    """High-performance response caching system."""

    def __init__(self, max_size: int = 1000, ttl: float = 3600):
        """Initialize response cache."""
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.access_order = []

    def get(self, key: str) -> Optional[Any]:
        """Get cached response."""
        if key in self.cache:
            response, timestamp = self.cache[key]

            # Check TTL
            if time.time() - timestamp < self.ttl:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                return response
            else:
                # Expired
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)

        return None

    def put(self, key: str, response: Any):
        """Cache a response."""
        # Remove old entry if exists
        if key in self.cache:
            del self.cache[key]
            self.access_order.remove(key)

        # Add new entry
        self.cache[key] = (response, time.time())
        self.access_order.append(key)

        # Evict if necessary (LRU)
        while len(self.cache) > self.max_size:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]

    def clear(self):
        """Clear all cached responses."""
        self.cache.clear()
        self.access_order.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
            "usage": len(self.cache) / self.max_size if self.max_size > 0 else 0,
        }


class PerformanceProfiler:
    """Performance profiling and optimization recommendations."""

    def __init__(self):
        """Initialize performance profiler."""
        self.metrics = PerformanceMetrics()
        self.profiling_active = False
        self.operation_times = {}
        self.recommendations = []

    def start_profiling(self):
        """Start performance profiling."""
        self.profiling_active = True
        self.metrics.update_memory_stats()

    def stop_profiling(self):
        """Stop performance profiling."""
        self.profiling_active = False
        self._generate_recommendations()

    def record_operation(self, operation: str, duration: float):
        """Record operation duration."""
        if operation not in self.operation_times:
            self.operation_times[operation] = []

        self.operation_times[operation].append(duration)

        # Keep only recent measurements
        if len(self.operation_times[operation]) > 100:
            self.operation_times[operation] = self.operation_times[operation][-50:]

    def _generate_recommendations(self):
        """Generate performance recommendations."""
        self.recommendations.clear()

        # Memory recommendations
        if self.metrics.memory_usage_mb > 2048:  # 2GB
            self.recommendations.append(
                {
                    "type": "memory",
                    "severity": "high",
                    "message": "Memory usage is high. Consider enabling model quantization or reducing cache size.",
                    "suggestion": "Add --load-in-4bit flag or reduce max_memory_gb setting",
                }
            )

        # Startup time recommendations
        if self.metrics.startup_time > 5.0:  # 5 seconds
            self.recommendations.append(
                {
                    "type": "startup",
                    "severity": "medium",
                    "message": "Startup time is slow. Consider enabling background model loading.",
                    "suggestion": "Enable model preloading or use a lighter default model",
                }
            )

        # Response time recommendations
        if self.metrics.avg_response_time > 2.0:  # 2 seconds
            self.recommendations.append(
                {
                    "type": "response",
                    "severity": "medium",
                    "message": "Response time is slow. Consider GPU acceleration or model optimization.",
                    "suggestion": "Use GPU acceleration or a smaller model variant",
                }
            )

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        self.metrics.update_memory_stats()

        # Calculate averages
        avg_times = {}
        for operation, times in self.operation_times.items():
            avg_times[operation] = sum(times) / len(times) if times else 0

        return {
            "metrics": {
                "startup_time": self.metrics.startup_time,
                "memory_usage_mb": self.metrics.memory_usage_mb,
                "peak_memory_mb": self.metrics.peak_memory_mb,
                "cpu_usage_percent": self.metrics.cpu_usage_percent,
                "gpu_memory_mb": self.metrics.gpu_memory_mb,
                "avg_response_time": self.metrics.avg_response_time,
                "total_requests": self.metrics.total_requests,
            },
            "operation_averages": avg_times,
            "recommendations": self.recommendations,
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
                "gpu_available": torch.cuda.is_available(),
                "gpu_device_count": (
                    torch.cuda.device_count() if torch.cuda.is_available() else 0
                ),
            },
        }


# Global instances
_model_cache = None
_startup_optimizer = None
_memory_optimizer = None
_response_cache = None
_performance_profiler = None


def get_model_cache() -> ModelCache:
    """Get global model cache instance."""
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelCache()
    return _model_cache


def get_startup_optimizer() -> StartupOptimizer:
    """Get global startup optimizer instance."""
    global _startup_optimizer
    if _startup_optimizer is None:
        _startup_optimizer = StartupOptimizer()
    return _startup_optimizer


def get_memory_optimizer() -> MemoryOptimizer:
    """Get global memory optimizer instance."""
    global _memory_optimizer
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer()
    return _memory_optimizer


def get_response_cache() -> ResponseCache:
    """Get global response cache instance."""
    global _response_cache
    if _response_cache is None:
        _response_cache = ResponseCache()
    return _response_cache


def get_performance_profiler() -> PerformanceProfiler:
    """Get global performance profiler instance."""
    global _performance_profiler
    if _performance_profiler is None:
        _performance_profiler = PerformanceProfiler()
    return _performance_profiler


async def optimize_openagent_performance():
    """Optimize OpenAgent performance with all available optimizations."""
    print("🚀 Starting OpenAgent performance optimization...")

    # Get optimizers
    startup_opt = get_startup_optimizer()
    memory_opt = get_memory_optimizer()
    model_cache = get_model_cache()

    # Start background optimizations
    tasks = [
        startup_opt.optimize_startup(),
        memory_opt.optimize_memory(),
    ]

    # Start monitoring
    memory_opt.start_monitoring()

    # Wait for initial optimizations
    results = await asyncio.gather(*tasks, return_exceptions=True)

    print("✅ OpenAgent performance optimization complete!")

    return {
        "startup_optimization": (
            results[0] if not isinstance(results[0], Exception) else str(results[0])
        ),
        "memory_optimization": (
            "completed" if not isinstance(results[1], Exception) else str(results[1])
        ),
        "model_cache_active": True,
        "memory_monitoring_active": memory_opt.monitoring_active,
    }
