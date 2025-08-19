"""
Advanced model caching system for OpenAgent.

This module provides:
- LRU cache for multiple model instances
- Background model preloading and warming
- Smart memory management with model swapping
- Cache hit/miss metrics and optimization
"""

import asyncio
import logging
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from openagent.core.exceptions import AgentError

logger = logging.getLogger(__name__)


@dataclass
class ModelCacheEntry:
    """Model cache entry with metadata."""

    model_name: str
    model_instance: Any  # HuggingFaceLLM or OllamaLLM
    load_time: float
    last_accessed: float
    access_count: int
    memory_usage_mb: float
    is_preloaded: bool = False
    is_warming: bool = False
    warm_time: Optional[float] = None


@dataclass
class CacheMetrics:
    """Cache performance metrics."""

    cache_hits: int = 0
    cache_misses: int = 0
    models_loaded: int = 0
    models_evicted: int = 0
    total_load_time: float = 0.0
    avg_load_time: float = 0.0
    memory_usage_mb: float = 0.0
    cache_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class ModelCache:
    """
    Advanced LRU model cache with preloading and background warming.

    Features:
    - LRU eviction policy
    - Background model preloading
    - Memory usage tracking
    - Cache warming for faster response times
    - Thread-safe operations
    """

    def __init__(
        self,
        max_models: int = 3,
        max_memory_mb: float = 4096,
        preload_popular: bool = True,
        background_warming: bool = True,
        warm_on_startup: bool = True,
    ):
        """
        Initialize model cache.

        Args:
            max_models: Maximum number of models to keep in cache
            max_memory_mb: Maximum memory usage in MB
            preload_popular: Whether to preload popular models
            background_warming: Whether to warm models in background
            warm_on_startup: Whether to warm models on startup
        """
        self.max_models = max_models
        self.max_memory_mb = max_memory_mb
        self.preload_popular = preload_popular
        self.background_warming = background_warming
        self.warm_on_startup = warm_on_startup

        # Thread-safe cache storage
        self._cache: OrderedDict[str, ModelCacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._metrics = CacheMetrics()

        # Background tasks
        self._warming_tasks: Dict[str, asyncio.Task] = {}
        self._cleanup_task: Optional[asyncio.Task] = None

        # Popular models to preload
        self._popular_models = {
            "tiny-llama",
            "codellama-7b",
            "mistral-7b",
            "ollama:llama3",
            "ollama:codellama",
            "ollama:mistral",
        }

        # Weak references to track model usage
        self._model_refs: Dict[str, weakref.WeakSet] = {}

        logger.info(
            f"ModelCache initialized: max_models={max_models}, max_memory={max_memory_mb}MB"
        )

        # Start background tasks
        if self.warm_on_startup:
            asyncio.create_task(self._startup_warming())

        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def get_model(self, model_name: str, **kwargs) -> Any:
        """
        Get model from cache or load it.

        Args:
            model_name: Name of the model to get
            **kwargs: Additional arguments for model loading

        Returns:
            Model instance
        """
        async with self._lock:
            # Check cache first
            if model_name in self._cache:
                entry = self._cache[model_name]

                # Move to end (most recently used)
                self._cache.move_to_end(model_name)

                # Update access info
                entry.last_accessed = time.time()
                entry.access_count += 1

                # Update metrics
                self._metrics.cache_hits += 1

                logger.debug(f"Cache hit for model: {model_name}")
                return entry.model_instance

            # Cache miss - need to load model
            self._metrics.cache_misses += 1
            logger.info(f"Cache miss for model: {model_name}, loading...")

            # Check if we need to evict models first
            await self._ensure_capacity()

            # Load the model
            start_time = time.time()
            model_instance = await self._load_model(model_name, **kwargs)
            load_time = time.time() - start_time

            # Estimate memory usage
            memory_usage = await self._estimate_memory_usage(model_instance)

            # Create cache entry
            entry = ModelCacheEntry(
                model_name=model_name,
                model_instance=model_instance,
                load_time=load_time,
                last_accessed=time.time(),
                access_count=1,
                memory_usage_mb=memory_usage,
            )

            # Add to cache
            self._cache[model_name] = entry

            # Update metrics
            self._metrics.models_loaded += 1
            self._metrics.total_load_time += load_time
            self._metrics.avg_load_time = (
                self._metrics.total_load_time / self._metrics.models_loaded
            )
            self._metrics.memory_usage_mb += memory_usage
            self._metrics.cache_size = len(self._cache)

            logger.info(
                f"Model loaded and cached: {model_name} "
                f"(load_time={load_time:.2f}s, memory={memory_usage:.1f}MB)"
            )

            # Start background warming if enabled
            if self.background_warming and not entry.is_warming:
                await self._schedule_warming(model_name)

            return model_instance

    async def preload_model(self, model_name: str, **kwargs) -> bool:
        """
        Preload a model into the cache.

        Args:
            model_name: Name of the model to preload
            **kwargs: Additional arguments for model loading

        Returns:
            True if preloaded successfully, False otherwise
        """
        try:
            async with self._lock:
                if model_name in self._cache:
                    logger.debug(f"Model {model_name} already in cache")
                    return True

                logger.info(f"Preloading model: {model_name}")

                # Check capacity
                await self._ensure_capacity()

                # Load model
                start_time = time.time()
                model_instance = await self._load_model(model_name, **kwargs)
                load_time = time.time() - start_time

                # Estimate memory usage
                memory_usage = await self._estimate_memory_usage(model_instance)

                # Create cache entry marked as preloaded
                entry = ModelCacheEntry(
                    model_name=model_name,
                    model_instance=model_instance,
                    load_time=load_time,
                    last_accessed=time.time(),
                    access_count=0,  # Not accessed by user yet
                    memory_usage_mb=memory_usage,
                    is_preloaded=True,
                )

                # Add to cache
                self._cache[model_name] = entry

                # Update metrics
                self._metrics.memory_usage_mb += memory_usage
                self._metrics.cache_size = len(self._cache)

                logger.info(f"Model preloaded: {model_name} (time={load_time:.2f}s)")
                return True

        except Exception as e:
            logger.error(f"Failed to preload model {model_name}: {e}")
            return False

    async def warm_model(self, model_name: str) -> bool:
        """
        Warm up a model by running a simple inference.

        Args:
            model_name: Name of the model to warm

        Returns:
            True if warmed successfully, False otherwise
        """
        try:
            async with self._lock:
                if model_name not in self._cache:
                    logger.warning(f"Cannot warm model not in cache: {model_name}")
                    return False

                entry = self._cache[model_name]
                if entry.is_warming:
                    logger.debug(f"Model {model_name} is already warming")
                    return True

                entry.is_warming = True

            # Warm outside of lock to avoid blocking
            logger.info(f"Warming model: {model_name}")
            start_time = time.time()

            try:
                model = entry.model_instance

                # Run a simple warm-up inference
                if hasattr(model, "generate_response"):
                    await model.generate_response(
                        "Hello", max_new_tokens=1, system_prompt="Be brief."
                    )
                elif hasattr(model, "generate"):
                    # For Ollama models
                    async for _ in model.stream_generate("Hi"):
                        break

                warm_time = time.time() - start_time

                async with self._lock:
                    entry.is_warming = False
                    entry.warm_time = warm_time

                logger.info(f"Model warmed: {model_name} (time={warm_time:.2f}s)")
                return True

            except Exception as e:
                async with self._lock:
                    entry.is_warming = False
                logger.warning(f"Model warming failed for {model_name}: {e}")
                return False

        except Exception as e:
            logger.error(f"Error warming model {model_name}: {e}")
            return False

    async def evict_model(self, model_name: str) -> bool:
        """
        Manually evict a model from cache.

        Args:
            model_name: Name of the model to evict

        Returns:
            True if evicted, False if not in cache
        """
        async with self._lock:
            if model_name not in self._cache:
                return False

            entry = self._cache.pop(model_name)

            # Cleanup model
            await self._cleanup_model(entry)

            # Update metrics
            self._metrics.models_evicted += 1
            self._metrics.memory_usage_mb -= entry.memory_usage_mb
            self._metrics.cache_size = len(self._cache)

            logger.info(f"Model evicted: {model_name}")
            return True

    async def clear_cache(self) -> None:
        """Clear all models from cache."""
        async with self._lock:
            logger.info("Clearing model cache")

            # Cleanup all models
            for entry in self._cache.values():
                await self._cleanup_model(entry)

            self._cache.clear()

            # Reset metrics
            self._metrics.memory_usage_mb = 0
            self._metrics.cache_size = 0

    def get_metrics(self) -> CacheMetrics:
        """Get cache performance metrics."""
        return self._metrics

    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        cache_info = {
            "cache_size": len(self._cache),
            "max_models": self.max_models,
            "memory_usage_mb": self._metrics.memory_usage_mb,
            "max_memory_mb": self.max_memory_mb,
            "hit_rate": self._metrics.hit_rate,
            "models": [],
        }

        for name, entry in self._cache.items():
            cache_info["models"].append(
                {
                    "name": name,
                    "last_accessed": entry.last_accessed,
                    "access_count": entry.access_count,
                    "memory_mb": entry.memory_usage_mb,
                    "is_preloaded": entry.is_preloaded,
                    "is_warming": entry.is_warming,
                    "warm_time": entry.warm_time,
                }
            )

        return cache_info

    async def _load_model(self, model_name: str, **kwargs) -> Any:
        """Load a model instance."""
        from openagent.core.llm import get_llm

        # Get model instance
        model = get_llm(model_name, **kwargs)

        # Ensure model is loaded
        if hasattr(model, "load_model"):
            await model.load_model()

        return model

    async def _estimate_memory_usage(self, model_instance: Any) -> float:
        """Estimate memory usage of a model in MB."""
        try:
            # Try to get actual memory usage
            if hasattr(model_instance, "get_model_info"):
                info = model_instance.get_model_info()
                # This is a placeholder - in practice you'd calculate based on model size

            # Default estimates based on model name
            model_name = getattr(model_instance, "model_name", "unknown").lower()

            if "tiny" in model_name or "1b" in model_name:
                return 512  # ~512MB for tiny models
            elif "7b" in model_name:
                return 2048  # ~2GB for 7B models
            elif "13b" in model_name:
                return 4096  # ~4GB for 13B models
            elif "34b" in model_name:
                return 8192  # ~8GB for 34B models
            else:
                return 1024  # Default 1GB estimate

        except Exception:
            return 1024  # Default fallback

    async def _ensure_capacity(self) -> None:
        """Ensure cache has capacity for new model."""
        # Check model count limit
        while len(self._cache) >= self.max_models:
            await self._evict_lru_model()

        # Check memory limit (rough estimate)
        while self._metrics.memory_usage_mb > self.max_memory_mb * 0.8:  # 80% threshold
            if not await self._evict_lru_model():
                break  # No more models to evict

    async def _evict_lru_model(self) -> bool:
        """Evict the least recently used model."""
        if not self._cache:
            return False

        # Get LRU model (first in OrderedDict)
        lru_model_name = next(iter(self._cache))
        entry = self._cache.pop(lru_model_name)

        # Cleanup model
        await self._cleanup_model(entry)

        # Update metrics
        self._metrics.models_evicted += 1
        self._metrics.memory_usage_mb -= entry.memory_usage_mb
        self._metrics.cache_size = len(self._cache)

        logger.info(f"Evicted LRU model: {lru_model_name}")
        return True

    async def _cleanup_model(self, entry: ModelCacheEntry) -> None:
        """Cleanup a model instance."""
        try:
            if hasattr(entry.model_instance, "unload_model"):
                await entry.model_instance.unload_model()
        except Exception as e:
            logger.warning(f"Error during model cleanup: {e}")

    async def _startup_warming(self) -> None:
        """Warm popular models on startup."""
        if not self.preload_popular:
            return

        logger.info("Starting model preloading and warming")

        # Preload popular models
        for model_name in list(self._popular_models)[: self.max_models]:
            try:
                await self.preload_model(model_name)
                await asyncio.sleep(0.1)  # Brief pause between loads
            except Exception as e:
                logger.warning(f"Failed to preload {model_name}: {e}")

    async def _schedule_warming(self, model_name: str) -> None:
        """Schedule background warming for a model."""
        if model_name in self._warming_tasks:
            return  # Already scheduled

        async def warming_task():
            await asyncio.sleep(1)  # Brief delay
            await self.warm_model(model_name)
            self._warming_tasks.pop(model_name, None)

        self._warming_tasks[model_name] = asyncio.create_task(warming_task())

    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup task."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                async with self._lock:
                    current_time = time.time()
                    stale_threshold = 30 * 60  # 30 minutes

                    # Find stale models (not accessed recently and not preloaded)
                    stale_models = []
                    for name, entry in self._cache.items():
                        if (
                            not entry.is_preloaded
                            and current_time - entry.last_accessed > stale_threshold
                            and entry.access_count < 2  # Rarely used
                        ):
                            stale_models.append(name)

                    # Evict stale models if cache is getting full
                    if len(self._cache) > self.max_models * 0.7:  # 70% full
                        for model_name in stale_models[:1]:  # Evict one at a time
                            await self.evict_model(model_name)

            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")


class ModelCacheManager:
    """Global model cache manager."""

    def __init__(self):
        self._cache: Optional[ModelCache] = None

    def initialize(self, **kwargs) -> ModelCache:
        """Initialize the global model cache."""
        if self._cache is None:
            self._cache = ModelCache(**kwargs)
        return self._cache

    def get_cache(self) -> Optional[ModelCache]:
        """Get the global model cache."""
        return self._cache

    async def shutdown(self) -> None:
        """Shutdown the model cache."""
        if self._cache:
            await self._cache.clear_cache()
            self._cache = None


# Global cache manager
_cache_manager = ModelCacheManager()


def get_model_cache(**kwargs) -> ModelCache:
    """Get or create the global model cache."""
    cache = _cache_manager.get_cache()
    if cache is None:
        cache = _cache_manager.initialize(**kwargs)
    return cache
