"""
Concurrent work queue system for OpenAgent.

This module provides:
- AsyncIO-based task queue with prioritization
- Resource limits per user/session
- Request throttling and load balancing
- Queue metrics and monitoring
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from openagent.core.exceptions import AgentError

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Request priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class QueuedRequest:
    """A queued request with metadata."""

    request_id: str
    user_id: Optional[str]
    priority: RequestPriority
    created_at: float
    task_func: Callable
    args: tuple
    kwargs: dict
    future: asyncio.Future
    timeout: Optional[float] = None
    retries: int = 0
    max_retries: int = 2

    @property
    def age(self) -> float:
        """Age of the request in seconds."""
        return time.time() - self.created_at


@dataclass
class QueueMetrics:
    """Work queue performance metrics."""

    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    queue_size: int = 0
    active_workers: int = 0
    avg_processing_time: float = 0.0
    avg_queue_time: float = 0.0
    throughput_per_minute: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.completed_requests + self.failed_requests
        return self.completed_requests / total if total > 0 else 0.0


@dataclass
class UserLimits:
    """Per-user resource limits."""

    max_concurrent: int = 5
    max_queue_size: int = 20
    rate_limit_per_minute: int = 100
    max_processing_time: float = 300.0  # 5 minutes


class WorkQueue:
    """
    Advanced work queue with prioritization, user limits, and resource management.

    Features:
    - Priority-based task scheduling
    - Per-user rate limiting and concurrency control
    - Request timeouts and retry logic
    - Load balancing across workers
    - Comprehensive metrics and monitoring
    """

    def __init__(
        self,
        max_workers: int = 10,
        max_queue_size: int = 1000,
        default_timeout: float = 60.0,
        enable_retries: bool = True,
        cleanup_interval: float = 300.0,  # 5 minutes
    ):
        """
        Initialize work queue.

        Args:
            max_workers: Maximum number of concurrent workers
            max_queue_size: Maximum total queue size
            default_timeout: Default request timeout in seconds
            enable_retries: Whether to enable automatic retries
            cleanup_interval: Cleanup interval in seconds
        """
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.default_timeout = default_timeout
        self.enable_retries = enable_retries
        self.cleanup_interval = cleanup_interval

        # Queue storage by priority
        self._queues: Dict[RequestPriority, deque] = {
            priority: deque() for priority in RequestPriority
        }

        # Active requests and workers
        self._active_requests: Dict[str, QueuedRequest] = {}
        self._workers: List[asyncio.Task] = []
        self._worker_semaphore = asyncio.Semaphore(max_workers)

        # Per-user tracking
        self._user_limits: Dict[str, UserLimits] = defaultdict(UserLimits)
        self._user_active: Dict[str, int] = defaultdict(int)
        self._user_queued: Dict[str, int] = defaultdict(int)
        self._user_rate_tracker: Dict[str, deque] = defaultdict(deque)

        # Metrics and monitoring
        self._metrics = QueueMetrics()
        self._start_time = time.time()
        self._last_throughput_check = time.time()
        self._completed_last_minute = 0

        # Synchronization
        self._queue_condition = asyncio.Condition()
        self._shutdown_event = asyncio.Event()

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None

        logger.info(
            f"WorkQueue initialized: max_workers={max_workers}, max_queue_size={max_queue_size}"
        )

        # Start background tasks
        self._start_background_tasks()

    async def submit(
        self,
        task_func: Callable,
        *args,
        priority: RequestPriority = RequestPriority.NORMAL,
        user_id: Optional[str] = None,
        timeout: Optional[float] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """
        Submit a task to the work queue.

        Args:
            task_func: Async function to execute
            *args: Positional arguments for the function
            priority: Request priority
            user_id: Optional user ID for rate limiting
            timeout: Request timeout (uses default if None)
            request_id: Optional request ID (generated if None)
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the task execution

        Raises:
            AgentError: If queue is full or user limits exceeded
        """
        # Generate request ID
        if not request_id:
            request_id = str(uuid.uuid4())

        # Check user limits
        if user_id:
            await self._check_user_limits(user_id)

        # Check global queue size
        total_queued = sum(len(q) for q in self._queues.values())
        if total_queued >= self.max_queue_size:
            raise AgentError("Work queue is full")

        # Create queued request
        future = asyncio.Future()
        request = QueuedRequest(
            request_id=request_id,
            user_id=user_id,
            priority=priority,
            created_at=time.time(),
            task_func=task_func,
            args=args,
            kwargs=kwargs,
            future=future,
            timeout=timeout or self.default_timeout,
        )

        # Add to appropriate queue
        async with self._queue_condition:
            self._queues[priority].append(request)

            # Update tracking
            if user_id:
                self._user_queued[user_id] += 1
                self._track_user_rate(user_id)

            # Update metrics
            self._metrics.total_requests += 1
            self._metrics.queue_size = total_queued + 1

            # Notify workers
            self._queue_condition.notify()

        logger.debug(
            f"Task submitted: {request_id} (priority={priority.name}, user={user_id})"
        )

        try:
            # Wait for result with timeout
            return await asyncio.wait_for(future, timeout=request.timeout)
        except asyncio.TimeoutError:
            # Handle timeout
            self._metrics.timeout_requests += 1
            await self._remove_request(request_id)
            raise AgentError(f"Request {request_id} timed out after {request.timeout}s")

    async def submit_nowait(
        self,
        task_func: Callable,
        *args,
        priority: RequestPriority = RequestPriority.NORMAL,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Submit a task without waiting for result (fire-and-forget).

        Args:
            task_func: Async function to execute
            *args: Positional arguments
            priority: Request priority
            user_id: Optional user ID
            **kwargs: Keyword arguments

        Returns:
            Request ID for tracking
        """
        request_id = str(uuid.uuid4())

        # Submit with a dummy future that we don't wait on
        async def _dummy_wait():
            try:
                await self.submit(
                    task_func,
                    *args,
                    priority=priority,
                    user_id=user_id,
                    request_id=request_id,
                    **kwargs,
                )
            except Exception as e:
                logger.warning(f"Fire-and-forget task {request_id} failed: {e}")

        asyncio.create_task(_dummy_wait())
        return request_id

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        queue_sizes = {
            priority.name: len(queue) for priority, queue in self._queues.items()
        }

        return {
            "queue_sizes": queue_sizes,
            "total_queued": sum(queue_sizes.values()),
            "active_requests": len(self._active_requests),
            "active_workers": self._metrics.active_workers,
            "max_workers": self.max_workers,
            "uptime": time.time() - self._start_time,
        }

    async def get_user_status(self, user_id: str) -> Dict[str, Any]:
        """Get status for a specific user."""
        limits = self._user_limits[user_id]
        current_rate = len(
            [
                t
                for t in self._user_rate_tracker[user_id]
                if time.time() - t < 60  # Last minute
            ]
        )

        return {
            "active_requests": self._user_active[user_id],
            "queued_requests": self._user_queued[user_id],
            "current_rate": current_rate,
            "limits": {
                "max_concurrent": limits.max_concurrent,
                "max_queue_size": limits.max_queue_size,
                "rate_limit_per_minute": limits.rate_limit_per_minute,
            },
        }

    def get_metrics(self) -> QueueMetrics:
        """Get queue performance metrics."""
        return self._metrics

    async def set_user_limits(self, user_id: str, limits: UserLimits) -> None:
        """Set custom limits for a user."""
        self._user_limits[user_id] = limits
        logger.info(f"Updated limits for user {user_id}: {limits}")

    async def cancel_request(self, request_id: str) -> bool:
        """
        Cancel a specific request.

        Args:
            request_id: ID of request to cancel

        Returns:
            True if cancelled, False if not found
        """
        # Check active requests
        if request_id in self._active_requests:
            request = self._active_requests[request_id]
            if not request.future.done():
                request.future.cancel()
                await self._remove_request(request_id)
                return True

        # Check queued requests
        async with self._queue_condition:
            for priority, queue in self._queues.items():
                for i, request in enumerate(queue):
                    if request.request_id == request_id:
                        queue.remove(request)
                        if request.user_id:
                            self._user_queued[request.user_id] -= 1

                        if not request.future.done():
                            request.future.cancel()

                        self._metrics.queue_size -= 1
                        return True

        return False

    async def shutdown(self, timeout: float = 30.0) -> None:
        """
        Gracefully shutdown the work queue.

        Args:
            timeout: Maximum time to wait for shutdown
        """
        logger.info("Shutting down work queue...")

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._metrics_task:
            self._metrics_task.cancel()

        # Wait for workers to finish or timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._workers, return_exceptions=True), timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning("Some workers did not finish in time, forcing shutdown")
            for worker in self._workers:
                worker.cancel()

        # Cancel remaining requests
        for request in self._active_requests.values():
            if not request.future.done():
                request.future.cancel()

        logger.info("Work queue shutdown complete")

    def _start_background_tasks(self) -> None:
        """Start background worker and maintenance tasks."""
        # Start worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self._workers.append(worker)

        # Start maintenance tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._metrics_task = asyncio.create_task(self._metrics_loop())

    async def _worker_loop(self, worker_name: str) -> None:
        """Main worker loop."""
        logger.debug(f"{worker_name} started")

        while not self._shutdown_event.is_set():
            try:
                # Wait for work
                request = await self._get_next_request()
                if not request:
                    continue

                # Acquire worker semaphore
                async with self._worker_semaphore:
                    self._metrics.active_workers += 1

                    try:
                        await self._process_request(request)
                    finally:
                        self._metrics.active_workers -= 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1)  # Brief pause before retrying

        logger.debug(f"{worker_name} stopped")

    async def _get_next_request(self) -> Optional[QueuedRequest]:
        """Get the next request to process."""
        async with self._queue_condition:
            while True:
                # Check queues in priority order (highest first)
                for priority in sorted(
                    RequestPriority, key=lambda x: x.value, reverse=True
                ):
                    queue = self._queues[priority]
                    if queue:
                        request = queue.popleft()

                        # Move to active requests
                        self._active_requests[request.request_id] = request

                        # Update tracking
                        if request.user_id:
                            self._user_queued[request.user_id] -= 1
                            self._user_active[request.user_id] += 1

                        self._metrics.queue_size -= 1
                        return request

                # No requests available, wait for notification
                if self._shutdown_event.is_set():
                    return None

                try:
                    await asyncio.wait_for(self._queue_condition.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

    async def _process_request(self, request: QueuedRequest) -> None:
        """Process a single request."""
        start_time = time.time()

        try:
            logger.debug(f"Processing request: {request.request_id}")

            # Check if request is already cancelled
            if request.future.cancelled():
                return

            # Execute the task
            if asyncio.iscoroutinefunction(request.task_func):
                result = await request.task_func(*request.args, **request.kwargs)
            else:
                result = request.task_func(*request.args, **request.kwargs)

            # Set result
            if not request.future.done():
                request.future.set_result(result)

            # Update metrics
            processing_time = time.time() - start_time
            queue_time = start_time - request.created_at

            self._metrics.completed_requests += 1
            self._update_avg_time("processing", processing_time)
            self._update_avg_time("queue", queue_time)

            logger.debug(
                f"Request completed: {request.request_id} "
                f"(processing={processing_time:.2f}s, queue={queue_time:.2f}s)"
            )

        except asyncio.CancelledError:
            # Request was cancelled
            if not request.future.done():
                request.future.cancel()
        except Exception as e:
            logger.error(f"Request {request.request_id} failed: {e}")

            # Check if we should retry
            if (
                self.enable_retries
                and request.retries < request.max_retries
                and not request.future.done()
            ):
                request.retries += 1
                logger.info(
                    f"Retrying request {request.request_id} (attempt {request.retries})"
                )

                # Re-queue with same priority
                async with self._queue_condition:
                    self._queues[request.priority].append(request)
                    self._queue_condition.notify()
                return

            # Set error result
            if not request.future.done():
                request.future.set_exception(e)

            self._metrics.failed_requests += 1

        finally:
            # Remove from active requests
            await self._remove_request(request.request_id)

    async def _remove_request(self, request_id: str) -> None:
        """Remove request from active tracking."""
        if request_id in self._active_requests:
            request = self._active_requests.pop(request_id)

            if request.user_id:
                self._user_active[request.user_id] -= 1

    async def _check_user_limits(self, user_id: str) -> None:
        """Check if user has exceeded limits."""
        limits = self._user_limits[user_id]

        # Check concurrent limit
        if self._user_active[user_id] >= limits.max_concurrent:
            raise AgentError(f"User {user_id} has reached concurrent request limit")

        # Check queue limit
        if self._user_queued[user_id] >= limits.max_queue_size:
            raise AgentError(f"User {user_id} has reached queue size limit")

        # Check rate limit
        recent_requests = [
            t
            for t in self._user_rate_tracker[user_id]
            if time.time() - t < 60  # Last minute
        ]
        if len(recent_requests) >= limits.rate_limit_per_minute:
            raise AgentError(f"User {user_id} has exceeded rate limit")

    def _track_user_rate(self, user_id: str) -> None:
        """Track user request rate."""
        now = time.time()
        tracker = self._user_rate_tracker[user_id]

        # Add current request
        tracker.append(now)

        # Remove old entries (older than 1 minute)
        while tracker and now - tracker[0] > 60:
            tracker.popleft()

    def _update_avg_time(self, metric_type: str, time_value: float) -> None:
        """Update average time metrics."""
        if metric_type == "processing":
            if self._metrics.avg_processing_time == 0:
                self._metrics.avg_processing_time = time_value
            else:
                # Exponential moving average
                self._metrics.avg_processing_time = (
                    0.9 * self._metrics.avg_processing_time + 0.1 * time_value
                )
        elif metric_type == "queue":
            if self._metrics.avg_queue_time == 0:
                self._metrics.avg_queue_time = time_value
            else:
                self._metrics.avg_queue_time = (
                    0.9 * self._metrics.avg_queue_time + 0.1 * time_value
                )

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of stale data."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.cleanup_interval)

                # Clean up user rate trackers
                current_time = time.time()
                for user_id, tracker in self._user_rate_tracker.items():
                    while tracker and current_time - tracker[0] > 300:  # 5 minutes
                        tracker.popleft()

                logger.debug("Performed queue cleanup")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _metrics_loop(self) -> None:
        """Update throughput metrics periodically."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Every minute

                # Calculate throughput
                now = time.time()
                if now - self._last_throughput_check >= 60:
                    completed_now = self._metrics.completed_requests
                    self._metrics.throughput_per_minute = (
                        completed_now - self._completed_last_minute
                    )
                    self._completed_last_minute = completed_now
                    self._last_throughput_check = now

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics loop: {e}")


class QueueManager:
    """Global work queue manager."""

    def __init__(self):
        self._queue: Optional[WorkQueue] = None

    def initialize(self, **kwargs) -> WorkQueue:
        """Initialize the global work queue."""
        if self._queue is None:
            self._queue = WorkQueue(**kwargs)
        return self._queue

    def get_queue(self) -> Optional[WorkQueue]:
        """Get the global work queue."""
        return self._queue

    async def shutdown(self) -> None:
        """Shutdown the work queue."""
        if self._queue:
            await self._queue.shutdown()
            self._queue = None


# Global queue manager
_queue_manager = QueueManager()


def get_work_queue(**kwargs) -> WorkQueue:
    """Get or create the global work queue."""
    queue = _queue_manager.get_queue()
    if queue is None:
        queue = _queue_manager.initialize(**kwargs)
    return queue
