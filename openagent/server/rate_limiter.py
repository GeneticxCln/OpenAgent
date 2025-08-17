"""
Rate limiting middleware for OpenAgent server.

Provides configurable rate limiting per user/IP with different
strategies and storage backends.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional, Tuple

from fastapi import HTTPException, Request, status


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""

    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    burst_limit: int = 10  # For token bucket strategy
    enabled: bool = True


class RateLimitExceeded(HTTPException):
    """Custom exception for rate limit exceeded."""

    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=message,
            headers={"Retry-After": str(retry_after)},
        )


class SlidingWindowCounter:
    """Sliding window rate limiter implementation."""

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.requests: deque = deque()

    def is_allowed(self, limit: int) -> Tuple[bool, int]:
        """Check if request is allowed and return remaining requests."""
        current_time = time.time()

        # Remove requests outside the window
        while self.requests and self.requests[0] <= current_time - self.window_size:
            self.requests.popleft()

        if len(self.requests) < limit:
            self.requests.append(current_time)
            return True, limit - len(self.requests)

        # Calculate retry after based on oldest request
        retry_after = int(self.requests[0] + self.window_size - current_time) + 1
        return False, retry_after


class TokenBucket:
    """Token bucket rate limiter implementation."""

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()

    def is_allowed(self, tokens_requested: int = 1) -> Tuple[bool, int]:
        """Check if request is allowed and return remaining tokens."""
        current_time = time.time()

        # Refill tokens based on elapsed time
        elapsed = current_time - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = current_time

        if self.tokens >= tokens_requested:
            self.tokens -= tokens_requested
            return True, int(self.tokens)

        # Calculate retry after
        tokens_needed = tokens_requested - self.tokens
        retry_after = int(tokens_needed / self.refill_rate) + 1
        return False, retry_after


class FixedWindowCounter:
    """Fixed window rate limiter implementation."""

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.current_window = 0
        self.request_count = 0

    def is_allowed(self, limit: int) -> Tuple[bool, int]:
        """Check if request is allowed and return remaining requests."""
        current_time = time.time()
        current_window = int(current_time // self.window_size)

        if current_window != self.current_window:
            self.current_window = current_window
            self.request_count = 0

        if self.request_count < limit:
            self.request_count += 1
            return True, limit - self.request_count

        # Calculate retry after (time until next window)
        retry_after = int((current_window + 1) * self.window_size - current_time) + 1
        return False, retry_after


class RateLimiter:
    """Rate limiter with multiple strategies and storage backends."""

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.user_limiters: Dict[str, Dict[str, any]] = defaultdict(dict)
        self.ip_limiters: Dict[str, Dict[str, any]] = defaultdict(dict)

    def _get_limiter(self, key: str, limit: int, window: int, storage: Dict):
        """Get or create a rate limiter for the given key."""
        if key not in storage:
            if self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                storage[key] = SlidingWindowCounter(window)
            elif self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
                refill_rate = limit / window  # tokens per second
                storage[key] = TokenBucket(self.config.burst_limit, refill_rate)
            else:  # FIXED_WINDOW
                storage[key] = FixedWindowCounter(window)

        return storage[key]

    def _check_limit(
        self, identifier: str, limit: int, window: int, storage: Dict
    ) -> Tuple[bool, int]:
        """Check if request is within rate limit."""
        limiter = self._get_limiter(identifier, limit, window, storage)
        return limiter.is_allowed(limit)

    def check_user_rate_limit(self, user_id: str) -> Tuple[bool, str, int]:
        """Check rate limit for a specific user."""
        if not self.config.enabled:
            return True, "", 0

        # Check minute limit
        allowed, retry_after = self._check_limit(
            f"{user_id}:minute", self.config.requests_per_minute, 60, self.user_limiters
        )
        if not allowed:
            return (
                False,
                f"Rate limit exceeded: {self.config.requests_per_minute} requests per minute",
                retry_after,
            )

        # Check hour limit
        allowed, retry_after = self._check_limit(
            f"{user_id}:hour", self.config.requests_per_hour, 3600, self.user_limiters
        )
        if not allowed:
            return (
                False,
                f"Rate limit exceeded: {self.config.requests_per_hour} requests per hour",
                retry_after,
            )

        # Check day limit
        allowed, retry_after = self._check_limit(
            f"{user_id}:day", self.config.requests_per_day, 86400, self.user_limiters
        )
        if not allowed:
            return (
                False,
                f"Rate limit exceeded: {self.config.requests_per_day} requests per day",
                retry_after,
            )

        return True, "", 0

    def check_ip_rate_limit(self, ip_address: str) -> Tuple[bool, str, int]:
        """Check rate limit for a specific IP address."""
        if not self.config.enabled:
            return True, "", 0

        # Use slightly higher limits for IP-based limiting
        ip_minute_limit = self.config.requests_per_minute * 2
        ip_hour_limit = self.config.requests_per_hour * 2
        ip_day_limit = self.config.requests_per_day * 2

        # Check minute limit
        allowed, retry_after = self._check_limit(
            f"{ip_address}:minute", ip_minute_limit, 60, self.ip_limiters
        )
        if not allowed:
            return (
                False,
                f"IP rate limit exceeded: {ip_minute_limit} requests per minute",
                retry_after,
            )

        # Check hour limit
        allowed, retry_after = self._check_limit(
            f"{ip_address}:hour", ip_hour_limit, 3600, self.ip_limiters
        )
        if not allowed:
            return (
                False,
                f"IP rate limit exceeded: {ip_hour_limit} requests per hour",
                retry_after,
            )

        # Check day limit
        allowed, retry_after = self._check_limit(
            f"{ip_address}:day", ip_day_limit, 86400, self.ip_limiters
        )
        if not allowed:
            return (
                False,
                f"IP rate limit exceeded: {ip_day_limit} requests per day",
                retry_after,
            )

        return True, "", 0

    def get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers (for reverse proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to direct client IP
        if hasattr(request, "client") and request.client:
            return request.client.host

        return "unknown"

    async def check_request(
        self, request: Request, user_id: Optional[str] = None
    ) -> None:
        """Check rate limits for a request and raise exception if exceeded."""
        if not self.config.enabled:
            return

        client_ip = self.get_client_ip(request)

        # Check IP-based rate limit first
        ip_allowed, ip_message, ip_retry_after = self.check_ip_rate_limit(client_ip)
        if not ip_allowed:
            raise RateLimitExceeded(ip_message, ip_retry_after)

        # Check user-based rate limit if user is identified
        if user_id:
            user_allowed, user_message, user_retry_after = self.check_user_rate_limit(
                user_id
            )
            if not user_allowed:
                raise RateLimitExceeded(user_message, user_retry_after)

    def get_rate_limit_info(
        self, user_id: Optional[str] = None, ip_address: Optional[str] = None
    ) -> Dict[str, any]:
        """Get current rate limit information."""
        info = {
            "enabled": self.config.enabled,
            "strategy": self.config.strategy.value,
            "limits": {
                "requests_per_minute": self.config.requests_per_minute,
                "requests_per_hour": self.config.requests_per_hour,
                "requests_per_day": self.config.requests_per_day,
            },
        }

        if user_id and self.config.enabled:
            # Get remaining requests for user
            user_minute_key = f"{user_id}:minute"
            user_hour_key = f"{user_id}:hour"
            user_day_key = f"{user_id}:day"

            minute_limiter = self.user_limiters.get(user_minute_key)
            hour_limiter = self.user_limiters.get(user_hour_key)
            day_limiter = self.user_limiters.get(user_day_key)

            info["user"] = {
                "user_id": user_id,
                "remaining_minute": self._get_remaining_requests(
                    minute_limiter, self.config.requests_per_minute
                ),
                "remaining_hour": self._get_remaining_requests(
                    hour_limiter, self.config.requests_per_hour
                ),
                "remaining_day": self._get_remaining_requests(
                    day_limiter, self.config.requests_per_day
                ),
            }

        if ip_address and self.config.enabled:
            # Get remaining requests for IP
            ip_minute_limit = self.config.requests_per_minute * 2
            ip_hour_limit = self.config.requests_per_hour * 2
            ip_day_limit = self.config.requests_per_day * 2

            ip_minute_key = f"{ip_address}:minute"
            ip_hour_key = f"{ip_address}:hour"
            ip_day_key = f"{ip_address}:day"

            minute_limiter = self.ip_limiters.get(ip_minute_key)
            hour_limiter = self.ip_limiters.get(ip_hour_key)
            day_limiter = self.ip_limiters.get(ip_day_key)

            info["ip"] = {
                "ip_address": ip_address,
                "remaining_minute": self._get_remaining_requests(
                    minute_limiter, ip_minute_limit
                ),
                "remaining_hour": self._get_remaining_requests(
                    hour_limiter, ip_hour_limit
                ),
                "remaining_day": self._get_remaining_requests(
                    day_limiter, ip_day_limit
                ),
            }

        return info

    def _get_remaining_requests(self, limiter, limit: int) -> int:
        """Get remaining requests for a limiter."""
        if not limiter:
            return limit

        if isinstance(limiter, SlidingWindowCounter):
            current_time = time.time()
            # Remove expired requests
            while (
                limiter.requests
                and limiter.requests[0] <= current_time - limiter.window_size
            ):
                limiter.requests.popleft()
            return max(0, limit - len(limiter.requests))

        elif isinstance(limiter, TokenBucket):
            return max(0, int(limiter.tokens))

        elif isinstance(limiter, FixedWindowCounter):
            current_time = time.time()
            current_window = int(current_time // limiter.window_size)
            if current_window != limiter.current_window:
                return limit
            return max(0, limit - limiter.request_count)

        return limit

    def reset_user_limits(self, user_id: str) -> None:
        """Reset all rate limits for a user."""
        keys_to_remove = []
        for key in self.user_limiters:
            if key.startswith(f"{user_id}:"):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.user_limiters[key]

    def reset_ip_limits(self, ip_address: str) -> None:
        """Reset all rate limits for an IP address."""
        keys_to_remove = []
        for key in self.ip_limiters:
            if key.startswith(f"{ip_address}:"):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.ip_limiters[key]

    def cleanup_expired_limiters(self) -> None:
        """Clean up expired rate limiters to free memory."""
        current_time = time.time()

        # Clean up user limiters
        expired_keys = []
        for key, limiter in self.user_limiters.items():
            if isinstance(limiter, SlidingWindowCounter):
                while (
                    limiter.requests
                    and limiter.requests[0] <= current_time - limiter.window_size
                ):
                    limiter.requests.popleft()
                if not limiter.requests:
                    expired_keys.append(key)

        for key in expired_keys:
            del self.user_limiters[key]

        # Clean up IP limiters
        expired_keys = []
        for key, limiter in self.ip_limiters.items():
            if isinstance(limiter, SlidingWindowCounter):
                while (
                    limiter.requests
                    and limiter.requests[0] <= current_time - limiter.window_size
                ):
                    limiter.requests.popleft()
                if not limiter.requests:
                    expired_keys.append(key)

        for key in expired_keys:
            del self.ip_limiters[key]


# Global rate limiter instance
rate_limiter = RateLimiter()


async def rate_limit_dependency(request: Request) -> None:
    """FastAPI dependency for rate limiting."""
    await rate_limiter.check_request(request)
