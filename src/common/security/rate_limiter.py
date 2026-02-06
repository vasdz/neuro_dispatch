"""
Rate Limiting Module.

Protects the API from abuse with:
- Token bucket algorithm
- Sliding window rate limiting
- Per-user and per-IP limits
- Endpoint-specific limits
- Graceful degradation

Security best practices:
- Defense in depth
- Fail securely
- Comprehensive logging
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import Optional, Callable
import hashlib

from fastapi import Request, HTTPException, status, Depends

from src.common.logging import get_logger
from src.common.config import settings

logger = get_logger(__name__)


class RateLimitExceeded(HTTPException):
    """Exception raised when rate limit is exceeded."""

    def __init__(
        self,
        detail: str = "Rate limit exceeded",
        retry_after: int = 60,
    ):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers={"Retry-After": str(retry_after)},
        )
        self.retry_after = retry_after


@dataclass
class RateLimitConfig:
    """Configuration for a rate limit rule."""
    requests: int  # Number of requests allowed
    window_seconds: int  # Time window in seconds

    # Optional identification
    name: str = "default"

    # Burst allowance (for token bucket)
    burst: Optional[int] = None

    @property
    def requests_per_second(self) -> float:
        return self.requests / self.window_seconds


@dataclass
class RateLimitState:
    """State for tracking rate limit usage."""
    tokens: float = 0.0
    last_update: float = field(default_factory=time.time)
    request_count: int = 0
    window_start: float = field(default_factory=time.time)


class TokenBucketLimiter:
    """
    Token bucket rate limiter.

    Allows bursts while enforcing average rate.
    More forgiving than fixed window approaches.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.rate = config.requests_per_second
        self.capacity = config.burst or config.requests
        self._states: dict[str, RateLimitState] = {}

    def is_allowed(self, key: str) -> tuple[bool, dict]:
        """
        Check if request is allowed.

        Returns (is_allowed, metadata).
        """
        now = time.time()

        if key not in self._states:
            self._states[key] = RateLimitState(tokens=self.capacity, last_update=now)

        state = self._states[key]

        # Refill tokens based on time passed
        time_passed = now - state.last_update
        state.tokens = min(
            self.capacity,
            state.tokens + time_passed * self.rate,
        )
        state.last_update = now

        # Check if we have tokens
        if state.tokens >= 1:
            state.tokens -= 1
            return True, {
                "remaining": int(state.tokens),
                "limit": self.capacity,
                "reset": int(now + (self.capacity - state.tokens) / self.rate),
            }

        # Calculate retry after
        retry_after = int((1 - state.tokens) / self.rate) + 1

        return False, {
            "remaining": 0,
            "limit": self.capacity,
            "reset": int(now + retry_after),
            "retry_after": retry_after,
        }

    def reset(self, key: str):
        """Reset rate limit for a key."""
        if key in self._states:
            del self._states[key]


class SlidingWindowLimiter:
    """
    Sliding window rate limiter.

    More accurate than fixed window but slightly more expensive.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> tuple[bool, dict]:
        """Check if request is allowed."""
        now = time.time()
        window_start = now - self.config.window_seconds

        # Clean old requests
        requests = self._requests[key]
        self._requests[key] = [t for t in requests if t > window_start]

        # Check limit
        current_count = len(self._requests[key])

        if current_count < self.config.requests:
            self._requests[key].append(now)
            return True, {
                "remaining": self.config.requests - current_count - 1,
                "limit": self.config.requests,
                "window": self.config.window_seconds,
            }

        # Calculate retry after (when oldest request expires)
        oldest = min(self._requests[key]) if self._requests[key] else now
        retry_after = int(oldest + self.config.window_seconds - now) + 1

        return False, {
            "remaining": 0,
            "limit": self.config.requests,
            "retry_after": max(1, retry_after),
        }

    def reset(self, key: str):
        """Reset rate limit for a key."""
        if key in self._requests:
            del self._requests[key]


class RateLimiter:
    """
    Main rate limiter with multiple strategies.

    Supports:
    - Per-IP limiting
    - Per-user limiting
    - Per-endpoint limiting
    - Combined limits
    """

    # Default limits
    DEFAULT_LIMITS = {
        "global": RateLimitConfig(requests=1000, window_seconds=60, name="global"),
        "auth": RateLimitConfig(requests=10, window_seconds=60, name="auth"),
        "api": RateLimitConfig(requests=100, window_seconds=60, name="api"),
        "heavy": RateLimitConfig(requests=10, window_seconds=60, name="heavy"),
    }

    def __init__(
        self,
        default_config: Optional[RateLimitConfig] = None,
        use_token_bucket: bool = True,
    ):
        self.default_config = default_config or self.DEFAULT_LIMITS["api"]
        self.use_token_bucket = use_token_bucket

        # Limiters by name
        self._limiters: dict[str, TokenBucketLimiter | SlidingWindowLimiter] = {}

        # Initialize default limiters
        for name, config in self.DEFAULT_LIMITS.items():
            self._create_limiter(name, config)

    def _create_limiter(self, name: str, config: RateLimitConfig):
        """Create a limiter for a config."""
        if self.use_token_bucket:
            self._limiters[name] = TokenBucketLimiter(config)
        else:
            self._limiters[name] = SlidingWindowLimiter(config)

    def get_limiter(self, name: str) -> TokenBucketLimiter | SlidingWindowLimiter:
        """Get or create a limiter by name."""
        if name not in self._limiters:
            self._create_limiter(name, self.default_config)
        return self._limiters[name]

    def check(
        self,
        key: str,
        limiter_name: str = "api",
    ) -> tuple[bool, dict]:
        """Check if request is allowed."""
        limiter = self.get_limiter(limiter_name)
        return limiter.is_allowed(key)

    def check_and_raise(
        self,
        key: str,
        limiter_name: str = "api",
    ):
        """Check rate limit and raise exception if exceeded."""
        allowed, metadata = self.check(key, limiter_name)

        if not allowed:
            logger.warning(
                "Rate limit exceeded",
                key=key,
                limiter=limiter_name,
                retry_after=metadata.get("retry_after", 60),
            )
            raise RateLimitExceeded(
                detail=f"Rate limit exceeded. Try again in {metadata.get('retry_after', 60)} seconds.",
                retry_after=metadata.get("retry_after", 60),
            )

        return metadata

    @staticmethod
    def get_client_key(request: Request) -> str:
        """Get unique identifier for client."""
        # Try to get real IP (considering proxies)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"

        return ip

    @staticmethod
    def get_user_key(user_id: str) -> str:
        """Get key for user-based limiting."""
        return f"user:{user_id}"

    @staticmethod
    def get_endpoint_key(request: Request) -> str:
        """Get key for endpoint-based limiting."""
        method = request.method
        path = request.url.path
        return hashlib.md5(f"{method}:{path}".encode()).hexdigest()[:8]


# Global rate limiter instance
_rate_limiter = RateLimiter()


def rate_limit(
    requests: int = 100,
    window_seconds: int = 60,
    key_func: Optional[Callable[[Request], str]] = None,
):
    """
    FastAPI dependency for rate limiting.

    Usage:
        @router.get("/api/data")
        async def get_data(
            _: None = Depends(rate_limit(requests=100, window_seconds=60))
        ):
            return {"data": "value"}
    """
    config = RateLimitConfig(requests=requests, window_seconds=window_seconds)
    limiter = TokenBucketLimiter(config)

    async def rate_limit_dependency(request: Request) -> dict:
        if key_func:
            key = key_func(request)
        else:
            key = RateLimiter.get_client_key(request)

        allowed, metadata = limiter.is_allowed(key)

        if not allowed:
            raise RateLimitExceeded(
                detail="Rate limit exceeded",
                retry_after=metadata.get("retry_after", 60),
            )

        return metadata

    return rate_limit_dependency


def rate_limit_by_user(requests: int = 100, window_seconds: int = 60):
    """Rate limit by authenticated user."""
    async def dependency(request: Request) -> dict:
        from src.common.security.auth import get_optional_user

        user = await get_optional_user(
            request.headers.get("Authorization", "").replace("Bearer ", "")
        )

        if user:
            key = RateLimiter.get_user_key(user.sub)
        else:
            key = RateLimiter.get_client_key(request)

        return _rate_limiter.check_and_raise(key, "api")

    return dependency


class RateLimitMiddleware:
    """
    ASGI middleware for global rate limiting.

    Applies rate limiting to all requests before routing.
    """

    def __init__(self, app, limiter: RateLimiter = None):
        self.app = app
        self.limiter = limiter or _rate_limiter

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Get client IP
        client = scope.get("client")
        ip = client[0] if client else "unknown"

        # Check global rate limit
        allowed, metadata = self.limiter.check(ip, "global")

        if not allowed:
            # Return 429 response
            response = {
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"retry-after", str(metadata.get("retry_after", 60)).encode()),
                ],
            }
            await send(response)

            body = b'{"detail": "Rate limit exceeded"}'
            await send({"type": "http.response.body", "body": body})
            return

        await self.app(scope, receive, send)

