"""
Redis client for caching and real-time state.
"""

from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as redis

from src.common.config import settings
from src.common.logging import get_logger

logger = get_logger(__name__)


class RedisClient:
    """Async Redis client wrapper."""

    def __init__(self) -> None:
        self._client: redis.Redis | None = None

    async def connect(self) -> None:
        """Initialize Redis connection."""
        self._client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        logger.info("Redis connection established", url=settings.redis_url)

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            logger.info("Redis connection closed")

    @property
    def client(self) -> redis.Redis:
        """Get Redis client instance."""
        if not self._client:
            raise RuntimeError("Redis not connected. Call connect() first.")
        return self._client

    async def get(self, key: str) -> str | None:
        """Get value by key."""
        return await self.client.get(key)

    async def set(
        self,
        key: str,
        value: str,
        ttl: int | None = None
    ) -> None:
        """Set value with optional TTL."""
        ttl = ttl or settings.redis_ttl_seconds
        await self.client.set(key, value, ex=ttl)

    async def delete(self, key: str) -> None:
        """Delete key."""
        await self.client.delete(key)

    async def hset(self, name: str, key: str, value: str) -> None:
        """Set hash field."""
        await self.client.hset(name, key, value)

    async def hget(self, name: str, key: str) -> str | None:
        """Get hash field."""
        return await self.client.hget(name, key)

    async def hgetall(self, name: str) -> dict[str, str]:
        """Get all hash fields."""
        return await self.client.hgetall(name)

    async def publish(self, channel: str, message: str) -> None:
        """Publish message to channel."""
        await self.client.publish(channel, message)

    async def health_check(self) -> bool:
        """Check Redis connection health."""
        try:
            await self.client.ping()
            return True
        except Exception as e:
            logger.error("Redis health check failed", error=str(e))
            return False


# Global Redis client instance
redis_client = RedisClient()


@asynccontextmanager
async def get_redis():
    """Context manager for Redis client."""
    try:
        yield redis_client
    finally:
        pass  # Connection managed by app lifecycle

