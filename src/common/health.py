"""
Health Check Module.

Production-grade health checks following Kubernetes patterns:
- Liveness: Is the service alive?
- Readiness: Is the service ready to accept traffic?
- Startup: Has the service started successfully?

Senior+ implementation with:
- Dependency health checks
- Circuit breaker pattern
- Detailed diagnostics
- SLA monitoring
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Awaitable
import asyncio
import time

from src.common.logging import get_logger

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class CheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class HealthReport:
    """Complete health report."""
    status: HealthStatus
    version: str
    uptime_seconds: float
    checks: list[CheckResult]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "version": self.version,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "timestamp": self.timestamp.isoformat(),
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "latency_ms": round(c.latency_ms, 2),
                    "details": c.details,
                }
                for c in self.checks
            ],
        }


class CircuitBreaker:
    """
    Circuit breaker for health checks.

    Prevents cascading failures by:
    - Tracking failure counts
    - Opening circuit after threshold
    - Half-open state for recovery testing
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        half_open_requests: int = 1,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests

        self._failures: dict[str, int] = {}
        self._last_failure: dict[str, datetime] = {}
        self._state: dict[str, str] = {}  # closed, open, half-open

    def is_open(self, name: str) -> bool:
        """Check if circuit is open for a check."""
        state = self._state.get(name, "closed")

        if state == "open":
            # Check if recovery timeout has passed
            last_fail = self._last_failure.get(name)
            if last_fail:
                elapsed = (datetime.now(timezone.utc) - last_fail).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self._state[name] = "half-open"
                    return False
            return True

        return False

    def record_success(self, name: str) -> None:
        """Record successful check."""
        self._failures[name] = 0
        self._state[name] = "closed"

    def record_failure(self, name: str) -> None:
        """Record failed check."""
        self._failures[name] = self._failures.get(name, 0) + 1
        self._last_failure[name] = datetime.now(timezone.utc)

        if self._failures[name] >= self.failure_threshold:
            self._state[name] = "open"
            logger.warning(f"Circuit breaker opened for: {name}")

    def get_state(self, name: str) -> str:
        """Get circuit state."""
        return self._state.get(name, "closed")


HealthCheckFunc = Callable[[], Awaitable[CheckResult]]


class HealthChecker:
    """
    Central health check coordinator.

    Manages:
    - Registration of health checks
    - Parallel execution
    - Result aggregation
    - Circuit breaker integration
    """

    def __init__(self, version: str = "0.1.0"):
        self.version = version
        self._start_time = datetime.now(timezone.utc)
        self._checks: dict[str, HealthCheckFunc] = {}
        self._check_types: dict[str, str] = {}  # liveness, readiness, startup
        self._circuit_breaker = CircuitBreaker()
        self._last_results: dict[str, CheckResult] = {}
        self._cache_ttl = timedelta(seconds=5)
        self._last_check_time: datetime | None = None

    def register(
        self,
        name: str,
        check_func: HealthCheckFunc,
        check_type: str = "readiness",
    ) -> None:
        """
        Register a health check.

        Args:
            name: Unique check name
            check_func: Async function returning CheckResult
            check_type: Type of check (liveness, readiness, startup)
        """
        self._checks[name] = check_func
        self._check_types[name] = check_type
        logger.debug(f"Registered health check: {name} ({check_type})")

    async def run_checks(
        self,
        check_type: str | None = None,
        use_cache: bool = True,
    ) -> HealthReport:
        """
        Run all registered health checks.

        Args:
            check_type: Filter by type (None = all)
            use_cache: Use cached results if fresh

        Returns:
            Complete health report
        """
        # Check cache
        if use_cache and self._last_check_time:
            elapsed = datetime.now(timezone.utc) - self._last_check_time
            if elapsed < self._cache_ttl:
                cached_checks = list(self._last_results.values())
                if check_type:
                    cached_checks = [
                        c for c in cached_checks
                        if self._check_types.get(c.name) == check_type
                    ]
                return self._build_report(cached_checks)

        # Filter checks by type
        checks_to_run = {
            name: func
            for name, func in self._checks.items()
            if check_type is None or self._check_types.get(name) == check_type
        }

        # Run checks in parallel
        results = await asyncio.gather(
            *[self._run_single_check(name, func) for name, func in checks_to_run.items()],
            return_exceptions=True,
        )

        # Process results
        check_results = []
        for result in results:
            if isinstance(result, Exception):
                check_results.append(
                    CheckResult(
                        name="unknown",
                        status=HealthStatus.UNHEALTHY,
                        message=str(result),
                    )
                )
            else:
                check_results.append(result)
                self._last_results[result.name] = result

        self._last_check_time = datetime.now(timezone.utc)

        return self._build_report(check_results)

    async def _run_single_check(
        self,
        name: str,
        func: HealthCheckFunc,
    ) -> CheckResult:
        """Run a single health check with circuit breaker."""
        # Check circuit breaker
        if self._circuit_breaker.is_open(name):
            return CheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="Circuit breaker open",
                details={"circuit_state": "open"},
            )

        start = time.perf_counter()

        try:
            result = await asyncio.wait_for(func(), timeout=5.0)
            latency = (time.perf_counter() - start) * 1000
            result.latency_ms = latency

            if result.status == HealthStatus.HEALTHY:
                self._circuit_breaker.record_success(name)
            else:
                self._circuit_breaker.record_failure(name)

            return result

        except asyncio.TimeoutError:
            self._circuit_breaker.record_failure(name)
            return CheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="Check timed out",
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            self._circuit_breaker.record_failure(name)
            return CheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=(time.perf_counter() - start) * 1000,
            )

    def _build_report(self, checks: list[CheckResult]) -> HealthReport:
        """Build health report from check results."""
        # Determine overall status
        if not checks:
            overall_status = HealthStatus.UNKNOWN
        elif any(c.status == HealthStatus.UNHEALTHY for c in checks):
            overall_status = HealthStatus.UNHEALTHY
        elif any(c.status == HealthStatus.DEGRADED for c in checks):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()

        return HealthReport(
            status=overall_status,
            version=self.version,
            uptime_seconds=uptime,
            checks=checks,
        )

    async def liveness(self) -> HealthReport:
        """Run liveness checks only."""
        return await self.run_checks(check_type="liveness")

    async def readiness(self) -> HealthReport:
        """Run readiness checks only."""
        return await self.run_checks(check_type="readiness")

    async def startup(self) -> HealthReport:
        """Run startup checks only."""
        return await self.run_checks(check_type="startup", use_cache=False)


# Global health checker instance
health_checker = HealthChecker()


# ============================================================================
# BUILT-IN HEALTH CHECKS
# ============================================================================

async def check_database() -> CheckResult:
    """Check database connectivity."""
    from src.common.database import check_db_connection

    try:
        is_healthy = await check_db_connection()

        if is_healthy:
            return CheckResult(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database connection OK",
            )
        else:
            return CheckResult(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message="Database connection failed",
            )
    except Exception as e:
        return CheckResult(
            name="database",
            status=HealthStatus.UNHEALTHY,
            message=f"Database error: {str(e)}",
        )


async def check_redis() -> CheckResult:
    """Check Redis connectivity."""
    from src.common.redis_client import redis_client

    try:
        is_healthy = await redis_client.health_check()

        if is_healthy:
            return CheckResult(
                name="redis",
                status=HealthStatus.HEALTHY,
                message="Redis connection OK",
            )
        else:
            return CheckResult(
                name="redis",
                status=HealthStatus.DEGRADED,
                message="Redis unavailable (cache disabled)",
            )
    except Exception as e:
        return CheckResult(
            name="redis",
            status=HealthStatus.DEGRADED,
            message=f"Redis error: {str(e)}",
        )


async def check_disk_space() -> CheckResult:
    """Check available disk space."""
    import shutil

    try:
        total, used, free = shutil.disk_usage("/")
        free_percent = (free / total) * 100

        status = HealthStatus.HEALTHY
        if free_percent < 10:
            status = HealthStatus.UNHEALTHY
        elif free_percent < 20:
            status = HealthStatus.DEGRADED

        return CheckResult(
            name="disk_space",
            status=status,
            message=f"{free_percent:.1f}% free",
            details={
                "total_gb": round(total / (1024**3), 2),
                "used_gb": round(used / (1024**3), 2),
                "free_gb": round(free / (1024**3), 2),
                "free_percent": round(free_percent, 2),
            },
        )
    except Exception as e:
        return CheckResult(
            name="disk_space",
            status=HealthStatus.UNKNOWN,
            message=f"Cannot check disk: {str(e)}",
        )


async def check_memory() -> CheckResult:
    """Check available memory."""
    try:
        import psutil

        memory = psutil.virtual_memory()
        available_percent = memory.available / memory.total * 100

        status = HealthStatus.HEALTHY
        if available_percent < 10:
            status = HealthStatus.UNHEALTHY
        elif available_percent < 20:
            status = HealthStatus.DEGRADED

        return CheckResult(
            name="memory",
            status=status,
            message=f"{available_percent:.1f}% available",
            details={
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_percent": round(memory.percent, 2),
            },
        )
    except ImportError:
        return CheckResult(
            name="memory",
            status=HealthStatus.UNKNOWN,
            message="psutil not installed",
        )
    except Exception as e:
        return CheckResult(
            name="memory",
            status=HealthStatus.UNKNOWN,
            message=f"Cannot check memory: {str(e)}",
        )


async def check_ml_models() -> CheckResult:
    """Check ML model availability."""
    from pathlib import Path

    models_dir = Path("data/models")

    if not models_dir.exists():
        return CheckResult(
            name="ml_models",
            status=HealthStatus.DEGRADED,
            message="Models directory not found",
            details={"models": []},
        )

    model_files = list(models_dir.glob("*.joblib"))

    if not model_files:
        return CheckResult(
            name="ml_models",
            status=HealthStatus.DEGRADED,
            message="No trained models found",
            details={"models": []},
        )

    return CheckResult(
        name="ml_models",
        status=HealthStatus.HEALTHY,
        message=f"{len(model_files)} models available",
        details={"models": [f.name for f in model_files]},
    )


def register_default_checks() -> None:
    """Register default health checks."""
    health_checker.register("database", check_database, "readiness")
    health_checker.register("redis", check_redis, "readiness")
    health_checker.register("disk_space", check_disk_space, "liveness")
    # health_checker.register("memory", check_memory, "liveness")  # Requires psutil
    health_checker.register("ml_models", check_ml_models, "readiness")

    logger.info("Registered default health checks")


# Register on import
register_default_checks()

