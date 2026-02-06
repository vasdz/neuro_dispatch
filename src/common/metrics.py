"""
Prometheus Metrics and Observability Module.

Production-grade metrics for:
- Request latency and throughput
- ML model inference metrics
- Business metrics (orders, couriers, surge)
- System health metrics

Senior+ implementation with:
- Custom collectors
- Histogram buckets optimized for latency SLOs
- Label cardinality management
- Metrics aggregation
"""

import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator
from enum import Enum

from src.common.logging import get_logger

logger = get_logger(__name__)


class MetricType(str, Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


# Histogram buckets for different use cases
LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0
)

ML_INFERENCE_BUCKETS = (
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5
)

ASSIGNMENT_BUCKETS = (
    0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0
)


class MetricsCollector:
    """
    Central metrics collector.

    In-memory metrics storage with Prometheus-compatible format.
    Can be extended to push to Prometheus Pushgateway or expose /metrics endpoint.
    """

    def __init__(self):
        self._counters: dict[str, dict[tuple, float]] = {}
        self._gauges: dict[str, dict[tuple, float]] = {}
        self._histograms: dict[str, dict[str, Any]] = {}
        self._summaries: dict[str, dict[tuple, list[float]]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def register_counter(
        self,
        name: str,
        description: str,
        labels: list[str] | None = None,
    ) -> None:
        """Register a counter metric."""
        self._counters[name] = {}
        self._metadata[name] = {
            "type": MetricType.COUNTER,
            "description": description,
            "labels": labels or [],
        }

    def register_gauge(
        self,
        name: str,
        description: str,
        labels: list[str] | None = None,
    ) -> None:
        """Register a gauge metric."""
        self._gauges[name] = {}
        self._metadata[name] = {
            "type": MetricType.GAUGE,
            "description": description,
            "labels": labels or [],
        }

    def register_histogram(
        self,
        name: str,
        description: str,
        buckets: tuple[float, ...] = LATENCY_BUCKETS,
        labels: list[str] | None = None,
    ) -> None:
        """Register a histogram metric."""
        self._histograms[name] = {
            "buckets": buckets,
            "observations": {},
        }
        self._metadata[name] = {
            "type": MetricType.HISTOGRAM,
            "description": description,
            "labels": labels or [],
            "buckets": buckets,
        }

    def inc_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment a counter."""
        if name not in self._counters:
            return

        label_key = tuple(sorted((labels or {}).items()))
        if label_key not in self._counters[name]:
            self._counters[name][label_key] = 0.0
        self._counters[name][label_key] += value

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Set a gauge value."""
        if name not in self._gauges:
            return

        label_key = tuple(sorted((labels or {}).items()))
        self._gauges[name][label_key] = value

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Observe a histogram value."""
        if name not in self._histograms:
            return

        label_key = tuple(sorted((labels or {}).items()))
        if label_key not in self._histograms[name]["observations"]:
            self._histograms[name]["observations"][label_key] = []
        self._histograms[name]["observations"][label_key].append(value)

    @contextmanager
    def timer(
        self,
        histogram_name: str,
        labels: dict[str, str] | None = None,
    ) -> Generator[None, None, None]:
        """Context manager for timing operations."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.observe_histogram(histogram_name, duration, labels)

    def get_counter(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ) -> float:
        """Get counter value."""
        if name not in self._counters:
            return 0.0
        label_key = tuple(sorted((labels or {}).items()))
        return self._counters[name].get(label_key, 0.0)

    def get_gauge(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ) -> float:
        """Get gauge value."""
        if name not in self._gauges:
            return 0.0
        label_key = tuple(sorted((labels or {}).items()))
        return self._gauges[name].get(label_key, 0.0)

    def get_histogram_stats(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ) -> dict[str, float]:
        """Get histogram statistics."""
        if name not in self._histograms:
            return {}

        label_key = tuple(sorted((labels or {}).items()))
        observations = self._histograms[name]["observations"].get(label_key, [])

        if not observations:
            return {
                "count": 0,
                "sum": 0.0,
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        sorted_obs = sorted(observations)
        count = len(sorted_obs)

        return {
            "count": count,
            "sum": sum(sorted_obs),
            "avg": sum(sorted_obs) / count,
            "min": sorted_obs[0],
            "max": sorted_obs[-1],
            "p50": sorted_obs[int(count * 0.50)],
            "p95": sorted_obs[min(int(count * 0.95), count - 1)],
            "p99": sorted_obs[min(int(count * 0.99), count - 1)],
        }

    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []

        # Counters
        for name, values in self._counters.items():
            meta = self._metadata.get(name, {})
            lines.append(f"# HELP {name} {meta.get('description', '')}")
            lines.append(f"# TYPE {name} counter")
            for label_key, value in values.items():
                labels_str = self._format_labels(label_key)
                lines.append(f"{name}{labels_str} {value}")

        # Gauges
        for name, values in self._gauges.items():
            meta = self._metadata.get(name, {})
            lines.append(f"# HELP {name} {meta.get('description', '')}")
            lines.append(f"# TYPE {name} gauge")
            for label_key, value in values.items():
                labels_str = self._format_labels(label_key)
                lines.append(f"{name}{labels_str} {value}")

        # Histograms
        for name, data in self._histograms.items():
            meta = self._metadata.get(name, {})
            lines.append(f"# HELP {name} {meta.get('description', '')}")
            lines.append(f"# TYPE {name} histogram")

            buckets = data["buckets"]
            for label_key, observations in data["observations"].items():
                sorted_obs = sorted(observations)
                count = len(sorted_obs)
                total = sum(sorted_obs)

                # Bucket counts
                for bucket in buckets:
                    bucket_count = sum(1 for o in sorted_obs if o <= bucket)
                    labels_str = self._format_labels(label_key, {"le": str(bucket)})
                    lines.append(f"{name}_bucket{labels_str} {bucket_count}")

                # +Inf bucket
                labels_str = self._format_labels(label_key, {"le": "+Inf"})
                lines.append(f"{name}_bucket{labels_str} {count}")

                # Sum and count
                labels_str = self._format_labels(label_key)
                lines.append(f"{name}_sum{labels_str} {total}")
                lines.append(f"{name}_count{labels_str} {count}")

        return "\n".join(lines)

    def _format_labels(
        self,
        label_key: tuple,
        extra: dict[str, str] | None = None,
    ) -> str:
        """Format labels for Prometheus output."""
        labels = dict(label_key)
        if extra:
            labels.update(extra)

        if not labels:
            return ""

        parts = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(parts) + "}"

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        for name in self._counters:
            self._counters[name] = {}
        for name in self._gauges:
            self._gauges[name] = {}
        for name in self._histograms:
            self._histograms[name]["observations"] = {}


# Global metrics instance
metrics = MetricsCollector()


# ============================================================================
# REGISTER APPLICATION METRICS
# ============================================================================

def register_default_metrics() -> None:
    """Register all default application metrics."""

    # HTTP metrics
    metrics.register_counter(
        "http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status"],
    )
    metrics.register_histogram(
        "http_request_duration_seconds",
        "HTTP request duration in seconds",
        LATENCY_BUCKETS,
        ["method", "endpoint"],
    )

    # Dispatch metrics
    metrics.register_counter(
        "dispatch_assignments_total",
        "Total order assignments",
        ["algorithm", "status"],
    )
    metrics.register_histogram(
        "dispatch_assignment_duration_seconds",
        "Time to assign an order",
        ASSIGNMENT_BUCKETS,
        ["algorithm"],
    )
    metrics.register_gauge(
        "dispatch_pending_orders",
        "Number of pending orders",
    )
    metrics.register_gauge(
        "dispatch_available_couriers",
        "Number of available couriers",
    )

    # ML inference metrics
    metrics.register_counter(
        "ml_predictions_total",
        "Total ML predictions",
        ["model", "status"],
    )
    metrics.register_histogram(
        "ml_inference_duration_seconds",
        "ML inference duration",
        ML_INFERENCE_BUCKETS,
        ["model"],
    )
    metrics.register_gauge(
        "ml_model_version",
        "Current model version",
        ["model"],
    )
    metrics.register_histogram(
        "ml_prediction_error",
        "Prediction error (actual - predicted)",
        (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
        ["model"],
    )

    # Pricing metrics
    metrics.register_counter(
        "pricing_calculations_total",
        "Total pricing calculations",
        ["strategy"],
    )
    metrics.register_histogram(
        "pricing_surge_coefficient",
        "Surge coefficient distribution",
        (1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0),
        ["zone_type"],
    )
    metrics.register_gauge(
        "pricing_avg_surge",
        "Average surge coefficient",
        ["zone_type"],
    )

    # Feature store metrics
    metrics.register_counter(
        "feature_store_requests_total",
        "Feature store requests",
        ["store", "operation", "status"],
    )
    metrics.register_histogram(
        "feature_store_latency_seconds",
        "Feature store operation latency",
        ML_INFERENCE_BUCKETS,
        ["store", "operation"],
    )
    metrics.register_gauge(
        "feature_store_cache_size",
        "Feature store cache size",
    )

    # Database metrics
    metrics.register_counter(
        "db_queries_total",
        "Total database queries",
        ["operation"],
    )
    metrics.register_histogram(
        "db_query_duration_seconds",
        "Database query duration",
        LATENCY_BUCKETS,
        ["operation"],
    )
    metrics.register_gauge(
        "db_pool_size",
        "Database connection pool size",
    )
    metrics.register_gauge(
        "db_pool_used",
        "Database connections in use",
    )

    # Redis metrics
    metrics.register_counter(
        "redis_operations_total",
        "Total Redis operations",
        ["operation", "status"],
    )
    metrics.register_histogram(
        "redis_operation_duration_seconds",
        "Redis operation duration",
        ML_INFERENCE_BUCKETS,
        ["operation"],
    )

    # Business metrics
    metrics.register_counter(
        "orders_created_total",
        "Total orders created",
        ["zone_type"],
    )
    metrics.register_counter(
        "orders_completed_total",
        "Total orders completed",
        ["zone_type"],
    )
    metrics.register_counter(
        "orders_cancelled_total",
        "Total orders cancelled",
        ["reason"],
    )
    metrics.register_histogram(
        "order_delivery_time_minutes",
        "Order delivery time in minutes",
        (5, 10, 15, 20, 25, 30, 40, 50, 60, 90),
        ["zone_type"],
    )
    metrics.register_gauge(
        "active_couriers",
        "Number of active couriers",
        ["status"],
    )

    logger.info("Registered default metrics")


# Register metrics on import
register_default_metrics()


# ============================================================================
# DECORATOR FOR METRIC COLLECTION
# ============================================================================

def track_time(histogram_name: str, labels: dict[str, str] | None = None):
    """Decorator to track function execution time."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with metrics.timer(histogram_name, labels):
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with metrics.timer(histogram_name, labels):
                return func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def count_calls(counter_name: str, labels: dict[str, str] | None = None):
    """Decorator to count function calls."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            metrics.inc_counter(counter_name, 1, labels)
            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            metrics.inc_counter(counter_name, 1, labels)
            return result

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator

