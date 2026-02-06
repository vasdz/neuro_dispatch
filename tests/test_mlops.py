"""
Tests for Feature Store Module.

Comprehensive tests covering:
- Feature definitions
- Online store operations
- Offline store operations
- Feature registry
- Cache behavior
"""

import pytest
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.feature_store.definitions import (
    Feature,
    FeatureGroup,
    FeatureType,
    AggregationType,
    get_feature_group,
    list_feature_groups,
    get_all_features,
    get_features_by_tag,
    COURIER_FEATURES,
    HEXAGON_FEATURES,
)
from src.feature_store.store import (
    OnlineStore,
    FeatureStore,
    get_feature_store,
)
from src.feature_store.registry import (
    FeatureRegistry,
    get_registry,
)


class TestFeatureDefinitions:
    """Tests for feature definitions."""

    def test_feature_creation(self):
        """Test creating a feature."""
        feature = Feature(
            name="test_feature",
            dtype=FeatureType.FLOAT,
            description="A test feature",
            default_value=0.0,
            ttl=timedelta(hours=1),
            tags=["test"],
        )

        assert feature.name == "test_feature"
        assert feature.dtype == FeatureType.FLOAT
        assert feature.default_value == 0.0
        assert feature.ttl == timedelta(hours=1)
        assert "test" in feature.tags

    def test_feature_validation(self):
        """Test feature value validation."""
        feature = Feature(
            name="rating",
            dtype=FeatureType.FLOAT,
            description="Rating (1-5)",
            validator=lambda x: 1.0 <= x <= 5.0,
        )

        assert feature.validate(3.5) is True
        assert feature.validate(0.5) is False
        assert feature.validate(5.5) is False

    def test_feature_to_dict(self):
        """Test feature serialization."""
        feature = Feature(
            name="test",
            dtype=FeatureType.INT,
            description="Test",
            default_value=0,
            aggregation=AggregationType.SUM,
            window=timedelta(hours=24),
        )

        data = feature.to_dict()

        assert data["name"] == "test"
        assert data["dtype"] == "int"
        assert data["aggregation"] == "sum"
        assert data["window_seconds"] == 86400.0

    def test_feature_group_creation(self):
        """Test creating a feature group."""
        features = [
            Feature(name="f1", dtype=FeatureType.FLOAT, description="F1"),
            Feature(name="f2", dtype=FeatureType.INT, description="F2"),
        ]

        group = FeatureGroup(
            name="test_group",
            description="Test group",
            entity="test",
            features=features,
        )

        assert group.name == "test_group"
        assert len(group.features) == 2
        assert group.get_feature_names() == ["f1", "f2"]

    def test_courier_features_defined(self):
        """Test that courier features are properly defined."""
        assert COURIER_FEATURES is not None
        assert len(COURIER_FEATURES.features) > 0
        assert COURIER_FEATURES.entity == "courier"

        # Check specific features exist
        feature_names = COURIER_FEATURES.get_feature_names()
        assert "courier_avg_speed_kmh" in feature_names
        assert "courier_completed_orders_24h" in feature_names

    def test_hexagon_features_defined(self):
        """Test that hexagon features are properly defined."""
        assert HEXAGON_FEATURES is not None
        assert len(HEXAGON_FEATURES.features) > 0
        assert HEXAGON_FEATURES.entity == "hexagon"

    def test_get_feature_group(self):
        """Test getting feature group by name."""
        group = get_feature_group("courier")
        assert group is not None
        assert group.entity == "courier"

        group = get_feature_group("nonexistent")
        assert group is None

    def test_list_feature_groups(self):
        """Test listing all feature groups."""
        groups = list_feature_groups()

        assert "courier" in groups
        assert "hexagon" in groups
        assert "order" in groups
        assert "restaurant" in groups
        assert "temporal" in groups

    def test_get_all_features(self):
        """Test getting all features."""
        features = get_all_features()

        assert len(features) > 20  # Should have many features
        assert all(isinstance(f, Feature) for f in features)

    def test_get_features_by_tag(self):
        """Test filtering features by tag."""
        performance_features = get_features_by_tag("performance")

        assert len(performance_features) > 0
        assert all("performance" in f.tags for f in performance_features)

        realtime_features = get_features_by_tag("realtime")
        assert len(realtime_features) > 0


class TestOnlineStore:
    """Tests for online (Redis) store."""

    @pytest.fixture
    def online_store(self):
        """Create online store instance."""
        return OnlineStore()

    def test_make_key(self, online_store):
        """Test key generation."""
        key = online_store._make_key("courier", "123")
        assert key == "fs:feature:courier:123"

    def test_serialize_value(self, online_store):
        """Test value serialization."""
        assert online_store._serialize_value(42) == "42"
        assert online_store._serialize_value(3.14) == "3.14"
        assert online_store._serialize_value([1, 2, 3]) == "[1, 2, 3]"

    def test_deserialize_value(self, online_store):
        """Test value deserialization."""
        assert online_store._deserialize_value("42") == 42
        assert online_store._deserialize_value("3.14") == 3.14
        assert online_store._deserialize_value("true") is True
        assert online_store._deserialize_value("[1, 2, 3]") == [1, 2, 3]
        assert online_store._deserialize_value(b"hello") == "hello"

    def test_get_defaults(self, online_store):
        """Test getting default values."""
        defaults = online_store._get_defaults("courier", None)

        assert "courier_avg_speed_kmh" in defaults
        assert defaults["courier_avg_speed_kmh"] == 15.0

    def test_get_default_single(self, online_store):
        """Test getting single default value."""
        default = online_store._get_default("courier", "courier_avg_speed_kmh")
        assert default == 15.0

        default = online_store._get_default("courier", "nonexistent")
        assert default is None


class TestFeatureStore:
    """Tests for unified feature store."""

    @pytest.fixture
    def feature_store(self):
        """Create feature store instance."""
        return FeatureStore()

    def test_get_feature_metadata(self, feature_store):
        """Test getting feature metadata."""
        metadata = feature_store.get_feature_metadata("courier")

        assert metadata["name"] == "courier_features"
        assert metadata["entity"] == "courier"
        assert "features" in metadata

    def test_list_features(self, feature_store):
        """Test listing all features."""
        features = feature_store.list_features()

        assert len(features) > 0
        assert all("name" in f for f in features)
        assert all("dtype" in f for f in features)


class TestFeatureRegistry:
    """Tests for feature registry."""

    @pytest.fixture
    def registry(self):
        """Create registry instance."""
        return FeatureRegistry()

    def test_get_active_version(self, registry):
        """Test getting active version."""
        version = registry.get_active_version("courier_features")
        assert version == "1.0"

    def test_get_version_history(self, registry):
        """Test getting version history."""
        history = registry.get_version_history("courier_features")

        assert len(history) >= 1
        assert history[0].version == "1.0"

    def test_validate_schema(self, registry):
        """Test schema validation."""
        # Valid data
        errors = registry.validate_schema("courier", {
            "courier_avg_speed_kmh": 20.0,
            "courier_rating": 4.5,
        })
        assert len(errors) == 0

        # Unknown feature
        errors = registry.validate_schema("courier", {
            "unknown_feature": 1.0,
        })
        assert len(errors) > 0

        # Invalid value
        errors = registry.validate_schema("courier", {
            "courier_rating": 10.0,  # Should be 1-5
        })
        assert len(errors) > 0

    def test_get_feature_lineage(self, registry):
        """Test getting feature lineage."""
        lineage = registry.get_feature_lineage("courier_avg_speed_kmh")

        assert lineage["name"] == "courier_avg_speed_kmh"
        assert lineage["group"] == "courier"
        assert lineage["dtype"] == "float"

        # Non-existent feature
        lineage = registry.get_feature_lineage("nonexistent")
        assert lineage == {}

    def test_export_schema(self, registry):
        """Test exporting complete schema."""
        schema = registry.export_schema()

        assert "version" in schema
        assert "generated_at" in schema
        assert "feature_groups" in schema
        assert "courier" in schema["feature_groups"]

    def test_get_stats(self, registry):
        """Test getting registry stats."""
        stats = registry.get_stats()

        assert "total_groups" in stats
        assert "total_features" in stats
        assert stats["total_groups"] >= 5
        assert stats["total_features"] > 20

    def test_global_registry(self):
        """Test global registry singleton."""
        registry1 = get_registry()
        registry2 = get_registry()

        assert registry1 is registry2


class TestMetrics:
    """Tests for metrics module."""

    def test_metrics_collector(self):
        """Test metrics collector."""
        from src.common.metrics import MetricsCollector

        collector = MetricsCollector()

        # Register and use counter
        collector.register_counter("test_counter", "Test counter", ["label"])
        collector.inc_counter("test_counter", 1, {"label": "a"})
        collector.inc_counter("test_counter", 2, {"label": "a"})

        assert collector.get_counter("test_counter", {"label": "a"}) == 3

    def test_gauge(self):
        """Test gauge metric."""
        from src.common.metrics import MetricsCollector

        collector = MetricsCollector()
        collector.register_gauge("test_gauge", "Test gauge")
        collector.set_gauge("test_gauge", 42.0)

        assert collector.get_gauge("test_gauge") == 42.0

    def test_histogram(self):
        """Test histogram metric."""
        from src.common.metrics import MetricsCollector

        collector = MetricsCollector()
        collector.register_histogram("test_histogram", "Test histogram")

        for i in range(100):
            collector.observe_histogram("test_histogram", i / 100)

        stats = collector.get_histogram_stats("test_histogram")

        assert stats["count"] == 100
        assert stats["min"] == 0.0
        assert stats["max"] == 0.99
        assert 0.4 < stats["p50"] < 0.6

    def test_timer_context_manager(self):
        """Test timer context manager."""
        from src.common.metrics import MetricsCollector
        import time

        collector = MetricsCollector()
        collector.register_histogram("test_timer", "Test timer")

        with collector.timer("test_timer"):
            time.sleep(0.01)

        stats = collector.get_histogram_stats("test_timer")
        assert stats["count"] == 1
        assert stats["min"] >= 0.01

    def test_prometheus_format(self):
        """Test Prometheus format export."""
        from src.common.metrics import MetricsCollector

        collector = MetricsCollector()
        collector.register_counter("http_requests", "HTTP requests", ["method"])
        collector.inc_counter("http_requests", 10, {"method": "GET"})

        output = collector.export_prometheus_format()

        assert "# HELP http_requests" in output
        assert "# TYPE http_requests counter" in output
        assert 'http_requests{method="GET"} 10' in output


class TestHealthChecks:
    """Tests for health check module."""

    def test_check_result(self):
        """Test check result creation."""
        from src.common.health import CheckResult, HealthStatus

        result = CheckResult(
            name="test",
            status=HealthStatus.HEALTHY,
            message="OK",
        )

        assert result.name == "test"
        assert result.status == HealthStatus.HEALTHY

    def test_health_report(self):
        """Test health report."""
        from src.common.health import HealthReport, CheckResult, HealthStatus

        checks = [
            CheckResult(name="db", status=HealthStatus.HEALTHY),
            CheckResult(name="redis", status=HealthStatus.HEALTHY),
        ]

        report = HealthReport(
            status=HealthStatus.HEALTHY,
            version="0.1.0",
            uptime_seconds=100.0,
            checks=checks,
        )

        data = report.to_dict()

        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert len(data["checks"]) == 2

    def test_circuit_breaker(self):
        """Test circuit breaker."""
        from src.common.health import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)

        # Initially closed
        assert cb.is_open("test") is False
        assert cb.get_state("test") == "closed"

        # Record failures
        cb.record_failure("test")
        cb.record_failure("test")
        assert cb.is_open("test") is False

        cb.record_failure("test")  # Third failure
        assert cb.is_open("test") is True
        assert cb.get_state("test") == "open"

        # Record success resets
        cb.record_success("test")
        assert cb.is_open("test") is False

