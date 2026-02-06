"""
Feature Definitions for NeuroDispatch.

Defines all features used across ML models with:
- Feature metadata (name, type, description)
- Aggregation windows
- Default values
- Validation rules

Follows Feature Store best practices similar to Feast, Tecton, Featureform.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Any, Callable


class FeatureType(str, Enum):
    """Supported feature data types."""
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOL = "bool"
    LIST_INT = "list_int"
    LIST_FLOAT = "list_float"
    EMBEDDING = "embedding"


class AggregationType(str, Enum):
    """Aggregation types for time-window features."""
    SUM = "sum"
    MEAN = "mean"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    STD = "std"
    LAST = "last"
    FIRST = "first"


@dataclass
class Feature:
    """
    Feature definition with metadata.

    Attributes:
        name: Unique feature name
        dtype: Feature data type
        description: Human-readable description
        default_value: Default value when feature is missing
        ttl: Time-to-live for online store caching
        aggregation: Aggregation type for time-window features
        window: Time window for aggregation
        tags: Custom tags for filtering/grouping
        validator: Optional validation function
    """
    name: str
    dtype: FeatureType
    description: str
    default_value: Any = None
    ttl: timedelta = field(default_factory=lambda: timedelta(hours=1))
    aggregation: AggregationType | None = None
    window: timedelta | None = None
    tags: list[str] = field(default_factory=list)
    validator: Callable[[Any], bool] | None = None

    def validate(self, value: Any) -> bool:
        """Validate feature value."""
        if self.validator:
            return self.validator(value)
        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "dtype": self.dtype.value,
            "description": self.description,
            "default_value": self.default_value,
            "ttl_seconds": self.ttl.total_seconds(),
            "aggregation": self.aggregation.value if self.aggregation else None,
            "window_seconds": self.window.total_seconds() if self.window else None,
            "tags": self.tags,
        }


@dataclass
class FeatureGroup:
    """
    Group of related features.

    Attributes:
        name: Group name
        description: Group description
        entity: Entity type (courier, order, hexagon, etc.)
        features: List of features in the group
        version: Group version for schema evolution
        online: Whether to serve from online store
        offline: Whether to serve from offline store
    """
    name: str
    description: str
    entity: str
    features: list[Feature]
    version: str = "1.0"
    online: bool = True
    offline: bool = True

    def get_feature(self, name: str) -> Feature | None:
        """Get feature by name."""
        for f in self.features:
            return f if f.name == name else None
        return None

    def get_feature_names(self) -> list[str]:
        """Get list of feature names."""
        return [f.name for f in self.features]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "entity": self.entity,
            "version": self.version,
            "online": self.online,
            "offline": self.offline,
            "features": [f.to_dict() for f in self.features],
        }


# ============================================================================
# COURIER FEATURES
# ============================================================================

COURIER_FEATURES = FeatureGroup(
    name="courier_features",
    description="Features related to courier performance and state",
    entity="courier",
    version="1.0",
    features=[
        Feature(
            name="courier_avg_speed_kmh",
            dtype=FeatureType.FLOAT,
            description="Average courier speed in km/h over last 7 days",
            default_value=15.0,
            ttl=timedelta(hours=6),
            aggregation=AggregationType.MEAN,
            window=timedelta(days=7),
            tags=["performance", "speed"],
        ),
        Feature(
            name="courier_completed_orders_24h",
            dtype=FeatureType.INT,
            description="Number of completed orders in last 24 hours",
            default_value=0,
            ttl=timedelta(minutes=30),
            aggregation=AggregationType.COUNT,
            window=timedelta(hours=24),
            tags=["performance", "workload"],
        ),
        Feature(
            name="courier_avg_delivery_time_min",
            dtype=FeatureType.FLOAT,
            description="Average delivery time in minutes over last 30 days",
            default_value=25.0,
            ttl=timedelta(hours=12),
            aggregation=AggregationType.MEAN,
            window=timedelta(days=30),
            tags=["performance", "efficiency"],
        ),
        Feature(
            name="courier_rating",
            dtype=FeatureType.FLOAT,
            description="Courier rating (1-5 scale)",
            default_value=4.5,
            ttl=timedelta(hours=24),
            tags=["quality"],
            validator=lambda x: 1.0 <= x <= 5.0,
        ),
        Feature(
            name="courier_acceptance_rate",
            dtype=FeatureType.FLOAT,
            description="Order acceptance rate (0-1)",
            default_value=0.8,
            ttl=timedelta(hours=6),
            aggregation=AggregationType.MEAN,
            window=timedelta(days=7),
            tags=["reliability"],
            validator=lambda x: 0.0 <= x <= 1.0,
        ),
        Feature(
            name="courier_current_load",
            dtype=FeatureType.INT,
            description="Current number of active orders",
            default_value=0,
            ttl=timedelta(seconds=60),
            tags=["realtime", "capacity"],
        ),
        Feature(
            name="courier_last_location_h3",
            dtype=FeatureType.STRING,
            description="Last known H3 hexagon index",
            default_value="",
            ttl=timedelta(seconds=30),
            tags=["realtime", "location"],
        ),
        Feature(
            name="courier_vehicle_type",
            dtype=FeatureType.STRING,
            description="Vehicle type (bike, car, foot)",
            default_value="bike",
            ttl=timedelta(days=1),
            tags=["static"],
        ),
        Feature(
            name="courier_experience_days",
            dtype=FeatureType.INT,
            description="Days since courier started working",
            default_value=0,
            ttl=timedelta(days=1),
            tags=["static"],
        ),
        Feature(
            name="courier_cancellation_rate",
            dtype=FeatureType.FLOAT,
            description="Order cancellation rate (0-1) over last 30 days",
            default_value=0.02,
            ttl=timedelta(hours=12),
            aggregation=AggregationType.MEAN,
            window=timedelta(days=30),
            tags=["reliability"],
            validator=lambda x: 0.0 <= x <= 1.0,
        ),
    ],
)


# ============================================================================
# HEXAGON/ZONE FEATURES
# ============================================================================

HEXAGON_FEATURES = FeatureGroup(
    name="hexagon_features",
    description="Features related to geographic hexagons (H3)",
    entity="hexagon",
    version="1.0",
    features=[
        Feature(
            name="hex_avg_orders_per_hour",
            dtype=FeatureType.FLOAT,
            description="Average orders per hour in hexagon",
            default_value=0.0,
            ttl=timedelta(hours=1),
            aggregation=AggregationType.MEAN,
            window=timedelta(days=7),
            tags=["demand"],
        ),
        Feature(
            name="hex_demand_rolling_24h",
            dtype=FeatureType.FLOAT,
            description="Rolling 24h demand count",
            default_value=0.0,
            ttl=timedelta(minutes=30),
            aggregation=AggregationType.SUM,
            window=timedelta(hours=24),
            tags=["demand", "realtime"],
        ),
        Feature(
            name="hex_active_couriers",
            dtype=FeatureType.INT,
            description="Number of active couriers in hexagon",
            default_value=0,
            ttl=timedelta(seconds=60),
            tags=["supply", "realtime"],
        ),
        Feature(
            name="hex_avg_delivery_time_min",
            dtype=FeatureType.FLOAT,
            description="Average delivery time from this hexagon",
            default_value=25.0,
            ttl=timedelta(hours=6),
            aggregation=AggregationType.MEAN,
            window=timedelta(days=14),
            tags=["performance"],
        ),
        Feature(
            name="hex_restaurant_count",
            dtype=FeatureType.INT,
            description="Number of restaurants in hexagon",
            default_value=0,
            ttl=timedelta(days=1),
            tags=["static"],
        ),
        Feature(
            name="hex_zone_type",
            dtype=FeatureType.STRING,
            description="Zone type (center, inner, middle, outer)",
            default_value="outer",
            ttl=timedelta(days=7),
            tags=["static"],
        ),
        Feature(
            name="hex_surge_coefficient",
            dtype=FeatureType.FLOAT,
            description="Current surge pricing coefficient",
            default_value=1.0,
            ttl=timedelta(minutes=5),
            tags=["pricing", "realtime"],
            validator=lambda x: 1.0 <= x <= 5.0,
        ),
        Feature(
            name="hex_weather_impact",
            dtype=FeatureType.FLOAT,
            description="Weather impact factor (1.0 = normal, >1.0 = bad weather)",
            default_value=1.0,
            ttl=timedelta(minutes=30),
            tags=["external"],
        ),
    ],
)


# ============================================================================
# ORDER FEATURES
# ============================================================================

ORDER_FEATURES = FeatureGroup(
    name="order_features",
    description="Features related to orders",
    entity="order",
    version="1.0",
    features=[
        Feature(
            name="order_estimated_prep_time_min",
            dtype=FeatureType.FLOAT,
            description="Estimated preparation time in minutes",
            default_value=15.0,
            ttl=timedelta(hours=1),
            tags=["estimation"],
        ),
        Feature(
            name="order_distance_km",
            dtype=FeatureType.FLOAT,
            description="Distance from restaurant to customer in km",
            default_value=2.0,
            ttl=timedelta(hours=24),
            tags=["routing"],
        ),
        Feature(
            name="order_item_count",
            dtype=FeatureType.INT,
            description="Number of items in order",
            default_value=3,
            ttl=timedelta(hours=24),
            tags=["size"],
        ),
        Feature(
            name="order_total_amount",
            dtype=FeatureType.FLOAT,
            description="Total order amount in rubles",
            default_value=500.0,
            ttl=timedelta(hours=24),
            tags=["value"],
        ),
        Feature(
            name="order_is_priority",
            dtype=FeatureType.BOOL,
            description="Whether order is priority/urgent",
            default_value=False,
            ttl=timedelta(hours=24),
            tags=["priority"],
        ),
        Feature(
            name="order_customer_orders_count",
            dtype=FeatureType.INT,
            description="Number of previous orders from this customer",
            default_value=0,
            ttl=timedelta(days=1),
            tags=["customer"],
        ),
    ],
)


# ============================================================================
# RESTAURANT FEATURES
# ============================================================================

RESTAURANT_FEATURES = FeatureGroup(
    name="restaurant_features",
    description="Features related to restaurants",
    entity="restaurant",
    version="1.0",
    features=[
        Feature(
            name="restaurant_avg_prep_time_min",
            dtype=FeatureType.FLOAT,
            description="Average preparation time in minutes",
            default_value=15.0,
            ttl=timedelta(hours=6),
            aggregation=AggregationType.MEAN,
            window=timedelta(days=14),
            tags=["performance"],
        ),
        Feature(
            name="restaurant_rating",
            dtype=FeatureType.FLOAT,
            description="Restaurant rating (1-5)",
            default_value=4.0,
            ttl=timedelta(days=1),
            tags=["quality"],
            validator=lambda x: 1.0 <= x <= 5.0,
        ),
        Feature(
            name="restaurant_order_volume_24h",
            dtype=FeatureType.INT,
            description="Number of orders in last 24 hours",
            default_value=0,
            ttl=timedelta(minutes=30),
            aggregation=AggregationType.COUNT,
            window=timedelta(hours=24),
            tags=["workload"],
        ),
        Feature(
            name="restaurant_current_queue",
            dtype=FeatureType.INT,
            description="Current number of orders in preparation",
            default_value=0,
            ttl=timedelta(seconds=60),
            tags=["realtime", "capacity"],
        ),
        Feature(
            name="restaurant_is_busy",
            dtype=FeatureType.BOOL,
            description="Whether restaurant is currently busy",
            default_value=False,
            ttl=timedelta(minutes=5),
            tags=["realtime"],
        ),
    ],
)


# ============================================================================
# TEMPORAL FEATURES
# ============================================================================

TEMPORAL_FEATURES = FeatureGroup(
    name="temporal_features",
    description="Time-based features for demand prediction",
    entity="timestamp",
    version="1.0",
    online=True,
    offline=True,
    features=[
        Feature(
            name="hour_of_day",
            dtype=FeatureType.INT,
            description="Hour of day (0-23)",
            default_value=12,
            ttl=timedelta(seconds=60),
            tags=["temporal"],
            validator=lambda x: 0 <= x <= 23,
        ),
        Feature(
            name="day_of_week",
            dtype=FeatureType.INT,
            description="Day of week (0=Monday, 6=Sunday)",
            default_value=0,
            ttl=timedelta(seconds=60),
            tags=["temporal"],
            validator=lambda x: 0 <= x <= 6,
        ),
        Feature(
            name="is_weekend",
            dtype=FeatureType.BOOL,
            description="Whether it's weekend",
            default_value=False,
            ttl=timedelta(seconds=60),
            tags=["temporal"],
        ),
        Feature(
            name="is_lunch_hour",
            dtype=FeatureType.BOOL,
            description="Whether it's lunch time (11:00-14:00)",
            default_value=False,
            ttl=timedelta(seconds=60),
            tags=["temporal"],
        ),
        Feature(
            name="is_dinner_hour",
            dtype=FeatureType.BOOL,
            description="Whether it's dinner time (18:00-21:00)",
            default_value=False,
            ttl=timedelta(seconds=60),
            tags=["temporal"],
        ),
        Feature(
            name="is_holiday",
            dtype=FeatureType.BOOL,
            description="Whether it's a public holiday",
            default_value=False,
            ttl=timedelta(hours=24),
            tags=["temporal", "external"],
        ),
        Feature(
            name="month",
            dtype=FeatureType.INT,
            description="Month of year (1-12)",
            default_value=1,
            ttl=timedelta(hours=1),
            tags=["temporal"],
            validator=lambda x: 1 <= x <= 12,
        ),
    ],
)


# ============================================================================
# FEATURE GROUP REGISTRY
# ============================================================================

_FEATURE_GROUPS: dict[str, FeatureGroup] = {
    "courier": COURIER_FEATURES,
    "hexagon": HEXAGON_FEATURES,
    "order": ORDER_FEATURES,
    "restaurant": RESTAURANT_FEATURES,
    "temporal": TEMPORAL_FEATURES,
}


def get_feature_group(name: str) -> FeatureGroup | None:
    """Get feature group by name."""
    return _FEATURE_GROUPS.get(name)


def list_feature_groups() -> list[str]:
    """List all available feature groups."""
    return list(_FEATURE_GROUPS.keys())


def get_all_features() -> list[Feature]:
    """Get all features from all groups."""
    features = []
    for group in _FEATURE_GROUPS.values():
        features.extend(group.features)
    return features


def get_features_by_tag(tag: str) -> list[Feature]:
    """Get all features with a specific tag."""
    features = []
    for group in _FEATURE_GROUPS.values():
        for feature in group.features:
            if tag in feature.tags:
                features.append(feature)
    return features

