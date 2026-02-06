"""
Feature Store Module.

Centralized feature management for ML models following Feast-like patterns.
Provides offline (batch) and online (real-time) feature serving.

Senior+ implementation with:
- Feature definitions with metadata
- Online/offline store abstraction
- Feature versioning
- TTL-based expiration
- Redis-backed online store
- PostgreSQL-backed offline store
"""

from src.feature_store.definitions import (
    Feature,
    FeatureGroup,
    FeatureType,
    get_feature_group,
    list_feature_groups,
)
from src.feature_store.store import (
    FeatureStore,
    get_feature_store,
)
from src.feature_store.registry import (
    FeatureRegistry,
)

__all__ = [
    "Feature",
    "FeatureGroup",
    "FeatureType",
    "FeatureStore",
    "FeatureRegistry",
    "get_feature_group",
    "get_feature_store",
    "list_feature_groups",
]

