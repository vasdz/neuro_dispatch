"""
Feature Registry - Feature Versioning and Schema Management.

Provides:
- Feature schema versioning
- Schema validation
- Migration support
- Feature lineage tracking

Senior+ implementation for production ML systems.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import hashlib
import json

from src.common.logging import get_logger
from src.feature_store.definitions import (
    FeatureGroup,
    get_feature_group,
    list_feature_groups,
)

logger = get_logger(__name__)


@dataclass
class FeatureVersion:
    """Represents a specific version of a feature group."""
    group_name: str
    version: str
    schema_hash: str
    created_at: datetime
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class FeatureRegistry:
    """
    Central registry for feature definitions.

    Manages:
    - Feature group registration
    - Version control
    - Schema validation
    - Backward compatibility checks
    """

    def __init__(self):
        self._versions: dict[str, list[FeatureVersion]] = {}
        self._active_versions: dict[str, str] = {}
        self._initialize_from_definitions()

    def _initialize_from_definitions(self) -> None:
        """Initialize registry from static definitions."""
        for group_name in list_feature_groups():
            group = get_feature_group(group_name)
            if group:
                self._register_group(group)

    def _register_group(self, group: FeatureGroup) -> None:
        """Register a feature group version."""
        schema_hash = self._compute_schema_hash(group)

        version = FeatureVersion(
            group_name=group.name,
            version=group.version,
            schema_hash=schema_hash,
            created_at=datetime.now(timezone.utc),
            is_active=True,
        )

        if group.name not in self._versions:
            self._versions[group.name] = []

        self._versions[group.name].append(version)
        self._active_versions[group.name] = group.version

        logger.debug(
            f"Registered feature group: {group.name} v{group.version}",
            schema_hash=schema_hash[:8],
        )

    def _compute_schema_hash(self, group: FeatureGroup) -> str:
        """Compute hash of feature group schema."""
        schema_data = {
            "name": group.name,
            "entity": group.entity,
            "features": sorted([
                {
                    "name": f.name,
                    "dtype": f.dtype.value,
                }
                for f in group.features
            ], key=lambda x: x["name"]),
        }

        schema_json = json.dumps(schema_data, sort_keys=True)
        return hashlib.sha256(schema_json.encode()).hexdigest()

    def get_active_version(self, group_name: str) -> str | None:
        """Get active version for a feature group."""
        return self._active_versions.get(group_name)

    def get_version_history(self, group_name: str) -> list[FeatureVersion]:
        """Get version history for a feature group."""
        return self._versions.get(group_name, [])

    def validate_schema(self, group_name: str, data: dict[str, Any]) -> list[str]:
        """
        Validate data against feature group schema.

        Returns list of validation errors.
        """
        errors = []
        group = get_feature_group(group_name)

        if not group:
            errors.append(f"Unknown feature group: {group_name}")
            return errors

        feature_names = {f.name for f in group.features}

        for key in data.keys():
            if key not in feature_names:
                errors.append(f"Unknown feature: {key}")

        for feature in group.features:
            if feature.name in data:
                value = data[feature.name]
                if not feature.validate(value):
                    errors.append(
                        f"Validation failed for {feature.name}: {value}"
                    )

        return errors

    def check_backward_compatibility(
        self,
        group_name: str,
        old_version: str,
        new_version: str,
    ) -> tuple[bool, list[str]]:
        """
        Check if new version is backward compatible with old version.

        Returns:
            (is_compatible, list of breaking changes)
        """
        breaking_changes = []

        versions = self._versions.get(group_name, [])
        old_ver = None
        new_ver = None

        for v in versions:
            if v.version == old_version:
                old_ver = v
            if v.version == new_version:
                new_ver = v

        if not old_ver or not new_ver:
            return False, ["Version not found"]

        # Same schema hash = compatible
        if old_ver.schema_hash == new_ver.schema_hash:
            return True, []

        # Check for removed features (breaking)
        old_group = get_feature_group(group_name)
        new_group = get_feature_group(group_name)

        if old_group and new_group:
            old_features = {f.name for f in old_group.features}
            new_features = {f.name for f in new_group.features}

            removed = old_features - new_features
            if removed:
                breaking_changes.append(
                    f"Removed features: {removed}"
                )

        return len(breaking_changes) == 0, breaking_changes

    def get_feature_lineage(self, feature_name: str) -> dict[str, Any]:
        """
        Get lineage information for a feature.

        Returns information about how the feature is computed.
        """
        for group_name in list_feature_groups():
            group = get_feature_group(group_name)
            if group:
                for feature in group.features:
                    if feature.name == feature_name:
                        return {
                            "name": feature.name,
                            "group": group_name,
                            "entity": group.entity,
                            "dtype": feature.dtype.value,
                            "aggregation": feature.aggregation.value if feature.aggregation else None,
                            "window": feature.window.total_seconds() if feature.window else None,
                            "tags": feature.tags,
                            "description": feature.description,
                        }

        return {}

    def export_schema(self) -> dict[str, Any]:
        """Export complete feature schema for documentation."""
        schema = {
            "version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "feature_groups": {},
        }

        for group_name in list_feature_groups():
            group = get_feature_group(group_name)
            if group:
                schema["feature_groups"][group_name] = {
                    "version": group.version,
                    "entity": group.entity,
                    "description": group.description,
                    "online": group.online,
                    "offline": group.offline,
                    "features": [f.to_dict() for f in group.features],
                }

        return schema

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        total_features = 0
        for group_name in list_feature_groups():
            group = get_feature_group(group_name)
            if group:
                total_features += len(group.features)

        return {
            "total_groups": len(list_feature_groups()),
            "total_features": total_features,
            "active_versions": dict(self._active_versions),
        }


# Global registry instance
_registry: FeatureRegistry | None = None


def get_registry() -> FeatureRegistry:
    """Get global feature registry instance."""
    global _registry
    if _registry is None:
        _registry = FeatureRegistry()
    return _registry

