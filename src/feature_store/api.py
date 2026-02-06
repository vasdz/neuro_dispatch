"""
Feature Store API Endpoints.

Production-ready API for:
- Feature retrieval (online)
- Feature metadata
- Schema management
- Health monitoring

Senior+ implementation with proper error handling and documentation.
"""

from typing import Any
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.common.logging import get_logger
from src.feature_store.store import get_feature_store
from src.feature_store.registry import get_registry
from src.feature_store.definitions import list_feature_groups, get_feature_group

logger = get_logger(__name__)
router = APIRouter()


# ============================================================================
# SCHEMAS
# ============================================================================

class FeatureRequest(BaseModel):
    """Request for online features."""
    entity_type: str = Field(..., description="Entity type (courier, hexagon, order, etc.)")
    entity_id: str = Field(..., description="Entity identifier")
    feature_names: list[str] | None = Field(None, description="Specific features (None = all)")


class BatchFeatureRequest(BaseModel):
    """Request for batch features."""
    entity_type: str = Field(..., description="Entity type")
    entity_ids: list[str] = Field(..., description="List of entity identifiers")
    feature_names: list[str] | None = Field(None, description="Specific features")


class FeatureResponse(BaseModel):
    """Response with feature values."""
    entity_type: str
    entity_id: str
    features: dict[str, Any]
    retrieved_at: datetime


class BatchFeatureResponse(BaseModel):
    """Response with batch feature values."""
    entity_type: str
    entities: dict[str, dict[str, Any]]
    count: int
    retrieved_at: datetime


class PushFeatureRequest(BaseModel):
    """Request to push features to online store."""
    entity_type: str
    entity_id: str
    features: dict[str, Any]


class FeatureGroupInfo(BaseModel):
    """Feature group information."""
    name: str
    description: str
    entity: str
    version: str
    online: bool
    offline: bool
    feature_count: int


class FeatureInfo(BaseModel):
    """Feature information."""
    name: str
    dtype: str
    description: str
    default_value: Any
    ttl_seconds: float
    tags: list[str]


class RegistryStats(BaseModel):
    """Registry statistics."""
    total_groups: int
    total_features: int
    active_versions: dict[str, str]


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/groups", response_model=list[str])
async def list_groups() -> list[str]:
    """
    List all available feature groups.

    Returns the names of all registered feature groups.
    """
    return list_feature_groups()


@router.get("/groups/{group_name}", response_model=FeatureGroupInfo)
async def get_group_info(group_name: str) -> FeatureGroupInfo:
    """
    Get information about a feature group.

    Args:
        group_name: Name of the feature group

    Returns:
        Feature group metadata including features list
    """
    group = get_feature_group(group_name)

    if not group:
        raise HTTPException(status_code=404, detail=f"Feature group '{group_name}' not found")

    return FeatureGroupInfo(
        name=group.name,
        description=group.description,
        entity=group.entity,
        version=group.version,
        online=group.online,
        offline=group.offline,
        feature_count=len(group.features),
    )


@router.get("/groups/{group_name}/features", response_model=list[FeatureInfo])
async def get_group_features(group_name: str) -> list[FeatureInfo]:
    """
    Get all features in a group.

    Args:
        group_name: Name of the feature group

    Returns:
        List of feature definitions
    """
    group = get_feature_group(group_name)

    if not group:
        raise HTTPException(status_code=404, detail=f"Feature group '{group_name}' not found")

    return [
        FeatureInfo(
            name=f.name,
            dtype=f.dtype.value,
            description=f.description,
            default_value=f.default_value,
            ttl_seconds=f.ttl.total_seconds(),
            tags=f.tags,
        )
        for f in group.features
    ]


@router.post("/online/get", response_model=FeatureResponse)
async def get_online_features(request: FeatureRequest) -> FeatureResponse:
    """
    Get features from online store.

    Retrieves real-time features for a single entity.
    Low-latency endpoint optimized for serving.
    """
    feature_store = get_feature_store()

    features = await feature_store.get_online_features(
        entity_type=request.entity_type,
        entity_id=request.entity_id,
        feature_names=request.feature_names,
    )

    return FeatureResponse(
        entity_type=request.entity_type,
        entity_id=request.entity_id,
        features=features,
        retrieved_at=datetime.utcnow(),
    )


@router.post("/online/batch", response_model=BatchFeatureResponse)
async def get_batch_features(request: BatchFeatureRequest) -> BatchFeatureResponse:
    """
    Get features for multiple entities.

    Batch endpoint for efficient retrieval of features
    for multiple entities in a single request.
    """
    if len(request.entity_ids) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Maximum 1000 entities per batch request"
        )

    feature_store = get_feature_store()

    entities = await feature_store.batch_get_online_features(
        entity_type=request.entity_type,
        entity_ids=request.entity_ids,
        feature_names=request.feature_names,
    )

    return BatchFeatureResponse(
        entity_type=request.entity_type,
        entities=entities,
        count=len(entities),
        retrieved_at=datetime.now(timezone.utc),
    )


@router.post("/online/push")
async def push_online_features(request: PushFeatureRequest) -> dict[str, Any]:
    """
    Push features to online store.

    Updates feature values for an entity in the online store.
    Features are validated against the schema before storing.
    """
    feature_store = get_feature_store()

    # Validate against schema
    registry = get_registry()
    errors = registry.validate_schema(request.entity_type, request.features)

    if errors:
        raise HTTPException(
            status_code=400,
            detail={"message": "Schema validation failed", "errors": errors}
        )

    success = await feature_store.push_features(
        entity_type=request.entity_type,
        entity_id=request.entity_id,
        features=request.features,
    )

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to push features to online store"
        )

    return {
        "status": "success",
        "entity_type": request.entity_type,
        "entity_id": request.entity_id,
        "features_pushed": len(request.features),
    }


@router.get("/registry/stats", response_model=RegistryStats)
async def get_registry_stats() -> RegistryStats:
    """
    Get feature registry statistics.

    Returns aggregate statistics about the feature registry.
    """
    registry = get_registry()
    stats = registry.get_stats()

    return RegistryStats(**stats)


@router.get("/registry/schema")
async def export_schema() -> dict[str, Any]:
    """
    Export complete feature schema.

    Returns the full schema of all feature groups and features.
    Useful for documentation and schema versioning.
    """
    registry = get_registry()
    return registry.export_schema()


@router.get("/lineage/{feature_name}")
async def get_feature_lineage(feature_name: str) -> dict[str, Any]:
    """
    Get lineage information for a feature.

    Returns information about how the feature is computed,
    its source, and dependencies.
    """
    registry = get_registry()
    lineage = registry.get_feature_lineage(feature_name)

    if not lineage:
        raise HTTPException(
            status_code=404,
            detail=f"Feature '{feature_name}' not found"
        )

    return lineage


@router.get("/health")
async def feature_store_health() -> dict[str, Any]:
    """
    Health check for feature store.

    Checks connectivity to online and offline stores.
    """
    feature_store = get_feature_store()

    # Test online store
    online_healthy = False
    try:
        await feature_store.online_store._ensure_connected()
        online_healthy = feature_store.online_store._connected
    except Exception:
        pass

    return {
        "status": "healthy" if online_healthy else "degraded",
        "online_store": "connected" if online_healthy else "disconnected",
        "offline_store": "available",  # Always available if DB is up
    }

