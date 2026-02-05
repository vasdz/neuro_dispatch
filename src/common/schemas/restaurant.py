"""
Restaurant schemas.
"""

from pydantic import BaseModel, ConfigDict, Field

from src.common.schemas.base import TimestampMixin


class RestaurantBase(BaseModel):
    """Base restaurant schema."""
    name: str = Field(..., min_length=1, max_length=255)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    avg_prep_time_minutes: int = Field(default=15, ge=1, le=120)
    is_active: bool = Field(default=True)


class RestaurantCreate(RestaurantBase):
    """Schema for creating a restaurant."""
    external_id: str = Field(..., min_length=1, max_length=64)


class RestaurantUpdate(BaseModel):
    """Schema for updating a restaurant."""
    name: str | None = Field(None, min_length=1, max_length=255)
    latitude: float | None = Field(None, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)
    avg_prep_time_minutes: int | None = Field(None, ge=1, le=120)
    is_active: bool | None = None


class RestaurantResponse(RestaurantBase, TimestampMixin):
    """Restaurant response schema."""
    id: int
    external_id: str
    h3_index: str

    model_config = ConfigDict(from_attributes=True)

