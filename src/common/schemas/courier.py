"""
Courier schemas.
"""

from pydantic import BaseModel, ConfigDict, Field

from src.common.models.base import CourierStatus
from src.common.schemas.base import TimestampMixin


class CourierBase(BaseModel):
    """Base courier schema."""
    name: str = Field(..., min_length=1, max_length=255)
    phone: str | None = Field(None, max_length=20)


class CourierCreate(CourierBase):
    """Schema for creating a courier."""
    external_id: str = Field(..., min_length=1, max_length=64)


class CourierUpdate(BaseModel):
    """Schema for updating a courier."""
    name: str | None = Field(None, min_length=1, max_length=255)
    phone: str | None = Field(None, max_length=20)
    status: CourierStatus | None = None


class CourierLocationUpdate(BaseModel):
    """Schema for updating courier location."""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    speed_kmh: float | None = Field(None, ge=0, le=100)
    heading: float | None = Field(None, ge=0, le=360)
    battery_level: int | None = Field(None, ge=0, le=100)


class CourierResponse(CourierBase, TimestampMixin):
    """Courier response schema."""
    id: int
    external_id: str
    latitude: float | None
    longitude: float | None
    h3_index: str | None
    status: CourierStatus
    avg_speed_kmh: float
    rating: float
    total_deliveries: int

    model_config = ConfigDict(from_attributes=True)

