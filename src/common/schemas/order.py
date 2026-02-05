"""
Order schemas.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.common.models.base import OrderStatus
from src.common.schemas.base import TimestampMixin


class OrderBase(BaseModel):
    """Base order schema."""
    restaurant_id: int
    customer_latitude: float = Field(..., ge=-90, le=90)
    customer_longitude: float = Field(..., ge=-180, le=180)


class OrderCreate(OrderBase):
    """Schema for creating an order."""
    external_id: str = Field(..., min_length=1, max_length=64)


class OrderAssignment(BaseModel):
    """Schema for order assignment."""
    order_id: int
    courier_id: int
    estimated_delivery_time_minutes: int | None = None


class OrderStatusUpdate(BaseModel):
    """Schema for updating order status."""
    status: OrderStatus


class OrderResponse(OrderBase, TimestampMixin):
    """Order response schema."""
    id: int
    external_id: str
    courier_id: int | None
    customer_h3_index: str
    status: OrderStatus
    surge_coefficient: float
    estimated_prep_time_minutes: int | None
    estimated_delivery_time_minutes: int | None
    actual_delivery_time_minutes: int | None
    assigned_at: datetime | None
    picked_up_at: datetime | None
    delivered_at: datetime | None

    model_config = ConfigDict(from_attributes=True)

