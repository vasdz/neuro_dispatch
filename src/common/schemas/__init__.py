"""
Pydantic schemas for API validation.
"""

from src.common.schemas.base import (
    GeoPoint,
    HealthResponse,
    PaginatedResponse,
    TimestampMixin,
)
from src.common.schemas.restaurant import (
    RestaurantCreate,
    RestaurantResponse,
    RestaurantUpdate,
)
from src.common.schemas.courier import (
    CourierCreate,
    CourierResponse,
    CourierUpdate,
    CourierLocationUpdate,
)
from src.common.schemas.order import (
    OrderCreate,
    OrderResponse,
    OrderAssignment,
    OrderStatusUpdate,
)

__all__ = [
    # Base
    "GeoPoint",
    "HealthResponse",
    "PaginatedResponse",
    "TimestampMixin",
    # Restaurant
    "RestaurantCreate",
    "RestaurantResponse",
    "RestaurantUpdate",
    # Courier
    "CourierCreate",
    "CourierResponse",
    "CourierUpdate",
    "CourierLocationUpdate",
    # Order
    "OrderCreate",
    "OrderResponse",
    "OrderAssignment",
    "OrderStatusUpdate",
]

