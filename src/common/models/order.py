"""
Order model.
"""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.common.database import Base
from src.common.models.base import OrderStatus, TimestampMixin

if TYPE_CHECKING:
    from src.common.models.courier import Courier
    from src.common.models.restaurant import Restaurant


class Order(Base, TimestampMixin):
    """Order entity."""

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)

    # Foreign keys
    restaurant_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("restaurants.id"), nullable=False
    )
    courier_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("couriers.id"), nullable=True
    )

    # Customer location
    customer_latitude: Mapped[float] = mapped_column(Float, nullable=False)
    customer_longitude: Mapped[float] = mapped_column(Float, nullable=False)
    customer_h3_index: Mapped[str] = mapped_column(String(15), nullable=False, index=True)

    # Status
    status: Mapped[OrderStatus] = mapped_column(
        Enum(
            OrderStatus,
            name="order_status",
            create_type=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        default=OrderStatus.PENDING,
        index=True,
    )

    # Pricing
    surge_coefficient: Mapped[float] = mapped_column(Float, default=1.0)

    # Time estimates
    estimated_prep_time_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_delivery_time_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    actual_delivery_time_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Timestamps
    assigned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    picked_up_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    restaurant: Mapped["Restaurant"] = relationship(back_populates="orders")
    courier: Mapped["Courier | None"] = relationship(back_populates="orders")

    def __repr__(self) -> str:
        return f"<Order(id={self.id}, status={self.status})>"

    @property
    def is_pending(self) -> bool:
        """Check if order is pending assignment."""
        return self.status == OrderStatus.PENDING

    @property
    def is_active(self) -> bool:
        """Check if order is in active delivery state."""
        return self.status in (
            OrderStatus.ASSIGNED,
            OrderStatus.PICKED_UP,
            OrderStatus.IN_TRANSIT,
        )

