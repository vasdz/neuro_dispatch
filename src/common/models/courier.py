"""
Courier model.
"""

from typing import TYPE_CHECKING

from sqlalchemy import Enum, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.common.database import Base
from src.common.models.base import CourierStatus, TimestampMixin

if TYPE_CHECKING:
    from src.common.models.order import Order


class Courier(Base, TimestampMixin):
    """Courier entity."""

    __tablename__ = "couriers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    phone: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # Current location
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    h3_index: Mapped[str | None] = mapped_column(String(15), nullable=True, index=True)

    # Status
    status: Mapped[CourierStatus] = mapped_column(
        Enum(
            CourierStatus,
            name="courier_status",
            create_type=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        default=CourierStatus.OFFLINE,
        index=True,
    )
    current_order_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Performance metrics
    avg_speed_kmh: Mapped[float] = mapped_column(Float, default=15.0)
    rating: Mapped[float] = mapped_column(Float, default=5.0)
    total_deliveries: Mapped[int] = mapped_column(Integer, default=0)

    # Relationships
    orders: Mapped[list["Order"]] = relationship(back_populates="courier")

    def __repr__(self) -> str:
        return f"<Courier(id={self.id}, name='{self.name}', status={self.status})>"

    @property
    def is_available(self) -> bool:
        """Check if courier is available for assignment."""
        return self.status == CourierStatus.AVAILABLE

    @property
    def has_location(self) -> bool:
        """Check if courier has valid location."""
        return self.latitude is not None and self.longitude is not None

