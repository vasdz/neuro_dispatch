"""
Restaurant model.
"""

from sqlalchemy import Boolean, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.common.database import Base
from src.common.models.base import TimestampMixin


class Restaurant(Base, TimestampMixin):
    """Restaurant entity."""

    __tablename__ = "restaurants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Location
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    h3_index: Mapped[str] = mapped_column(String(15), nullable=False, index=True)

    # Operational
    avg_prep_time_minutes: Mapped[int] = mapped_column(Integer, default=15)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    orders: Mapped[list["Order"]] = relationship(back_populates="restaurant")

    def __repr__(self) -> str:
        return f"<Restaurant(id={self.id}, name='{self.name}')>"


# Avoid circular import
from src.common.models.order import Order

