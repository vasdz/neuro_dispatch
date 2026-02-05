"""
Time-series models for telemetry and analytics.
"""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, JSON
from sqlalchemy.orm import Mapped, mapped_column

from src.common.database import Base


class CourierTelemetry(Base):
    """Courier location telemetry (time-series)."""

    __tablename__ = "courier_telemetry"

    time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, nullable=False
    )
    courier_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("couriers.id"), primary_key=True, nullable=False
    )

    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    h3_index: Mapped[str] = mapped_column(String(15), nullable=False)

    speed_kmh: Mapped[float | None] = mapped_column(Float, nullable=True)
    heading: Mapped[float | None] = mapped_column(Float, nullable=True)
    battery_level: Mapped[int | None] = mapped_column(Integer, nullable=True)

    def __repr__(self) -> str:
        return f"<CourierTelemetry(courier_id={self.courier_id}, time={self.time})>"


class OrderEvent(Base):
    """Order events for analytics (time-series)."""

    __tablename__ = "order_events"

    time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, nullable=False
    )
    order_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("orders.id"), primary_key=True, nullable=False
    )
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    h3_index: Mapped[str | None] = mapped_column(String(15), nullable=True)
    event_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    def __repr__(self) -> str:
        return f"<OrderEvent(order_id={self.order_id}, type={self.event_type})>"


class DemandHourly(Base):
    """Hourly demand aggregation per hexagon."""

    __tablename__ = "demand_hourly"

    time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, nullable=False
    )
    h3_index: Mapped[str] = mapped_column(String(15), primary_key=True, nullable=False)

    order_count: Mapped[int] = mapped_column(Integer, default=0)
    avg_surge: Mapped[float] = mapped_column(Float, default=1.0)
    available_couriers: Mapped[int] = mapped_column(Integer, default=0)

    def __repr__(self) -> str:
        return f"<DemandHourly(h3={self.h3_index}, time={self.time}, count={self.order_count})>"

