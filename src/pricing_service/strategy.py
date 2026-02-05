"""
Pricing Strategy - Surge coefficient calculation.

Implements dynamic pricing based on supply/demand balance.
"""

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.models import Order, Courier, OrderStatus, CourierStatus, DemandHourly
from src.common.config import settings
from src.common.logging import get_logger

logger = get_logger(__name__)


class PricingEngine:
    """
    Dynamic pricing engine.

    Calculates surge coefficients based on:
    - Current demand (pending orders in area)
    - Available supply (couriers in area)
    - Historical patterns
    - Time of day
    """

    # Surge limits
    MIN_SURGE = 1.0
    MAX_SURGE = 3.0

    # Thresholds
    HIGH_DEMAND_THRESHOLD = 10  # orders
    LOW_SUPPLY_THRESHOLD = 3   # couriers

    def __init__(self, session: AsyncSession):
        self.session = session

    async def calculate_surge(self, h3_index: str) -> dict[str, Any]:
        """
        Calculate surge coefficient for a hexagon.

        Surge = f(demand / supply) * time_modifier
        """
        # Get current demand (pending orders)
        demand = await self._get_current_demand(h3_index)

        # Get current supply (available couriers)
        supply = await self._get_current_supply(h3_index)

        # Calculate base surge
        if supply == 0:
            surge = self.MAX_SURGE if demand > 0 else self.MIN_SURGE
            reason = "No couriers available"
        elif demand == 0:
            surge = self.MIN_SURGE
            reason = "No pending orders"
        else:
            ratio = demand / supply

            if ratio <= 0.5:
                surge = 1.0
                reason = "Balanced supply/demand"
            elif ratio <= 1.0:
                surge = 1.0 + (ratio - 0.5) * 0.4  # 1.0 - 1.2
                reason = "Slightly elevated demand"
            elif ratio <= 2.0:
                surge = 1.2 + (ratio - 1.0) * 0.3  # 1.2 - 1.5
                reason = "High demand"
            elif ratio <= 4.0:
                surge = 1.5 + (ratio - 2.0) * 0.25  # 1.5 - 2.0
                reason = "Very high demand"
            else:
                surge = min(2.0 + (ratio - 4.0) * 0.1, self.MAX_SURGE)
                reason = "Extreme demand"

        # Apply time-of-day modifier
        time_modifier = self._get_time_modifier()
        surge = surge * time_modifier

        # Clamp to limits
        surge = max(self.MIN_SURGE, min(self.MAX_SURGE, surge))

        # Determine demand/supply levels for display
        demand_level = self._categorize_level(demand, [3, 7, 15])
        supply_level = self._categorize_level(supply, [2, 5, 10])

        logger.debug(
            "Surge calculated",
            h3_index=h3_index,
            demand=demand,
            supply=supply,
            surge=round(surge, 2),
        )

        return {
            "coefficient": round(surge, 2),
            "reason": reason,
            "demand": demand,
            "supply": supply,
            "demand_level": demand_level,
            "supply_level": supply_level,
        }

    async def _get_current_demand(self, h3_index: str) -> int:
        """Get number of pending orders in hexagon."""
        result = await self.session.execute(
            select(func.count(Order.id))
            .where(
                Order.customer_h3_index == h3_index,
                Order.status == OrderStatus.PENDING,
            )
        )
        return result.scalar() or 0

    async def _get_current_supply(self, h3_index: str) -> int:
        """Get number of available couriers in hexagon."""
        result = await self.session.execute(
            select(func.count(Courier.id))
            .where(
                Courier.h3_index == h3_index,
                Courier.status == CourierStatus.AVAILABLE,
            )
        )
        return result.scalar() or 0

    def _get_time_modifier(self) -> float:
        """
        Get time-of-day surge modifier.

        Peak hours have higher base multiplier.
        """
        hour = datetime.utcnow().hour

        # Peak hours (local time adjustments may be needed)
        if 11 <= hour <= 14:  # Lunch
            return 1.1
        elif 18 <= hour <= 21:  # Dinner
            return 1.15
        elif 22 <= hour or hour <= 6:  # Night
            return 0.9
        else:
            return 1.0

    def _categorize_level(self, value: int, thresholds: list[int]) -> str:
        """Categorize a value into low/medium/high/extreme."""
        if value <= thresholds[0]:
            return "low"
        elif value <= thresholds[1]:
            return "medium"
        elif value <= thresholds[2]:
            return "high"
        else:
            return "extreme"

    async def get_city_surge_map(self) -> list[dict]:
        """
        Get surge map for entire city.

        Returns surge coefficient for all hexagons with activity.
        """
        # Get all hexagons with pending orders or available couriers
        orders_result = await self.session.execute(
            select(Order.customer_h3_index)
            .where(Order.status == OrderStatus.PENDING)
            .distinct()
        )

        couriers_result = await self.session.execute(
            select(Courier.h3_index)
            .where(
                Courier.status == CourierStatus.AVAILABLE,
                Courier.h3_index.isnot(None),
            )
            .distinct()
        )

        h3_indexes = set()
        h3_indexes.update([r[0] for r in orders_result.all()])
        h3_indexes.update([r[0] for r in couriers_result.all() if r[0]])

        surge_map = []
        for h3_index in h3_indexes:
            result = await self.calculate_surge(h3_index)
            surge_map.append({
                "h3_index": h3_index,
                "surge": result["coefficient"],
                "demand": result["demand"],
                "supply": result["supply"],
            })

        # Sort by surge descending
        surge_map.sort(key=lambda x: x["surge"], reverse=True)

        return surge_map

