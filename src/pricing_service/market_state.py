"""
Market State Module.

Aggregates real-time supply and demand data to provide
a comprehensive view of market conditions for pricing decisions.

Components:
- MarketStateAggregator: Collects data from various sources
- MarketAnalyzer: Analyzes trends and anomalies
- MarketStateCache: Caches market state for performance
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import h3
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.logging import get_logger
from src.common.models import Order, Courier, OrderStatus, CourierStatus, DemandHourly
from src.common.config import settings
from src.pricing_service.strategies.base import MarketState

logger = get_logger(__name__)


class MarketStateAggregator:
    """
    Aggregates real-time market data from multiple sources.

    Provides:
    - Current supply (couriers) and demand (orders) by hexagon
    - Historical averages for trend detection
    - Integration with demand forecast predictions
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self._cache: dict[str, tuple[datetime, MarketState]] = {}
        self._cache_ttl_seconds = 30  # Cache for 30 seconds

    async def get_market_state(
        self,
        h3_index: str,
        include_neighbors: bool = False,
        use_cache: bool = True,
    ) -> MarketState:
        """
        Get current market state for a hexagon.

        Args:
            h3_index: H3 hexagon index
            include_neighbors: Include neighboring hexagon data
            use_cache: Use cached data if fresh

        Returns:
            MarketState with current metrics
        """
        cache_key = f"{h3_index}:{include_neighbors}"

        # Check cache
        if use_cache and cache_key in self._cache:
            cached_time, cached_state = self._cache[cache_key]
            if (datetime.now(timezone.utc) - cached_time).total_seconds() < self._cache_ttl_seconds:
                return cached_state

        # Collect data
        now = datetime.now(timezone.utc)

        # Get target hexagons
        target_hexagons = {h3_index}
        if include_neighbors:
            neighbors = h3.k_ring(h3_index, 1)
            target_hexagons.update(neighbors)

        # Get current orders and couriers
        pending_orders = await self._get_pending_orders(target_hexagons)
        available_couriers = await self._get_available_couriers(target_hexagons)
        busy_couriers = await self._get_busy_couriers(target_hexagons)

        # Get historical averages
        avg_this_hour = await self._get_hourly_average(h3_index, now.hour, days_back=7)
        avg_same_hour_last_week = await self._get_specific_hour_average(
            h3_index, now - timedelta(days=7), now - timedelta(days=6)
        )

        # Get demand forecast prediction (if available)
        predicted_demand = await self._get_predicted_demand(h3_index)

        state = MarketState(
            h3_index=h3_index,
            timestamp=now,
            pending_orders=pending_orders,
            available_couriers=available_couriers,
            busy_couriers=busy_couriers,
            avg_orders_this_hour=avg_this_hour,
            avg_orders_same_hour_last_week=avg_same_hour_last_week,
            predicted_demand_1h=predicted_demand.get("demand", 0.0),
            predicted_demand_confidence=predicted_demand.get("confidence", 0.0),
            weather_multiplier=1.0,  # TODO: Integrate weather API
        )

        # Update cache
        self._cache[cache_key] = (now, state)

        logger.debug(
            "Market state collected",
            h3_index=h3_index,
            pending_orders=pending_orders,
            available_couriers=available_couriers,
        )

        return state

    async def get_city_market_overview(
        self,
        limit: int = 50,
    ) -> list[MarketState]:
        """
        Get market state overview for the entire city.

        Returns top hexagons by activity.
        """
        # Get all active hexagons
        active_hexagons = await self._get_active_hexagons(limit)

        states = []
        for h3_index in active_hexagons:
            state = await self.get_market_state(h3_index, include_neighbors=False)
            states.append(state)

        # Sort by demand/supply imbalance
        states.sort(key=lambda s: s.demand_supply_ratio, reverse=True)

        return states

    async def _get_pending_orders(self, hexagons: set[str]) -> int:
        """Get count of pending orders in hexagons."""
        result = await self.session.execute(
            select(func.count(Order.id))
            .where(
                Order.customer_h3_index.in_(hexagons),
                Order.status == OrderStatus.PENDING,
            )
        )
        return result.scalar() or 0

    async def _get_available_couriers(self, hexagons: set[str]) -> int:
        """Get count of available couriers in hexagons."""
        result = await self.session.execute(
            select(func.count(Courier.id))
            .where(
                Courier.h3_index.in_(hexagons),
                Courier.status == CourierStatus.AVAILABLE,
            )
        )
        return result.scalar() or 0

    async def _get_busy_couriers(self, hexagons: set[str]) -> int:
        """Get count of busy couriers in hexagons."""
        result = await self.session.execute(
            select(func.count(Courier.id))
            .where(
                Courier.h3_index.in_(hexagons),
                Courier.status.in_([CourierStatus.BUSY, CourierStatus.ASSIGNED]),
            )
        )
        return result.scalar() or 0

    async def _get_hourly_average(
        self,
        h3_index: str,
        hour: int,
        days_back: int = 7,
    ) -> float:
        """Get average orders for this hour over past N days."""
        since = datetime.now(timezone.utc) - timedelta(days=days_back)

        result = await self.session.execute(
            select(func.avg(DemandHourly.order_count))
            .where(
                DemandHourly.h3_index == h3_index,
                DemandHourly.time >= since,
                func.extract('hour', DemandHourly.time) == hour,
            )
        )
        return float(result.scalar() or 0.0)

    async def _get_specific_hour_average(
        self,
        h3_index: str,
        start: datetime,
        end: datetime,
    ) -> float:
        """Get average orders for a specific time range."""
        result = await self.session.execute(
            select(func.avg(DemandHourly.order_count))
            .where(
                DemandHourly.h3_index == h3_index,
                DemandHourly.time >= start,
                DemandHourly.time < end,
            )
        )
        return float(result.scalar() or 0.0)

    async def _get_predicted_demand(self, h3_index: str) -> dict[str, float]:
        """Get predicted demand from forecast service."""
        try:
            from src.demand_forecast import DemandPredictor

            predictor = DemandPredictor(self.session)
            prediction = await predictor.predict(h3_index, hours_ahead=1)

            return {
                "demand": prediction.get("demand", 0.0),
                "confidence": prediction.get("confidence", 0.0),
            }
        except Exception as e:
            logger.debug(f"Could not get demand prediction: {e}")
            return {"demand": 0.0, "confidence": 0.0}

    async def _get_active_hexagons(self, limit: int) -> list[str]:
        """Get hexagons with recent activity."""
        # Get hexagons with pending orders
        orders_result = await self.session.execute(
            select(Order.customer_h3_index)
            .where(Order.status == OrderStatus.PENDING)
            .distinct()
            .limit(limit)
        )

        # Get hexagons with available couriers
        couriers_result = await self.session.execute(
            select(Courier.h3_index)
            .where(
                Courier.status == CourierStatus.AVAILABLE,
                Courier.h3_index.isnot(None),
            )
            .distinct()
            .limit(limit)
        )

        hexagons = set()
        hexagons.update([r[0] for r in orders_result.all() if r[0]])
        hexagons.update([r[0] for r in couriers_result.all() if r[0]])

        return list(hexagons)[:limit]


class MarketAnalyzer:
    """
    Analyzes market conditions and detects patterns.

    Provides:
    - Trend detection (rising/falling demand)
    - Anomaly detection (unusual patterns)
    - Zone health scoring
    """

    def __init__(self, aggregator: MarketStateAggregator):
        self.aggregator = aggregator

    def analyze_trend(self, state: MarketState) -> str:
        """
        Analyze demand trend.

        Returns: "rising", "falling", or "stable"
        """
        trend = state.demand_trend

        if trend > 0.2:
            return "rising"
        elif trend < -0.2:
            return "falling"
        else:
            return "stable"

    def calculate_zone_health(self, state: MarketState) -> float:
        """
        Calculate zone health score (0-1).

        Higher score = healthier supply/demand balance.
        """
        ratio = state.demand_supply_ratio

        if ratio == float('inf'):
            return 0.0  # No couriers = unhealthy

        # Optimal ratio is around 0.5-1.0
        if 0.3 <= ratio <= 1.0:
            return 1.0
        elif ratio < 0.3:
            # Over-supply (not critical but not optimal)
            return 0.8
        elif ratio <= 2.0:
            return 0.6
        elif ratio <= 4.0:
            return 0.3
        else:
            return 0.1

    def detect_anomaly(
        self,
        state: MarketState,
        historical_avg: float,
        std_threshold: float = 2.0,
    ) -> bool:
        """
        Detect if current demand is anomalous.

        Uses simple z-score approach.
        """
        if historical_avg == 0:
            return state.pending_orders > 10  # Arbitrary threshold

        # Assuming std is roughly 30% of mean
        estimated_std = historical_avg * 0.3

        if estimated_std == 0:
            return False

        z_score = abs(state.pending_orders - historical_avg) / estimated_std

        return z_score > std_threshold

    async def get_market_summary(self) -> dict[str, Any]:
        """Get overall market summary."""
        states = await self.aggregator.get_city_market_overview(limit=100)

        if not states:
            return {
                "total_pending_orders": 0,
                "total_available_couriers": 0,
                "avg_surge": 1.0,
                "unhealthy_zones": 0,
            }

        total_orders = sum(s.pending_orders for s in states)
        total_couriers = sum(s.available_couriers for s in states)

        health_scores = [self.calculate_zone_health(s) for s in states]
        avg_health = sum(health_scores) / len(health_scores)

        unhealthy_count = sum(1 for h in health_scores if h < 0.5)

        return {
            "total_pending_orders": total_orders,
            "total_available_couriers": total_couriers,
            "active_zones": len(states),
            "avg_zone_health": round(avg_health, 2),
            "unhealthy_zones": unhealthy_count,
            "global_ratio": round(total_orders / max(total_couriers, 1), 2),
        }

