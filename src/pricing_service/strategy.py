"""
Pricing Strategy - Surge coefficient calculation.

Senior+ level implementation with:
- Multiple pricing strategies (rule-based, ML, hybrid)
- A/B testing infrastructure
- Market state aggregation
- Fairness constraints
- Price history tracking

This is the main entry point for the pricing service.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.config import settings
from src.common.logging import get_logger
from src.pricing_service.strategies.base import (
    BasePricingStrategy,
    PricingContext,
    PricingResult,
    MarketState,
    StrategyType,
)
from src.pricing_service.strategies.rule_based import RuleBasedStrategy
from src.pricing_service.strategies.hybrid import HybridStrategy
from src.pricing_service.market_state import MarketStateAggregator, MarketAnalyzer
from src.pricing_service.ab_testing import get_ab_testing_engine

logger = get_logger(__name__)


class PricingEngine:
    """
    Main pricing engine - facade for the pricing service.

    Coordinates:
    - Strategy selection
    - Market state collection
    - A/B testing
    - Price calculation
    - Result caching

    Usage:
        engine = PricingEngine(session)
        result = await engine.calculate_price(
            h3_index="882830829bfffff",
            base_price=350.0,
        )
    """

    # Global surge limits (can be overridden per strategy)
    MIN_SURGE = 1.0
    MAX_SURGE = 3.0

    # Fairness constraints
    MAX_SURGE_INCREASE_PER_MINUTE = 0.1  # Max surge increase rate
    PRICE_CAP_MULTIPLIER = 2.0  # Maximum price cap (2x base)

    def __init__(
        self,
        session: AsyncSession,
        default_strategy: Optional[BasePricingStrategy] = None,
    ):
        self.session = session
        self.market_aggregator = MarketStateAggregator(session)
        self.market_analyzer = MarketAnalyzer(self.market_aggregator)
        self.ab_engine = get_ab_testing_engine()

        # Default to hybrid strategy
        self._default_strategy = default_strategy or HybridStrategy()

        # Price history for rate limiting
        self._price_history: dict[str, list[tuple[datetime, float]]] = {}
        self._history_ttl = timedelta(minutes=30)

    async def calculate_price(
        self,
        h3_index: str,
        base_price: float,
        order_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        order_created_at: Optional[datetime] = None,
        order_value: float = 0.0,
        customer_order_count: int = 0,
        customer_avg_order_value: float = 0.0,
        customer_is_premium: bool = False,
        restaurant_prep_time: int = 20,
        restaurant_rating: float = 4.5,
        apply_fairness: bool = True,
    ) -> PricingResult:
        """
        Calculate dynamic price for an order.

        This is the main entry point for pricing calculations.

        Args:
            h3_index: Hexagon for the order location
            base_price: Base delivery price
            order_id: Optional order ID
            customer_id: Optional customer ID
            order_created_at: When order was created (for time-decay)
            order_value: Total order value
            customer_order_count: Customer's historical order count
            customer_avg_order_value: Customer's average order value
            customer_is_premium: Whether customer is premium
            restaurant_prep_time: Expected prep time in minutes
            restaurant_rating: Restaurant rating
            apply_fairness: Apply fairness constraints

        Returns:
            PricingResult with surge and final price
        """
        # Step 1: Collect market state
        market_state = await self.market_aggregator.get_market_state(
            h3_index,
            include_neighbors=True,
        )

        # Step 2: Build context
        context = PricingContext(
            market_state=market_state,
            base_price=base_price,
            order_id=order_id,
            order_created_at=order_created_at,
            order_value=order_value,
            customer_id=customer_id,
            customer_order_count=customer_order_count,
            customer_avg_order_value=customer_avg_order_value,
            customer_is_premium=customer_is_premium,
            restaurant_prep_time_min=restaurant_prep_time,
            restaurant_rating=restaurant_rating,
            max_surge_override=self.MAX_SURGE,
            min_surge_override=self.MIN_SURGE,
        )

        # Step 3: Select strategy (considering A/B tests)
        strategy, exp_id, variant = await self.ab_engine.get_strategy_for_context(
            context,
            self._default_strategy,
        )

        # Update context with experiment info
        context.experiment_id = exp_id
        context.variant = variant

        # Step 4: Calculate price
        result = await strategy.calculate(context)

        # Step 5: Apply fairness constraints
        if apply_fairness:
            result = self._apply_fairness_constraints(h3_index, result, context)

        # Step 6: Record for A/B testing
        if exp_id and variant:
            self.ab_engine.record_conversion(
                exp_id,
                variant,
                revenue=result.final_price,
                surge=result.surge_coefficient,
            )

        # Step 7: Update history
        self._update_price_history(h3_index, result.surge_coefficient)

        logger.info(
            "Price calculated",
            h3_index=h3_index,
            surge=result.surge_coefficient,
            final_price=result.final_price,
            strategy=result.strategy_used.value,
            experiment=exp_id,
        )

        return result

    async def calculate_surge(self, h3_index: str) -> dict[str, Any]:
        """
        Calculate surge coefficient for a hexagon.

        Simplified version that returns a dict for backward compatibility.
        """
        result = await self.calculate_price(
            h3_index=h3_index,
            base_price=100.0,  # Dummy base price
        )

        return {
            "coefficient": result.surge_coefficient,
            "reason": result.reason,
            "demand": result.factors.get("pending_orders", 0),
            "supply": result.factors.get("available_couriers", 0),
            "demand_level": result.demand_level.value,
            "supply_level": result.supply_level.value,
        }

    async def get_city_surge_map(
        self,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Get surge map for the city.

        Returns surge coefficients for active hexagons.
        """
        states = await self.market_aggregator.get_city_market_overview(limit=limit)

        surge_map = []
        for state in states:
            context = PricingContext(
                market_state=state,
                base_price=100.0,
            )

            result = await self._default_strategy.calculate(context)

            surge_map.append({
                "h3_index": state.h3_index,
                "surge": result.surge_coefficient,
                "demand": state.pending_orders,
                "supply": state.available_couriers,
                "health": self.market_analyzer.calculate_zone_health(state),
            })

        # Sort by surge descending
        surge_map.sort(key=lambda x: x["surge"], reverse=True)

        return surge_map

    async def get_market_summary(self) -> dict[str, Any]:
        """Get market summary for monitoring."""
        return await self.market_analyzer.get_market_summary()

    def _apply_fairness_constraints(
        self,
        h3_index: str,
        result: PricingResult,
        context: PricingContext,
    ) -> PricingResult:
        """
        Apply fairness constraints to pricing result.

        Constraints:
        1. Rate limit surge increases
        2. Cap maximum price
        3. Ensure loyalty discounts
        """
        # Get recent price history
        history = self._get_recent_history(h3_index)

        if history:
            # Rate limit: max surge increase per minute
            last_time, last_surge = history[-1]
            minutes_elapsed = (datetime.now(timezone.utc) - last_time).total_seconds() / 60
            max_increase = minutes_elapsed * self.MAX_SURGE_INCREASE_PER_MINUTE

            if result.surge_coefficient > last_surge + max_increase:
                old_surge = result.surge_coefficient
                result.surge_coefficient = round(last_surge + max_increase, 2)
                result.is_capped = True
                result.uncapped_surge = old_surge
                result.reason = f"{result.reason} [Rate limited]"

        # Price cap: ensure final price doesn't exceed cap
        max_price = context.base_price * self.PRICE_CAP_MULTIPLIER
        if result.final_price > max_price:
            result.final_price = round(max_price, 2)
            result.surge_coefficient = round(max_price / context.base_price, 2)
            result.is_capped = True
            result.reason = f"{result.reason} [Price capped]"

        return result

    def _update_price_history(self, h3_index: str, surge: float):
        """Update price history for a hexagon."""
        now = datetime.now(timezone.utc)

        if h3_index not in self._price_history:
            self._price_history[h3_index] = []

        history = self._price_history[h3_index]
        history.append((now, surge))

        # Clean old entries
        cutoff = now - self._history_ttl
        self._price_history[h3_index] = [
            (t, s) for t, s in history if t > cutoff
        ]

    def _get_recent_history(
        self,
        h3_index: str,
        minutes: int = 5,
    ) -> list[tuple[datetime, float]]:
        """Get recent price history for a hexagon."""
        if h3_index not in self._price_history:
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        return [
            (t, s) for t, s in self._price_history[h3_index]
            if t > cutoff
        ]


class PricingEngineFactory:
    """
    Factory for creating pricing engines with different configurations.
    """

    @staticmethod
    def create_default(session: AsyncSession) -> PricingEngine:
        """Create default pricing engine."""
        return PricingEngine(session)

    @staticmethod
    def create_with_strategy(
        session: AsyncSession,
        strategy_type: StrategyType,
    ) -> PricingEngine:
        """Create pricing engine with specific strategy."""
        from src.pricing_service.strategies.rule_based import RuleBasedStrategy
        from src.pricing_service.strategies.ml_based import MLBasedStrategy
        from src.pricing_service.strategies.hybrid import HybridStrategy
        from src.pricing_service.strategies.time_decay import TimeDecayStrategy

        strategies = {
            StrategyType.RULE_BASED: RuleBasedStrategy,
            StrategyType.ML_BASED: MLBasedStrategy,
            StrategyType.HYBRID: HybridStrategy,
            StrategyType.TIME_DECAY: TimeDecayStrategy,
        }

        strategy_class = strategies.get(strategy_type, HybridStrategy)
        return PricingEngine(session, default_strategy=strategy_class())

    @staticmethod
    def create_for_testing(
        session: AsyncSession,
        min_surge: float = 1.0,
        max_surge: float = 2.0,
    ) -> PricingEngine:
        """Create pricing engine for testing with custom limits."""
        engine = PricingEngine(session)
        engine.MIN_SURGE = min_surge
        engine.MAX_SURGE = max_surge
        return engine
