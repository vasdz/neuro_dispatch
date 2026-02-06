"""
Time Decay Pricing Strategy.

Implements time-sensitive pricing for orders that have been
waiting too long. Gradually increases courier incentives
to ensure order pickup.

Use cases:
- Aging orders that haven't been accepted
- Peak hour order backlog management
- Restaurant delay compensation
"""

import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from src.common.logging import get_logger
from src.pricing_service.strategies.base import (
    BasePricingStrategy,
    PricingContext,
    PricingResult,
    StrategyType,
)

logger = get_logger(__name__)


class TimeDecayStrategy(BasePricingStrategy):
    """
    Time-based dynamic pricing for aging orders.

    As orders wait longer, we need to:
    1. Increase courier payout to incentivize pickup
    2. Potentially reduce customer price to maintain satisfaction
    3. Balance cost between platform subsidy and customer charge

    This strategy is typically composed with other strategies.
    """

    strategy_type = StrategyType.TIME_DECAY
    version = "1.0.0"

    # Time thresholds (in seconds) and their multipliers
    TIME_TIERS = [
        (0, 1.0),       # 0-5 min: no adjustment
        (300, 1.0),     # 5 min: still normal
        (600, 1.1),     # 10 min: slight increase
        (900, 1.25),    # 15 min: moderate increase
        (1200, 1.4),    # 20 min: significant increase
        (1500, 1.6),    # 25 min: urgent
        (1800, 1.8),    # 30 min: critical
        (2400, 2.0),    # 40 min: maximum urgency
    ]

    # Maximum time-based multiplier
    MAX_TIME_MULTIPLIER = 2.5

    # Customer discount to offset some of the delay
    CUSTOMER_DELAY_DISCOUNT_RATE = 0.1  # % per 10 min after 15 min
    MAX_CUSTOMER_DISCOUNT = 0.3  # Maximum 30% discount

    def __init__(
        self,
        base_strategy: Optional[BasePricingStrategy] = None,
        courier_bonus_enabled: bool = True,
        customer_discount_enabled: bool = True,
    ):
        """
        Initialize time decay strategy.

        Args:
            base_strategy: Base strategy to apply time decay on top of
            courier_bonus_enabled: Whether to increase courier payout
            customer_discount_enabled: Whether to offer customer discount
        """
        self.base_strategy = base_strategy
        self.courier_bonus_enabled = courier_bonus_enabled
        self.customer_discount_enabled = customer_discount_enabled

    async def calculate(self, context: PricingContext) -> PricingResult:
        """Calculate time-adjusted pricing."""
        start_time = time.perf_counter()

        # Get order age
        order_age = context.order_age_seconds

        if order_age is None:
            # No order context, can't apply time decay
            if self.base_strategy:
                return await self.base_strategy.calculate(context)

            # Return neutral result
            return PricingResult(
                surge_coefficient=1.0,
                final_price=context.base_price,
                base_surge=1.0,
                strategy_used=self.strategy_type,
                reason="No order context for time decay",
            )

        # Get base result if we have a base strategy
        if self.base_strategy:
            base_result = await self.base_strategy.calculate(context)
            base_surge = base_result.surge_coefficient
        else:
            base_result = None
            base_surge = 1.0

        # Calculate time-based multiplier for courier
        courier_multiplier = self._calculate_time_multiplier(order_age)

        # Calculate customer discount for delays
        customer_discount = self._calculate_delay_discount(order_age)

        # Final surge for courier (what we pay them)
        courier_surge = base_surge * courier_multiplier
        courier_surge = min(courier_surge, self.MAX_SURGE * self.MAX_TIME_MULTIPLIER)

        # Final price for customer (what they pay)
        customer_surge = base_surge * (1 - customer_discount)
        customer_surge = max(customer_surge, self.MIN_SURGE)

        # Platform absorbs the difference
        platform_subsidy = (courier_surge - customer_surge) * context.base_price

        final_price = context.base_price * customer_surge

        calculation_time = (time.perf_counter() - start_time) * 1000

        # Build reason
        reason_parts = []
        if order_age > 600:
            reason_parts.append(f"Order waiting {order_age // 60:.0f} min")
        if courier_multiplier > 1.0:
            reason_parts.append(f"Courier bonus {courier_multiplier:.0%}")
        if customer_discount > 0:
            reason_parts.append(f"Delay discount {customer_discount:.0%}")

        reason = "; ".join(reason_parts) if reason_parts else "Normal timing"

        market = context.market_state

        return PricingResult(
            surge_coefficient=round(customer_surge, 2),
            final_price=round(final_price, 2),
            base_surge=round(base_surge, 2),
            time_modifier=round(courier_multiplier, 2),
            weather_modifier=market.weather_multiplier if market else 1.0,
            loyalty_discount=round(customer_discount, 2),  # Using this field for delay discount
            demand_level=base_result.demand_level if base_result else self.classify_demand(0),
            supply_level=base_result.supply_level if base_result else self.classify_supply(0),
            strategy_used=self.strategy_type,
            reason=reason,
            factors={
                "order_age_seconds": order_age,
                "order_age_minutes": round(order_age / 60, 1),
                "courier_multiplier": round(courier_multiplier, 2),
                "courier_surge": round(courier_surge, 2),
                "customer_discount": round(customer_discount, 2),
                "customer_surge": round(customer_surge, 2),
                "platform_subsidy": round(platform_subsidy, 2),
            },
            confidence=1.0,
            is_capped=False,
            uncapped_surge=round(base_surge * courier_multiplier, 2),
            calculation_time_ms=round(calculation_time, 3),
        )

    def _calculate_time_multiplier(self, age_seconds: float) -> float:
        """
        Calculate time-based multiplier for courier payout.

        Uses piecewise linear interpolation between time tiers.
        """
        if not self.courier_bonus_enabled:
            return 1.0

        if age_seconds <= 0:
            return 1.0

        # Find the tier
        prev_time, prev_mult = self.TIME_TIERS[0]
        for tier_time, tier_mult in self.TIME_TIERS[1:]:
            if age_seconds <= tier_time:
                # Linear interpolation
                t = (age_seconds - prev_time) / (tier_time - prev_time)
                return prev_mult + t * (tier_mult - prev_mult)
            prev_time, prev_mult = tier_time, tier_mult

        # Beyond last tier, cap at maximum
        return min(self.MAX_TIME_MULTIPLIER, prev_mult)

    def _calculate_delay_discount(self, age_seconds: float) -> float:
        """
        Calculate customer discount for delays.

        Kicks in after 15 minutes and increases with wait time.
        """
        if not self.customer_discount_enabled:
            return 0.0

        # No discount for first 15 minutes
        if age_seconds <= 900:
            return 0.0

        # Calculate discount: 10% per 10 min after 15 min
        extra_minutes = (age_seconds - 900) / 60
        discount = (extra_minutes / 10) * self.CUSTOMER_DELAY_DISCOUNT_RATE

        return min(discount, self.MAX_CUSTOMER_DISCOUNT)


class UrgentOrderStrategy(TimeDecayStrategy):
    """
    Strategy specifically for urgent orders.

    Some orders are marked as urgent (e.g., scheduled orders
    that are running late, VIP customers, etc.).

    These get higher initial courier incentives.
    """

    version = "1.1.0"

    # Urgent orders start with higher base multiplier
    URGENT_BASE_MULTIPLIER = 1.3

    # Time tiers are more aggressive for urgent orders
    TIME_TIERS = [
        (0, 1.3),       # Start at 1.3x
        (300, 1.5),     # 5 min: 1.5x
        (600, 1.8),     # 10 min: 1.8x
        (900, 2.0),     # 15 min: 2x
        (1200, 2.5),    # 20 min: 2.5x
    ]

    MAX_TIME_MULTIPLIER = 3.0

    def __init__(self, base_strategy: Optional[BasePricingStrategy] = None):
        super().__init__(base_strategy, courier_bonus_enabled=True)

    async def calculate(self, context: PricingContext) -> PricingResult:
        """Calculate with urgent order priority."""
        result = await super().calculate(context)

        result.factors["is_urgent"] = True
        result.reason = f"[URGENT] {result.reason}"

        return result

