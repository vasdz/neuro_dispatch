"""
Rule-Based Pricing Strategy.

Traditional surge pricing using predefined rules based on
supply/demand ratio and time-of-day patterns.

This is the baseline strategy that serves as a fallback
and comparison benchmark for ML-based approaches.
"""

import time
from datetime import datetime, timezone

from src.common.logging import get_logger
from src.pricing_service.strategies.base import (
    BasePricingStrategy,
    PricingContext,
    PricingResult,
    StrategyType,
    DemandLevel,
    SupplyLevel,
)

logger = get_logger(__name__)


class RuleBasedStrategy(BasePricingStrategy):
    """
    Rule-based surge pricing strategy.

    Uses a piecewise linear function to map supply/demand ratio
    to surge coefficient, with time-of-day adjustments.

    Surge Calculation:
    1. Base surge from demand/supply ratio
    2. Time-of-day modifier (peak hours)
    3. Weather modifier (external factor)
    4. Loyalty discount (customer-specific)
    5. Apply min/max constraints
    """

    strategy_type = StrategyType.RULE_BASED
    version = "2.0.0"

    # Surge curve breakpoints: (ratio, surge)
    # Piecewise linear interpolation between points
    SURGE_CURVE = [
        (0.0, 1.0),    # No demand
        (0.5, 1.0),    # Balanced
        (1.0, 1.15),   # Slight imbalance
        (1.5, 1.3),    # Moderate imbalance
        (2.0, 1.5),    # High demand
        (3.0, 1.8),    # Very high demand
        (4.0, 2.0),    # Extreme demand
        (6.0, 2.5),    # Crisis mode
        (float('inf'), 3.0),  # Maximum surge
    ]

    # Time-of-day modifiers (hour -> modifier)
    TIME_MODIFIERS = {
        # Early morning (6-9)
        6: 0.95, 7: 1.0, 8: 1.0, 9: 1.0,
        # Late morning (10-11)
        10: 1.0, 11: 1.05,
        # Lunch peak (12-14)
        12: 1.15, 13: 1.2, 14: 1.1,
        # Afternoon (15-17)
        15: 1.0, 16: 1.0, 17: 1.05,
        # Dinner peak (18-21)
        18: 1.15, 19: 1.25, 20: 1.2, 21: 1.1,
        # Late night (22-5)
        22: 1.0, 23: 0.95, 0: 0.9, 1: 0.85,
        2: 0.8, 3: 0.8, 4: 0.85, 5: 0.9,
    }

    # Day-of-week modifiers
    DAY_MODIFIERS = {
        0: 1.0,   # Monday
        1: 1.0,   # Tuesday
        2: 1.0,   # Wednesday
        3: 1.05,  # Thursday
        4: 1.1,   # Friday
        5: 1.15,  # Saturday
        6: 1.1,   # Sunday
    }

    async def calculate(self, context: PricingContext) -> PricingResult:
        """Calculate surge using rule-based logic."""
        start_time = time.perf_counter()

        market = context.market_state

        # Step 1: Calculate base surge from ratio
        ratio = market.demand_supply_ratio
        base_surge = self._interpolate_surge(ratio)

        # Step 2: Apply time modifier
        # Use market timestamp for determinism, fallback to now
        now = market.timestamp or datetime.now(timezone.utc)
        time_modifier = self._get_time_modifier(now)
        day_modifier = self.DAY_MODIFIERS.get(now.weekday(), 1.0)
        combined_time_modifier = time_modifier * day_modifier

        surge_after_time = base_surge * combined_time_modifier

        # Step 3: Apply weather modifier
        weather_modifier = market.weather_multiplier
        surge_after_weather = surge_after_time * weather_modifier

        # Step 4: Apply constraints
        final_surge, is_capped = self.apply_constraints(surge_after_weather, context)

        # Step 5: Calculate loyalty discount
        loyalty_discount = self.calculate_loyalty_discount(context)

        # Step 6: Calculate final price
        effective_surge = final_surge * (1 - loyalty_discount)
        final_price = context.base_price * effective_surge

        # Build result
        calculation_time = (time.perf_counter() - start_time) * 1000

        reason = self._generate_reason(market, ratio, combined_time_modifier)

        return PricingResult(
            surge_coefficient=round(final_surge, 2),
            final_price=round(final_price, 2),
            base_surge=round(base_surge, 2),
            time_modifier=round(combined_time_modifier, 2),
            weather_modifier=round(weather_modifier, 2),
            loyalty_discount=round(loyalty_discount, 2),
            demand_level=self.classify_demand(market.pending_orders),
            supply_level=self.classify_supply(market.available_couriers),
            strategy_used=self.strategy_type,
            reason=reason,
            factors={
                "demand_supply_ratio": round(ratio, 2) if ratio != float('inf') else -1,
                "pending_orders": market.pending_orders,
                "available_couriers": market.available_couriers,
                "courier_utilization": round(market.courier_utilization, 2),
                "hour": now.hour,
                "day_of_week": now.weekday(),
            },
            confidence=1.0,  # Rule-based is deterministic
            is_capped=is_capped,
            uncapped_surge=round(surge_after_weather, 2),
            calculation_time_ms=round(calculation_time, 3),
        )

    def _interpolate_surge(self, ratio: float) -> float:
        """
        Interpolate surge from the surge curve.

        Uses piecewise linear interpolation between breakpoints.
        """
        if ratio <= 0:
            return self.SURGE_CURVE[0][1]

        # Find the two breakpoints to interpolate between
        prev_point = self.SURGE_CURVE[0]
        for point in self.SURGE_CURVE[1:]:
            if ratio <= point[0]:
                # Linear interpolation
                r1, s1 = prev_point
                r2, s2 = point

                if r2 == float('inf'):
                    return s2

                t = (ratio - r1) / (r2 - r1)
                return s1 + t * (s2 - s1)

            prev_point = point

        return self.SURGE_CURVE[-1][1]

    def _get_time_modifier(self, dt: datetime) -> float:
        """Get time-of-day modifier."""
        return self.TIME_MODIFIERS.get(dt.hour, 1.0)

    def _generate_reason(
        self,
        market,
        ratio: float,
        time_modifier: float,
    ) -> str:
        """Generate human-readable reason for surge."""
        reasons = []

        # Supply/demand explanation
        if market.available_couriers == 0:
            reasons.append("No couriers available in area")
        elif ratio > 3.0:
            reasons.append(f"Extreme demand ({market.pending_orders} orders vs {market.available_couriers} couriers)")
        elif ratio > 1.5:
            reasons.append(f"High demand ({market.pending_orders} orders vs {market.available_couriers} couriers)")
        elif ratio > 1.0:
            reasons.append("Slightly elevated demand")
        elif market.pending_orders == 0:
            reasons.append("No pending orders")
        else:
            reasons.append("Balanced supply and demand")

        # Time explanation
        if time_modifier > 1.1:
            reasons.append("Peak hours surcharge")
        elif time_modifier < 0.95:
            reasons.append("Off-peak discount")

        return "; ".join(reasons)


class ZoneBasedStrategy(RuleBasedStrategy):
    """
    Extension of rule-based strategy with zone-specific adjustments.

    Different areas of the city may have different surge thresholds
    based on historical patterns and business strategy.
    """

    strategy_type = StrategyType.RULE_BASED
    version = "2.1.0"

    # Zone-specific max surge limits
    ZONE_MAX_SURGE = {
        "center": 2.5,     # City center - more competition
        "inner": 2.8,      # Inner ring
        "middle": 3.0,     # Default
        "outer": 3.5,      # Outer areas - higher delivery costs
    }

    # Zone-specific base adjustments
    ZONE_BASE_ADJUSTMENT = {
        "center": 1.1,     # Premium area
        "inner": 1.05,
        "middle": 1.0,
        "outer": 0.95,     # Slightly lower base due to less competition
    }

    def __init__(self, zone: str = "middle"):
        self.zone = zone
        self.MAX_SURGE = self.ZONE_MAX_SURGE.get(zone, 3.0)
        self._zone_adjustment = self.ZONE_BASE_ADJUSTMENT.get(zone, 1.0)

    async def calculate(self, context: PricingContext) -> PricingResult:
        """Calculate with zone-specific adjustments."""
        result = await super().calculate(context)

        # Apply zone adjustment
        adjusted_surge = result.surge_coefficient * self._zone_adjustment
        adjusted_surge, is_capped = self.apply_constraints(adjusted_surge, context)

        result.surge_coefficient = round(adjusted_surge, 2)
        result.final_price = round(context.base_price * adjusted_surge, 2)
        result.factors["zone"] = self.zone
        result.factors["zone_adjustment"] = self._zone_adjustment
        result.is_capped = is_capped

        return result

