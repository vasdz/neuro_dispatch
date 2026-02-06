"""
Hybrid Pricing Strategy.

Combines rule-based and ML-based strategies with configurable weights.
Uses the strengths of both approaches:
- Rule-based: Interpretability, stability, guaranteed bounds
- ML-based: Adaptability, pattern recognition, optimization

Features:
- Weighted average of multiple strategies
- Confidence-based weighting
- Automatic fallback handling
- Explanation aggregation
"""

import time
from datetime import datetime, timezone
from typing import Optional

from src.common.logging import get_logger
from src.pricing_service.strategies.base import (
    BasePricingStrategy,
    PricingContext,
    PricingResult,
    StrategyType,
)
from src.pricing_service.strategies.rule_based import RuleBasedStrategy
from src.pricing_service.strategies.ml_based import MLBasedStrategy

logger = get_logger(__name__)


class HybridStrategy(BasePricingStrategy):
    """
    Hybrid strategy combining rule-based and ML approaches.

    Weighting Modes:
    - fixed: Fixed weights for each strategy
    - confidence: Weight by prediction confidence
    - adaptive: Learn optimal weights over time

    The hybrid approach provides:
    - More robust predictions
    - Graceful degradation when ML fails
    - Explainability from rules + adaptability from ML
    """

    strategy_type = StrategyType.HYBRID
    version = "1.0.0"

    def __init__(
        self,
        rule_weight: float = 0.4,
        ml_weight: float = 0.6,
        use_confidence_weighting: bool = True,
        min_ml_confidence: float = 0.7,
    ):
        """
        Initialize hybrid strategy.

        Args:
            rule_weight: Base weight for rule-based strategy
            ml_weight: Base weight for ML strategy
            use_confidence_weighting: Adjust weights by ML confidence
            min_ml_confidence: Minimum ML confidence to use ML predictions
        """
        if abs(rule_weight + ml_weight - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")

        self.base_rule_weight = rule_weight
        self.base_ml_weight = ml_weight
        self.use_confidence_weighting = use_confidence_weighting
        self.min_ml_confidence = min_ml_confidence

        self.rule_strategy = RuleBasedStrategy()
        self.ml_strategy = MLBasedStrategy()

    async def calculate(self, context: PricingContext) -> PricingResult:
        """Calculate using hybrid approach."""
        start_time = time.perf_counter()

        # Get predictions from both strategies
        rule_result = await self.rule_strategy.calculate(context)
        ml_result = await self.ml_strategy.calculate(context)

        # Determine weights
        rule_weight, ml_weight = self._compute_weights(ml_result)

        # Weighted average of surge coefficients
        hybrid_surge = (
            rule_result.surge_coefficient * rule_weight +
            ml_result.surge_coefficient * ml_weight
        )

        # Apply constraints
        final_surge, is_capped = self.apply_constraints(hybrid_surge, context)

        # Calculate loyalty discount (use rule-based calculation)
        loyalty_discount = self.calculate_loyalty_discount(context)

        # Final price
        effective_surge = final_surge * (1 - loyalty_discount)
        final_price = context.base_price * effective_surge

        calculation_time = (time.perf_counter() - start_time) * 1000

        # Combine factors from both strategies
        combined_factors = {
            "rule_surge": rule_result.surge_coefficient,
            "ml_surge": ml_result.surge_coefficient,
            "rule_weight": round(rule_weight, 2),
            "ml_weight": round(ml_weight, 2),
            "ml_confidence": ml_result.confidence,
            **rule_result.factors,
        }

        # Combined reason
        reason = self._combine_reasons(rule_result, ml_result, rule_weight, ml_weight)

        # Use demand/supply classification from rule-based (more consistent)
        market = context.market_state

        return PricingResult(
            surge_coefficient=round(final_surge, 2),
            final_price=round(final_price, 2),
            base_surge=round(hybrid_surge, 2),
            time_modifier=rule_result.time_modifier,
            weather_modifier=market.weather_multiplier,
            loyalty_discount=round(loyalty_discount, 2),
            demand_level=rule_result.demand_level,
            supply_level=rule_result.supply_level,
            strategy_used=self.strategy_type,
            reason=reason,
            factors=combined_factors,
            confidence=round(
                rule_weight * 1.0 + ml_weight * ml_result.confidence, 2
            ),
            is_capped=is_capped,
            uncapped_surge=round(hybrid_surge, 2),
            calculation_time_ms=round(calculation_time, 3),
        )

    def _compute_weights(self, ml_result: PricingResult) -> tuple[float, float]:
        """
        Compute effective weights based on ML confidence.

        If ML confidence is low, shift weight toward rules.
        """
        if not self.use_confidence_weighting:
            return self.base_rule_weight, self.base_ml_weight

        ml_confidence = ml_result.confidence

        # If ML confidence too low, use pure rule-based
        if ml_confidence < self.min_ml_confidence:
            logger.debug(
                f"Low ML confidence ({ml_confidence:.2f}), using pure rules"
            )
            return 1.0, 0.0

        # If ML was actually a fallback, use pure rules
        if ml_result.strategy_used == StrategyType.RULE_BASED:
            return 1.0, 0.0

        # Adjust weights by confidence
        # High confidence -> more ML weight
        # confidence = 0.7 -> no adjustment
        # confidence = 1.0 -> +15% to ML weight
        # confidence = 0.5 -> -25% from ML weight

        confidence_adjustment = (ml_confidence - 0.7) * 0.5

        adjusted_ml_weight = self.base_ml_weight + confidence_adjustment
        adjusted_ml_weight = max(0.0, min(1.0, adjusted_ml_weight))

        adjusted_rule_weight = 1.0 - adjusted_ml_weight

        return adjusted_rule_weight, adjusted_ml_weight

    def _combine_reasons(
        self,
        rule_result: PricingResult,
        ml_result: PricingResult,
        rule_weight: float,
        ml_weight: float,
    ) -> str:
        """Combine explanations from both strategies."""
        parts = []

        if rule_weight > 0:
            parts.append(f"Rules ({rule_weight:.0%}): {rule_result.reason}")

        if ml_weight > 0:
            parts.append(f"ML ({ml_weight:.0%}): {ml_result.reason}")

        return " | ".join(parts)


class AdaptiveHybridStrategy(HybridStrategy):
    """
    Adaptive hybrid strategy that learns optimal weights over time.

    Uses online learning to adjust weights based on:
    - Prediction accuracy vs actual outcomes
    - Conversion rate at different surge levels
    - Customer satisfaction metrics

    This is a placeholder for a more sophisticated approach
    that would require integration with outcome tracking.
    """

    version = "1.1.0"

    def __init__(
        self,
        initial_rule_weight: float = 0.5,
        initial_ml_weight: float = 0.5,
        learning_rate: float = 0.01,
    ):
        super().__init__(initial_rule_weight, initial_ml_weight)
        self.learning_rate = learning_rate

        # Weight history for analysis
        self._weight_history: list[tuple[datetime, float, float]] = []
        self._max_history = 1000

    def record_outcome(
        self,
        context: PricingContext,
        result: PricingResult,
        order_accepted: bool,
        delivery_completed: bool,
        customer_rating: Optional[float] = None,
    ):
        """
        Record outcome to learn from.

        This would be called by the order service after delivery completion
        to help the strategy learn which approach works better.
        """
        # Calculate reward signal
        reward = 0.0

        if order_accepted:
            reward += 0.5
        if delivery_completed:
            reward += 0.3
        if customer_rating is not None and customer_rating >= 4.0:
            reward += 0.2

        # Record for future weight updates
        # In a real implementation, this would update weights using
        # techniques like Thompson Sampling or UCB
        logger.info(
            f"Recorded outcome: accepted={order_accepted}, "
            f"completed={delivery_completed}, rating={customer_rating}, "
            f"reward={reward:.2f}"
        )

