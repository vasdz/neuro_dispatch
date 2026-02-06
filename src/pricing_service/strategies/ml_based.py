"""
ML-Based Pricing Strategy.

Uses machine learning to predict optimal surge coefficients
based on historical data and real-time features.

Features:
- Gradient boosting model for surge prediction
- Feature engineering from market state
- Elasticity-aware pricing
- Confidence intervals for predictions
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.common.logging import get_logger
from src.common.config import settings
from src.pricing_service.strategies.base import (
    BasePricingStrategy,
    PricingContext,
    PricingResult,
    StrategyType,
    MarketState,
)

logger = get_logger(__name__)

# Model storage path
PRICING_MODELS_DIR = Path("data/models/pricing")


class PricingFeatureExtractor:
    """
    Feature extraction for ML-based pricing.

    Converts market state and context into model-ready features.
    """

    FEATURE_NAMES = [
        # Supply/Demand features
        "pending_orders",
        "available_couriers",
        "busy_couriers",
        "demand_supply_ratio",
        "courier_utilization",

        # Temporal features
        "hour",
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "is_weekend",
        "is_lunch_peak",
        "is_dinner_peak",
        "is_late_night",

        # Historical features
        "demand_trend",
        "predicted_demand_1h",
        "predicted_demand_confidence",

        # Context features
        "order_value_normalized",
        "restaurant_rating",
        "restaurant_prep_time",

        # Weather
        "weather_multiplier",
    ]

    def extract(self, context: PricingContext) -> dict[str, float]:
        """Extract features from pricing context."""
        market = context.market_state
        now = datetime.now(timezone.utc)

        # Handle infinite ratio
        ratio = market.demand_supply_ratio
        if ratio == float('inf'):
            ratio = 10.0  # Cap for model input

        features = {
            # Supply/Demand
            "pending_orders": float(market.pending_orders),
            "available_couriers": float(market.available_couriers),
            "busy_couriers": float(market.busy_couriers),
            "demand_supply_ratio": min(ratio, 10.0),
            "courier_utilization": market.courier_utilization,

            # Temporal
            "hour": float(now.hour),
            "hour_sin": float(np.sin(2 * np.pi * now.hour / 24)),
            "hour_cos": float(np.cos(2 * np.pi * now.hour / 24)),
            "day_of_week": float(now.weekday()),
            "is_weekend": float(now.weekday() >= 5),
            "is_lunch_peak": float(11 <= now.hour <= 14),
            "is_dinner_peak": float(18 <= now.hour <= 21),
            "is_late_night": float(now.hour >= 22 or now.hour <= 5),

            # Historical
            "demand_trend": market.demand_trend,
            "predicted_demand_1h": market.predicted_demand_1h,
            "predicted_demand_confidence": market.predicted_demand_confidence,

            # Context
            "order_value_normalized": context.order_value / 500.0,  # Normalize to ~1
            "restaurant_rating": context.restaurant_rating,
            "restaurant_prep_time": float(context.restaurant_prep_time_min),

            # Weather
            "weather_multiplier": market.weather_multiplier,
        }

        return features

    def to_array(self, features: dict[str, float]) -> np.ndarray:
        """Convert features dict to ordered array."""
        return np.array([features.get(name, 0.0) for name in self.FEATURE_NAMES])


class SurgePredictionModel:
    """
    ML model for surge prediction.

    Wraps a trained model with prediction and uncertainty estimation.
    """

    def __init__(self):
        self.model = None
        self.is_loaded = False
        self.version = "1.0.0"
        self.feature_names = PricingFeatureExtractor.FEATURE_NAMES

    def load(self, path: Optional[Path] = None) -> bool:
        """Load model from disk."""
        if path is None:
            path = PRICING_MODELS_DIR / "surge_model_latest.joblib"

        if not path.exists():
            logger.warning(f"Model not found at {path}")
            return False

        try:
            import joblib
            model_data = joblib.load(path)

            self.model = model_data["model"]
            self.version = model_data.get("version", "1.0.0")
            self.feature_names = model_data.get("feature_names", self.feature_names)
            self.is_loaded = True

            logger.info(f"Loaded pricing model v{self.version}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def predict(self, features: np.ndarray) -> tuple[float, float]:
        """
        Predict surge and confidence.

        Returns:
            Tuple of (predicted_surge, confidence)
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")

        # Reshape for single prediction
        X = features.reshape(1, -1)

        # Predict
        prediction = self.model.predict(X)[0]

        # Estimate confidence (if model supports it)
        confidence = self._estimate_confidence(X)

        return float(prediction), confidence

    def _estimate_confidence(self, X: np.ndarray) -> float:
        """
        Estimate prediction confidence.

        Uses ensemble variance if available, otherwise fixed confidence.
        """
        if hasattr(self.model, 'estimators_'):
            # For ensemble models, use prediction variance
            predictions = np.array([
                est.predict(X)[0] for est in self.model.estimators_[:10]
            ])
            std = predictions.std()
            # Convert std to confidence (lower std = higher confidence)
            confidence = max(0.5, min(0.99, 1.0 - std / 2))
            return confidence

        return 0.85  # Default confidence


class MLBasedStrategy(BasePricingStrategy):
    """
    Machine learning based pricing strategy.

    Uses a trained model to predict optimal surge coefficients.
    Falls back to rule-based strategy if model is unavailable.
    """

    strategy_type = StrategyType.ML_BASED
    version = "1.0.0"

    def __init__(self):
        self.model = SurgePredictionModel()
        self.feature_extractor = PricingFeatureExtractor()
        self._fallback_strategy = None

        # Try to load model
        self.model.load()

    @property
    def fallback_strategy(self):
        """Lazy-load fallback strategy."""
        if self._fallback_strategy is None:
            from src.pricing_service.strategies.rule_based import RuleBasedStrategy
            self._fallback_strategy = RuleBasedStrategy()
        return self._fallback_strategy

    async def calculate(self, context: PricingContext) -> PricingResult:
        """Calculate surge using ML model."""
        start_time = time.perf_counter()

        # Use fallback if model not available
        if not self.model.is_loaded:
            logger.debug("ML model not loaded, using fallback")
            result = await self.fallback_strategy.calculate(context)
            result.strategy_used = StrategyType.RULE_BASED
            result.reason = f"[Fallback] {result.reason}"
            return result

        try:
            # Extract features
            features = self.feature_extractor.extract(context)
            features_array = self.feature_extractor.to_array(features)

            # Predict
            predicted_surge, confidence = self.model.predict(features_array)

            # Apply constraints
            final_surge, is_capped = self.apply_constraints(predicted_surge, context)

            # Calculate loyalty discount
            loyalty_discount = self.calculate_loyalty_discount(context)

            # Calculate final price
            effective_surge = final_surge * (1 - loyalty_discount)
            final_price = context.base_price * effective_surge

            calculation_time = (time.perf_counter() - start_time) * 1000

            market = context.market_state

            return PricingResult(
                surge_coefficient=round(final_surge, 2),
                final_price=round(final_price, 2),
                base_surge=round(predicted_surge, 2),
                time_modifier=1.0,  # Incorporated in ML model
                weather_modifier=market.weather_multiplier,
                loyalty_discount=round(loyalty_discount, 2),
                demand_level=self.classify_demand(market.pending_orders),
                supply_level=self.classify_supply(market.available_couriers),
                strategy_used=self.strategy_type,
                reason=f"ML prediction (confidence: {confidence:.0%})",
                factors=features,
                confidence=round(confidence, 2),
                is_capped=is_capped,
                uncapped_surge=round(predicted_surge, 2),
                calculation_time_ms=round(calculation_time, 3),
            )

        except Exception as e:
            logger.error(f"ML prediction failed: {e}, using fallback")
            result = await self.fallback_strategy.calculate(context)
            result.strategy_used = StrategyType.RULE_BASED
            result.reason = f"[Fallback due to error] {result.reason}"
            return result


class ElasticityAwareStrategy(MLBasedStrategy):
    """
    ML strategy with price elasticity awareness.

    Adjusts surge based on estimated customer price sensitivity
    to maximize revenue while maintaining conversion.
    """

    strategy_type = StrategyType.ML_BASED
    version = "1.1.0"

    # Elasticity parameters (negative values indicate demand decreases with price)
    BASE_ELASTICITY = -1.2  # 1% price increase -> 1.2% demand decrease

    # Elasticity adjustments by customer segment
    SEGMENT_ELASTICITY = {
        "premium": -0.8,     # Less price sensitive
        "regular": -1.2,     # Average sensitivity
        "price_sensitive": -1.8,  # More price sensitive
    }

    def _estimate_customer_segment(self, context: PricingContext) -> str:
        """Estimate customer price sensitivity segment."""
        if context.customer_is_premium:
            return "premium"

        if context.customer_avg_order_value > 800:
            return "premium"
        elif context.customer_avg_order_value < 300:
            return "price_sensitive"

        return "regular"

    def _apply_elasticity_adjustment(
        self,
        surge: float,
        context: PricingContext,
    ) -> float:
        """
        Adjust surge based on elasticity.

        For very elastic customers, we might want to be more
        conservative with surge pricing to maintain conversion.
        """
        segment = self._estimate_customer_segment(context)
        elasticity = self.SEGMENT_ELASTICITY.get(segment, self.BASE_ELASTICITY)

        # If customer is very price sensitive and surge is high,
        # consider a slight reduction to maintain conversion
        if elasticity < -1.5 and surge > 1.5:
            # Soft cap for price-sensitive customers
            adjusted_surge = 1.0 + (surge - 1.0) * 0.85
            logger.debug(
                f"Elasticity adjustment: {surge:.2f} -> {adjusted_surge:.2f} "
                f"(segment: {segment})"
            )
            return adjusted_surge

        return surge

    async def calculate(self, context: PricingContext) -> PricingResult:
        """Calculate with elasticity awareness."""
        result = await super().calculate(context)

        if result.strategy_used == StrategyType.ML_BASED:
            # Apply elasticity adjustment
            original_surge = result.surge_coefficient
            adjusted_surge = self._apply_elasticity_adjustment(original_surge, context)

            if adjusted_surge != original_surge:
                result.surge_coefficient = round(adjusted_surge, 2)
                result.final_price = round(context.base_price * adjusted_surge, 2)
                result.factors["elasticity_adjusted"] = True
                result.factors["customer_segment"] = self._estimate_customer_segment(context)

        return result

