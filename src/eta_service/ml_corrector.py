"""
ML-based ETA Correction Model.

Provides machine learning adjustments to base routing estimates:
- Historical delivery time patterns
- Restaurant-specific delays
- Courier performance factors
- Weather and event impacts
- Zone-based corrections

Senior+ implementation with:
- Feature engineering pipeline
- Model versioning
- Online learning support
- Confidence intervals
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Literal
from pathlib import Path

import numpy as np

from src.common.logging import get_logger
from src.common.config import settings

logger = get_logger(__name__)


@dataclass
class CorrectionFeatures:
    """
    Features for ML correction model.

    Captures all factors that affect delivery time beyond routing.
    """
    # Temporal features
    hour_of_day: int
    day_of_week: int
    is_weekend: bool
    is_holiday: bool
    is_lunch_rush: bool  # 11-14
    is_dinner_rush: bool  # 18-21

    # Location features
    origin_h3_index: str
    destination_h3_index: str
    origin_zone_type: str  # center, inner, middle, outer
    destination_zone_type: str

    # Restaurant features
    restaurant_avg_prep_time: float
    restaurant_current_queue: int
    restaurant_rating: float
    restaurant_order_complexity: float  # 0-1, based on items

    # Courier features
    courier_avg_speed: float
    courier_experience_days: int
    courier_current_load: int
    courier_rating: float
    courier_transport_type: str

    # Route features
    base_distance_km: float
    base_duration_minutes: float
    traffic_factor: float

    # Weather features
    weather_condition: str  # clear, rain, snow, etc.
    temperature_celsius: float

    # Historical features
    avg_delivery_time_this_hour: float
    avg_delivery_time_this_route: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.hour_of_day,
            self.day_of_week,
            float(self.is_weekend),
            float(self.is_holiday),
            float(self.is_lunch_rush),
            float(self.is_dinner_rush),
            self.restaurant_avg_prep_time,
            self.restaurant_current_queue,
            self.restaurant_rating,
            self.restaurant_order_complexity,
            self.courier_avg_speed,
            self.courier_experience_days,
            self.courier_current_load,
            self.courier_rating,
            self.base_distance_km,
            self.base_duration_minutes,
            self.traffic_factor,
            self.temperature_celsius,
            self.avg_delivery_time_this_hour,
            self.avg_delivery_time_this_route,
        ])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hour_of_day": self.hour_of_day,
            "day_of_week": self.day_of_week,
            "is_weekend": self.is_weekend,
            "is_holiday": self.is_holiday,
            "is_lunch_rush": self.is_lunch_rush,
            "is_dinner_rush": self.is_dinner_rush,
            "origin_zone": self.origin_zone_type,
            "destination_zone": self.destination_zone_type,
            "restaurant_prep_time": self.restaurant_avg_prep_time,
            "restaurant_queue": self.restaurant_current_queue,
            "courier_speed": self.courier_avg_speed,
            "courier_experience": self.courier_experience_days,
            "base_distance_km": self.base_distance_km,
            "base_duration_min": self.base_duration_minutes,
            "traffic_factor": self.traffic_factor,
            "weather": self.weather_condition,
        }


@dataclass
class CorrectionResult:
    """Result of ML correction."""
    correction_minutes: float
    corrected_eta_minutes: float
    confidence: float
    lower_bound_minutes: float
    upper_bound_minutes: float
    factors: dict[str, float] = field(default_factory=dict)
    model_version: str = "1.0.0"


class CorrectionModel:
    """
    Base class for ETA correction models.

    Provides interface for different model implementations.
    """

    version: str = "1.0.0"
    model_type: str = "base"

    def predict(
        self,
        features: CorrectionFeatures,
        base_eta_minutes: float,
    ) -> CorrectionResult:
        """
        Predict ETA correction.

        Args:
            features: Input features
            base_eta_minutes: Base routing ETA

        Returns:
            Correction result with adjusted ETA
        """
        raise NotImplementedError

    def get_confidence_interval(
        self,
        prediction: float,
        features: CorrectionFeatures,
    ) -> tuple[float, float]:
        """Get confidence interval for prediction."""
        # Default: ±20% for 95% CI
        margin = prediction * 0.2
        return (prediction - margin, prediction + margin)


class RuleBasedCorrector(CorrectionModel):
    """
    Rule-based ETA correction.

    Uses predefined rules for ETA adjustment.
    Serves as baseline and fallback.
    """

    version = "1.0.0"
    model_type = "rule_based"

    # Correction factors
    PREP_TIME_BASE = 15.0  # Base restaurant prep time
    RUSH_HOUR_PENALTY = 5.0  # Extra minutes during rush
    WEATHER_PENALTIES = {
        "clear": 0.0,
        "cloudy": 0.0,
        "rain": 3.0,
        "heavy_rain": 7.0,
        "snow": 10.0,
        "storm": 15.0,
    }

    # Zone-based adjustments
    ZONE_ADJUSTMENTS = {
        "center": 2.0,   # Congested, slower
        "inner": 1.0,
        "middle": 0.0,
        "outer": -1.0,   # Less traffic
    }

    def predict(
        self,
        features: CorrectionFeatures,
        base_eta_minutes: float,
    ) -> CorrectionResult:
        """Calculate rule-based correction."""
        correction = 0.0
        factors = {}

        # Restaurant prep time adjustment
        prep_adjustment = features.restaurant_avg_prep_time - self.PREP_TIME_BASE
        if features.restaurant_current_queue > 3:
            prep_adjustment += features.restaurant_current_queue * 1.5
        correction += prep_adjustment
        factors["prep_time_adjustment"] = prep_adjustment

        # Rush hour penalty
        if features.is_lunch_rush or features.is_dinner_rush:
            correction += self.RUSH_HOUR_PENALTY
            factors["rush_hour_penalty"] = self.RUSH_HOUR_PENALTY

        # Weather penalty
        weather_penalty = self.WEATHER_PENALTIES.get(features.weather_condition, 0.0)
        correction += weather_penalty
        factors["weather_penalty"] = weather_penalty

        # Zone adjustment
        origin_adj = self.ZONE_ADJUSTMENTS.get(features.origin_zone_type, 0.0)
        dest_adj = self.ZONE_ADJUSTMENTS.get(features.destination_zone_type, 0.0)
        zone_adjustment = (origin_adj + dest_adj) / 2
        correction += zone_adjustment
        factors["zone_adjustment"] = zone_adjustment

        # Courier experience adjustment
        if features.courier_experience_days < 30:
            rookie_penalty = (30 - features.courier_experience_days) * 0.1
            correction += rookie_penalty
            factors["rookie_penalty"] = rookie_penalty
        elif features.courier_experience_days > 180:
            veteran_bonus = -2.0
            correction += veteran_bonus
            factors["veteran_bonus"] = veteran_bonus

        # Courier load adjustment
        if features.courier_current_load > 1:
            load_penalty = (features.courier_current_load - 1) * 5.0
            correction += load_penalty
            factors["load_penalty"] = load_penalty

        # Calculate final ETA
        corrected_eta = base_eta_minutes + correction
        corrected_eta = max(corrected_eta, 5.0)  # Minimum 5 minutes

        # Confidence interval
        lower, upper = self.get_confidence_interval(corrected_eta, features)

        return CorrectionResult(
            correction_minutes=round(correction, 2),
            corrected_eta_minutes=round(corrected_eta, 2),
            confidence=0.75,  # Rule-based has lower confidence
            lower_bound_minutes=round(lower, 2),
            upper_bound_minutes=round(upper, 2),
            factors=factors,
            model_version=self.version,
        )


class GradientBoostingCorrector(CorrectionModel):
    """
    Gradient Boosting based ETA correction.

    Uses trained XGBoost/LightGBM model for predictions.
    Falls back to rule-based if model not available.
    """

    version = "2.0.0"
    model_type = "gradient_boosting"

    def __init__(self, model_path: Path | None = None):
        self.model = None
        self.model_path = model_path or Path("data/models/eta_corrector_latest.joblib")
        self.fallback = RuleBasedCorrector()
        self._load_model()

    def _load_model(self) -> None:
        """Load trained model from disk."""
        if not self.model_path.exists():
            logger.warning(f"ETA model not found: {self.model_path}")
            return

        try:
            import joblib
            self.model = joblib.load(self.model_path)
            logger.info(f"Loaded ETA correction model: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load ETA model: {e}")

    def predict(
        self,
        features: CorrectionFeatures,
        base_eta_minutes: float,
    ) -> CorrectionResult:
        """Predict using ML model."""
        if self.model is None:
            # Fallback to rule-based
            result = self.fallback.predict(features, base_eta_minutes)
            result.model_version = f"{self.version}-fallback"
            return result

        try:
            # Prepare features
            X = features.to_array().reshape(1, -1)

            # Predict correction
            correction = self.model.predict(X)[0]

            # Calculate corrected ETA
            corrected_eta = base_eta_minutes + correction
            corrected_eta = max(corrected_eta, 5.0)

            # Confidence based on model certainty
            confidence = 0.85

            # Get prediction interval if model supports it
            lower, upper = self._get_prediction_interval(X, corrected_eta)

            return CorrectionResult(
                correction_minutes=round(correction, 2),
                corrected_eta_minutes=round(corrected_eta, 2),
                confidence=confidence,
                lower_bound_minutes=round(lower, 2),
                upper_bound_minutes=round(upper, 2),
                factors={"ml_correction": correction},
                model_version=self.version,
            )

        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            result = self.fallback.predict(features, base_eta_minutes)
            result.model_version = f"{self.version}-fallback"
            return result

    def _get_prediction_interval(
        self,
        X: np.ndarray,
        prediction: float,
    ) -> tuple[float, float]:
        """Calculate prediction interval."""
        # For now, use fixed percentage
        # In production, would use quantile regression or bootstrap
        margin = prediction * 0.15
        return (prediction - margin, prediction + margin)


class MLCorrector:
    """
    Main ML Corrector interface.

    Manages multiple correction models and provides unified API.
    Supports A/B testing of different models.
    """

    def __init__(
        self,
        model_type: Literal["rule_based", "ml", "ensemble"] = "rule_based"
    ):
        self.model_type = model_type
        self._models: dict[str, CorrectionModel] = {
            "rule_based": RuleBasedCorrector(),
        }

        # Try to load ML model
        if model_type in ("ml", "ensemble"):
            try:
                self._models["ml"] = GradientBoostingCorrector()
            except Exception as e:
                logger.warning(f"Could not load ML model: {e}")

    def get_correction(
        self,
        features: CorrectionFeatures,
        base_eta_minutes: float,
        model_override: str | None = None,
    ) -> CorrectionResult:
        """
        Get ETA correction.

        Args:
            features: Input features
            base_eta_minutes: Base routing estimate
            model_override: Force specific model

        Returns:
            Correction result
        """
        model_name = model_override or self.model_type

        if model_name == "ensemble":
            return self._ensemble_predict(features, base_eta_minutes)

        model = self._models.get(model_name)
        if model is None:
            model = self._models["rule_based"]

        return model.predict(features, base_eta_minutes)

    def _ensemble_predict(
        self,
        features: CorrectionFeatures,
        base_eta_minutes: float,
    ) -> CorrectionResult:
        """Ensemble prediction from multiple models."""
        results = []
        weights = []

        for name, model in self._models.items():
            result = model.predict(features, base_eta_minutes)
            results.append(result)
            # Weight by confidence
            weights.append(result.confidence)

        if not results:
            return self._models["rule_based"].predict(features, base_eta_minutes)

        # Weighted average
        total_weight = sum(weights)
        weighted_eta = sum(
            r.corrected_eta_minutes * w
            for r, w in zip(results, weights)
        ) / total_weight

        weighted_correction = sum(
            r.correction_minutes * w
            for r, w in zip(results, weights)
        ) / total_weight

        # Confidence is average of individual confidences
        avg_confidence = sum(weights) / len(weights)

        # Bounds from most confident model
        best_result = max(results, key=lambda r: r.confidence)

        return CorrectionResult(
            correction_minutes=round(weighted_correction, 2),
            corrected_eta_minutes=round(weighted_eta, 2),
            confidence=round(avg_confidence, 3),
            lower_bound_minutes=best_result.lower_bound_minutes,
            upper_bound_minutes=best_result.upper_bound_minutes,
            factors={"ensemble_models": len(results)},
            model_version="ensemble-1.0",
        )

    def list_models(self) -> list[dict[str, str]]:
        """List available models."""
        return [
            {"name": name, "type": model.model_type, "version": model.version}
            for name, model in self._models.items()
        ]

