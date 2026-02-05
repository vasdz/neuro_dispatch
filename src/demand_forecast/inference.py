"""
Demand Prediction Model Inference.

Provides predictions for order demand by hexagon and time.
Currently uses a rule-based baseline; will be replaced with ML model.
"""

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.models import DemandHourly
from src.common.logging import get_logger

logger = get_logger(__name__)


class DemandPredictor:
    """
    Demand prediction service.

    Currently implements a simple rule-based baseline.
    Will be upgraded to XGBoost/LightGBM in Phase 2.
    """

    MODEL_VERSION = "baseline-v1"

    def __init__(self, session: AsyncSession):
        self.session = session

    async def predict(
        self,
        h3_index: str,
        hours_ahead: int = 1,
    ) -> dict[str, Any]:
        """
        Predict demand for a given hexagon.

        Args:
            h3_index: H3 hexagon index
            hours_ahead: Number of hours to predict ahead

        Returns:
            Dictionary with prediction and metadata
        """
        # Get historical average for this hour of day
        target_time = datetime.utcnow() + timedelta(hours=hours_ahead)
        target_hour = target_time.hour
        target_day = target_time.weekday()

        # Look at same hour from past 7 days
        historical_demands = await self._get_historical_average(
            h3_index,
            target_hour,
            days_back=7,
        )

        if historical_demands:
            predicted_demand = sum(historical_demands) / len(historical_demands)
            confidence = min(0.9, 0.5 + len(historical_demands) * 0.05)
        else:
            # Fallback to time-of-day heuristic
            predicted_demand = self._time_based_heuristic(target_hour, target_day)
            confidence = 0.3

        return {
            "demand": round(predicted_demand, 1),
            "confidence": round(confidence, 2),
            "model_version": self.MODEL_VERSION,
            "features_used": ["hour_of_day", "day_of_week", "historical_avg"],
        }

    async def _get_historical_average(
        self,
        h3_index: str,
        hour: int,
        days_back: int = 7,
    ) -> list[int]:
        """Get historical demand values for the same hour."""
        since = datetime.utcnow() - timedelta(days=days_back)

        result = await self.session.execute(
            select(DemandHourly.order_count)
            .where(
                DemandHourly.h3_index == h3_index,
                DemandHourly.time >= since,
                func.extract("hour", DemandHourly.time) == hour,
            )
        )

        return [row[0] for row in result.all() if row[0] is not None]

    def _time_based_heuristic(self, hour: int, day_of_week: int) -> float:
        """
        Simple time-of-day heuristic for demand.

        Peak hours:
        - Lunch: 11-14
        - Dinner: 18-21

        Weekend adjustment: +20%
        """
        base_demand = 5.0

        # Time of day adjustments
        if 11 <= hour <= 14:
            base_demand = 12.0  # Lunch peak
        elif 18 <= hour <= 21:
            base_demand = 15.0  # Dinner peak
        elif 22 <= hour or hour <= 6:
            base_demand = 2.0  # Night low

        # Weekend boost
        if day_of_week >= 5:  # Saturday, Sunday
            base_demand *= 1.2

        return base_demand

    async def batch_predict(
        self,
        h3_indexes: list[str],
        hours_ahead: int = 1,
    ) -> dict[str, dict]:
        """Batch prediction for multiple hexagons."""
        predictions = {}

        for h3_index in h3_indexes:
            predictions[h3_index] = await self.predict(h3_index, hours_ahead)

        return predictions

