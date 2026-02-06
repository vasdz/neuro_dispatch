"""
Feature Engineering for Demand Forecasting.

This module provides feature extraction pipelines for time-series
demand prediction using H3 hexagonal grid data.

Senior-level implementation with:
- Temporal features (hour, day, week, holidays)
- Spatial features (neighbors, zones)
- Lag features (historical patterns)
- Rolling statistics
"""

from datetime import datetime, timedelta
from typing import Any

import h3
import numpy as np
import pandas as pd
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.config import settings
from src.common.models import DemandHourly, CourierTelemetry, Order, OrderStatus
from src.common.logging import get_logger

logger = get_logger(__name__)


# Russian holidays for 2024-2026 (simplified)
RUSSIAN_HOLIDAYS = {
    # New Year holidays
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
    # Defender of the Fatherland Day
    (2, 23),
    # International Women's Day
    (3, 8),
    # Spring and Labour Day
    (5, 1),
    # Victory Day
    (5, 9),
    # Russia Day
    (6, 12),
    # Unity Day
    (11, 4),
}


class FeatureExtractor:
    """
    Advanced feature extractor for demand prediction.

    Features are organized in groups:
    1. Temporal - time-based patterns
    2. Spatial - location-based patterns
    3. Historical - lag and rolling features
    4. External - weather, events (future)
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self._feature_cache: dict[str, Any] = {}

    async def extract_features(
        self,
        h3_index: str,
        target_time: datetime,
        include_neighbors: bool = True,
    ) -> dict[str, float]:
        """
        Extract all features for a given hexagon and target time.

        Args:
            h3_index: H3 hexagon index
            target_time: Target datetime for prediction
            include_neighbors: Whether to include neighbor hexagon features

        Returns:
            Dictionary of feature names to values
        """
        features: dict[str, float] = {}

        # 1. Temporal features
        features.update(self._extract_temporal_features(target_time))

        # 2. Spatial features
        features.update(await self._extract_spatial_features(h3_index, target_time))

        # 3. Historical lag features
        features.update(await self._extract_lag_features(h3_index, target_time))

        # 4. Rolling statistics
        features.update(await self._extract_rolling_features(h3_index, target_time))

        # 5. Neighbor features (optional, computationally expensive)
        if include_neighbors:
            features.update(await self._extract_neighbor_features(h3_index, target_time))

        return features

    def _extract_temporal_features(self, dt: datetime) -> dict[str, float]:
        """
        Extract time-based features.

        Uses cyclical encoding for periodic features to preserve
        continuity (e.g., hour 23 is close to hour 0).
        """
        features = {}

        # Basic time components
        features["hour"] = float(dt.hour)
        features["day_of_week"] = float(dt.weekday())
        features["day_of_month"] = float(dt.day)
        features["month"] = float(dt.month)
        features["week_of_year"] = float(dt.isocalendar()[1])

        # Cyclical encoding for hour (sin/cos transformation)
        hour_rad = 2 * np.pi * dt.hour / 24
        features["hour_sin"] = float(np.sin(hour_rad))
        features["hour_cos"] = float(np.cos(hour_rad))

        # Cyclical encoding for day of week
        dow_rad = 2 * np.pi * dt.weekday() / 7
        features["dow_sin"] = float(np.sin(dow_rad))
        features["dow_cos"] = float(np.cos(dow_rad))

        # Cyclical encoding for month
        month_rad = 2 * np.pi * (dt.month - 1) / 12
        features["month_sin"] = float(np.sin(month_rad))
        features["month_cos"] = float(np.cos(month_rad))

        # Boolean flags
        features["is_weekend"] = float(dt.weekday() >= 5)
        features["is_holiday"] = float((dt.month, dt.day) in RUSSIAN_HOLIDAYS)

        # Meal time flags (key for food delivery)
        features["is_breakfast"] = float(7 <= dt.hour < 10)
        features["is_lunch"] = float(11 <= dt.hour < 14)
        features["is_dinner"] = float(18 <= dt.hour < 22)
        features["is_late_night"] = float(dt.hour >= 22 or dt.hour < 6)

        # Peak hours indicator
        features["is_peak"] = float(
            (11 <= dt.hour < 14) or (18 <= dt.hour < 21)
        )

        # Friday/Saturday evening (high demand period)
        features["is_fri_sat_evening"] = float(
            dt.weekday() in (4, 5) and 18 <= dt.hour < 23
        )

        return features

    async def _extract_spatial_features(
        self,
        h3_index: str,
        target_time: datetime,
    ) -> dict[str, float]:
        """
        Extract location-based features.
        """
        features = {}

        # H3 cell properties
        resolution = h3.h3_get_resolution(h3_index)
        features["h3_resolution"] = float(resolution)

        # Cell center coordinates
        lat, lng = h3.h3_to_geo(h3_index)
        features["latitude"] = lat
        features["longitude"] = lng

        # Distance from city center (Moscow: 55.7558, 37.6173)
        city_center_lat, city_center_lng = settings.default_city_lat, settings.default_city_lng
        features["distance_from_center_km"] = self._haversine_distance(
            lat, lng, city_center_lat, city_center_lng
        )

        # Zone classification based on distance
        dist = features["distance_from_center_km"]
        features["zone_center"] = float(dist < 5)
        features["zone_inner"] = float(5 <= dist < 10)
        features["zone_middle"] = float(10 <= dist < 15)
        features["zone_outer"] = float(dist >= 15)

        # Current courier availability in this hexagon
        features["available_couriers"] = await self._get_available_couriers_count(
            h3_index, target_time
        )

        return features

    async def _extract_lag_features(
        self,
        h3_index: str,
        target_time: datetime,
    ) -> dict[str, float]:
        """
        Extract historical lag features.

        Lags capture autocorrelation in time series:
        - Short-term (1-3 hours)
        - Same hour previous days
        - Same hour same weekday
        """
        features = {}

        # Short-term lags (1, 2, 3 hours ago)
        for lag in [1, 2, 3]:
            lag_time = target_time - timedelta(hours=lag)
            demand = await self._get_demand_at_time(h3_index, lag_time)
            features[f"demand_lag_{lag}h"] = demand

        # Same hour yesterday
        yesterday = target_time - timedelta(days=1)
        features["demand_lag_24h"] = await self._get_demand_at_time(h3_index, yesterday)

        # Same hour, same weekday last week
        last_week = target_time - timedelta(days=7)
        features["demand_lag_7d"] = await self._get_demand_at_time(h3_index, last_week)

        # Same hour, 2 weeks ago
        two_weeks_ago = target_time - timedelta(days=14)
        features["demand_lag_14d"] = await self._get_demand_at_time(h3_index, two_weeks_ago)

        # Difference features (momentum)
        features["demand_diff_1h"] = features["demand_lag_1h"] - features.get("demand_lag_2h", 0)
        features["demand_diff_24h"] = features["demand_lag_1h"] - features["demand_lag_24h"]

        return features

    async def _extract_rolling_features(
        self,
        h3_index: str,
        target_time: datetime,
    ) -> dict[str, float]:
        """
        Extract rolling window statistics.

        Rolling features capture trends and seasonality:
        - Mean, std, min, max over various windows
        - Trend indicators
        """
        features = {}

        # Last 6 hours rolling stats
        rolling_6h = await self._get_rolling_stats(h3_index, target_time, hours=6)
        features["rolling_6h_mean"] = rolling_6h["mean"]
        features["rolling_6h_std"] = rolling_6h["std"]
        features["rolling_6h_max"] = rolling_6h["max"]
        features["rolling_6h_min"] = rolling_6h["min"]

        # Last 24 hours rolling stats
        rolling_24h = await self._get_rolling_stats(h3_index, target_time, hours=24)
        features["rolling_24h_mean"] = rolling_24h["mean"]
        features["rolling_24h_std"] = rolling_24h["std"]

        # Last 7 days rolling stats (daily average)
        rolling_7d = await self._get_rolling_stats(h3_index, target_time, hours=168)
        features["rolling_7d_mean"] = rolling_7d["mean"]

        # Trend: compare recent to historical
        if rolling_24h["mean"] > 0 and rolling_7d["mean"] > 0:
            features["trend_ratio"] = rolling_24h["mean"] / rolling_7d["mean"]
        else:
            features["trend_ratio"] = 1.0

        return features

    async def _extract_neighbor_features(
        self,
        h3_index: str,
        target_time: datetime,
    ) -> dict[str, float]:
        """
        Extract features from neighboring hexagons.

        Spatial autocorrelation: nearby areas have similar demand.
        """
        features = {}

        # Get ring of neighbors (k-ring with k=1)
        neighbors = h3.k_ring(h3_index, 1)
        neighbors = [n for n in neighbors if n != h3_index]

        if not neighbors:
            features["neighbor_mean_demand"] = 0.0
            features["neighbor_total_couriers"] = 0.0
            return features

        # Get demand for all neighbors
        neighbor_demands = []
        neighbor_couriers = 0

        for neighbor in neighbors:
            demand = await self._get_demand_at_time(neighbor, target_time - timedelta(hours=1))
            neighbor_demands.append(demand)
            neighbor_couriers += await self._get_available_couriers_count(neighbor, target_time)

        features["neighbor_mean_demand"] = float(np.mean(neighbor_demands)) if neighbor_demands else 0.0
        features["neighbor_max_demand"] = float(np.max(neighbor_demands)) if neighbor_demands else 0.0
        features["neighbor_total_couriers"] = float(neighbor_couriers)
        features["neighbor_count"] = float(len(neighbors))

        return features

    async def _get_demand_at_time(
        self,
        h3_index: str,
        time: datetime,
    ) -> float:
        """Get demand value at specific time (with tolerance)."""
        start_time = time - timedelta(minutes=30)
        end_time = time + timedelta(minutes=30)

        result = await self.session.execute(
            select(func.avg(DemandHourly.order_count))
            .where(
                DemandHourly.h3_index == h3_index,
                DemandHourly.time >= start_time,
                DemandHourly.time <= end_time,
            )
        )

        value = result.scalar()
        return float(value) if value is not None else 0.0

    async def _get_rolling_stats(
        self,
        h3_index: str,
        end_time: datetime,
        hours: int,
    ) -> dict[str, float]:
        """Calculate rolling statistics over a time window."""
        start_time = end_time - timedelta(hours=hours)

        result = await self.session.execute(
            select(
                func.avg(DemandHourly.order_count),
                func.stddev(DemandHourly.order_count),
                func.max(DemandHourly.order_count),
                func.min(DemandHourly.order_count),
            )
            .where(
                DemandHourly.h3_index == h3_index,
                DemandHourly.time >= start_time,
                DemandHourly.time < end_time,
            )
        )

        row = result.one_or_none()

        if row and row[0] is not None:
            return {
                "mean": float(row[0]),
                "std": float(row[1]) if row[1] is not None else 0.0,
                "max": float(row[2]) if row[2] is not None else 0.0,
                "min": float(row[3]) if row[3] is not None else 0.0,
            }

        return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}

    async def _get_available_couriers_count(
        self,
        h3_index: str,
        time: datetime,
    ) -> float:
        """Get count of available couriers in hexagon at given time."""
        # For now, use the latest DemandHourly record
        result = await self.session.execute(
            select(DemandHourly.available_couriers)
            .where(DemandHourly.h3_index == h3_index)
            .order_by(DemandHourly.time.desc())
            .limit(1)
        )

        value = result.scalar()
        return float(value) if value is not None else 0.0

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in kilometers."""
        R = 6371  # Earth's radius in km

        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)

        a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return float(R * c)


class BatchFeatureExtractor:
    """
    Batch feature extraction for training data preparation.

    Optimized for bulk extraction with database query batching.
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.feature_extractor = FeatureExtractor(session)

    async def extract_training_dataset(
        self,
        start_date: datetime,
        end_date: datetime,
        h3_indices: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Extract features for training dataset.

        Args:
            start_date: Start of training period
            end_date: End of training period
            h3_indices: Optional list of specific hexagons (all if None)

        Returns:
            DataFrame with features and target
        """
        logger.info(
            "Extracting training dataset",
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )

        # Get all demand records in range
        query = select(DemandHourly).where(
            DemandHourly.time >= start_date,
            DemandHourly.time <= end_date,
        )

        if h3_indices:
            query = query.where(DemandHourly.h3_index.in_(h3_indices))

        result = await self.session.execute(query.order_by(DemandHourly.time))
        records = result.scalars().all()

        logger.info(f"Found {len(records)} demand records for feature extraction")

        # Extract features for each record
        data_rows = []

        for i, record in enumerate(records):
            try:
                features = await self.feature_extractor.extract_features(
                    record.h3_index,
                    record.time,
                    include_neighbors=True,
                )

                # Add target and identifiers
                features["target"] = float(record.order_count)
                features["h3_index"] = record.h3_index
                features["timestamp"] = record.time

                data_rows.append(features)

                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(records)} records")

            except Exception as e:
                logger.warning(
                    f"Failed to extract features for record",
                    h3_index=record.h3_index,
                    time=record.time,
                    error=str(e),
                )

        df = pd.DataFrame(data_rows)
        logger.info(f"Extracted {len(df)} feature rows with {len(df.columns)} columns")

        return df

    @staticmethod
    def get_feature_columns() -> list[str]:
        """Get list of feature column names (excluding target and identifiers)."""
        # All features extracted by FeatureExtractor
        return [
            # Temporal
            "hour", "day_of_week", "day_of_month", "month", "week_of_year",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
            "is_weekend", "is_holiday", "is_breakfast", "is_lunch", "is_dinner",
            "is_late_night", "is_peak", "is_fri_sat_evening",
            # Spatial
            "h3_resolution", "latitude", "longitude", "distance_from_center_km",
            "zone_center", "zone_inner", "zone_middle", "zone_outer", "available_couriers",
            # Lag
            "demand_lag_1h", "demand_lag_2h", "demand_lag_3h",
            "demand_lag_24h", "demand_lag_7d", "demand_lag_14d",
            "demand_diff_1h", "demand_diff_24h",
            # Rolling
            "rolling_6h_mean", "rolling_6h_std", "rolling_6h_max", "rolling_6h_min",
            "rolling_24h_mean", "rolling_24h_std", "rolling_7d_mean", "trend_ratio",
            # Neighbors
            "neighbor_mean_demand", "neighbor_max_demand",
            "neighbor_total_couriers", "neighbor_count",
        ]

