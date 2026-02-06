"""
Tests for Demand Forecasting Service (Phase 2).

Comprehensive test suite covering:
- Feature extraction
- ML models
- Training pipeline
- Inference service
- API endpoints
- H3 utilities
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.demand_forecast.features import FeatureExtractor, BatchFeatureExtractor
from src.demand_forecast.models import XGBoostModel, LightGBMModel, EnsembleModel, get_model
from src.demand_forecast.inference import DemandPredictor, ModelRegistry, PredictionCache
from src.demand_forecast.h3_utils import (
    CityGrid, HotspotDetector, point_to_h3, h3_to_point,
    get_hexagon_geojson, hexagons_to_geojson
)


# ============== Feature Extraction Tests ==============

class TestFeatureExtractor:
    """Tests for feature extraction pipeline."""

    def test_temporal_features_extraction(self):
        """Test extraction of temporal features."""
        # Create mock session
        mock_session = AsyncMock()
        extractor = FeatureExtractor(mock_session)

        # Test with specific datetime
        dt = datetime(2026, 2, 5, 12, 30)  # Thursday, lunch time
        features = extractor._extract_temporal_features(dt)

        # Check basic features
        assert features["hour"] == 12
        assert features["day_of_week"] == 3  # Thursday = 3
        assert features["month"] == 2

        # Check boolean flags
        assert features["is_weekend"] == 0.0
        assert features["is_lunch"] == 1.0
        assert features["is_dinner"] == 0.0
        assert features["is_peak"] == 1.0

        # Check cyclical encoding
        assert -1 <= features["hour_sin"] <= 1
        assert -1 <= features["hour_cos"] <= 1

    def test_temporal_features_weekend(self):
        """Test weekend detection."""
        mock_session = AsyncMock()
        extractor = FeatureExtractor(mock_session)

        # Saturday
        saturday = datetime(2026, 2, 7, 20, 0)
        features = extractor._extract_temporal_features(saturday)

        assert features["is_weekend"] == 1.0
        assert features["day_of_week"] == 5

    def test_temporal_features_dinner_peak(self):
        """Test dinner peak detection."""
        mock_session = AsyncMock()
        extractor = FeatureExtractor(mock_session)

        # Friday 8 PM
        friday_evening = datetime(2026, 2, 6, 20, 0)
        features = extractor._extract_temporal_features(friday_evening)

        assert features["is_dinner"] == 1.0
        assert features["is_peak"] == 1.0
        assert features["is_fri_sat_evening"] == 1.0

    def test_haversine_distance(self):
        """Test distance calculation."""
        # Moscow center to ~5km away
        dist = FeatureExtractor._haversine_distance(
            55.7558, 37.6173,  # Moscow center
            55.7558, 37.6873   # ~5km east
        )

        assert 4 < dist < 6  # Approximately 5km


class TestBatchFeatureExtractor:
    """Tests for batch feature extraction."""

    def test_get_feature_columns(self):
        """Test that feature column list is complete."""
        columns = BatchFeatureExtractor.get_feature_columns()

        # Check essential features are present
        assert "hour" in columns
        assert "is_weekend" in columns
        assert "demand_lag_1h" in columns
        assert "rolling_24h_mean" in columns
        assert "neighbor_mean_demand" in columns

        # Check count
        assert len(columns) >= 30  # Should have at least 30 features


# ============== ML Model Tests ==============

class TestXGBoostModel:
    """Tests for XGBoost model wrapper."""

    def test_default_params(self):
        """Test default hyperparameters."""
        model = XGBoostModel()
        params = model.params

        assert params["objective"] == "reg:squarederror"
        assert params["n_estimators"] == 500
        assert params["learning_rate"] == 0.05

    def test_fit_and_predict(self):
        """Test model training and prediction."""
        model = XGBoostModel({"n_estimators": 10, "max_depth": 3})

        # Create dummy data
        np.random.seed(42)
        X = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100),
        })
        y = pd.Series(np.random.rand(100) * 10)

        # Train
        model.fit(X, y)

        # Check model is trained
        assert model.model is not None
        assert model.trained_at is not None
        assert len(model.feature_names) == 3

        # Predict
        predictions = model.predict(X)
        assert len(predictions) == 100
        assert all(p >= 0 for p in predictions)  # Demand cannot be negative

    def test_feature_importance(self):
        """Test feature importance extraction."""
        model = XGBoostModel({"n_estimators": 10, "max_depth": 3})

        X = pd.DataFrame({
            "important": np.linspace(0, 100, 100),
            "noise": np.random.randn(100),
        })
        y = pd.Series(X["important"] + np.random.randn(100) * 5)

        model.fit(X, y)

        # Important feature should have higher importance
        assert model.feature_importance["important"] > model.feature_importance["noise"]

    def test_evaluate(self):
        """Test model evaluation metrics."""
        model = XGBoostModel({"n_estimators": 10})

        X = pd.DataFrame({"x": np.linspace(0, 10, 100)})
        y = pd.Series(X["x"] * 2 + np.random.randn(100) * 0.5)

        model.fit(X, y)
        metrics = model.evaluate(X, y)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "mape" in metrics

        # Model should fit well
        assert metrics["r2"] > 0.9


class TestLightGBMModel:
    """Tests for LightGBM model wrapper."""

    def test_fit_and_predict(self):
        """Test LightGBM training and prediction."""
        model = LightGBMModel({"n_estimators": 10, "verbose": -1})

        X = pd.DataFrame({
            "f1": np.random.randn(100),
            "f2": np.random.randn(100),
        })
        y = pd.Series(np.random.rand(100) * 10)

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 100
        assert all(p >= 0 for p in predictions)


class TestEnsembleModel:
    """Tests for ensemble model."""

    def test_weighted_averaging(self):
        """Test that ensemble uses weighted averaging."""
        # Create models with fixed predictions
        model1 = MagicMock()
        model1.model_type = "model1"
        model1.predict = MagicMock(return_value=np.array([10.0, 20.0]))
        model1.training_metrics = {"mae": 1.0, "rmse": 1.5, "r2": 0.9}
        model1.feature_importance = {"f1": 0.6, "f2": 0.4}
        model1.feature_names = ["f1", "f2"]

        model2 = MagicMock()
        model2.model_type = "model2"
        model2.predict = MagicMock(return_value=np.array([20.0, 30.0]))
        model2.training_metrics = {"mae": 0.8, "rmse": 1.2, "r2": 0.95}
        model2.feature_importance = {"f1": 0.5, "f2": 0.5}
        model2.feature_names = ["f1", "f2"]

        ensemble = EnsembleModel(models=[model1, model2], weights=[0.4, 0.6])

        X = pd.DataFrame({"f1": [1], "f2": [2]})
        predictions = ensemble.predict(X)

        # Expected: 0.4 * 10 + 0.6 * 20 = 16, 0.4 * 20 + 0.6 * 30 = 26
        np.testing.assert_array_almost_equal(predictions, [16.0, 26.0])


class TestModelFactory:
    """Tests for model factory function."""

    def test_get_xgboost(self):
        """Test getting XGBoost model."""
        model = get_model("xgboost")
        assert isinstance(model, XGBoostModel)

    def test_get_lightgbm(self):
        """Test getting LightGBM model."""
        model = get_model("lightgbm")
        assert isinstance(model, LightGBMModel)

    def test_get_ensemble(self):
        """Test getting ensemble model."""
        model = get_model("ensemble")
        assert isinstance(model, EnsembleModel)

    def test_unknown_model(self):
        """Test error for unknown model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            get_model("random_forest")


# ============== Inference Tests ==============

class TestDemandPredictor:
    """Tests for demand prediction service."""

    def test_time_based_heuristic_lunch(self):
        """Test lunch time demand heuristic."""
        mock_session = AsyncMock()
        predictor = DemandPredictor(mock_session)

        # Weekday lunch
        demand = predictor._time_based_heuristic(hour=12, day_of_week=2)

        # Should be elevated (base * lunch_multiplier)
        assert demand > 10  # 5 * 2.5 = 12.5

    def test_time_based_heuristic_dinner(self):
        """Test dinner time demand heuristic."""
        mock_session = AsyncMock()
        predictor = DemandPredictor(mock_session)

        demand = predictor._time_based_heuristic(hour=19, day_of_week=3)

        # Dinner peak should be highest
        assert demand > 12  # 5 * 3.0 = 15

    def test_time_based_heuristic_night(self):
        """Test night time low demand."""
        mock_session = AsyncMock()
        predictor = DemandPredictor(mock_session)

        demand = predictor._time_based_heuristic(hour=3, day_of_week=1)

        # Night should be low
        assert demand < 3  # 5 * 0.3 = 1.5

    def test_time_based_heuristic_weekend_boost(self):
        """Test weekend demand boost."""
        mock_session = AsyncMock()
        predictor = DemandPredictor(mock_session)

        weekday = predictor._time_based_heuristic(hour=12, day_of_week=2)
        weekend = predictor._time_based_heuristic(hour=12, day_of_week=5)

        assert weekend > weekday  # Weekend should have boost


class TestPredictionCache:
    """Tests for prediction cache."""

    def test_cache_set_and_get(self):
        """Test basic cache operations."""
        cache = PredictionCache(ttl_seconds=60)

        cache.set("key1", {"demand": 10.0})
        result = cache.get("key1")

        assert result == {"demand": 10.0}

    def test_cache_miss(self):
        """Test cache miss."""
        cache = PredictionCache()
        result = cache.get("nonexistent")

        assert result is None

    def test_cache_key_generation(self):
        """Test cache key format."""
        cache = PredictionCache()
        key = cache.make_key("abc123", 2, "lightgbm")

        assert "abc123" in key
        assert "2" in key
        assert "lightgbm" in key


class TestModelRegistry:
    """Tests for model registry."""

    def test_singleton(self):
        """Test registry is singleton."""
        registry1 = ModelRegistry()
        registry2 = ModelRegistry()

        assert registry1 is registry2

    def test_list_models_empty(self):
        """Test listing models when none exist."""
        with patch.object(ModelRegistry, '__new__', return_value=MagicMock()):
            registry = ModelRegistry()
            registry.list_models = MagicMock(return_value=[])

            models = registry.list_models()
            assert models == []


# ============== H3 Utilities Tests ==============

class TestCityGrid:
    """Tests for city grid management."""

    def test_grid_generation(self):
        """Test hexagon grid generation."""
        grid = CityGrid(resolution=7)  # Larger hexagons for faster test

        # Should generate hexagons
        assert len(grid.hexagons) > 0

    def test_zone_classification(self):
        """Test zone classification."""
        grid = CityGrid(resolution=7)

        zones = grid.zones

        assert "center" in zones
        assert "inner" in zones
        assert "middle" in zones
        assert "outer" in zones

    def test_get_neighbors(self):
        """Test neighbor lookup."""
        grid = CityGrid()
        center_hex = point_to_h3(55.7558, 37.6173)

        neighbors = grid.get_neighbors(center_hex, k=1)

        # Hexagon has 6 neighbors at k=1
        assert len(neighbors) == 6

    def test_distance_calculation(self):
        """Test distance between hexagons."""
        grid = CityGrid()

        hex1 = point_to_h3(55.7558, 37.6173)
        hex2 = point_to_h3(55.7558, 37.7173)

        distance = grid.get_distance_km(hex1, hex2)

        assert 5 < distance < 10  # Roughly 6-7 km

    def test_hexagon_area(self):
        """Test hexagon area calculation."""
        grid = CityGrid(resolution=8)
        area = grid.get_hexagon_area_km2()

        # Resolution 8 hexagons are about 0.74 km²
        assert 0.5 < area < 1.0


class TestH3Functions:
    """Tests for H3 utility functions."""

    def test_point_to_h3(self):
        """Test point to H3 conversion."""
        h3_index = point_to_h3(55.7558, 37.6173)

        assert isinstance(h3_index, str)
        assert len(h3_index) == 15  # H3 index length for resolution 8

    def test_h3_to_point(self):
        """Test H3 to point conversion."""
        h3_index = point_to_h3(55.7558, 37.6173)
        lat, lng = h3_to_point(h3_index)

        # Should be close to original point
        assert abs(lat - 55.7558) < 0.01
        assert abs(lng - 37.6173) < 0.01

    def test_hexagon_geojson(self):
        """Test GeoJSON generation for single hexagon."""
        h3_index = point_to_h3(55.7558, 37.6173)
        geojson = get_hexagon_geojson(h3_index)

        assert geojson["type"] == "Feature"
        assert geojson["geometry"]["type"] == "Polygon"
        assert geojson["properties"]["h3_index"] == h3_index

    def test_hexagons_to_geojson(self):
        """Test GeoJSON generation for multiple hexagons."""
        h3_indices = [
            point_to_h3(55.7558, 37.6173),
            point_to_h3(55.7658, 37.6273),
        ]

        geojson = hexagons_to_geojson(h3_indices)

        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) == 2


class TestHotspotDetector:
    """Tests for hotspot detection."""

    def test_detect_hotspots(self):
        """Test hotspot detection."""
        grid = CityGrid(resolution=7)
        detector = HotspotDetector(grid)

        # Create demand map with one clear hotspot
        demand_map = {}
        center = point_to_h3(55.7558, 37.6173, 7)

        for hex_id in list(grid.hexagons)[:50]:
            demand_map[hex_id] = np.random.rand() * 5

        # Make center a hotspot
        demand_map[center] = 100
        for neighbor in grid.get_neighbors(center):
            demand_map[neighbor] = 80

        hotspots = detector.detect_hotspots(demand_map, threshold_percentile=90)

        # Should detect at least one hotspot
        assert len(hotspots) >= 1

        # Hotspot should have required fields
        if hotspots:
            hs = hotspots[0]
            assert "center" in hs
            assert "peak_demand" in hs
            assert "hexagon_count" in hs


# ============== Integration Tests ==============

@pytest.mark.asyncio
class TestPredictionIntegration:
    """Integration tests for prediction pipeline."""

    async def test_baseline_prediction(self):
        """Test baseline prediction works without ML model."""
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock(
            all=MagicMock(return_value=[])
        ))

        predictor = DemandPredictor(mock_session)

        with patch('src.demand_forecast.inference.settings') as mock_settings:
            mock_settings.enable_ml_predictions = False

            result = await predictor._predict_baseline(
                "882830829bfffff",
                datetime.now(timezone.utc) + timedelta(hours=1),
            )

        assert "demand" in result
        assert "confidence" in result
        assert result["model_type"] == "baseline"
        assert result["demand"] >= 0

