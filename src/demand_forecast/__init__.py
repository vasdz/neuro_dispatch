"""
Demand Forecasting Service - Predicts order demand by location and time.

This service provides ML-based demand prediction using:
- XGBoost / LightGBM gradient boosting models
- H3 hexagonal grid for spatial indexing
- Time-series feature engineering
- Rolling predictions with confidence intervals

Components:
- features: Feature extraction pipelines
- models: ML model wrappers (XGBoost, LightGBM, Ensemble)
- training: Training pipelines with MLflow integration
- inference: Production inference service
- h3_utils: Hexagonal grid utilities
- api: FastAPI endpoints

Usage:
    from src.demand_forecast import DemandPredictor
    from src.demand_forecast.training import train_demand_model
    from src.demand_forecast.h3_utils import get_city_grid
"""

from src.demand_forecast.inference import DemandPredictor, ModelRegistry
from src.demand_forecast.api import router as demand_router

__all__ = [
    "DemandPredictor",
    "ModelRegistry",
    "demand_router",
]
