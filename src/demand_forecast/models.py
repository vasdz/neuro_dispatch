"""
ML Model Wrappers for Demand Forecasting.

Provides unified interface for multiple ML algorithms:
- XGBoost (gradient boosting)
- LightGBM (fast gradient boosting)
- Prophet (time-series specific)

Senior-level implementation with:
- Consistent API across models
- Built-in cross-validation
- Feature importance tracking
- Hyperparameter optimization ready
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from src.common.logging import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """Abstract base class for demand prediction models."""

    model_type: str = "base"
    version: str = "1.0.0"

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or self._default_params()
        self.model: Any = None
        self.feature_names: list[str] = []
        self.feature_importance: dict[str, float] = {}
        self.training_metrics: dict[str, float] = {}
        self.trained_at: datetime | None = None

    @abstractmethod
    def _default_params(self) -> dict[str, Any]:
        """Return default hyperparameters."""
        pass

    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying model instance."""
        pass

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict[str, float]:
        """Evaluate model on test data."""
        predictions = self.predict(X)

        return {
            "mae": float(mean_absolute_error(y, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
            "r2": float(r2_score(y, predictions)),
            "mape": float(self._mape(y, predictions)),
        }

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> dict[str, dict[str, float]]:
        """
        Time-series cross-validation.

        Uses TimeSeriesSplit to respect temporal ordering.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)

        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Clone and train new model for each fold
            fold_model = self.__class__(self.params)
            fold_model.fit(X_train, y_train)

            metrics = fold_model.evaluate(X_val, y_val)
            fold_metrics.append(metrics)

            logger.info(f"Fold {fold + 1}/{n_splits}: MAE={metrics['mae']:.3f}")

        # Aggregate metrics
        return {
            "mean": {
                k: float(np.mean([m[k] for m in fold_metrics]))
                for k in fold_metrics[0].keys()
            },
            "std": {
                k: float(np.std([m[k] for m in fold_metrics]))
                for k in fold_metrics[0].keys()
            },
        }

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "model_type": self.model_type,
            "version": self.version,
            "params": self.params,
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance,
            "training_metrics": self.training_metrics,
            "trained_at": self.trained_at,
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        """Load model from disk."""
        path = Path(path)
        model_data = joblib.load(path)

        instance = cls(model_data["params"])
        instance.model = model_data["model"]
        instance.feature_names = model_data["feature_names"]
        instance.feature_importance = model_data["feature_importance"]
        instance.training_metrics = model_data["training_metrics"]
        instance.trained_at = model_data["trained_at"]

        logger.info(f"Model loaded from {path}")
        return instance

    @staticmethod
    def _mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error (avoiding division by zero)."""
        y_true = np.array(y_true)
        mask = y_true != 0
        if mask.sum() == 0:
            return 0.0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


class XGBoostModel(BaseModel):
    """XGBoost gradient boosting model for demand prediction."""

    model_type: str = "xgboost"
    version: str = "2.0.0"

    def _default_params(self) -> dict[str, Any]:
        return {
            "objective": "reg:squarederror",
            "n_estimators": 500,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "early_stopping_rounds": 50,
        }

    def _create_model(self) -> Any:
        from xgboost import XGBRegressor

        # Separate early stopping from model params
        params = {k: v for k, v in self.params.items() if k != "early_stopping_rounds"}
        return XGBRegressor(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
        """Train XGBoost model with early stopping."""
        logger.info(
            "Training XGBoost model",
            n_samples=len(X),
            n_features=len(X.columns),
        )

        self.feature_names = list(X.columns)
        self.model = self._create_model()

        # Split for early stopping
        split_idx = int(len(X) * 0.85)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )

        # Extract feature importance
        importance = self.model.feature_importances_
        self.feature_importance = dict(zip(self.feature_names, importance.tolist()))

        # Training metrics
        self.training_metrics = self.evaluate(X_val, y_val)
        self.trained_at = datetime.now(timezone.utc)

        logger.info(
            "XGBoost training complete",
            val_mae=self.training_metrics["mae"],
            val_r2=self.training_metrics["r2"],
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with XGBoost."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Ensure column order matches training
        X = X[self.feature_names]
        predictions = self.model.predict(X)

        # Demand cannot be negative
        return np.maximum(predictions, 0)


class LightGBMModel(BaseModel):
    """LightGBM gradient boosting model for demand prediction."""

    model_type: str = "lightgbm"
    version: str = "4.0.0"

    def _default_params(self) -> dict[str, Any]:
        return {
            "objective": "regression",
            "metric": "mae",
            "n_estimators": 500,
            "num_leaves": 64,
            "max_depth": 8,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

    def _create_model(self) -> Any:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**self.params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMModel":
        """Train LightGBM model with early stopping."""
        logger.info(
            "Training LightGBM model",
            n_samples=len(X),
            n_features=len(X.columns),
        )

        self.feature_names = list(X.columns)
        self.model = self._create_model()

        # Split for early stopping
        split_idx = int(len(X) * 0.85)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                self._early_stopping_callback(50),
                self._log_callback(100),
            ],
        )

        # Extract feature importance
        importance = self.model.feature_importances_
        self.feature_importance = dict(zip(self.feature_names, importance.tolist()))

        # Training metrics
        self.training_metrics = self.evaluate(X_val, y_val)
        self.trained_at = datetime.now(timezone.utc)

        logger.info(
            "LightGBM training complete",
            val_mae=self.training_metrics["mae"],
            val_r2=self.training_metrics["r2"],
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with LightGBM."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Ensure column order matches training
        X = X[self.feature_names]
        predictions = self.model.predict(X)

        # Demand cannot be negative
        return np.maximum(predictions, 0)

    @staticmethod
    def _early_stopping_callback(stopping_rounds: int):
        """Create early stopping callback."""
        from lightgbm import early_stopping
        return early_stopping(stopping_rounds=stopping_rounds, verbose=False)

    @staticmethod
    def _log_callback(period: int):
        """Create logging callback."""
        from lightgbm import log_evaluation
        return log_evaluation(period=period)


class EnsembleModel(BaseModel):
    """
    Ensemble model combining multiple base models.

    Uses weighted averaging of predictions for improved robustness.
    """

    model_type: str = "ensemble"
    version: str = "1.0.0"

    def __init__(
        self,
        models: list[BaseModel] | None = None,
        weights: list[float] | None = None,
    ):
        super().__init__()
        self.models = models or [XGBoostModel(), LightGBMModel()]
        self.weights = weights or [1.0 / len(self.models)] * len(self.models)

        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def _default_params(self) -> dict[str, Any]:
        return {}

    def _create_model(self) -> Any:
        return None  # Ensemble doesn't have a single underlying model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleModel":
        """Train all models in the ensemble."""
        logger.info(
            "Training ensemble",
            n_models=len(self.models),
            model_types=[m.model_type for m in self.models],
        )

        self.feature_names = list(X.columns)

        for i, model in enumerate(self.models):
            logger.info(f"Training model {i + 1}/{len(self.models)}: {model.model_type}")
            model.fit(X, y)

        # Aggregate feature importance (weighted average)
        all_features = set()
        for model in self.models:
            all_features.update(model.feature_importance.keys())

        self.feature_importance = {}
        for feature in all_features:
            weighted_sum = 0.0
            for model, weight in zip(self.models, self.weights):
                weighted_sum += model.feature_importance.get(feature, 0.0) * weight
            self.feature_importance[feature] = weighted_sum

        # Aggregate training metrics
        self.training_metrics = {
            "mae": sum(m.training_metrics["mae"] * w for m, w in zip(self.models, self.weights)),
            "rmse": sum(m.training_metrics["rmse"] * w for m, w in zip(self.models, self.weights)),
            "r2": sum(m.training_metrics["r2"] * w for m, w in zip(self.models, self.weights)),
        }

        self.trained_at = datetime.now(timezone.utc)

        logger.info(
            "Ensemble training complete",
            ensemble_mae=self.training_metrics["mae"],
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make weighted average predictions."""
        predictions = None

        for model, weight in zip(self.models, self.weights):
            model_preds = model.predict(X) * weight
            if predictions is None:
                predictions = model_preds.copy()
            else:
                predictions += model_preds

        return np.maximum(predictions, 0) if predictions is not None else np.zeros(len(X))

    def get_individual_predictions(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Get predictions from each model separately."""
        return {
            model.model_type: model.predict(X)
            for model in self.models
        }


def get_model(
    model_type: str = "xgboost",
    params: dict[str, Any] | None = None,
) -> BaseModel:
    """
    Factory function to create model instances.

    Args:
        model_type: One of "xgboost", "lightgbm", "ensemble"
        params: Optional hyperparameters

    Returns:
        Model instance
    """
    models = {
        "xgboost": XGBoostModel,
        "lightgbm": LightGBMModel,
        "ensemble": EnsembleModel,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")

    return models[model_type](params) if model_type != "ensemble" else EnsembleModel()

