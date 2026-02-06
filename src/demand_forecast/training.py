"""
Training Pipeline for Demand Forecasting Models.

This module provides end-to-end training pipelines with:
- Data preparation and feature extraction
- Model training and validation
- Experiment tracking with MLflow
- Model registration and versioning

Senior-level implementation following MLOps best practices.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.config import settings
from src.common.database import get_db_context
from src.common.logging import get_logger
from src.demand_forecast.features import BatchFeatureExtractor
from src.demand_forecast.models import BaseModel, get_model

logger = get_logger(__name__)


# Default paths
MODELS_DIR = Path("data/models")
DATASETS_DIR = Path("data/datasets")


class TrainingConfig:
    """Configuration for training pipeline."""

    def __init__(
        self,
        model_type: Literal["xgboost", "lightgbm", "ensemble"] = "lightgbm",
        test_size: float = 0.15,
        cv_folds: int = 5,
        use_mlflow: bool = False,
        experiment_name: str = "demand_forecast",
        run_name: str | None = None,
        model_params: dict[str, Any] | None = None,
        feature_selection: bool = True,
        feature_importance_threshold: float = 0.01,
    ):
        self.model_type = model_type
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.use_mlflow = use_mlflow
        self.experiment_name = experiment_name
        self.run_name = run_name or f"{model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.model_params = model_params
        self.feature_selection = feature_selection
        self.feature_importance_threshold = feature_importance_threshold


class TrainingPipeline:
    """
    End-to-end training pipeline for demand forecasting.

    Workflow:
    1. Extract features from database
    2. Prepare train/test splits
    3. Train model with cross-validation
    4. Evaluate on holdout set
    5. Save model artifacts
    6. (Optional) Track with MLflow
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model: BaseModel | None = None
        self.training_data: pd.DataFrame | None = None
        self.feature_columns: list[str] = []
        self.metrics: dict[str, Any] = {}

    async def run(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        h3_indices: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run complete training pipeline.

        Args:
            start_date: Start of training period (default: 30 days ago)
            end_date: End of training period (default: now)
            h3_indices: Optional specific hexagons to train on

        Returns:
            Dictionary with training results and model path
        """
        # Default date range: last 30 days
        end_date = end_date or datetime.utcnow()
        start_date = start_date or (end_date - timedelta(days=30))

        logger.info(
            "Starting training pipeline",
            model_type=self.config.model_type,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )

        # Initialize MLflow tracking if enabled
        if self.config.use_mlflow:
            self._init_mlflow()

        try:
            # Step 1: Extract features
            logger.info("Step 1: Extracting features...")
            self.training_data = await self._extract_features(start_date, end_date, h3_indices)

            if len(self.training_data) == 0:
                raise ValueError("No training data available for the specified period")

            # Step 2: Prepare data
            logger.info("Step 2: Preparing data...")
            X_train, X_test, y_train, y_test = self._prepare_data()

            # Step 3: Train model
            logger.info("Step 3: Training model...")
            self.model = self._train_model(X_train, y_train)

            # Step 4: Cross-validation
            logger.info("Step 4: Cross-validation...")
            cv_metrics = self.model.cross_validate(X_train, y_train, n_splits=self.config.cv_folds)

            # Step 5: Evaluate on test set
            logger.info("Step 5: Evaluating on test set...")
            test_metrics = self.model.evaluate(X_test, y_test)

            # Step 6: Feature selection (optional)
            if self.config.feature_selection:
                logger.info("Step 6: Feature importance analysis...")
                selected_features = self._analyze_features()
                self.feature_columns = selected_features

            # Step 7: Save model
            logger.info("Step 7: Saving model...")
            model_path = self._save_model()

            # Aggregate results
            self.metrics = {
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "n_features": len(self.feature_columns),
                "cv_metrics": cv_metrics,
                "test_metrics": test_metrics,
                "top_features": self._get_top_features(10),
                "model_path": str(model_path),
            }

            # Log to MLflow if enabled
            if self.config.use_mlflow:
                self._log_to_mlflow()

            logger.info(
                "Training pipeline complete",
                test_mae=test_metrics["mae"],
                test_r2=test_metrics["r2"],
                model_path=str(model_path),
            )

            return self.metrics

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise

    async def _extract_features(
        self,
        start_date: datetime,
        end_date: datetime,
        h3_indices: list[str] | None,
    ) -> pd.DataFrame:
        """Extract features from database."""
        async with get_db_context() as session:
            extractor = BatchFeatureExtractor(session)
            df = await extractor.extract_training_dataset(start_date, end_date, h3_indices)

        return df

    def _prepare_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare train/test splits."""
        df = self.training_data

        # Remove non-feature columns
        non_feature_cols = ["target", "h3_index", "timestamp"]
        self.feature_columns = [c for c in df.columns if c not in non_feature_cols]

        X = df[self.feature_columns]
        y = df["target"]

        # Fill missing values
        X = X.fillna(0)

        # Time-based split (test = most recent data)
        if "timestamp" in df.columns:
            df_sorted = df.sort_values("timestamp")
            split_idx = int(len(df_sorted) * (1 - self.config.test_size))

            train_idx = df_sorted.index[:split_idx]
            test_idx = df_sorted.index[split_idx:]

            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, shuffle=False
            )

        logger.info(
            "Data prepared",
            train_size=len(X_train),
            test_size=len(X_test),
            n_features=len(self.feature_columns),
        )

        return X_train, X_test, y_train, y_test

    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> BaseModel:
        """Train the model."""
        model = get_model(self.config.model_type, self.config.model_params)
        model.fit(X_train, y_train)
        return model

    def _analyze_features(self) -> list[str]:
        """Analyze feature importance and select features."""
        if not self.model:
            return self.feature_columns

        importance = self.model.feature_importance

        # Sort by importance
        sorted_features = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Filter by threshold
        total_importance = sum(importance.values())
        if total_importance > 0:
            selected = [
                name for name, imp in sorted_features
                if imp / total_importance >= self.config.feature_importance_threshold
            ]
        else:
            selected = self.feature_columns

        logger.info(
            f"Feature selection: {len(selected)}/{len(self.feature_columns)} features kept"
        )

        return selected

    def _get_top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Get top N most important features."""
        if not self.model:
            return []

        sorted_features = sorted(
            self.model.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [(name, round(imp, 4)) for name, imp in sorted_features[:n]]

    def _save_model(self) -> Path:
        """Save model to disk."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_name = f"demand_{self.config.model_type}_{timestamp}.joblib"
        model_path = MODELS_DIR / model_name

        self.model.save(model_path)

        # Also save as "latest"
        latest_path = MODELS_DIR / f"demand_{self.config.model_type}_latest.joblib"
        self.model.save(latest_path)

        return model_path

    def _init_mlflow(self) -> None:
        """Initialize MLflow tracking."""
        try:
            import mlflow

            mlflow.set_experiment(self.config.experiment_name)
            mlflow.start_run(run_name=self.config.run_name)

            # Log config
            mlflow.log_params({
                "model_type": self.config.model_type,
                "test_size": self.config.test_size,
                "cv_folds": self.config.cv_folds,
            })

        except ImportError:
            logger.warning("MLflow not installed, skipping tracking")
            self.config.use_mlflow = False

    def _log_to_mlflow(self) -> None:
        """Log metrics and artifacts to MLflow."""
        try:
            import mlflow

            # Log test metrics
            for name, value in self.metrics["test_metrics"].items():
                mlflow.log_metric(f"test_{name}", value)

            # Log CV metrics
            for name, value in self.metrics["cv_metrics"]["mean"].items():
                mlflow.log_metric(f"cv_mean_{name}", value)

            # Log model
            mlflow.log_artifact(self.metrics["model_path"])

            mlflow.end_run()

        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")


class HyperparameterTuner:
    """
    Hyperparameter tuning for demand forecasting models.

    Uses Optuna for Bayesian optimization (optional).
    Falls back to grid search if Optuna not available.
    """

    def __init__(
        self,
        model_type: Literal["xgboost", "lightgbm"] = "lightgbm",
        n_trials: int = 50,
        cv_folds: int = 3,
    ):
        self.model_type = model_type
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.best_params: dict[str, Any] = {}
        self.best_score: float = float("inf")

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict[str, Any]:
        """
        Run hyperparameter tuning.

        Returns best parameters found.
        """
        try:
            return self._tune_with_optuna(X, y)
        except ImportError:
            logger.warning("Optuna not installed, using default params")
            return self._get_default_params()

    def _tune_with_optuna(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """Tune hyperparameters using Optuna."""
        import optuna
        from sklearn.model_selection import cross_val_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            if self.model_type == "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                }
            else:  # lightgbm
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                }

            model = get_model(self.model_type, params)

            # Use cross-validation score
            cv_results = model.cross_validate(X, y, n_splits=self.cv_folds)

            return cv_results["mean"]["mae"]

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        self.best_params = study.best_params
        self.best_score = study.best_value

        logger.info(
            "Hyperparameter tuning complete",
            best_mae=self.best_score,
            best_params=self.best_params,
        )

        return self.best_params

    def _get_default_params(self) -> dict[str, Any]:
        """Return default parameters."""
        model = get_model(self.model_type)
        return model.params


async def train_demand_model(
    model_type: str = "lightgbm",
    days_back: int = 30,
    use_mlflow: bool = False,
) -> dict[str, Any]:
    """
    Convenience function to train a demand model.

    Args:
        model_type: Model type ("xgboost", "lightgbm", "ensemble")
        days_back: Number of days of historical data to use
        use_mlflow: Whether to track with MLflow

    Returns:
        Training results dictionary
    """
    config = TrainingConfig(
        model_type=model_type,  # type: ignore
        use_mlflow=use_mlflow,
    )

    pipeline = TrainingPipeline(config)

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)

    return await pipeline.run(start_date=start_date, end_date=end_date)


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train demand forecasting model")
    parser.add_argument("--model", type=str, default="lightgbm", choices=["xgboost", "lightgbm", "ensemble"])
    parser.add_argument("--days", type=int, default=30, help="Days of historical data")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking")

    args = parser.parse_args()

    result = asyncio.run(train_demand_model(
        model_type=args.model,
        days_back=args.days,
        use_mlflow=args.mlflow,
    ))

    print(f"\nTraining complete!")
    print(f"Model saved to: {result['model_path']}")
    print(f"Test MAE: {result['test_metrics']['mae']:.3f}")
    print(f"Test R²: {result['test_metrics']['r2']:.3f}")

