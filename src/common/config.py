"""
Application configuration using Pydantic Settings.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/neuro_dispatch"
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_ttl_seconds: int = 300

    # Security
    secret_key: str = "super-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7

    # H3 Geo Configuration
    default_city_lat: float = 55.7558  # Moscow
    default_city_lng: float = 37.6173
    h3_resolution: int = 8  # ~460m edge length

    # Feature Flags
    enable_surge_pricing: bool = True
    enable_ml_predictions: bool = False

    # Dispatch settings
    max_courier_distance_km: float = 5.0
    assignment_interval_seconds: int = 30

    # ============================================
    # MLOps Configuration
    # ============================================

    # MLflow
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment_name: str = "neuro_dispatch"
    mlflow_enable: bool = False

    # Model Registry
    model_registry_path: str = "data/models"
    model_auto_reload: bool = True
    model_reload_interval_seconds: int = 60

    # Feature Store
    feature_store_online_ttl_seconds: int = 3600
    feature_store_cache_size: int = 10000

    # Training
    training_data_days: int = 30
    training_cv_folds: int = 5
    training_test_size: float = 0.15

    # Monitoring
    metrics_enable: bool = True
    metrics_prefix: str = "neuro_dispatch"
    health_check_timeout_seconds: int = 5

    # Alerting thresholds
    alert_prediction_error_threshold: float = 5.0
    alert_latency_p95_threshold_ms: float = 500.0
    alert_error_rate_threshold: float = 0.05

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_staging(self) -> bool:
        return self.app_env == "staging"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()

