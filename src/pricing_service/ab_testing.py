"""
A/B Testing Engine for Pricing Strategies.

Provides infrastructure for running pricing experiments:
- Strategy variant assignment
- Traffic splitting
- Metrics collection
- Statistical analysis

Senior-level implementation following experimentation best practices.
"""

import hashlib
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Optional
import json

from src.common.logging import get_logger
from src.common.config import settings
from src.pricing_service.strategies.base import (
    BasePricingStrategy,
    PricingContext,
    PricingResult,
    StrategyType,
)

logger = get_logger(__name__)


class ExperimentStatus(str, Enum):
    """Experiment lifecycle status."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass
class ExperimentVariant:
    """A variant in an A/B experiment."""
    name: str
    strategy: BasePricingStrategy
    traffic_percentage: float  # 0-100
    description: str = ""

    # Metrics (updated during experiment)
    impressions: int = 0
    conversions: int = 0
    total_revenue: float = 0.0
    total_surge: float = 0.0

    @property
    def conversion_rate(self) -> float:
        if self.impressions == 0:
            return 0.0
        return self.conversions / self.impressions

    @property
    def avg_surge(self) -> float:
        if self.impressions == 0:
            return 0.0
        return self.total_surge / self.impressions

    @property
    def avg_revenue(self) -> float:
        if self.conversions == 0:
            return 0.0
        return self.total_revenue / self.conversions


@dataclass
class Experiment:
    """
    A/B Experiment configuration.

    An experiment compares multiple pricing strategy variants
    to determine which performs best.
    """
    id: str
    name: str
    description: str
    variants: list[ExperimentVariant]

    # Lifecycle
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    # Targeting
    target_hexagons: Optional[set[str]] = None  # None = all hexagons
    target_percentage: float = 100.0  # % of eligible traffic

    # Guardrails
    min_sample_size: int = 1000
    max_duration_days: int = 14
    min_conversion_rate: float = 0.5  # Abort if below this

    def __post_init__(self):
        # Validate traffic percentages sum to 100
        total = sum(v.traffic_percentage for v in self.variants)
        if abs(total - 100.0) > 0.01:
            raise ValueError(f"Variant traffic must sum to 100%, got {total}%")

    @property
    def is_active(self) -> bool:
        return self.status == ExperimentStatus.RUNNING

    @property
    def total_impressions(self) -> int:
        return sum(v.impressions for v in self.variants)

    @property
    def has_sufficient_data(self) -> bool:
        return all(v.impressions >= self.min_sample_size for v in self.variants)


class VariantAssigner:
    """
    Assigns users/requests to experiment variants.

    Uses deterministic hashing for consistent assignment:
    - Same user always gets same variant
    - Assignment is evenly distributed
    """

    @staticmethod
    def assign(
        experiment: Experiment,
        identifier: str,  # customer_id, order_id, etc.
    ) -> Optional[ExperimentVariant]:
        """
        Assign an identifier to a variant.

        Uses consistent hashing for deterministic assignment.
        """
        if not experiment.is_active:
            return None

        # Create hash from experiment + identifier
        hash_input = f"{experiment.id}:{identifier}"
        hash_bytes = hashlib.md5(hash_input.encode()).digest()
        hash_value = int.from_bytes(hash_bytes[:4], 'big')

        # Convert to percentage (0-100)
        percentage = (hash_value % 10000) / 100.0

        # Check if in experiment at all
        if percentage > experiment.target_percentage:
            return None

        # Assign to variant based on traffic split
        cumulative = 0.0
        for variant in experiment.variants:
            cumulative += variant.traffic_percentage
            if percentage < cumulative:
                return variant

        # Fallback to last variant
        return experiment.variants[-1]

    @staticmethod
    def is_in_experiment(
        experiment: Experiment,
        h3_index: str,
    ) -> bool:
        """Check if a hexagon is targeted by the experiment."""
        if experiment.target_hexagons is None:
            return True
        return h3_index in experiment.target_hexagons


class ABTestingEngine:
    """
    Main A/B testing engine for pricing experiments.

    Manages:
    - Experiment lifecycle
    - Variant assignment
    - Metrics collection
    - Winner determination
    """

    def __init__(self):
        self._experiments: dict[str, Experiment] = {}
        self._active_experiment_id: Optional[str] = None

    def create_experiment(
        self,
        name: str,
        variants: list[tuple[str, BasePricingStrategy, float]],
        description: str = "",
        target_hexagons: Optional[set[str]] = None,
    ) -> Experiment:
        """
        Create a new experiment.

        Args:
            name: Experiment name
            variants: List of (name, strategy, traffic_percentage) tuples
            description: Experiment description
            target_hexagons: Optional set of target hexagons

        Returns:
            Created experiment
        """
        experiment_id = f"exp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"

        variant_objects = [
            ExperimentVariant(
                name=name,
                strategy=strategy,
                traffic_percentage=traffic,
            )
            for name, strategy, traffic in variants
        ]

        experiment = Experiment(
            id=experiment_id,
            name=name,
            description=description,
            variants=variant_objects,
            target_hexagons=target_hexagons,
        )

        self._experiments[experiment_id] = experiment

        logger.info(
            f"Created experiment",
            experiment_id=experiment_id,
            name=name,
            variants=[v.name for v in variant_objects],
        )

        return experiment

    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment."""
        if experiment_id not in self._experiments:
            logger.error(f"Experiment not found: {experiment_id}")
            return False

        experiment = self._experiments[experiment_id]

        if experiment.status != ExperimentStatus.DRAFT:
            logger.error(f"Cannot start experiment in {experiment.status} status")
            return False

        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.now(timezone.utc)
        self._active_experiment_id = experiment_id

        logger.info(f"Started experiment", experiment_id=experiment_id)
        return True

    def stop_experiment(
        self,
        experiment_id: str,
        status: ExperimentStatus = ExperimentStatus.COMPLETED,
    ) -> bool:
        """Stop an experiment."""
        if experiment_id not in self._experiments:
            return False

        experiment = self._experiments[experiment_id]
        experiment.status = status
        experiment.ended_at = datetime.now(timezone.utc)

        if self._active_experiment_id == experiment_id:
            self._active_experiment_id = None

        logger.info(
            f"Stopped experiment",
            experiment_id=experiment_id,
            status=status,
        )
        return True

    def get_active_experiment(self) -> Optional[Experiment]:
        """Get currently active experiment."""
        if self._active_experiment_id is None:
            return None
        return self._experiments.get(self._active_experiment_id)

    async def get_strategy_for_context(
        self,
        context: PricingContext,
        default_strategy: BasePricingStrategy,
    ) -> tuple[BasePricingStrategy, Optional[str], Optional[str]]:
        """
        Get pricing strategy for a context, considering active experiments.

        Returns:
            Tuple of (strategy, experiment_id, variant_name)
        """
        experiment = self.get_active_experiment()

        if experiment is None:
            return default_strategy, None, None

        # Check if hexagon is targeted
        h3_index = context.market_state.h3_index
        if not VariantAssigner.is_in_experiment(experiment, h3_index):
            return default_strategy, None, None

        # Get assignment identifier
        identifier = context.customer_id or context.order_id or h3_index
        if identifier is None:
            identifier = h3_index

        # Assign to variant
        variant = VariantAssigner.assign(experiment, identifier)

        if variant is None:
            return default_strategy, None, None

        # Record impression
        variant.impressions += 1

        return variant.strategy, experiment.id, variant.name

    def record_conversion(
        self,
        experiment_id: str,
        variant_name: str,
        revenue: float,
        surge: float,
    ):
        """Record a conversion for experiment analysis."""
        if experiment_id not in self._experiments:
            return

        experiment = self._experiments[experiment_id]

        for variant in experiment.variants:
            if variant.name == variant_name:
                variant.conversions += 1
                variant.total_revenue += revenue
                variant.total_surge += surge
                break

    def get_experiment_results(
        self,
        experiment_id: str,
    ) -> Optional[dict[str, Any]]:
        """Get experiment results and analysis."""
        if experiment_id not in self._experiments:
            return None

        experiment = self._experiments[experiment_id]

        results = {
            "experiment_id": experiment.id,
            "name": experiment.name,
            "status": experiment.status.value,
            "started_at": experiment.started_at.isoformat() if experiment.started_at else None,
            "total_impressions": experiment.total_impressions,
            "has_sufficient_data": experiment.has_sufficient_data,
            "variants": [],
        }

        for variant in experiment.variants:
            results["variants"].append({
                "name": variant.name,
                "traffic_percentage": variant.traffic_percentage,
                "impressions": variant.impressions,
                "conversions": variant.conversions,
                "conversion_rate": round(variant.conversion_rate, 4),
                "avg_surge": round(variant.avg_surge, 2),
                "avg_revenue": round(variant.avg_revenue, 2),
            })

        # Determine winner (if sufficient data)
        if experiment.has_sufficient_data and len(experiment.variants) >= 2:
            winner = self._determine_winner(experiment)
            results["winner"] = winner

        return results

    def _determine_winner(self, experiment: Experiment) -> dict[str, Any]:
        """
        Determine experiment winner using statistical analysis.

        Uses conversion rate as primary metric.
        """
        variants = experiment.variants

        # Sort by conversion rate
        sorted_variants = sorted(
            variants,
            key=lambda v: v.conversion_rate,
            reverse=True,
        )

        best = sorted_variants[0]
        second = sorted_variants[1] if len(sorted_variants) > 1 else None

        # Calculate lift
        if second and second.conversion_rate > 0:
            lift = (best.conversion_rate - second.conversion_rate) / second.conversion_rate
        else:
            lift = 0.0

        # Simple significance check (would use proper statistical test in production)
        is_significant = (
            best.impressions >= experiment.min_sample_size and
            lift > 0.05  # 5% lift threshold
        )

        return {
            "variant": best.name,
            "conversion_rate": round(best.conversion_rate, 4),
            "lift_vs_second": round(lift, 4),
            "is_significant": is_significant,
            "recommendation": "Deploy winner" if is_significant else "Continue experiment",
        }

    def list_experiments(self) -> list[dict[str, Any]]:
        """List all experiments."""
        return [
            {
                "id": exp.id,
                "name": exp.name,
                "status": exp.status.value,
                "variants": [v.name for v in exp.variants],
                "total_impressions": exp.total_impressions,
            }
            for exp in self._experiments.values()
        ]


# Global singleton
_ab_testing_engine: Optional[ABTestingEngine] = None


def get_ab_testing_engine() -> ABTestingEngine:
    """Get global A/B testing engine instance."""
    global _ab_testing_engine
    if _ab_testing_engine is None:
        _ab_testing_engine = ABTestingEngine()
    return _ab_testing_engine

