"""
Base Pricing Strategy Interface and Core Types.

Defines the abstract base class and common types for all pricing strategies.
Follows Strategy Pattern for extensibility and A/B testing support.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from src.common.logging import get_logger

logger = get_logger(__name__)


class StrategyType(str, Enum):
    """Available pricing strategy types."""
    RULE_BASED = "rule_based"
    ML_BASED = "ml_based"
    HYBRID = "hybrid"
    TIME_DECAY = "time_decay"
    COMPETITOR_AWARE = "competitor_aware"


class DemandLevel(str, Enum):
    """Demand level categories."""
    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    EXTREME = "extreme"


class SupplyLevel(str, Enum):
    """Supply level categories."""
    CRITICAL = "critical"
    LOW = "low"
    NORMAL = "normal"
    ADEQUATE = "adequate"
    SURPLUS = "surplus"


@dataclass
class MarketState:
    """
    Current market state for a geographic area.

    Captures supply/demand balance and historical context.
    """
    h3_index: str
    timestamp: datetime

    # Current metrics
    pending_orders: int = 0
    available_couriers: int = 0
    busy_couriers: int = 0

    # Historical context (for trend detection)
    avg_orders_this_hour: float = 0.0
    avg_orders_same_hour_last_week: float = 0.0

    # Predicted metrics (from demand forecast)
    predicted_demand_1h: float = 0.0
    predicted_demand_confidence: float = 0.0

    # Weather factor (external)
    weather_multiplier: float = 1.0

    # Computed properties
    @property
    def demand_supply_ratio(self) -> float:
        """Calculate demand/supply ratio."""
        if self.available_couriers == 0:
            return float('inf') if self.pending_orders > 0 else 0.0
        return self.pending_orders / self.available_couriers

    @property
    def total_couriers(self) -> int:
        """Total couriers in area."""
        return self.available_couriers + self.busy_couriers

    @property
    def courier_utilization(self) -> float:
        """Courier utilization rate (0-1)."""
        if self.total_couriers == 0:
            return 0.0
        return self.busy_couriers / self.total_couriers

    @property
    def demand_trend(self) -> float:
        """Demand trend vs historical average (-1 to +inf)."""
        if self.avg_orders_same_hour_last_week == 0:
            return 0.0
        return (self.pending_orders - self.avg_orders_same_hour_last_week) / self.avg_orders_same_hour_last_week


@dataclass
class PricingContext:
    """
    Context for pricing calculation.

    Contains all information needed to make a pricing decision.
    """
    market_state: MarketState
    base_price: float

    # Order context (optional)
    order_id: Optional[str] = None
    order_created_at: Optional[datetime] = None
    order_value: float = 0.0

    # Customer context (for personalization)
    customer_id: Optional[str] = None
    customer_order_count: int = 0
    customer_avg_order_value: float = 0.0
    customer_is_premium: bool = False

    # Restaurant context
    restaurant_id: Optional[str] = None
    restaurant_prep_time_min: int = 20
    restaurant_rating: float = 4.5

    # A/B test context
    experiment_id: Optional[str] = None
    variant: Optional[str] = None

    # Constraints
    max_surge_override: Optional[float] = None
    min_surge_override: Optional[float] = None

    @property
    def order_age_seconds(self) -> Optional[float]:
        """Age of order in seconds."""
        if self.order_created_at is None:
            return None
        return (datetime.now(timezone.utc) - self.order_created_at).total_seconds()


@dataclass
class PricingResult:
    """
    Result of pricing calculation.

    Contains the calculated surge coefficient and metadata.
    """
    # Core result
    surge_coefficient: float
    final_price: float

    # Breakdown
    base_surge: float  # Before adjustments
    time_modifier: float = 1.0
    weather_modifier: float = 1.0
    loyalty_discount: float = 0.0

    # Classification
    demand_level: DemandLevel = DemandLevel.NORMAL
    supply_level: SupplyLevel = SupplyLevel.NORMAL

    # Explanation
    strategy_used: StrategyType = StrategyType.RULE_BASED
    reason: str = ""
    factors: dict[str, float] = field(default_factory=dict)

    # Confidence (for ML strategies)
    confidence: float = 1.0

    # Fairness metrics
    is_capped: bool = False
    uncapped_surge: float = 0.0

    # Timing
    calculated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    calculation_time_ms: float = 0.0


class BasePricingStrategy(ABC):
    """
    Abstract base class for pricing strategies.

    All pricing strategies must implement this interface.
    Follows the Strategy Pattern for flexibility and testability.
    """

    # Default surge limits (can be overridden per strategy)
    MIN_SURGE: float = 1.0
    MAX_SURGE: float = 3.0

    # Strategy identification
    strategy_type: StrategyType = StrategyType.RULE_BASED
    version: str = "1.0.0"

    @abstractmethod
    async def calculate(self, context: PricingContext) -> PricingResult:
        """
        Calculate surge coefficient for given context.

        Args:
            context: Pricing context with market state and order info

        Returns:
            PricingResult with surge coefficient and metadata
        """
        pass

    def apply_constraints(
        self,
        surge: float,
        context: PricingContext,
    ) -> tuple[float, bool]:
        """
        Apply surge constraints and return capped value.

        Args:
            surge: Calculated surge coefficient
            context: Pricing context (may have overrides)

        Returns:
            Tuple of (capped_surge, was_capped)
        """
        min_surge = context.min_surge_override or self.MIN_SURGE
        max_surge = context.max_surge_override or self.MAX_SURGE

        capped_surge = max(min_surge, min(max_surge, surge))
        was_capped = capped_surge != surge

        return capped_surge, was_capped

    def classify_demand(self, pending_orders: int) -> DemandLevel:
        """Classify demand level based on pending orders."""
        if pending_orders <= 2:
            return DemandLevel.LOW
        elif pending_orders <= 5:
            return DemandLevel.NORMAL
        elif pending_orders <= 10:
            return DemandLevel.ELEVATED
        elif pending_orders <= 20:
            return DemandLevel.HIGH
        else:
            return DemandLevel.EXTREME

    def classify_supply(self, available_couriers: int) -> SupplyLevel:
        """Classify supply level based on available couriers."""
        if available_couriers == 0:
            return SupplyLevel.CRITICAL
        elif available_couriers <= 2:
            return SupplyLevel.LOW
        elif available_couriers <= 5:
            return SupplyLevel.NORMAL
        elif available_couriers <= 10:
            return SupplyLevel.ADEQUATE
        else:
            return SupplyLevel.SURPLUS

    def calculate_loyalty_discount(self, context: PricingContext) -> float:
        """
        Calculate loyalty discount for returning customers.

        Premium customers and frequent orderers get discounts.
        """
        if context.customer_is_premium:
            return 0.10  # 10% discount for premium

        if context.customer_order_count >= 50:
            return 0.05  # 5% for 50+ orders
        elif context.customer_order_count >= 20:
            return 0.03  # 3% for 20+ orders
        elif context.customer_order_count >= 10:
            return 0.02  # 2% for 10+ orders

        return 0.0

