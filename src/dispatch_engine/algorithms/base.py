"""
Base Assignment Algorithm Interface and Core Types.

Defines abstract base class and common types for all assignment algorithms.
Follows Strategy Pattern for flexibility and testability.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import time

from src.common.logging import get_logger

logger = get_logger(__name__)


class OptimizationObjective(str, Enum):
    """Optimization objectives for assignment."""
    MINIMIZE_DISTANCE = "minimize_distance"
    MINIMIZE_TIME = "minimize_time"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_FAIRNESS = "maximize_fairness"
    BALANCED = "balanced"  # Multi-objective


class AssignmentStatus(str, Enum):
    """Status of an assignment."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


@dataclass
class OrderInfo:
    """Order information for assignment."""
    id: str
    restaurant_lat: float
    restaurant_lng: float
    customer_lat: float
    customer_lng: float
    h3_index: str
    created_at: datetime
    priority: int = 0  # Higher = more urgent
    order_value: float = 0.0
    estimated_prep_time_min: int = 15

    # Constraints
    max_delivery_time_min: Optional[int] = None
    required_vehicle_type: Optional[str] = None

    @property
    def age_seconds(self) -> float:
        """Order age in seconds."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()

    @property
    def is_urgent(self) -> bool:
        """Check if order is urgent (waiting too long)."""
        return self.age_seconds > 600  # > 10 minutes


@dataclass
class CourierInfo:
    """Courier information for assignment."""
    id: str
    lat: float
    lng: float
    h3_index: str

    # Capabilities
    vehicle_type: str = "bike"  # bike, scooter, car
    max_distance_km: float = 10.0
    avg_speed_kmh: float = 15.0

    # Current state
    current_load: int = 0  # Current number of orders
    max_load: int = 2  # Max simultaneous orders

    # Performance metrics
    rating: float = 4.5
    completion_rate: float = 0.95
    avg_delivery_time_min: float = 25.0

    # Fairness tracking
    orders_today: int = 0
    earnings_today: float = 0.0

    @property
    def has_capacity(self) -> bool:
        """Check if courier can take more orders."""
        return self.current_load < self.max_load

    @property
    def utilization(self) -> float:
        """Current utilization rate."""
        return self.current_load / self.max_load if self.max_load > 0 else 0.0


@dataclass
class AssignmentConstraints:
    """Constraints for assignment optimization."""
    # Distance constraints
    max_pickup_distance_km: float = 5.0
    max_total_distance_km: float = 15.0

    # Time constraints
    max_pickup_time_min: int = 15
    max_delivery_time_min: int = 45

    # Fairness constraints
    max_orders_per_courier: int = 20  # Per day
    min_earnings_per_order: float = 50.0

    # Vehicle matching
    enforce_vehicle_matching: bool = True

    # Load balancing
    prefer_underutilized_couriers: bool = True

    # Quality thresholds
    min_courier_rating: float = 3.5
    min_completion_rate: float = 0.8


@dataclass
class Assignment:
    """A single order-to-courier assignment."""
    order_id: str
    courier_id: str

    # Estimated metrics
    pickup_distance_km: float
    delivery_distance_km: float
    total_distance_km: float
    estimated_pickup_time_min: float
    estimated_delivery_time_min: float

    # Cost components
    cost: float = 0.0
    courier_payout: float = 0.0

    # Assignment metadata
    status: AssignmentStatus = AssignmentStatus.PENDING
    assigned_at: Optional[datetime] = None
    reason: str = ""

    @property
    def total_time_min(self) -> float:
        """Total estimated time for this assignment."""
        return self.estimated_pickup_time_min + self.estimated_delivery_time_min


@dataclass
class AssignmentProblem:
    """
    Assignment problem definition.

    Contains all inputs needed to solve the assignment problem.
    """
    orders: list[OrderInfo]
    couriers: list[CourierInfo]
    constraints: AssignmentConstraints = field(default_factory=AssignmentConstraints)
    objective: OptimizationObjective = OptimizationObjective.BALANCED

    # Problem metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    timeout_seconds: float = 5.0

    @property
    def n_orders(self) -> int:
        return len(self.orders)

    @property
    def n_couriers(self) -> int:
        return len(self.couriers)

    @property
    def is_feasible(self) -> bool:
        """Check if problem has any feasible solution."""
        return self.n_orders > 0 and self.n_couriers > 0


@dataclass
class AssignmentSolution:
    """
    Solution to an assignment problem.

    Contains assignments and solution metadata.
    """
    assignments: list[Assignment]

    # Problem info
    n_orders: int = 0
    n_couriers: int = 0

    # Solution quality
    total_cost: float = 0.0
    total_distance_km: float = 0.0
    avg_delivery_time_min: float = 0.0

    # Unassigned
    unassigned_orders: list[str] = field(default_factory=list)
    unassigned_reason: str = ""

    # Solver metadata
    algorithm: str = "unknown"
    solve_time_ms: float = 0.0
    is_optimal: bool = False

    # Timestamp
    solved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def n_assigned(self) -> int:
        return len(self.assignments)

    @property
    def n_unassigned(self) -> int:
        return len(self.unassigned_orders)

    @property
    def assignment_rate(self) -> float:
        """Percentage of orders successfully assigned."""
        if self.n_orders == 0:
            return 0.0
        return self.n_assigned / self.n_orders


class BaseAssigner(ABC):
    """
    Abstract base class for assignment algorithms.

    All assignment algorithms must implement this interface.
    """

    algorithm_name: str = "base"

    @abstractmethod
    def solve(self, problem: AssignmentProblem) -> AssignmentSolution:
        """
        Solve the assignment problem.

        Args:
            problem: Assignment problem definition

        Returns:
            AssignmentSolution with assignments and metadata
        """
        pass

    def validate_problem(self, problem: AssignmentProblem) -> list[str]:
        """
        Validate problem inputs.

        Returns list of validation errors (empty if valid).
        """
        errors = []

        if problem.n_orders == 0:
            errors.append("No orders to assign")

        if problem.n_couriers == 0:
            errors.append("No couriers available")

        # Check for duplicate IDs
        order_ids = [o.id for o in problem.orders]
        if len(order_ids) != len(set(order_ids)):
            errors.append("Duplicate order IDs found")

        courier_ids = [c.id for c in problem.couriers]
        if len(courier_ids) != len(set(courier_ids)):
            errors.append("Duplicate courier IDs found")

        return errors

    def create_empty_solution(
        self,
        problem: AssignmentProblem,
        reason: str = "",
    ) -> AssignmentSolution:
        """Create an empty solution (no assignments)."""
        return AssignmentSolution(
            assignments=[],
            n_orders=problem.n_orders,
            n_couriers=problem.n_couriers,
            unassigned_orders=[o.id for o in problem.orders],
            unassigned_reason=reason,
            algorithm=self.algorithm_name,
        )

    def log_solution(self, solution: AssignmentSolution):
        """Log solution summary."""
        logger.info(
            f"Assignment solution",
            algorithm=solution.algorithm,
            assigned=solution.n_assigned,
            unassigned=solution.n_unassigned,
            total_distance=round(solution.total_distance_km, 2),
            solve_time_ms=round(solution.solve_time_ms, 2),
        )

