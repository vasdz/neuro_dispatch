"""
Greedy Assignment Algorithm.

Fast O(n*m) algorithm that assigns each order to the best available courier.
Not globally optimal but very fast and works well for small problems.

Advantages:
- Simple and fast
- Easy to understand and debug
- Good enough for real-time scenarios

Disadvantages:
- Not globally optimal
- Order of processing matters
"""

import time
from typing import Optional
import numpy as np

from src.common.logging import get_logger
from src.dispatch_engine.algorithms.base import (
    BaseAssigner,
    AssignmentProblem,
    AssignmentSolution,
    Assignment,
    AssignmentStatus,
)
from src.dispatch_engine.algorithms.cost_matrix import CostMatrixBuilder, INFINITY_COST

logger = get_logger(__name__)


class GreedyAssigner(BaseAssigner):
    """
    Greedy assignment algorithm.

    Strategy:
    1. Build cost matrix
    2. Sort orders by priority (urgent first)
    3. For each order, assign to lowest-cost available courier
    4. Mark courier as used (if single-order capacity)
    """

    algorithm_name = "greedy"

    def __init__(self, allow_multi_order: bool = False):
        """
        Initialize greedy assigner.

        Args:
            allow_multi_order: Allow assigning multiple orders to same courier
        """
        self.allow_multi_order = allow_multi_order
        self.cost_builder = CostMatrixBuilder()

    def solve(self, problem: AssignmentProblem) -> AssignmentSolution:
        """Solve using greedy algorithm."""
        start_time = time.perf_counter()

        # Validate
        errors = self.validate_problem(problem)
        if errors:
            return self.create_empty_solution(problem, "; ".join(errors))

        # Build cost matrix
        self.cost_builder = CostMatrixBuilder(objective=problem.objective)
        cost_matrix = self.cost_builder.build_fast(problem)

        # Sort orders by urgency (age) descending
        order_indices = sorted(
            range(problem.n_orders),
            key=lambda i: problem.orders[i].age_seconds,
            reverse=True,
        )

        assignments = []
        assigned_couriers = set()
        unassigned_orders = []

        for order_idx in order_indices:
            order = problem.orders[order_idx]

            # Find best available courier
            best_courier_idx = self._find_best_courier(
                cost_matrix[order_idx],
                problem.couriers,
                assigned_couriers,
            )

            if best_courier_idx is not None:
                courier = problem.couriers[best_courier_idx]
                cost = cost_matrix[order_idx, best_courier_idx]

                # Get assignment details
                details = self._get_assignment_details(order, courier, problem)

                assignment = Assignment(
                    order_id=order.id,
                    courier_id=courier.id,
                    pickup_distance_km=details["pickup_distance_km"],
                    delivery_distance_km=details["delivery_distance_km"],
                    total_distance_km=details["total_distance_km"],
                    estimated_pickup_time_min=details["estimated_pickup_time_min"],
                    estimated_delivery_time_min=details["estimated_delivery_time_min"],
                    cost=cost,
                    status=AssignmentStatus.ASSIGNED,
                    reason="Greedy: lowest cost available",
                )

                assignments.append(assignment)

                if not self.allow_multi_order:
                    assigned_couriers.add(best_courier_idx)
            else:
                unassigned_orders.append(order.id)

        # Build solution
        solve_time = (time.perf_counter() - start_time) * 1000

        solution = AssignmentSolution(
            assignments=assignments,
            n_orders=problem.n_orders,
            n_couriers=problem.n_couriers,
            total_cost=sum(a.cost for a in assignments),
            total_distance_km=sum(a.total_distance_km for a in assignments),
            avg_delivery_time_min=(
                sum(a.estimated_delivery_time_min for a in assignments) / len(assignments)
                if assignments else 0.0
            ),
            unassigned_orders=unassigned_orders,
            unassigned_reason="No feasible courier found" if unassigned_orders else "",
            algorithm=self.algorithm_name,
            solve_time_ms=solve_time,
            is_optimal=False,  # Greedy is not guaranteed optimal
        )

        self.log_solution(solution)
        return solution

    def _find_best_courier(
        self,
        costs: np.ndarray,
        couriers: list,
        assigned_couriers: set,
    ) -> Optional[int]:
        """Find best available courier for an order."""
        best_idx = None
        best_cost = INFINITY_COST

        for j, cost in enumerate(costs):
            if j in assigned_couriers:
                continue

            if cost < best_cost and cost < INFINITY_COST:
                best_cost = cost
                best_idx = j

        return best_idx

    def _get_assignment_details(self, order, courier, problem) -> dict:
        """Get assignment details for logging."""
        from src.dispatch_engine.algorithms.cost_matrix import haversine_distance

        pickup_distance = haversine_distance(
            courier.lat, courier.lng,
            order.restaurant_lat, order.restaurant_lng,
        )
        delivery_distance = haversine_distance(
            order.restaurant_lat, order.restaurant_lng,
            order.customer_lat, order.customer_lng,
        )

        return self.cost_builder.get_assignment_details(
            order, courier, pickup_distance, delivery_distance
        )


class PriorityGreedyAssigner(GreedyAssigner):
    """
    Greedy assigner with configurable priority function.

    Allows custom prioritization of orders beyond just age.
    """

    algorithm_name = "priority_greedy"

    def __init__(
        self,
        priority_weights: Optional[dict] = None,
        allow_multi_order: bool = False,
    ):
        super().__init__(allow_multi_order)

        # Priority weights for different factors
        self.priority_weights = priority_weights or {
            "age": 1.0,        # Order age (older = higher priority)
            "value": 0.3,      # Order value (higher = higher priority)
            "urgency": 0.5,    # Explicit priority flag
        }

    def _calculate_order_priority(self, order) -> float:
        """Calculate composite priority score for order."""
        priority = 0.0

        # Age factor (normalized to 0-1 for 0-30 min)
        age_factor = min(order.age_seconds / 1800, 1.0)
        priority += age_factor * self.priority_weights["age"]

        # Value factor (normalized to 0-1 for 0-2000 order value)
        value_factor = min(order.order_value / 2000, 1.0)
        priority += value_factor * self.priority_weights["value"]

        # Urgency flag
        if order.is_urgent:
            priority += self.priority_weights["urgency"]

        # Explicit priority
        priority += order.priority * 0.1

        return priority

    def solve(self, problem: AssignmentProblem) -> AssignmentSolution:
        """Solve with custom priority ordering."""
        start_time = time.perf_counter()

        errors = self.validate_problem(problem)
        if errors:
            return self.create_empty_solution(problem, "; ".join(errors))

        # Build cost matrix
        self.cost_builder = CostMatrixBuilder(objective=problem.objective)
        cost_matrix = self.cost_builder.build_fast(problem)

        # Sort orders by priority
        order_priorities = [
            (i, self._calculate_order_priority(problem.orders[i]))
            for i in range(problem.n_orders)
        ]
        order_indices = [i for i, _ in sorted(order_priorities, key=lambda x: x[1], reverse=True)]

        assignments = []
        assigned_couriers = set()
        unassigned_orders = []

        for order_idx in order_indices:
            order = problem.orders[order_idx]

            best_courier_idx = self._find_best_courier(
                cost_matrix[order_idx],
                problem.couriers,
                assigned_couriers,
            )

            if best_courier_idx is not None:
                courier = problem.couriers[best_courier_idx]
                cost = cost_matrix[order_idx, best_courier_idx]
                details = self._get_assignment_details(order, courier, problem)

                assignment = Assignment(
                    order_id=order.id,
                    courier_id=courier.id,
                    pickup_distance_km=details["pickup_distance_km"],
                    delivery_distance_km=details["delivery_distance_km"],
                    total_distance_km=details["total_distance_km"],
                    estimated_pickup_time_min=details["estimated_pickup_time_min"],
                    estimated_delivery_time_min=details["estimated_delivery_time_min"],
                    cost=cost,
                    status=AssignmentStatus.ASSIGNED,
                    reason=f"Priority greedy (priority: {self._calculate_order_priority(order):.2f})",
                )

                assignments.append(assignment)

                if not self.allow_multi_order:
                    assigned_couriers.add(best_courier_idx)
            else:
                unassigned_orders.append(order.id)

        solve_time = (time.perf_counter() - start_time) * 1000

        solution = AssignmentSolution(
            assignments=assignments,
            n_orders=problem.n_orders,
            n_couriers=problem.n_couriers,
            total_cost=sum(a.cost for a in assignments),
            total_distance_km=sum(a.total_distance_km for a in assignments),
            avg_delivery_time_min=(
                sum(a.estimated_delivery_time_min for a in assignments) / len(assignments)
                if assignments else 0.0
            ),
            unassigned_orders=unassigned_orders,
            unassigned_reason="No feasible courier found" if unassigned_orders else "",
            algorithm=self.algorithm_name,
            solve_time_ms=solve_time,
            is_optimal=False,
        )

        self.log_solution(solution)
        return solution

