"""
Hungarian Algorithm (Kuhn-Munkres) for Optimal Assignment.

Solves the assignment problem optimally in O(n³) time.
Finds the minimum-cost perfect matching in a bipartite graph.

This is the optimal algorithm for order-courier assignment
when we need to minimize total cost.

Implementation uses scipy.optimize.linear_sum_assignment
which is a highly optimized C implementation.
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
from src.dispatch_engine.algorithms.cost_matrix import (
    CostMatrixBuilder,
    INFINITY_COST,
    haversine_distance,
)

logger = get_logger(__name__)


class HungarianAssigner(BaseAssigner):
    """
    Hungarian (Kuhn-Munkres) optimal assignment algorithm.

    Finds the globally optimal assignment that minimizes total cost.

    Time complexity: O(n³) where n = max(orders, couriers)
    Space complexity: O(n²)

    Best for:
    - Small to medium problems (up to ~1000 orders)
    - When optimality is required
    - Batch processing scenarios
    """

    algorithm_name = "hungarian"

    def __init__(self, pad_matrix: bool = True):
        """
        Initialize Hungarian assigner.

        Args:
            pad_matrix: Pad matrix to square if needed
        """
        self.pad_matrix = pad_matrix
        self.cost_builder = CostMatrixBuilder()

    def solve(self, problem: AssignmentProblem) -> AssignmentSolution:
        """Solve using Hungarian algorithm."""
        start_time = time.perf_counter()

        # Validate
        errors = self.validate_problem(problem)
        if errors:
            return self.create_empty_solution(problem, "; ".join(errors))

        # Build cost matrix
        self.cost_builder = CostMatrixBuilder(objective=problem.objective)
        cost_matrix = self.cost_builder.build_fast(problem)

        # Handle rectangular matrices
        original_shape = cost_matrix.shape
        if self.pad_matrix:
            cost_matrix = self._pad_to_square(cost_matrix)

        # Solve using scipy
        try:
            from scipy.optimize import linear_sum_assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        except ImportError:
            logger.warning("scipy not available, falling back to numpy implementation")
            row_indices, col_indices = self._hungarian_numpy(cost_matrix)

        # Extract valid assignments
        assignments = []
        unassigned_orders = []

        for row_idx, col_idx in zip(row_indices, col_indices):
            # Skip padded rows/columns
            if row_idx >= original_shape[0] or col_idx >= original_shape[1]:
                continue

            cost = cost_matrix[row_idx, col_idx]

            # Skip infeasible assignments
            if cost >= INFINITY_COST:
                unassigned_orders.append(problem.orders[row_idx].id)
                continue

            order = problem.orders[row_idx]
            courier = problem.couriers[col_idx]

            # Get assignment details
            details = self._get_assignment_details(order, courier)

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
                reason="Hungarian: optimal assignment",
            )

            assignments.append(assignment)

        # Find unassigned orders (not in solution)
        assigned_order_ids = {a.order_id for a in assignments}
        for order in problem.orders:
            if order.id not in assigned_order_ids and order.id not in unassigned_orders:
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
            unassigned_reason="Infeasible or no courier available" if unassigned_orders else "",
            algorithm=self.algorithm_name,
            solve_time_ms=solve_time,
            is_optimal=True,  # Hungarian is optimal
        )

        self.log_solution(solution)
        return solution

    def _pad_to_square(self, matrix: np.ndarray) -> np.ndarray:
        """Pad matrix to square with high cost values."""
        rows, cols = matrix.shape

        if rows == cols:
            return matrix

        size = max(rows, cols)
        padded = np.full((size, size), INFINITY_COST)
        padded[:rows, :cols] = matrix

        return padded

    def _hungarian_numpy(self, cost_matrix: np.ndarray) -> tuple:
        """
        Pure numpy implementation of Hungarian algorithm.

        Used as fallback when scipy is not available.
        This is a simplified O(n³) implementation.
        """
        n = cost_matrix.shape[0]

        # Initialize
        u = np.zeros(n)
        v = np.zeros(n)
        p = np.zeros(n, dtype=int)
        way = np.zeros(n, dtype=int)

        for i in range(n):
            p[0] = i
            j0 = 0
            minv = np.full(n, np.inf)
            used = np.zeros(n, dtype=bool)

            while p[j0] != 0:
                used[j0] = True
                i0 = p[j0]
                delta = np.inf
                j1 = 0

                for j in range(1, n):
                    if not used[j]:
                        cur = cost_matrix[i0, j] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j

                for j in range(n):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta

                j0 = j1

            while j0 != 0:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1

        # Extract solution
        row_ind = np.zeros(n, dtype=int)
        col_ind = np.zeros(n, dtype=int)

        for j in range(1, n + 1):
            if j <= n and p[j] != 0:
                row_ind[j-1] = p[j]
                col_ind[j-1] = j - 1

        return row_ind[:n], col_ind[:n]

    def _get_assignment_details(self, order, courier) -> dict:
        """Get assignment details."""
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


class AuctionAssigner(BaseAssigner):
    """
    Auction algorithm for assignment.

    Alternative to Hungarian with better parallelization potential.
    Works by having orders "bid" for couriers iteratively.

    Good for:
    - Large problems where parallelization helps
    - Distributed systems
    - Approximate solutions with speed tradeoff
    """

    algorithm_name = "auction"

    def __init__(self, epsilon: float = 0.01, max_iterations: int = 100):
        """
        Initialize auction assigner.

        Args:
            epsilon: Bid increment (controls convergence)
            max_iterations: Maximum auction rounds
        """
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.cost_builder = CostMatrixBuilder()

    def solve(self, problem: AssignmentProblem) -> AssignmentSolution:
        """Solve using auction algorithm."""
        start_time = time.perf_counter()

        errors = self.validate_problem(problem)
        if errors:
            return self.create_empty_solution(problem, "; ".join(errors))

        # Build cost matrix (negate for maximization)
        self.cost_builder = CostMatrixBuilder(objective=problem.objective)
        benefit_matrix = -self.cost_builder.build_fast(problem)

        n_orders = problem.n_orders
        n_couriers = problem.n_couriers

        # Initialize
        prices = np.zeros(n_couriers)  # Courier prices
        assignment = np.full(n_orders, -1, dtype=int)  # Order -> courier
        courier_to_order = np.full(n_couriers, -1, dtype=int)  # Courier -> order

        # Auction iterations
        for iteration in range(self.max_iterations):
            unassigned = np.where(assignment == -1)[0]

            if len(unassigned) == 0:
                break

            for order_idx in unassigned:
                # Calculate net values (benefit - price)
                net_values = benefit_matrix[order_idx] - prices

                # Find best and second-best couriers
                sorted_indices = np.argsort(net_values)[::-1]
                best_j = sorted_indices[0]

                if net_values[best_j] <= -INFINITY_COST:
                    continue  # No feasible courier

                if len(sorted_indices) > 1:
                    second_best_value = net_values[sorted_indices[1]]
                else:
                    second_best_value = -INFINITY_COST

                # Calculate bid
                bid = benefit_matrix[order_idx, best_j] - second_best_value + self.epsilon

                # Remove previous assignment of this courier
                if courier_to_order[best_j] != -1:
                    prev_order = courier_to_order[best_j]
                    assignment[prev_order] = -1

                # Assign
                assignment[order_idx] = best_j
                courier_to_order[best_j] = order_idx
                prices[best_j] = benefit_matrix[order_idx, best_j] - second_best_value + self.epsilon

        # Build assignments
        assignments = []
        unassigned_orders = []

        for order_idx, courier_idx in enumerate(assignment):
            order = problem.orders[order_idx]

            if courier_idx == -1:
                unassigned_orders.append(order.id)
                continue

            courier = problem.couriers[courier_idx]
            cost = -benefit_matrix[order_idx, courier_idx]

            if cost >= INFINITY_COST:
                unassigned_orders.append(order.id)
                continue

            details = self._get_assignment_details(order, courier)

            assignment_obj = Assignment(
                order_id=order.id,
                courier_id=courier.id,
                pickup_distance_km=details["pickup_distance_km"],
                delivery_distance_km=details["delivery_distance_km"],
                total_distance_km=details["total_distance_km"],
                estimated_pickup_time_min=details["estimated_pickup_time_min"],
                estimated_delivery_time_min=details["estimated_delivery_time_min"],
                cost=cost,
                status=AssignmentStatus.ASSIGNED,
                reason="Auction algorithm",
            )

            assignments.append(assignment_obj)

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
            unassigned_reason="Auction did not converge or infeasible" if unassigned_orders else "",
            algorithm=self.algorithm_name,
            solve_time_ms=solve_time,
            is_optimal=False,  # Auction is approximately optimal
        )

        self.log_solution(solution)
        return solution

    def _get_assignment_details(self, order, courier) -> dict:
        """Get assignment details."""
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

