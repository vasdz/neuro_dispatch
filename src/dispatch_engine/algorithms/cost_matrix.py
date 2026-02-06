"""
Cost Matrix Builder.

Builds cost matrices for assignment algorithms considering:
- Distance (pickup + delivery)
- Time (ETA)
- Courier performance
- Fairness factors
- Constraints violations

The cost matrix is the foundation for optimization algorithms.
"""

import math
from typing import Optional
import numpy as np

from src.common.logging import get_logger
from src.dispatch_engine.algorithms.base import (
    AssignmentProblem,
    AssignmentConstraints,
    OrderInfo,
    CourierInfo,
    OptimizationObjective,
)

logger = get_logger(__name__)

# Large cost for infeasible assignments
INFINITY_COST = 1e9


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great circle distance between two points in kilometers.

    Optimized implementation using numpy for batch operations.
    """
    R = 6371.0  # Earth radius in km

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2 +
        math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def haversine_vectorized(
    lats1: np.ndarray,
    lons1: np.ndarray,
    lats2: np.ndarray,
    lons2: np.ndarray,
) -> np.ndarray:
    """
    Vectorized haversine distance calculation.

    Much faster for large matrices.
    """
    R = 6371.0

    lats1_rad = np.radians(lats1)
    lats2_rad = np.radians(lats2)
    delta_lat = np.radians(lats2 - lats1)
    delta_lon = np.radians(lons2 - lons1)

    a = (
        np.sin(delta_lat / 2) ** 2 +
        np.cos(lats1_rad) * np.cos(lats2_rad) * np.sin(delta_lon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


class CostMatrixBuilder:
    """
    Builds cost matrices for assignment optimization.

    The cost matrix C[i][j] represents the cost of assigning
    order i to courier j. Lower cost = better assignment.

    Cost components:
    1. Distance cost: Based on pickup + delivery distance
    2. Time cost: Based on estimated delivery time
    3. Fairness cost: Penalize overworked couriers
    4. Quality bonus: Prefer high-performing couriers
    5. Constraint penalty: Infinite cost for constraint violations
    """

    # Weight parameters for multi-objective optimization
    DEFAULT_WEIGHTS = {
        "distance": 1.0,
        "time": 0.5,
        "fairness": 0.3,
        "quality": 0.2,
        "urgency": 0.4,
    }

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        objective: OptimizationObjective = OptimizationObjective.BALANCED,
    ):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.objective = objective

        # Adjust weights based on objective
        self._adjust_weights_for_objective()

    def _adjust_weights_for_objective(self):
        """Adjust weights based on optimization objective."""
        if self.objective == OptimizationObjective.MINIMIZE_DISTANCE:
            self.weights = {"distance": 1.0, "time": 0.1, "fairness": 0.0, "quality": 0.0, "urgency": 0.0}
        elif self.objective == OptimizationObjective.MINIMIZE_TIME:
            self.weights = {"distance": 0.3, "time": 1.0, "fairness": 0.0, "quality": 0.0, "urgency": 0.2}
        elif self.objective == OptimizationObjective.MAXIMIZE_FAIRNESS:
            self.weights = {"distance": 0.3, "time": 0.2, "fairness": 1.0, "quality": 0.0, "urgency": 0.1}
        # BALANCED uses default weights

    def build(self, problem: AssignmentProblem) -> np.ndarray:
        """
        Build the cost matrix for the assignment problem.

        Args:
            problem: Assignment problem with orders and couriers

        Returns:
            Cost matrix of shape (n_orders, n_couriers)
        """
        n_orders = problem.n_orders
        n_couriers = problem.n_couriers

        if n_orders == 0 or n_couriers == 0:
            return np.array([])

        # Initialize cost matrix
        cost_matrix = np.zeros((n_orders, n_couriers))

        # Build distance matrices first (vectorized for speed)
        pickup_distances = self._build_pickup_distance_matrix(problem)
        delivery_distances = self._build_delivery_distance_matrix(problem)

        # Calculate costs for each order-courier pair
        for i, order in enumerate(problem.orders):
            for j, courier in enumerate(problem.couriers):
                cost = self._calculate_assignment_cost(
                    order=order,
                    courier=courier,
                    pickup_distance=pickup_distances[i, j],
                    delivery_distance=delivery_distances[i],
                    constraints=problem.constraints,
                )
                cost_matrix[i, j] = cost

        return cost_matrix

    def build_fast(self, problem: AssignmentProblem) -> np.ndarray:
        """
        Fast vectorized cost matrix building.

        Uses numpy broadcasting for better performance on large problems.
        """
        n_orders = problem.n_orders
        n_couriers = problem.n_couriers

        if n_orders == 0 or n_couriers == 0:
            return np.array([])

        # Extract coordinates
        order_rest_lats = np.array([o.restaurant_lat for o in problem.orders])
        order_rest_lngs = np.array([o.restaurant_lng for o in problem.orders])
        order_cust_lats = np.array([o.customer_lat for o in problem.orders])
        order_cust_lngs = np.array([o.customer_lng for o in problem.orders])

        courier_lats = np.array([c.lat for c in problem.couriers])
        courier_lngs = np.array([c.lng for c in problem.couriers])
        courier_speeds = np.array([c.avg_speed_kmh for c in problem.couriers])
        courier_orders_today = np.array([c.orders_today for c in problem.couriers])
        courier_ratings = np.array([c.rating for c in problem.couriers])

        # Calculate pickup distances (couriers to restaurants)
        # Shape: (n_orders, n_couriers)
        pickup_distances = np.zeros((n_orders, n_couriers))
        for j in range(n_couriers):
            pickup_distances[:, j] = haversine_vectorized(
                courier_lats[j] * np.ones(n_orders),
                courier_lngs[j] * np.ones(n_orders),
                order_rest_lats,
                order_rest_lngs,
            )

        # Calculate delivery distances (restaurants to customers)
        # Shape: (n_orders,)
        delivery_distances = haversine_vectorized(
            order_rest_lats,
            order_rest_lngs,
            order_cust_lats,
            order_cust_lngs,
        )

        # Total distances
        total_distances = pickup_distances + delivery_distances[:, np.newaxis]

        # Estimated times (distance / speed)
        # Shape: (n_orders, n_couriers)
        speeds_matrix = courier_speeds[np.newaxis, :]  # (1, n_couriers)
        estimated_times = (total_distances / speeds_matrix) * 60  # Convert to minutes

        # Distance cost
        distance_cost = total_distances * self.weights["distance"]

        # Time cost
        time_cost = estimated_times * self.weights["time"]

        # Fairness cost (penalize overworked couriers)
        fairness_cost = (courier_orders_today[np.newaxis, :] / 20.0) * self.weights["fairness"]

        # Quality bonus (prefer higher-rated couriers)
        quality_bonus = ((5.0 - courier_ratings[np.newaxis, :]) / 5.0) * self.weights["quality"]

        # Urgency cost for old orders
        order_ages = np.array([o.age_seconds for o in problem.orders])
        urgency_factor = np.clip(order_ages / 600.0, 0, 2)[:, np.newaxis]  # Normalize to 10 min
        urgency_cost = urgency_factor * self.weights["urgency"]

        # Combine costs
        cost_matrix = distance_cost + time_cost + fairness_cost + quality_bonus + urgency_cost

        # Apply constraint penalties
        cost_matrix = self._apply_constraint_penalties(
            cost_matrix,
            problem,
            pickup_distances,
            total_distances,
            estimated_times,
        )

        return cost_matrix

    def _build_pickup_distance_matrix(self, problem: AssignmentProblem) -> np.ndarray:
        """Build matrix of distances from couriers to order restaurants."""
        n_orders = problem.n_orders
        n_couriers = problem.n_couriers

        distances = np.zeros((n_orders, n_couriers))

        for i, order in enumerate(problem.orders):
            for j, courier in enumerate(problem.couriers):
                distances[i, j] = haversine_distance(
                    courier.lat, courier.lng,
                    order.restaurant_lat, order.restaurant_lng,
                )

        return distances

    def _build_delivery_distance_matrix(self, problem: AssignmentProblem) -> np.ndarray:
        """Build array of distances from restaurants to customers."""
        return np.array([
            haversine_distance(
                o.restaurant_lat, o.restaurant_lng,
                o.customer_lat, o.customer_lng,
            )
            for o in problem.orders
        ])

    def _calculate_assignment_cost(
        self,
        order: OrderInfo,
        courier: CourierInfo,
        pickup_distance: float,
        delivery_distance: float,
        constraints: AssignmentConstraints,
    ) -> float:
        """
        Calculate cost for a single order-courier assignment.

        Returns INFINITY_COST if assignment violates hard constraints.
        """
        # Check hard constraints
        if not self._is_feasible_assignment(order, courier, pickup_distance, delivery_distance, constraints):
            return INFINITY_COST

        total_distance = pickup_distance + delivery_distance

        # Estimate time
        estimated_time = (total_distance / courier.avg_speed_kmh) * 60  # minutes

        # Distance cost
        distance_cost = total_distance * self.weights["distance"]

        # Time cost
        time_cost = estimated_time * self.weights["time"]

        # Fairness cost
        fairness_cost = (courier.orders_today / 20.0) * self.weights["fairness"]

        # Quality bonus (lower rating = higher cost)
        quality_cost = ((5.0 - courier.rating) / 5.0) * self.weights["quality"]

        # Urgency bonus for old orders (prioritize assignment)
        urgency_bonus = 0.0
        if order.is_urgent:
            urgency_bonus = -self.weights["urgency"]  # Negative cost = bonus

        return distance_cost + time_cost + fairness_cost + quality_cost + urgency_bonus

    def _is_feasible_assignment(
        self,
        order: OrderInfo,
        courier: CourierInfo,
        pickup_distance: float,
        delivery_distance: float,
        constraints: AssignmentConstraints,
    ) -> bool:
        """Check if assignment satisfies all hard constraints."""
        # Courier capacity
        if not courier.has_capacity:
            return False

        # Pickup distance constraint
        if pickup_distance > constraints.max_pickup_distance_km:
            return False

        # Total distance constraint
        total_distance = pickup_distance + delivery_distance
        if total_distance > constraints.max_total_distance_km:
            return False

        # Courier max distance
        if total_distance > courier.max_distance_km:
            return False

        # Vehicle type matching
        if constraints.enforce_vehicle_matching and order.required_vehicle_type:
            if courier.vehicle_type != order.required_vehicle_type:
                return False

        # Courier quality thresholds
        if courier.rating < constraints.min_courier_rating:
            return False

        if courier.completion_rate < constraints.min_completion_rate:
            return False

        # Daily order limit
        if courier.orders_today >= constraints.max_orders_per_courier:
            return False

        # Time constraints
        estimated_pickup_time = (pickup_distance / courier.avg_speed_kmh) * 60
        if estimated_pickup_time > constraints.max_pickup_time_min:
            return False

        estimated_delivery_time = estimated_pickup_time + order.estimated_prep_time_min + (delivery_distance / courier.avg_speed_kmh) * 60
        if estimated_delivery_time > constraints.max_delivery_time_min:
            return False

        return True

    def _apply_constraint_penalties(
        self,
        cost_matrix: np.ndarray,
        problem: AssignmentProblem,
        pickup_distances: np.ndarray,
        total_distances: np.ndarray,
        estimated_times: np.ndarray,
    ) -> np.ndarray:
        """Apply infinite penalties for constraint violations."""
        constraints = problem.constraints

        # Pickup distance violations
        cost_matrix[pickup_distances > constraints.max_pickup_distance_km] = INFINITY_COST

        # Total distance violations
        cost_matrix[total_distances > constraints.max_total_distance_km] = INFINITY_COST

        # Time violations
        cost_matrix[estimated_times > constraints.max_delivery_time_min] = INFINITY_COST

        # Courier-specific constraints
        for j, courier in enumerate(problem.couriers):
            if not courier.has_capacity:
                cost_matrix[:, j] = INFINITY_COST

            if courier.rating < constraints.min_courier_rating:
                cost_matrix[:, j] = INFINITY_COST

            if courier.completion_rate < constraints.min_completion_rate:
                cost_matrix[:, j] = INFINITY_COST

            if courier.orders_today >= constraints.max_orders_per_courier:
                cost_matrix[:, j] = INFINITY_COST

        return cost_matrix

    def get_assignment_details(
        self,
        order: OrderInfo,
        courier: CourierInfo,
        pickup_distance: float,
        delivery_distance: float,
    ) -> dict:
        """Get detailed breakdown of assignment metrics."""
        total_distance = pickup_distance + delivery_distance

        pickup_time = (pickup_distance / courier.avg_speed_kmh) * 60
        delivery_time = (delivery_distance / courier.avg_speed_kmh) * 60
        total_time = pickup_time + order.estimated_prep_time_min + delivery_time

        return {
            "pickup_distance_km": round(pickup_distance, 2),
            "delivery_distance_km": round(delivery_distance, 2),
            "total_distance_km": round(total_distance, 2),
            "estimated_pickup_time_min": round(pickup_time, 1),
            "estimated_prep_time_min": order.estimated_prep_time_min,
            "estimated_delivery_time_min": round(total_time, 1),
            "courier_speed_kmh": courier.avg_speed_kmh,
        }

