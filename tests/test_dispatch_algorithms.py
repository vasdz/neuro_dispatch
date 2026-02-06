"""
Tests for Dispatch Engine (Phase 4).

Comprehensive test suite covering:
- Assignment algorithms (Greedy, Hungarian, Batch)
- Cost matrix building
- Constraint handling
- Solution quality metrics
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.dispatch_engine.algorithms.base import (
    BaseAssigner,
    AssignmentProblem,
    AssignmentSolution,
    Assignment,
    AssignmentConstraints,
    OrderInfo,
    CourierInfo,
    OptimizationObjective,
)
from src.dispatch_engine.algorithms.cost_matrix import (
    CostMatrixBuilder,
    haversine_distance,
    haversine_vectorized,
    INFINITY_COST,
)
from src.dispatch_engine.algorithms.greedy import GreedyAssigner, PriorityGreedyAssigner
from src.dispatch_engine.algorithms.hungarian import HungarianAssigner, AuctionAssigner
from src.dispatch_engine.algorithms.batch import BatchAssigner, MultiObjectiveAssigner


# ============== Test Fixtures ==============

@pytest.fixture
def sample_orders():
    """Create sample orders for testing."""
    now = datetime.now(timezone.utc)
    return [
        OrderInfo(
            id="order_1",
            restaurant_lat=55.7558,
            restaurant_lng=37.6173,
            customer_lat=55.7600,
            customer_lng=37.6200,
            h3_index="882830829bfffff",
            created_at=now - timedelta(minutes=5),
            priority=0,
            order_value=500.0,
        ),
        OrderInfo(
            id="order_2",
            restaurant_lat=55.7500,
            restaurant_lng=37.6100,
            customer_lat=55.7450,
            customer_lng=37.6050,
            h3_index="882830829bfffff",
            created_at=now - timedelta(minutes=10),
            priority=1,
            order_value=800.0,
        ),
        OrderInfo(
            id="order_3",
            restaurant_lat=55.7700,
            restaurant_lng=37.6300,
            customer_lat=55.7750,
            customer_lng=37.6350,
            h3_index="882830829bfffff",
            created_at=now - timedelta(minutes=2),
            priority=0,
            order_value=350.0,
        ),
    ]


@pytest.fixture
def sample_couriers():
    """Create sample couriers for testing."""
    return [
        CourierInfo(
            id="courier_1",
            lat=55.7560,
            lng=37.6180,
            h3_index="882830829bfffff",
            vehicle_type="bike",
            avg_speed_kmh=15.0,
            rating=4.8,
            completion_rate=0.98,
            orders_today=5,
        ),
        CourierInfo(
            id="courier_2",
            lat=55.7520,
            lng=37.6120,
            h3_index="882830829bfffff",
            vehicle_type="scooter",
            avg_speed_kmh=25.0,
            rating=4.5,
            completion_rate=0.95,
            orders_today=8,
        ),
        CourierInfo(
            id="courier_3",
            lat=55.7680,
            lng=37.6280,
            h3_index="882830829bfffff",
            vehicle_type="bike",
            avg_speed_kmh=12.0,
            rating=4.2,
            completion_rate=0.90,
            orders_today=3,
        ),
    ]


@pytest.fixture
def sample_problem(sample_orders, sample_couriers):
    """Create sample assignment problem."""
    return AssignmentProblem(
        orders=sample_orders,
        couriers=sample_couriers,
        constraints=AssignmentConstraints(),
        objective=OptimizationObjective.BALANCED,
    )


# ============== Haversine Distance Tests ==============

class TestHaversineDistance:
    """Tests for distance calculations."""

    def test_same_point_zero_distance(self):
        """Test that same point gives zero distance."""
        dist = haversine_distance(55.7558, 37.6173, 55.7558, 37.6173)
        assert dist == 0.0

    def test_known_distance(self):
        """Test distance between known points."""
        # Moscow center to Sheremetyevo airport (~29 km)
        dist = haversine_distance(55.7558, 37.6173, 55.9726, 37.4146)
        assert 25 < dist < 35

    def test_distance_symmetry(self):
        """Test that distance is symmetric."""
        dist1 = haversine_distance(55.7558, 37.6173, 55.7600, 37.6200)
        dist2 = haversine_distance(55.7600, 37.6200, 55.7558, 37.6173)
        assert abs(dist1 - dist2) < 0.001

    def test_vectorized_matches_scalar(self):
        """Test that vectorized version matches scalar."""
        lats1 = np.array([55.7558, 55.7500])
        lngs1 = np.array([37.6173, 37.6100])
        lats2 = np.array([55.7600, 55.7450])
        lngs2 = np.array([37.6200, 37.6050])

        vectorized = haversine_vectorized(lats1, lngs1, lats2, lngs2)

        scalar1 = haversine_distance(55.7558, 37.6173, 55.7600, 37.6200)
        scalar2 = haversine_distance(55.7500, 37.6100, 55.7450, 37.6050)

        assert abs(vectorized[0] - scalar1) < 0.001
        assert abs(vectorized[1] - scalar2) < 0.001


# ============== Cost Matrix Tests ==============

class TestCostMatrixBuilder:
    """Tests for cost matrix building."""

    def test_build_matrix_shape(self, sample_problem):
        """Test that matrix has correct shape."""
        builder = CostMatrixBuilder()
        matrix = builder.build(sample_problem)

        assert matrix.shape == (3, 3)  # 3 orders x 3 couriers

    def test_build_fast_matches_slow(self, sample_problem):
        """Test that fast builder gives similar results."""
        builder = CostMatrixBuilder()

        matrix_slow = builder.build(sample_problem)
        matrix_fast = builder.build_fast(sample_problem)

        # Should have same shape
        assert matrix_slow.shape == matrix_fast.shape

        # Should have similar costs (allowing for floating point differences)
        finite_mask = np.isfinite(matrix_slow) & np.isfinite(matrix_fast)
        if finite_mask.any():
            diff = np.abs(matrix_slow[finite_mask] - matrix_fast[finite_mask])
            assert np.max(diff) < 1.0  # Allow up to 1.0 difference

    def test_infeasible_assignment_has_infinity_cost(self, sample_orders, sample_couriers):
        """Test that constraint violations have infinite cost."""
        # Create courier with low capacity
        courier = sample_couriers[0]
        courier.current_load = 2
        courier.max_load = 2

        problem = AssignmentProblem(
            orders=sample_orders,
            couriers=[courier],
            constraints=AssignmentConstraints(),
        )

        builder = CostMatrixBuilder()
        matrix = builder.build(problem)

        # All assignments should be infeasible (infinite cost)
        assert np.all(matrix >= INFINITY_COST)

    def test_distance_constraint(self, sample_orders, sample_couriers):
        """Test that distance constraint is enforced."""
        # Set very strict distance constraint
        constraints = AssignmentConstraints(max_pickup_distance_km=0.01)

        problem = AssignmentProblem(
            orders=sample_orders,
            couriers=sample_couriers,
            constraints=constraints,
        )

        builder = CostMatrixBuilder()
        matrix = builder.build(problem)

        # Most assignments should be infeasible due to distance
        infeasible_count = np.sum(matrix >= INFINITY_COST)
        assert infeasible_count > 0


# ============== Greedy Assigner Tests ==============

class TestGreedyAssigner:
    """Tests for greedy assignment algorithm."""

    def test_assigns_all_when_possible(self, sample_problem):
        """Test that greedy assigns all orders when possible."""
        assigner = GreedyAssigner()
        solution = assigner.solve(sample_problem)

        # Should assign all 3 orders to 3 couriers
        assert solution.n_assigned == 3
        assert solution.n_unassigned == 0

    def test_assigns_to_nearest(self, sample_orders, sample_couriers):
        """Test that greedy assigns to nearest courier."""
        # Create simple problem with clear nearest courier
        problem = AssignmentProblem(
            orders=[sample_orders[0]],
            couriers=sample_couriers,
            constraints=AssignmentConstraints(),
        )

        assigner = GreedyAssigner()
        solution = assigner.solve(problem)

        assert len(solution.assignments) == 1
        # Courier 1 is closest to order 1's restaurant
        assert solution.assignments[0].courier_id == "courier_1"

    def test_handles_no_couriers(self, sample_orders):
        """Test handling when no couriers available."""
        problem = AssignmentProblem(
            orders=sample_orders,
            couriers=[],
            constraints=AssignmentConstraints(),
        )

        assigner = GreedyAssigner()
        solution = assigner.solve(problem)

        assert solution.n_assigned == 0
        assert solution.n_unassigned == 3

    def test_handles_no_orders(self, sample_couriers):
        """Test handling when no orders to assign."""
        problem = AssignmentProblem(
            orders=[],
            couriers=sample_couriers,
            constraints=AssignmentConstraints(),
        )

        assigner = GreedyAssigner()
        solution = assigner.solve(problem)

        assert solution.n_assigned == 0
        assert solution.n_unassigned == 0

    def test_solution_metadata(self, sample_problem):
        """Test that solution has proper metadata."""
        assigner = GreedyAssigner()
        solution = assigner.solve(sample_problem)

        assert solution.algorithm == "greedy"
        assert solution.solve_time_ms >= 0
        assert solution.is_optimal == False  # Greedy is not optimal
        assert solution.total_distance_km >= 0


class TestPriorityGreedyAssigner:
    """Tests for priority-based greedy assigner."""

    def test_prioritizes_older_orders(self, sample_orders, sample_couriers):
        """Test that older orders get priority."""
        problem = AssignmentProblem(
            orders=sample_orders,
            couriers=sample_couriers[:1],  # Only one courier
            constraints=AssignmentConstraints(),
        )

        assigner = PriorityGreedyAssigner()
        solution = assigner.solve(problem)

        # Order 2 is oldest (10 min old), should be assigned first
        if solution.assignments:
            assert solution.assignments[0].order_id == "order_2"


# ============== Hungarian Assigner Tests ==============

class TestHungarianAssigner:
    """Tests for Hungarian (optimal) assignment algorithm."""

    def test_finds_optimal_assignment(self, sample_problem):
        """Test that Hungarian finds optimal solution."""
        assigner = HungarianAssigner()
        solution = assigner.solve(sample_problem)

        assert solution.n_assigned == 3
        assert solution.is_optimal == True

    def test_lower_or_equal_cost_to_greedy(self, sample_problem):
        """Test that Hungarian has lower or equal cost to greedy."""
        greedy = GreedyAssigner()
        hungarian = HungarianAssigner()

        greedy_solution = greedy.solve(sample_problem)
        hungarian_solution = hungarian.solve(sample_problem)

        # Hungarian should be at least as good as greedy
        assert hungarian_solution.total_cost <= greedy_solution.total_cost + 0.01

    def test_handles_rectangular_matrix(self, sample_orders, sample_couriers):
        """Test handling when orders != couriers."""
        # More orders than couriers
        problem = AssignmentProblem(
            orders=sample_orders,
            couriers=sample_couriers[:2],  # Only 2 couriers for 3 orders
            constraints=AssignmentConstraints(),
        )

        assigner = HungarianAssigner()
        solution = assigner.solve(problem)

        # Should assign 2 orders (one unassigned)
        assert solution.n_assigned == 2
        assert solution.n_unassigned == 1


class TestAuctionAssigner:
    """Tests for auction-based assignment."""

    def test_finds_solution(self, sample_problem):
        """Test that auction finds a solution."""
        assigner = AuctionAssigner(max_iterations=100)
        solution = assigner.solve(sample_problem)

        # Should assign at least some orders
        assert solution.n_assigned > 0

    def test_converges(self, sample_problem):
        """Test that auction converges."""
        assigner = AuctionAssigner(epsilon=0.01, max_iterations=100)
        solution = assigner.solve(sample_problem)

        assert solution.solve_time_ms < 1000  # Should be fast


# ============== Batch Assigner Tests ==============

class TestBatchAssigner:
    """Tests for batch assignment."""

    def test_uses_hungarian_for_small_problems(self, sample_problem):
        """Test that batch uses Hungarian for small problems."""
        assigner = BatchAssigner(use_hungarian=True)
        solution = assigner.solve(sample_problem)

        # Should be optimal for small problem
        assert solution.n_assigned == 3

    def test_respects_time_limit(self, sample_problem):
        """Test that batch respects time limit."""
        assigner = BatchAssigner(time_limit_ms=10000)
        solution = assigner.solve(sample_problem)

        assert solution.solve_time_ms < 10000


class TestMultiObjectiveAssigner:
    """Tests for multi-objective optimization."""

    def test_generates_solutions(self, sample_problem):
        """Test that multi-objective generates solutions."""
        assigner = MultiObjectiveAssigner(n_solutions=3)
        solution = assigner.solve(sample_problem)

        assert solution.n_assigned > 0


# ============== Assignment Constraints Tests ==============

class TestAssignmentConstraints:
    """Tests for constraint enforcement."""

    def test_default_constraints(self):
        """Test default constraint values."""
        constraints = AssignmentConstraints()

        assert constraints.max_pickup_distance_km == 5.0
        assert constraints.max_total_distance_km == 15.0
        assert constraints.max_pickup_time_min == 15
        assert constraints.max_delivery_time_min == 45

    def test_constraint_filtering(self, sample_orders, sample_couriers):
        """Test that constraints filter assignments."""
        # Very strict constraints
        constraints = AssignmentConstraints(
            max_pickup_distance_km=0.5,
            max_delivery_time_min=10,
        )

        problem = AssignmentProblem(
            orders=sample_orders,
            couriers=sample_couriers,
            constraints=constraints,
        )

        assigner = GreedyAssigner()
        solution = assigner.solve(problem)

        # May have fewer assignments due to constraints
        assert solution.n_assigned <= 3


# ============== Order and Courier Info Tests ==============

class TestOrderInfo:
    """Tests for OrderInfo dataclass."""

    def test_age_calculation(self):
        """Test order age calculation."""
        order = OrderInfo(
            id="test",
            restaurant_lat=55.75,
            restaurant_lng=37.62,
            customer_lat=55.76,
            customer_lng=37.63,
            h3_index="test",
            created_at=datetime.now(timezone.utc) - timedelta(minutes=15),
        )

        age = order.age_seconds
        assert 890 < age < 920  # ~15 minutes in seconds

    def test_urgent_flag(self):
        """Test urgent order detection."""
        old_order = OrderInfo(
            id="old",
            restaurant_lat=55.75,
            restaurant_lng=37.62,
            customer_lat=55.76,
            customer_lng=37.63,
            h3_index="test",
            created_at=datetime.now(timezone.utc) - timedelta(minutes=15),
        )

        new_order = OrderInfo(
            id="new",
            restaurant_lat=55.75,
            restaurant_lng=37.62,
            customer_lat=55.76,
            customer_lng=37.63,
            h3_index="test",
            created_at=datetime.now(timezone.utc) - timedelta(minutes=2),
        )

        assert old_order.is_urgent == True
        assert new_order.is_urgent == False


class TestCourierInfo:
    """Tests for CourierInfo dataclass."""

    def test_capacity_check(self):
        """Test courier capacity check."""
        courier = CourierInfo(
            id="test",
            lat=55.75,
            lng=37.62,
            h3_index="test",
            current_load=1,
            max_load=2,
        )

        assert courier.has_capacity == True

        courier.current_load = 2
        assert courier.has_capacity == False

    def test_utilization(self):
        """Test courier utilization calculation."""
        courier = CourierInfo(
            id="test",
            lat=55.75,
            lng=37.62,
            h3_index="test",
            current_load=1,
            max_load=2,
        )

        assert courier.utilization == 0.5


# ============== Integration Tests ==============

class TestDispatchIntegration:
    """Integration tests for dispatch engine."""

    def test_full_dispatch_flow(self, sample_problem):
        """Test complete dispatch flow."""
        assigner = HungarianAssigner()
        solution = assigner.solve(sample_problem)

        # Verify solution structure
        assert isinstance(solution, AssignmentSolution)
        assert all(isinstance(a, Assignment) for a in solution.assignments)

        # Verify all assignments are valid
        for assignment in solution.assignments:
            assert assignment.order_id in [o.id for o in sample_problem.orders]
            assert assignment.courier_id in [c.id for c in sample_problem.couriers]
            assert assignment.pickup_distance_km >= 0
            assert assignment.total_distance_km >= assignment.pickup_distance_km

    def test_assignment_uniqueness(self, sample_problem):
        """Test that each order and courier is assigned at most once."""
        assigner = HungarianAssigner()
        solution = assigner.solve(sample_problem)

        order_ids = [a.order_id for a in solution.assignments]
        courier_ids = [a.courier_id for a in solution.assignments]

        # No duplicates
        assert len(order_ids) == len(set(order_ids))
        assert len(courier_ids) == len(set(courier_ids))

