"""
Dispatch Solver - Order assignment algorithms.

Senior+ implementation with:
- Multiple algorithm support (Greedy, Hungarian, Batch)
- Cost matrix optimization
- Constraint handling
- Performance monitoring
- Integration with pricing service

This is the main entry point for dispatch operations.
"""

from datetime import datetime, timezone
from typing import Any, Optional, Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.models import Order, Courier, Restaurant, OrderStatus, CourierStatus
from src.common.logging import get_logger
from src.dispatch_engine.algorithms.base import (
    AssignmentProblem,
    AssignmentSolution,
    AssignmentConstraints,
    OrderInfo,
    CourierInfo,
    OptimizationObjective,
)
from src.dispatch_engine.algorithms.greedy import GreedyAssigner, PriorityGreedyAssigner
from src.dispatch_engine.algorithms.hungarian import HungarianAssigner, AuctionAssigner
from src.dispatch_engine.algorithms.batch import BatchAssigner, MultiObjectiveAssigner
from src.dispatch_engine.algorithms.cost_matrix import haversine_distance

logger = get_logger(__name__)


class DispatchSolver:
    """
    Main dispatch solver orchestrating order-courier assignment.

    Features:
    - Multiple algorithm selection
    - Automatic algorithm fallback
    - Constraint configuration
    - Performance tracking
    - Integration with demand forecast and pricing
    """

    # Available algorithms
    ALGORITHMS = {
        "greedy": GreedyAssigner,
        "priority_greedy": PriorityGreedyAssigner,
        "hungarian": HungarianAssigner,
        "auction": AuctionAssigner,
        "batch": BatchAssigner,
        "multi_objective": MultiObjectiveAssigner,
    }

    def __init__(
        self,
        session: AsyncSession,
        default_algorithm: str = "hungarian",
        constraints: Optional[AssignmentConstraints] = None,
    ):
        self.session = session
        self.default_algorithm = default_algorithm
        self.constraints = constraints or AssignmentConstraints()

        # Performance tracking
        self._solve_count = 0
        self._total_solve_time_ms = 0.0
        self._total_assigned = 0
        self._total_unassigned = 0

    async def solve(
        self,
        algorithm: Optional[str] = None,
        objective: OptimizationObjective = OptimizationObjective.BALANCED,
        constraints: Optional[AssignmentConstraints] = None,
    ) -> AssignmentSolution:
        """
        Run dispatch algorithm and return assignments.

        Args:
            algorithm: Algorithm to use (default: self.default_algorithm)
            objective: Optimization objective
            constraints: Override default constraints

        Returns:
            AssignmentSolution with assignments and metadata
        """
        algorithm = algorithm or self.default_algorithm
        constraints = constraints or self.constraints

        # Build problem
        problem = await self._build_problem(objective, constraints)

        if not problem.is_feasible:
            logger.info(
                "No feasible assignment problem",
                orders=problem.n_orders,
                couriers=problem.n_couriers,
            )
            return AssignmentSolution(
                assignments=[],
                n_orders=problem.n_orders,
                n_couriers=problem.n_couriers,
                unassigned_orders=[o.id for o in problem.orders],
                unassigned_reason="No orders or couriers available",
                algorithm=algorithm,
            )

        # Get assigner
        assigner = self._get_assigner(algorithm)

        # Solve
        solution = assigner.solve(problem)

        # Apply assignments to database
        await self._apply_assignments(solution)

        # Update stats
        self._solve_count += 1
        self._total_solve_time_ms += solution.solve_time_ms
        self._total_assigned += solution.n_assigned
        self._total_unassigned += solution.n_unassigned

        logger.info(
            "Dispatch completed",
            algorithm=algorithm,
            assigned=solution.n_assigned,
            unassigned=solution.n_unassigned,
            solve_time_ms=round(solution.solve_time_ms, 2),
        )

        return solution

    async def solve_single(
        self,
        order_id: str,
        algorithm: str = "greedy",
    ) -> Optional[dict]:
        """
        Find best courier for a single order.

        Useful for real-time assignment as orders come in.
        """
        # Get order
        order = await self.session.get(Order, order_id)
        if not order or order.status != OrderStatus.PENDING:
            return None

        # Get restaurant
        restaurant = await self.session.get(Restaurant, order.restaurant_id)
        if not restaurant:
            return None

        # Build single-order problem
        order_info = OrderInfo(
            id=str(order.id),
            restaurant_lat=restaurant.latitude,
            restaurant_lng=restaurant.longitude,
            customer_lat=order.customer_latitude,
            customer_lng=order.customer_longitude,
            h3_index=order.customer_h3_index or "",
            created_at=order.created_at,
            priority=order.priority or 0,
            order_value=float(order.total_amount or 0),
        )

        # Get available couriers
        couriers = await self._get_available_couriers()

        if not couriers:
            return {"status": "no_couriers", "order_id": order_id}

        problem = AssignmentProblem(
            orders=[order_info],
            couriers=couriers,
            constraints=self.constraints,
        )

        assigner = self._get_assigner(algorithm)
        solution = assigner.solve(problem)

        if solution.assignments:
            assignment = solution.assignments[0]
            await self._apply_single_assignment(order, assignment)

            return {
                "status": "assigned",
                "order_id": order_id,
                "courier_id": assignment.courier_id,
                "estimated_delivery_time_min": assignment.estimated_delivery_time_min,
                "pickup_distance_km": assignment.pickup_distance_km,
            }

        return {
            "status": "no_match",
            "order_id": order_id,
            "reason": solution.unassigned_reason,
        }

    async def get_recommendations(
        self,
        order_id: str,
        top_n: int = 5,
    ) -> list[dict]:
        """
        Get top N courier recommendations for an order.

        Useful for manual assignment with suggestions.
        """
        order = await self.session.get(Order, order_id)
        if not order:
            return []

        restaurant = await self.session.get(Restaurant, order.restaurant_id)
        if not restaurant:
            return []

        # Get available couriers
        couriers = await self._get_available_couriers()

        # Calculate scores for each courier
        recommendations = []
        for courier in couriers:
            pickup_distance = haversine_distance(
                courier.lat, courier.lng,
                restaurant.latitude, restaurant.longitude,
            )

            delivery_distance = haversine_distance(
                restaurant.latitude, restaurant.longitude,
                order.customer_latitude, order.customer_longitude,
            )

            total_distance = pickup_distance + delivery_distance
            estimated_time = (total_distance / courier.avg_speed_kmh) * 60

            # Skip if constraints violated
            if pickup_distance > self.constraints.max_pickup_distance_km:
                continue

            if total_distance > self.constraints.max_total_distance_km:
                continue

            recommendations.append({
                "courier_id": courier.id,
                "pickup_distance_km": round(pickup_distance, 2),
                "delivery_distance_km": round(delivery_distance, 2),
                "total_distance_km": round(total_distance, 2),
                "estimated_time_min": round(estimated_time, 1),
                "courier_rating": courier.rating,
                "score": round(1 / (total_distance + 0.1), 4),  # Simple scoring
            })

        # Sort by score descending
        recommendations.sort(key=lambda x: x["score"], reverse=True)

        return recommendations[:top_n]

    def get_stats(self) -> dict:
        """Get solver statistics."""
        return {
            "solve_count": self._solve_count,
            "avg_solve_time_ms": (
                self._total_solve_time_ms / self._solve_count
                if self._solve_count > 0 else 0
            ),
            "total_assigned": self._total_assigned,
            "total_unassigned": self._total_unassigned,
            "assignment_rate": (
                self._total_assigned / (self._total_assigned + self._total_unassigned)
                if (self._total_assigned + self._total_unassigned) > 0 else 0
            ),
        }

    def _get_assigner(self, algorithm: str):
        """Get assigner instance for algorithm."""
        if algorithm not in self.ALGORITHMS:
            logger.warning(f"Unknown algorithm {algorithm}, using greedy")
            algorithm = "greedy"

        return self.ALGORITHMS[algorithm]()

    async def _build_problem(
        self,
        objective: OptimizationObjective,
        constraints: AssignmentConstraints,
    ) -> AssignmentProblem:
        """Build assignment problem from database."""
        orders = await self._get_pending_orders()
        couriers = await self._get_available_couriers()

        return AssignmentProblem(
            orders=orders,
            couriers=couriers,
            constraints=constraints,
            objective=objective,
        )

    async def _get_pending_orders(self) -> list[OrderInfo]:
        """Get pending orders from database."""
        result = await self.session.execute(
            select(Order, Restaurant)
            .join(Restaurant, Order.restaurant_id == Restaurant.id)
            .where(Order.status == OrderStatus.PENDING)
        )

        orders = []
        for order, restaurant in result.all():
            orders.append(OrderInfo(
                id=str(order.id),
                restaurant_lat=restaurant.latitude,
                restaurant_lng=restaurant.longitude,
                customer_lat=order.customer_latitude,
                customer_lng=order.customer_longitude,
                h3_index=order.customer_h3_index or "",
                created_at=order.created_at,
                priority=order.priority or 0,
                order_value=float(order.total_amount or 0),
                estimated_prep_time_min=restaurant.average_prep_time_minutes or 15,
            ))

        return orders

    async def _get_available_couriers(self) -> list[CourierInfo]:
        """Get available couriers from database."""
        result = await self.session.execute(
            select(Courier).where(
                Courier.status == CourierStatus.AVAILABLE,
                Courier.latitude.isnot(None),
                Courier.longitude.isnot(None),
            )
        )

        couriers = []
        for courier in result.scalars().all():
            couriers.append(CourierInfo(
                id=str(courier.id),
                lat=courier.latitude,
                lng=courier.longitude,
                h3_index=courier.h3_index or "",
                vehicle_type=courier.vehicle_type or "bike",
                avg_speed_kmh=courier.average_speed_kmh or 15.0,
                rating=courier.rating or 4.5,
                completion_rate=courier.completion_rate or 0.95,
                orders_today=courier.orders_today or 0,
            ))

        return couriers

    async def _apply_assignments(self, solution: AssignmentSolution):
        """Apply solution assignments to database."""
        for assignment in solution.assignments:
            order = await self.session.get(Order, assignment.order_id)
            courier = await self.session.get(Courier, assignment.courier_id)

            if order and courier:
                order.courier_id = courier.id
                order.status = OrderStatus.ASSIGNED
                order.assigned_at = datetime.now(timezone.utc)
                order.estimated_delivery_time_minutes = int(assignment.estimated_delivery_time_min)

                courier.status = CourierStatus.BUSY
                courier.current_order_id = order.id

                logger.debug(
                    f"Assigned order {order.id} to courier {courier.id}",
                    pickup_distance=assignment.pickup_distance_km,
                    eta=assignment.estimated_delivery_time_min,
                )

        await self.session.commit()

    async def _apply_single_assignment(self, order: Order, assignment):
        """Apply single assignment to database."""
        courier = await self.session.get(Courier, assignment.courier_id)

        if courier:
            order.courier_id = courier.id
            order.status = OrderStatus.ASSIGNED
            order.assigned_at = datetime.now(timezone.utc)
            order.estimated_delivery_time_minutes = int(assignment.estimated_delivery_time_min)

            courier.status = CourierStatus.BUSY
            courier.current_order_id = order.id

            await self.session.commit()


class DispatcherFactory:
    """Factory for creating configured dispatch solvers."""

    @staticmethod
    def create_default(session: AsyncSession) -> DispatchSolver:
        """Create default dispatcher."""
        return DispatchSolver(session)

    @staticmethod
    def create_fast(session: AsyncSession) -> DispatchSolver:
        """Create fast dispatcher (greedy algorithm)."""
        return DispatchSolver(session, default_algorithm="greedy")

    @staticmethod
    def create_optimal(session: AsyncSession) -> DispatchSolver:
        """Create optimal dispatcher (Hungarian algorithm)."""
        return DispatchSolver(session, default_algorithm="hungarian")

    @staticmethod
    def create_with_constraints(
        session: AsyncSession,
        max_pickup_distance: float = 5.0,
        max_delivery_time: int = 45,
    ) -> DispatchSolver:
        """Create dispatcher with custom constraints."""
        constraints = AssignmentConstraints(
            max_pickup_distance_km=max_pickup_distance,
            max_delivery_time_min=max_delivery_time,
        )
        return DispatchSolver(session, constraints=constraints)
