"""
Dispatch Solver - Order assignment algorithms.

Implements various assignment strategies:
- Greedy (nearest courier)
- Hungarian Algorithm (optimal bipartite matching)
- Min-cost max-flow (for complex constraints)
"""

from datetime import datetime
from typing import Any
import math

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.models import Order, Courier, OrderStatus, CourierStatus
from src.common.logging import get_logger

logger = get_logger(__name__)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    Returns distance in kilometers.
    """
    R = 6371  # Radius of earth in kilometers

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


class DispatchSolver:
    """
    Dispatch solver for order-courier assignment.

    Currently implements a greedy algorithm.
    Future: Hungarian algorithm for optimal matching.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def solve(self, strategy: str = "greedy") -> list[dict]:
        """
        Run dispatch algorithm and return assignments.

        Args:
            strategy: Algorithm to use ('greedy', 'hungarian', 'mincost')

        Returns:
            List of assignment results
        """
        if strategy == "greedy":
            return await self._solve_greedy()
        elif strategy == "hungarian":
            return await self._solve_hungarian()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    async def _solve_greedy(self) -> list[dict]:
        """
        Greedy assignment: assign each order to nearest available courier.

        Time complexity: O(n*m) where n=orders, m=couriers
        Not optimal globally but fast.
        """
        # Get pending orders
        orders_result = await self.session.execute(
            select(Order).where(Order.status == OrderStatus.PENDING)
        )
        pending_orders = list(orders_result.scalars().all())

        # Get available couriers
        couriers_result = await self.session.execute(
            select(Courier).where(
                Courier.status == CourierStatus.AVAILABLE,
                Courier.latitude.isnot(None),
                Courier.longitude.isnot(None),
            )
        )
        available_couriers = list(couriers_result.scalars().all())

        logger.info(
            "Running greedy dispatch",
            pending_orders=len(pending_orders),
            available_couriers=len(available_couriers),
        )

        assignments = []
        assigned_courier_ids = set()

        for order in pending_orders:
            best_courier = None
            best_distance = float("inf")

            # Find nearest available courier
            for courier in available_couriers:
                if courier.id in assigned_courier_ids:
                    continue

                if not courier.has_location:
                    continue

                # Get restaurant location for pickup
                restaurant = await self.session.get(
                    Order.restaurant.property.mapper.class_,
                    order.restaurant_id,
                )

                if not restaurant:
                    continue

                # Distance from courier to restaurant
                distance = haversine_distance(
                    courier.latitude,
                    courier.longitude,
                    restaurant.latitude,
                    restaurant.longitude,
                )

                if distance < best_distance:
                    best_distance = distance
                    best_courier = courier

            # Assign if found
            if best_courier:
                order.courier_id = best_courier.id
                order.status = OrderStatus.ASSIGNED
                order.assigned_at = datetime.utcnow()

                best_courier.status = CourierStatus.BUSY
                best_courier.current_order_id = order.id

                assigned_courier_ids.add(best_courier.id)

                # Estimate delivery time based on distance and speed
                total_distance = best_distance + haversine_distance(
                    order.customer_latitude,
                    order.customer_longitude,
                    restaurant.latitude if restaurant else order.customer_latitude,
                    restaurant.longitude if restaurant else order.customer_longitude,
                )
                eta_minutes = int((total_distance / best_courier.avg_speed_kmh) * 60)
                order.estimated_delivery_time_minutes = eta_minutes + (order.estimated_prep_time_minutes or 15)

                assignments.append({
                    "order_id": order.id,
                    "courier_id": best_courier.id,
                    "distance_km": round(best_distance, 2),
                    "eta_minutes": order.estimated_delivery_time_minutes,
                })

                logger.info(
                    "Order assigned (greedy)",
                    order_id=order.id,
                    courier_id=best_courier.id,
                    distance_km=round(best_distance, 2),
                )

        await self.session.commit()

        return assignments

    async def _solve_hungarian(self) -> list[dict]:
        """
        Hungarian algorithm for optimal bipartite matching.

        Minimizes total assignment cost (distance).
        Time complexity: O(n^3)

        TODO: Implement using scipy.optimize.linear_sum_assignment
        or custom C++ implementation via pybind11.
        """
        logger.warning("Hungarian algorithm not yet implemented, falling back to greedy")
        return await self._solve_greedy()

    def _build_cost_matrix(
        self,
        orders: list[Order],
        couriers: list[Courier],
    ) -> list[list[float]]:
        """
        Build cost matrix for assignment problem.

        cost[i][j] = distance from courier j to order i's restaurant
        """
        n_orders = len(orders)
        n_couriers = len(couriers)

        # Initialize with infinity
        cost_matrix = [[float("inf")] * n_couriers for _ in range(n_orders)]

        for i, order in enumerate(orders):
            for j, courier in enumerate(couriers):
                if courier.has_location:
                    # Simplified: just distance to customer
                    # Full version would include restaurant pickup
                    cost_matrix[i][j] = haversine_distance(
                        courier.latitude,
                        courier.longitude,
                        order.customer_latitude,
                        order.customer_longitude,
                    )

        return cost_matrix

