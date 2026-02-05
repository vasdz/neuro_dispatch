"""
Dispatch Engine API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.common.database import get_session
from src.common.models import Order, Courier, OrderStatus, CourierStatus
from src.common.schemas import OrderResponse, OrderAssignment
from src.common.logging import get_logger
from src.dispatch_engine.solver import DispatchSolver

router = APIRouter()
logger = get_logger(__name__)


@router.get("/orders/pending", response_model=list[OrderResponse])
async def get_pending_orders(
    session: AsyncSession = Depends(get_session),
) -> list[Order]:
    """Get all pending orders waiting for assignment."""
    result = await session.execute(
        select(Order).where(Order.status == OrderStatus.PENDING)
    )
    return list(result.scalars().all())


@router.get("/couriers/available")
async def get_available_couriers(
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """Get all available couriers."""
    result = await session.execute(
        select(Courier).where(Courier.status == CourierStatus.AVAILABLE)
    )
    couriers = result.scalars().all()
    return [
        {
            "id": c.id,
            "name": c.name,
            "latitude": c.latitude,
            "longitude": c.longitude,
            "h3_index": c.h3_index,
        }
        for c in couriers
    ]


@router.post("/assign")
async def assign_order(
    assignment: OrderAssignment,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Manually assign an order to a courier."""
    # Get order
    order = await session.get(Order, assignment.order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    if order.status != OrderStatus.PENDING:
        raise HTTPException(status_code=400, detail="Order is not pending")

    # Get courier
    courier = await session.get(Courier, assignment.courier_id)
    if not courier:
        raise HTTPException(status_code=404, detail="Courier not found")

    if courier.status != CourierStatus.AVAILABLE:
        raise HTTPException(status_code=400, detail="Courier is not available")

    # Assign
    order.courier_id = courier.id
    order.status = OrderStatus.ASSIGNED
    courier.status = CourierStatus.BUSY
    courier.current_order_id = order.id

    if assignment.estimated_delivery_time_minutes:
        order.estimated_delivery_time_minutes = assignment.estimated_delivery_time_minutes

    await session.commit()

    logger.info(
        "Order assigned",
        order_id=order.id,
        courier_id=courier.id,
    )

    return {"status": "assigned", "order_id": order.id, "courier_id": courier.id}


@router.post("/dispatch/run")
async def run_dispatch(
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Run automatic dispatch algorithm to assign pending orders."""
    solver = DispatchSolver(session)
    assignments = await solver.solve()

    return {
        "status": "completed",
        "assignments": len(assignments),
        "details": assignments,
    }

