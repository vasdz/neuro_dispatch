"""
Demand Forecasting API endpoints.
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from src.common.database import get_session
from src.common.models import DemandHourly
from src.common.logging import get_logger
from src.demand_forecast.inference import DemandPredictor

router = APIRouter()
logger = get_logger(__name__)


@router.get("/history/{h3_index}")
async def get_demand_history(
    h3_index: str,
    hours: int = Query(default=24, ge=1, le=168),
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """Get historical demand for a specific hexagon."""
    since = datetime.utcnow() - timedelta(hours=hours)

    result = await session.execute(
        select(DemandHourly)
        .where(
            DemandHourly.h3_index == h3_index,
            DemandHourly.time >= since,
        )
        .order_by(DemandHourly.time.desc())
    )

    demands = result.scalars().all()

    return [
        {
            "time": d.time.isoformat(),
            "h3_index": d.h3_index,
            "order_count": d.order_count,
            "avg_surge": d.avg_surge,
            "available_couriers": d.available_couriers,
        }
        for d in demands
    ]


@router.get("/predict/{h3_index}")
async def predict_demand(
    h3_index: str,
    hours_ahead: int = Query(default=1, ge=1, le=24),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Predict demand for a specific hexagon."""
    predictor = DemandPredictor(session)
    prediction = await predictor.predict(h3_index, hours_ahead)

    return {
        "h3_index": h3_index,
        "hours_ahead": hours_ahead,
        "predicted_demand": prediction["demand"],
        "confidence": prediction["confidence"],
        "model_version": prediction["model_version"],
    }


@router.get("/heatmap")
async def get_demand_heatmap(
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """Get current demand heatmap across all hexagons."""
    # Get last hour's demand aggregated by hexagon
    since = datetime.utcnow() - timedelta(hours=1)

    result = await session.execute(
        select(
            DemandHourly.h3_index,
            func.sum(DemandHourly.order_count).label("total_demand"),
            func.avg(DemandHourly.avg_surge).label("avg_surge"),
        )
        .where(DemandHourly.time >= since)
        .group_by(DemandHourly.h3_index)
        .order_by(func.sum(DemandHourly.order_count).desc())
        .limit(100)
    )

    return [
        {
            "h3_index": row.h3_index,
            "demand": int(row.total_demand or 0),
            "surge": round(float(row.avg_surge or 1.0), 2),
        }
        for row in result.all()
    ]

