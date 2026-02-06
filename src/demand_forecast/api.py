"""
Demand Forecasting API endpoints.

Production-ready API with:
- Single and batch predictions
- Heatmap generation
- Historical data access
- Model management endpoints
- Training trigger endpoint
"""

from datetime import datetime, timedelta
from typing import Literal

from fastapi import APIRouter, Depends, Query, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from src.common.database import get_session
from src.common.models import DemandHourly
from src.common.logging import get_logger
from src.demand_forecast.inference import DemandPredictor, ModelRegistry

router = APIRouter()
logger = get_logger(__name__)


# ============== Pydantic Schemas ==============

class DemandPredictionRequest(BaseModel):
    """Request for single hexagon prediction."""
    h3_index: str = Field(..., description="H3 hexagon index")
    hours_ahead: int = Field(default=1, ge=1, le=24, description="Hours to predict ahead")
    model_type: Literal["xgboost", "lightgbm", "ensemble", "baseline"] = Field(
        default="lightgbm", description="Model type to use"
    )


class DemandPredictionResponse(BaseModel):
    """Response for demand prediction."""
    h3_index: str
    hours_ahead: int
    predicted_demand: float
    confidence: float
    model_version: str
    model_type: str
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    h3_indices: list[str] = Field(..., min_length=1, max_length=100)
    hours_ahead: int = Field(default=1, ge=1, le=24)
    model_type: Literal["xgboost", "lightgbm", "ensemble"] = Field(default="lightgbm")


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: list[dict]
    total: int
    model_type: str
    timestamp: str


class HeatmapCell(BaseModel):
    """Single cell in demand heatmap."""
    h3_index: str
    demand: float
    surge: float
    lat: float
    lng: float


class HeatmapResponse(BaseModel):
    """Response for demand heatmap."""
    cells: list[HeatmapCell]
    generated_at: str
    hours_ahead: int


class DemandHistoryResponse(BaseModel):
    """Response for historical demand data."""
    h3_index: str
    data: list[dict]
    from_time: str
    to_time: str


class ModelInfo(BaseModel):
    """Information about a trained model."""
    model_type: str
    path: str
    modified: str


class TrainingTriggerRequest(BaseModel):
    """Request to trigger model training."""
    model_type: Literal["xgboost", "lightgbm", "ensemble"] = Field(default="lightgbm")
    days_back: int = Field(default=30, ge=7, le=90)
    use_mlflow: bool = Field(default=False)


# ============== API Endpoints ==============

@router.get("/history/{h3_index}", response_model=DemandHistoryResponse)
async def get_demand_history(
    h3_index: str,
    hours: int = Query(default=24, ge=1, le=168),
    session: AsyncSession = Depends(get_session),
) -> DemandHistoryResponse:
    """
    Get historical demand for a specific hexagon.

    Returns hourly demand data for the specified time window.
    """
    to_time = datetime.now(timezone.utc)
    from_time = to_time - timedelta(hours=hours)

    result = await session.execute(
        select(DemandHourly)
        .where(
            DemandHourly.h3_index == h3_index,
            DemandHourly.time >= from_time,
        )
        .order_by(DemandHourly.time.desc())
    )

    demands = result.scalars().all()

    return DemandHistoryResponse(
        h3_index=h3_index,
        data=[
            {
                "time": d.time.isoformat(),
                "order_count": d.order_count,
                "avg_surge": d.avg_surge,
                "available_couriers": d.available_couriers,
            }
            for d in demands
        ],
        from_time=from_time.isoformat(),
        to_time=to_time.isoformat(),
    )


@router.get("/predict/{h3_index}", response_model=DemandPredictionResponse)
async def predict_demand(
    h3_index: str,
    hours_ahead: int = Query(default=1, ge=1, le=24),
    model_type: Literal["xgboost", "lightgbm", "ensemble", "baseline"] = Query(default="lightgbm"),
    session: AsyncSession = Depends(get_session),
) -> DemandPredictionResponse:
    """
    Predict demand for a specific hexagon.

    Uses trained ML model or falls back to baseline if unavailable.
    """
    predictor = DemandPredictor(session)

    try:
        prediction = await predictor.predict(h3_index, hours_ahead, model_type)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return DemandPredictionResponse(
        h3_index=h3_index,
        hours_ahead=hours_ahead,
        predicted_demand=prediction["demand"],
        confidence=prediction["confidence"],
        model_version=prediction["model_version"],
        model_type=prediction["model_type"],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_demand_batch(
    request: BatchPredictionRequest,
    session: AsyncSession = Depends(get_session),
) -> BatchPredictionResponse:
    """
    Batch prediction for multiple hexagons.

    More efficient than multiple individual requests.
    Limited to 100 hexagons per request.
    """
    predictor = DemandPredictor(session)

    try:
        predictions = await predictor.predict_batch(
            request.h3_indices,
            request.hours_ahead,
            request.model_type,
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        model_type=request.model_type,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/heatmap", response_model=HeatmapResponse)
async def get_demand_heatmap(
    hours_ahead: int = Query(default=0, ge=0, le=24),
    limit: int = Query(default=100, ge=10, le=500),
    session: AsyncSession = Depends(get_session),
) -> HeatmapResponse:
    """
    Get current or predicted demand heatmap across all hexagons.

    Returns top hexagons by demand for visualization.
    """
    import h3

    if hours_ahead == 0:
        # Current demand from last hour
        since = datetime.now(timezone.utc) - timedelta(hours=1)

        result = await session.execute(
            select(
                DemandHourly.h3_index,
                func.sum(DemandHourly.order_count).label("total_demand"),
                func.avg(DemandHourly.avg_surge).label("avg_surge"),
            )
            .where(DemandHourly.time >= since)
            .group_by(DemandHourly.h3_index)
            .order_by(func.sum(DemandHourly.order_count).desc())
            .limit(limit)
        )

        cells = []
        for row in result.all():
            lat, lng = h3.h3_to_geo(row.h3_index)
            cells.append(HeatmapCell(
                h3_index=row.h3_index,
                demand=float(row.total_demand or 0),
                surge=round(float(row.avg_surge or 1.0), 2),
                lat=lat,
                lng=lng,
            ))
    else:
        # Predicted demand - get unique hexagons and predict
        result = await session.execute(
            select(DemandHourly.h3_index)
            .distinct()
            .limit(limit)
        )

        h3_indices = [row[0] for row in result.all()]

        if h3_indices:
            predictor = DemandPredictor(session)
            predictions = await predictor.predict_batch(h3_indices, hours_ahead)

            cells = []
            for pred in predictions:
                lat, lng = h3.h3_to_geo(pred["h3_index"])
                cells.append(HeatmapCell(
                    h3_index=pred["h3_index"],
                    demand=pred["demand"],
                    surge=1.0,  # Predicted surge not implemented yet
                    lat=lat,
                    lng=lng,
                ))
        else:
            cells = []

    return HeatmapResponse(
        cells=cells,
        generated_at=datetime.now(timezone.utc).isoformat(),
        hours_ahead=hours_ahead,
    )


@router.get("/models", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    """
    List all available trained models.
    """
    registry = ModelRegistry()
    models = registry.list_models()

    return [ModelInfo(**m) for m in models]


@router.post("/train", status_code=202)
async def trigger_training(
    request: TrainingTriggerRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    """
    Trigger model training in background.

    Returns immediately with job ID.
    Training runs asynchronously.
    """
    import uuid

    job_id = str(uuid.uuid4())[:8]

    # Add training task to background
    background_tasks.add_task(
        _run_training,
        job_id,
        request.model_type,
        request.days_back,
        request.use_mlflow,
    )

    logger.info(
        "Training job triggered",
        job_id=job_id,
        model_type=request.model_type,
    )

    return {
        "status": "accepted",
        "job_id": job_id,
        "message": f"Training job {job_id} started for {request.model_type}",
    }


async def _run_training(
    job_id: str,
    model_type: str,
    days_back: int,
    use_mlflow: bool,
) -> None:
    """Background training task."""
    from src.demand_forecast.training import train_demand_model

    try:
        logger.info(f"Starting training job {job_id}")

        result = await train_demand_model(
            model_type=model_type,
            days_back=days_back,
            use_mlflow=use_mlflow,
        )

        logger.info(
            f"Training job {job_id} complete",
            test_mae=result["test_metrics"]["mae"],
        )

    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}")


@router.get("/stats")
async def get_forecast_stats(
    session: AsyncSession = Depends(get_session),
) -> dict:
    """
    Get forecasting service statistics.
    """
    # Get total demand records
    result = await session.execute(
        select(func.count()).select_from(DemandHourly)
    )
    total_records = result.scalar() or 0

    # Get unique hexagons
    result = await session.execute(
        select(func.count(func.distinct(DemandHourly.h3_index)))
    )
    unique_hexagons = result.scalar() or 0

    # Get time range
    result = await session.execute(
        select(func.min(DemandHourly.time), func.max(DemandHourly.time))
    )
    time_range = result.one()

    # Get available models
    registry = ModelRegistry()
    models = registry.list_models()

    return {
        "total_demand_records": total_records,
        "unique_hexagons": unique_hexagons,
        "data_from": time_range[0].isoformat() if time_range[0] else None,
        "data_to": time_range[1].isoformat() if time_range[1] else None,
        "available_models": len(models),
        "models": models,
    }
