"""
Dynamic Pricing API endpoints.

Production-ready API with:
- Price calculation endpoints
- Surge map visualization
- A/B testing management
- Market analytics
- Strategy selection
"""

from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Query, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
import h3

from src.common.database import get_session
from src.common.config import settings
from src.common.logging import get_logger
from src.pricing_service.strategy import PricingEngine, PricingEngineFactory
from src.pricing_service.strategies.base import StrategyType, DemandLevel, SupplyLevel
from src.pricing_service.ab_testing import get_ab_testing_engine

router = APIRouter()
logger = get_logger(__name__)


# ============== Pydantic Schemas ==============

class PriceRequest(BaseModel):
    """Request for price calculation."""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    base_price: float = Field(default=100.0, ge=0)
    order_value: float = Field(default=0.0, ge=0)
    customer_id: Optional[str] = None
    customer_order_count: int = Field(default=0, ge=0)
    customer_is_premium: bool = False
    restaurant_prep_time: int = Field(default=20, ge=0, le=120)
    restaurant_rating: float = Field(default=4.5, ge=1, le=5)


class PriceResponse(BaseModel):
    """Price calculation response."""
    h3_index: str
    surge_coefficient: float
    base_price: float
    final_price: float
    surge_reason: str
    demand_level: str
    supply_level: str
    strategy_used: str
    confidence: float
    is_capped: bool
    factors: dict
    calculated_at: str


class SurgeMapItem(BaseModel):
    """Single item in surge map."""
    h3_index: str
    surge: float
    demand: int
    supply: int
    health: float
    lat: float
    lng: float


class SurgeMapResponse(BaseModel):
    """Surge map response."""
    items: list[SurgeMapItem]
    total: int
    generated_at: str


class MarketSummaryResponse(BaseModel):
    """Market summary response."""
    total_pending_orders: int
    total_available_couriers: int
    active_zones: int
    avg_zone_health: float
    unhealthy_zones: int
    global_ratio: float
    timestamp: str


class ExperimentCreateRequest(BaseModel):
    """Request to create A/B experiment."""
    name: str
    description: str = ""
    control_strategy: Literal["rule_based", "ml_based", "hybrid"] = "rule_based"
    treatment_strategy: Literal["rule_based", "ml_based", "hybrid"] = "hybrid"
    control_traffic: float = Field(default=50.0, ge=10, le=90)


class ExperimentResponse(BaseModel):
    """Experiment info response."""
    id: str
    name: str
    status: str
    variants: list[dict]
    total_impressions: int
    has_sufficient_data: bool
    winner: Optional[dict] = None


# ============== API Endpoints ==============

@router.post("/calculate", response_model=PriceResponse)
async def calculate_price(
    request: PriceRequest,
    session: AsyncSession = Depends(get_session),
) -> PriceResponse:
    """
    Calculate dynamic price for a location.

    Takes into account:
    - Current supply/demand balance
    - Time of day
    - Customer loyalty
    - Restaurant factors
    """
    h3_index = h3.geo_to_h3(request.latitude, request.longitude, settings.h3_resolution)

    engine = PricingEngine(session)

    try:
        result = await engine.calculate_price(
            h3_index=h3_index,
            base_price=request.base_price,
            customer_id=request.customer_id,
            order_value=request.order_value,
            customer_order_count=request.customer_order_count,
            customer_is_premium=request.customer_is_premium,
            restaurant_prep_time=request.restaurant_prep_time,
            restaurant_rating=request.restaurant_rating,
        )
    except Exception as e:
        logger.error(f"Price calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return PriceResponse(
        h3_index=h3_index,
        surge_coefficient=result.surge_coefficient,
        base_price=request.base_price,
        final_price=result.final_price,
        surge_reason=result.reason,
        demand_level=result.demand_level.value,
        supply_level=result.supply_level.value,
        strategy_used=result.strategy_used.value,
        confidence=result.confidence,
        is_capped=result.is_capped,
        factors=result.factors,
        calculated_at=result.calculated_at.isoformat(),
    )


@router.get("/surge/{h3_index}")
async def get_surge_coefficient(
    h3_index: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Get current surge coefficient for a hexagon."""
    engine = PricingEngine(session)

    try:
        result = await engine.calculate_surge(h3_index)
    except Exception as e:
        logger.error(f"Surge calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "h3_index": h3_index,
        "surge_coefficient": result["coefficient"],
        "reason": result["reason"],
        "demand_level": result["demand_level"],
        "supply_level": result["supply_level"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/surge-map", response_model=SurgeMapResponse)
async def get_surge_map(
    limit: int = Query(default=50, ge=10, le=200),
    session: AsyncSession = Depends(get_session),
) -> SurgeMapResponse:
    """
    Get surge coefficients for all active hexagons.

    Returns a map of surge values for visualization.
    """
    engine = PricingEngine(session)

    try:
        surge_data = await engine.get_city_surge_map(limit=limit)
    except Exception as e:
        logger.error(f"Surge map generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    items = []
    for item in surge_data:
        lat, lng = h3.h3_to_geo(item["h3_index"])
        items.append(SurgeMapItem(
            h3_index=item["h3_index"],
            surge=item["surge"],
            demand=item["demand"],
            supply=item["supply"],
            health=item["health"],
            lat=lat,
            lng=lng,
        ))

    return SurgeMapResponse(
        items=items,
        total=len(items),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/market-summary", response_model=MarketSummaryResponse)
async def get_market_summary(
    session: AsyncSession = Depends(get_session),
) -> MarketSummaryResponse:
    """
    Get overall market summary.

    Provides high-level metrics for monitoring and dashboards.
    """
    engine = PricingEngine(session)

    try:
        summary = await engine.get_market_summary()
    except Exception as e:
        logger.error(f"Market summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return MarketSummaryResponse(
        total_pending_orders=summary.get("total_pending_orders", 0),
        total_available_couriers=summary.get("total_available_couriers", 0),
        active_zones=summary.get("active_zones", 0),
        avg_zone_health=summary.get("avg_zone_health", 0.0),
        unhealthy_zones=summary.get("unhealthy_zones", 0),
        global_ratio=summary.get("global_ratio", 0.0),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ============== A/B Testing Endpoints ==============

@router.post("/experiments", response_model=ExperimentResponse)
async def create_experiment(
    request: ExperimentCreateRequest,
) -> ExperimentResponse:
    """
    Create a new A/B testing experiment.

    Allows testing different pricing strategies against each other.
    """
    from src.pricing_service.strategies.rule_based import RuleBasedStrategy
    from src.pricing_service.strategies.ml_based import MLBasedStrategy
    from src.pricing_service.strategies.hybrid import HybridStrategy

    strategy_map = {
        "rule_based": RuleBasedStrategy,
        "ml_based": MLBasedStrategy,
        "hybrid": HybridStrategy,
    }

    control_strategy = strategy_map[request.control_strategy]()
    treatment_strategy = strategy_map[request.treatment_strategy]()
    treatment_traffic = 100.0 - request.control_traffic

    ab_engine = get_ab_testing_engine()

    experiment = ab_engine.create_experiment(
        name=request.name,
        variants=[
            ("control", control_strategy, request.control_traffic),
            ("treatment", treatment_strategy, treatment_traffic),
        ],
        description=request.description,
    )

    return ExperimentResponse(
        id=experiment.id,
        name=experiment.name,
        status=experiment.status.value,
        variants=[
            {"name": v.name, "traffic": v.traffic_percentage}
            for v in experiment.variants
        ],
        total_impressions=experiment.total_impressions,
        has_sufficient_data=experiment.has_sufficient_data,
    )


@router.post("/experiments/{experiment_id}/start")
async def start_experiment(experiment_id: str) -> dict:
    """Start an A/B experiment."""
    ab_engine = get_ab_testing_engine()

    success = ab_engine.start_experiment(experiment_id)
    if not success:
        raise HTTPException(status_code=400, detail="Could not start experiment")

    return {"status": "started", "experiment_id": experiment_id}


@router.post("/experiments/{experiment_id}/stop")
async def stop_experiment(experiment_id: str) -> dict:
    """Stop an A/B experiment."""
    ab_engine = get_ab_testing_engine()

    success = ab_engine.stop_experiment(experiment_id)
    if not success:
        raise HTTPException(status_code=400, detail="Could not stop experiment")

    return {"status": "stopped", "experiment_id": experiment_id}


@router.get("/experiments/{experiment_id}/results")
async def get_experiment_results(experiment_id: str) -> dict:
    """Get A/B experiment results and analysis."""
    ab_engine = get_ab_testing_engine()

    results = ab_engine.get_experiment_results(experiment_id)
    if results is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return results


@router.get("/experiments")
async def list_experiments() -> list[dict]:
    """List all A/B experiments."""
    ab_engine = get_ab_testing_engine()
    return ab_engine.list_experiments()


# ============== Strategy Selection ==============

@router.get("/strategies")
async def list_strategies() -> list[dict]:
    """List available pricing strategies."""
    return [
        {
            "type": "rule_based",
            "name": "Rule-Based Strategy",
            "description": "Traditional surge pricing using predefined rules",
            "version": "2.0.0",
        },
        {
            "type": "ml_based",
            "name": "ML-Based Strategy",
            "description": "Machine learning powered pricing with elasticity awareness",
            "version": "1.0.0",
        },
        {
            "type": "hybrid",
            "name": "Hybrid Strategy",
            "description": "Combination of rule-based and ML approaches",
            "version": "1.0.0",
        },
        {
            "type": "time_decay",
            "name": "Time Decay Strategy",
            "description": "Time-sensitive pricing for aging orders",
            "version": "1.0.0",
        },
    ]
