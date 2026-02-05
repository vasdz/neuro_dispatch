"""
Dynamic Pricing API endpoints.
"""

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
import h3

from src.common.database import get_session
from src.common.config import settings
from src.common.logging import get_logger
from src.pricing_service.strategy import PricingEngine

router = APIRouter()
logger = get_logger(__name__)


class PriceRequest(BaseModel):
    """Request for price calculation."""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    base_price: float = Field(default=100.0, ge=0)


class PriceResponse(BaseModel):
    """Price calculation response."""
    h3_index: str
    surge_coefficient: float
    base_price: float
    final_price: float
    surge_reason: str


@router.post("/calculate", response_model=PriceResponse)
async def calculate_price(
    request: PriceRequest,
    session: AsyncSession = Depends(get_session),
) -> PriceResponse:
    """Calculate dynamic price for a location."""
    h3_index = h3.geo_to_h3(request.latitude, request.longitude, settings.h3_resolution)

    engine = PricingEngine(session)
    result = await engine.calculate_surge(h3_index)

    final_price = request.base_price * result["coefficient"]

    return PriceResponse(
        h3_index=h3_index,
        surge_coefficient=result["coefficient"],
        base_price=request.base_price,
        final_price=round(final_price, 2),
        surge_reason=result["reason"],
    )


@router.get("/surge/{h3_index}")
async def get_surge_coefficient(
    h3_index: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Get current surge coefficient for a hexagon."""
    engine = PricingEngine(session)
    result = await engine.calculate_surge(h3_index)

    return {
        "h3_index": h3_index,
        "surge_coefficient": result["coefficient"],
        "reason": result["reason"],
        "demand_level": result["demand_level"],
        "supply_level": result["supply_level"],
    }


@router.get("/surge-map")
async def get_surge_map(
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """Get surge coefficients for all active hexagons."""
    engine = PricingEngine(session)
    surge_map = await engine.get_city_surge_map()

    return surge_map

