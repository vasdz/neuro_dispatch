"""
ETA Service API Endpoints.

Production-ready REST API for ETA predictions:
- Full delivery ETA (courier → restaurant → customer)
- Simple point-to-point routing
- Batch calculations
- Real-time updates

Senior+ implementation with comprehensive documentation.
"""

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.common.logging import get_logger
from src.eta_service.calculator import ETACalculator, ETARequest, ETAResult
from src.eta_service.routing import TransportType, RoutePoint

logger = get_logger(__name__)
router = APIRouter()


# ============================================================================
# SCHEMAS
# ============================================================================

class LocationSchema(BaseModel):
    """Location coordinates."""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude")


class CourierInfoSchema(BaseModel):
    """Courier information for ETA calculation."""
    id: str | None = Field(None, description="Courier ID")
    transport_type: str = Field("bike", description="Transport type: foot, bike, car, scooter")
    avg_speed_kmh: float = Field(15.0, ge=1, le=100, description="Average speed in km/h")
    experience_days: int = Field(30, ge=0, description="Days of experience")
    current_load: int = Field(1, ge=1, le=5, description="Current number of orders")
    rating: float = Field(4.5, ge=1, le=5, description="Courier rating")


class RestaurantInfoSchema(BaseModel):
    """Restaurant information for ETA calculation."""
    id: str | None = Field(None, description="Restaurant ID")
    avg_prep_time_min: float = Field(15.0, ge=0, le=120, description="Average preparation time")
    current_queue: int = Field(2, ge=0, le=50, description="Current queue size")
    rating: float = Field(4.0, ge=1, le=5, description="Restaurant rating")
    order_complexity: float = Field(0.5, ge=0, le=1, description="Order complexity (0-1)")


class WeatherSchema(BaseModel):
    """Weather conditions."""
    condition: str = Field("clear", description="Weather: clear, rain, snow, etc.")
    temperature_celsius: float = Field(20.0, description="Temperature in Celsius")


class FullETARequest(BaseModel):
    """Request for full delivery ETA calculation."""
    courier_location: LocationSchema
    restaurant_location: LocationSchema
    customer_location: LocationSchema
    courier_info: CourierInfoSchema = Field(default_factory=CourierInfoSchema)
    restaurant_info: RestaurantInfoSchema = Field(default_factory=RestaurantInfoSchema)
    weather: WeatherSchema = Field(default_factory=WeatherSchema)
    order_id: str | None = None
    use_ml_correction: bool = Field(True, description="Use ML model for correction")


class SimpleETARequest(BaseModel):
    """Request for simple point-to-point ETA."""
    origin: LocationSchema
    destination: LocationSchema
    transport_type: str = Field("bike", description="Transport type")


class BatchETARequest(BaseModel):
    """Request for batch ETA calculations."""
    requests: list[SimpleETARequest] = Field(..., max_length=100)


class ETABreakdownSchema(BaseModel):
    """ETA breakdown by phase."""
    phase: str
    duration_minutes: float
    distance_km: float
    confidence: float
    details: dict[str, Any] = {}


class ConfidenceIntervalSchema(BaseModel):
    """Confidence interval for ETA."""
    lower_minutes: float
    upper_minutes: float
    earliest_delivery: datetime
    latest_delivery: datetime


class ETAResponseSchema(BaseModel):
    """Full ETA response."""
    estimated_delivery_time: datetime
    total_duration_minutes: float
    total_distance_km: float
    confidence: float
    confidence_interval: ConfidenceIntervalSchema
    breakdown: list[ETABreakdownSchema]
    model_version: str
    calculated_at: datetime
    calculation_time_ms: float


class SimpleETAResponse(BaseModel):
    """Simple ETA response."""
    distance_km: float
    duration_minutes: float
    transport_type: str


class BatchETAResponse(BaseModel):
    """Batch ETA response."""
    results: list[SimpleETAResponse]
    total_calculations: int
    total_time_ms: float


# ============================================================================
# CALCULATOR INSTANCE
# ============================================================================

_calculator: ETACalculator | None = None


def get_calculator() -> ETACalculator:
    """Get or create ETA calculator instance."""
    global _calculator
    if _calculator is None:
        _calculator = ETACalculator(use_ml_correction=True, correction_model="rule_based")
    return _calculator


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/calculate", response_model=ETAResponseSchema)
async def calculate_eta(request: FullETARequest) -> ETAResponseSchema:
    """
    Calculate full delivery ETA.

    Computes estimated time for complete delivery journey:
    1. Courier travels to restaurant
    2. Restaurant prepares order
    3. Courier delivers to customer

    Returns total ETA with confidence interval and phase breakdown.
    """
    calculator = get_calculator()

    # Parse transport type
    try:
        transport = TransportType(request.courier_info.transport_type)
    except ValueError:
        transport = TransportType.BIKE

    # Build internal request
    eta_request = ETARequest(
        courier_lat=request.courier_location.latitude,
        courier_lon=request.courier_location.longitude,
        restaurant_lat=request.restaurant_location.latitude,
        restaurant_lon=request.restaurant_location.longitude,
        customer_lat=request.customer_location.latitude,
        customer_lon=request.customer_location.longitude,
        courier_id=request.courier_info.id,
        restaurant_id=request.restaurant_info.id,
        order_id=request.order_id,
        transport_type=transport,
        courier_avg_speed=request.courier_info.avg_speed_kmh,
        courier_experience_days=request.courier_info.experience_days,
        courier_current_load=request.courier_info.current_load,
        courier_rating=request.courier_info.rating,
        restaurant_avg_prep_time=request.restaurant_info.avg_prep_time_min,
        restaurant_current_queue=request.restaurant_info.current_queue,
        restaurant_rating=request.restaurant_info.rating,
        order_complexity=request.restaurant_info.order_complexity,
        weather_condition=request.weather.condition,
        temperature_celsius=request.weather.temperature_celsius,
    )

    # Calculate ETA
    result = calculator.calculate(eta_request)

    # Build response
    return ETAResponseSchema(
        estimated_delivery_time=result.estimated_delivery_time,
        total_duration_minutes=result.total_duration_minutes,
        total_distance_km=result.total_distance_km,
        confidence=result.confidence,
        confidence_interval=ConfidenceIntervalSchema(
            lower_minutes=result.lower_bound_minutes,
            upper_minutes=result.upper_bound_minutes,
            earliest_delivery=result.earliest_delivery,
            latest_delivery=result.latest_delivery,
        ),
        breakdown=[
            ETABreakdownSchema(
                phase=b.phase.value,
                duration_minutes=b.duration_minutes,
                distance_km=b.distance_km,
                confidence=b.confidence,
                details=b.details,
            )
            for b in result.breakdown
        ],
        model_version=result.model_version,
        calculated_at=result.calculated_at,
        calculation_time_ms=result.calculation_time_ms,
    )


@router.post("/simple", response_model=SimpleETAResponse)
async def calculate_simple_eta(request: SimpleETARequest) -> SimpleETAResponse:
    """
    Calculate simple point-to-point ETA.

    For basic routing without full delivery context.
    """
    calculator = get_calculator()

    try:
        transport = TransportType(request.transport_type)
    except ValueError:
        transport = TransportType.BIKE

    result = calculator.calculate_simple(
        origin_lat=request.origin.latitude,
        origin_lon=request.origin.longitude,
        destination_lat=request.destination.latitude,
        destination_lon=request.destination.longitude,
        transport_type=transport,
    )

    return SimpleETAResponse(
        distance_km=result["distance_km"],
        duration_minutes=result["duration_minutes"],
        transport_type=result["transport_type"],
    )


@router.post("/batch", response_model=BatchETAResponse)
async def calculate_batch_eta(request: BatchETARequest) -> BatchETAResponse:
    """
    Calculate ETA for multiple routes.

    Efficient batch processing for route planning.
    Maximum 100 routes per request.
    """
    import time
    start = time.perf_counter()

    calculator = get_calculator()
    results = []

    for req in request.requests:
        try:
            transport = TransportType(req.transport_type)
        except ValueError:
            transport = TransportType.BIKE

        result = calculator.calculate_simple(
            origin_lat=req.origin.latitude,
            origin_lon=req.origin.longitude,
            destination_lat=req.destination.latitude,
            destination_lon=req.destination.longitude,
            transport_type=transport,
        )

        results.append(SimpleETAResponse(
            distance_km=result["distance_km"],
            duration_minutes=result["duration_minutes"],
            transport_type=result["transport_type"],
        ))

    total_time = (time.perf_counter() - start) * 1000

    return BatchETAResponse(
        results=results,
        total_calculations=len(results),
        total_time_ms=round(total_time, 3),
    )


@router.get("/models")
async def list_models() -> list[dict[str, str]]:
    """
    List available ETA correction models.

    Returns information about ML models used for ETA adjustment.
    """
    calculator = get_calculator()
    return calculator.get_models_info()


@router.get("/transport-types")
async def list_transport_types() -> list[dict[str, Any]]:
    """
    List supported transport types with their characteristics.
    """
    from src.eta_service.routing import TRANSPORT_SPEEDS

    return [
        {
            "type": t.value,
            "avg_speed_kmh": TRANSPORT_SPEEDS[t],
            "description": {
                TransportType.FOOT: "Walking",
                TransportType.BIKE: "Bicycle",
                TransportType.CAR: "Car (urban traffic)",
                TransportType.SCOOTER: "Electric scooter",
            }.get(t, ""),
        }
        for t in TransportType
    ]


@router.get("/health")
async def eta_service_health() -> dict[str, Any]:
    """
    Health check for ETA service.
    """
    calculator = get_calculator()

    # Test calculation
    try:
        result = calculator.calculate_simple(
            origin_lat=55.7558,
            origin_lon=37.6173,
            destination_lat=55.7600,
            destination_lon=37.6200,
        )
        healthy = result["duration_minutes"] > 0
    except Exception:
        healthy = False

    return {
        "status": "healthy" if healthy else "unhealthy",
        "models": calculator.get_models_info(),
        "routing_engine": "internal",
    }

