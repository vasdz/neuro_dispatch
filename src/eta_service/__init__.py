"""
ETA Service - Estimated Time of Arrival Prediction.

Production-ready ETA calculation system with:
- Graph-based routing (simulated OSRM/GraphHopper)
- ML-based time corrections
- Real-time traffic adjustments
- Historical pattern learning
- Confidence intervals

Senior+ implementation following industry best practices.
"""

from src.eta_service.calculator import (
    ETACalculator,
    ETAResult,
    ETABreakdown,
    ETARequest,
)
from src.eta_service.ml_corrector import (
    MLCorrector,
    CorrectionModel,
    CorrectionFeatures,
)
from src.eta_service.routing import (
    RoutingEngine,
    Route,
    RoutePoint,
    RouteSegment,
)

__all__ = [
    "ETACalculator",
    "ETAResult",
    "ETABreakdown",
    "ETARequest",
    "MLCorrector",
    "CorrectionModel",
    "CorrectionFeatures",
    "RoutingEngine",
    "Route",
    "RoutePoint",
    "RouteSegment",
]

