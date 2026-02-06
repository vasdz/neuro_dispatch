"""
Routing Engine - Graph-based Route Calculation.

Simulates OSRM/GraphHopper functionality with:
- Distance calculation (Haversine + road factor)
- Duration estimation based on transport type
- Route segmentation
- Traffic-aware adjustments

In production, this would integrate with:
- OSRM (Open Source Routing Machine)
- GraphHopper
- Google Directions API
- Yandex Maps API
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import math

from src.common.logging import get_logger

logger = get_logger(__name__)


class TransportType(str, Enum):
    """Transport types for routing."""
    FOOT = "foot"
    BIKE = "bike"
    CAR = "car"
    SCOOTER = "scooter"


# Average speeds in km/h for different transport types
TRANSPORT_SPEEDS: dict[TransportType, float] = {
    TransportType.FOOT: 5.0,
    TransportType.BIKE: 15.0,
    TransportType.CAR: 30.0,  # Urban average with traffic
    TransportType.SCOOTER: 20.0,
}

# Speed adjustments for different conditions
SPEED_ADJUSTMENTS = {
    "rush_hour": 0.6,      # 40% slower during rush hour
    "night": 1.2,          # 20% faster at night (less traffic)
    "rain": 0.8,           # 20% slower in rain
    "snow": 0.5,           # 50% slower in snow
    "weekend": 1.1,        # 10% faster on weekends
}


@dataclass
class RoutePoint:
    """A point on a route."""
    latitude: float
    longitude: float
    name: str = ""
    h3_index: str = ""

    def to_tuple(self) -> tuple[float, float]:
        """Return as (lat, lon) tuple."""
        return (self.latitude, self.longitude)


@dataclass
class RouteSegment:
    """A segment of a route between two points."""
    start: RoutePoint
    end: RoutePoint
    distance_km: float
    duration_minutes: float
    transport_type: TransportType
    segment_type: str = "road"  # road, pedestrian, etc.
    traffic_factor: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start": {"lat": self.start.latitude, "lon": self.start.longitude},
            "end": {"lat": self.end.latitude, "lon": self.end.longitude},
            "distance_km": round(self.distance_km, 3),
            "duration_minutes": round(self.duration_minutes, 2),
            "transport_type": self.transport_type.value,
            "segment_type": self.segment_type,
            "traffic_factor": self.traffic_factor,
        }


@dataclass
class Route:
    """Complete route with all segments."""
    segments: list[RouteSegment]
    total_distance_km: float
    total_duration_minutes: float
    transport_type: TransportType
    calculated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Metadata
    origin: RoutePoint | None = None
    destination: RoutePoint | None = None
    waypoints: list[RoutePoint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_distance_km": round(self.total_distance_km, 3),
            "total_duration_minutes": round(self.total_duration_minutes, 2),
            "transport_type": self.transport_type.value,
            "segments": [s.to_dict() for s in self.segments],
            "calculated_at": self.calculated_at.isoformat(),
        }


class RoutingEngine:
    """
    Routing engine for ETA calculations.

    Provides distance and duration estimates between points.
    In production, would integrate with external routing APIs.
    """

    # Road factor: actual road distance is typically 1.3-1.5x straight-line distance
    ROAD_FACTOR = 1.4

    # Earth radius in km
    EARTH_RADIUS_KM = 6371.0

    def __init__(self, default_transport: TransportType = TransportType.BIKE):
        self.default_transport = default_transport
        self._cache: dict[str, Route] = {}
        self._cache_ttl_minutes = 5

    def calculate_route(
        self,
        origin: RoutePoint,
        destination: RoutePoint,
        transport_type: TransportType | None = None,
        waypoints: list[RoutePoint] | None = None,
        departure_time: datetime | None = None,
    ) -> Route:
        """
        Calculate route between two points.

        Args:
            origin: Starting point
            destination: Ending point
            transport_type: Transport mode (default: bike)
            waypoints: Optional intermediate points
            departure_time: Departure time for traffic estimation

        Returns:
            Route object with segments and totals
        """
        transport = transport_type or self.default_transport
        departure = departure_time or datetime.now(timezone.utc)

        # Build list of all points
        points = [origin]
        if waypoints:
            points.extend(waypoints)
        points.append(destination)

        # Calculate segments
        segments = []
        total_distance = 0.0
        total_duration = 0.0

        for i in range(len(points) - 1):
            segment = self._calculate_segment(
                points[i],
                points[i + 1],
                transport,
                departure,
            )
            segments.append(segment)
            total_distance += segment.distance_km
            total_duration += segment.duration_minutes

        return Route(
            segments=segments,
            total_distance_km=total_distance,
            total_duration_minutes=total_duration,
            transport_type=transport,
            origin=origin,
            destination=destination,
            waypoints=waypoints or [],
        )

    def _calculate_segment(
        self,
        start: RoutePoint,
        end: RoutePoint,
        transport: TransportType,
        departure_time: datetime,
    ) -> RouteSegment:
        """Calculate a single route segment."""
        # Calculate straight-line distance
        straight_distance = self._haversine_distance(
            start.latitude, start.longitude,
            end.latitude, end.longitude,
        )

        # Apply road factor
        road_distance = straight_distance * self.ROAD_FACTOR

        # Get base speed
        base_speed = TRANSPORT_SPEEDS.get(transport, 15.0)

        # Apply time-based adjustments
        traffic_factor = self._get_traffic_factor(departure_time)
        adjusted_speed = base_speed * traffic_factor

        # Calculate duration (distance / speed * 60 for minutes)
        if adjusted_speed > 0:
            duration = (road_distance / adjusted_speed) * 60
        else:
            duration = float('inf')

        return RouteSegment(
            start=start,
            end=end,
            distance_km=road_distance,
            duration_minutes=duration,
            transport_type=transport,
            traffic_factor=traffic_factor,
        )

    def _haversine_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float,
    ) -> float:
        """
        Calculate the great-circle distance between two points.

        Uses the Haversine formula.
        Returns distance in kilometers.
        """
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        # Haversine formula
        a = (
            math.sin(delta_lat / 2) ** 2 +
            math.cos(lat1_rad) * math.cos(lat2_rad) *
            math.sin(delta_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return self.EARTH_RADIUS_KM * c

    def _get_traffic_factor(self, dt: datetime) -> float:
        """
        Get traffic adjustment factor for given time.

        Returns a multiplier for speed (< 1 = slower, > 1 = faster).
        """
        hour = dt.hour
        weekday = dt.weekday()

        # Weekend adjustment
        if weekday >= 5:
            base_factor = SPEED_ADJUSTMENTS["weekend"]
        else:
            base_factor = 1.0

        # Rush hour (7-9 AM, 5-8 PM on weekdays)
        if weekday < 5:
            if 7 <= hour <= 9 or 17 <= hour <= 20:
                base_factor *= SPEED_ADJUSTMENTS["rush_hour"]

        # Night (11 PM - 6 AM)
        if hour >= 23 or hour <= 6:
            base_factor *= SPEED_ADJUSTMENTS["night"]

        return base_factor

    def get_distance(
        self,
        origin: RoutePoint,
        destination: RoutePoint,
    ) -> float:
        """Get distance in km between two points."""
        straight = self._haversine_distance(
            origin.latitude, origin.longitude,
            destination.latitude, destination.longitude,
        )
        return straight * self.ROAD_FACTOR

    def get_duration(
        self,
        origin: RoutePoint,
        destination: RoutePoint,
        transport_type: TransportType | None = None,
    ) -> float:
        """Get estimated duration in minutes."""
        route = self.calculate_route(origin, destination, transport_type)
        return route.total_duration_minutes

