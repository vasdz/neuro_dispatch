"""
H3 Hexagonal Grid Utilities.

This module provides utilities for working with Uber's H3
hexagonal hierarchical spatial index.

Features:
- City grid generation
- Zone classification
- Distance calculations
- Visualization helpers
"""

from typing import Iterator

import h3
import numpy as np

from src.common.config import settings
from src.common.logging import get_logger

logger = get_logger(__name__)


# Moscow city boundaries (approximate bounding box)
MOSCOW_BOUNDS = {
    "min_lat": 55.55,
    "max_lat": 55.95,
    "min_lng": 37.35,
    "max_lng": 37.85,
}

# H3 resolution reference (edge length in meters)
H3_RESOLUTION_EDGE_LENGTH = {
    4: 22606,  # 22.6 km
    5: 8544,   # 8.5 km
    6: 3229,   # 3.2 km
    7: 1220,   # 1.2 km
    8: 461,    # 461 m
    9: 174,    # 174 m
    10: 66,    # 66 m
}


class CityGrid:
    """
    Manages hexagonal grid for a city.

    Provides methods for:
    - Generating hexagons covering the city
    - Zone classification
    - Neighbor lookups
    - Distance calculations
    """

    def __init__(
        self,
        center_lat: float | None = None,
        center_lng: float | None = None,
        resolution: int | None = None,
        bounds: dict[str, float] | None = None,
    ):
        self.center_lat = center_lat or settings.default_city_lat
        self.center_lng = center_lng or settings.default_city_lng
        self.resolution = resolution or settings.h3_resolution
        self.bounds = bounds or MOSCOW_BOUNDS

        self._hexagons: set[str] | None = None
        self._zones: dict[str, list[str]] | None = None

    @property
    def hexagons(self) -> set[str]:
        """Get all hexagons in the city grid."""
        if self._hexagons is None:
            self._hexagons = self._generate_city_hexagons()
        return self._hexagons

    @property
    def zones(self) -> dict[str, list[str]]:
        """Get hexagons organized by zone."""
        if self._zones is None:
            self._zones = self._classify_zones()
        return self._zones

    def _generate_city_hexagons(self) -> set[str]:
        """Generate all hexagons covering the city bounds."""
        # Create polygon from bounds
        polygon = [
            (self.bounds["min_lat"], self.bounds["min_lng"]),
            (self.bounds["min_lat"], self.bounds["max_lng"]),
            (self.bounds["max_lat"], self.bounds["max_lng"]),
            (self.bounds["max_lat"], self.bounds["min_lng"]),
        ]

        # Get hexagons covering the polygon
        hexagons = h3.polyfill_geojson(
            {
                "type": "Polygon",
                "coordinates": [[
                    [lng, lat] for lat, lng in polygon
                ] + [[polygon[0][1], polygon[0][0]]]],  # Close polygon
            },
            self.resolution,
        )

        logger.info(
            f"Generated city grid",
            hexagon_count=len(hexagons),
            resolution=self.resolution,
        )

        return hexagons

    def _classify_zones(self) -> dict[str, list[str]]:
        """Classify hexagons into zones based on distance from center."""
        zones = {
            "center": [],    # < 5 km
            "inner": [],     # 5-10 km
            "middle": [],    # 10-15 km
            "outer": [],     # > 15 km
        }

        for hex_id in self.hexagons:
            lat, lng = h3.h3_to_geo(hex_id)
            distance = self.haversine_distance(
                lat, lng, self.center_lat, self.center_lng
            )

            if distance < 5:
                zones["center"].append(hex_id)
            elif distance < 10:
                zones["inner"].append(hex_id)
            elif distance < 15:
                zones["middle"].append(hex_id)
            else:
                zones["outer"].append(hex_id)

        return zones

    def get_zone(self, h3_index: str) -> str:
        """Get zone name for a hexagon."""
        for zone, hexagons in self.zones.items():
            if h3_index in hexagons:
                return zone
        return "unknown"

    def get_neighbors(self, h3_index: str, k: int = 1) -> list[str]:
        """Get neighboring hexagons within k rings."""
        neighbors = h3.k_ring(h3_index, k)
        return [n for n in neighbors if n != h3_index]

    def get_neighbors_in_city(self, h3_index: str, k: int = 1) -> list[str]:
        """Get neighbors that are within city bounds."""
        neighbors = self.get_neighbors(h3_index, k)
        return [n for n in neighbors if n in self.hexagons]

    def get_distance_km(self, hex1: str, hex2: str) -> float:
        """Get distance between two hexagon centers in kilometers."""
        lat1, lng1 = h3.h3_to_geo(hex1)
        lat2, lng2 = h3.h3_to_geo(hex2)
        return self.haversine_distance(lat1, lng1, lat2, lng2)

    def get_hexagon_for_point(self, lat: float, lng: float) -> str:
        """Get hexagon index for a point."""
        return h3.geo_to_h3(lat, lng, self.resolution)

    def is_in_city(self, h3_index: str) -> bool:
        """Check if hexagon is within city bounds."""
        return h3_index in self.hexagons

    def get_hexagon_area_km2(self) -> float:
        """Get approximate area of a hexagon in km²."""
        # Use first hexagon as reference
        if not self.hexagons:
            return 0.0

        sample_hex = next(iter(self.hexagons))
        area_m2 = h3.cell_area(sample_hex, unit='m^2')
        return area_m2 / 1_000_000

    def get_hexagon_boundary(self, h3_index: str) -> list[tuple[float, float]]:
        """Get boundary coordinates of a hexagon."""
        boundary = h3.h3_to_geo_boundary(h3_index)
        return [(lat, lng) for lat, lng in boundary]

    @staticmethod
    def haversine_distance(
        lat1: float, lng1: float,
        lat2: float, lng2: float,
    ) -> float:
        """Calculate distance between two points in kilometers."""
        R = 6371  # Earth's radius in km

        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lng = np.radians(lng2 - lng1)

        a = (
            np.sin(delta_lat / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lng / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return float(R * c)


class HotspotDetector:
    """
    Detects demand hotspots using hexagonal grid analysis.

    Uses spatial clustering to identify areas of high activity.
    """

    def __init__(self, grid: CityGrid):
        self.grid = grid

    def detect_hotspots(
        self,
        demand_map: dict[str, float],
        threshold_percentile: float = 90,
    ) -> list[dict]:
        """
        Detect hotspots based on demand values.

        Args:
            demand_map: Dictionary of h3_index -> demand value
            threshold_percentile: Percentile threshold for hotspot

        Returns:
            List of hotspot info dictionaries
        """
        if not demand_map:
            return []

        # Calculate threshold
        values = list(demand_map.values())
        threshold = np.percentile(values, threshold_percentile)

        # Find hotspots
        hotspots = []
        processed = set()

        for h3_index, demand in sorted(
            demand_map.items(), key=lambda x: x[1], reverse=True
        ):
            if h3_index in processed:
                continue

            if demand < threshold:
                continue

            # Found a hotspot center
            cluster = self._grow_cluster(
                h3_index, demand_map, threshold * 0.7, processed
            )

            lat, lng = h3.h3_to_geo(h3_index)

            hotspots.append({
                "center": h3_index,
                "center_lat": lat,
                "center_lng": lng,
                "peak_demand": demand,
                "avg_demand": np.mean([demand_map.get(h, 0) for h in cluster]),
                "hexagon_count": len(cluster),
                "hexagons": list(cluster),
                "zone": self.grid.get_zone(h3_index),
            })

            processed.update(cluster)

        return hotspots[:10]  # Top 10 hotspots

    def _grow_cluster(
        self,
        center: str,
        demand_map: dict[str, float],
        min_demand: float,
        processed: set[str],
    ) -> set[str]:
        """Grow cluster from center using flood fill."""
        cluster = {center}
        queue = [center]
        processed.add(center)

        while queue:
            current = queue.pop(0)

            for neighbor in self.grid.get_neighbors(current, k=1):
                if neighbor in processed:
                    continue

                if demand_map.get(neighbor, 0) >= min_demand:
                    cluster.add(neighbor)
                    queue.append(neighbor)
                    processed.add(neighbor)

        return cluster


def get_city_grid() -> CityGrid:
    """Get singleton city grid instance."""
    return CityGrid()


def point_to_h3(lat: float, lng: float, resolution: int | None = None) -> str:
    """Convert point to H3 index."""
    res = resolution or settings.h3_resolution
    return h3.geo_to_h3(lat, lng, res)


def h3_to_point(h3_index: str) -> tuple[float, float]:
    """Convert H3 index to center point."""
    return h3.h3_to_geo(h3_index)


def get_hexagon_geojson(h3_index: str) -> dict:
    """Get GeoJSON representation of a hexagon."""
    boundary = h3.h3_to_geo_boundary(h3_index, geo_json=True)

    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [list(boundary) + [boundary[0]]],
        },
        "properties": {
            "h3_index": h3_index,
            "resolution": h3.h3_get_resolution(h3_index),
        },
    }


def hexagons_to_geojson(
    h3_indices: list[str],
    properties: dict[str, dict] | None = None,
) -> dict:
    """
    Convert list of hexagons to GeoJSON FeatureCollection.

    Args:
        h3_indices: List of H3 indices
        properties: Optional dict of h3_index -> properties

    Returns:
        GeoJSON FeatureCollection
    """
    features = []

    for h3_index in h3_indices:
        boundary = h3.h3_to_geo_boundary(h3_index, geo_json=True)

        props = {"h3_index": h3_index}
        if properties and h3_index in properties:
            props.update(properties[h3_index])

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [list(boundary) + [boundary[0]]],
            },
            "properties": props,
        })

    return {
        "type": "FeatureCollection",
        "features": features,
    }

