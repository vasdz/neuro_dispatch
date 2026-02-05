"""
Tests for dispatch engine solver.
"""

import pytest
from src.dispatch_engine.solver import haversine_distance


class TestHaversineDistance:
    """Tests for haversine distance calculation."""

    def test_same_point_zero_distance(self):
        """Same point should return 0 distance."""
        distance = haversine_distance(55.7558, 37.6173, 55.7558, 37.6173)
        assert distance == 0.0

    def test_known_distance(self):
        """Test with known distance between two Moscow points."""
        # Red Square to Sparrow Hills (~5km)
        lat1, lon1 = 55.7539, 37.6208  # Red Square
        lat2, lon2 = 55.7105, 37.5419  # Sparrow Hills

        distance = haversine_distance(lat1, lon1, lat2, lon2)

        # Should be approximately 7-8 km
        assert 6 < distance < 9

    def test_distance_symmetry(self):
        """Distance should be same in both directions."""
        lat1, lon1 = 55.7558, 37.6173
        lat2, lon2 = 55.8000, 37.7000

        d1 = haversine_distance(lat1, lon1, lat2, lon2)
        d2 = haversine_distance(lat2, lon2, lat1, lon1)

        assert abs(d1 - d2) < 0.001

