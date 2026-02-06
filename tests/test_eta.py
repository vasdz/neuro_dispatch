"""
Tests for ETA Service.

Comprehensive test suite covering:
- Routing engine (Haversine, road factor, traffic)
- ML correction models
- Full ETA calculator
- API endpoints
"""

from datetime import datetime, timezone, timedelta
import math

import pytest

from src.eta_service.routing import (
    RoutingEngine,
    RoutePoint,
    Route,
    RouteSegment,
    TransportType,
    TRANSPORT_SPEEDS,
)
from src.eta_service.ml_corrector import (
    MLCorrector,
    RuleBasedCorrector,
    CorrectionFeatures,
    CorrectionResult,
)
from src.eta_service.calculator import (
    ETACalculator,
    ETARequest,
    ETAResult,
    ETABreakdown,
    ETAPhase,
)


# ============================================================================
# ROUTING ENGINE TESTS
# ============================================================================

class TestRoutingEngine:
    """Tests for the routing engine."""

    @pytest.fixture
    def engine(self):
        """Create routing engine."""
        return RoutingEngine()

    @pytest.fixture
    def moscow_center(self):
        """Moscow center point."""
        return RoutePoint(latitude=55.7558, longitude=37.6173, name="Red Square")

    @pytest.fixture
    def moscow_north(self):
        """Moscow north point."""
        return RoutePoint(latitude=55.8000, longitude=37.6000, name="North")

    def test_haversine_distance(self, engine):
        """Test Haversine distance calculation."""
        # Distance from Red Square to approximately 5km north
        distance = engine._haversine_distance(
            55.7558, 37.6173,  # Red Square
            55.8000, 37.6173,  # ~5km north
        )

        # Should be approximately 4.9km
        assert 4.5 <= distance <= 5.5

    def test_haversine_same_point(self, engine):
        """Test distance between same point is zero."""
        distance = engine._haversine_distance(
            55.7558, 37.6173,
            55.7558, 37.6173,
        )

        assert distance == 0.0

    def test_road_factor_applied(self, engine, moscow_center, moscow_north):
        """Test that road factor is applied to straight-line distance."""
        straight_distance = engine._haversine_distance(
            moscow_center.latitude, moscow_center.longitude,
            moscow_north.latitude, moscow_north.longitude,
        )

        route = engine.calculate_route(moscow_center, moscow_north)

        # Road distance should be greater than straight-line
        assert route.total_distance_km > straight_distance
        assert route.total_distance_km == pytest.approx(
            straight_distance * engine.ROAD_FACTOR, rel=0.01
        )

    def test_different_transport_speeds(self, engine, moscow_center, moscow_north):
        """Test that different transports have different durations."""
        foot_route = engine.calculate_route(
            moscow_center, moscow_north, TransportType.FOOT
        )
        bike_route = engine.calculate_route(
            moscow_center, moscow_north, TransportType.BIKE
        )
        car_route = engine.calculate_route(
            moscow_center, moscow_north, TransportType.CAR
        )

        # Foot should be slowest, car fastest
        assert foot_route.total_duration_minutes > bike_route.total_duration_minutes
        assert bike_route.total_duration_minutes > car_route.total_duration_minutes

        # Distances should be the same
        assert foot_route.total_distance_km == bike_route.total_distance_km

    def test_route_segments(self, engine, moscow_center, moscow_north):
        """Test route has correct segments."""
        route = engine.calculate_route(moscow_center, moscow_north)

        assert len(route.segments) == 1
        segment = route.segments[0]

        assert segment.start.latitude == moscow_center.latitude
        assert segment.end.latitude == moscow_north.latitude
        assert segment.distance_km > 0
        assert segment.duration_minutes > 0

    def test_route_with_waypoints(self, engine):
        """Test route with intermediate waypoints."""
        origin = RoutePoint(latitude=55.75, longitude=37.60)
        waypoint = RoutePoint(latitude=55.76, longitude=37.61)
        destination = RoutePoint(latitude=55.77, longitude=37.62)

        route = engine.calculate_route(
            origin, destination, waypoints=[waypoint]
        )

        assert len(route.segments) == 2
        assert route.waypoints == [waypoint]

    def test_traffic_factor_rush_hour(self, engine):
        """Test traffic factor during rush hour."""
        # Rush hour (8 AM on Monday)
        rush_time = datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc)
        factor = engine._get_traffic_factor(rush_time)

        assert factor < 1.0  # Should be slower

    def test_traffic_factor_night(self, engine):
        """Test traffic factor at night."""
        # Night (2 AM)
        night_time = datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc)
        factor = engine._get_traffic_factor(night_time)

        assert factor > 1.0  # Should be faster

    def test_traffic_factor_weekend(self, engine):
        """Test traffic factor on weekend."""
        # Saturday 10 AM
        weekend_time = datetime(2024, 1, 6, 10, 0, tzinfo=timezone.utc)
        factor = engine._get_traffic_factor(weekend_time)

        assert factor >= 1.0  # Should be same or faster

    def test_route_to_dict(self, engine, moscow_center, moscow_north):
        """Test route serialization."""
        route = engine.calculate_route(moscow_center, moscow_north)
        data = route.to_dict()

        assert "total_distance_km" in data
        assert "total_duration_minutes" in data
        assert "transport_type" in data
        assert "segments" in data
        assert len(data["segments"]) == 1


class TestRoutePoint:
    """Tests for RoutePoint."""

    def test_route_point_creation(self):
        """Test creating a route point."""
        point = RoutePoint(
            latitude=55.7558,
            longitude=37.6173,
            name="Test",
            h3_index="abc123",
        )

        assert point.latitude == 55.7558
        assert point.longitude == 37.6173
        assert point.name == "Test"

    def test_to_tuple(self):
        """Test conversion to tuple."""
        point = RoutePoint(latitude=55.7558, longitude=37.6173)
        assert point.to_tuple() == (55.7558, 37.6173)


# ============================================================================
# ML CORRECTOR TESTS
# ============================================================================

class TestCorrectionFeatures:
    """Tests for correction features."""

    @pytest.fixture
    def sample_features(self):
        """Create sample features."""
        return CorrectionFeatures(
            hour_of_day=12,
            day_of_week=1,
            is_weekend=False,
            is_holiday=False,
            is_lunch_rush=True,
            is_dinner_rush=False,
            origin_h3_index="abc123",
            destination_h3_index="def456",
            origin_zone_type="center",
            destination_zone_type="middle",
            restaurant_avg_prep_time=15.0,
            restaurant_current_queue=3,
            restaurant_rating=4.5,
            restaurant_order_complexity=0.5,
            courier_avg_speed=15.0,
            courier_experience_days=60,
            courier_current_load=1,
            courier_rating=4.8,
            courier_transport_type="bike",
            base_distance_km=3.5,
            base_duration_minutes=20.0,
            traffic_factor=0.8,
            weather_condition="clear",
            temperature_celsius=20.0,
            avg_delivery_time_this_hour=25.0,
            avg_delivery_time_this_route=22.0,
        )

    def test_to_array(self, sample_features):
        """Test conversion to numpy array."""
        arr = sample_features.to_array()

        assert len(arr) == 20
        assert arr[0] == 12  # hour_of_day

    def test_to_dict(self, sample_features):
        """Test conversion to dictionary."""
        data = sample_features.to_dict()

        assert data["hour_of_day"] == 12
        assert data["is_lunch_rush"] is True
        assert data["restaurant_prep_time"] == 15.0


class TestRuleBasedCorrector:
    """Tests for rule-based correction."""

    @pytest.fixture
    def corrector(self):
        """Create rule-based corrector."""
        return RuleBasedCorrector()

    @pytest.fixture
    def base_features(self):
        """Create base features for testing."""
        return CorrectionFeatures(
            hour_of_day=10,
            day_of_week=2,
            is_weekend=False,
            is_holiday=False,
            is_lunch_rush=False,
            is_dinner_rush=False,
            origin_h3_index="",
            destination_h3_index="",
            origin_zone_type="middle",
            destination_zone_type="middle",
            restaurant_avg_prep_time=15.0,
            restaurant_current_queue=2,
            restaurant_rating=4.0,
            restaurant_order_complexity=0.5,
            courier_avg_speed=15.0,
            courier_experience_days=60,
            courier_current_load=1,
            courier_rating=4.5,
            courier_transport_type="bike",
            base_distance_km=3.0,
            base_duration_minutes=15.0,
            traffic_factor=1.0,
            weather_condition="clear",
            temperature_celsius=20.0,
            avg_delivery_time_this_hour=25.0,
            avg_delivery_time_this_route=22.0,
        )

    def test_base_correction(self, corrector, base_features):
        """Test correction with neutral conditions."""
        result = corrector.predict(base_features, 20.0)

        assert isinstance(result, CorrectionResult)
        assert result.corrected_eta_minutes > 0
        assert 0 < result.confidence <= 1

    def test_rush_hour_penalty(self, corrector, base_features):
        """Test that rush hour adds penalty."""
        base_features.is_lunch_rush = True

        result = corrector.predict(base_features, 20.0)

        assert "rush_hour_penalty" in result.factors
        assert result.factors["rush_hour_penalty"] > 0

    def test_weather_penalty(self, corrector, base_features):
        """Test weather penalty."""
        base_features.weather_condition = "rain"

        result = corrector.predict(base_features, 20.0)

        assert "weather_penalty" in result.factors
        assert result.factors["weather_penalty"] > 0

    def test_heavy_weather_penalty(self, corrector, base_features):
        """Test heavy weather has larger penalty."""
        base_features.weather_condition = "snow"
        result_snow = corrector.predict(base_features, 20.0)

        base_features.weather_condition = "rain"
        result_rain = corrector.predict(base_features, 20.0)

        assert result_snow.factors["weather_penalty"] > result_rain.factors["weather_penalty"]

    def test_queue_penalty(self, corrector, base_features):
        """Test restaurant queue penalty."""
        base_features.restaurant_current_queue = 10

        result = corrector.predict(base_features, 20.0)

        # Should have higher correction due to queue
        assert result.correction_minutes > 0

    def test_rookie_courier_penalty(self, corrector, base_features):
        """Test penalty for inexperienced courier."""
        base_features.courier_experience_days = 5

        result = corrector.predict(base_features, 20.0)

        assert "rookie_penalty" in result.factors
        assert result.factors["rookie_penalty"] > 0

    def test_veteran_courier_bonus(self, corrector, base_features):
        """Test bonus for experienced courier."""
        base_features.courier_experience_days = 365

        result = corrector.predict(base_features, 20.0)

        assert "veteran_bonus" in result.factors
        assert result.factors["veteran_bonus"] < 0  # Negative = faster

    def test_courier_load_penalty(self, corrector, base_features):
        """Test penalty for courier with multiple orders."""
        base_features.courier_current_load = 3

        result = corrector.predict(base_features, 20.0)

        assert "load_penalty" in result.factors
        assert result.factors["load_penalty"] > 0

    def test_minimum_eta(self, corrector, base_features):
        """Test that ETA doesn't go below minimum."""
        # Very short base ETA with lots of negative corrections
        base_features.courier_experience_days = 365

        result = corrector.predict(base_features, 3.0)

        assert result.corrected_eta_minutes >= 5.0

    def test_confidence_interval(self, corrector, base_features):
        """Test confidence interval calculation."""
        result = corrector.predict(base_features, 20.0)

        assert result.lower_bound_minutes < result.corrected_eta_minutes
        assert result.upper_bound_minutes > result.corrected_eta_minutes


class TestMLCorrector:
    """Tests for main ML corrector interface."""

    def test_corrector_creation(self):
        """Test creating ML corrector."""
        corrector = MLCorrector(model_type="rule_based")

        assert "rule_based" in corrector._models

    def test_list_models(self):
        """Test listing available models."""
        corrector = MLCorrector()
        models = corrector.list_models()

        assert len(models) >= 1
        assert all("name" in m for m in models)
        assert all("version" in m for m in models)


# ============================================================================
# ETA CALCULATOR TESTS
# ============================================================================

class TestETACalculator:
    """Tests for main ETA calculator."""

    @pytest.fixture
    def calculator(self):
        """Create ETA calculator."""
        return ETACalculator(use_ml_correction=True)

    @pytest.fixture
    def sample_request(self):
        """Create sample ETA request."""
        return ETARequest(
            courier_lat=55.7500,
            courier_lon=37.6000,
            restaurant_lat=55.7558,
            restaurant_lon=37.6173,
            customer_lat=55.7600,
            customer_lon=37.6200,
            transport_type=TransportType.BIKE,
            courier_avg_speed=15.0,
            courier_experience_days=60,
            courier_current_load=1,
            courier_rating=4.5,
            restaurant_avg_prep_time=15.0,
            restaurant_current_queue=2,
            restaurant_rating=4.0,
            order_complexity=0.5,
            weather_condition="clear",
            temperature_celsius=20.0,
        )

    def test_full_eta_calculation(self, calculator, sample_request):
        """Test complete ETA calculation."""
        result = calculator.calculate(sample_request)

        assert isinstance(result, ETAResult)
        assert result.total_duration_minutes > 0
        assert result.total_distance_km > 0
        assert result.confidence > 0

    def test_eta_breakdown(self, calculator, sample_request):
        """Test ETA breakdown by phase."""
        result = calculator.calculate(sample_request)

        assert len(result.breakdown) == 4  # 3 phases + total

        phases = [b.phase for b in result.breakdown]
        assert ETAPhase.COURIER_TO_RESTAURANT in phases
        assert ETAPhase.RESTAURANT_PREPARATION in phases
        assert ETAPhase.RESTAURANT_TO_CUSTOMER in phases
        assert ETAPhase.TOTAL in phases

    def test_confidence_interval(self, calculator, sample_request):
        """Test confidence interval in result."""
        result = calculator.calculate(sample_request)

        assert result.lower_bound_minutes < result.total_duration_minutes
        assert result.upper_bound_minutes > result.total_duration_minutes
        assert result.earliest_delivery < result.estimated_delivery_time
        assert result.latest_delivery > result.estimated_delivery_time

    def test_calculation_time_tracked(self, calculator, sample_request):
        """Test that calculation time is tracked."""
        result = calculator.calculate(sample_request)

        assert result.calculation_time_ms > 0
        assert result.calculation_time_ms < 1000  # Should be fast

    def test_simple_eta(self, calculator):
        """Test simple point-to-point ETA."""
        result = calculator.calculate_simple(
            origin_lat=55.7558,
            origin_lon=37.6173,
            destination_lat=55.7700,
            destination_lon=37.6300,
        )

        assert "distance_km" in result
        assert "duration_minutes" in result
        assert result["distance_km"] > 0
        assert result["duration_minutes"] > 0

    def test_different_transports(self, calculator, sample_request):
        """Test ETA varies by transport type."""
        sample_request.transport_type = TransportType.FOOT
        result_foot = calculator.calculate(sample_request)

        sample_request.transport_type = TransportType.CAR
        result_car = calculator.calculate(sample_request)

        # Walking should take longer
        assert result_foot.total_duration_minutes > result_car.total_duration_minutes

    def test_result_to_dict(self, calculator, sample_request):
        """Test result serialization."""
        result = calculator.calculate(sample_request)
        data = result.to_dict()

        assert "estimated_delivery_time" in data
        assert "total_duration_minutes" in data
        assert "confidence_interval" in data
        assert "breakdown" in data

    def test_get_phase(self, calculator, sample_request):
        """Test getting specific phase."""
        result = calculator.calculate(sample_request)

        pickup_phase = result.get_phase(ETAPhase.COURIER_TO_RESTAURANT)
        assert pickup_phase is not None
        assert pickup_phase.duration_minutes > 0

        unknown_phase = result.get_phase(ETAPhase.TOTAL)
        assert unknown_phase is not None

    def test_models_info(self, calculator):
        """Test getting model info."""
        models = calculator.get_models_info()

        assert len(models) >= 1


class TestETABreakdown:
    """Tests for ETA breakdown."""

    def test_breakdown_creation(self):
        """Test creating breakdown."""
        breakdown = ETABreakdown(
            phase=ETAPhase.COURIER_TO_RESTAURANT,
            duration_minutes=10.0,
            distance_km=2.5,
            confidence=0.85,
        )

        assert breakdown.phase == ETAPhase.COURIER_TO_RESTAURANT
        assert breakdown.duration_minutes == 10.0

    def test_breakdown_to_dict(self):
        """Test breakdown serialization."""
        breakdown = ETABreakdown(
            phase=ETAPhase.RESTAURANT_PREPARATION,
            duration_minutes=15.0,
            distance_km=0.0,
            confidence=0.75,
            details={"queue_size": 3},
        )

        data = breakdown.to_dict()

        assert data["phase"] == "restaurant_preparation"
        assert data["duration_minutes"] == 15.0
        assert data["details"]["queue_size"] == 3


# ============================================================================
# API TESTS
# ============================================================================

class TestETAAPI:
    """Tests for ETA API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)

    def test_simple_eta_endpoint(self, client):
        """Test simple ETA endpoint."""
        response = client.post(
            "/api/v1/eta/simple",
            json={
                "origin": {"latitude": 55.7558, "longitude": 37.6173},
                "destination": {"latitude": 55.7700, "longitude": 37.6300},
                "transport_type": "bike",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "distance_km" in data
        assert "duration_minutes" in data

    def test_full_eta_endpoint(self, client):
        """Test full ETA endpoint."""
        response = client.post(
            "/api/v1/eta/calculate",
            json={
                "courier_location": {"latitude": 55.7500, "longitude": 37.6000},
                "restaurant_location": {"latitude": 55.7558, "longitude": 37.6173},
                "customer_location": {"latitude": 55.7600, "longitude": 37.6200},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "estimated_delivery_time" in data
        assert "breakdown" in data

    def test_batch_eta_endpoint(self, client):
        """Test batch ETA endpoint."""
        response = client.post(
            "/api/v1/eta/batch",
            json={
                "requests": [
                    {
                        "origin": {"latitude": 55.75, "longitude": 37.60},
                        "destination": {"latitude": 55.76, "longitude": 37.61},
                        "transport_type": "bike",
                    },
                    {
                        "origin": {"latitude": 55.77, "longitude": 37.62},
                        "destination": {"latitude": 55.78, "longitude": 37.63},
                        "transport_type": "car",
                    },
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_calculations"] == 2
        assert len(data["results"]) == 2

    def test_transport_types_endpoint(self, client):
        """Test transport types listing."""
        response = client.get("/api/v1/eta/transport-types")

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 3
        assert any(t["type"] == "bike" for t in data)

    def test_models_endpoint(self, client):
        """Test models listing."""
        response = client.get("/api/v1/eta/models")

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1

    def test_health_endpoint(self, client):
        """Test ETA health check."""
        response = client.get("/api/v1/eta/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data

