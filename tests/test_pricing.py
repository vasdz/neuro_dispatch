"""
Tests for Dynamic Pricing Service (Phase 3).

Comprehensive test suite covering:
- Pricing strategies (rule-based, ML, hybrid, time-decay)
- Market state aggregation
- A/B testing engine
- Fairness constraints
- API endpoints
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.pricing_service.strategies.base import (
    BasePricingStrategy,
    PricingContext,
    PricingResult,
    MarketState,
    StrategyType,
    DemandLevel,
    SupplyLevel,
)
from src.pricing_service.strategies.rule_based import RuleBasedStrategy, ZoneBasedStrategy
from src.pricing_service.strategies.ml_based import (
    MLBasedStrategy,
    PricingFeatureExtractor,
    ElasticityAwareStrategy,
)
from src.pricing_service.strategies.hybrid import HybridStrategy, AdaptiveHybridStrategy
from src.pricing_service.strategies.time_decay import TimeDecayStrategy, UrgentOrderStrategy
from src.pricing_service.ab_testing import (
    ABTestingEngine,
    Experiment,
    ExperimentVariant,
    VariantAssigner,
    ExperimentStatus,
)


# ============== Test Fixtures ==============

@pytest.fixture
def sample_market_state():
    """Create a sample market state for testing."""
    # Use a fixed neutral time (10:00 UTC) where time modifier is 1.0
    neutral_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    return MarketState(
        h3_index="882830829bfffff",
        timestamp=neutral_time,
        pending_orders=5,
        available_couriers=3,
        busy_couriers=7,
        avg_orders_this_hour=4.0,
        avg_orders_same_hour_last_week=4.5,
        predicted_demand_1h=6.0,
        predicted_demand_confidence=0.85,
        weather_multiplier=1.0,
    )


@pytest.fixture
def sample_context(sample_market_state):
    """Create a sample pricing context."""
    return PricingContext(
        market_state=sample_market_state,
        base_price=350.0,
        order_id="order_123",
        customer_id="customer_456",
        customer_order_count=15,
        customer_avg_order_value=500.0,
        customer_is_premium=False,
        restaurant_prep_time_min=25,
        restaurant_rating=4.3,
    )


# ============== Market State Tests ==============

class TestMarketState:
    """Tests for MarketState dataclass."""

    def test_demand_supply_ratio(self, sample_market_state):
        """Test demand/supply ratio calculation."""
        ratio = sample_market_state.demand_supply_ratio
        expected = 5 / 3  # 5 orders / 3 couriers
        assert abs(ratio - expected) < 0.01

    def test_demand_supply_ratio_zero_couriers(self):
        """Test ratio with no couriers."""
        state = MarketState(
            h3_index="test",
            timestamp=datetime.now(timezone.utc),
            pending_orders=5,
            available_couriers=0,
        )
        assert state.demand_supply_ratio == float('inf')

    def test_courier_utilization(self, sample_market_state):
        """Test courier utilization calculation."""
        util = sample_market_state.courier_utilization
        expected = 7 / 10  # 7 busy / 10 total
        assert abs(util - expected) < 0.01

    def test_demand_trend(self, sample_market_state):
        """Test demand trend calculation."""
        trend = sample_market_state.demand_trend
        expected = (5 - 4.5) / 4.5  # (current - historical) / historical
        assert abs(trend - expected) < 0.01


# ============== Rule-Based Strategy Tests ==============

class TestRuleBasedStrategy:
    """Tests for rule-based pricing strategy."""

    @pytest.mark.asyncio
    async def test_balanced_market(self, sample_context):
        """Test pricing in balanced market."""
        # Set balanced state (ratio ~1)
        sample_context.market_state.pending_orders = 3
        sample_context.market_state.available_couriers = 3

        strategy = RuleBasedStrategy()
        result = await strategy.calculate(sample_context)

        # Should have mild surge
        assert 1.0 <= result.surge_coefficient <= 1.3
        assert result.strategy_used == StrategyType.RULE_BASED

    @pytest.mark.asyncio
    async def test_high_demand(self, sample_context):
        """Test pricing with high demand."""
        sample_context.market_state.pending_orders = 15
        sample_context.market_state.available_couriers = 3

        strategy = RuleBasedStrategy()
        result = await strategy.calculate(sample_context)

        # Should have significant surge
        assert result.surge_coefficient > 1.5
        assert result.demand_level in [DemandLevel.HIGH, DemandLevel.EXTREME]

    @pytest.mark.asyncio
    async def test_no_couriers(self, sample_context):
        """Test pricing with no available couriers."""
        sample_context.market_state.pending_orders = 5
        sample_context.market_state.available_couriers = 0
        sample_context.customer_order_count = 0  # No loyalty discount

        strategy = RuleBasedStrategy()
        result = await strategy.calculate(sample_context)

        # Should hit max surge (or very close due to rounding)
        assert result.surge_coefficient >= strategy.MAX_SURGE * 0.99
        assert result.supply_level == SupplyLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_no_demand(self, sample_context):
        """Test pricing with no demand."""
        sample_context.market_state.pending_orders = 0
        sample_context.market_state.available_couriers = 10

        strategy = RuleBasedStrategy()
        result = await strategy.calculate(sample_context)

        # Should be at minimum surge
        assert result.surge_coefficient == strategy.MIN_SURGE
        assert result.demand_level == DemandLevel.LOW

    @pytest.mark.asyncio
    async def test_final_price_calculation(self, sample_context):
        """Test that final price is calculated correctly."""
        strategy = RuleBasedStrategy()
        result = await strategy.calculate(sample_context)

        expected_price = sample_context.base_price * result.surge_coefficient * (1 - result.loyalty_discount)
        # Allow for rounding differences
        assert abs(result.final_price - expected_price) < 2.0

    def test_surge_curve_interpolation(self):
        """Test surge curve interpolation logic."""
        strategy = RuleBasedStrategy()

        # Test at exact breakpoints
        assert strategy._interpolate_surge(0.0) == 1.0
        assert strategy._interpolate_surge(0.5) == 1.0

        # Test interpolation between points
        surge_at_0_75 = strategy._interpolate_surge(0.75)
        assert 1.0 < surge_at_0_75 < 1.15

        # Test high ratio
        surge_at_5 = strategy._interpolate_surge(5.0)
        assert surge_at_5 > 2.0


class TestZoneBasedStrategy:
    """Tests for zone-based pricing strategy."""

    @pytest.mark.asyncio
    async def test_center_zone_adjustment(self, sample_context):
        """Test that center zone has higher base adjustment."""
        center_strategy = ZoneBasedStrategy(zone="center")
        outer_strategy = ZoneBasedStrategy(zone="outer")

        result_center = await center_strategy.calculate(sample_context)
        result_outer = await outer_strategy.calculate(sample_context)

        # Center should have higher surge (premium area)
        assert result_center.surge_coefficient >= result_outer.surge_coefficient

    @pytest.mark.asyncio
    async def test_zone_max_surge(self, sample_context):
        """Test zone-specific max surge limits."""
        # Set extreme demand
        sample_context.market_state.pending_orders = 50
        sample_context.market_state.available_couriers = 1

        center_strategy = ZoneBasedStrategy(zone="center")
        result = await center_strategy.calculate(sample_context)

        # Should respect zone-specific max
        assert result.surge_coefficient <= center_strategy.MAX_SURGE


# ============== ML-Based Strategy Tests ==============

class TestPricingFeatureExtractor:
    """Tests for feature extraction."""

    def test_feature_extraction(self, sample_context):
        """Test that all features are extracted."""
        extractor = PricingFeatureExtractor()
        features = extractor.extract(sample_context)

        # Check essential features exist
        assert "pending_orders" in features
        assert "available_couriers" in features
        assert "demand_supply_ratio" in features
        assert "hour" in features
        assert "is_weekend" in features

    def test_feature_array_conversion(self, sample_context):
        """Test conversion to numpy array."""
        extractor = PricingFeatureExtractor()
        features = extractor.extract(sample_context)
        array = extractor.to_array(features)

        assert isinstance(array, np.ndarray)
        assert len(array) == len(extractor.FEATURE_NAMES)


class TestMLBasedStrategy:
    """Tests for ML-based pricing strategy."""

    @pytest.mark.asyncio
    async def test_fallback_to_rules(self, sample_context):
        """Test fallback when ML model not loaded."""
        strategy = MLBasedStrategy()
        # Model won't be loaded in test environment

        result = await strategy.calculate(sample_context)

        # Should fall back to rule-based
        assert result.strategy_used == StrategyType.RULE_BASED
        assert "[Fallback]" in result.reason


class TestElasticityAwareStrategy:
    """Tests for elasticity-aware pricing."""

    def test_customer_segment_detection(self, sample_context):
        """Test customer segment detection."""
        strategy = ElasticityAwareStrategy()

        # Premium customer
        sample_context.customer_is_premium = True
        assert strategy._estimate_customer_segment(sample_context) == "premium"

        # High spender
        sample_context.customer_is_premium = False
        sample_context.customer_avg_order_value = 1000.0
        assert strategy._estimate_customer_segment(sample_context) == "premium"

        # Low spender
        sample_context.customer_avg_order_value = 200.0
        assert strategy._estimate_customer_segment(sample_context) == "price_sensitive"

        # Regular
        sample_context.customer_avg_order_value = 500.0
        assert strategy._estimate_customer_segment(sample_context) == "regular"


# ============== Hybrid Strategy Tests ==============

class TestHybridStrategy:
    """Tests for hybrid pricing strategy."""

    @pytest.mark.asyncio
    async def test_combines_strategies(self, sample_context):
        """Test that hybrid combines rule and ML strategies."""
        strategy = HybridStrategy(rule_weight=0.5, ml_weight=0.5)
        result = await strategy.calculate(sample_context)

        # Should use hybrid
        assert result.strategy_used == StrategyType.HYBRID

        # Should have combined factors
        assert "rule_surge" in result.factors
        assert "ml_surge" in result.factors
        assert "rule_weight" in result.factors
        assert "ml_weight" in result.factors

    def test_weight_validation(self):
        """Test that weights must sum to 1."""
        with pytest.raises(ValueError):
            HybridStrategy(rule_weight=0.5, ml_weight=0.3)  # Doesn't sum to 1

    @pytest.mark.asyncio
    async def test_confidence_weighting(self, sample_context):
        """Test confidence-based weight adjustment."""
        strategy = HybridStrategy(use_confidence_weighting=True)
        result = await strategy.calculate(sample_context)

        # Weights should be adjusted based on ML confidence
        assert "rule_weight" in result.factors
        assert "ml_weight" in result.factors


# ============== Time Decay Strategy Tests ==============

class TestTimeDecayStrategy:
    """Tests for time-decay pricing strategy."""

    @pytest.mark.asyncio
    async def test_no_decay_for_new_orders(self, sample_context):
        """Test that new orders have no time decay."""
        sample_context.order_created_at = datetime.now(timezone.utc) - timedelta(seconds=60)

        strategy = TimeDecayStrategy()
        result = await strategy.calculate(sample_context)

        # Should have minimal time modifier
        assert result.time_modifier == 1.0

    @pytest.mark.asyncio
    async def test_decay_for_old_orders(self, sample_context):
        """Test that old orders get higher courier incentive."""
        sample_context.order_created_at = datetime.now(timezone.utc) - timedelta(minutes=20)

        strategy = TimeDecayStrategy()
        result = await strategy.calculate(sample_context)

        # Should have significant time modifier
        assert result.time_modifier > 1.3
        assert "courier_multiplier" in result.factors
        assert result.factors["courier_multiplier"] > 1.3

    @pytest.mark.asyncio
    async def test_customer_discount_for_delays(self, sample_context):
        """Test customer discount for long delays."""
        sample_context.order_created_at = datetime.now(timezone.utc) - timedelta(minutes=25)

        strategy = TimeDecayStrategy()
        result = await strategy.calculate(sample_context)

        # Should have customer discount
        assert result.factors.get("customer_discount", 0) > 0

    def test_time_multiplier_calculation(self):
        """Test time multiplier calculation."""
        strategy = TimeDecayStrategy()

        # 5 minutes - no change
        assert strategy._calculate_time_multiplier(300) == 1.0

        # 20 minutes - should be elevated
        mult_20 = strategy._calculate_time_multiplier(1200)
        assert mult_20 > 1.3

        # 40 minutes - should be at max
        mult_40 = strategy._calculate_time_multiplier(2400)
        assert mult_40 == 2.0


class TestUrgentOrderStrategy:
    """Tests for urgent order pricing."""

    @pytest.mark.asyncio
    async def test_urgent_starts_higher(self, sample_context):
        """Test that urgent orders start with higher multiplier."""
        # Set order to be 5 minutes old
        sample_context.order_created_at = datetime.now(timezone.utc) - timedelta(minutes=5)

        regular_strategy = TimeDecayStrategy()
        urgent_strategy = UrgentOrderStrategy()

        regular_result = await regular_strategy.calculate(sample_context)
        urgent_result = await urgent_strategy.calculate(sample_context)

        # Urgent should have higher courier payout (check factors instead of time_modifier)
        assert urgent_result.factors.get("courier_multiplier", 1.0) >= regular_result.factors.get("courier_multiplier", 1.0)
        assert "[URGENT]" in urgent_result.reason


# ============== A/B Testing Engine Tests ==============

class TestVariantAssigner:
    """Tests for variant assignment."""

    def test_deterministic_assignment(self):
        """Test that same identifier gets same variant."""
        strategy1 = RuleBasedStrategy()
        strategy2 = HybridStrategy()

        experiment = Experiment(
            id="exp_test",
            name="Test Experiment",
            description="Test",
            variants=[
                ExperimentVariant("control", strategy1, 50.0),
                ExperimentVariant("treatment", strategy2, 50.0),
            ],
            status=ExperimentStatus.RUNNING,
        )

        # Same ID should always get same variant
        variant1 = VariantAssigner.assign(experiment, "customer_123")
        variant2 = VariantAssigner.assign(experiment, "customer_123")

        assert variant1.name == variant2.name

    def test_traffic_distribution(self):
        """Test that traffic is roughly distributed according to weights."""
        strategy1 = RuleBasedStrategy()
        strategy2 = HybridStrategy()

        experiment = Experiment(
            id="exp_dist",
            name="Distribution Test",
            description="Test",
            variants=[
                ExperimentVariant("control", strategy1, 70.0),
                ExperimentVariant("treatment", strategy2, 30.0),
            ],
            status=ExperimentStatus.RUNNING,
        )

        # Assign many identifiers
        control_count = 0
        treatment_count = 0

        for i in range(1000):
            variant = VariantAssigner.assign(experiment, f"id_{i}")
            if variant.name == "control":
                control_count += 1
            else:
                treatment_count += 1

        # Should be roughly 70/30 (with some tolerance)
        control_ratio = control_count / 1000
        assert 0.6 < control_ratio < 0.8


class TestABTestingEngine:
    """Tests for A/B testing engine."""

    def test_create_experiment(self):
        """Test experiment creation."""
        engine = ABTestingEngine()

        experiment = engine.create_experiment(
            name="Test Experiment",
            variants=[
                ("control", RuleBasedStrategy(), 50.0),
                ("treatment", HybridStrategy(), 50.0),
            ],
        )

        assert experiment.name == "Test Experiment"
        assert len(experiment.variants) == 2
        assert experiment.status == ExperimentStatus.DRAFT

    def test_start_experiment(self):
        """Test starting experiment."""
        engine = ABTestingEngine()

        experiment = engine.create_experiment(
            name="Test",
            variants=[
                ("A", RuleBasedStrategy(), 50.0),
                ("B", HybridStrategy(), 50.0),
            ],
        )

        success = engine.start_experiment(experiment.id)
        assert success
        assert experiment.status == ExperimentStatus.RUNNING
        assert experiment.started_at is not None

    def test_stop_experiment(self):
        """Test stopping experiment."""
        engine = ABTestingEngine()

        experiment = engine.create_experiment(
            name="Test",
            variants=[
                ("A", RuleBasedStrategy(), 50.0),
                ("B", HybridStrategy(), 50.0),
            ],
        )

        engine.start_experiment(experiment.id)
        success = engine.stop_experiment(experiment.id)

        assert success
        assert experiment.status == ExperimentStatus.COMPLETED
        assert experiment.ended_at is not None

    def test_get_results(self):
        """Test getting experiment results."""
        engine = ABTestingEngine()

        experiment = engine.create_experiment(
            name="Test",
            variants=[
                ("control", RuleBasedStrategy(), 50.0),
                ("treatment", HybridStrategy(), 50.0),
            ],
        )

        # Simulate some data
        experiment.variants[0].impressions = 100
        experiment.variants[0].conversions = 80
        experiment.variants[0].total_revenue = 40000

        experiment.variants[1].impressions = 100
        experiment.variants[1].conversions = 85
        experiment.variants[1].total_revenue = 45000

        results = engine.get_experiment_results(experiment.id)

        assert results is not None
        assert results["total_impressions"] == 200
        assert len(results["variants"]) == 2


# ============== Base Strategy Tests ==============

class TestBaseStrategy:
    """Tests for base strategy functionality."""

    def test_apply_constraints(self, sample_context):
        """Test surge constraint application."""
        strategy = RuleBasedStrategy()

        # Test capping high surge
        surge, was_capped = strategy.apply_constraints(5.0, sample_context)
        assert surge == strategy.MAX_SURGE
        assert was_capped

        # Test capping low surge
        surge, was_capped = strategy.apply_constraints(0.5, sample_context)
        assert surge == strategy.MIN_SURGE
        assert was_capped

        # Test no capping
        surge, was_capped = strategy.apply_constraints(1.5, sample_context)
        assert surge == 1.5
        assert not was_capped

    def test_classify_demand(self):
        """Test demand classification."""
        strategy = RuleBasedStrategy()

        assert strategy.classify_demand(1) == DemandLevel.LOW
        assert strategy.classify_demand(3) == DemandLevel.NORMAL
        assert strategy.classify_demand(8) == DemandLevel.ELEVATED
        assert strategy.classify_demand(15) == DemandLevel.HIGH
        assert strategy.classify_demand(25) == DemandLevel.EXTREME

    def test_classify_supply(self):
        """Test supply classification."""
        strategy = RuleBasedStrategy()

        assert strategy.classify_supply(0) == SupplyLevel.CRITICAL
        assert strategy.classify_supply(1) == SupplyLevel.LOW
        assert strategy.classify_supply(4) == SupplyLevel.NORMAL
        assert strategy.classify_supply(8) == SupplyLevel.ADEQUATE
        assert strategy.classify_supply(15) == SupplyLevel.SURPLUS

    def test_loyalty_discount(self, sample_context):
        """Test loyalty discount calculation."""
        strategy = RuleBasedStrategy()

        # Regular customer
        sample_context.customer_order_count = 5
        sample_context.customer_is_premium = False
        assert strategy.calculate_loyalty_discount(sample_context) == 0.0

        # Frequent customer (10+ orders)
        sample_context.customer_order_count = 12
        assert strategy.calculate_loyalty_discount(sample_context) == 0.02

        # Very frequent customer (50+ orders)
        sample_context.customer_order_count = 55
        assert strategy.calculate_loyalty_discount(sample_context) == 0.05

        # Premium customer
        sample_context.customer_is_premium = True
        assert strategy.calculate_loyalty_discount(sample_context) == 0.10


# ============== Integration Tests ==============

class TestPricingIntegration:
    """Integration tests for pricing service."""

    @pytest.mark.asyncio
    async def test_full_pricing_flow(self, sample_context):
        """Test complete pricing calculation flow."""
        strategy = HybridStrategy()
        result = await strategy.calculate(sample_context)

        # Verify result structure
        assert isinstance(result, PricingResult)
        assert result.surge_coefficient >= 1.0
        assert result.final_price > 0
        assert result.calculated_at is not None
        assert len(result.factors) > 0

    @pytest.mark.asyncio
    async def test_strategy_consistency(self, sample_context):
        """Test that same input gives same output."""
        strategy = RuleBasedStrategy()

        result1 = await strategy.calculate(sample_context)
        result2 = await strategy.calculate(sample_context)

        # Should be deterministic
        assert result1.surge_coefficient == result2.surge_coefficient

