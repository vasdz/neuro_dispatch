"""
Dynamic Pricing Service - Surge pricing and price balancing.

Senior+ level implementation with:
- Multiple pricing strategies (rule-based, ML, hybrid, time-decay)
- A/B testing infrastructure for strategy comparison
- Market state aggregation and analysis
- Fairness constraints and rate limiting
- Customer loyalty discounts
- Elasticity-aware pricing

Components:
- strategies/: Pricing strategy implementations
- market_state: Supply/demand aggregation
- ab_testing: A/B experimentation engine
- strategy: Main pricing engine facade
- api: FastAPI endpoints

Usage:
    from src.pricing_service import PricingEngine
    from src.pricing_service.strategies import RuleBasedStrategy, HybridStrategy

    engine = PricingEngine(session)
    result = await engine.calculate_price(h3_index, base_price=350.0)
"""

from src.pricing_service.strategy import PricingEngine, PricingEngineFactory
from src.pricing_service.api import router as pricing_router

__all__ = [
    "PricingEngine",
    "PricingEngineFactory",
    "pricing_router",
]
