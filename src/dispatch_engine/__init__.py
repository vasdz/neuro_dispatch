"""
Dispatch Engine - Order assignment and routing service.

Senior+ implementation with:
- Multiple assignment algorithms (Greedy, Hungarian, Auction, Batch)
- Cost matrix optimization with constraint handling
- Multi-objective optimization support
- Performance tracking and monitoring
- Integration with pricing and demand forecast

Components:
- algorithms/: Assignment algorithm implementations
- solver: Main DispatchSolver orchestrator
- api: FastAPI endpoints

Usage:
    from src.dispatch_engine import DispatchSolver, DispatcherFactory

    solver = DispatchSolver(session)
    solution = await solver.solve(algorithm="hungarian")
"""

from src.dispatch_engine.solver import DispatchSolver, DispatcherFactory
from src.dispatch_engine.api import router as dispatch_router

__all__ = [
    "DispatchSolver",
    "DispatcherFactory",
    "dispatch_router",
]
