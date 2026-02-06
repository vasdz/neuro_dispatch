"""
Dispatch Algorithms Module.

Provides multiple order-to-courier assignment algorithms:
- GreedyAssigner: Fast nearest-courier assignment
- HungarianAssigner: Optimal bipartite matching (Kuhn-Munkres)
- BatchAssigner: Batch optimization with constraints
- RebalancingAssigner: Considers courier repositioning

Senior+ implementation with:
- Algorithm interface (Strategy Pattern)
- Performance optimization
- Constraint handling
- Multi-objective optimization
"""

from src.dispatch_engine.algorithms.base import (
    BaseAssigner,
    AssignmentProblem,
    AssignmentSolution,
    Assignment,
    AssignmentConstraints,
    OptimizationObjective,
)
from src.dispatch_engine.algorithms.greedy import GreedyAssigner
from src.dispatch_engine.algorithms.hungarian import HungarianAssigner
from src.dispatch_engine.algorithms.batch import BatchAssigner
from src.dispatch_engine.algorithms.cost_matrix import CostMatrixBuilder

__all__ = [
    "BaseAssigner",
    "AssignmentProblem",
    "AssignmentSolution",
    "Assignment",
    "AssignmentConstraints",
    "OptimizationObjective",
    "GreedyAssigner",
    "HungarianAssigner",
    "BatchAssigner",
    "CostMatrixBuilder",
]

