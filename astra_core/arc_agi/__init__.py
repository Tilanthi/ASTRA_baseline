#!/usr/bin/env python3
"""
ARC-AGI Solver Package (Enhanced v2.0)

A comprehensive solver for the ARC-AGI benchmark using:
- Grid DSL with transformation primitives
- Hypothesis generation and testing (40+ generators)
- Compositional pattern library
- Systematic search with beam search and pruning
- Deep program synthesis (depth 5)
- Neural pattern recognition with embeddings
- Iterative refinement with error correction
- Analogical transfer from solved tasks
"""


def _degraded_warn(_module: str, _exc: BaseException) -> None:
    """Log why an optional import degraded instead of failing silently."""
    import logging
    logging.getLogger(__name__).warning(
        "%s unavailable (%s: %s) - dependent names set to None",
        _module, type(_exc).__name__, _exc,
    )


try:
    from .grid_dsl import (
    Grid, GridObject, BoundingBox,
    Color, Direction, Symmetry,
    empty_grid, from_objects
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".grid_dsl", _exc)
    Grid = GridObject = BoundingBox = Color = Direction = Symmetry = empty_grid = from_objects = None  # degraded: unavailable

try:
    from .hypothesis_engine import (
    TransformationHypothesis, TransformationType,
    HypothesisGenerator, HypothesisTester
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".hypothesis_engine", _exc)
    TransformationHypothesis = TransformationType = HypothesisGenerator = HypothesisTester = None  # degraded: unavailable

try:
    from .pattern_library import (
    Pattern, PatternType,
    PatternDetector, PatternPrimitives,
    ObjectRelationships, CompositeTransform
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".pattern_library", _exc)
    Pattern = PatternType = PatternDetector = PatternPrimitives = ObjectRelationships = CompositeTransform = None  # degraded: unavailable

try:
    from .systematic_search import (
    SearchState, TaskAnalysis,
    ConstraintPropagator, ProgramSynthesizer,
    BeamSearchSolver, AnalogicalTransfer,
    ARCSolver
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".systematic_search", _exc)
    SearchState = TaskAnalysis = ConstraintPropagator = ProgramSynthesizer = BeamSearchSolver = AnalogicalTransfer = ARCSolver = None  # degraded: unavailable

try:
    from .extended_generators import ExtendedGenerators
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".extended_generators", _exc)
    ExtendedGenerators = None  # degraded: unavailable

try:
    from .deep_synthesis import (
    DeepProgramSynthesizer, EnumerativeSynthesizer,
    ProgramNode, TypedPrimitive
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".deep_synthesis", _exc)
    DeepProgramSynthesizer = EnumerativeSynthesizer = ProgramNode = TypedPrimitive = None  # degraded: unavailable

try:
    from .neural_patterns import (
    GridEmbedding, GridEncoder,
    TransformationEmbedding, PatternMatcher,
    PatternCluster, TransformationPrioritizer
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".neural_patterns", _exc)
    GridEmbedding = GridEncoder = TransformationEmbedding = PatternMatcher = PatternCluster = TransformationPrioritizer = None  # degraded: unavailable

try:
    from .iterative_refinement import (
    SolutionAttempt, ErrorAnalysis,
    ErrorAnalyzer, SolutionRefiner,
    IterativeRefinementSolver, HypothesisCombiner,
    ConstraintBasedRepair
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".iterative_refinement", _exc)
    SolutionAttempt = ErrorAnalysis = ErrorAnalyzer = SolutionRefiner = IterativeRefinementSolver = HypothesisCombiner = ConstraintBasedRepair = None  # degraded: unavailable

try:
    from .enhanced_solver import EnhancedARCSolver, SolveResult
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".enhanced_solver", _exc)
    EnhancedARCSolver = SolveResult = None  # degraded: unavailable

__all__ = [
    # Grid DSL
    'Grid', 'GridObject', 'BoundingBox',
    'Color', 'Direction', 'Symmetry',
    'empty_grid', 'from_objects',

    # Hypothesis Engine
    'TransformationHypothesis', 'TransformationType',
    'HypothesisGenerator', 'HypothesisTester',

    # Pattern Library
    'Pattern', 'PatternType',
    'PatternDetector', 'PatternPrimitives',
    'ObjectRelationships', 'CompositeTransform',

    # Systematic Search
    'SearchState', 'TaskAnalysis',
    'ConstraintPropagator', 'ProgramSynthesizer',
    'BeamSearchSolver', 'AnalogicalTransfer',
    'ARCSolver',

    # Extended Generators
    'ExtendedGenerators',

    # Deep Synthesis
    'DeepProgramSynthesizer', 'EnumerativeSynthesizer',
    'ProgramNode', 'TypedPrimitive',

    # Neural Patterns
    'GridEmbedding', 'GridEncoder',
    'TransformationEmbedding', 'PatternMatcher',
    'PatternCluster', 'TransformationPrioritizer',

    # Iterative Refinement
    'SolutionAttempt', 'ErrorAnalysis',
    'ErrorAnalyzer', 'SolutionRefiner',
    'IterativeRefinementSolver', 'HypothesisCombiner',
    'ConstraintBasedRepair',

    # Enhanced Solver
    'EnhancedARCSolver', 'SolveResult',
]

__version__ = '2.0.0'
