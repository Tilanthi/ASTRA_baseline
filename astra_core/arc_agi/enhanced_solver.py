#!/usr/bin/env python3
# Copyright 2026 Glenn J. White
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Enhanced ARC-AGI Solver

Integrates all advanced components:
- Extended pattern generators (40+ generators)
- Deep program synthesis (depth 5)
- Neural pattern recognition with embeddings
- Iterative refinement with error correction
"""


def _degraded_warn(_module: str, _exc: BaseException) -> None:
    """Log why an optional import degraded instead of failing silently."""
    import logging
    logging.getLogger(__name__).warning(
        "%s unavailable (%s: %s) - dependent names set to None",
        _module, type(_exc).__name__, _exc,
    )


try:
    import numpy as np
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("<unknown>", _exc)
    np = None  # degraded: unavailable
try:
    from typing import List, Tuple, Dict, Set, Optional, Callable, Any
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("typing", _exc)
    List = Tuple = Dict = Set = Optional = Callable = Any = None  # degraded: unavailable
try:
    from dataclasses import dataclass
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("dataclasses", _exc)
    dataclass = None  # degraded: unavailable
try:
    import time
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("<unknown>", _exc)
    time = None  # degraded: unavailable

try:
    from .grid_dsl import Grid, GridObject, empty_grid
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".grid_dsl", _exc)
    Grid = GridObject = empty_grid = None  # degraded: unavailable
try:
    from .hypothesis_engine import (
    TransformationHypothesis, TransformationType,
    HypothesisGenerator, HypothesisTester
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".hypothesis_engine", _exc)
    TransformationHypothesis = TransformationType = HypothesisGenerator = HypothesisTester = None  # degraded: unavailable
try:
    from .pattern_library import PatternDetector, PatternPrimitives, CompositeTransform
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".pattern_library", _exc)
    PatternDetector = PatternPrimitives = CompositeTransform = None  # degraded: unavailable
try:
    from .systematic_search import (
    ConstraintPropagator, ProgramSynthesizer, BeamSearchSolver,
    AnalogicalTransfer, TaskAnalysis
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".systematic_search", _exc)
    ConstraintPropagator = ProgramSynthesizer = BeamSearchSolver = AnalogicalTransfer = TaskAnalysis = None  # degraded: unavailable
try:
    from .extended_generators import ExtendedGenerators
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".extended_generators", _exc)
    ExtendedGenerators = None  # degraded: unavailable
try:
    from .deep_synthesis import DeepProgramSynthesizer, EnumerativeSynthesizer
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".deep_synthesis", _exc)
    DeepProgramSynthesizer = EnumerativeSynthesizer = None  # degraded: unavailable
try:
    from .neural_patterns import (
    GridEncoder, PatternMatcher, TransformationPrioritizer, PatternCluster
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".neural_patterns", _exc)
    GridEncoder = PatternMatcher = TransformationPrioritizer = PatternCluster = None  # degraded: unavailable
try:
    from .iterative_refinement import (
    ErrorAnalyzer, SolutionRefiner, IterativeRefinementSolver,
    HypothesisCombiner, ConstraintBasedRepair
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".iterative_refinement", _exc)
    ErrorAnalyzer = SolutionRefiner = IterativeRefinementSolver = HypothesisCombiner = ConstraintBasedRepair = None  # degraded: unavailable


@dataclass
class SolveResult:
    """Result of solving a task"""
    solution: Optional[Grid]
    strategy_used: str
    confidence: float
    iterations: int
    time_taken: float


class EnhancedARCSolver:
    """
    Enhanced ARC-AGI solver combining all advanced techniques.
    """

    def __init__(self, timeout_seconds: float = 60.0):
        self.timeout = timeout_seconds

        # Core components
        self.base_generator = HypothesisGenerator()
        self.extended_generator = ExtendedGenerators()
        self.hypothesis_tester = HypothesisTester()

        # Advanced components
        self.deep_synthesizer = DeepProgramSynthesizer(max_depth=5, timeout_seconds=30.0)
        self.enum_synthesizer = EnumerativeSynthesizer(max_depth=4)

        # Neural components
        self.encoder = GridEncoder()
        self.pattern_matcher = PatternMatcher()
        self.prioritizer = TransformationPrioritizer()

        # Refinement components
        self.error_analyzer = ErrorAnalyzer()
        self.refiner = SolutionRefiner()
        self.iterative_solver = IterativeRefinementSolver(max_iterations=5)
        self.combiner = HypothesisCombiner()
        self.constraint_repair = ConstraintBasedRepair()

        # Search components
        self.constraint_prop = ConstraintPropagator()
        self.beam_search = BeamSearchSolver(beam_width=15, max_iterations=200)
        self.analogical = AnalogicalTransfer()

        # Statistics
        self.solve_stats = {
            'direct_hypothesis': 0,
            'extended_hypothesis': 0,
            'deep_synthesis': 0,
            'enum_synthesis': 0,
            'analogical_transfer': 0,
            'iterative_refinement': 0,
            'beam_search': 0,
        }

    def solve(self, train_pairs: List[Tuple[Grid, Grid]],
             test_input: Grid) -> SolveResult:
        """
        Solve an ARC task using multiple strategies in order of efficiency.
        """
        start_time = time.time()

        # Strategy 1: Direct hypothesis matching (fast)
        result = self._try_direct_hypothesis(train_pairs, test_input, start_time)
        if result.solution is not None:
            self.solve_stats['direct_hypothesis'] += 1
            return result

        # Strategy 2: Extended generators (moderate)
        result = self._try_extended_generators(train_pairs, test_input, start_time)
        if result.solution is not None:
            self.solve_stats['extended_hypothesis'] += 1
            return result

        # Strategy 3: Analogical transfer (fast if library has matches)
        result = self._try_analogical_transfer(train_pairs, test_input, start_time)
        if result.solution is not None:
            self.solve_stats['analogical_transfer'] += 1
            return result

        # Strategy 4: Deep program synthesis (slower, more thorough)
        result = self._try_deep_synthesis(train_pairs, test_input, start_time)
        if result.solution is not None:
            self.solve_stats['deep_synthesis'] += 1
            return result

        # Strategy 5: Enumerative synthesis (thorough)
        result = self._try_enum_synthesis(train_pairs, test_input, start_time)
        if result.solution is not None:
            self.solve_stats['enum_synthesis'] += 1
            return result

        # Strategy 6: Iterative refinement with combinations
        result = self._try_iterative_refinement(train_pairs, test_input, start_time)
        if result.solution is not None:
            self.solve_stats['iterative_refinement'] += 1
            return result

        # Strategy 7: Beam search (slowest, most thorough)
        result = self._try_beam_search(train_pairs, test_input, start_time)
        if result.solution is not None:
            self.solve_stats['beam_search'] += 1
            return result

        return SolveResult(
            solution=None,
            strategy_used='none',
            confidence=0.0,
            iterations=0,
            time_taken=time.time() - start_time
        )

    def _try_direct_hypothesis(self, train_pairs: List[Tuple[Grid, Grid]],
                               test_input: Grid,
                               start_time: float) -> SolveResult:
        """Try direct hypothesis matching"""
        if time.time() - start_time > self.timeout:
            return SolveResult(None, 'timeout', 0.0, 0, time.time() - start_time)
