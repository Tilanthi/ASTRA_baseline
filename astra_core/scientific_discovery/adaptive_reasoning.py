"""
Adaptive Reasoning Controller for Scientific Discovery
======================================================

Manages dynamic reasoning mode switching based on discovery phase,
confidence levels, and metacognitive assessment. Integrates with
V41 Orchestrator for high-level reasoning control.

Key Components:
- AdaptiveReasoningController: Main mode selection logic
- MetacognitiveMonitor: Quality assessment using V41
- UncertaintyTracker: Confidence calibration
- ReasoningModeSelector: Phase → Mode mapping

Reasoning Modes (from V41):
- ANALYTICAL: Deep systematic analysis
- CREATIVE: Novel solutions via analogical reasoning
- CRITICAL: Evaluation and falsification
- INTEGRATIVE: Synthesis and unification
- ADAPTIVE: Dynamic responsive reasoning
- DELIBERATIVE: Multi-perspective consideration

Version: 1.0.0
Date: 2025-12-27
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto
from collections import defaultdict

# Import V41 components (try both absolute and relative imports)
try:
    from ..reasoning.v41_orchestrator import (
        ReasoningMode, TaskComplexity, ReasoningTask, ReasoningResult
    )
    from ..reasoning.metacognition import (
        get_metacognitive_controller, ReasoningStrategy, ReasoningTrace
    )
    V41_AVAILABLE = True
except (ImportError, ValueError):
    try:
        # Fallback to absolute import (for when run from outside package)
        from astra_core.reasoning.v41_orchestrator import (
            ReasoningMode, TaskComplexity, ReasoningTask, ReasoningResult
        )
        from astra_core.reasoning.metacognition import (
            get_metacognitive_controller, ReasoningStrategy, ReasoningTrace
        )
        V41_AVAILABLE = True
    except ImportError as e:
        logging.warning(f"Could not import V41 components: {e}")
        V41_AVAILABLE = False
    # Define fallback enums if import fails
    class ReasoningMode(Enum):
        ANALYTICAL = auto()
        CREATIVE = auto()
        CRITICAL = auto()
        INTEGRATIVE = auto()
        ADAPTIVE = auto()
        DELIBERATIVE = auto()

logger = logging.getLogger(__name__)


# =============================================================================
# Discovery Phases
# =============================================================================

class DiscoveryPhase(Enum):
    """Phases of the scientific discovery cycle"""
    LITERATURE_REVIEW = "literature_review"
    DATA_GATHERING = "data_gathering"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    EXPERIMENTAL_DESIGN = "experimental_design"
    ANALYSIS_EXECUTION = "analysis_execution"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"


# Default phase-to-mode mappings
PHASE_TO_MODE_MAP = {
    DiscoveryPhase.LITERATURE_REVIEW: ReasoningMode.ANALYTICAL,
    DiscoveryPhase.DATA_GATHERING: ReasoningMode.ANALYTICAL,
    DiscoveryPhase.HYPOTHESIS_GENERATION: ReasoningMode.CREATIVE,
    DiscoveryPhase.EXPERIMENTAL_DESIGN: ReasoningMode.CREATIVE,
    DiscoveryPhase.ANALYSIS_EXECUTION: ReasoningMode.ANALYTICAL,
    DiscoveryPhase.SYNTHESIS: ReasoningMode.INTEGRATIVE,
    DiscoveryPhase.VALIDATION: ReasoningMode.CRITICAL,
}


# =============================================================================
# Reasoning State Tracking
# =============================================================================

@dataclass
class ReasoningState:
    """Current reasoning state"""
    phase: DiscoveryPhase
    mode: ReasoningMode
    confidence: float = 0.5
    quality_score: float = 0.5
    iterations: int = 0
    stuck_count: int = 0  # Times we've been stuck without progress

    # Performance metrics
    time_in_phase: float = 0.0
    phase_start_time: float = field(default_factory=time.time)

    # History
    mode_history: List[ReasoningMode] = field(default_factory=list)
    phase_history: List[DiscoveryPhase] = field(default_factory=list)

    def update_phase(self, new_phase: DiscoveryPhase):
        """Update to new phase"""
        self.phase_history.append(self.phase)
        self.phase = new_phase
        self.time_in_phase = 0.0
        self.phase_start_time = time.time()
        self.iterations = 0
        self.stuck_count = 0

    def update_mode(self, new_mode: ReasoningMode):
        """Update reasoning mode"""
        self.mode_history.append(self.mode)
        self.mode = new_mode

    def tick(self):
        """Update time tracking"""
        self.time_in_phase = time.time() - self.phase_start_time
        self.iterations += 1


# =============================================================================
# Uncertainty Tracker
# =============================================================================

@dataclass
class ConfidenceRecord:
    """One confidence-vs-outcome observation for calibration"""
    confidence: float
    success: bool
    timestamp: float = field(default_factory=time.time)


class UncertaintyTracker:
    """
    Confidence calibration and drift detection.

    Records (stated confidence, realised success) pairs and estimates the
    systematic bias between them, so downstream mode selection can tell
    over-confidence from genuine uncertainty.
    """

    def __init__(self, max_records: int = 500):
        self.records: List[ConfidenceRecord] = []
        self.max_records = max_records

    def record(self, confidence: float, success: bool) -> None:
        self.records.append(ConfidenceRecord(float(confidence), bool(success)))
        if len(self.records) > self.max_records:
            del self.records[:len(self.records) - self.max_records]

    @property
    def bias(self) -> float:
        """mean(confidence) - mean(success); >0 means over-confident."""
        if not self.records:
            return 0.0
        conf = sum(r.confidence for r in self.records) / len(self.records)
        succ = sum(1.0 for r in self.records if r.success) / len(self.records)
        return conf - succ

    def calibrate(self, confidence: float) -> float:
        """Remove the estimated bias from a stated confidence."""
        return float(min(1.0, max(0.0, confidence - 0.5 * self.bias)))

    def summary(self) -> Dict[str, Any]:
        return {'n_records': len(self.records), 'bias': self.bias}


def _clip01(x: float) -> float:
    return min(1.0, max(0.0, float(x)))


class MetacognitiveMonitor:
    """
    Quality assessment for reasoning episodes.

    Uses the V41 metacognitive controller when available for monitoring
    hooks; the quality score itself is a weighted aggregate of caller
    supplied metrics (no invented numbers).
    """

    # Default weights for the recognised quality metrics
    DEFAULT_WEIGHTS = {
        'completeness': 0.3,
        'consistency': 0.3,
        'novelty': 0.2,
        'evidence_strength': 0.2,
    }

    def __init__(self):
        self.v41_controller = None
        if V41_AVAILABLE:
            try:
                self.v41_controller = get_metacognitive_controller()
            except Exception:
                self.v41_controller = None  # degraded: monitoring unavailable
        self.quality_history: List[float] = []

    def assess_quality(self, metrics: Optional[Dict[str, float]] = None) -> float:
        """Weighted mean of supplied metrics in [0, 1]; 0.5 when absent."""
        if not metrics:
            return 0.5
        total_w = 0.0
        score = 0.0
        for key, weight in self.DEFAULT_WEIGHTS.items():
            if key in metrics:
                score += weight * _clip01(metrics[key])
                total_w += weight
        # Unknown extra metrics are averaged in at a small residual weight
        extra = [k for k in metrics if k not in self.DEFAULT_WEIGHTS]
        if extra and total_w < 1.0:
            residual = (1.0 - total_w) / len(extra)
            for key in extra:
                score += residual * _clip01(metrics[key])
                total_w += residual
        quality = score / total_w if total_w > 0 else 0.5
        self.quality_history.append(quality)
        return quality

    @property
    def quality_trend(self) -> float:
        """Slope of recent quality scores (positive = improving)."""
        h = self.quality_history[-10:]
        if len(h) < 2:
            return 0.0
        return (h[-1] - h[0]) / (len(h) - 1)


class ReasoningModeSelector:
    """
    Phase-to-mode mapping with adaptive overrides.

    Base mapping comes from PHASE_TO_MODE_MAP; overrides respond to being
    stuck (switch to DELIBERATIVE for a fresh perspective) and low
    confidence mid-analysis (switch to ADAPTIVE).
    """

    def __init__(self, phase_map: Optional[Dict[DiscoveryPhase, ReasoningMode]] = None):
        self.phase_map = phase_map or dict(PHASE_TO_MODE_MAP)

    def select(self, state: ReasoningState) -> Tuple[ReasoningMode, str]:
        """Return (mode, reason) for the current state."""
        base = self.phase_map.get(state.phase, ReasoningMode.ADAPTIVE)

        if state.stuck_count >= 3:
            if state.phase != DiscoveryPhase.VALIDATION:
                return ReasoningMode.DELIBERATIVE, (
                    f"stuck {state.stuck_count}x in {state.phase.value}; "
                    "opening multiple perspectives")

        if (state.phase == DiscoveryPhase.ANALYSIS_EXECUTION
                and state.confidence < 0.3 and state.iterations > 5):
            return ReasoningMode.ADAPTIVE, (
                "low confidence during analysis; switching to responsive mode")

        return base, f"default mapping for {state.phase.value}"

    def suggest_next_phase(self, state: ReasoningState) -> DiscoveryPhase:
        """Advance the cycle when quality and confidence are both healthy."""
        order = list(DiscoveryPhase)
        if state.confidence >= 0.7 and state.quality_score >= 0.7:
            return order[(order.index(state.phase) + 1) % len(order)]
        return state.phase


class AdaptiveReasoningController:
    """
    Main controller: tracks discovery state and switches reasoning modes.

    Typical use::

        controller = get_adaptive_reasoning_controller()
        controller.update_phase(DiscoveryPhase.HYPOTHESIS_GENERATION)
        ...
        controller.record_iteration(confidence=0.6, quality_metrics={...})
        mode = controller.get_current_mode()
    """

    def __init__(self,
                 initial_phase: DiscoveryPhase = DiscoveryPhase.LITERATURE_REVIEW,
                 initial_confidence: float = 0.5):
        self.uncertainty = UncertaintyTracker()
        self.monitor = MetacognitiveMonitor()
        self.selector = ReasoningModeSelector()
        self.state = ReasoningState(phase=initial_phase, mode=ReasoningMode.ANALYTICAL,
                                    confidence=initial_confidence)
        self.state.mode, _ = self.selector.select(self.state)
        self.switch_log: List[Dict[str, Any]] = []

    # ----------------------------------------------------------------- state
    def update_phase(self, phase: DiscoveryPhase, reason: str = "") -> None:
        self.state.update_phase(phase)
        self._reselect(reason=f"phase -> {phase.value}")

    def _reselect(self, reason: str = "") -> None:
        new_mode, why = self.selector.select(self.state)
        if new_mode is not self.state.mode:
            self.switch_log.append({
                'from': self.state.mode.name, 'to': new_mode.name,
                'reason': f"{reason} | {why}", 'time': time.time()})
            logger.info("Mode switch %s -> %s (%s)",
                        self.state.mode.name, new_mode.name, why)
            self.state.update_mode(new_mode)

    # ---------------------------------------------------------------- events
    def record_iteration(self, confidence: Optional[float] = None,
                         success: Optional[bool] = None,
                         quality_metrics: Optional[Dict[str, float]] = None,
                         stuck: bool = False) -> ReasoningMode:
        """
        Record one reasoning iteration and re-select the mode if warranted.

        Returns the (possibly switched) current mode.
        """
        self.state.tick()
        if confidence is not None:
            calibrated = self.uncertainty.calibrate(confidence)
            # exponential smoothing of confidence
            alpha = 0.3
            self.state.confidence = (alpha * calibrated
                                     + (1 - alpha) * self.state.confidence)
            if success is not None:
                self.uncertainty.record(calibrated, success)
        if quality_metrics is not None:
            self.state.quality_score = self.monitor.assess_quality(quality_metrics)

        if stuck:
            self.state.stuck_count += 1
        else:
            self.state.stuck_count = 0

        self._reselect()
        return self.state.mode

    # ------------------------------------------------------------------ read
    def get_current_mode(self) -> ReasoningMode:
        return self.state.mode

    def suggest_next_phase(self) -> DiscoveryPhase:
        return self.selector.suggest_next_phase(self.state)

    def get_state_summary(self) -> Dict[str, Any]:
        s = self.state
        return {
            'mode': s.mode.name,
            'phase': s.phase.value,
            'confidence': s.confidence,
            'quality_score': s.quality_score,
            'iterations': s.iterations,
            'stuck_count': s.stuck_count,
            'time_in_phase': s.time_in_phase,
            'mode_history': [m.name for m in s.mode_history],
            'phase_history': [p.value for p in s.phase_history],
            'calibration_bias': self.uncertainty.bias,
            'quality_trend': self.monitor.quality_trend,
            'n_mode_switches': len(self.switch_log),
            'v41_available': V41_AVAILABLE,
        }


# Module-level singleton (mirrors reasoning.metacognition pattern)
_adaptive_reasoning_controller: Optional[AdaptiveReasoningController] = None


def get_adaptive_reasoning_controller() -> AdaptiveReasoningController:
    """Get or create the global adaptive reasoning controller."""
    global _adaptive_reasoning_controller
    if _adaptive_reasoning_controller is None:
        _adaptive_reasoning_controller = AdaptiveReasoningController()
    return _adaptive_reasoning_controller
