"""
V94 Complete System - Embodied Learning and Grounded Cognition Architecture

This is the complete V94 system that integrates all embodied learning components
with previous STAN versions, representing the paradigm shift from simulated
intelligence to experienced intelligence.
"""


def _degraded_warn(_module: str, _exc: BaseException) -> None:
    """Log why an optional import degraded instead of failing silently."""
    import logging
    logging.getLogger(__name__).warning(
        "%s unavailable (%s: %s) - dependent names set to None",
        _module, type(_exc).__name__, _exc,
    )


try:
    from dataclasses import dataclass, field
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("dataclasses", _exc)
    dataclass = field = None  # degraded: unavailable
try:
    from typing import Dict, List, Optional, Any, Union, Callable
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("typing", _exc)
    Dict = List = Optional = Any = Union = Callable = None  # degraded: unavailable
try:
    import logging
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("<unknown>", _exc)
    logging = None  # degraded: unavailable
try:
    import time
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("<unknown>", _exc)
    time = None  # degraded: unavailable
try:
    import numpy as np
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("<unknown>", _exc)
    np = None  # degraded: unavailable

# Import V94 components
try:
    from .embodied_learning_engine import EmbodiedLearningEngine, LearningState
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".embodied_learning_engine", _exc)
    EmbodiedLearningEngine = LearningState = None  # degraded: unavailable
try:
    from .sensorimotor_system import SensorimotorInterface, WorldAction, Experience
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".sensorimotor_system", _exc)
    SensorimotorInterface = WorldAction = Experience = None  # degraded: unavailable
try:
    from .developmental_learning import DevelopmentalLearning, DevelopmentalStage
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".developmental_learning", _exc)
    DevelopmentalLearning = DevelopmentalStage = None  # degraded: unavailable
try:
    from .common_sense_engine import CommonSenseEngine
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".common_sense_engine", _exc)
    CommonSenseEngine = None  # degraded: unavailable
try:
    from .language_grounding import LanguageGroundingEngine
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".language_grounding", _exc)
    LanguageGroundingEngine = None  # degraded: unavailable

# Import previous versions for integration
try:
    from ..v80 import V80CompleteSystem
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("..v80", _exc)
    V80CompleteSystem = None  # degraded: unavailable
try:
    from ..v91 import V91CompleteSystem
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("..v91", _exc)
    V91CompleteSystem = None  # degraded: unavailable
try:
    from ..v92 import V92CompleteSystem
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("..v92", _exc)
    V92CompleteSystem = None  # degraded: unavailable
try:
    from ..v93 import V93CompleteSystem
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("..v93", _exc)
    V93CompleteSystem = None  # degraded: unavailable


@dataclass
class V94Config:
    """Configuration for V94 Embodied Learning System"""
    # Embodied learning parameters
    embodied_learning_rate: float = 0.01
    curiosity_drive_strength: float = 0.8
    exploration_tendency: float = 0.7

    # Developmental learning
    developmental_stage: str = "sensorimotor"
    play_based_learning: bool = True
    social_learning_enabled: bool = True

    # Common sense parameters
    common_sense_confidence_threshold: float = 0.7
    physics_intuition_enabled: bool = True
    social_reasoning_enabled: bool = True

    # Language grounding
    language_grounding_enabled: bool = True
    multimodal_integration: bool = True
    concept_confidence_threshold: float = 0.6

    # Integration with previous versions
    v80_integration_enabled: bool = True  # Neural-symbolic
    v91_integration_enabled: bool = True  # Social AGI
    v92_integration_enabled: bool = True  # Scientific discovery
    v93_integration_enabled: bool = True  # Self-modifying

    # Performance
    max_experience_buffer_size: int = 10000
    update_frequency: float = 1.0  # Hz
    enable_parallel_processing: bool = True


class V94CompleteSystem:
    """
    Complete V94 System implementing Embodied Learning and Grounded Cognition.

    This represents the paradigm shift from simulated intelligence to
    experienced intelligence through:
    - Sensorimotor interaction with the world
    - Developmental learning through play and exploration
    - Common sense reasoning from embodied experience
    - Language grounding in real-world experiences
    - Integration with all previous STAN versions
    """

    def __init__(self, config: Optional[V94Config] = None):
        self.config = config or V94Config()
        self.logger = logging.getLogger(__name__)

        # Core V94 components
        self.embodied_engine = EmbodiedLearningEngine(self.config.__dict__)
        self.sensorimotor_system = self.embodied_engine.sensorimotor_system
        self.developmental_learning = self.embodied_engine.developmental_learning
        self.common_sense_engine = self.embodied_engine.common_sense_engine
        self.language_grounding = self.embodied_engine.language_grounding

        # Integration with previous versions
        self.previous_versions = {}
        self._initialize_previous_versions()

        # Learning state tracking
        self.learning_state = LearningState()
        self.developmental_milestones = []
        self.grounded_concepts = {}

        # Performance metrics
        self.start_time = time.time()
        self.total_interactions = 0
        self.successful_groundings = 0
        self.discovery_count = 0

        # Mode-specific configurations
        self.current_mode = "standard"
        self.mode_configurations = {
            "standard": self._get_standard_config(),
            "exploration": self._get_exploration_config(),
            "learning": self._get_learning_config(),
            "social": self._get_social_config(),
            "scientific": self._get_scientific_config()
        }
