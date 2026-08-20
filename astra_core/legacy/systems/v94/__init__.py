"""
STAN V94: Embodied Learning and Grounded Cognition Architecture (ASTRO Version)

This module implements the paradigm shift from simulated intelligence to experienced intelligence
through embodied learning, sensorimotor integration, and grounded cognition.
Enhanced for astrophysics applications with cosmic-scale embodied understanding.
"""


def _degraded_warn(_module: str, _exc: BaseException) -> None:
    """Log why an optional import degraded instead of failing silently."""
    import logging
    logging.getLogger(__name__).warning(
        "%s unavailable (%s: %s) - dependent names set to None",
        _module, type(_exc).__name__, _exc,
    )


try:
    from .embodied_learning_engine import EmbodiedLearningEngine
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".embodied_learning_engine", _exc)
    EmbodiedLearningEngine = None  # degraded: unavailable
try:
    from .sensorimotor_system import SensorimotorInterface, WorldAction, Experience
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".sensorimotor_system", _exc)
    SensorimotorInterface = WorldAction = Experience = None  # degraded: unavailable
try:
    from .developmental_learning import DevelopmentalLearning, PlayfulExplorer
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".developmental_learning", _exc)
    DevelopmentalLearning = PlayfulExplorer = None  # degraded: unavailable
try:
    from .common_sense_engine import CommonSenseEngine, PhysicsIntuitionModule
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".common_sense_engine", _exc)
    CommonSenseEngine = PhysicsIntuitionModule = None  # degraded: unavailable
try:
    from .language_grounding import LanguageGroundingEngine, ConceptGroundingEngine
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".language_grounding", _exc)
    LanguageGroundingEngine = ConceptGroundingEngine = None  # degraded: unavailable
try:
    from .v94_complete import V94CompleteSystem, V94Config
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".v94_complete", _exc)
    V94CompleteSystem = V94Config = None  # degraded: unavailable
try:
    from .astro_embodied_integration import AstroEmbodiedIntegrator
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".astro_embodied_integration", _exc)
    AstroEmbodiedIntegrator = None  # degraded: unavailable

__all__ = [
    'EmbodiedLearningEngine',
    'SensorimotorInterface',
    'WorldAction',
    'Experience',
    'DevelopmentalLearning',
    'PlayfulExplorer',
    'CommonSenseEngine',
    'PhysicsIntuitionModule',
    'LanguageGroundingEngine',
    'ConceptGroundingEngine',
    'V94CompleteSystem',
    'V94Config',
    'AstroEmbodiedIntegrator'
]