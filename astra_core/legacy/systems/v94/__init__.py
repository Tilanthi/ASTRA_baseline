"""
STAN V94: Embodied Learning and Grounded Cognition Architecture (ASTRO Version)

This module implements the paradigm shift from simulated intelligence to experienced intelligence
through embodied learning, sensorimotor integration, and grounded cognition.
Enhanced for astrophysics applications with cosmic-scale embodied understanding.
"""

try:
    from .embodied_learning_engine import EmbodiedLearningEngine
except Exception:
    EmbodiedLearningEngine = None  # degraded: unavailable
try:
    from .sensorimotor_system import SensorimotorInterface, WorldAction, Experience
except Exception:
    SensorimotorInterface = WorldAction = Experience = None  # degraded: unavailable
try:
    from .developmental_learning import DevelopmentalLearning, PlayfulExplorer
except Exception:
    DevelopmentalLearning = PlayfulExplorer = None  # degraded: unavailable
try:
    from .common_sense_engine import CommonSenseEngine, PhysicsIntuitionModule
except Exception:
    CommonSenseEngine = PhysicsIntuitionModule = None  # degraded: unavailable
try:
    from .language_grounding import LanguageGroundingEngine, ConceptGroundingEngine
except Exception:
    LanguageGroundingEngine = ConceptGroundingEngine = None  # degraded: unavailable
try:
    from .v94_complete import V94CompleteSystem, V94Config
except Exception:
    V94CompleteSystem = V94Config = None  # degraded: unavailable
try:
    from .astro_embodied_integration import AstroEmbodiedIntegrator
except Exception:
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