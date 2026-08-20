"""
V92 Automated Scientific Discovery Engine
=========================================

This module represents the pinnacle of STAN's evolution - an automated
scientific discovery system capable of generating hypotheses, discovering
causal relationships, applying mathematical intuition, and designing experiments.
"""

try:
    from .v92_system import (
    V92CompleteSystem,
    V92Config,
    ScientificDiscovery,
    create_v92_system,
    create_v92_explorer,
    create_v92_validator,
    create_v92_mathematician,
    create_v92_experimentalist
    )
except Exception:
    V92CompleteSystem = V92Config = ScientificDiscovery = create_v92_system = create_v92_explorer = create_v92_validator = create_v92_mathematician = create_v92_experimentalist = None  # degraded: unavailable

try:
    from .hypothesis_engine import (
    HypothesisGenerator,
    Hypothesis,
    HypothesisType
    )
except Exception:
    HypothesisGenerator = Hypothesis = HypothesisType = None  # degraded: unavailable

try:
    from .mathematical_intuition import (
    MathematicalIntuitionModule,
    MathematicalConjecture,
    Proof,
    MathDomain,
    ProofStatus
    )
except Exception:
    MathematicalIntuitionModule = MathematicalConjecture = Proof = MathDomain = ProofStatus = None  # degraded: unavailable

try:
    from .causal_discovery import (
    CausalDiscoveryEngine,
    CausalModel,
    CausalRelation,
    Intervention,
    Counterfactual,
    DiscoveryMethod
    )
except Exception:
    CausalDiscoveryEngine = CausalModel = CausalRelation = Intervention = Counterfactual = DiscoveryMethod = None  # degraded: unavailable

try:
    from .experimental_design import (
    ExperimentalDesignEngine,
    ExperimentalDesign,
    ExperimentalVariable,
    Treatment,
    ExperimentalType,
    SimulationResult
    )
except Exception:
    ExperimentalDesignEngine = ExperimentalDesign = ExperimentalVariable = Treatment = ExperimentalType = SimulationResult = None  # degraded: unavailable

__all__ = [
    # Main system
    'V92CompleteSystem',
    'V92Config',
    'ScientificDiscovery',
    'create_v92_system',
    'create_v92_explorer',
    'create_v92_validator',
    'create_v92_mathematician',
    'create_v92_experimentalist',

    # Hypothesis generation
    'HypothesisGenerator',
    'Hypothesis',
    'HypothesisType',

    # Mathematical intuition
    'MathematicalIntuitionModule',
    'MathematicalConjecture',
    'Proof',
    'MathDomain',
    'ProofStatus',

    # Causal discovery
    'CausalDiscoveryEngine',
    'CausalModel',
    'CausalRelation',
    'Intervention',
    'Counterfactual',
    'DiscoveryMethod',

    # Experimental design
    'ExperimentalDesignEngine',
    'ExperimentalDesign',
    'ExperimentalVariable',
    'Treatment',
    'ExperimentalType',
    'SimulationResult'
]