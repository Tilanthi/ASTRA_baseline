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
V92 Automated Scientific Discovery Engine
=========================================

This module represents the pinnacle of STAN's evolution - an automated
scientific discovery system capable of generating hypotheses, discovering
causal relationships, applying mathematical intuition, and designing experiments.
"""


def _degraded_warn(_module: str, _exc: BaseException) -> None:
    """Log why an optional import degraded instead of failing silently."""
    import logging
    logging.getLogger(__name__).warning(
        "%s unavailable (%s: %s) - dependent names set to None",
        _module, type(_exc).__name__, _exc,
    )


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
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".v92_system", _exc)
    V92CompleteSystem = V92Config = ScientificDiscovery = create_v92_system = create_v92_explorer = create_v92_validator = create_v92_mathematician = create_v92_experimentalist = None  # degraded: unavailable

try:
    from .hypothesis_engine import (
    HypothesisGenerator,
    Hypothesis,
    HypothesisType
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".hypothesis_engine", _exc)
    HypothesisGenerator = Hypothesis = HypothesisType = None  # degraded: unavailable

try:
    from .mathematical_intuition import (
    MathematicalIntuitionModule,
    MathematicalConjecture,
    Proof,
    MathDomain,
    ProofStatus
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".mathematical_intuition", _exc)
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
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".causal_discovery", _exc)
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
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".experimental_design", _exc)
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