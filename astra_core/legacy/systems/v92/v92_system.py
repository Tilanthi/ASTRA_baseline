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

Complete V92 system integrating hypothesis generation, mathematical intuition,
causal discovery, and experimental design for automated scientific reasoning.

This is the culmination of STAN's evolution - a true automated scientist capable
of generating novel hypotheses, discovering causal relationships, understanding
mathematics intuitively, and designing experiments to test theories.

Capabilities:
- Generate novel scientific hypotheses from existing knowledge
- Mathematical intuition and automated proof generation
- Discover causal relationships from observational data
- Design optimal experiments to test hypotheses
- Integrate all components for end-to-end scientific discovery
- Cross-domain scientific reasoning
- Automated theory building and testing
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
    import pandas as pd
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("<unknown>", _exc)
    pd = None  # degraded: unavailable
try:
    from typing import Dict, List, Any, Optional, Tuple, Set
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("typing", _exc)
    Dict = List = Any = Optional = Tuple = Set = None  # degraded: unavailable
try:
    from dataclasses import dataclass, field
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("dataclasses", _exc)
    dataclass = field = None  # degraded: unavailable
try:
    from enum import Enum
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("enum", _exc)
    Enum = None  # degraded: unavailable
try:
    import time
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("<unknown>", _exc)
    time = None  # degraded: unavailable
try:
    import json
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("<unknown>", _exc)
    json = None  # degraded: unavailable
try:
    from collections import defaultdict
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("collections", _exc)
    defaultdict = None  # degraded: unavailable

try:
    from .hypothesis_engine import (
    HypothesisGenerator, Hypothesis, HypothesisType
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".hypothesis_engine", _exc)
    HypothesisGenerator = Hypothesis = HypothesisType = None  # degraded: unavailable
try:
    from .mathematical_intuition import (
    MathematicalIntuitionModule, MathematicalConjecture,
    Proof, MathDomain, ProofStatus
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".mathematical_intuition", _exc)
    MathematicalIntuitionModule = MathematicalConjecture = Proof = MathDomain = ProofStatus = None  # degraded: unavailable
try:
    from .causal_discovery import (
    CausalDiscoveryEngine, CausalModel, CausalRelation,
    Intervention, Counterfactual, DiscoveryMethod
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".causal_discovery", _exc)
    CausalDiscoveryEngine = CausalModel = CausalRelation = Intervention = Counterfactual = DiscoveryMethod = None  # degraded: unavailable
try:
    from .experimental_design import (
    ExperimentalDesignEngine, ExperimentalDesign, ExperimentalVariable,
    Treatment, ExperimentalType, SimulationResult
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".experimental_design", _exc)
    ExperimentalDesignEngine = ExperimentalDesign = ExperimentalVariable = Treatment = ExperimentalType = SimulationResult = None  # degraded: unavailable


@dataclass
class V92Config:
    """Configuration for V92 system"""
    enable_hypothesis_generation: bool = True
    enable_mathematical_intuition: bool = True
    enable_causal_discovery: bool = True
    enable_experimental_design: bool = True
    enable_cross_domain_synthesis: bool = True

    # Module configurations
    max_hypotheses_per_cycle: int = 50
    max_conjectures_per_cycle: int = 30
    max_experiments_per_design: int = 10
    confidence_threshold: float = 0.7

    # Scientific domains to focus on
    primary_domains: List[str] = field(default_factory=lambda: ['physics', 'biology', 'mathematics'])
    secondary_domains: List[str] = field(default_factory=lambda: ['chemistry', 'computer_science', 'economics'])

    # Learning parameters
    knowledge_retention: float = 0.95  # How much knowledge to retain between cycles
    exploration_rate: float = 0.3  # Rate of exploring novel ideas vs. exploiting known ones
    synthesis_depth: int = 3  # Depth of cross-domain synthesis


@dataclass
class ScientificDiscovery:
    """A complete scientific discovery process"""
    id: str
    domain: str
    initial_observation: str
    generated_hypotheses: List[Hypothesis]
    mathematical_conjectures: List[MathematicalConjecture]
    causal_models: List[CausalModel]
    experimental_designs: List[ExperimentalDesign]
    verified_results: Dict[str, Any]
    novelty_score: float
    confidence_score: float
    impact_assessment: str
    timestamp: float = field(default_factory=time.time)


class V92CompleteSystem:
    """
    Complete V92 Automated Scientific Discovery System.

    This system represents the pinnacle of STAN's capabilities - an automated
    scientist that can generate hypotheses, discover causal relationships,
    understand mathematics intuitively, and design experiments to test theories.
    """

    def __init__(self, config: Optional[V92Config] = None):
        self.config = config or V92Config()

        # Initialize all V92 modules
        if self.config.enable_hypothesis_generation:
            self.hypothesis_generator = HypothesisGenerator()

        if self.config.enable_mathematical_intuition:
            self.math_intuition = MathematicalIntuitionModule()

        if self.config.enable_causal_discovery:
            self.causal_discovery = CausalDiscoveryEngine()

        if self.config.enable_experimental_design:
            self.experiment_design = ExperimentalDesignEngine()

        # Knowledge integration
        self.knowledge_graph = defaultdict(dict)
        self.discovery_history = []
        self.cross_domain_mappings = defaultdict(list)

        # Scientific metrics
        self.total_discoveries = 0
        self.verified_hypotheses = 0
        self.novel_theories = 0
        self.successful_experiments = 0

        print("V92 Automated Scientific Discovery Engine initialized")

    def discover(self, observation: str, domain: str = "general",
                data: Optional[pd.DataFrame] = None,
                depth: int = 3) -> Dict[str, Any]:
        """
        End-to-end scientific discovery from observation.

        This method orchestrates all V92 capabilities to take an initial observation
        and generate a complete scientific discovery process.
        """
        print(f"\n{'='*60}")
        print(f"V92 DISCOVERY ENGINE: {domain.upper()}")
        print(f"Initial Observation: {observation}")
        print(f"{'='*60}")

        discovery_id = f"discovery_{int(time.time())}"

        # Phase 1: Hypothesis Generation
        print("\n🔍 Phase 1: Generating hypotheses...")
        hypotheses = self._generate_hypotheses(observation, domain)
        print(f"   Generated {len(hypotheses)} hypotheses")

        # Phase 2: Mathematical Intuition (if applicable)
        conjectures = []
        if self.config.enable_mathematical_intuition and domain in ['mathematics', 'physics', 'computer_science']:
            print("\n🧮 Phase 2: Applying mathematical intuition...")
            conjectures = self._generate_mathematical_conjectures(observation, domain)
            print(f"   Generated {len(conjectures)} mathematical conjectures")

        # Phase 3: Causal Discovery (if data provided)
        causal_models = []
        if data is not None and self.config.enable_causal_discovery:
            print("\n🔗 Phase 3: Discovering causal relationships...")
            causal_models = self._discover_causal_structure(data, domain)
            print(f"   Discovered {len(causal_models)} causal models")

        # Phase 4: Experimental Design
        experimental_designs = []
        if self.config.enable_experimental_design and hypotheses:
            print("\n🧪 Phase 4: Designing experiments...")
            experimental_designs = self._design_experiments(hypotheses, domain, data)
            print(f"   Designed {len(experimental_designs)} experiments")

        # Phase 5: Cross-Domain Synthesis
        synthesis_results = []
        if self.config.enable_cross_domain_synthesis and depth > 1:
            print("\n🌐 Phase 5: Cross-domain synthesis...")
            synthesis_results = self._cross_domain_synthesis(hypotheses, domain)
            print(f"   Generated {len(synthesis_results)} cross-domain insights")

        # Assemble discovery
        discovery = ScientificDiscovery(
            id=discovery_id,
            domain=domain,
            initial_observation=observation,
            generated_hypotheses=hypotheses,
            mathematical_conjectures=conjectures,
            causal_models=causal_models,
            experimental_designs=experimental_designs,
            verified_results={},
            novelty_score=self._calculate_novelty_score(hypotheses, conjectures),
            confidence_score=self._calculate_confidence_score(hypotheses, causal_models),
            impact_assessment=self._assess_impact(hypotheses, domain)
        )

        self.discovery_history.append(discovery)
        self.total_discoveries += 1

        # Generate final report
        result = {
            'discovery_id': discovery_id,
            'domain': domain,
            'observation': observation,
            'hypotheses': [{'id': h.id, 'statement': h.statement, 'confidence': h.confidence} for h in hypotheses[:5]],
            'mathematical_conjectures': [{'id': c.id, 'statement': c.statement} for c in conjectures[:3]],
            'causal_models': [{'variables': list(m.variables), 'edges': len(m.edges)} for m in causal_models[:3]],
            'experimental_designs': [{'name': d.name, 'conditions': len(d.conditions)} for d in experimental_designs[:3]],
            'cross_domain_synthesis': synthesis_results,
            'novelty_score': discovery.novelty_score,
            'confidence_score': discovery.confidence_score,
            'impact_assessment': discovery.impact_assessment,
            'recommendations': self._generate_recommendations(discovery)
        }

        print(f"\n✅ Discovery Complete! Score: {discovery.confidence_score:.2f}")
        return result

    def think(self, question: str, domain: str = "general") -> Dict[str, Any]:
        """
        Enhanced thinking with scientific discovery capabilities.

        This integrates V92's scientific reasoning with V90-V91 capabilities.
        """
        # Parse question to determine appropriate approach
        question_type = self._classify_question(question)

        result = {
            'question': question,
            'domain': domain,
            'reasoning_mode': 'scientific_discovery',
            'timestamp': time.time()
        }

        if question_type == 'hypothesis_generation':
            # Generate hypotheses about the question
            hypotheses = self.hypothesis_generator.generate_hypotheses(domain, 5)
            result['hypotheses'] = [{'statement': h.statement, 'confidence': h.confidence} for h in hypotheses]
            result['answer'] = f"Based on scientific reasoning, I propose {len(hypotheses)} testable hypotheses."

        elif question_type == 'mathematical_reasoning':
            # Apply mathematical intuition
            observations = [question]  # Treat question as observation
            conjectures = self.math_intuition.generate_conjectures(
                MathDomain.ANALYSIS, observations, 3
            )
            result['conjectures'] = [{'statement': c.statement, 'confidence': c.confidence} for c in conjectures]
            result['answer'] = f"Mathematical intuition suggests {len(conjectures)} conjectures."

        elif question_type == 'causal_inference':
            # Provide causal reasoning framework
            result['causal_framework'] = {
                'approach': 'Apply causal discovery algorithms to observational data',
                'methods': ['PC algorithm', 'GES', 'LiNGAM', 'Additive Noise Models'],
                'requirements': ['Sufficient sample size', 'No unmeasured confounders', 'Causal sufficiency']
            }
            result['answer'] = "To answer this question causally, we need to apply causal discovery methods to appropriate data."

        elif question_type == 'experimental_design':
            # Design experiment to answer question
            design_type = ExperimentalType.RANDOMIZED_CONTROLLED
            variables = [
                # FIX(audit): `levels[...]` subscripted an undefined name;
                # `levels` is a List[Any] field of ExperimentalVariable.
                ExperimentalVariable("treatment", "independent", "categorical",
                                     levels=["control", "experimental"]),
                ExperimentalVariable("outcome", "dependent", "continuous")
            ]
            design = self.experiment_design.design_experiment(question, variables, design_type)
            result['experimental_design'] = {
                'type': design.design_type.value,
                'conditions': len(design.conditions),
                'estimated_sample_size': design.sample_size,
                'estimated_cost': design.estimated_cost
            }
            result['answer'] = f"Designed {design.design_type.value} experiment with {design.sample_size} total subjects."

        else:
            # General scientific reasoning
            result['scientific_approach'] = {
                'step1': 'Formulate testable hypotheses',
                'step2': 'Apply mathematical reasoning if applicable',
                'step3': 'Seek causal relationships',
                'step4': 'Design experiments to test hypotheses',
                'step5': 'Iterate based on evidence'
            }
            result['answer'] = "I'll approach this systematically using the scientific method with hypothesis generation, causal reasoning, and experimental design."

        # Add V92 capabilities summary
        result['v92_capabilities'] = {
            'hypothesis_generation': self.config.enable_hypothesis_generation,
            'mathematical_intuition': self.config.enable_mathematical_intuition,
            'causal_discovery': self.config.enable_causal_discovery,
            'experimental_design': self.config.enable_experimental_design,
            'cross_domain_synthesis': self.config.enable_cross_domain_synthesis
        }

        return result

    def _generate_hypotheses(self, observation: str, domain: str) -> List[Hypothesis]:
        """Generate hypotheses from observation"""
        # Create knowledge from observation
        knowledge = {
            'concepts': [observation],
            'relations': [{'subject': 'observation', 'object': domain, 'type': 'observed_in'}],
            'domain': domain
        }

        # Add to knowledge graph
        self.hypothesis_generator.add_knowledge(knowledge)

        # Generate hypotheses
        hypotheses = self.hypothesis_generator.generate_hypotheses(domain, self.config.max_hypotheses_per_cycle)

        # Filter by confidence
        filtered_hypotheses = [h for h in hypotheses if h.confidence >= self.config.confidence_threshold]

        return filtered_hypotheses

    def _generate_mathematical_conjectures(self, observation: str, domain: str) -> List[MathematicalConjecture]:
        """Generate mathematical conjectures"""
        # Determine mathematical domain
        math_domain = MathDomain.ANALYSIS  # Default
        if 'geometry' in domain.lower() or 'space' in observation.lower():
            math_domain = MathDomain.GEOMETRY
        elif 'number' in observation.lower() or 'prime' in observation.lower():
            math_domain = MathDomain.NUMBER_THEORY
        elif 'algebra' in domain.lower():
            math_domain = MathDomain.ALGEBRA

        # Generate observations for conjecture generation
        observations = [observation, f"Mathematical patterns in {domain}"]

        # Generate conjectures
        conjectures = self.math_intuition.generate_conjectures(
            math_domain, observations, self.config.max_conjectures_per_cycle
        )

        # Filter by confidence
        filtered_conjectures = [c for c in conjectures if c.confidence >= self.config.confidence_threshold]

        return filtered_conjectures

# =============================================================================
# FACTORY FUNCTIONS
# (Re-implemented 2026-08; bodies were lost to file truncation before the
#  audit. Each factory is a documented configuration preset over the real
#  V92CompleteSystem.)
# =============================================================================

def create_v92_system(config: Optional[V92Config] = None) -> V92CompleteSystem:
    """General-purpose V92 scientific discovery system (default config)."""
    return V92CompleteSystem(config)


def create_v92_explorer(config: Optional[V92Config] = None) -> V92CompleteSystem:
    """Explorer preset: high exploration, low threshold, wide domains."""
    config = config or V92Config()
    config.exploration_rate = max(config.exploration_rate, 0.7)
    config.confidence_threshold = min(config.confidence_threshold, 0.4)
    config.primary_domains = list(dict.fromkeys(
        config.primary_domains + config.secondary_domains))
    return V92CompleteSystem(config)


def create_v92_validator(config: Optional[V92Config] = None) -> V92CompleteSystem:
    """Validator preset: conservative - high confidence bar, low novelty."""
    config = config or V92Config()
    config.confidence_threshold = max(config.confidence_threshold, 0.9)
    config.exploration_rate = min(config.exploration_rate, 0.1)
    return V92CompleteSystem(config)


def create_v92_mathematician(config: Optional[V92Config] = None) -> V92CompleteSystem:
    """Mathematician preset: mathematics-first domains and depth."""
    config = config or V92Config()
    config.primary_domains = ["mathematics", "physics"]
    config.secondary_domains = ["computer_science"]
    config.synthesis_depth = max(config.synthesis_depth, 4)
    return V92CompleteSystem(config)


def create_v92_experimentalist(config: Optional[V92Config] = None) -> V92CompleteSystem:
    """Experimentalist preset: maximize experimental design throughput."""
    config = config or V92Config()
    config.max_experiments_per_design = max(config.max_experiments_per_design, 20)
    config.enable_experimental_design = True
    return V92CompleteSystem(config)
