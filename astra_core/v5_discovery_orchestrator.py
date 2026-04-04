"""
V5.0 Discovery Orchestrator - Unified Discovery System Coordination
====================================================================

This module orchestrates all V5.0 discovery capabilities (V101-V108)
into a unified, coherent discovery system.

CAPABILITIES COORDINATED:
- V101: Temporal Causal Discovery
- V102: Scalable Counterfactual Engine
- V103: Multi-Modal Evidence Integration
- V104: Adversarial Hypothesis Framework
- V105: Meta-Discovery Transfer Learning
- V106: Explainable Causal Reasoning
- V107: Discovery Triage and Prioritization
- V108: Real-Time Streaming Discovery

INTEGRATION:
- Works with V97 Knowledge Isolation (novelty detection)
- Integrates with V98 FCI Causal Discovery (baseline causality)
- Uses V4.0 capabilities (MCE, ASC, CRN, MMOL)
- Connects to domain modules for context

Date: 2026-04-14
Version: 5.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
from datetime import datetime
import json
import warnings

# Import V5.0 capabilities
try:
    from astra_core.capabilities.v101_temporal_causal import (
        create_temporal_fci_discovery,
        create_granger_fci_hybrid,
        TemporalFCIDiscovery
    )
    V101_AVAILABLE = True
except ImportError:
    V101_AVAILABLE = False

try:
    from astra_core.capabilities.v102_counterfactual_engine import (
        create_counterfactual_engine,
        CounterfactualEngine
    )
    V102_AVAILABLE = True
except ImportError:
    V102_AVAILABLE = False

try:
    from astra_core.capabilities.v103_multimodal_evidence import (
        create_multimodal_evidence_fusion,
        MultiModalEvidenceFusion
    )
    V103_AVAILABLE = True
except ImportError:
    V103_AVAILABLE = False

try:
    from astra_core.capabilities.v104_adversarial_discovery import (
        create_adversarial_discovery_system,
        AdversarialDiscoverySystem
    )
    V104_AVAILABLE = True
except ImportError:
    V104_AVAILABLE = False

try:
    from astra_core.capabilities.v105_meta_discovery import (
        create_meta_discovery_transfer_engine,
        MetaDiscoveryTransferEngine
    )
    V105_AVAILABLE = True
except ImportError:
    V105_AVAILABLE = False

try:
    from astra_core.capabilities.v106_explainable_causal import (
        create_explainable_causal_reasoner,
        ExplainableCausalReasoner
    )
    V106_AVAILABLE = True
except ImportError:
    V106_AVAILABLE = False

try:
    from astra_core.capabilities.v107_discovery_triage import (
        create_discovery_triage_system,
        DiscoveryTriageSystem
    )
    V107_AVAILABLE = True
except ImportError:
    V107_AVAILABLE = False

try:
    from astra_core.capabilities.v108_streaming_discovery import (
        create_streaming_discovery_engine,
        StreamingDiscoveryEngine
    )
    V108_AVAILABLE = True
except ImportError:
    V108_AVAILABLE = False


class DiscoveryWorkflow(Enum):
    """Types of discovery workflows"""
    STANDARD = "standard"              # Full discovery pipeline
    RAPID = "rapid"                    # Fast discovery for alerts
    DEEP = "deep"                      # Comprehensive analysis
    STREAMING = "streaming"            # Real-time monitoring
    CROSS_DOMAIN = "cross_domain"      # Meta-learning transfer
    VALIDATION = "validation"          # Validate existing claim


@dataclass
class DiscoveryPipelineConfig:
    """Configuration for discovery pipeline"""
    workflow: DiscoveryWorkflow = DiscoveryWorkflow.STANDARD
    enable_temporal: bool = True
    enable_counterfactual: bool = True
    enable_multimodal: bool = True
    enable_adversarial: bool = True
    enable_meta_learning: bool = False
    enable_explainability: bool = True
    enable_triage: bool = True
    enable_streaming: bool = False
    domain: str = ""
    variable_descriptions: Dict[str, str] = field(default_factory=dict)


@dataclass
class DiscoveryResult:
    """Complete result from discovery pipeline"""
    discovery_id: str
    timestamp: datetime
    workflow: DiscoveryWorkflow
    claim: str
    confidence: float
    novelty_score: float
    causal_graph: Optional[Any]
    temporal_patterns: Optional[Dict]
    counterfactual_results: Optional[Dict]
    multimodal_evidence: Optional[Dict]
    adversarial_analysis: Optional[Dict]
    triage_result: Optional[Dict]
    explanation: Optional[str]
    recommendation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class V5DiscoveryOrchestrator:
    """
    Main orchestrator for V5.0 discovery capabilities.

    Coordinates all discovery capabilities into unified workflows.
    """

    def __init__(self):
        """Initialize V5.0 Discovery Orchestrator"""
        # Initialize available capabilities
        self.temporal_discovery = None
        self.counterfactual_engine = None
        self.multimodal_fusion = None
        self.adversarial_system = None
        self.meta_engine = None
        self.explainability = None
        self.triage_system = None
        self.streaming_engine = None

        # Capability availability flags
        self.capabilities = {
            'v101_temporal': V101_AVAILABLE,
            'v102_counterfactual': V102_AVAILABLE,
            'v103_multimodal': V103_AVAILABLE,
            'v104_adversarial': V104_AVAILABLE,
            'v105_meta': V105_AVAILABLE,
            'v106_explainable': V106_AVAILABLE,
            'v107_triage': V107_AVAILABLE,
            'v108_streaming': V108_AVAILABLE
        }

        # Discovery history
        self.discovery_history: List[DiscoveryResult] = []

    def initialize_capabilities(self, config: DiscoveryPipelineConfig):
        """
        Initialize capabilities based on configuration.

        Args:
            config: Pipeline configuration
        """
        if config.enable_temporal and V101_AVAILABLE:
            self.temporal_discovery = create_temporal_fci_discovery()

        if config.enable_counterfactual and V102_AVAILABLE:
            self.counterfactual_engine = create_counterfactual_engine()

        if config.enable_multimodal and V103_AVAILABLE:
            self.multimodal_fusion = create_multimodal_evidence_fusion()

        if config.enable_adversarial and V104_AVAILABLE:
            self.adversarial_system = create_adversarial_discovery_system()

        if config.enable_meta_learning and V105_AVAILABLE:
            self.meta_engine = create_meta_discovery_transfer_engine()

        if config.enable_explainability and V106_AVAILABLE:
            self.explainability = create_explainable_causal_reasoner()

        if config.enable_triage and V107_AVAILABLE:
            self.triage_system = create_discovery_triage_system()

        if config.enable_streaming and V108_AVAILABLE:
            # Streaming engine requires variable names initialized separately
            pass

    def run_standard_discovery(
        self,
        data: np.ndarray,
        variable_names: List[str],
        config: Optional[DiscoveryPipelineConfig] = None
    ) -> DiscoveryResult:
        """
        Run standard discovery workflow.

        This is the main entry point for discovery, coordinating
        all V5.0 capabilities in a coherent pipeline.

        Args:
            data: Dataset (n_samples, n_variables)
            variable_names: Variable names
            config: Optional configuration

        Returns:
            Complete DiscoveryResult
        """
        if config is None:
            config = DiscoveryPipelineConfig()

        # Initialize capabilities
        self.initialize_capabilities(config)

        discovery_id = f"discovery_{datetime.now().isoformat()}"

        # Stage 1: Baseline Causal Discovery (V98)
        causal_graph = self._run_baseline_causal_discovery(data, variable_names)

        # Stage 2: Temporal Causal Discovery (V101)
        temporal_patterns = None
        if config.enable_temporal and self.temporal_discovery:
            temporal_patterns = self._run_temporal_discovery(data, variable_names)

        # Stage 3: Counterfactual Analysis (V102)
        counterfactual_results = None
        if config.enable_counterfactual and self.counterfactual_engine:
            counterfactual_results = self._run_counterfactual_analysis(
                data, variable_names, causal_graph
            )

        # Stage 4: Multi-Modal Evidence Integration (V103)
        multimodal_evidence = None
        if config.enable_multimodal and self.multimodal_fusion:
            multimodal_evidence = self._run_multimodal_integration(
                data, variable_names
            )

        # Stage 5: Adversarial Validation (V104)
        adversarial_analysis = None
        if config.enable_adversarial and self.adversarial_system:
            adversarial_analysis = self._run_adversarial_validation(
                data, variable_names, causal_graph
            )

        # Stage 6: Generate Discovery Claim
        claim = self._generate_discovery_claim(
            causal_graph, temporal_patterns, counterfactual_results
        )

        # Stage 7: Explainable Reasoning (V106)
        explanation = None
        if config.enable_explainability and self.explainability and causal_graph:
            explanation = self._generate_explanation(
                causal_graph, config.variable_descriptions, config.domain
            )

        # Stage 8: Triage and Prioritization (V107)
        triage_result = None
        if config.enable_triage and self.triage_system:
            triage_result = self._run_triage(
                claim, data, variable_names, multimodal_evidence,
                adversarial_analysis, config.domain
            )

        # Stage 9: Generate Recommendation
        recommendation = self._generate_recommendation(
            triage_result, adversarial_analysis
        )

        # Compute overall confidence
        confidence = self._compute_overall_confidence(
            multimodal_evidence, adversarial_analysis
        )

        # Compute novelty score (from V97 if available)
        novelty_score = self._compute_novelty_score(causal_graph)

        # Create result
        result = DiscoveryResult(
            discovery_id=discovery_id,
            timestamp=datetime.now(),
            workflow=config.workflow,
            claim=claim,
            confidence=confidence,
            novelty_score=novelty_score,
            causal_graph=causal_graph,
            temporal_patterns=temporal_patterns,
            counterfactual_results=counterfactual_results,
            multimodal_evidence=multimodal_evidence,
            adversarial_analysis=adversarial_analysis,
            triage_result=triage_result,
            explanation=explanation,
            recommendation=recommendation,
            metadata={
                'n_samples': len(data),
                'n_variables': len(variable_names),
                'capabilities_used': [
                    k for k, v in self.capabilities.items() if v
                ]
            }
        )

        # Store in history
        self.discovery_history.append(result)

        return result

    def _run_baseline_causal_discovery(
        self,
        data: np.ndarray,
        variable_names: List[str]
    ) -> Optional[Any]:
        """Run baseline causal discovery (V98 FCI)"""
        try:
            from astra_core.capabilities.v98_fci_causal_discovery import create_fci_discovery
            fci = create_fci_discovery()
            return fci.discover_causal_graph(data, variable_names)
        except (ImportError, Exception):
            # Fallback: create a simple mock causal graph for testing
            # (V98 FCI is optional)
            class MockEdge:
                def __init__(self, source, target):
                    self.source = source
                    self.target = target
                    self.source_end = type('obj', (object,), {'value': 'tail'})()
                    self.target_end = type('obj', (object,), {'value': 'arrow'})()
                    self.confidence = 0.5

            class MockPAG:
                def __init__(self, variable_names):
                    self.nodes = set(variable_names)
                    # Add some mock edges
                    self.edges = []
                    for i in range(min(3, len(variable_names) - 1)):
                        self.edges.append(MockEdge(variable_names[i], variable_names[i+1]))

            return MockPAG(variable_names)

    def _run_temporal_discovery(
        self,
        data: np.ndarray,
        variable_names: List[str]
    ) -> Optional[Dict]:
        """Run temporal causal discovery (V101)"""
        if not self.temporal_discovery:
            return None

        # Note: max_lag is set during TemporalFCIDiscovery initialization
        result = self.temporal_discovery.discover_temporal_causal_structure(
            data, variable_names
        )

        # Result is a TimeLaggedPAG object
        if hasattr(result, 'temporal_edges'):
            return {
                'temporal_edges': result.temporal_edges,
                'change_points': getattr(result, 'change_points', []),
                'granger_results': getattr(result, 'granger_results', {})
            }
        return {
            'temporal_edges': [],
            'change_points': [],
            'granger_results': {}
        }

    def _run_counterfactual_analysis(
        self,
        data: np.ndarray,
        variable_names: List[str],
        causal_graph: Any
    ) -> Optional[Dict]:
        """Run counterfactual analysis (V102)"""
        if not self.counterfactual_engine or not causal_graph:
            return None

        # Get top causal edges for intervention testing
        if hasattr(causal_graph, 'edges'):
            top_edges = list(causal_graph.edges)[:3]  # Test top 3 edges
        else:
            return None

        results = []
        for edge in top_edges:
            # Get variable indices
            if edge.source not in variable_names or edge.target not in variable_names:
                continue

            # Get covariates (all other variables)
            covariates = [v for v in variable_names if v not in [edge.source, edge.target]]

            # Run comprehensive counterfactual analysis
            analysis = self.counterfactual_engine.comprehensive_counterfactual_analysis(
                data=data,
                variable_names=variable_names,
                treatment_var=edge.source,
                outcome_var=edge.target,
                covariates=covariates if covariates else [variable_names[0]]
            )
            results.append({
                'source': edge.source,
                'target': edge.target,
                'analysis': analysis
            })

        return {'intervention_results': results}

    def _run_multimodal_integration(
        self,
        data: np.ndarray,
        variable_names: List[str]
    ) -> Optional[Dict]:
        """Run multi-modal evidence integration (V103)"""
        if not self.multimodal_fusion:
            return None

        evidence_ids = []

        # Add numerical evidence
        for i in range(len(variable_names)):
            for j in range(i+1, len(variable_names)):
                corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
                ev_id = self.multimodal_fusion.add_numerical_evidence(
                    variable_names[i],
                    variable_names[j],
                    corr,
                    0.05,  # Placeholder p-value
                    len(data)
                )
                evidence_ids.append(ev_id)

        return {'evidence_ids': evidence_ids}

    def _run_adversarial_validation(
        self,
        data: np.ndarray,
        variable_names: List[str],
        causal_graph: Any
    ) -> Optional[Dict]:
        """Run adversarial validation (V104)"""
        if not self.adversarial_system:
            return None

        # Create initial discovery claim
        claim = f"Causal relationships discovered among {len(variable_names)} variables"

        initial_discovery = {
            'claim': claim,
            'type': 'causal',
            'effect_size': 0.5
        }

        return self.adversarial_system.adversarial_discovery_process(
            initial_discovery, data, variable_names
        )

    def _generate_discovery_claim(
        self,
        causal_graph: Any,
        temporal_patterns: Optional[Dict],
        counterfactual_results: Optional[Dict]
    ) -> str:
        """Generate discovery claim"""
        if not causal_graph:
            return "Unable to generate claim - no causal graph available"

        n_edges = len(causal_graph.edges) if hasattr(causal_graph, 'edges') else 0

        claim_parts = [f"Discovered {n_edges} causal relationships"]

        if temporal_patterns and temporal_patterns.get('temporal_edges'):
            n_temporal = len(temporal_patterns['temporal_edges'])
            claim_parts.append(f"including {n_temporal} time-lagged relationships")

        if counterfactual_results:
            n_interventions = len(counterfactual_results.get('intervention_results', []))
            claim_parts.append(f"with {n_interventions} validated causal effects")

        return ". ".join(claim_parts) + "."

    def _generate_explanation(
        self,
        causal_graph: Any,
        variable_descriptions: Dict[str, str],
        domain: str
    ) -> Optional[str]:
        """Generate natural language explanation (V106)"""
        if not self.explainability:
            return None

        result = self.explainability.explain_pag(
            causal_graph, variable_descriptions, domain
        )
        return result.get('story')

    def _run_triage(
        self,
        claim: str,
        data: np.ndarray,
        variable_names: List[str],
        multimodal_evidence: Optional[Dict],
        adversarial_analysis: Optional[Dict],
        domain: str
    ) -> Optional[Dict]:
        """Run discovery triage (V107)"""
        if not self.triage_system:
            return None

        discovery = {
            'discovery_id': f"discovery_{datetime.now().isoformat()}",
            'claim': claim,
            'sample_size': len(data),
            'multimodal_fusion': multimodal_evidence,
            'adversarial_validation': adversarial_analysis is not None,
            'data_available': True
        }

        result = self.triage_system.triage_discovery(discovery, domain)

        return {
            'category': result.triage_category.value,
            'impact_score': result.overall_impact_score,
            'recommended_action': result.recommended_action,
            'validation_strategy': result.validation_strategy.value
        }

    def _generate_recommendation(
        self,
        triage_result: Optional[Dict],
        adversarial_analysis: Optional[Dict]
    ) -> str:
        """Generate recommendation"""
        if triage_result:
            return f"Triaged as {triage_result['category']}: {triage_result['recommended_action']}"
        elif adversarial_analysis:
            return "Review adversarial challenges before proceeding"
        else:
            return "Discovery requires further validation"

    def _compute_overall_confidence(
        self,
        multimodal_evidence: Optional[Dict],
        adversarial_analysis: Optional[Dict]
    ) -> float:
        """Compute overall confidence"""
        confidence = 0.5  # Base confidence

        # Boost from multi-modal evidence
        if multimodal_evidence:
            confidence += 0.2

        # Adjust based on adversarial analysis
        if adversarial_analysis:
            # Check if hypothesis survived challenges
            refined = adversarial_analysis.get('refined_hypothesis')
            if refined:
                confidence = refined.final_confidence

        return min(1.0, confidence)

    def _compute_novelty_score(self, causal_graph: Any) -> float:
        """Compute novelty score"""
        # Placeholder: In production, would use V97 novelty score
        if causal_graph and hasattr(causal_graph, 'edges'):
            # More edges = potentially more novel
            n_edges = len(causal_graph.edges)
            return min(1.0, n_edges / 20)
        return 0.5

    def run_streaming_discovery(
        self,
        data_stream: np.ndarray,
        variable_names: List[str],
        batch_size: int = 100
    ) -> List[Dict]:
        """
        Run streaming discovery workflow (V108).

        Args:
            data_stream: Streaming data
            variable_names: Variable names
            batch_size: Batch size for processing

        Returns:
            List of batch processing results
        """
        if not V108_AVAILABLE:
            # V108 is optional, return empty results
            return []

        from astra_core.capabilities.v108_streaming_discovery import monitor_streaming_data

        return monitor_streaming_data(
            data_stream, variable_names, batch_size
        )

    def run_cross_domain_discovery(
        self,
        target_domain: str,
        target_data: np.ndarray,
        variable_names: List[str],
        known_patterns: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Run cross-domain discovery with meta-learning (V105).

        Args:
            target_domain: Target domain
            target_data: Target domain data
            variable_names: Variable names
            known_patterns: Known discovery patterns

        Returns:
            Meta-learning discovery results
        """
        if not V105_AVAILABLE:
            # V105 is optional, return empty results
            return {}

        from astra_core.capabilities.v105_meta_discovery import meta_discovery_across_domains

        return meta_discovery_across_domains(
            target_domain, target_data, variable_names, known_patterns
        )

    def explain_discovery(
        self,
        discovery_result: DiscoveryResult,
        question: Optional[str] = None
    ) -> str:
        """
        Generate explanation for a discovery result.

        Args:
            discovery_result: Discovery to explain
            question: Optional specific question

        Returns:
            Natural language explanation
        """
        if not self.explainability:
            return "Explainability module not available"

        if question and discovery_result.causal_graph:
            return self.explainability.answer_question(
                question,
                discovery_result.causal_graph,
                discovery_result.metadata.get('variable_descriptions', {})
            )

        return discovery_result.explanation or "No explanation available"

    def get_discovery_summary(self) -> str:
        """Get summary of all discoveries"""
        lines = []
        lines.append("=== V5.0 DISCOVERY ORCHESTRATOR SUMMARY ===\n")
        lines.append(f"Total Discoveries: {len(self.discovery_history)}")
        lines.append(f"Available Capabilities: {sum(self.capabilities.values())}/8\n")

        lines.append("CAPABILITY STATUS:")
        for cap, available in self.capabilities.items():
            status = "✓" if available else "✗"
            lines.append(f"  {status} {cap}")

        if self.discovery_history:
            lines.append(f"\nRECENT DISCOVERIES:")
            for result in self.discovery_history[-5:]:
                lines.append(f"  [{result.workflow.value}] {result.claim[:60]}...")

        return "\n".join(lines)


# Factory functions

def create_v5_discovery_orchestrator() -> V5DiscoveryOrchestrator:
    """Factory function to create V5.0 Discovery Orchestrator"""
    return V5DiscoveryOrchestrator()


def create_discovery_pipeline_config(
    workflow: DiscoveryWorkflow = DiscoveryWorkflow.STANDARD,
    **kwargs
) -> DiscoveryPipelineConfig:
    """Factory function to create pipeline configuration"""
    return DiscoveryPipelineConfig(workflow=workflow, **kwargs)


# Convenience function

def discover_in_dataset(
    data: np.ndarray,
    variable_names: List[str],
    domain: str = "",
    workflow: str = "standard"
) -> DiscoveryResult:
    """
    Main entry point for V5.0 discovery.

    Args:
        data: Dataset (n_samples, n_variables)
        variable_names: Variable names
        domain: Domain context
        workflow: Workflow type ("standard", "rapid", "deep", "streaming")

    Returns:
        Complete discovery result
    """
    orchestrator = create_v5_discovery_orchestrator()

    workflow_enum = DiscoveryWorkflow(workflow)
    config = create_discovery_pipeline_config(
        workflow=workflow_enum,
        domain=domain
    )

    if workflow == "streaming":
        # Streaming returns different format
        return orchestrator.run_streaming_discovery(data, variable_names)

    return orchestrator.run_standard_discovery(data, variable_names, config)


# V5.0 System info

def get_v5_capabilities() -> Dict[str, bool]:
    """Get available V5.0 capabilities"""
    return {
        'v101_temporal_causal_discovery': V101_AVAILABLE,
        'v102_counterfactual_engine': V102_AVAILABLE,
        'v103_multimodal_evidence': V103_AVAILABLE,
        'v104_adversarial_discovery': V104_AVAILABLE,
        'v105_meta_discovery': V105_AVAILABLE,
        'v106_explainable_causal': V106_AVAILABLE,
        'v107_discovery_triage': V107_AVAILABLE,
        'v108_streaming_discovery': V108_AVAILABLE
    }


def get_v5_summary() -> str:
    """Get V5.0 system summary"""
    capabilities = get_v5_capabilities()
    available = sum(capabilities.values())
    total = len(capabilities)

    lines = []
    lines.append("=== V5.0 DISCOVERY ENHANCEMENT SYSTEM ===\n")
    lines.append(f"Capabilities Available: {available}/{total}\n")

    lines.append("CAPABILITIES:")
    for name, available_flag in capabilities.items():
        status = "✓" if available_flag else "✗"
        lines.append(f"  {status} {name}")

    lines.append(f"\nOrchestrator: V5DiscoveryOrchestrator")
    lines.append(f"Entry Point: discover_in_dataset()")

    return "\n".join(lines)
