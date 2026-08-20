"""
Scientific Discovery Orchestrator - Autonomous Research Conductor
=================================================================

Central coordinator for autonomous scientific discovery in astronomy
and astrophysics. Orchestrates the complete discovery cycle from
literature review through hypothesis generation, experimental design,
analysis, and synthesis.

6-Phase Discovery Loop:
1. LITERATURE REVIEW: Read and synthesize research papers
2. DATA GATHERING: Access databases and archives
3. HYPOTHESIS GENERATION: Generate novel hypotheses
4. EXPERIMENTAL DESIGN: Design experiments and observations
5. ANALYSIS & TESTING: Execute analysis and test hypotheses
6. SYNTHESIS: Integrate findings and discover new knowledge

Integrations:
- V41 Orchestrator: Complex reasoning and metacognition
- V50 Discovery Engine: World simulation and program synthesis
- V92 Scientific Discovery: Hypothesis generation and experimental design
- AstroSwarm: Physics-based Bayesian inference
- Integration Bus: Event-driven communication
- MORK: Knowledge persistence

Version: 1.0.0
Date: 2025-12-27
"""

try:
    import time
except Exception:
    time = None  # degraded: unavailable
try:
    import uuid
except Exception:
    uuid = None  # degraded: unavailable
try:
    import logging
except Exception:
    logging = None  # degraded: unavailable
try:
    from dataclasses import dataclass, field
except Exception:
    dataclass = field = None  # degraded: unavailable
try:
    from typing import Dict, List, Optional, Any, Tuple
except Exception:
    Dict = List = Optional = Any = Tuple = None  # degraded: unavailable
try:
    from enum import Enum
except Exception:
    Enum = None  # degraded: unavailable
try:
    from pathlib import Path
except Exception:
    Path = None  # degraded: unavailable
try:
    import json
except Exception:
    json = None  # degraded: unavailable

# Import discovery components
try:
    from .adaptive_reasoning import (
    AdaptiveReasoningController, DiscoveryPhase,
    get_adaptive_reasoning_controller
    )
except Exception:
    AdaptiveReasoningController = DiscoveryPhase = get_adaptive_reasoning_controller = None  # degraded: unavailable
try:
    from .feasibility_checker import (
    FeasibilityAssessor, SafetyLimits, FeasibilityResult,
    create_feasibility_assessor
    )
except Exception:
    FeasibilityAssessor = SafetyLimits = FeasibilityResult = create_feasibility_assessor = None  # degraded: unavailable

# Import V41, V50, V92 components (try both relative and absolute imports)
try:
    from ..reasoning.v41_orchestrator import (
        V41Orchestrator, ReasoningMode, ReasoningTask, ReasoningResult
    )
    from ..reasoning.integration_bus import (
        get_integration_bus, Event, EventType, EventPriority
    )
    from ..core_legacy.v92.v92_system import V92CompleteSystem
    from ..core_legacy.v50.v50_discovery_engine import V50DiscoveryEngine
    HAS_V41_V50_V92 = True
except (ImportError, ValueError):
    try:
        # Fallback to absolute imports
        from astra_core.reasoning.v41_orchestrator import (
            V41Orchestrator, ReasoningMode, ReasoningTask, ReasoningResult
        )
        from astra_core.reasoning.integration_bus import (
            get_integration_bus, Event, EventType, EventPriority
        )
        from astra_core.core_legacy.v92.v92_system import V92CompleteSystem
        from astra_core.core_legacy.v50.v50_discovery_engine import V50DiscoveryEngine
        HAS_V41_V50_V92 = True
    except ImportError as e:
        logging.warning(f"Could not import V41/V50/V92 components: {e}")
        HAS_V41_V50_V92 = False

# Import AstroSwarm
try:
    from ..astro_physics.core import AstroSwarmSystem
    HAS_ASTROSWARM = True
except (ImportError, ValueError):
    try:
        from astra_core.astro_physics.core import AstroSwarmSystem
        HAS_ASTROSWARM = True
    except ImportError as e:
        logging.warning(f"Could not import AstroSwarm: {e}")
        HAS_ASTROSWARM = False

# Import MORK
try:
    from ..swarm.client import MORKClient
    HAS_MORK = True
except (ImportError, ValueError):
    try:
        from astra_core.swarm.client import MORKClient
        HAS_MORK = True
    except ImportError as e:
        logging.warning(f"Could not import MORK: {e}")
        HAS_MORK = False

logger = logging.getLogger(__name__)


# =============================================================================
# Discovery Task and Result Dataclasses
# =============================================================================

@dataclass
class DiscoveryTask:
    """A scientific discovery task"""
    task_id: str
    research_question: str
    domain: str = "astrophysics"

    # Configuration
    enable_literature_review: bool = True
    enable_data_access: bool = True
    enable_hypothesis_generation: bool = True
    enable_experimental_design: bool = True
    enable_analysis: bool = True
    enable_simulations: bool = True

    # Constraints
    max_time_hours: float = 48.0
    max_papers: int = 50
    max_data_gb: float = 100.0
    safety_limits: Optional[SafetyLimits] = None

    # Context
    prior_knowledge: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)

    # Status tracking
    status: str = "pending"
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def __post_init__(self):
        if not self.task_id:
            self.task_id = f"DISCOVERY-{uuid.uuid4().hex[:8]}"


@dataclass
class Hypothesis:
    """A scientific hypothesis"""
    hypothesis_id: str
    statement: str
    domain: str
    plausibility: float  # 0-1
    testability: float  # 0-1
    novelty: float  # 0-1
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    proposed_tests: List[str] = field(default_factory=list)

    def overall_score(self) -> float:
        """Calculate overall hypothesis score"""
        return (0.4 * self.plausibility +
                0.3 * self.testability +
                0.3 * self.novelty)


@dataclass
class ExperimentProposal:
    """Proposed experiment or observation"""
    experiment_id: str
    description: str
    experiment_type: str  # 'observational', 'computational', 'theoretical'
    target_hypothesis: str

    # Feasibility
    feasibility: Optional[FeasibilityResult] = None

    # Requirements
    required_data: List[str] = field(default_factory=list)
    required_compute: Dict[str, float] = field(default_factory=dict)
    required_time_hours: float = 1.0

    # Expected outcomes
    expected_outcomes: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)


@dataclass
class LiteratureReview:
    """Results of literature review"""
    num_papers_reviewed: int
    key_findings: List[str]
    identified_gaps: List[str]
    extracted_hypotheses: List[Hypothesis]
    citation_network_stats: Dict[str, Any]
    synthesis_summary: str


@dataclass
class DiscoveryResult:
    """Complete results of discovery process"""
    task_id: str
    research_question: str
    success: bool

    # Phase results
    literature_review: Optional[LiteratureReview] = None
    data_gathered: Dict[str, Any] = field(default_factory=dict)
    hypotheses_generated: List[Hypothesis] = field(default_factory=list)
    experiments_proposed: List[ExperimentProposal] = field(default_factory=list)
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    synthesized_knowledge: Dict[str, Any] = field(default_factory=dict)

    # Discoveries
    novel_insights: List[str] = field(default_factory=list)
    new_research_directions: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)

    # Metadata
    total_time_hours: float = 0.0
    phases_completed: List[str] = field(default_factory=list)
    reasoning_trace: List[Dict[str, Any]] = field(default_factory=list)

    # Assessment
    discovery_quality: float = 0.0
    novelty_score: float = 0.0
    impact_assessment: str = ""

    def summary(self) -> str:
        """Generate summary string"""
        status = "SUCCESS" if self.success else "INCOMPLETE"
        return f"""
{'='*60}
DISCOVERY RESULT: {status}
{'='*60}
Research Question: {self.research_question}
Total Time: {self.total_time_hours:.2f} hours
Phases Completed: {', '.join(self.phases_completed)}

Literature Review:
  - Papers Reviewed: {self.literature_review.num_papers_reviewed if self.literature_review else 0}
  - Key Findings: {len(self.literature_review.key_findings) if self.literature_review else 0}

Hypotheses Generated: {len(self.hypotheses_generated)}
Experiments Proposed: {len(self.experiments_proposed)}
Novel Insights: {len(self.novel_insights)}

Discovery Quality: {self.discovery_quality:.2f}
Novelty Score: {self.novelty_score:.2f}

Top Hypotheses:
{self._format_top_hypotheses()}

New Research Directions:
{self._format_research_directions()}
{'='*60}
"""

    def _format_top_hypotheses(self) -> str:
        """Format top 3 hypotheses"""
        if not self.hypotheses_generated:
            return "  (none)"

        sorted_hyp = sorted(self.hypotheses_generated,
                           key=lambda h: h.overall_score(),
                           reverse=True)
        lines = []
        for i, hyp in enumerate(sorted_hyp[:3], 1):
            lines.append(f"  {i}. {hyp.statement}")
            lines.append(f"     Score: {hyp.overall_score():.2f} "
                        f"(P={hyp.plausibility:.2f}, "
                        f"T={hyp.testability:.2f}, "
                        f"N={hyp.novelty:.2f})")
        return "\n".join(lines)

    def _format_research_directions(self) -> str:
        """Format research directions"""
        if not self.new_research_directions:
            return "  (none)"
        return "\n".join(f"  - {d}" for d in self.new_research_directions[:5])


# =============================================================================
# Scientific Discovery Orchestrator
# =============================================================================

class ScientificDiscoveryOrchestrator:
    """
    Main orchestrator for autonomous scientific discovery.

    Coordinates the complete 6-phase discovery cycle, integrating
    V41/V50/V92 reasoning systems with AstroSwarm inference and
    domain-specific analysis capabilities.
    """

    def __init__(self,
                 safety_limits: Optional[SafetyLimits] = None,
                 storage_path: Optional[Path] = None,
                 enable_mork: bool = True):
        """
        Initialize discovery orchestrator.

        Args:
            safety_limits: Resource and safety constraints
            storage_path: Path for storing discovery results
            enable_mork: Whether to use MORK persistence
        """
        # Core controllers
        self.reasoning_controller = get_adaptive_reasoning_controller()
        self.feasibility_assessor = create_feasibility_assessor(
            custom_limits=safety_limits
        )

        # Initialize subsystems
        self._init_v41_v50_v92()
        self._init_astroswarm()
        self._init_mork(enable_mork)
        self._init_integration_bus()

        # Storage
        self.storage_path = storage_path or Path.home() / ".stan_discovery"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # State tracking
        self.current_task: Optional[DiscoveryTask] = None
        self.discovery_history: List[DiscoveryResult] = []

    # ==================================================================
    # Subsystem initialisation
    # (re-implemented 2026-08; bodies lost to file truncation before the
    #  audit. V92/V50 engines lived in the retired core_legacy package.)
    # ==================================================================

    def _init_v41_v50_v92(self) -> None:
        """Attach V41 reasoning orchestrator (V92/V50 legacy: removed)."""
        v41_cls = globals().get('V41Orchestrator')
        self.v41_orchestrator = None
        if v41_cls is not None:
            try:
                self.v41_orchestrator = v41_cls()
            except Exception as e:
                logger.warning(f"V41 orchestrator unavailable: {e}")
        self.v92_system = None    # degraded: core_legacy removed
        self.v50_engine = None    # degraded: core_legacy removed

    def _init_astroswarm(self) -> None:
        """Attach ASTRO-SWARM inference system."""
        self.astroswarm = None
        if HAS_ASTROSWARM:
            try:
                self.astroswarm = AstroSwarmSystem()
            except Exception as e:
                logger.warning(f"AstroSwarm unavailable: {e}")

    def _init_mork(self, enable_mork: bool) -> None:
        """Attach MORK persistence client."""
        self.mork_client = None
        if enable_mork and HAS_MORK:
            try:
                self.mork_client = MORKClient(storage_mode="local")
            except Exception as e:
                logger.warning(f"MORK client unavailable: {e}")

    def _init_integration_bus(self) -> None:
        """Attach the shared reasoning integration bus."""
        get_bus = globals().get('get_integration_bus')
        self.integration_bus = None
        if get_bus is not None:
            try:
                self.integration_bus = get_bus()
            except Exception as e:
                logger.warning(f"Integration bus unavailable: {e}")

    # ==================================================================
    # Discovery pipeline
    # ==================================================================

    def run_discovery(self, task: DiscoveryTask) -> DiscoveryResult:
        """
        Execute the discovery cycle for a task.

        Phases run in order (literature review -> hypothesis generation ->
        experimental design -> synthesis), each gated by the task's
        enable_* flags and by component availability.  Every phase appends
        to the reasoning trace; skipped phases are simply absent from
        phases_completed.
        """
        start = time.time()
        task.status = "running"
        task.started_at = start
        self.current_task = task

        result = DiscoveryResult(
            task_id=task.task_id,
            research_question=task.research_question,
            success=False,
        )

        # Phase 1: literature review ------------------------------------
        if task.enable_literature_review:
            self._phase_literature_review(task, result)

        # Phase 2: hypothesis generation --------------------------------
        if task.enable_hypothesis_generation:
            self._phase_hypothesis_generation(task, result)

        # Phase 3: experimental design ----------------------------------
        if task.enable_experimental_design:
            self._phase_experimental_design(task, result)

        # Phase 4: synthesis --------------------------------------------
        self._phase_synthesis(task, result)

        task.completed_at = time.time()
        task.status = "completed"
        result.total_time_hours = (task.completed_at - start) / 3600.0
        result.success = bool(result.phases_completed)
        self.discovery_history.append(result)
        self._persist_result(result)
        return result

    # ------------------------------------------------------------------
    def _phase_literature_review(self, task: DiscoveryTask,
                                 result: DiscoveryResult) -> None:
        """Query the local paper library for prior work on the question."""
        num_papers, findings, gaps, extracted = 0, [], [], []
        try:
            from .paper_rag_query import PaperRAGSystem
            rag = PaperRAGSystem()
            stats = rag.library.get_stats()
            if stats['total_papers'] > 0:
                qr = rag.query(task.research_question, k=min(5, task.max_papers))
                num_papers = stats['total_papers']
                findings = [f"{s['citation']}: {s['title']}"
                            for s in qr.sources]
        except Exception as e:
            logger.warning(f"Literature review degraded: {e}")

        # Gaps: question facets with no library coverage
        facets = self._question_facets(task.research_question)
        covered_terms = {' '.join(f.lower().split() for f in findings)}
        for facet in facets:
            if not any(facet.lower() in c for c in covered_terms):
                gaps.append(f"No local literature coverage for '{facet}'")

        result.literature_review = LiteratureReview(
            num_papers_reviewed=num_papers,
            key_findings=findings,
            identified_gaps=gaps,
            extracted_hypotheses=extracted,
            citation_network_stats={'local_library_papers': num_papers},
            synthesis_summary=(f"Reviewed {num_papers} local papers; "
                               f"{len(gaps)} coverage gaps identified."),
        )
        result.phases_completed.append('literature_review')
        result.reasoning_trace.append({
            'phase': 'literature_review',
            'papers': num_papers, 'gaps': len(gaps),
        })

    def _phase_hypothesis_generation(self, task: DiscoveryTask,
                                     result: DiscoveryResult) -> None:
        """Generate candidate hypotheses from the question facets and gaps."""
        facets = self._question_facets(task.research_question)
        gaps = (result.literature_review.identified_gaps
                if result.literature_review else [])
        seeds = facets + [g.replace('No local literature coverage for ', '')
                          for g in gaps]

        hypotheses = []
        for i, seed in enumerate(seeds[:8]):
            statement = (f"Hypothesis {i + 1}: measurable physics connecting "
                         f"'{seed}' to '{task.research_question}'")
            # Testability: concrete measurable terms raise it
            measurable = any(w in seed.lower() for w in
                             ('mass', 'temperature', 'density', 'flux',
                              'velocity', 'luminosity', 'radius', 'rate'))
            # Novelty: uncovered facets are more novel
            novel = seed in gaps or any(g and seed in g for g in gaps)
            hypotheses.append(Hypothesis(
                hypothesis_id=f"HYP-{task.task_id[-8:]}-{i + 1}",
                statement=statement,
                domain=task.domain,
                plausibility=0.5,
                testability=0.7 if measurable else 0.4,
                novelty=0.7 if novel else 0.3,
                supporting_evidence=[],
                proposed_tests=[],
            ))
        result.hypotheses_generated = hypotheses
        result.phases_completed.append('hypothesis_generation')
        result.reasoning_trace.append({
            'phase': 'hypothesis_generation',
            'n_hypotheses': len(hypotheses),
        })

    def _phase_experimental_design(self, task: DiscoveryTask,
                                   result: DiscoveryResult) -> None:
        """Propose and feasibility-check experiments for top hypotheses."""
        top = sorted(result.hypotheses_generated,
                     key=lambda h: h.overall_score(), reverse=True)[:3]
        proposals = []
        for hyp in top:
            exp_type = ('computational' if hyp.testability < 0.5
                        else 'observational')
            description = (f"{exp_type.capitalize()} test of: "
                           f"{hyp.statement}")
            feasibility = None
            if self.feasibility_assessor is not None:
                try:
                    feasibility = self.feasibility_assessor.assess_experiment({
                        'description': description,
                        'type': exp_type,
                        'domain': task.domain,
                    })
                except Exception:
                    feasibility = None
            proposals.append(ExperimentProposal(
                experiment_id=f"EXP-{task.task_id[-8:]}-{len(proposals) + 1}",
                description=description,
                experiment_type=exp_type,
                target_hypothesis=hyp.hypothesis_id,
                feasibility=feasibility,
                required_data=[task.domain],
                expected_outcomes=['confirm', 'refute', 'inconclusive'],
                success_criteria=['detection significance > 3 sigma'],
            ))
        result.experiments_proposed = proposals
        result.phases_completed.append('experimental_design')
        result.reasoning_trace.append({
            'phase': 'experimental_design', 'n_experiments': len(proposals),
        })

    def _phase_synthesis(self, task: DiscoveryTask,
                         result: DiscoveryResult) -> None:
        """Aggregate phase outputs into insights and quality scores."""
        insights = []
        if result.literature_review:
            insights.extend(f"Prior work: {f}" for f in
                            result.literature_review.key_findings[:3])
        for hyp in sorted(result.hypotheses_generated,
                          key=lambda h: h.overall_score(), reverse=True)[:3]:
            insights.append(f"Candidate: {hyp.statement}")
        for exp in result.experiments_proposed:
            insights.append(f"Testable: {exp.description}")

        result.novel_insights = insights
        result.new_research_directions = [
            g for g in (result.literature_review.identified_gaps
                        if result.literature_review else [])
        ][:5]

        n_hyp = len(result.hypotheses_generated)
        n_exp = len(result.experiments_proposed)
        result.discovery_quality = min(1.0, 0.2 * len(result.phases_completed))
        result.novelty_score = (
            sum(h.novelty for h in result.hypotheses_generated) / n_hyp
            if n_hyp else 0.0)
        result.confidence_scores = {
            'hypothesis_coverage': min(1.0, n_hyp / 5.0),
            'experimental_coverage': min(1.0, n_exp / 3.0),
        }
        result.phases_completed.append('synthesis')
        result.reasoning_trace.append({'phase': 'synthesis'})

    # ------------------------------------------------------------------
    @staticmethod
    def _question_facets(question: str) -> List[str]:
        """Split a research question into investigateable facets."""
        stop = {'what', 'how', 'why', 'does', 'do', 'is', 'are', 'the', 'a',
                'an', 'of', 'in', 'on', 'for', 'to', 'and', 'or', 'with',
                'between', 'affect', 'affects', 'cause', 'causes'}
        words = [w for w in question.lower().split()
                 if w.strip('?,.') not in stop]
        facets = []
        for i in range(0, len(words), 3):
            facet = ' '.join(words[i:i + 3]).strip()
            if len(facet) > 3:
                facets.append(facet)
        return facets or [question]

    def _persist_result(self, result: DiscoveryResult) -> None:
        """Save the result summary as JSON under storage_path."""
        if json is None:
            return
        try:
            payload = {
                'task_id': result.task_id,
                'research_question': result.research_question,
                'success': result.success,
                'phases_completed': result.phases_completed,
                'n_hypotheses': len(result.hypotheses_generated),
                'n_experiments': len(result.experiments_proposed),
                'novel_insights': result.novel_insights,
                'discovery_quality': result.discovery_quality,
                'novelty_score': result.novelty_score,
                'total_time_hours': result.total_time_hours,
            }
            path = self.storage_path / f"{result.task_id}.json"
            path.write_text(json.dumps(payload, indent=2))
        except Exception as e:
            logger.warning(f"Could not persist discovery result: {e}")


# =============================================================================
# Module-level convenience API
# (exported by scientific_discovery/__init__.py)
# =============================================================================

def create_discovery_system(safety_limits: Optional[SafetyLimits] = None,
                            storage_path: Optional[Path] = None,
                            enable_mork: bool = True
                            ) -> ScientificDiscoveryOrchestrator:
    """Create a configured scientific discovery system."""
    return ScientificDiscoveryOrchestrator(
        safety_limits=safety_limits,
        storage_path=storage_path,
        enable_mork=enable_mork,
    )


def autonomous_discovery(research_question: str,
                         domain: str = "astrophysics",
                         max_time_hours: float = 48.0
                         ) -> DiscoveryResult:
    """Run one autonomous discovery cycle for a research question."""
    orchestrator = create_discovery_system()
    task = DiscoveryTask(
        task_id="",
        research_question=research_question,
        domain=domain,
        max_time_hours=max_time_hours,
    )
    return orchestrator.run_discovery(task)


def review_literature(research_question: str,
                      max_papers: int = 50) -> LiteratureReview:
    """Literature-review phase only, for a research question."""
    orchestrator = create_discovery_system()
    task = DiscoveryTask(task_id="", research_question=research_question,
                         enable_hypothesis_generation=False,
                         enable_experimental_design=False)
    result = DiscoveryResult(task_id=task.task_id,
                             research_question=research_question, success=False)
    orchestrator._phase_literature_review(task, result)
    return result.literature_review or LiteratureReview(
        num_papers_reviewed=0, key_findings=[], identified_gaps=[],
        extracted_hypotheses=[], citation_network_stats={},
        synthesis_summary="Literature review did not run.")


def propose_experiment(hypothesis_statement: str,
                       domain: str = "astrophysics"
                       ) -> ExperimentProposal:
    """Design and feasibility-check one experiment for a hypothesis."""
    orchestrator = create_discovery_system()
    hyp = Hypothesis(
        hypothesis_id="HYP-SINGLE",
        statement=hypothesis_statement,
        domain=domain,
        plausibility=0.5, testability=0.5, novelty=0.5,
    )
    task = DiscoveryTask(task_id="", research_question=hypothesis_statement,
                         enable_literature_review=False,
                         enable_hypothesis_generation=False)
    result = DiscoveryResult(task_id=task.task_id,
                             research_question=hypothesis_statement, success=False)
    result.hypotheses_generated = [hyp]
    orchestrator._phase_experimental_design(task, result)
    return result.experiments_proposed[0]
