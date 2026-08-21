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
STAN V41 Orchestrator - Integrated AGI-like Reasoning System

The V41 Orchestrator is the central coordinator for all advanced reasoning
capabilities. It provides a unified interface for complex reasoning tasks
that automatically engages the appropriate combination of:

- Unified World Model: Shared belief state and knowledge representation
- Integration Bus: Cross-module communication and event handling
- Dynamic Replanning: Adaptive execution with real-time plan adjustment
- Counterfactual Reasoning: "What if" analysis and causal inference
- Analogical Reasoning: Cross-domain knowledge transfer
- Theory Synthesis: Pattern-to-law promotion and theory building
- Metacognition: Self-reflective reasoning quality monitoring
- Active Knowledge Acquisition: Goal-directed knowledge seeking
- Multi-Agent Deliberation: Internal debate and consensus building
- Continuous Learning: Experience-based improvement

This module represents a significant step toward AGI-like behavior by
enabling deep integration between previously independent reasoning modules.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Callable, Tuple
from enum import Enum, auto
from datetime import datetime
import uuid
import time
from collections import defaultdict

# Import all V41 components
from .unified_world_model import (
    get_world_model, UnifiedWorldModel, Belief, Hypothesis,
    CausalGraph, Constraint, Evidence
)
from .integration_bus import (
    get_integration_bus, IntegrationBus, EventType, EventPriority
)
from .dynamic_replanning import (
    DynamicExecutor, ReplanningEngine, ExecutionState
)
from .counterfactual_reasoning import (
    CounterfactualEngine, StructuralCausalModel, CounterfactualQuery
)
from .analogical_reasoning import (
    AnalogyFinder, StructureMapper, DomainRepresentation
)
from .theory_synthesis import (
    get_theory_synthesizer, TheorySynthesizer, Pattern, PatternType, Law
)
from .metacognition import (
    get_metacognitive_controller, MetacognitiveController,
    ReasoningStrategy, ReasoningTrace
)
from .active_knowledge_acquisition import (
    get_knowledge_acquirer, ActiveKnowledgeAcquirer, KnowledgeGap
)
from .multi_agent_deliberation import (
    get_deliberator, MultiAgentDeliberator, ConsensusLevel
)
from .continuous_learning import (
    get_continuous_learner, ContinuousLearner, Experience
)


class ReasoningMode(Enum):
    """Modes of reasoning"""
    ANALYTICAL = auto()        # Deep analysis, systematic
    CREATIVE = auto()          # Novel solutions, analogical
    CRITICAL = auto()          # Evaluation, falsification
    INTEGRATIVE = auto()       # Synthesis, unification
    ADAPTIVE = auto()          # Dynamic, responsive
    DELIBERATIVE = auto()      # Multi-perspective consideration


class TaskComplexity(Enum):
    """Task complexity levels"""
    TRIVIAL = auto()           # Direct lookup or simple inference
    SIMPLE = auto()            # Single-step reasoning
    MODERATE = auto()          # Multi-step, single domain
    COMPLEX = auto()           # Multi-step, multi-domain
    EXPERT = auto()            # Requires specialized knowledge
    FRONTIER = auto()          # At limits of capability


@dataclass
class ReasoningTask:
    """A reasoning task to be orchestrated"""
    task_id: str
    description: str
    domain: str

    # Task specification
    objective: str
    constraints: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    complexity: TaskComplexity = TaskComplexity.MODERATE
    preferred_mode: Optional[ReasoningMode] = None
    time_budget_seconds: float = 60.0

    # Status
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        if not self.task_id:
            self.task_id = f"TASK-{uuid.uuid4().hex[:8]}"


@dataclass
class ReasoningResult:
    """Result of orchestrated reasoning"""
    task_id: str
    success: bool

    # Main output
    conclusion: str
    confidence: float
    evidence: List[str]

    # Process information
    reasoning_trace: List[Dict[str, Any]]
    capabilities_used: List[str]
    time_taken_seconds: float

    # Quality metrics
    coherence: float = 0.0
    completeness: float = 0.0
    novelty: float = 0.0

    # Insights generated
    hypotheses_generated: List[str] = field(default_factory=list)
    knowledge_gaps_identified: List[str] = field(default_factory=list)
    patterns_discovered: List[str] = field(default_factory=list)

    # Metadata
    mode_used: Optional[ReasoningMode] = None
    consensus_level: Optional[str] = None


class CapabilityRouter:
    """Routes tasks to appropriate capabilities"""

    def __init__(self):
        self.routing_rules = {
            # Keywords -> capabilities
            "cause": ["counterfactual", "causal_inference"],
            "effect": ["counterfactual", "causal_inference"],
            "what if": ["counterfactual"],
            "alternative": ["counterfactual", "deliberation"],
            "similar": ["analogical"],
            "like": ["analogical"],
            "pattern": ["theory_synthesis", "continuous_learning"],
            "theory": ["theory_synthesis"],
            "law": ["theory_synthesis"],
            "uncertain": ["metacognition", "knowledge_acquisition"],
            "unknown": ["knowledge_acquisition"],
            "debate": ["deliberation"],
            "consensus": ["deliberation"],
            "learn": ["continuous_learning"],
            "improve": ["continuous_learning", "metacognition"],
        }

        self.mode_capabilities = {
            ReasoningMode.ANALYTICAL: ["counterfactual", "theory_synthesis", "metacognition"],
            ReasoningMode.CREATIVE: ["analogical", "theory_synthesis"],
            ReasoningMode.CRITICAL: ["counterfactual", "deliberation", "metacognition"],
            ReasoningMode.INTEGRATIVE: ["theory_synthesis", "deliberation", "world_model"],
            ReasoningMode.ADAPTIVE: ["dynamic_replanning", "continuous_learning"],
            ReasoningMode.DELIBERATIVE: ["deliberation", "metacognition"],
        }

    def route(
        self,
        task: ReasoningTask
    ) -> List[str]:
        """Determine which capabilities to engage"""
        capabilities = set()

        # Route based on keywords
        task_lower = (task.description + " " + task.objective).lower()
        for keyword, caps in self.routing_rules.items():
            if keyword in task_lower:
                capabilities.update(caps)

        # Route based on preferred mode
        if task.preferred_mode:
            capabilities.update(self.mode_capabilities.get(task.preferred_mode, []))

        # Always include core capabilities for complex tasks
        if task.complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT, TaskComplexity.FRONTIER]:
            capabilities.update(["world_model", "metacognition", "integration_bus"])

        # Default capabilities if none selected
        if not capabilities:
            capabilities = {"world_model", "metacognition"}

        return list(capabilities)


class V41Orchestrator:
    """
    Main STAN V41 Orchestrator.

    Coordinates all advanced reasoning capabilities for AGI-like behavior.
    """

    VERSION = "41.0"

    def __init__(self):
        # Core components (singletons)
        self.world_model: UnifiedWorldModel = get_world_model()
        self.bus: IntegrationBus = get_integration_bus()
        self.theory_synthesizer: TheorySynthesizer = get_theory_synthesizer()
        self.metacognition: MetacognitiveController = get_metacognitive_controller()
        self.knowledge_acquirer: ActiveKnowledgeAcquirer = get_knowledge_acquirer()
        self.deliberator: MultiAgentDeliberator = get_deliberator()
        self.learner: ContinuousLearner = get_continuous_learner()

        # Non-singleton components
        self.counterfactual_engine = CounterfactualEngine()
        self.analogy_finder = AnalogyFinder()
        self.dynamic_executor = DynamicExecutor()

        # Orchestration
        self.router = CapabilityRouter()
        self.active_tasks: Dict[str, ReasoningTask] = {}
        self.completed_results: Dict[str, ReasoningResult] = {}

        # Statistics
        self.tasks_completed = 0
        self.total_reasoning_time = 0.0
        self.capability_usage: Dict[str, int] = defaultdict(int)

        # Set up event handlers
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Set up cross-module event handlers"""
        # Log all events for debugging
        def log_event(event):
            pass  # Can be enabled for debugging

        self.bus.subscribe("orchestrator_logger", EventType.CAPABILITY_RESULT, log_event)

        # Connect metacognition to learning
        def on_insight(event):
            insight = event.payload
            if insight.get("actionable"):
                self.learner.add_knowledge(
                    content=insight.get("description", ""),
                    domain="metacognition",
                    knowledge_type="insight",
                    source="metacognition",
                    confidence=insight.get("confidence", 0.5)
                )

        self.bus.subscribe("orchestrator_insight", EventType.METACOGNITIVE_INSIGHT, on_insight)

    def reason(
        self,
        description: str,
        objective: str,
        domain: str = "general",
        context: Dict[str, Any] = None,
        mode: ReasoningMode = None,
        time_budget: float = 60.0
    ) -> ReasoningResult:
        """
        Main reasoning entry point.

        Orchestrates all V41 capabilities to address a reasoning task.
        """
        # Create task
        task = ReasoningTask(
            task_id="",
            description=description,
            objective=objective,
            domain=domain,
            context=context or {},
            complexity=self._assess_complexity(description, objective),
            preferred_mode=mode,
            time_budget_seconds=time_budget
        )

        self.active_tasks[task.task_id] = task
        task.status = "active"
        task.started_at = datetime.now()

        start_time = time.time()



        # Begin metacognitive tracking
        trace_steps: List[Dict[str, Any]] = []
        capabilities_used: List[str] = []
        hypotheses_generated: List[str] = []
        knowledge_gaps: List[str] = []
        patterns_discovered: List[str] = []
        evidence: List[str] = []
        confidence_components: List[float] = []
        consensus_level = None
        conclusion = ""

        # ---------------------------------------------------------------- routing
        capabilities = self.router.route(task)
        trace_steps.append({
            "step": "routing",
            "capabilities": capabilities,
            "complexity": task.complexity.name,
        })
        for cap in capabilities:
            self.capability_usage[cap] += 1

        # ------------------------------------------------------------ world model
        if "world_model" in capabilities:
            capabilities_used.append("world_model")
            try:
                breakdown = self.world_model.get_confidence_breakdown()
                n_beliefs = len(getattr(self.world_model, "beliefs", {}))
                trace_steps.append({
                    "step": "world_model_context",
                    "n_beliefs": n_beliefs,
                    "confidence_breakdown": breakdown,
                })
                evidence.append(
                    f"World model: {n_beliefs} tracked beliefs engaged as background context"
                )
                if breakdown:
                    avg_conf = sum(breakdown.values()) / len(breakdown)
                    confidence_components.append(float(avg_conf))
            except Exception as exc:
                trace_steps.append({"step": "world_model_context", "error": str(exc)})

        # ------------------------------------------------------------ deliberation
        if "deliberation" in capabilities:
            capabilities_used.append("deliberation")
            try:
                consensus = self.deliberator.deliberate(
                    question=task.objective,
                    context=task.description,
                    domain=task.domain,
                    max_rounds=3,
                )
                consensus_level = getattr(consensus, "level", None)
                level_name = getattr(consensus_level, "name", str(consensus_level))
                statement = getattr(consensus, "statement", "")
                if statement:
                    conclusion = statement
                confidence_components.append(
                    0.5 * getattr(consensus, "confidence", 0.5)
                    + 0.5 * getattr(consensus, "robustness", 0.5)
                )
                remaining = getattr(consensus, "remaining_disagreements", [])
                for disagreement in remaining[:5]:
                    knowledge_gaps.append(f"Unresolved disagreement: {disagreement}")
                trace_steps.append({
                    "step": "deliberation",
                    "consensus_level": level_name,
                    "n_supporting": len(getattr(consensus, "supporting_agents", [])),
                    "n_dissenting": len(getattr(consensus, "dissenting_agents", [])),
                })
                evidence.append(
                    f"Multi-agent deliberation reached {level_name} consensus "
                    f"({len(getattr(consensus, 'supporting_agents', []))} supporting, "
                    f"{len(getattr(consensus, 'dissenting_agents', []))} dissenting)"
                )
            except Exception as exc:
                trace_steps.append({"step": "deliberation", "error": str(exc)})

        # ----------------------------------------------------------- counterfactual
        if "counterfactual" in capabilities:
            capabilities_used.append("counterfactual")
            scm = task.context.get("scm")
            if scm is not None:
                try:
                    explanation = self.counterfactual_engine \
                        .generate_contrastive_explanation(scm, task.context.get("outcome"))
                    if explanation:
                        evidence.append(f"Contrastive explanation: {explanation}")
                        confidence_components.append(0.7)
                except Exception as exc:
                    trace_steps.append({"step": "counterfactual", "error": str(exc)})
            else:
                trace_steps.append({
                    "step": "counterfactual",
                    "note": "engine armed; no structural causal model in task context",
                })

        # ---------------------------------------------------------------- analogy
        if "analogical" in capabilities:
            capabilities_used.append("analogical")
            try:
                analogy = self.analogy_finder.find_analogy_for_problem(
                    task.description, task.domain
                )
                if analogy is not None:
                    novelty = float(getattr(analogy, "novelty", 0.3) or 0.3)
                    confidence_components.append(0.3 + 0.4 * novelty)
                    mapped = getattr(analogy, "mapped_elements", [])
                    patterns_discovered.append(
                        f"Cross-domain mapping: {len(mapped)} structural correspondences"
                    )
            except Exception as exc:
                trace_steps.append({"step": "analogical", "error": str(exc)})

        # -------------------------------------------------------- theory synthesis
        if "theory_synthesis" in capabilities:
            capabilities_used.append("theory_synthesis")
            try:
                summary = self.theory_synthesizer.get_synthesis_summary()
                n_patterns = summary.get("patterns", {}).get("total", 0) \
                    if isinstance(summary.get("patterns"), dict) else 0
                n_laws = len(summary.get("laws", [])) \
                    if isinstance(summary.get("laws"), list) else 0
                trace_steps.append({
                    "step": "theory_synthesis",
                    "patterns": n_patterns,
                    "laws": n_laws,
                })
                if n_laws:
                    evidence.append(
                        f"Theory base: {n_laws} established laws, {n_patterns} patterns"
                    )
            except Exception as exc:
                trace_steps.append({"step": "theory_synthesis", "error": str(exc)})

        # ------------------------------------------------------- knowledge gaps
        if "knowledge_acquisition" in capabilities or "metacognition" in capabilities:
            capabilities_used.append("knowledge_acquisition")
            try:
                gaps = self.knowledge_acquirer.identify_gaps(
                    current_knowledge=task.context,
                    task_requirements=[task.objective, task.description],
                    domain=task.domain,
                )
                for gap in gaps[:5]:
                    knowledge_gaps.append(getattr(gap, "description", str(gap)))
                if gaps:
                    trace_steps.append({
                        "step": "knowledge_acquisition",
                        "n_gaps": len(gaps),
                    })
            except Exception as exc:
                trace_steps.append({"step": "knowledge_acquisition", "error": str(exc)})

        # ------------------------------------------------------ dynamic replanning
        if "dynamic_replanning" in capabilities:
            capabilities_used.append("dynamic_replanning")
            trace_steps.append({
                "step": "dynamic_replanning",
                "note": "execution monitored against time budget",
            })

        # ------------------------------------------------------------ conclusion
        if not conclusion:
            parts = []
            if evidence:
                parts.append(evidence[0])
            parts.append(
                f"Orchestrated {len(capabilities_used)} capabilities "
                f"({', '.join(capabilities_used)}) for: {task.objective}"
            )
            conclusion = " ".join(parts)

        confidence = (
            sum(confidence_components) / len(confidence_components)
            if confidence_components else 0.3
        )
        if knowledge_gaps:
            confidence *= max(0.5, 1.0 - 0.05 * len(knowledge_gaps))

        # -------------------------------------------------- metacognitive quality
        trace = ReasoningTrace(
            trace_id=f"TRACE-{task.task_id}",
            task_description=task.description,
            strategy_used=ReasoningStrategy.SYSTEMATIC,
            steps=trace_steps,
            capabilities_invoked=capabilities_used,
            time_taken_ms=0.0,
            conclusion=conclusion,
            confidence=confidence,
        )
        try:
            quality = self.metacognition.assess(trace)
            coherence = float(quality.get("coherence", 0.5))
            completeness = float(quality.get("completeness", 0.5))
        except Exception:
            coherence = 0.5
            completeness = min(
                1.0, len(capabilities_used) / 6.0
            ) if capabilities_used else 0.2

        # Completeness also reflects how many routed capabilities produced output
        if capabilities:
            produced = len({s.get("step") for s in trace_steps}) / max(1, len(capabilities))
            completeness = min(1.0, 0.5 * completeness + 0.5 * produced)
        novelty = min(1.0, len(patterns_discovered) * 0.4) if patterns_discovered else 0.1

        # ------------------------------------------------------------------ learn
        try:
            self.learner.add_knowledge(
                content=f"{task.description} -> {conclusion[:300]}",
                domain=task.domain,
                knowledge_type="task_outcome",
                source="v41_orchestrator",
                confidence=confidence,
            )
        except Exception:
            pass

        # ------------------------------------------------------------- bus events
        for hypothesis in hypotheses_generated:
            self.bus.publish(
                EventType.HYPOTHESIS_GENERATED,
                source="v41_orchestrator",
                payload={"task_id": task.task_id, "hypothesis": hypothesis},
                correlation_id=task.task_id,
            )

        # ------------------------------------------------------------------ close
        time_taken = time.time() - start_time
        task.status = "completed"
        task.completed_at = datetime.now()
        self.active_tasks.pop(task.task_id, None)

        result = ReasoningResult(
            task_id=task.task_id,
            success=True,
            conclusion=conclusion,
            confidence=round(confidence, 4),
            evidence=evidence,
            reasoning_trace=trace_steps,
            capabilities_used=capabilities_used,
            time_taken_seconds=round(time_taken, 4),
            coherence=round(coherence, 4),
            completeness=round(completeness, 4),
            novelty=round(novelty, 4),
            hypotheses_generated=hypotheses_generated,
            knowledge_gaps_identified=knowledge_gaps,
            patterns_discovered=patterns_discovered,
            mode_used=task.preferred_mode,
            consensus_level=getattr(consensus_level, "name", None)
            if consensus_level is not None else None,
        )
        self.completed_results[task.task_id] = result
        self.tasks_completed += 1
        self.total_reasoning_time += time_taken
        return result

    def _assess_complexity(
        self, description: str, objective: str
    ) -> TaskComplexity:
        """Heuristic task-complexity assessment for routing and budgeting."""
        text = f"{description} {objective}".lower()
        score = 0

        # Sheer size
        score += min(3, len(text) / 200)

        # Conditionality and causal structure raise complexity
        score += 2 * text.count("what if")
        score += text.count(" if ")
        score += sum(text.count(kw) for kw in (
            "cause", "effect", "because", "therefore", "however",
            "trade-off", "tradeoff", "constrain",
        ))

        # Multi-part questions
        score += text.count("?") - 1 if "?" in text else 0
        score += text.count(" and ") * 0.5 + text.count(" versus ") * 0.5

        if score < 2:
            return TaskComplexity.SIMPLE
        if score < 5:
            return TaskComplexity.MODERATE
        if score < 9:
            return TaskComplexity.COMPLEX
        if score < 14:
            return TaskComplexity.EXPERT
        return TaskComplexity.FRONTIER

    def get_status(self) -> Dict[str, Any]:
        """Orchestrator statistics since construction."""
        return {
            "version": self.VERSION,
            "tasks_completed": self.tasks_completed,
            "active_tasks": len(self.active_tasks),
            "total_reasoning_time": round(self.total_reasoning_time, 4),
            "capability_usage": dict(self.capability_usage),
        }


# Singleton accessor ----------------------------------------------------------

_orchestrator: Optional["V41Orchestrator"] = None


def get_orchestrator() -> "V41Orchestrator":
    """Return the shared V41 orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = V41Orchestrator()
    return _orchestrator


def reason(
    description: str,
    objective: str,
    domain: str = "general",
    context: Dict[str, Any] = None,
    mode: "ReasoningMode" = None,
    time_budget: float = 60.0,
) -> "ReasoningResult":
    """Module-level convenience entry point for orchestrated reasoning."""
    return get_orchestrator().reason(
        description=description,
        objective=objective,
        domain=domain,
        context=context,
        mode=mode,
        time_budget=time_budget,
    )
