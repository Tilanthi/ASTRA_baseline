"""
Curriculum Generator for STAR-Learn

Autonomously generates training problems at appropriate difficulty levels
to drive self-teaching. The curriculum adapts based on:

1. Current performance (success rate, reward trends)
2. Knowledge gaps (identified via MORK ontology analysis)
3. Cross-domain synthesis opportunities
4. Progressive difficulty scaling
5. Novelty generation (using LLM for creative problems)

This ensures the system always has appropriate challenges to learn from.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import random
from datetime import datetime


class ProblemDifficulty(Enum):
    """Difficulty levels for generated problems"""
    TRIVIAL = "trivial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"
    RESEARCH = "research"  # Open-ended scientific problems


class DomainTask(Enum):
    """Types of tasks by scientific domain"""
    CAUSAL_INFERENCE = "causal_inference"
    ASTRONOMY = "astronomy"
    PHYSICS = "physics"
    MATHEMATICS = "mathematics"
    BIOLOGY = "biology"
    CHEMISTRY = "chemistry"
    CROSS_DOMAIN = "cross_domain"  # Requires multiple domains
    META_LEARNING = "meta_learning"  # Learning how to learn
    EXPERIMENTAL_DESIGN = "experimental_design"
    THEORY_CONSTRUCTION = "theory_construction"


@dataclass
class GeneratedProblem:
    """A generated training problem"""
    question: str
    domain: str
    difficulty: float  # 0-1 scale
    task_type: str

    # Additional context
    context: str = ""
    hints: List[str] = field(default_factory=list)
    expected_skills: List[str] = field(default_factory=list)

    # Metadata
    problem_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "curriculum_generator"

    # Evaluation criteria
    evaluation_criteria: Dict[str, Any] = field(default_factory=dict)
    success_threshold: float = 0.5


@dataclass
class CurriculumConfig:
    """Configuration for the curriculum generator"""
    # Domain preferences
    domain_weights: Dict[str, float] = field(default_factory=lambda: {
        'astrophysics': 0.25,
        'causal_inference': 0.20,
        'physics': 0.15,
        'mathematics': 0.15,
        'cross_domain': 0.15,
        'experimental_design': 0.10
    })

    # Difficulty progression
    initial_difficulty: float = 0.3
    difficulty_step: float = 0.05
    adaptive_difficulty: bool = True

    # Novelty parameters
    novelty_rate: float = 0.3  # Fraction of novel problems
    novelty_generation_method: str = "llm"  # "llm" or "template"

    # Cross-domain synthesis
    cross_domain_rate: float = 0.2
    min_domains_for_synthesis: int = 2

    # Knowledge gap targeting
    target_knowledge_gaps: bool = True
    gap_analysis_frequency: int = 10  # Analyze gaps every N problems

    # Problem variety
    repetition_penalty: float = 0.1  # Avoid similar recent problems
    variety_bonus: float = 0.2  # Reward problem diversity

    # Template-based generation
    use_templates: bool = True
    n_templates_per_domain: int = 50

    # LLM-based generation
    llm_creativity: float = 0.7  # Temperature for LLM
    llm_max_tokens: int = 500


class CurriculumGenerator:
    """
    Curriculum Generator for autonomous problem generation.

    Creates training problems that:
    1. Match current skill level (adaptive difficulty)
    2. Target knowledge gaps
    3. Encourage cross-domain thinking
    4. Include novel, creative problems
    5. Cover all scientific domains
    """

    def __init__(
        self,
        config: Optional[CurriculumConfig] = None,
        memory=None
    ):
        """
        Initialize the curriculum generator.

        Args:
            config: Curriculum configuration
            memory: Stigmergic memory for gap analysis
        """
        self.config = config or CurriculumConfig()
        self.memory = memory

        # Current curriculum state
        self.current_difficulty = self.config.initial_difficulty
        self.recent_problems: List[GeneratedProblem] = []
        self.domain_counters: Dict[str, int] = {}

        # Knowledge gap tracking
        self.knowledge_gaps: Dict[str, float] = {}
        self.last_gap_analysis = 0

        # Problem templates
        self.templates = self._initialize_templates()

        # Statistics
        self.problems_generated = 0
        self.domains_used = set()
        self.performance_history: List[float] = []

    def generate_problem(
        self,
        difficulty: Optional[float] = None,
        domain: Optional[str] = None,
        exploration_rate: float = 0.2
    ) -> GeneratedProblem:
        """
        Generate a training problem.

        Args:
            difficulty: Target difficulty (0-1). If None, uses adaptive difficulty.
            domain: Target domain. If None, selects based on weights and gaps.
            exploration_rate: Rate of exploration vs exploitation

        Returns:
            GeneratedProblem with all problem details
        """
        # Determine domain
        if domain is None:
            domain = self._select_domain(exploration_rate)

        # Determine difficulty
        if difficulty is None:
            difficulty = self._select_difficulty(domain)

        # Generate problem
        problem = self._generate(domain, difficulty)

        # Update tracking
        self._update_tracking(problem)

        return problem

    def _select_domain(self, exploration_rate: float) -> str:
        """Select a domain based on weights, gaps, and exploration."""
        # Periodic gap analysis
        if self.problems_generated % self.config.gap_analysis_frequency == 0:
            self._analyze_knowledge_gaps()

        # Exploration vs exploitation
        if random.random() < exploration_rate:
            # Explore: select underrepresented domain
            all_domains = list(self.config.domain_weights.keys())
            underrepresented = [
                d for d in all_domains
                if self.domain_counters.get(d, 0) < self.problems_generated / len(all_domains)
            ]
            if underrepresented:
                return random.choice(underrepresented)

        # Exploit: select based on weighted gaps
        weights = {}
        for domain, base_weight in self.config.domain_weights.items():
            # Adjust for knowledge gaps
            gap_bonus = self.knowledge_gaps.get(domain, 0)
            weights[domain] = base_weight + gap_bonus

        # Normalize weights
        total = sum(weights.values())
        normalized = {d: w/total for d, w in weights.items()}

        # Sample domain
        domains = list(normalized.keys())
        probs = [normalized[d] for d in domains]
        return np.random.choice(domains, p=probs)

    def _select_difficulty(self, domain: str) -> float:
        """Select appropriate difficulty for the domain."""
        if not self.config.adaptive_difficulty:
            return self.config.initial_difficulty

        # Check domain performance
        recent_domain_problems = [
            p for p in self.recent_problems[-50:]
            if p.domain == domain
        ]

        if len(recent_domain_problems) >= 5:
            # Adjust based on recent performance
            avg_difficulty = np.mean([p.difficulty for p in recent_domain_problems])

            # If recent problems were easy, increase difficulty
            # If recent problems were hard, decrease difficulty
            # For now, use current adaptive difficulty
            return self.current_difficulty

        return self.current_difficulty

    def _generate(self, domain: str, difficulty: float) -> GeneratedProblem:
        """Generate a problem for the given domain and difficulty."""
        # Decide generation method
        use_novel = random.random() < self.config.novelty_rate
        use_cross_domain = (
            random.random() < self.config.cross_domain_rate and
            len(self.domains_used) >= self.config.min_domains_for_synthesis
        )

        if use_cross_domain:
            return self._generate_cross_domain_problem(domain, difficulty)
        elif use_novel and self.config.novelty_generation_method == "llm":
            return self._generate_novel_problem(domain, difficulty)
        else:
            return self._generate_template_problem(domain, difficulty)

    def _generate_template_problem(
        self,
        domain: str,
        difficulty: float
    ) -> GeneratedProblem:
        """Generate a problem from templates."""
        templates = self.templates.get(domain, self._get_default_templates(domain))
        template = random.choice(templates)

        # Adjust template based on difficulty
        question = self._adjust_template_difficulty(template, difficulty)

        # Determine task type
        task_type = self._domain_to_task_type(domain)

        # Generate hints based on difficulty
        hints = self._generate_hints(domain, difficulty)

        # Determine expected skills
        expected_skills = self._get_expected_skills(domain, difficulty)

        return GeneratedProblem(
            question=question,
            domain=domain,
            difficulty=difficulty,
            task_type=task_type,
            hints=hints,
            expected_skills=expected_skills,
            problem_id=self._generate_problem_id(),
            source="template"
        )

    def _generate_cross_domain_problem(
        self,
        primary_domain: str,
        difficulty: float
    ) -> GeneratedProblem:
        """Generate a problem requiring multiple domains."""
        # Select secondary domains
        available_domains = [d for d in self.domains_used if d != primary_domain]

        if len(available_domains) < 1:
            # Fall back to single domain
            return self._generate_template_problem(primary_domain, difficulty)

        secondary_domain = random.choice(available_domains)

        # Generate cross-domain problem
        templates = self.templates.get('cross_domain', self._get_cross_domain_templates())

        # Find relevant template
        relevant_templates = [
            t for t in templates
            if primary_domain in t.lower() or secondary_domain in t.lower()
        ]

        if not relevant_templates:
            relevant_templates = templates

        template = random.choice(relevant_templates)

        # Customize template
        question = template.format(
            domain1=primary_domain,
            domain2=secondary_domain
        )

        # Adjust for difficulty
        question = self._adjust_template_difficulty(question, difficulty)

        return GeneratedProblem(
            question=question,
            domain=f"{primary_domain}+{secondary_domain}",
            difficulty=difficulty + 0.1,  # Cross-domain is harder
            task_type=DomainTask.CROSS_DOMAIN.value,
            hints=[
                f"Consider principles from both {primary_domain} and {secondary_domain}",
                "Look for analogous concepts between domains"
            ],
            expected_skills=[
                f"{primary_domain}_reasoning",
                f"{secondary_domain}_reasoning",
                "cross_domain_synthesis",
                "analogical_reasoning"
            ],
            problem_id=self._generate_problem_id(),
            source="cross_domain"
        )

    def _generate_novel_problem(
        self,
        domain: str,
        difficulty: float,
        focus_area: Optional[str] = None
    ) -> GeneratedProblem:
        """
        Generate a novel problem using creative templates.

        Args:
            domain: Target domain
            difficulty: Problem difficulty (0-1)
            focus_area: Optional specific area to focus on

        Returns:
            GeneratedProblem with novel content
        """
        # Get novel problem templates
        templates = self.templates.get('novel', self._get_novel_templates())

        # Find relevant template
        relevant_templates = [
            t for t in templates
            if focus_area is None or focus_area.lower() in t.lower()
        ]

        if not relevant_templates:
            relevant_templates = templates

        template = random.choice(relevant_templates)

        # Customize template
        question = template.format(domain=domain, focus=focus_area or "this area")

        # Adjust for difficulty
        question = self._adjust_template_difficulty(question, difficulty)

        return GeneratedProblem(
            question=question,
            domain=domain,
            difficulty=difficulty,
            task_type=DomainTask.THEORY_CONSTRUCTION.value,
            hints=[
                f"Consider novel approaches in {domain}",
                "Think beyond standard solutions"
            ],
            expected_skills=[
                f"{domain}_creativity",
                "novel_thinking",
                "theory_construction"
            ],
            problem_id=self._generate_problem_id(),
            source="novel_generator"
        )

    def _generate_problem_id(self) -> str:
        """Generate unique problem ID."""
        return f"prob_{int(datetime.now().timestamp())}_{random.randint(1000, 9999)}"

    def _get_cross_domain_templates(self) -> List[str]:
        """Get cross-domain problem templates."""
        return [
            "How might principles from {domain1} apply to {domain2}?",
            "Compare and contrast {domain1} and {domain2} approaches to similar phenomena",
            "Design an experiment combining {domain1} and {domain2} methods"
        ]

    def _get_novel_templates(self) -> List[str]:
        """Get novel problem templates."""
        return [
            "Propose a new theoretical framework for {domain}",
            "Identify gaps in current {domain} understanding",
            "How would you extend {domain} to explain {focus}?"
        ]

    # ------------------------------------------------------------------
    # Templates and problem enrichment
    # (re-implemented 2026-08; lost to file truncation before the audit)
    # ------------------------------------------------------------------

    def _initialize_templates(self) -> Dict[str, List[str]]:
        """Base problem templates per domain (placeholders substituted at use)."""
        return {
            'astrophysics': [
                "Estimate the Jeans mass of a molecular cloud with given size and temperature",
                "Explain why Lyman-alpha photons scatter many times before escaping",
                "Derive the virial mass of a galaxy cluster from its X-ray temperature",
                "How does metallicity affect the Cepheid period-luminosity relation?",
                "What sets the minimum mass for hydrogen burning in stars?",
            ],
            'causal_inference': [
                "Given observational data, how would you test whether X causes Y?",
                "Explain the difference between a confounder and a collider",
                "Design an instrumental-variable analysis for a given causal graph",
                "When does regression adjustment fail to remove confounding?",
                "How would you estimate a causal effect with unmeasured confounding?",
            ],
            'physics': [
                "Derive the equation of motion for a damped driven oscillator",
                "Explain the physical origin of the greenhouse effect quantitatively",
                "Compute the entropy change for free expansion of an ideal gas",
                "What determines whether a flow becomes turbulent?",
                "Derive the dispersion relation for deep-water waves",
            ],
            'mathematics': [
                "Prove that the union of countably many countable sets is countable",
                "Solve a first-order linear ODE with an integrating factor",
                "Explain the spectral theorem for symmetric matrices",
                "When does a power series converge uniformly?",
                "Derive the Fourier transform of a Gaussian",
            ],
            'experimental_design': [
                "Design a controlled experiment to test a given hypothesis",
                "How would you choose a sample size for a target statistical power?",
                "Identify confounders in a proposed observational study",
                "Propose a blinding scheme for a measurement prone to bias",
                "How would you validate a calibration against a known standard?",
            ],
            'cross_domain': self._get_cross_domain_templates(),
            'novel': self._get_novel_templates(),
        }

    def _get_default_templates(self, domain: str) -> List[str]:
        """Generic fallback templates for domains without dedicated ones."""
        return [
            f"Formulate and solve a key open problem in {domain}",
            f"Identify the assumptions underlying standard {domain} reasoning",
            f"Design a decisive test of a central {domain} claim",
        ]

    def _adjust_template_difficulty(self, template: str, difficulty: float) -> str:
        """Append difficulty-appropriate instructions to a problem template."""
        if difficulty < 0.35:
            level = ("State the key concepts involved and give one concrete "
                     "illustrative example")
        elif difficulty < 0.7:
            level = ("Give a quantitative treatment, including one worked "
                     "numerical estimate")
        else:
            level = ("Derive the relevant relations, quantify the dominant "
                     "uncertainties, and state one falsifiable prediction")
        return f"{template}. ({level}.)"

    def _domain_to_task_type(self, domain: str) -> str:
        """Map a domain name to a DomainTask value."""
        mapping = {
            'astrophysics': DomainTask.ASTRONOMY,
            'astronomy': DomainTask.ASTRONOMY,
            'causal_inference': DomainTask.CAUSAL_INFERENCE,
            'physics': DomainTask.PHYSICS,
            'mathematics': DomainTask.MATHEMATICS,
            'biology': DomainTask.BIOLOGY,
            'chemistry': DomainTask.CHEMISTRY,
            'experimental_design': DomainTask.EXPERIMENTAL_DESIGN,
            'meta_learning': DomainTask.META_LEARNING,
        }
        return mapping.get(domain, DomainTask.PHYSICS).value

    def _generate_hints(self, domain: str, difficulty: float) -> List[str]:
        """Difficulty-scaled hints for a generated problem."""
        hints = [f"Which core {domain} principles are relevant?"]
        if difficulty >= 0.35:
            hints.append("What quantities can you estimate from the given information?")
        if difficulty >= 0.7:
            hints.append("What would a wrong answer get wrong - and how could you check it?")
        return hints

    def _get_expected_skills(self, domain: str, difficulty: float) -> List[str]:
        """Skills a problem is expected to exercise."""
        skills = [f"{domain}_knowledge", "quantitative_reasoning"]
        if difficulty >= 0.7:
            skills.extend(["derivation", "uncertainty_analysis"])
        if domain == 'causal_inference':
            skills.append("causal_graph_analysis")
        return skills

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    def _update_tracking(self, problem: GeneratedProblem) -> None:
        """Update counters and histories after generating a problem."""
        self.problems_generated += 1
        self.domain_counters[problem.domain] = \
            self.domain_counters.get(problem.domain, 0) + 1
        self.domains_used.add(problem.domain.split('+')[0])
        self.recent_problems.append(problem)
        if len(self.recent_problems) > 200:
            del self.recent_problems[:len(self.recent_problems) - 200]

    def _analyze_knowledge_gaps(self) -> None:
        """
        Estimate per-domain knowledge gaps from the stigmergic memory.

        Domains with few high-strength trails relative to the number of
        problems posed are flagged as gaps (gap in [0, 1]).
        """
        if self.memory is None:
            return
        counts: Dict[str, int] = {}
        try:
            for trail in getattr(self.memory, 'trails', []):
                domain = getattr(trail, 'domain', '') or 'unknown'
                counts[domain] = counts.get(domain, 0) + 1
        except Exception:
            return
        max_count = max(counts.values(), default=1)
        for domain in self.config.domain_weights:
            # few trails -> large gap
            self.knowledge_gaps[domain] = 1.0 - counts.get(domain, 0) / max_count
        self.last_gap_analysis = self.problems_generated

    def get_curriculum_status(self) -> Dict[str, Any]:
        """Get current curriculum status."""
        return {
            "domains_covered": list(self.config.domain_weights.keys()),
            "current_difficulty": self.current_difficulty,
            "problems_generated": self.problems_generated,
            "domains_used": sorted(self.domains_used),
            "domain_counters": dict(self.domain_counters),
            "performance_history": self.performance_history[-10:]  # Last 10
        }


# Factory functions
def create_curriculum_generator(config: Optional[CurriculumConfig] = None) -> CurriculumGenerator:
    """Create a curriculum generator."""
    return CurriculumGenerator(config or CurriculumConfig())
