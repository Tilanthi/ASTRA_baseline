"""
STAR-Learn Integrated System
============================

Top-level facade assembling the STAR-Learn self-teaching architecture:

    SelfRewardingEngine      - intrinsic reward for attempts
    CurriculumGenerator      - autonomous problem generation
    RecursiveImprover        - metacognitive self-improvement
    StigmergicMemory         - shared biological-field memory
    AutonomousTrainingLoop   - the iterate/score/improve cycle

(Re-implemented 2026-08; the original facade was lost to file truncation
before the August 2026 audit. This version composes the surviving component
implementations directly.)
"""

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .self_rewarding import SelfRewardingEngine
from .curriculum_generator import CurriculumGenerator, CurriculumConfig, GeneratedProblem
from .recursive_improver import RecursiveImprover
from .stigmergic_memory import StigmergicMemory
from .autonomous_loop import AutonomousTrainingLoop, LoopConfig, IterationResult


class STARLearnSystem:
    """Integrated STAR-Learn self-teaching system."""

    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 reward_engine: Optional[SelfRewardingEngine] = None,
                 curriculum_generator: Optional[CurriculumGenerator] = None,
                 recursive_improver: Optional[RecursiveImprover] = None,
                 stigmergic_memory: Optional[StigmergicMemory] = None):
        config = dict(config or {})
        self.reward_engine = reward_engine or SelfRewardingEngine()
        self.memory = stigmergic_memory or StigmergicMemory()
        self.curriculum = curriculum_generator or CurriculumGenerator(
            config=CurriculumConfig(), memory=self.memory)
        self.improver = recursive_improver or RecursiveImprover(
            memory=self.memory, reward_engine=self.reward_engine)
        self.loop = AutonomousTrainingLoop(
            config=LoopConfig(**{k: v for k, v in config.items()
                                 if k in LoopConfig.__dataclass_fields__}),
            reward_engine=self.reward_engine,
            curriculum_generator=self.curriculum,
            recursive_improver=self.improver,
            stigmergic_memory=self.memory)

        # Bookkeeping over and above the loop's own history
        self.iteration_count = 0
        self.reward_history: List[float] = []
        self.discovery_archive: List[Dict[str, Any]] = []
        self.domain_attempts: Dict[str, int] = {}

    # ---------------------------------------------------------------- training
    def train_autonomously(self,
                           n_iterations: int = 5,
                           generate_report: bool = False,
                           verbose: bool = False) -> List[IterationResult]:
        """
        Run the self-teaching cycle ``n_iterations`` times.

        Each iteration: curriculum generates a problem, the loop attempts and
        scores it, notable attempts are archived as discoveries.
        """
        results: List[IterationResult] = []
        for _ in range(n_iterations):
            problem = self.curriculum.generate_problem()
            problem_dict = asdict(problem) if isinstance(problem, GeneratedProblem) \
                else dict(problem)
            self.domain_attempts[problem_dict.get('domain', 'unknown')] = \
                self.domain_attempts.get(problem_dict.get('domain', 'unknown'), 0) + 1

            result = self.loop.run_iteration(problem_dict)
            results.append(result)
            self.iteration_count += 1
            self.reward_history.append(result.total_reward)

            if result.total_reward >= 0.4 or result.iteration.discovery_made:
                self.discovery_archive.append({
                    'iteration': self.iteration_count,
                    'problem': problem_dict.get('question', ''),
                    'domain': problem_dict.get('domain', ''),
                    'reward': result.total_reward,
                    'novelty': result.iteration.novelty_score,
                })
            if verbose:
                print(f"[STAR-Learn] iter {self.iteration_count}: "
                      f"reward={result.total_reward:.3f}")

        if generate_report:
            self._write_report()
        return results

    # ------------------------------------------------------------------ status
    def get_status(self) -> Dict[str, Any]:
        rewards = self.reward_history
        return {
            'iteration_count': self.iteration_count,
            'best_reward': max(rewards) if rewards else 0.0,
            'average_reward': sum(rewards) / len(rewards) if rewards else 0.0,
            'total_discoveries': len(self.discovery_archive),
            'domains_attempted': dict(self.domain_attempts),
            'components': {
                'reward_engine': type(self.reward_engine).__name__,
                'curriculum': type(self.curriculum).__name__,
                'improver': type(self.improver).__name__,
                'memory': type(self.memory).__name__,
            },
            'loop': self.loop.get_status(),
        }

    def get_enhanced_capabilities(self) -> Dict[str, str]:
        """
        Report availability of the V2.0 enhanced-feature modules.

        (Method restored 2026-08; exercised by tests/test_enhanced_features.py.)
        """
        import importlib
        components = {
            'embedding_novelty': 'astra_core.self_teaching.embedding_novelty',
            'scientific_data': 'astra_core.self_teaching.scientific_data',
            'multi_agent_swarm': 'astra_core.self_teaching.multi_agent_swarm',
            'arxiv_integration': 'astra_core.self_teaching.arxiv_integration',
        }
        capabilities: Dict[str, str] = {}
        for name, module in components.items():
            try:
                importlib.import_module(module)
                capabilities[name] = 'available'
            except Exception:
                capabilities[name] = 'unavailable'
        return capabilities

    # ------------------------------------------------------------ capabilities
    def assess_self_teaching_capability(self) -> Dict[str, Any]:
        """
        Measure self-teaching capability from the ACTUAL recorded history.

        learning_rate      - normalised reward trend over iterations
        transfer_efficiency - coverage of curriculum domains achieved
        autonomy_level     - fraction of self-contained components active
        """
        # Learning rate: slope of rewards normalised by their mean
        if len(self.reward_history) >= 2:
            mean_r = max(sum(self.reward_history) / len(self.reward_history), 1e-9)
            n = len(self.reward_history)
            slope = (self.reward_history[-1] - self.reward_history[0]) / (n - 1)
            learning_rate = min(1.0, max(0.0, slope / mean_r + 0.5))
        else:
            learning_rate = 0.0

        # Transfer efficiency: distinct domains exercised vs configured domains
        configured = set(self.curriculum.config.domain_weights)
        covered = set(self.domain_attempts) & configured
        transfer_efficiency = len(covered) / max(1, len(configured))

        # Autonomy: all four components self-contained
        autonomy_level = sum([
            self.reward_engine is not None,
            self.curriculum is not None,
            self.improver is not None,
            self.memory is not None,
        ]) / 4.0

        self_teaching_score = (0.4 * learning_rate
                               + 0.3 * transfer_efficiency
                               + 0.3 * autonomy_level)
        return {
            'self_teaching_score': round(self_teaching_score, 4),
            'learning_rate': round(learning_rate, 4),
            'transfer_efficiency': round(transfer_efficiency, 4),
            'autonomy_level': round(autonomy_level, 4),
            'n_iterations': self.iteration_count,
        }

    # ----------------------------------------------------------------- reports
    def _write_report(self) -> None:
        status = self.get_status()
        print("=" * 60)
        print("STAR-Learn Training Report")
        print("=" * 60)
        for key in ('iteration_count', 'best_reward', 'average_reward',
                    'total_discoveries'):
            print(f"  {key}: {status[key]}")
        print(f"  capability: {self.assess_self_teaching_capability()}")


def create_star_learn_system(config: Optional[Dict[str, Any]] = None) -> STARLearnSystem:
    """Factory for the integrated STAR-Learn system."""
    return STARLearnSystem(config=config)
