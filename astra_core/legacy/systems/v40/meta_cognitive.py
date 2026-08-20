"""
Meta-Cognitive Controller for STAN V40

Implements:
- Uncertainty estimation
- Strategy selection
- Resource allocation
- Self-monitoring and adaptation

Target: +10-15% through optimal strategy routing

Date: 2025-12-11
Version: 40.0
"""

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum


class ReasoningStrategy(Enum):
    """Available reasoning strategies"""
    DIRECT = "direct"                   # Simple answer retrieval
    DECOMPOSITION = "decomposition"     # Multi-step decomposition
    HYPOTHESIS = "hypothesis"           # Hypothesis generation & testing
    FORMAL_LOGIC = "formal_logic"       # Z3/Prolog reasoning
    THEOREM_PROVING = "theorem_proving" # Neural-symbolic proof
    CAUSAL = "causal"                   # Causal world model
    RETRIEVAL = "retrieval"             # Knowledge retrieval
    SELF_CONSISTENCY = "self_consistency"  # Multiple samples + voting
    ENSEMBLE = "ensemble"               # Combine multiple strategies


class ConfidenceLevel(Enum):
    """Confidence levels"""
    VERY_LOW = "very_low"       # < 0.2
    LOW = "low"                 # 0.2 - 0.4
    MEDIUM = "medium"           # 0.4 - 0.6
    HIGH = "high"               # 0.6 - 0.8
    VERY_HIGH = "very_high"     # > 0.8


@dataclass
class ResourceBudget:
    """Resource budget for problem solving"""
    max_time_seconds: float = 30.0
    max_llm_calls: int = 5
    max_tool_calls: int = 10
    max_iterations: int = 3

    # Current usage
    time_used: float = 0.0
    llm_calls_used: int = 0
    tool_calls_used: int = 0
    iterations_used: int = 0

    def remaining_time(self) -> float:
        return max(0, self.max_time_seconds - self.time_used)

    def remaining_llm_calls(self) -> int:
        return max(0, self.max_llm_calls - self.llm_calls_used)

    def is_exhausted(self) -> bool:
        return (self.time_used >= self.max_time_seconds or
                self.llm_calls_used >= self.max_llm_calls)

    def to_dict(self) -> Dict:
        return {
            'time_remaining': self.remaining_time(),
            'llm_calls_remaining': self.remaining_llm_calls(),
            'tool_calls_remaining': self.max_tool_calls - self.tool_calls_used
        }


@dataclass
class StrategyResult:
    """Result from applying a strategy"""
    strategy: ReasoningStrategy
    answer: Any
    confidence: float
    reasoning_trace: List[str] = field(default_factory=list)
    resources_used: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'strategy': self.strategy.value,
            'answer': str(self.answer),
            'confidence': self.confidence,
            'trace_length': len(self.reasoning_trace)
        }


@dataclass
class ProblemCharacteristics:
    """Characteristics of a problem for strategy selection"""
    # Content type
    is_mathematical: bool = False
    is_logical: bool = False
    is_factual: bool = False
    is_causal: bool = False
    is_comparative: bool = False

    # Structure
    complexity: float = 0.5  # 0-1 scale
    has_multiple_parts: bool = False
    requires_precision: bool = False

    # Domain
    domain: str = "general"
    subdomain: str = ""

    # Format
    answer_type: str = "text"  # text, number, choice, yes_no
