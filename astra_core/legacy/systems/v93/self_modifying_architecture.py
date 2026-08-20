"""
Self-Modifying Architecture for V93
====================================

Provides architecture that can modify its own structure for the
V93 Recursive Self-Modifying Metacognitive Architecture.

This is a simplified version for compatibility purposes.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum


class ModificationType(Enum):
    """Types of architectural modifications"""
    ADD_CONNECTION = "add_connection"
    REMOVE_CONNECTION = "remove_connection"
    ADJUST_WEIGHTS = "adjust_weights"
    ADD_MODULE = "add_module"
    REMOVE_MODULE = "remove_module"
    RECONFIGURE = "reconfigure"


@dataclass
class ArchitectureModification:
    """Represents a modification to the architecture"""
    modification_type: ModificationType
    target: str
    parameters: Dict[str, Any]
    expected_impact: float
    confidence: float


class SelfModifyingArchitecture:
    """Architecture that can modify its own structure"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.architecture_state = self._initialize_architecture()
        self.modification_history = []
        self.performance_metrics = []
        self.metacognition_depth = self.config.get('metacognition_depth', 5)
        self.evolution_autonomy = self.config.get('evolution_autonomy', 0.6)


    # ------------------------------------------------------------ architecture
    def _initialize_architecture(self) -> Dict[str, Any]:
        """
        Build the initial cognitive architecture: a weighted graph of
        modules (deterministic - no randomness in the baseline).
        """
        modules = {
            "perception":   {"weight": 1.0, "health": 1.0, "activations": 0},
            "memory":       {"weight": 1.0, "health": 1.0, "activations": 0},
            "reasoning":    {"weight": 1.0, "health": 1.0, "activations": 0},
            "metacognition": {"weight": 1.0, "health": 1.0, "activations": 0},
            "action":       {"weight": 1.0, "health": 1.0, "activations": 0},
        }
        connections = {
            ("perception", "memory"): 0.8,
            ("perception", "reasoning"): 0.6,
            ("memory", "reasoning"): 0.9,
            ("reasoning", "metacognition"): 0.7,
            ("reasoning", "action"): 0.8,
            ("metacognition", "reasoning"): 0.5,   # feedback path
        }
        return {
            "modules": modules,
            "connections": connections,
            "version": 1,
        }

    def get_state(self) -> Dict[str, Any]:
        """Snapshot of the current architecture."""
        return {
            "version": self.architecture_state["version"],
            "modules": {name: dict(spec)
                        for name, spec in self.architecture_state["modules"].items()},
            "connections": dict(self.architecture_state["connections"]),
            "n_modifications": len(self.modification_history),
        }

    # ------------------------------------------------------------ performance
    def record_performance(self, metric: float) -> None:
        """Record a measured system performance metric in [0, 1]."""
        self.performance_metrics.append(float(metric))

    def _recent_trend(self, window: int = 10) -> float:
        """Mean performance delta over the recent window (None-safe)."""
        recent = self.performance_metrics[-window:]
        if len(recent) < 2:
            return 0.0
        first, last = sum(recent[: len(recent) // 2]), sum(recent[len(recent) // 2:])
        halves = len(recent) // 2
        if halves == 0 or (first + last) == 0:
            return 0.0
        return (last - first) / (first + last)

    # ------------------------------------------------------------- modification
    def propose_modification(self) -> Optional[ArchitectureModification]:
        """
        Propose the next self-modification from MEASURED performance data.

        Strategy (data-driven, no fabricated impacts):
        - If a module's health is degraded, propose strengthening its
          inputs (ADJUST_WEIGHTS) with expected impact tied to the deficit.
        - If performance is rising and a useful pathway is missing, propose
          ADD_CONNECTION between reasoning and the most-activated module.
        - If performance is flat and the graph is dense, propose
          REMOVE_CONNECTION pruning of the weakest link.
        - Otherwise no proposal (None) - no change without evidence.
        """
        modules = self.architecture_state["modules"]
        connections = self.architecture_state["connections"]

        # 1. Weakest module -> strengthen its inbound weights
        weakest = min(modules, key=lambda m: modules[m]["health"])
        deficit = 1.0 - modules[weakest]["health"]
        if deficit > 0.2:
            return ArchitectureModification(
                modification_type=ModificationType.ADJUST_WEIGHTS,
                target=weakest,
                parameters={"scale": 1.0 + deficit},
                expected_impact=round(deficit * 0.5, 4),
                confidence=round(min(0.9, 0.5 + deficit), 4),
            )

        # 2. Improving trend + missing metacognition feedback -> add it
        trend = self._recent_trend()
        feedback_exists = any(
            src == "metacognition" and dst == "reasoning"
            for (src, dst) in connections
        )
        if trend > 0.05 and not feedback_exists:
            return ArchitectureModification(
                modification_type=ModificationType.ADD_CONNECTION,
                target="metacognition->reasoning",
                parameters={"weight": 0.4},
                expected_impact=round(trend * 0.3, 4),
                confidence=0.6,
            )

        # 3. Flat performance + dense graph -> prune weakest link
        if abs(trend) <= 0.02 and len(connections) > 5:
            weakest_link = min(connections, key=connections.get)
            return ArchitectureModification(
                modification_type=ModificationType.REMOVE_CONNECTION,
                target=f"{weakest_link[0]}->{weakest_link[1]}",
                parameters={},
                expected_impact=round(0.1 * connections[weakest_link], 4),
                confidence=0.5,
            )

        return None

    def apply_modification(self, modification: ArchitectureModification) -> bool:
        """
        Apply a modification to the architecture.

        Respects evolution_autonomy: modifications below the autonomy
        threshold in confidence require it to be exceeded (returns False).
        Returns True if the architecture changed.
        """
        if modification.confidence < self.evolution_autonomy:
            return False

        modules = self.architecture_state["modules"]
        connections = self.architecture_state["connections"]
        mtype = modification.modification_type

        if mtype == ModificationType.ADJUST_WEIGHTS:
            if modification.target not in modules:
                return False
            scale = modification.parameters.get("scale", 1.1)
            for (src, dst) in list(connections):
                if dst == modification.target:
                    connections[(src, dst)] = min(1.0,
                                                  connections[(src, dst)] * scale)
            modules[modification.target]["health"] = min(
                1.0, modules[modification.target]["health"]
                * modification.parameters.get("scale", 1.1))

        elif mtype == ModificationType.ADD_CONNECTION:
            if "->" not in modification.target:
                return False
            src, dst = modification.target.split("->", 1)
            if src not in modules or dst not in modules:
                return False
            connections[(src, dst)] = modification.parameters.get("weight", 0.5)

        elif mtype == ModificationType.REMOVE_CONNECTION:
            if "->" not in modification.target:
                return False
            src, dst = modification.target.split("->", 1)
            connections.pop((src, dst), None)

        elif mtype == ModificationType.ADD_MODULE:
            name = modification.target
            modules[name] = {"weight": modification.parameters.get("weight", 1.0),
                             "health": 1.0, "activations": 0}

        elif mtype == ModificationType.REMOVE_MODULE:
            if modification.target not in modules or len(modules) <= 2:
                return False
            modules.pop(modification.target)
            for edge in list(connections):
                if modification.target in edge:
                    connections.pop(edge)

        elif mtype == ModificationType.RECONFIGURE:
            for name, spec in modules.items():
                spec["weight"] *= modification.parameters.get("scale", 1.0)

        self.architecture_state["version"] += 1
        self.modification_history.append({
            "version": self.architecture_state["version"],
            "type": mtype.value,
            "target": modification.target,
            "expected_impact": modification.expected_impact,
            "confidence": modification.confidence,
            "metrics_before": len(self.performance_metrics),
            "measured_impact": None,   # filled by evaluate_last_modification
        })
        return True

    def evaluate_last_modification(self) -> Optional[float]:
        """
        Measure the actual performance impact of the last modification.

        Compares mean performance after the modification against the mean
        before it (needs >= 2 post-modification samples). Honest: returns
        None when there is not yet enough data.
        """
        if not self.modification_history:
            return None
        last = self.modification_history[-1]
        if last["measured_impact"] is not None:
            return last["measured_impact"]
        split = last.get("metrics_before", 0)
        pre = self.performance_metrics[:split]
        post = self.performance_metrics[split:]
        if len(pre) < 1 or len(post) < 2:
            return None  # not yet enough post-modification data
        impact = (sum(post) / len(post)) - (sum(pre) / len(pre))
        last["measured_impact"] = round(impact, 4)
        return last["measured_impact"]

    # -------------------------------------------------------------- discovery
    def integrate_discovery(self, discovery: Dict[str, Any]) -> bool:
        """
        Integrate a discovery (e.g. a found causal link) into the
        architecture as a new weighted connection.
        """
        description = str(discovery.get("description", "")).strip()
        if not description:
            return False
        # New pathway: discovery feeds both reasoning and memory
        name = "discovery"
        modules = self.architecture_state["modules"]
        if name not in modules:
            modules[name] = {"weight": 0.8, "health": 1.0, "activations": 0}
        for dst in ("reasoning", "memory"):
            modules.setdefault(dst, {"weight": 1.0, "health": 1.0,
                                     "activations": 0})
            self.architecture_state["connections"][(name, dst)] = 0.6
        self.architecture_state["version"] += 1
        self.modification_history.append({
            "version": self.architecture_state["version"],
            "type": ModificationType.ADD_MODULE.value,
            "target": name,
            "expected_impact": 0.0,
            "confidence": float(discovery.get("confidence", 0.5)),
            "measured_impact": None,
            "discovery": description[:200],
        })
        return True

    # ----------------------------------------------------------- metacognition
    def metacognitive_report(self) -> List[Dict[str, Any]]:
        """
        Depth-limited reflection: real summaries of the actual architecture
        state at increasing levels of abstraction (level 1 = modules,
        level 2 = structure, ... up to metacognition_depth).
        """
        state = self.architecture_state
        levels = [
            {"level": 1, "abstraction": "modules",
             "summary": {name: round(spec["health"], 3)
                         for name, spec in state["modules"].items()}},
            {"level": 2, "abstraction": "connectivity",
             "summary": {"n_modules": len(state["modules"]),
                         "n_connections": len(state["connections"]),
                         "mean_weight": round(
                             sum(state["connections"].values())
                             / max(1, len(state["connections"])), 3)}},
            {"level": 3, "abstraction": "history",
             "summary": {"n_modifications": len(self.modification_history),
                         "version": state["version"]}},
        ]
        deeper = [
            {"level": level, "abstraction": "recursive-reflection",
             "summary": {"levels_below": level - 1,
                         "report_possible": level <= self.metacognition_depth}}
            for level in range(4, self.metacognition_depth + 1)
        ]
        return levels + deeper


# Compatibility alias: V94's embodied learning engine imports this name.
DynamicArchitecture = SelfModifyingArchitecture
