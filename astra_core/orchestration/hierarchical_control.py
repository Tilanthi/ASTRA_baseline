"""
Hierarchical Meta-Control System for ASTRA Orchestration

Phase 4: Implement explicit operational, tactical, and strategic control layers.
Improves decision quality and system responsiveness through layered control.

Based on orchestration patterns for hierarchical control systems.
"""

import asyncio
import logging
import time
from typing import (
    Dict, List, Optional, Any, Set, Tuple, Callable
)
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import json

from .event_bus import DomainEvent, EventType, get_global_event_bus

logger = logging.getLogger(__name__)


class ControlLayer(Enum):
    """Hierarchical control layers"""
    OPERATIONAL = "operational"  # Millisecond decisions, real-time control
    TACTICAL = "tactical"        # Query-level optimization, seconds
    STRATEGIC = "strategic"      # Session-level learning, minutes to hours


class ControlHorizon(Enum):
    """Time horizons for control decisions"""
    IMMEDIATE = "immediate"     # 0-1 seconds
    SHORT_TERM = "short_term"   # 1-60 seconds
    MEDIUM_TERM = "medium_term" # 1-60 minutes
    LONG_TERM = "long_term"     # 1+ hours


@dataclass
class ControlDecision:
    """A control decision made by a control layer"""
    layer: ControlLayer
    decision_id: str
    action: str
    target: str  # What this decision affects
    parameters: Dict[str, Any]
    priority: int  # 0-100
    timestamp: float
    horizon: ControlHorizon
    expected_impact: float  # 0-1, expected effectiveness
    confidence: float  # 0-1
    rationale: str
    parent_decision: Optional[str] = None  # For hierarchical tracking


@dataclass
class LayerMetrics:
    """Metrics for a control layer"""
    decisions_made: int = 0
    decisions_successful: int = 0
    avg_response_time: float = 0.0
    total_impact: float = 0.0
    last_decision_time: float = 0.0

    def record_decision(self, response_time: float, success: bool, impact: float):
        """Record a decision outcome"""
        self.decisions_made += 1
        if success:
            self.decisions_successful += 1
            self.total_impact += impact

        # Update average response time
        if self.decisions_made == 1:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = (
                0.9 * self.avg_response_time + 0.1 * response_time
            )

        self.last_decision_time = time.time()

    def get_success_rate(self) -> float:
        """Get success rate for this layer"""
        if self.decisions_made == 0:
            return 0.0
        return self.decisions_successful / self.decisions_made


class OperationalController:
    """
    Operational control layer for real-time decisions.

    Handles:
    - Millisecond-scale resource allocation
    - Immediate query routing
    - Real-time performance optimization
    - Emergency responses

    Time horizon: 0-1 seconds
    """

    def __init__(self):
        """Initialize operational controller"""
        self.active = True
        self.metrics = LayerMetrics()
        self.decision_history: deque[ControlDecision] = deque(maxlen=1000)

        # Current state
        self.current_load = 0.0
        self.active_queries = 0
        self.performance_thresholds = {
            'cpu': 80.0,
            'memory': 85.0,
            'latency': 1.0  # seconds
        }

        # Event bus
        self.event_bus = get_global_event_bus()

        logger.info("OperationalController initialized")

    async def make_decision(
        self,
        context: Dict[str, Any]
    ) -> Optional[ControlDecision]:
        """
        Make an operational decision based on current context.

        Args:
            context: Current operational context

        Returns:
            Control decision or None if no action needed
        """
        start_time = time.time()

        # Check for immediate issues
        decision = None

        # Check CPU pressure
        if context.get('cpu_percent', 0) > self.performance_thresholds['cpu']:
            decision = ControlDecision(
                layer=ControlLayer.OPERATIONAL,
                decision_id=f"operational_cpu_{int(time.time() * 1000)}",
                action="throttle_non_critical",
                target="cpu_usage",
                parameters={
                    'threshold': self.performance_thresholds['cpu'],
                    'current': context.get('cpu_percent', 0)
                },
                priority=90,
                timestamp=start_time,
                horizon=ControlHorizon.IMMEDIATE,
                expected_impact=0.8,
                confidence=0.9,
                rationale=f"CPU usage {context.get('cpu_percent', 0):.1f}% exceeds threshold"
            )

        # Check memory pressure
        elif context.get('memory_percent', 0) > self.performance_thresholds['memory']:
            decision = ControlDecision(
                layer=ControlLayer.OPERATIONAL,
                decision_id=f"operational_mem_{int(time.time() * 1000)}",
                action="trigger_gc",
                target="memory_usage",
                parameters={
                    'threshold': self.performance_thresholds['memory'],
                    'current': context.get('memory_percent', 0)
                },
                priority=95,
                timestamp=start_time,
                horizon=ControlHorizon.IMMEDIATE,
                expected_impact=0.9,
                confidence=0.95,
                rationale=f"Memory usage {context.get('memory_percent', 0):.1f}% exceeds threshold"
            )

        # Check latency issues
        elif context.get('avg_latency', 0) > self.performance_thresholds['latency']:
            decision = ControlDecision(
                layer=ControlLayer.OPERATIONAL,
                decision_id=f"operational_lat_{int(time.time() * 1000)}",
                action="increase_capacity",
                target="query_processing",
                parameters={
                    'threshold': self.performance_thresholds['latency'],
                    'current': context.get('avg_latency', 0)
                },
                priority=85,
                timestamp=start_time,
                horizon=ControlHorizon.IMMEDIATE,
                expected_impact=0.75,
                confidence=0.85,
                rationale=f"Latency {context.get('avg_latency', 0):.2f}s exceeds threshold"
            )

        if decision:
            self.decision_history.append(decision)

            # Publish decision
            self.event_bus.publish(DomainEvent(
                event_type=EventType.DOMAIN_LOADED,  # Reuse event type
                source="operational_controller",
                data={
                    'decision': decision.to_dict() if hasattr(decision, 'to_dict') else
                    {'layer': decision.layer.value, 'action': decision.action}
                }
            ))

            response_time = time.time() - start_time
            self.metrics.record_decision(response_time, True, decision.expected_impact)

            logger.debug(f"Operational decision: {decision.action} (priority: {decision.priority})")

        return decision

    def get_status(self) -> Dict[str, Any]:
        """Get operational controller status"""
        return {
            'active': self.active,
            'current_load': self.current_load,
            'active_queries': self.active_queries,
            'decisions_made': self.metrics.decisions_made,
            'success_rate': self.metrics.get_success_rate(),
            'avg_response_time': self.metrics.avg_response_time
        }


class TacticalController:
    """
    Tactical control layer for query-level optimization.

    Handles:
    - Query routing and capability selection
    - Resource allocation for queries
    - Performance optimization strategies
    - Adaptation to query patterns

    Time horizon: 1-60 seconds
    """

    def __init__(self):
        """Initialize tactical controller"""
        self.active = True
        self.metrics = LayerMetrics()
        self.decision_history: deque[ControlDecision] = deque(maxlen=500)

        # Query statistics
        self.query_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'success_count': 0,
            'avg_time': 0.0,
            'success_rate': 1.0
        })

        # Routing strategies
        self.routing_strategies = {
            'fast_path': {
                'condition': lambda ctx: ctx.get('complexity', 0.5) < 0.3,
                'capabilities': ['cache', 'llm_inference']
            },
            'standard_path': {
                'condition': lambda ctx: 0.3 <= ctx.get('complexity', 0.5) < 0.7,
                'capabilities': ['causal_discovery', 'abductive_inference']
            },
            'deep_analysis': {
                'condition': lambda ctx: ctx.get('complexity', 0.5) >= 0.7,
                'capabilities': ['causal_discovery', 'abductive_inference', 'meta_learning']
            }
        }

        # Event bus
        self.event_bus = get_global_event_bus()

        logger.info("TacticalController initialized")

    async def make_decision(
        self,
        query_context: Dict[str, Any]
    ) -> Optional[ControlDecision]:
        """
        Make a tactical decision for query handling.

        Args:
            query_context: Query context information

        Returns:
            Control decision for query routing
        """
        start_time = time.time()

        # Determine routing strategy
        strategy = self._select_strategy(query_context)

        if strategy:
            decision = ControlDecision(
                layer=ControlLayer.TACTICAL,
                decision_id=f"tactical_route_{int(time.time() * 1000)}",
                action="route_query",
                target=query_context.get('query_id', 'unknown'),
                parameters={
                    'strategy': strategy,
                    'capabilities': self.routing_strategies[strategy]['capabilities'],
                    'complexity': query_context.get('complexity', 0.5)
                },
                priority=70,
                timestamp=start_time,
                horizon=ControlHorizon.SHORT_TERM,
                expected_impact=0.7,
                confidence=0.8,
                rationale=f"Query complexity {query_context.get('complexity', 0.5):.2f} requires {strategy} strategy"
            )

            self.decision_history.append(decision)

            # Publish decision
            self.event_bus.publish(DomainEvent(
                event_type=EventType.QUERY_ASSIGNED,
                source="tactical_controller",
                data={
                    'decision': {
                        'layer': decision.layer.value,
                        'action': decision.action,
                        'strategy': strategy
                    }
                }
            ))

            response_time = time.time() - start_time
            self.metrics.record_decision(response_time, True, decision.expected_impact)

            logger.debug(f"Tactical decision: Route to {strategy} (priority: {decision.priority})")

            return decision

        return None

    def _select_strategy(self, context: Dict[str, Any]) -> Optional[str]:
        """Select appropriate routing strategy"""
        for strategy_name, strategy_config in self.routing_strategies.items():
            try:
                if strategy_config['condition'](context):
                    return strategy_name
            except Exception as e:
                logger.warning(f"Error evaluating strategy {strategy_name}: {e}")

        return 'standard_path'  # Default

    def update_query_stats(self, query_type: str, execution_time: float, success: bool):
        """Update query statistics for learning"""
        stats = self.query_stats[query_type]
        stats['count'] += 1
        stats['total_time'] += execution_time
        stats['avg_time'] = stats['total_time'] / stats['count']

        if success:
            stats['success_count'] += 1

        stats['success_rate'] = stats['success_count'] / stats['count']

    def get_status(self) -> Dict[str, Any]:
        """Get tactical controller status"""
        return {
            'active': self.active,
            'decisions_made': self.metrics.decisions_made,
            'success_rate': self.metrics.get_success_rate(),
            'avg_response_time': self.metrics.avg_response_time,
            'query_types': len(self.query_stats),
            'top_query_types': dict(
                sorted(self.query_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:5]
            )
        }


class StrategicController:
    """
    Strategic control layer for session-level optimization.

    Handles:
    - Long-term learning and adaptation
    - Session-level policy optimization
    - Resource allocation strategies
    - System-wide optimization

    Time horizon: 1-60 minutes to hours
    """

    def __init__(self, optimization_interval: int = 300):  # 5 minutes
        """
        Initialize strategic controller.

        Args:
            optimization_interval: Seconds between optimization runs
        """
        self.active = True
        self.optimization_interval = optimization_interval
        self.metrics = LayerMetrics()
        self.decision_history: deque[ControlDecision] = deque(maxlen=100)

        # Session statistics
        self.session_start_time = time.time()
        self.session_metrics = {
            'total_queries': 0,
            'total_decisions': 0,
            'total_impact': 0.0
        }

        # Learned policies
        self.policies: Dict[str, Dict[str, Any]] = {}

        # Event bus
        self.event_bus = get_global_event_bus()

        # Background task
        self._optimizer_task = None
        self._running = False

        logger.info("StrategicController initialized")

    async def start(self):
        """Start the strategic controller"""
        if self._running:
            return

        self._running = True
        self._optimizer_task = asyncio.create_task(self._optimization_loop())

        logger.info("StrategicController started")

    async def stop(self):
        """Stop the strategic controller"""
        if not self._running:
            return

        self._running = False

        if self._optimizer_task:
            self._optimizer_task.cancel()
            try:
                await self._optimizer_task
            except asyncio.CancelledError:
                pass

        logger.info("StrategicController stopped")

    async def _optimization_loop(self):
        """Background optimization loop"""
        while self._running:
            try:
                await asyncio.sleep(self.optimization_interval)
                await self._run_optimization()

            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")

    async def _run_optimization(self):
        """Run strategic optimization"""
        start_time = time.time()

        # Analyze session performance
        session_duration = time.time() - self.session_start_time
        avg_impact = (
            self.session_metrics['total_impact'] /
            max(1, self.session_metrics['total_decisions'])
        )

        # Generate optimization recommendations
        decision = ControlDecision(
            layer=ControlLayer.STRATEGIC,
            decision_id=f"strategic_opt_{int(time.time())}",
            action="optimize_policies",
            target="system_policies",
            parameters={
                'session_duration': session_duration,
                'avg_impact': avg_impact,
                'total_queries': self.session_metrics['total_queries']
            },
            priority=50,
            timestamp=start_time,
            horizon=ControlHorizon.LONG_TERM,
            expected_impact=0.6,
            confidence=0.7,
            rationale=f"Session optimization after {session_duration:.0f}s"
        )

        self.decision_history.append(decision)

        # Publish optimization event
        self.event_bus.publish(DomainEvent(
            event_type=EventType.SYSTEM_STATUS,
            source="strategic_controller",
            data={
                'optimization': {
                    'session_duration': session_duration,
                    'avg_impact': avg_impact,
                    'recommendations': self._generate_recommendations()
                }
            }
        ))

        response_time = time.time() - start_time
        self.metrics.record_decision(response_time, True, decision.expected_impact)

        logger.info(f"Strategic optimization completed (impact: {avg_impact:.2f})")

    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        # Analyze session metrics
        if self.session_metrics['total_queries'] > 100:
            recommendations.append("Consider caching frequent queries")

        if self.metrics.get_success_rate() < 0.8:
            recommendations.append("Review decision accuracy thresholds")

        recommendations.append("Continue monitoring system performance")

        return recommendations

    async def make_decision(
        self,
        session_context: Dict[str, Any]
    ) -> Optional[ControlDecision]:
        """
        Make a strategic decision for session optimization.

        Args:
            session_context: Session-level context

        Returns:
            Control decision for strategic actions
        """
        # Strategic decisions are mostly made by the optimization loop
        # This method is for ad-hoc strategic decisions

        if session_context.get('trigger_optimization', False):
            return await self._run_optimization_decision()

        return None

    async def _run_optimization_decision(self) -> ControlDecision:
        """Create an optimization decision"""
        start_time = time.time()

        decision = ControlDecision(
            layer=ControlLayer.STRATEGIC,
            decision_id=f"strategic_manual_{int(time.time())}",
            action="trigger_optimization",
            target="optimization_loop",
            parameters={'manual_trigger': True},
            priority=60,
            timestamp=start_time,
            horizon=ControlHorizon.MEDIUM_TERM,
            expected_impact=0.65,
            confidence=0.75,
            rationale="Manual optimization trigger"
        )

        self.decision_history.append(decision)

        # Trigger optimization
        await self._run_optimization()

        return decision

    def get_status(self) -> Dict[str, Any]:
        """Get strategic controller status"""
        session_duration = time.time() - self.session_start_time

        return {
            'active': self.active,
            'session_duration': session_duration,
            'total_queries': self.session_metrics['total_queries'],
            'total_decisions': self.session_metrics['total_decisions'],
            'decisions_made': self.metrics.decisions_made,
            'success_rate': self.metrics.get_success_rate(),
            'avg_response_time': self.metrics.avg_response_time,
            'optimization_interval': self.optimization_interval,
            'policies_count': len(self.policies)
        }


class HierarchicalMetaController:
    """
    Main hierarchical meta-controller with explicit operational, tactical,
    and strategic layers.

    Coordinates decision-making across time horizons and ensures
    appropriate responses to different types of situations.
    """

    def __init__(self, optimization_interval: int = 300):
        """
        Initialize hierarchical meta-controller.

        Args:
            optimization_interval: Interval for strategic optimization (seconds)
        """
        # Control layers
        self.operational = OperationalController()
        self.tactical = TacticalController()
        self.strategic = StrategicController(optimization_interval)

        # Decision tracking
        self.all_decisions: deque[ControlDecision] = deque(maxlen=2000)
        self.decision_chain: Dict[str, List[str]] = {}  # parent -> children

        # Event bus
        self.event_bus = get_global_event_bus()

        # Background task
        self._running = False

        logger.info("HierarchicalMetaController initialized")

    async def start(self):
        """Start the meta-controller"""
        if self._running:
            return

        self._running = True

        # Start strategic controller
        await self.strategic.start()

        # Subscribe to events
        self.event_bus.subscribe(
            "meta_controller",
            EventType.QUERY_RECEIVED,
            self._on_query_received
        )

        logger.info("HierarchicalMetaController started")

    async def stop(self):
        """Stop the meta-controller"""
        if not self._running:
            return

        self._running = False

        await self.strategic.stop()

        logger.info("HierarchicalMetaController stopped")

    async def _on_query_received(self, event: DomainEvent):
        """Handle query received event"""
        query_context = event.data

        # Route through control layers
        await self.route_decision(query_context)

    async def route_decision(self, context: Dict[str, Any]) -> List[ControlDecision]:
        """
        Route context through appropriate control layers.

        Args:
            context: Decision context

        Returns:
            List of decisions made by control layers
        """
        decisions = []

        # Try operational layer first (fastest)
        if self._needs_operational_response(context):
            operational_decision = await self.operational.make_decision(context)
            if operational_decision:
                decisions.append(operational_decision)
                self.all_decisions.append(operational_decision)

        # Try tactical layer
        if self._needs_tactical_response(context):
            tactical_decision = await self.tactical.make_decision(context)
            if tactical_decision:
                decisions.append(tactical_decision)
                self.all_decisions.append(tactical_decision)

                # Link to operational decision if exists
                if len(decisions) > 1:
                    self.decision_chain[decisions[0].decision_id] = [tactical_decision.decision_id]

        return decisions

    def _needs_operational_response(self, context: Dict[str, Any]) -> bool:
        """Check if context requires operational response"""
        return (
            context.get('cpu_percent', 0) > 80 or
            context.get('memory_percent', 0) > 85 or
            context.get('avg_latency', 0) > 1.0
        )

    def _needs_tactical_response(self, context: Dict[str, Any]) -> bool:
        """Check if context requires tactical response"""
        return 'query_id' in context or 'query_text' in context

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get status of all control layers"""
        return {
            'operational': self.operational.get_status(),
            'tactical': self.tactical.get_status(),
            'strategic': self.strategic.get_status(),
            'total_decisions': len(self.all_decisions),
            'active_layers': sum([
                self.operational.active,
                self.tactical.active,
                self.strategic.active
            ])
        }

    def get_decision_history(
        self,
        layer: Optional[ControlLayer] = None,
        limit: int = 100
    ) -> List[ControlDecision]:
        """Get decision history with optional layer filter"""
        decisions = list(self.all_decisions)

        if layer is not None:
            decisions = [d for d in decisions if d.layer == layer]

        return decisions[-limit:]


# Factory function
def create_hierarchical_meta_controller(
    optimization_interval: int = 300
) -> HierarchicalMetaController:
    """Create a hierarchical meta-controller"""
    return HierarchicalMetaController(optimization_interval=optimization_interval)


# Add to_dict method to ControlDecision for serialization
def control_decision_to_dict(self) -> Dict[str, Any]:
    """Convert decision to dictionary"""
    return {
        'layer': self.layer.value,
        'decision_id': self.decision_id,
        'action': self.action,
        'target': self.target,
        'parameters': self.parameters,
        'priority': self.priority,
        'timestamp': self.timestamp,
        'horizon': self.horizon.value,
        'expected_impact': self.expected_impact,
        'confidence': self.confidence,
        'rationale': self.rationale,
        'parent_decision': self.parent_decision
    }

ControlDecision.to_dict = control_decision_to_dict
