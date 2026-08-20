"""
Integrated Orchestrator for ASTRA

Combines all four orchestration phases into a unified system:
- Phase 1: Event Bus Foundation
- Phase 2: Anticipatory Preloading
- Phase 3: Adaptive Resource Management
- Phase 4: Hierarchical Meta-Control

This is the main entry point for using ASTRA's orchestration improvements.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .event_bus import DomainEvent, EventType, get_global_event_bus
from .anticipatory import AnticipatoryOrchestrator, create_anticipatory_orchestrator
from .adaptive_resources import DynamicCapabilityManager, create_dynamic_capability_manager
from .hierarchical_control import HierarchicalMetaController, create_hierarchical_meta_controller

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorMetrics:
    """Comprehensive metrics for the integrated orchestrator"""
    # Event bus metrics
    events_published: int = 0
    events_processed: int = 0

    # Anticipatory metrics
    prediction_accuracy: float = 0.0
    preload_hit_rate: float = 0.0
    preloaded_capabilities: int = 0

    # Resource metrics
    total_executions: int = 0
    overall_success_rate: float = 0.0
    working_memory_capacity: int = 7
    working_memory_utilization: float = 0.0

    # Control metrics
    operational_decisions: int = 0
    tactical_decisions: int = 0
    strategic_decisions: int = 0


class IntegratedOrchestrator:
    """
    Unified orchestrator combining all orchestration phases.

    Provides:
    - Event-driven communication between components
    - Anticipatory capability preloading
    - Adaptive resource management
    - Hierarchical meta-control

    Usage:
        orchestrator = IntegratedOrchestrator()
        await orchestrator.start()

        # Process queries
        result = await orchestrator.process_query(
            query="What causes filament width variations?",
            context={'domain': 'ism'}
        )

        await orchestrator.stop()
    """

    def __init__(
        self,
        preload_threshold: float = 0.5,
        max_preloaded: int = 20,
        optimization_interval: int = 300
    ):
        """
        Initialize integrated orchestrator.

        Args:
            preload_threshold: Minimum probability for preloading capabilities
            max_preloaded: Maximum number of preloaded capabilities
            optimization_interval: Interval for strategic optimization (seconds)
        """
        # Event bus
        self.event_bus = get_global_event_bus()

        # Phase 2: Anticipatory orchestrator
        self.anticipatory = create_anticipatory_orchestrator(
            preload_threshold=preload_threshold,
            max_preloaded=max_preloaded
        )

        # Phase 3: Resource manager
        self.resource_manager = create_dynamic_capability_manager()

        # Phase 4: Meta-controller
        self.meta_controller = create_hierarchical_meta_controller(
            optimization_interval=optimization_interval
        )

        # Capability registry
        self.capability_registry: Dict[str, Any] = {}

        # State
        self._running = False

        logger.info("IntegratedOrchestrator initialized")

    async def start(self):
        """Start the integrated orchestrator"""
        if self._running:
            logger.warning("IntegratedOrchestrator already running")
            return

        self._running = True

        # Start event bus
        await self.event_bus.start()

        # Start anticipatory orchestrator
        await self.anticipatory.start()

        # Start resource manager
        await self.resource_manager.start()

        # Start meta-controller
        await self.meta_controller.start()

        logger.info("IntegratedOrchestrator started with all phases active")

    async def stop(self):
        """Stop the integrated orchestrator"""
        if not self._running:
            return

        self._running = False

        # Stop in reverse order
        await self.meta_controller.stop()
        await self.resource_manager.stop()
        await self.anticipatory.stop()
        await self.event_bus.stop()

        logger.info("IntegratedOrchestrator stopped")

    def register_capability(self, name: str, capability: Any):
        """Register a capability with the orchestrator"""
        self.capability_registry[name] = capability
        self.resource_manager.register_capability(name, capability)

        logger.debug(f"Registered capability: {name}")

    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a query through the orchestration system.

        Args:
            query: Query text
            context: Additional context (domain, complexity, etc.)

        Returns:
            Processing result with metadata
        """
        context = context or {}
        start_time = asyncio.get_event_loop().time()

        # Publish query received event
        event_data = {
            'query': query,
            'query_id': f"query_{int(start_time * 1000)}",
        }
        # Add context, converting sets to lists for JSON serialization
        for key, value in context.items():
            if isinstance(value, set):
                event_data[key] = list(value)
            else:
                event_data[key] = value

        self.event_bus.publish(DomainEvent(
            event_type=EventType.QUERY_RECEIVED,
            source="integrated_orchestrator",
            data=event_data
        ))

        # Get resource allocation
        allocation = self.resource_manager.request_allocation(
            cap_name=context.get('domain', 'general'),
            task_context=context
        )

        # Route through meta-controller
        control_context = {
            **context,
            'query': query,
            'query_id': f"query_{int(start_time * 1000)}",
            'cpu_percent': 0,  # Could be filled from monitoring
            'memory_percent': 0,
            'avg_latency': 0
        }

        decisions = await self.meta_controller.route_decision(control_context)

        # Process query with allocated capabilities
        result = {
            'query': query,
            'success': True,
            'answer': "Query processed through orchestration system",
            'decisions_made': len(decisions),
            'allocation': {
                'allowed': allocation.allowed,
                'priority': allocation.cpu_priority
            }
        }

        # Publish query completed event
        processing_time = asyncio.get_event_loop().time() - start_time

        # Convert any sets to lists for JSON serialization
        keywords = context.get('keywords', set())
        if isinstance(keywords, set):
            keywords = list(keywords)

        self.event_bus.publish(DomainEvent(
            event_type=EventType.QUERY_COMPLETED,
            source="integrated_orchestrator",
            data={
                'query': query,
                'query_id': context.get('query_id', f"query_{int(start_time * 1000)}"),
                'success': result['success'],
                'execution_time': processing_time,
                'capabilities_used': context.get('capabilities_used', []),
                'predicted_capabilities': context.get('predicted_capabilities', []),
                'domain': context.get('domain'),
                'complexity': context.get('complexity', 0.5),
                'keywords': keywords
            }
        ))

        return result

    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all orchestration phases"""
        return {
            'event_bus': self.event_bus.get_metrics(),
            'anticipatory': self.anticipatory.get_metrics(),
            'resources': self.resource_manager.get_metrics(),
            'meta_control': self.meta_controller.get_comprehensive_status(),
            'summary': self._summarize_metrics()
        }

    def _summarize_metrics(self) -> Dict[str, Any]:
        """Summarize key metrics across all phases"""
        return {
            'total_events_published': self.event_bus.get_metrics()['events_published'],
            'prediction_accuracy': self.anticipatory.get_metrics()['prediction_accuracy'],
            'preload_hit_rate': self.anticipatory.get_metrics()['preload_hit_rate'],
            'overall_success_rate': self.resource_manager.get_metrics()['overall_success_rate'],
            'total_decisions': (
                self.meta_controller.operational.metrics.decisions_made +
                self.meta_controller.tactical.metrics.decisions_made +
                self.meta_controller.strategic.metrics.decisions_made
            ),
            'active_capabilities': len(self.capability_registry),
            'system_running': self._running
        }


# Factory function
def create_integrated_orchestrator(
    preload_threshold: float = 0.5,
    max_preloaded: int = 20,
    optimization_interval: int = 300
) -> IntegratedOrchestrator:
    """Create an integrated orchestrator"""
    return IntegratedOrchestrator(
        preload_threshold=preload_threshold,
        max_preloaded=max_preloaded,
        optimization_interval=optimization_interval
    )


__all__ = [
    'IntegratedOrchestrator',
    'OrchestratorMetrics',
    'create_integrated_orchestrator'
]
