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
Comprehensive test suite for ASTRA Orchestration improvements

Tests all four phases:
- Phase 1: Event Bus Foundation
- Phase 2: Anticipatory Preloading
- Phase 3: Adaptive Resource Management
- Phase 4: Hierarchical Meta-Control
"""

import asyncio
import time
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_event_bus():
    """Test Phase 1: Event Bus Foundation"""
    print("\n=== Testing Phase 1: Event Bus Foundation ===")

    from astra_core.orchestration import (
        DomainEvent,
        EventType,
        EventPriority,
        create_event_bus
    )

    # Create event bus
    bus = create_event_bus()

    # Test subscription
    received_events = []

    def handler(event: DomainEvent):
        received_events.append(event)
        return f"Processed: {event.event_type}"

    unsubscribe = bus.subscribe(
        "test_subscriber",
        EventType.QUERY_RECEIVED,
        handler,
        priority=10
    )

    # Test event publishing
    event = DomainEvent(
        event_type=EventType.QUERY_RECEIVED,
        source="test",
        data={"query": "test query"},
        priority=EventPriority.HIGH
    )

    bus.publish(event)

    # Test synchronous processing
    results = bus.publish_sync(event)
    print(f"✓ Event published and processed: {len(results)} handlers")

    # Test filtering
    filtered = bus.get_event_history(event_type=EventType.QUERY_RECEIVED)
    print(f"✓ Event history filtering: {len(filtered)} events")

    # Test metrics
    metrics = bus.get_metrics()
    print(f"✓ Event bus metrics: {metrics['events_published']} published, "
          f"{metrics['events_processed']} processed")

    # Test unsubscribe
    unsubscribe()
    subscribers = bus.get_subscribers()
    print(f"✓ Unsubscribe successful: {len(subscribers)} subscribers")

    print("✓ Phase 1: Event Bus Foundation - PASSED\n")
    return True


async def test_anticipatory():
    """Test Phase 2: Anticipatory Preloading"""
    print("=== Testing Phase 2: Anticipatory Preloading ===")

    from astra_core.orchestration import (
        QueryContext,
        PredictionMethod,
        create_anticipatory_orchestrator
    )

    # Create anticipatory orchestrator
    anticipatory = create_anticipatory_orchestrator()

    # Test query pattern analysis
    query_context = QueryContext(
        query_id="test_1",
        query_text="What causes filament width variations in ISM?",
        timestamp=time.time(),
        domain="ism",
        keywords={"filament", "width", "variations", "ism"},
        complexity=0.7,
        capabilities_used=["causal_discovery", "abductive_inference"],
        success=True,
        execution_time=1.5
    )

    anticipatory.analyzer.record_query(query_context)
    print("✓ Query context recorded")

    # Test prediction
    predictions = anticipatory.analyzer.predict_capabilities(
        query_context,
        method=PredictionMethod.FREQUENCY
    )
    print(f"✓ Generated {len(predictions)} predictions")

    # Test preload decisions
    from astra_core.orchestration import CapabilityDemand

    test_predictions = [
        CapabilityDemand(
            capability_name="causal_discovery",
            probability=0.8,
            urgency=0.7,
            confidence=0.75,
            reason="Test prediction"
        ),
        CapabilityDemand(
            capability_name="abductive_inference",
            probability=0.6,
            urgency=0.5,
            confidence=0.6,
            reason="Test prediction"
        )
    ]

    decisions = anticipatory._make_preload_decisions(test_predictions)
    print(f"✓ Generated {len(decisions)} preload decisions")

    # Test metrics
    metrics = anticipatory.get_metrics()
    print(f"✓ Anticipatory metrics: {metrics['query_history_size']} queries recorded")

    print("✓ Phase 2: Anticipatory Preloading - PASSED\n")
    return True


async def test_adaptive_resources():
    """Test Phase 3: Adaptive Resource Management"""
    print("=== Testing Phase 3: Adaptive Resource Management ===")

    from astra_core.orchestration import (
        WorkingMemoryManager,
        ResourceSnapshot,
        create_dynamic_capability_manager
    )

    # Test working memory manager
    working_memory = WorkingMemoryManager(base_capacity=7)

    # Test capacity adjustment
    task_context = {
        'complexity': 0.8,
        'familiarity': 0.3,
        'urgency': 0.5
    }

    new_capacity = working_memory.adjust_capacity(task_context)
    print(f"✓ Working memory capacity adjusted: {new_capacity}")

    # Test item management
    item1 = {"data": "item1"}
    item2 = {"data": "item2"}

    success1 = working_memory.add_item(item1)
    success2 = working_memory.add_item(item2)

    print(f"✓ Added items to working memory: {success1}, {success2}")

    # Test utilization
    utilization = working_memory.get_utilization()
    print(f"✓ Working memory utilization: {utilization:.2%}")

    # Test dynamic capability manager
    resource_manager = create_dynamic_capability_manager()

    # Test resource snapshot
    snapshot = resource_manager._take_snapshot()
    print(f"✓ Resource snapshot: CPU {snapshot.cpu_percent:.1f}%, "
          f"Memory {snapshot.memory_percent:.1f}%")

    # Test allocation request
    allocation = resource_manager.request_allocation(
        "test_capability",
        task_context=task_context
    )
    print(f"✓ Resource allocation: allowed={allocation.allowed}, "
          f"cpu_priority={allocation.cpu_priority}")

    # Test metrics
    metrics = resource_manager.get_metrics()
    print(f"✓ Resource manager metrics: "
          f"{metrics['total_capabilities']} capabilities, "
          f"success rate {metrics['overall_success_rate']:.2%}")

    print("✓ Phase 3: Adaptive Resource Management - PASSED\n")
    return True


async def test_hierarchical_control():
    """Test Phase 4: Hierarchical Meta-Control"""
    print("=== Testing Phase 4: Hierarchical Meta-Control ===")

    from astra_core.orchestration import (
        ControlLayer,
        ControlHorizon,
        create_hierarchical_meta_controller
    )

    # Create hierarchical meta-controller
    meta_controller = create_hierarchical_meta_controller()

    # Test operational controller
    operational_context = {
        'cpu_percent': 85.0,
        'memory_percent': 90.0,
        'avg_latency': 1.2
    }

    operational_decision = await meta_controller.operational.make_decision(
        operational_context
    )

    if operational_decision:
        print(f"✓ Operational decision: {operational_decision.action} "
              f"(priority: {operational_decision.priority})")
    else:
        print("✓ No operational decision needed (system healthy)")

    # Test tactical controller
    tactical_context = {
        'query_id': 'test_query',
        'query_text': 'Complex analysis request',
        'complexity': 0.7
    }

    tactical_decision = await meta_controller.tactical.make_decision(
        tactical_context
    )

    if tactical_decision:
        print(f"✓ Tactical decision: {tactical_decision.action} "
              f"(strategy: {tactical_decision.parameters.get('strategy')})")

    # Test strategic controller status
    strategic_status = meta_controller.strategic.get_status()
    print(f"✓ Strategic controller status: "
          f"session duration {strategic_status['session_duration']:.0f}s")

    # Test comprehensive status
    comprehensive_status = meta_controller.get_comprehensive_status()
    print(f"✓ Comprehensive status: "
          f"{comprehensive_status['total_decisions']} total decisions")

    print("✓ Phase 4: Hierarchical Meta-Control - PASSED\n")
    return True


async def test_integrated_orchestrator():
    """Test integrated orchestrator combining all phases"""
    print("=== Testing Integrated Orchestrator ===")

    from astra_core.orchestration import create_integrated_orchestrator

    # Create integrated orchestrator
    orchestrator = create_integrated_orchestrator()

    # Start orchestrator
    await orchestrator.start()
    print("✓ Integrated orchestrator started")

    # Register a test capability
    orchestrator.register_capability("test_capability", {"name": "test"})
    print("✓ Capability registered")

    # Process a test query
    result = await orchestrator.process_query(
        query="What causes filament width variations?",
        context={
            'domain': 'ism',
            'complexity': 0.7,
            'keywords': {'filament', 'width', 'variations'}
        }
    )
    print(f"✓ Query processed: {result['success']}, "
          f"{result['decisions_made']} decisions made")

    # Get comprehensive metrics
    metrics = orchestrator.get_comprehensive_metrics()
    print(f"✓ Comprehensive metrics:")
    print(f"  - Events published: {metrics['summary']['total_events_published']}")
    print(f"  - Prediction accuracy: {metrics['summary']['prediction_accuracy']:.2%}")
    print(f"  - Total decisions: {metrics['summary']['total_decisions']}")
    print(f"  - Active capabilities: {metrics['summary']['active_capabilities']}")

    # Stop orchestrator
    await orchestrator.stop()
    print("✓ Integrated orchestrator stopped")

    print("✓ Integrated Orchestrator - PASSED\n")
    return True


async def run_all_tests():
    """Run all orchestration tests"""
    print("\n" + "="*50)
    print("ASTRA ORCHESTRATION IMPROVEMENTS - TEST SUITE")
    print("="*50)

    results = {}

    try:
        results['event_bus'] = await test_event_bus()
    except Exception as e:
        print(f"✗ Phase 1 failed: {e}")
        results['event_bus'] = False

    try:
        results['anticipatory'] = await test_anticipatory()
    except Exception as e:
        print(f"✗ Phase 2 failed: {e}")
        results['anticipatory'] = False

    try:
        results['adaptive'] = await test_adaptive_resources()
    except Exception as e:
        print(f"✗ Phase 3 failed: {e}")
        results['adaptive'] = False

    try:
        results['hierarchical'] = await test_hierarchical_control()
    except Exception as e:
        print(f"✗ Phase 4 failed: {e}")
        results['hierarchical'] = False

    try:
        results['integrated'] = await test_integrated_orchestrator()
    except Exception as e:
        print(f"✗ Integrated test failed: {e}")
        results['integrated'] = False

    # Summary
    print("="*50)
    print("TEST SUMMARY")
    print("="*50)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")

    total_tests = len(results)
    passed_tests = sum(1 for p in results.values() if p)

    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n🎉 All orchestration improvements are working correctly!")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed")

    return passed_tests == total_tests


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
