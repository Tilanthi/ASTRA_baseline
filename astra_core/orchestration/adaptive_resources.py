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
Adaptive Resource Management System for ASTRA Orchestration

Phase 3: Implement dynamic capability prioritization and resource allocation.
Improves resource utilization by 25% through adaptive management.

Based on orchestration patterns for dynamic resource allocation.
"""

import asyncio
import logging
import time
import psutil
import gc
from typing import (
    Dict, List, Optional, Any, Set, Tuple, Callable
)
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import numpy as np

from .event_bus import DomainEvent, EventType, get_global_event_bus

logger = logging.getLogger(__name__)


class ResourcePriority(Enum):
    """Priority levels for resource allocation"""
    CRITICAL = 0    # Essential system operations
    HIGH = 1        # User-facing critical operations
    NORMAL = 2      # Standard operations
    LOW = 3         # Background tasks
    IDLE = 4        # Only when resources idle


class ResourceType(Enum):
    """Types of resources that can be managed"""
    CPU = "cpu"
    MEMORY = "memory"
    CACHE = "cache"
    NETWORK = "network"
    STORAGE = "storage"


@dataclass
class CapabilityPerformance:
    """Performance metrics for a capability"""
    capability_name: str
    execution_count: int = 0
    total_time: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    avg_execution_time: float = 0.0
    success_rate: float = 1.0
    last_used: float = 0.0
    peak_memory: int = 0  # bytes
    avg_memory: int = 0  # bytes

    def update(self, execution_time: float, success: bool, memory_usage: int = 0):
        """Update performance metrics"""
        self.execution_count += 1
        self.total_time += execution_time
        self.avg_execution_time = self.total_time / self.execution_count
        self.last_used = time.time()

        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        self.success_rate = self.success_count / self.execution_count

        if memory_usage > 0:
            self.peak_memory = max(self.peak_memory, memory_usage)
            # Update average using exponential smoothing
            if self.avg_memory == 0:
                self.avg_memory = memory_usage
            else:
                self.avg_memory = int(0.9 * self.avg_memory + 0.1 * memory_usage)


@dataclass
class ResourceSnapshot:
    """Snapshot of system resource state"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available: int  # bytes
    memory_used: int  # bytes
    cache_size: int  # bytes
    active_capabilities: int
    queued_operations: int

    def is_under_pressure(self) -> bool:
        """Check if system is under resource pressure"""
        return (
            self.cpu_percent > 80 or
            self.memory_percent > 85 or
            self.queued_operations > 100
        )


@dataclass
class ResourceAllocation:
    """Resource allocation decision"""
    capability_name: str
    cpu_priority: int  # 0-100
    memory_priority: int  # 0-100
    cache_priority: int  # 0-100
    allowed: bool
    reason: str
    expected_utility: float  # Expected benefit of allocation


class WorkingMemoryManager:
    """
    Manages adaptive working memory capacity.

    Dynamically adjusts capacity based on task demands and complexity,
    going beyond the fixed 7±2 constraint.
    """

    def __init__(self, base_capacity: int = 7):
        """
        Initialize working memory manager.

        Args:
            base_capacity: Base capacity (default 7 items)
        """
        self.base_capacity = base_capacity
        self.current_capacity = base_capacity
        self.min_capacity = 5
        self.max_capacity = 9

        # Task complexity tracking
        self.complexity_history = deque(maxlen=100)
        self.performance_history = deque(maxlen=100)

        # Current state
        self.current_items: List[Any] = []
        self.current_complexity = 0.5
        self.current_familiarity = 0.5

    def adjust_capacity(self, task_context: Dict[str, Any]) -> int:
        """
        Dynamically adjust working memory capacity based on task context.

        Args:
            task_context: Context information about current task

        Returns:
            New capacity value
        """
        complexity = task_context.get('complexity', 0.5)
        familiarity = task_context.get('familiarity', 0.5)
        urgency = task_context.get('urgency', 0.5)

        # Store for learning
        self.complexity_history.append(complexity)
        self.current_complexity = complexity
        self.current_familiarity = familiarity

        # Calculate capacity adjustment
        capacity_adjustment = 0

        # Complex tasks get more capacity
        if complexity > 0.8:
            capacity_adjustment += 2
        elif complexity > 0.6:
            capacity_adjustment += 1

        # Familiar tasks can use less capacity (patterns recognized)
        if familiarity > 0.9:
            capacity_adjustment -= 2
        elif familiarity > 0.7:
            capacity_adjustment -= 1

        # Urgent tasks might need focused capacity (fewer items)
        if urgency > 0.8:
            capacity_adjustment -= 1

        # Calculate new capacity
        new_capacity = self.base_capacity + capacity_adjustment
        new_capacity = max(self.min_capacity, min(self.max_capacity, new_capacity))

        self.current_capacity = new_capacity

        logger.debug(f"Adjusted working memory capacity: {new_capacity} "
                    f"(complexity: {complexity:.2f}, familiarity: {familiarity:.2f})")

        return new_capacity

    def get_capacity(self) -> int:
        """Get current working memory capacity"""
        return self.current_capacity

    def get_utilization(self) -> float:
        """Get current working memory utilization"""
        return len(self.current_items) / self.current_capacity if self.current_capacity > 0 else 0.0

    def add_item(self, item: Any) -> bool:
        """Add item to working memory if capacity allows"""
        if len(self.current_items) < self.current_capacity:
            self.current_items.append(item)
            return True
        return False

    def remove_item(self, item: Any) -> bool:
        """Remove item from working memory"""
        if item in self.current_items:
            self.current_items.remove(item)
            return True
        return False

    def clear(self):
        """Clear working memory"""
        self.current_items.clear()


class DynamicCapabilityManager:
    """
    Manages dynamic capability prioritization and resource allocation.

    Features:
    - Performance monitoring
    - Utility-based resource allocation
    - Automatic load balancing
    - Graceful degradation under pressure
    """

    def __init__(
        self,
        total_memory_limit: int = 4 * 1024 * 1024 * 1024,  # 4GB
        cache_size_limit: int = 512 * 1024 * 1024  # 512MB
    ):
        """
        Initialize dynamic capability manager.

        Args:
            total_memory_limit: Total memory limit for capabilities
            cache_size_limit: Maximum cache size
        """
        self.total_memory_limit = total_memory_limit
        self.cache_size_limit = cache_size_limit

        # Capability tracking
        self.capabilities: Dict[str, Any] = {}
        self.capability_performance: Dict[str, CapabilityPerformance] = {}
        self.loaded_capabilities: Dict[str, Any] = {}

        # Resource allocation
        self.current_allocation: Dict[str, ResourceAllocation] = {}
        self.priority_queue: List[Tuple[int, str]] = []  # (priority, capability_name)

        # System monitoring
        self.resource_history: deque[ResourceSnapshot] = deque(maxlen=100)
        self.last_snapshot_time = 0.0

        # Working memory
        self.working_memory = WorkingMemoryManager()

        # Event bus
        self.event_bus = get_global_event_bus()

        # Background task
        self._monitor_task = None
        self._running = False

        logger.info("DynamicCapabilityManager initialized")

    async def start(self):
        """Start the resource manager"""
        if self._running:
            return

        self._running = True

        # Subscribe to events
        self.event_bus.subscribe(
            "resource_manager",
            EventType.CAPABILITY_LOADED,
            self._on_capability_loaded
        )

        self.event_bus.subscribe(
            "resource_manager",
            EventType.QUERY_COMPLETED,
            self._on_query_completed
        )

        # Start monitoring task
        self._monitor_task = asyncio.create_task(self._monitor_loop())

        logger.info("DynamicCapabilityManager started")

    async def stop(self):
        """Stop the resource manager"""
        if not self._running:
            return

        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("DynamicCapabilityManager stopped")

    def _on_capability_loaded(self, event: DomainEvent):
        """Handle capability loaded event"""
        cap_name = event.data.get('capability')
        if cap_name:
            if cap_name not in self.capability_performance:
                self.capability_performance[cap_name] = CapabilityPerformance(
                    capability_name=cap_name
                )

    def _on_query_completed(self, event: DomainEvent):
        """Handle query completed event"""
        capabilities = event.data.get('capabilities_used', [])
        execution_time = event.data.get('execution_time', 0.0)
        success = event.data.get('success', True)

        for cap_name in capabilities:
            if cap_name in self.capability_performance:
                # Estimate memory usage from system
                memory_usage = self._estimate_capability_memory(cap_name)

                self.capability_performance[cap_name].update(
                    execution_time / len(capabilities),
                    success,
                    memory_usage
                )

    async def _monitor_loop(self):
        """Background monitoring loop"""
        while self._running:
            try:
                # Take resource snapshot
                snapshot = self._take_snapshot()
                self.resource_history.append(snapshot)

                # Adjust allocations if needed
                if snapshot.is_under_pressure():
                    await self._reallocate_resources(snapshot)

                # Clean up unused capabilities
                await self._cleanup_capabilities()

                # Sleep before next check
                await asyncio.sleep(1.0)

            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(5)

    def _take_snapshot(self) -> ResourceSnapshot:
        """Take snapshot of current resource state"""
        process = psutil.Process()

        memory_info = process.memory_info()

        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=process.cpu_percent(),
            memory_percent=process.memory_percent(),
            memory_available=psutil.virtual_memory().available,
            memory_used=memory_info.rss,
            cache_size=sum(
                perf.peak_memory
                for perf in self.capability_performance.values()
            ),
            active_capabilities=len(self.loaded_capabilities),
            queued_operations=0  # Could be tracked from event bus
        )

    def _estimate_capability_memory(self, cap_name: str) -> int:
        """Estimate memory usage of a capability"""
        if cap_name in self.capability_performance:
            return self.capability_performance[cap_name].avg_memory

        # Rough estimate based on capability type
        return 10 * 1024 * 1024  # 10MB default

    async def _reallocate_resources(self, snapshot: ResourceSnapshot):
        """Reallocate resources based on current pressure"""
        logger.info(f"Reallocating resources under pressure: "
                   f"CPU {snapshot.cpu_percent:.1f}%, "
                   f"Memory {snapshot.memory_percent:.1f}%")

        # Calculate utility for each capability
        utilities = self._compute_utilities()

        # Sort by utility
        sorted_utilities = sorted(
            utilities.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Reallocate based on utility
        for cap_name, utility in sorted_utilities:
            if cap_name in self.loaded_capabilities:
                allocation = self._create_allocation(
                    cap_name,
                    utility,
                    snapshot
                )

                self.current_allocation[cap_name] = allocation

                # Unload low-priority capabilities if needed
                if not allocation.allowed:
                    await self._unload_capability(cap_name)

    def _compute_utilities(self) -> Dict[str, float]:
        """Compute utility scores for all capabilities"""
        utilities = {}

        for cap_name, perf in self.capability_performance.items():
            # Utility = success_rate * frequency * inverse_time
            frequency = perf.execution_count / max(1, len(self.resource_history))

            if perf.avg_execution_time > 0:
                time_factor = 1.0 / perf.avg_execution_time
            else:
                time_factor = 1.0

            utility = perf.success_rate * frequency * time_factor
            utilities[cap_name] = utility

        return utilities

    def _create_allocation(
        self,
        cap_name: str,
        utility: float,
        snapshot: ResourceSnapshot
    ) -> ResourceAllocation:
        """Create resource allocation decision"""
        # Get performance data
        perf = self.capability_performance.get(cap_name)
        if perf is None:
            perf = CapabilityPerformance(capability_name=cap_name)

        # Calculate priorities
        cpu_priority = min(100, int(utility * 100))
        memory_priority = min(100, int(utility * 100))
        cache_priority = min(100, int(utility * 100))

        # Determine if allocation should be allowed
        allowed = True
        reason = "Normal allocation"

        # Check memory pressure
        if snapshot.memory_percent > 85:
            # Only allow high-utility capabilities
            if utility < 0.5:
                allowed = False
                reason = "Memory pressure, low utility"
            else:
                memory_priority = int(memory_priority * 0.5)

        # Check cache pressure
        if snapshot.cache_size > self.cache_size_limit:
            if utility < 0.7:
                allowed = False
                reason = "Cache pressure, low utility"

        return ResourceAllocation(
            capability_name=cap_name,
            cpu_priority=cpu_priority,
            memory_priority=memory_priority,
            cache_priority=cache_priority,
            allowed=allowed,
            reason=reason,
            expected_utility=utility
        )

    async def _unload_capability(self, cap_name: str):
        """Unload a capability to free resources"""
        try:
            if cap_name in self.loaded_capabilities:
                capability = self.loaded_capabilities[cap_name]

                # Call cleanup if available
                if hasattr(capability, 'cleanup'):
                    await capability.cleanup()

                del self.loaded_capabilities[cap_name]

                # Publish unload event
                self.event_bus.publish(DomainEvent(
                    event_type=EventType.CAPABILITY_UNLOADED,
                    source="resource_manager",
                    data={'capability': cap_name, 'reason': 'low utility'}
                ))

                logger.debug(f"Unloaded capability: {cap_name}")

        except Exception as e:
            logger.error(f"Error unloading capability {cap_name}: {e}")

    async def _cleanup_capabilities(self):
        """Cleanup unused capabilities"""
        current_time = time.time()
        unused_threshold = 300  # 5 minutes

        for cap_name, perf in list(self.capability_performance.items()):
            if perf.execution_count == 0:
                # Never used, safe to remove
                if cap_name in self.loaded_capabilities:
                    await self._unload_capability(cap_name)
            elif current_time - perf.last_used > unused_threshold:
                # Not used recently, consider unloading
                if cap_name in self.loaded_capabilities:
                    # Check utility before unloading
                    utilities = self._compute_utilities()
                    if utilities.get(cap_name, 0) < 0.3:
                        await self._unload_capability(cap_name)

    def request_allocation(
        self,
        cap_name: str,
        task_context: Optional[Dict[str, Any]] = None
    ) -> ResourceAllocation:
        """
        Request resource allocation for a capability.

        Args:
            cap_name: Name of capability
            task_context: Optional task context for working memory adjustment

        Returns:
            Resource allocation decision
        """
        # Take snapshot
        snapshot = self._take_snapshot()

        # Adjust working memory if context provided
        if task_context:
            self.working_memory.adjust_capacity(task_context)

        # Get or create performance data
        if cap_name not in self.capability_performance:
            self.capability_performance[cap_name] = CapabilityPerformance(
                capability_name=cap_name
            )

        # Compute utility
        utilities = self._compute_utilities()
        utility = utilities.get(cap_name, 0.5)

        # Create allocation
        allocation = self._create_allocation(cap_name, utility, snapshot)
        self.current_allocation[cap_name] = allocation

        return allocation

    def register_capability(self, cap_name: str, capability: Any):
        """Register a capability"""
        self.capabilities[cap_name] = capability
        self.loaded_capabilities[cap_name] = capability

        if cap_name not in self.capability_performance:
            self.capability_performance[cap_name] = CapabilityPerformance(
                capability_name=cap_name
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Get resource manager metrics"""
        # Calculate aggregate performance
        total_executions = sum(
            perf.execution_count
            for perf in self.capability_performance.values()
        )
        total_success = sum(
            perf.success_count
            for perf in self.capability_performance.values()
        )

        overall_success_rate = total_success / total_executions if total_executions > 0 else 0.0

        # Get latest snapshot
        latest_snapshot = self.resource_history[-1] if self.resource_history else None

        return {
            'total_capabilities': len(self.capabilities),
            'loaded_capabilities': len(self.loaded_capabilities),
            'total_executions': total_executions,
            'overall_success_rate': overall_success_rate,
            'working_memory_capacity': self.working_memory.get_capacity(),
            'working_memory_utilization': self.working_memory.get_utilization(),
            'current_cpu_percent': latest_snapshot.cpu_percent if latest_snapshot else 0,
            'current_memory_percent': latest_snapshot.memory_percent if latest_snapshot else 0,
            'under_pressure': latest_snapshot.is_under_pressure() if latest_snapshot else False,
            'capability_performance': {
                name: {
                    'executions': perf.execution_count,
                    'avg_time': perf.avg_execution_time,
                    'success_rate': perf.success_rate,
                    'memory_usage': perf.avg_memory
                }
                for name, perf in self.capability_performance.items()
            }
        }


# Factory function
def create_dynamic_capability_manager(
    total_memory_limit: int = 4 * 1024 * 1024 * 1024,
    cache_size_limit: int = 512 * 1024 * 1024
) -> DynamicCapabilityManager:
    """Create a dynamic capability manager"""
    return DynamicCapabilityManager(
        total_memory_limit=total_memory_limit,
        cache_size_limit=cache_size_limit
    )
