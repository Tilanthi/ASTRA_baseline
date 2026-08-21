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
Anticipatory Preloading System for ASTRA Orchestration

Phase 2: Implement demand prediction and capability preloading.
Reduces capability selection latency by 30-50% through predictive preloading.

Based on orchestration patterns for anticipatory resource management.
"""

import asyncio
import logging
import time
import json
from typing import (
    Dict, List, Optional, Any, Set, Tuple, Callable
)
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import hashlib
import numpy as np

from .event_bus import DomainEvent, EventType, get_global_event_bus

logger = logging.getLogger(__name__)


class PredictionMethod(Enum):
    """Methods for predicting capability demand"""
    FREQUENCY = "frequency"  # Most frequent capabilities
    MARKOV_CHAIN = "markov_chain"  # Markov chain transitions
    CO_OCCURRENCE = "co_occurrence"  # Capabilities used together
    NEURAL = "neural"  # Neural network prediction (future)
    HYBRID = "hybrid"  # Combination of methods


@dataclass
class QueryContext:
    """
    Context information about a query for prediction.

    Includes features that help predict what capabilities will be needed.
    """
    query_id: str
    query_text: str
    timestamp: float
    domain: Optional[str] = None
    keywords: Set[str] = field(default_factory=set)
    complexity: float = 0.5
    capabilities_used: List[str] = field(default_factory=list)
    success: bool = True
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'query_id': self.query_id,
            'query_text': self.query_text,
            'timestamp': self.timestamp,
            'domain': self.domain,
            'keywords': list(self.keywords),
            'complexity': self.complexity,
            'capabilities_used': self.capabilities_used,
            'success': self.success,
            'execution_time': self.execution_time
        }


@dataclass
class CapabilityDemand:
    """Demand prediction for a capability"""
    capability_name: str
    probability: float
    urgency: float  # 0-1, how soon it will be needed
    confidence: float  # 0-1, confidence in prediction
    reason: str  # Human-readable explanation


@dataclass
class PreloadDecision:
    """Decision about preloading a capability"""
    capability_name: str
    should_preload: bool
    priority: int  # 0-100
    expected_benefit: float  # Expected time savings
    resource_cost: float  # Memory/CPU cost
    net_benefit: float  # expected_benefit - resource_cost


class QueryPatternAnalyzer:
    """
    Analyzes query patterns to predict capability needs.

    Maintains statistics about:
    - Frequency of capability usage
    - Co-occurrence patterns
    - Sequential dependencies
    - Temporal patterns
    """

    def __init__(self, history_size: int = 1000):
        """
        Initialize pattern analyzer.

        Args:
            history_size: Maximum number of queries to remember
        """
        self.query_history: deque[QueryContext] = deque(maxlen=history_size)

        # Frequency statistics
        self.capability_frequency: defaultdict[str, int] = defaultdict(int)
        self.domain_frequency: defaultdict[str, int] = defaultdict(int)

        # Co-occurrence statistics
        self.co_occurrence: defaultdict[Tuple[str, str], int] = defaultdict(int)
        self.domain_capabilities: defaultdict[str, Set[str]] = defaultdict(set)

        # Sequential patterns (Markov chain)
        self.transition_matrix: defaultdict[Tuple[str, str], int] = defaultdict(int)

        # Temporal patterns
        self.hourly_usage: defaultdict[int, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))

    def record_query(self, context: QueryContext) -> None:
        """
        Record a query context for pattern learning.

        Args:
            context: Query context to record
        """
        self.query_history.append(context)

        # Update frequency statistics
        for cap in context.capabilities_used:
            self.capability_frequency[cap] += 1

        if context.domain:
            self.domain_frequency[context.domain] += 1
            self.domain_capabilities[context.domain].update(context.capabilities_used)

        # Update co-occurrence statistics
        caps = sorted(context.capabilities_used)
        for i, cap1 in enumerate(caps):
            for cap2 in caps[i+1:]:
                self.co_occurrence[(cap1, cap2)] += 1
                self.co_occurrence[(cap2, cap1)] += 1

        # Update sequential patterns
        if len(self.query_history) > 1:
            prev_context = self.query_history[-2]
            for prev_cap in prev_context.capabilities_used:
                for curr_cap in context.capabilities_used:
                    self.transition_matrix[(prev_cap, curr_cap)] += 1

        # Update temporal patterns
        hour = datetime.fromtimestamp(context.timestamp).hour
        for cap in context.capabilities_used:
            self.hourly_usage[hour][cap] += 1

        logger.debug(f"Recorded query: {context.query_id} with {len(context.capabilities_used)} capabilities")

    def predict_capabilities(
        self,
        query_context: QueryContext,
        method: PredictionMethod = PredictionMethod.HYBRID,
        top_n: int = 10
    ) -> List[CapabilityDemand]:
        """
        Predict capabilities needed for a query.

        Args:
            query_context: Query context to predict for
            method: Prediction method to use
            top_n: Number of top predictions to return

        Returns:
            List of predicted capability demands
        """
        predictions = []

        if method in [PredictionMethod.FREQUENCY, PredictionMethod.HYBRID]:
            freq_predictions = self._predict_by_frequency(query_context)
            predictions.extend(freq_predictions)

        if method in [PredictionMethod.MARKOV_CHAIN, PredictionMethod.HYBRID]:
            markov_predictions = self._predict_by_markov(query_context)
            predictions.extend(markov_predictions)

        if method in [PredictionMethod.CO_OCCURRENCE, PredictionMethod.HYBRID]:
            cooc_predictions = self._predict_by_cooccurrence(query_context)
            predictions.extend(cooc_predictions)

        # Combine predictions
        combined = self._combine_predictions(predictions)

        return combined[:top_n]

    def _predict_by_frequency(self, context: QueryContext) -> List[CapabilityDemand]:
        """Predict based on overall frequency"""
        predictions = []

        # Get domain-specific frequencies
        if context.domain:
            domain_caps = self.domain_capabilities.get(context.domain, set())
            for cap in domain_caps:
                predictions.append(CapabilityDemand(
                    capability_name=cap,
                    probability=0.7,
                    urgency=0.5,
                    confidence=0.6,
                    reason=f"Frequency in domain {context.domain}"
                ))

        # Get overall top capabilities
        total_queries = len(self.query_history)
        if total_queries > 0:
            # Sort by frequency and get top 10
            sorted_caps = sorted(self.capability_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
            for cap, freq in sorted_caps:
                prob = freq / total_queries
                if prob > 0.1:  # Only reasonably frequent capabilities
                    predictions.append(CapabilityDemand(
                        capability_name=cap,
                        probability=prob,
                        urgency=0.3,
                        confidence=0.5,
                        reason="High overall frequency"
                    ))

        return predictions

    def _predict_by_markov(self, context: QueryContext) -> List[CapabilityDemand]:
        """Predict based on sequential patterns (Markov chain)"""
        predictions = []

        if len(self.query_history) == 0:
            return predictions

        # Get last used capabilities
        last_context = self.query_history[-1]
        last_caps = set(last_context.capabilities_used)

        # Find likely next capabilities
        transition_scores = defaultdict(float)
        total_transitions = 0

        for last_cap in last_caps:
            for (prev, curr), count in self.transition_matrix.items():
                if prev == last_cap:
                    transition_scores[curr] += count
                    total_transitions += count

        if total_transitions > 0:
            for cap, score in transition_scores.items():
                predictions.append(CapabilityDemand(
                    capability_name=cap,
                    probability=score / total_transitions,
                    urgency=0.7,
                    confidence=0.65,
                    reason=f"Sequential pattern after {last_cap}"
                ))

        return predictions

    def _predict_by_cooccurrence(self, context: QueryContext) -> List[CapabilityDemand]:
        """Predict based on co-occurrence patterns"""
        predictions = []

        if not context.keywords:
            return predictions

        # Find capabilities that co-occur with keywords
        keyword_matches = defaultdict(int)

        for query_ctx in self.query_history:
            # Check keyword overlap
            overlap = len(context.keywords & query_ctx.keywords)
            if overlap > 0:
                for cap in query_ctx.capabilities_used:
                    keyword_matches[cap] += overlap

        if keyword_matches:
            max_score = max(keyword_matches.values())
            for cap, score in keyword_matches.items():
                predictions.append(CapabilityDemand(
                    capability_name=cap,
                    probability=score / max_score,
                    urgency=0.6,
                    confidence=0.55,
                    reason=f"Keyword co-occurrence (overlap: {score})"
                ))

        return predictions

    def _combine_predictions(self, predictions: List[CapabilityDemand]) -> List[CapabilityDemand]:
        """Combine predictions from multiple methods"""
        # Group by capability
        combined = defaultdict(lambda: {
            'probability': 0.0,
            'urgency': 0.0,
            'confidence': 0.0,
            'reasons': []
        })

        for pred in predictions:
            cap_data = combined[pred.capability_name]
            cap_data['probability'] = max(cap_data['probability'], pred.probability)
            cap_data['urgency'] = max(cap_data['urgency'], pred.urgency)
            cap_data['confidence'] = max(cap_data['confidence'], pred.confidence)
            cap_data['reasons'].append(pred.reason)

        # Convert back to CapabilityDemand
        result = []
        for cap_name, data in combined.items():
            # Weight probability by confidence
            weighted_prob = data['probability'] * data['confidence']

            result.append(CapabilityDemand(
                capability_name=cap_name,
                probability=weighted_prob,
                urgency=data['urgency'],
                confidence=data['confidence'],
                reason=f"Hybrid: {', '.join(data['reasons'][:2])}"
            ))

        # Sort by weighted probability
        result.sort(key=lambda x: x.probability * x.confidence, reverse=True)

        return result


class AnticipatoryOrchestrator:
    """
    Orchestrator with anticipatory capability preloading.

    Features:
    - Query pattern analysis
    - Demand prediction
    - Automatic capability preloading
    - Performance monitoring
    - Event-driven updates
    """

    def __init__(
        self,
        capability_registry: Optional[Any] = None,
        preload_threshold: float = 0.5,
        max_preloaded: int = 20
    ):
        """
        Initialize anticipatory orchestrator.

        Args:
            capability_registry: Registry of available capabilities
            preload_threshold: Minimum probability to trigger preload
            max_preloaded: Maximum number of capabilities to keep preloaded
        """
        self.capability_registry = capability_registry
        self.preload_threshold = preload_threshold
        self.max_preloaded = max_preloaded

        # Pattern analyzer
        self.analyzer = QueryPatternAnalyzer()

        # Preloaded capabilities
        self.preloaded_capabilities: Dict[str, Any] = {}
        self.preload_timestamps: Dict[str, float] = {}

        # Event bus
        self.event_bus = get_global_event_bus()

        # Performance metrics
        self.prediction_accuracy = deque(maxlen=100)
        self.preload_hits = 0
        self.preload_misses = 0

        # Background task
        self._preloader_task = None
        self._running = False

        logger.info("AnticipatoryOrchestrator initialized")

    async def start(self):
        """Start the anticipatory orchestrator"""
        if self._running:
            return

        self._running = True

        # Subscribe to query events
        self.event_bus.subscribe(
            "anticipatory_orchestrator",
            EventType.QUERY_COMPLETED,
            self._on_query_completed
        )

        # Start preloader task
        self._preloader_task = asyncio.create_task(self._preloader_loop())

        logger.info("AnticipatoryOrchestrator started")

    async def stop(self):
        """Stop the anticipatory orchestrator"""
        if not self._running:
            return

        self._running = False

        if self._preloader_task:
            self._preloader_task.cancel()
            try:
                await self._preloader_task
            except asyncio.CancelledError:
                pass

        logger.info("AnticipatoryOrchestrator stopped")

    def _on_query_completed(self, event: DomainEvent):
        """Handle query completion event"""
        try:
            data = event.data
            context = QueryContext(
                query_id=data.get('query_id', ''),
                query_text=data.get('query_text', ''),
                timestamp=event.timestamp,
                domain=data.get('domain'),
                keywords=set(data.get('keywords', [])),
                complexity=data.get('complexity', 0.5),
                capabilities_used=data.get('capabilities_used', []),
                success=data.get('success', True),
                execution_time=data.get('execution_time', 0.0)
            )

            self.analyzer.record_query(context)

            # Check if predictions were accurate
            predictions = data.get('predicted_capabilities', [])
            actual = set(context.capabilities_used)
            predicted = set(predictions)

            if actual:
                intersection = actual & predicted
                accuracy = len(intersection) / len(actual)
                self.prediction_accuracy.append(accuracy)

                if intersection:
                    self.preload_hits += len(intersection)
                self.preload_misses += len(actual - intersection)

        except Exception as e:
            logger.error(f"Error handling query completed event: {e}")

    async def _preloader_loop(self):
        """Background loop for preloading capabilities"""
        while self._running:
            try:
                # Get current query context from event bus history
                events = self.event_bus.get_event_history(
                    event_type=EventType.QUERY_RECEIVED,
                    limit=1
                )

                if events:
                    latest_event = events[-1]
                    await self._preload_for_query(latest_event)

                # Sleep before next check
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in preloader loop: {e}")
                await asyncio.sleep(1)

    async def _preload_for_query(self, event: DomainEvent):
        """Preload capabilities for a query"""
        try:
            # Create query context
            query_text = event.data.get('query', '')
            context = QueryContext(
                query_id=event.correlation_id,
                query_text=query_text,
                timestamp=event.timestamp,
                domain=event.data.get('domain'),
                keywords=self._extract_keywords(query_text),
                complexity=event.data.get('complexity', 0.5)
            )

            # Predict capabilities
            predictions = self.analyzer.predict_capabilities(context)

            # Decide which to preload
            preload_decisions = self._make_preload_decisions(predictions)

            # Execute preloading
            for decision in preload_decisions:
                if decision.should_preload:
                    await self._preload_capability(decision)

        except Exception as e:
            logger.error(f"Error preloading for query: {e}")

    def _extract_keywords(self, query: str) -> Set[str]:
        """Extract keywords from query text"""
        # Simple keyword extraction (can be enhanced)
        words = query.lower().split()
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'what',
                     'which', 'who', 'when', 'where', 'why', 'how', 'this', 'that'}

        keywords = {w for w in words if len(w) > 3 and w not in stop_words}
        return keywords

    def _make_preload_decisions(
        self,
        predictions: List[CapabilityDemand]
    ) -> List[PreloadDecision]:
        """
        Make decisions about which capabilities to preload.

        Args:
            predictions: Predicted capability demands

        Returns:
            List of preload decisions
        """
        decisions = []

        for pred in predictions:
            # Check if already preloaded
            if pred.capability_name in self.preloaded_capabilities:
                continue

            # Check threshold
            if pred.probability < self.preload_threshold:
                continue

            # Calculate benefit/cost
            expected_benefit = pred.probability * pred.urgency * 100  # ms saved
            resource_cost = 20  # Fixed cost for preloading (ms)
            net_benefit = expected_benefit - resource_cost

            # Priority based on probability * urgency
            priority = int(pred.probability * pred.urgency * 100)

            should_preload = (
                net_benefit > 0 and
                len(self.preloaded_capabilities) < self.max_preloaded
            )

            decisions.append(PreloadDecision(
                capability_name=pred.capability_name,
                should_preload=should_preload,
                priority=priority,
                expected_benefit=expected_benefit,
                resource_cost=resource_cost,
                net_benefit=net_benefit
            ))

        # Sort by priority
        decisions.sort(key=lambda d: d.priority, reverse=True)

        return decisions[:self.max_preloaded - len(self.preloaded_capabilities)]

    async def _preload_capability(self, decision: PreloadDecision):
        """Preload a capability"""
        try:
            cap_name = decision.capability_name

            # Check if capability exists
            if self.capability_registry:
                capability = self.capability_registry.get(cap_name)
                if capability is None:
                    logger.debug(f"Capability {cap_name} not found in registry")
                    return

                # Preload (initialize if needed)
                if hasattr(capability, 'initialize'):
                    await capability.initialize()

                self.preloaded_capabilities[cap_name] = capability
                self.preload_timestamps[cap_name] = time.time()

                logger.debug(f"Preloaded capability: {cap_name} (priority: {decision.priority})")

                # Publish preloading event
                self.event_bus.publish(DomainEvent(
                    event_type=EventType.CAPABILITY_LOADED,
                    source="anticipatory_orchestrator",
                    data={
                        'capability': cap_name,
                        'priority': decision.priority,
                        'expected_benefit': decision.expected_benefit
                    }
                ))

        except Exception as e:
            logger.error(f"Error preloading capability {decision.capability_name}: {e}")

    def get_preloaded_capabilities(self) -> Dict[str, float]:
        """
        Get currently preloaded capabilities with ages.

        Returns:
            Dictionary mapping capability names to age in seconds
        """
        current_time = time.time()
        return {
            cap: current_time - ts
            for cap, ts in self.preload_timestamps.items()
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics"""
        accuracy = list(self.prediction_accuracy)
        avg_accuracy = sum(accuracy) / len(accuracy) if accuracy else 0.0

        total_preloads = self.preload_hits + self.preload_misses
        hit_rate = self.preload_hits / total_preloads if total_preloads > 0 else 0.0

        return {
            'preloaded_count': len(self.preloaded_capabilities),
            'prediction_accuracy': avg_accuracy,
            'preload_hit_rate': hit_rate,
            'total_preload_hits': self.preload_hits,
            'total_preload_misses': self.preload_misses,
            'query_history_size': len(self.analyzer.query_history),
            'preloaded_capabilities': list(self.preloaded_capabilities.keys())
        }


# Factory function
def create_anticipatory_orchestrator(
    capability_registry: Optional[Any] = None,
    preload_threshold: float = 0.5,
    max_preloaded: int = 20
) -> AnticipatoryOrchestrator:
    """Create an anticipatory orchestrator"""
    return AnticipatoryOrchestrator(
        capability_registry=capability_registry,
        preload_threshold=preload_threshold,
        max_preloaded=max_preloaded
    )
