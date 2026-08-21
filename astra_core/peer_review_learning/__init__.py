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
Peer Review Learning Architecture for ASTRA

This module implements architectural improvements for learning from expert peer review,
enhancing reasoning, inference, and discovery capabilities through:
- Causal validation with confounder enumeration
- Adversarial hypothesis generation and criticism
- Statistical defense frameworks
- Physics consistency checking
- Peer review memory and feedback integration
- Domain validation gatekeepers
- Epistemic humility and uncertainty quantification
- Peer review simulation

Author: ASTRA Development Team
Version: 1.0
"""

from .causal_validation_framework import CausalValidationFramework
from .hypothesis_critic_engine import HypothesisCriticEngine
from .statistical_defense_framework import StatisticalDefenseFramework
from .physics_consistency_sentinel import PhysicsConsistencySentinel
from .peer_review_memory import PeerReviewMemory
from .domain_validation_gatekeeper import DomainValidationGatekeeper
from .epistemic_humility_engine import EpistemicHumilityEngine
from .peer_review_simulator import PeerReviewSimulator

__all__ = [
    'CausalValidationFramework',
    'HypothesisCriticEngine',
    'StatisticalDefenseFramework',
    'PhysicsConsistencySentinel',
    'PeerReviewMemory',
    'DomainValidationGatekeeper',
    'EpistemicHumilityEngine',
    'PeerReviewSimulator',
]
