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
