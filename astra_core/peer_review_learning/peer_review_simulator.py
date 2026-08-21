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
Peer Review Simulation Module

Simulates expert peer review before publication, generating pre-emptive criticism
and enabling stronger claims through adversarial testing.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
from datetime import datetime

# Import other peer review learning modules
from .causal_validation_framework import CausalValidationFramework, CausalClaim
from .hypothesis_critic_engine import HypothesisCriticEngine, Hypothesis, CritiqueResult
from .statistical_defense_framework import StatisticalDefenseFramework, StatisticalClaim
from .physics_consistency_sentinel import PhysicsConsistencySentinel, PhysicalClaim
from .domain_validation_gatekeeper import DomainValidationGatekeeper, DomainClaim
from .epistemic_humility_engine import EpistemicHumilityEngine, UncertaintyQuantification, KnowledgeBoundary, EvidentialStrength
from .peer_review_memory import PeerReviewMemory


@dataclass
class SimulatedReview:
    """Result of peer review simulation"""
    paper_id: str
    timestamp: datetime
    overall_verdict: str  # 'accept', 'minor_revisions', 'major_revisions', 'reject'
    readiness_score: float
    concerns_by_category: Dict[str, List[Dict[str, Any]]]
    strengths: List[str]
    weaknesses: List[str]
    recommended_revisions: List[Dict[str, Any]]
    estimated_referee_response: str


@dataclass
class PreDefenseRebuttal:
    """Result of pre-emptive defense strengthening"""
    original_claims: List[str]
    strengthened_claims: List[str]
    addressed_concerns: List[str]
    remaining_vulnerabilities: List[str]
    confidence_improvement: float


class PeerReviewSimulator:
    """
    Simulates peer review before submission to catch weaknesses early
    and strengthen claims through adversarial testing.
    """

    def __init__(self):
        # Initialize all framework components
        self.causal_framework = CausalValidationFramework()
        self.hypothesis_critic = HypothesisCriticEngine()
        self.statistical_defense = StatisticalDefenseFramework()
        self.physics_sentinel = PhysicsConsistencySentinel()
        self.domain_gatekeeper = DomainValidationGatekeeper()
        self.humility_engine = EpistemicHumilityEngine()
        self.review_memory = PeerReviewMemory()

        # Review criteria weights
        self.category_weights = {
            'causal_rigor': 0.20,
            'statistical_validity': 0.20,
            'physics_consistency': 0.20,
            'domain_appropriateness': 0.15,
            'methodological_rigor': 0.15,
            'uncertainty_quantification': 0.10
        }

    def simulate_review(self,
                       paper_content: Dict[str, Any],
                       domain: str = 'astrophysics') -> SimulatedReview:
        """
        Run comprehensive peer review simulation.

        Addresses peer review: "Review issues before referees see them"
        """
        paper_id = paper_content.get('paper_id', 'unknown')

        # Initialize concerns by category
        concerns_by_category = {
            'causal_rigor': [],
            'statistical_validity': [],
            'physics_consistency': [],
            'domain_appropriateness': [],
            'methodological_rigor': [],
            'uncertainty_quantification': []
        }

        # Analyze claims
        claims = paper_content.get('claims', [])
        methods = paper_content.get('methods', {})
        results = paper_content.get('results', {})

        # 1. Causal Rigor Analysis
        causal_concerns = self._analyze_causal_rigor(claims, methods, results)
        concerns_by_category['causal_rigor'] = causal_concerns

        # 2. Statistical Validity Analysis
        statistical_concerns = self._analyze_statistical_validity(claims, methods, results)
        concerns_by_category['statistical_validity'] = statistical_concerns

        # 3. Physics Consistency Analysis
        physics_concerns = self._analyze_physics_consistency(claims, methods)
        concerns_by_category['physics_consistency'] = physics_concerns

        # 4. Domain Appropriateness Analysis
        domain_concerns = self._analyze_domain_appropriateness(claims, domain)
        concerns_by_category['domain_appropriateness'] = domain_concerns

        # 5. Methodological Rigor Analysis
        method_concerns = self._analyze_methodological_rigor(claims, methods)
        concerns_by_category['methodological_rigor'] = method_concerns

        # 6. Uncertainty Quantification Analysis
        uncertainty_concerns = self._analyze_uncertainty_quantification(claims, results)
        concerns_by_category['uncertainty_quantification'] = uncertainty_concerns

        # Calculate readiness score
        readiness_score = self._calculate_readiness_score(concerns_by_category)

        # Determine overall verdict
        overall_verdict = self._determine_verdict(readiness_score, concerns_by_category)

        # Identify strengths
        strengths = self._identify_strengths(paper_content, concerns_by_category)

        # Identify weaknesses
        weaknesses = self._identify_weaknesses(concerns_by_category)

        # Generate recommended revisions
        recommended_revisions = self._generate_revisions(concerns_by_category)

        # Estimate referee response
        estimated_response = self._estimate_referee_response(
            overall_verdict, concerns_by_category, strengths, weaknesses
        )

        return SimulatedReview(
            paper_id=paper_id,
            timestamp=datetime.now(),
            overall_verdict=overall_verdict,
            readiness_score=readiness_score,
            concerns_by_category=concerns_by_category,
            strengths=strengths,
            weaknesses=weaknesses,
            recommended_revisions=recommended_revisions,
            estimated_response=estimated_response
        )

    def pre_defense_rebuttal(self,
                            original_review: SimulatedReview,
                            paper_content: Dict[str, Any]) -> PreDefenseRebuttal:
        """
        Strengthen claims by pre-emptively addressing reviewer concerns.

        Addresses peer review: "Address issues before submission"
        """
        original_claims = paper_content.get('claims', [])
        strengthened_claims = []
        addressed_concerns = []
        remaining_vulnerabilities = []

        for i, claim in enumerate(original_claims):
            claim_text = claim.get('statement', '')

            # Check which categories have concerns about this claim
            claim_concerns = []
            for category, concerns in original_review.concerns_by_category.items():
                for concern in concerns:
                    if claim_text in concern.get('affected_claim', ''):
                        claim_concerns.append((category, concern))

            if not claim_concerns:
                # No concerns - keep as is
                strengthened_claims.append(claim_text)
                continue

            # Address concerns
            strengthened = claim_text
            confidence_improvement = 0.0

            for category, concern in claim_concerns:
                if category == 'causal_rigor':
                    strengthened = self._strengthen_causal_claim(strengthened, concern)
                    addressed_concerns.append(concern.get('concern', 'causal_issue'))
                    confidence_improvement += 0.1

                elif category == 'statistical_validity':
                    strengthened = self._strengthen_statistical_claim(strengthened, concern)
                    addressed_concerns.append(concern.get('concern', 'statistical_issue'))
                    confidence_improvement += 0.05

                elif category == 'physics_consistency':
                    strengthened = self._strengthen_physics_claim(strengthened, concern)
                    addressed_concerns.append(concern.get('concern', 'physics_issue'))
                    confidence_improvement += 0.15

                elif category == 'uncertainty_quantification':
                    strengthened = self._strengthen_uncertainty_claim(strengthened, concern)
                    addressed_concerns.append(concern.get('concern', 'uncertainty_issue'))
                    confidence_improvement += 0.1

            # Check if all concerns addressed
            all_addressed = confidence_improvement > 0.3

            if all_addressed:
                strengthened_claims.append(strengthened)
            else:
                remaining_vulnerabilities.append(
                    f"Claim '{claim_text[:50]}...'仍有未解决的担忧"
                )
                strengthened_claims.append(strengthened)

        return PreDefenseRebuttal(
            original_claims=[c.get('statement', '') for c in original_claims],
            strengthened_claims=strengthened_claims,
            addressed_concerns=addressed_concerns,
            remaining_vulnerabilities=remaining_vulnerabilities,
            confidence_improvement=0.0  # Would compute actual improvement
        )

    def _analyze_causal_rigor(self,
                             claims: List[Dict],
                             methods: Dict,
                             results: Dict) -> List[Dict[str, Any]]:
        """Analyze causal rigor of claims"""
        concerns = []

        for claim in claims:
            statement = claim.get('statement', '')

            # Check for causal language
            if any(word in statement.lower() for word in ['causes', 'leads to', 'drives', 'determines']):
                # This is a causal claim - check rigor

                # Check 1: Confounders considered?
                if 'confounders' not in claim.get('analysis', {}):
                    concerns.append({
                        'severity': 'major',
                        'concern': 'Causal claim without confounder analysis',
                        'affected_claim': statement,
                        'suggestion': 'Enumerate and test potential confounders'
                    })

                # Check 2: Intervention or counterfactual?
                if 'intervention' not in methods and 'counterfactual' not in methods:
                    concerns.append({
                        'severity': 'major',
                        'concern': 'Causal claim without intervention analysis',
                        'affected_claim': statement,
                        'suggestion': 'Add do-calculus or counterfactual reasoning'
                    })

                # Check 3: Alternative mechanisms?
                if 'alternatives' not in claim:
                    concerns.append({
                        'severity': 'minor',
                        'concern': 'Causal claim without considering alternatives',
                        'affected_claim': statement,
                        'suggestion': 'Generate and test alternative explanations'
                    })

        return concerns

    def _analyze_statistical_validity(self,
                                     claims: List[Dict],
                                     methods: Dict,
                                     results: Dict) -> List[Dict[str, Any]]:
        """Analyze statistical validity"""
        concerns = []

        sample_size = methods.get('sample_size', 0)

        # Check 1: Sample size adequate?
        if sample_size < 30:
            concerns.append({
                'severity': 'major',
                'concern': f'Small sample size (N={sample_size})',
                'affected_claim': 'all_statistical_claims',
                'suggestion': 'Conduct power analysis and report statistical power'
            })

        # Check 2: Power analysis?
        if 'power_analysis' not in methods:
            concerns.append({
                'severity': 'minor',
                'concern': 'No power analysis reported',
                'affected_claim': 'statistical_claims',
                'suggestion': 'Add power analysis to methods section'
            })

        # Check 3: Multiple testing correction?
        n_tests = methods.get('n_tests', 1)
        if n_tests > 3 and 'multiple_testing_correction' not in methods:
            concerns.append({
                'severity': 'major',
                'concern': f'{n_tests} tests performed without multiple testing correction',
                'affected_claim': 'statistical_significance',
                'suggestion': 'Apply FDR or family-wise error rate correction'
            })

        # Check 4: Effect sizes reported?
        if 'effect_sizes' not in results:
            concerns.append({
                'severity': 'minor',
                'concern': 'Effect sizes not reported',
                'affected_claim': 'statistical_results',
                'suggestion': 'Report effect sizes with confidence intervals'
            })

        return concerns

    def _analyze_physics_consistency(self,
                                    claims: List[Dict],
                                    methods: Dict) -> List[Dict[str, Any]]:
        """Analyze physics consistency"""
        concerns = []

        for claim in claims:
            statement = claim.get('statement', '')

            # Check for extreme values
            if 'infinite' in statement.lower() or 'infinity' in statement.lower():
                concerns.append({
                    'severity': 'critical',
                    'concern': 'Infinite value in physical claim',
                    'affected_claim': statement,
                    'suggestion': 'Verify limit behavior and add physical constraints'
                })

            # Check for superluminal claims
            if any(phrase in statement.lower() for phrase in
                   ['faster than light', 'superluminal', 'ftl']):
                concerns.append({
                    'severity': 'critical',
                    'concern': 'Potential violation of relativistic causality',
                    'affected_claim': statement,
                    'suggestion': 'Verify consistency with special relativity'
                })

            # Check dimensional consistency
            if 'equation' in claim:
                # Would implement full dimensional analysis
                pass

        return concerns

    def _analyze_domain_appropriateness(self,
                                       claims: List[Dict],
                                       domain: str) -> List[Dict[str, Any]]:
        """Analyze domain appropriateness"""
        concerns = []

        # Check for proper terminology
        terminology_issues = self.domain_gatekeeper.check_terminology_consistency(
            ' '.join([c.get('statement', '') for c in claims]),
            domain
        )

        for issue in terminology_issues:
            if not issue.correct_usage:
                concerns.append({
                    'severity': 'minor',
                    'concern': f"Terminology issue: '{issue.term}'",
                    'affected_claim': issue.context,
                    'suggestion': f"Use domain convention: {issue.domain_convention}"
                })

        # Check against domain precedents
        for claim in claims:
            precedents = self.domain_gatekeeper.find_historical_precedents(
                DomainClaim(
                    claim=claim.get('statement', ''),
                    domain=domain,
                    terminology=[],
                    theoretical_basis=[],
                    observational_support=[],
                    consistency_checks=[],
                    confidence_by_domain_expert=0.5
                )
            )

            if precedents:
                # Check if current claim contradicts established results
                concerns.append({
                    'severity': 'minor',
                    'concern': f'Claim related to precedent: {precedents[0]["precedent_claim"]}',
                    'affected_claim': claim.get('statement', ''),
                    'suggestion': f'Acknowledge and distinguish from: {precedents[0]["authors"]} ({precedents[0]["year"]})'
                })

        return concerns

    def _analyze_methodological_rigor(self,
                                     claims: List[Dict],
                                     methods: Dict) -> List[Dict[str, Any]]:
        """Analyze methodological rigor"""
        concerns = []

        # Check 1: Reproducibility information
        if 'code_availability' not in methods:
            concerns.append({
                'severity': 'minor',
                'concern': 'Code availability not specified',
                'affected_claim': 'reproducibility',
                'suggestion': 'State whether code is publicly available'
            })

        # Check 2: Data provenance
        if 'data_sources' not in methods:
            concerns.append({
                'severity': 'minor',
                'concern': 'Data sources not fully documented',
                'affected_claim': 'data_provenance',
                'suggestion': 'Document all data sources and versions'
            })

        # Check 3: Alternative methods tested
        if 'alternative_methods' not in methods:
            concerns.append({
                'severity': 'minor',
                'concern': 'Robustness to alternative methods not tested',
                'affected_claim': 'methodological_robustness',
                'suggestion': 'Test robustness to different methodological choices'
            })

        return concerns

    def _analyze_uncertainty_quantification(self,
                                          claims: List[Dict],
                                          results: Dict) -> List[Dict[str, Any]]:
        """Analyze uncertainty quantification"""
        concerns = []

        for claim in claims:
            statement = claim.get('statement', '')

            # Check 1: Uncertainties reported?
            if 'uncertainty' not in claim and 'error' not in statement.lower():
                concerns.append({
                    'severity': 'minor',
                    'concern': 'Uncertainty not quantified',
                    'affected_claim': statement,
                    'suggestion': 'Add uncertainty estimates'
                })

            # Check 2: Confidence intervals?
            if 'confidence_interval' not in claim:
                concerns.append({
                    'severity': 'minor',
                    'concern': 'Confidence interval not reported',
                    'affected_claim': statement,
                    'suggestion': 'Report 95% confidence intervals'
                })

        # Check 3: Uncertainty propagation?
        if 'uncertainty_propagation' not in results:
            concerns.append({
                'severity': 'minor',
                'concern': 'Uncertainty propagation not shown',
                'affected_claim': 'uncertainty_analysis',
                'suggestion': 'Show how uncertainties propagate through analysis'
            })

        return concerns

    def _calculate_readiness_score(self,
                                  concerns_by_category: Dict[str, List[Dict]]) -> float:
        """Calculate overall readiness score"""
        # Start with perfect score
        score = 1.0

        # Deduct for concerns
        for category, concerns in concerns_by_category.items():
            weight = self.category_weights.get(category, 0.1)

            for concern in concerns:
                severity = concern.get('severity', 'minor')

                if severity == 'critical':
                    score -= 0.15 * weight
                elif severity == 'major':
                    score -= 0.10 * weight
                elif severity == 'minor':
                    score -= 0.05 * weight

        return max(0.0, score)

    def _determine_verdict(self,
                          readiness_score: float,
                          concerns_by_category: Dict[str, List[Dict]]) -> str:
        """Determine overall verdict"""
        # Count critical concerns
        n_critical = sum(len([c for c in concerns if c.get('severity') == 'critical'])
                        for concerns in concerns_by_category.values())

        # Count major concerns
        n_major = sum(len([c for c in concerns if c.get('severity') == 'major'])
                     for concerns in concerns_by_category.values())

        if n_critical > 0:
            return 'reject'
        elif readiness_score >= 0.8 and n_major <= 2:
            return 'accept'
        elif readiness_score >= 0.6 and n_major <= 5:
            return 'minor_revisions'
        else:
            return 'major_revisions'

    def _identify_strengths(self,
                           paper_content: Dict[str, Any],
                           concerns_by_category: Dict[str, List[Dict]]) -> List[str]:
        """Identify paper strengths"""
        strengths = []

        # Check for strengths that have no concerns
        if not concerns_by_category.get('causal_rigor'):
            strengths.append('Causal claims are well-supported')

        if not concerns_by_category.get('statistical_validity'):
            strengths.append('Statistical methods are rigorous')

        if not concerns_by_category.get('physics_consistency'):
            strengths.append('Physics consistency is maintained')

        if not concerns_by_category.get('uncertainty_quantification'):
            strengths.append('Uncertainties are properly quantified')

        # Check for positive attributes
        methods = paper_content.get('methods', {})
        if methods.get('sample_size', 0) > 100:
            strengths.append('Large sample size')

        if methods.get('replication_studies', 0) > 0:
            strengths.append('Includes replication analysis')

        if 'code_availability' in methods:
            strengths.append('Code is publicly available')

        return strengths

    def _identify_weaknesses(self,
                            concerns_by_category: Dict[str, List[Dict]]) -> List[str]:
        """Identify paper weaknesses"""
        weaknesses = []

        # Group concerns by type
        for category, concerns in concerns_by_category.items():
            if concerns:
                # Find most frequent concern type
                concern_types = {}
                for concern in concerns:
                    concern_text = concern.get('concern', 'unknown')
                    concern_types[concern_text] = concern_types.get(concern_text, 0) + 1

                # Get most common concern
                most_common = max(concern_types, key=concern_types.get)
                weaknesses.append(most_common)

        return weaknesses

    def _generate_revisions(self,
                          concerns_by_category: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """Generate recommended revisions"""
        revisions = []

        # Group concerns by category for systematic revision
        for category, concerns in concerns_by_category.items():
            if not concerns:
                continue

            # Create revision group
            revision = {
                'category': category,
                'priority': self._get_revision_priority(concerns),
                'concerns': [c.get('concern') for c in concerns],
                'actions': [c.get('suggestion') for c in concerns]
            }

            revisions.append(revision)

        # Sort by priority
        priority_order = {'critical': 0, 'major': 1, 'minor': 2}
        revisions.sort(key=lambda r: priority_order.get(r['priority'], 3))

        return revisions

    def _get_revision_priority(self, concerns: List[Dict]) -> str:
        """Get priority level for revision"""
        # Check for any critical concerns
        if any(c.get('severity') == 'critical' for c in concerns):
            return 'critical'
        # Check for major concerns
        if any(c.get('severity') == 'major' for c in concerns):
            return 'major'
        return 'minor'

    def _estimate_referee_response(self,
                                  verdict: str,
                                  concerns_by_category: Dict[str, List[Dict]],
                                  strengths: List[str],
                                  weaknesses: List[str]) -> str:
        """Generate estimated referee response"""
        if verdict == 'accept':
            response = "Referee likely to accept with minor comments. "
            response += f"Strengths: {', '.join(strengths[:3])}. "
            response += "Minor revisions may be requested for clarity."
        elif verdict == 'minor_revisions':
            response = "Referee likely to request minor revisions. "
            response += f"Address: {', '.join(weaknesses[:2])}. "
            response += "Core contributions appear sound."
        elif verdict == 'major_revisions':
            response = "Referee likely to request major revisions. "
            n_concerns = sum(len(c) for c in concerns_by_category.values())
            response += f"{n_concerns} concerns need addressing. "
            response += f"Focus on: {', '.join(weaknesses[:3])}."
        else:  # reject
            response = "Referee likely to recommend rejection. "
            response += "Critical issues must be resolved. "
            n_critical = sum(len([c for c in concerns if c.get('severity') == 'critical'])
                           for concerns in concerns_by_category.values())
            response += f"{n_critical} critical concerns identified."

        return response

    def _strengthen_causal_claim(self, claim: str, concern: Dict) -> str:
        """Strengthen causal claim by adding qualifiers"""
        if 'confounder' in concern.get('suggestion', '').lower():
            return claim + " (after accounting for measured confounders)"
        elif 'intervention' in concern.get('suggestion', '').lower():
            return claim + " (based on observational analysis; causal interpretation requires further testing)"
        elif 'alternative' in concern.get('suggestion', '').lower():
            return claim + " (while alternative mechanisms cannot be ruled out)"
        return claim

    def _strengthen_statistical_claim(self, claim: str, concern: Dict) -> str:
        """Strengthen statistical claim by adding statistical details"""
        if 'sample size' in concern.get('concern', '').lower():
            return claim + " (with current sample size limitations)"
        elif 'power' in concern.get('concern', '').lower():
            return claim + " (statistical power should be verified)"
        return claim

    def _strengthen_physics_claim(self, claim: str, concern: Dict) -> str:
        """Strengthen physics claim by adding constraints"""
        if 'limit' in concern.get('suggestion', '').lower():
            return claim + " (within valid physical regime)"
        elif 'relativity' in concern.get('concern', '').lower():
            return claim + " (assuming sub-relativistic velocities)"
        return claim

    def _strengthen_uncertainty_claim(self, claim: str, concern: Dict) -> str:
        """Strengthen claim by adding uncertainty"""
        return claim + " (± uncertainty)"


def create_peer_review_simulator() -> PeerReviewSimulator:
    """Factory function for PeerReviewSimulator"""
    return PeerReviewSimulator()
