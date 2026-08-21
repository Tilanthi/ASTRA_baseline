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
Domain Validation Gatekeeper

Validates claims against expert domain knowledge, ensures terminology consistency,
finds historical precedents, and encodes domain-specific heuristics.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import re
import json
from pathlib import Path


@dataclass
class DomainClaim:
    """A claim with domain-specific metadata"""
    claim: str
    domain: str
    terminology: List[str]
    theoretical_basis: List[str]
    observational_support: List[str]
    consistency_checks: List[str]
    confidence_by_domain_expert: float


@dataclass
class ValidationWarning:
    """A domain-specific validation warning"""
    warning_type: str
    severity: str  # 'critical', 'major', 'minor'
    description: str
    affected_claim: str
    domain_rule: str
    suggested_correction: str


@dataclass
class TerminologyCheck:
    """Result of terminology consistency check"""
    term: str
    correct_usage: bool
    context: str
    suggested_alternative: Optional[str]
    domain_convention: str


class DomainValidationGatekeeper:
    """
    Domain-specific validation that addresses peer review concerns about
    expert knowledge, terminology, and domain consistency.
    """

    def __init__(self):
        # Load domain knowledge bases
        self.domain_rules = self._load_domain_rules()
        self.terminology_database = self._load_terminology()
        self.historical_precedents = self._load_precedents()
        self.heuristic_rules = self._load_heuristics()

    def validate_against_expert_knowledge(self,
                                        claim: DomainClaim) -> List[ValidationWarning]:
        """
        Check claims against established domain knowledge.

        Addresses peer review: "This contradicts known results in X"
        """
        warnings = []

        # Check against domain rules
        for rule in self.domain_rules.get(claim.domain, []):
            if self._violates_rule(claim.claim, rule):
                warnings.append(ValidationWarning(
                    warning_type='domain_rule_violation',
                    severity='major',
                    description=f"Claim may violate domain rule: {rule['description']}",
                    affected_claim=claim.claim,
                    domain_rule=rule['name'],
                    suggested_correction=rule['correction']
                ))

        # Check theoretical basis
        if claim.theoretical_basis:
            for theory in claim.theoretical_basis:
                if not self._is_theory_valid(theory, claim.domain):
                    warnings.append(ValidationWarning(
                        warning_type='invalid_theory',
                        severity='major',
                        description=f"Theory '{theory}' may not be applicable to {claim.domain}",
                        affected_claim=claim.claim,
                        domain_rule='theoretical_validity',
                        suggested_correction='Verify theory applicability'
                    ))

        # Check observational support
        if not claim.observational_support:
            warnings.append(ValidationWarning(
                warning_type='insufficient_observation',
                severity='minor',
                description='Claim lacks observational support',
                affected_claim=claim.claim,
                domain_rule='observational_validation',
                suggested_correction='Add supporting observational evidence'
            ))

        return warnings

    def check_terminology_consistency(self,
                                     text: str,
                                     domain: str) -> List[TerminologyCheck]:
        """
        Ensure proper use of field-specific terms.

        Addresses peer review: "Inconsistent use of terminology"
        """
        checks = []

        # Get domain-specific terminology
        domain_terms = self.terminology_database.get(domain, {})

        # Check each term
        for term, definition in domain_terms.items():
            # Find term usage in text
            pattern = r'\b' + re.escape(term) + r'\b'
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                context = self._get_context(text, match.start(), match.end())

                # Check if usage is correct
                correct = self._check_term_usage(term, context, domain)

                checks.append(TerminologyCheck(
                    term=term,
                    correct_usage=correct,
                    context=context,
                    suggested_alternative=definition.get('alternative') if not correct else None,
                    domain_convention=definition.get('convention', 'standard')
                ))

        return checks

    def find_historical_precedents(self,
                                  claim: DomainClaim) -> List[Dict[str, Any]]:
        """
        Find analogous cases in domain history.

        Addresses peer review: "Has this been studied before?"
        """
        precedents = []

        domain_history = self.historical_precedents.get(claim.domain, [])

        # Search for similar claims
        for precedent in domain_history:
            similarity = self._compute_claim_similarity(claim.claim, precedent['claim'])

            if similarity > 0.5:  # Threshold for similarity
                precedents.append({
                    'precedent_claim': precedent['claim'],
                    'similarity': similarity,
                    'year': precedent['year'],
                    'authors': precedent['authors'],
                    'outcome': precedent['outcome'],
                    'relevance': self._assess_relevance(claim, precedent)
                })

        # Sort by relevance
        precedents.sort(key=lambda x: x['relevance'], reverse=True)

        return precedents[:5]  # Return top 5

    def apply_domain_heuristics(self,
                               claim: DomainClaim,
                               analysis: Dict[str, Any]) -> List[str]:
        """
        Encode expert "rules of thumb" for the domain.

        Addresses peer review: "Domain experts would know that..."
        """
        heuristic_feedback = []

        domain_heuristics = self.heuristic_rules.get(claim.domain, [])

        for heuristic in domain_heuristics:
            if self._triggers_heuristic(claim, analysis, heuristic):
                heuristic_feedback.append(
                    f"Heuristic: {heuristic['rule']}\n"
                    f"Reason: {heuristic['explanation']}\n"
                    f"Action: {heuristic['action']}"
                )

        return heuristic_feedback

    def validate_cross_domain_consistency(self,
                                         claims: List[DomainClaim]) -> List[ValidationWarning]:
        """
        Check consistency across different astrophysical domains.

        Addresses peer review: "Claims in Section X contradict Section Y"
        """
        warnings = []

        # Group claims by physical quantities
        quantity_claims = self._group_by_quantity(claims)

        # Check for contradictions
        for quantity, claim_list in quantity_claims.items():
            if len(claim_list) > 1:
                # Multiple claims about same quantity - check consistency
                for i, claim1 in enumerate(claim_list):
                    for claim2 in claim_list[i+1:]:
                        if self._claims_contradict(claim1, claim2):
                            warnings.append(ValidationWarning(
                                warning_type='cross_domain_contradiction',
                                severity='major',
                                description=f"Claims about {quantity} contradict: "
                                          f"'{claim1.claim}' vs '{claim2.claim}'",
                                affected_claim=f"{claim1.claim} | {claim2.claim}",
                                domain_rule='cross_domain_consistency',
                                suggested_correction='Reconcile contradictory claims'
                            ))

        return warnings

    def _load_domain_rules(self) -> Dict[str, List[Dict]]:
        """Load domain-specific rules"""
        return {
            'astrophysics': [
                {
                    'name': 'energy_conservation',
                    'description': 'Energy must be conserved in all processes',
                    'correction': 'Add energy source/sink terms'
                },
                {
                    'name': 'causality',
                    'description': 'No effect can precede its cause',
                    'correction': 'Check temporal ordering'
                },
                {
                    'name': 'luminosity_limit',
                    'description': 'Eddington luminosity limit for spherical accretion',
                    'correction': 'Check against Eddington limit'
                },
                {
                    'name': 'Jeans_mass',
                    'description': 'Minimum mass for gravitational collapse',
                    'correction': 'Verify mass exceeds Jeans mass'
                }
            ],
            'cosmology': [
                {
                    'name': 'causality_horizon',
                    'description': 'No causal contact outside particle horizon',
                    'correction': 'Check comoving distances'
                },
                {
                    'name': 'Lambda_constraint',
                    'description': 'Dark energy density constrained by observations',
                    'correction': 'Use Planck values for Lambda'
                }
            ],
            'ism': [
                {
                    'name': 'virial_theorem',
                    'description': 'Virial equilibrium requires 2K + U = 0',
                    'correction': 'Check virial balance'
                },
                {
                    'name': 'sonic_scale',
                    'description': 'Turbulence transitions at sonic scale',
                    'correction': 'Consider sonic scale effects'
                }
            ]
        }

    def _load_terminology(self) -> Dict[str, Dict[str, Dict]]:
        """Load domain-specific terminology database"""
        return {
            'astrophysics': {
                'luminosity': {
                    'units': 'erg/s',
                    'convention': 'log scale often used',
                    'alternative': 'flux (erg/s/cm^2) for observed values'
                },
                'flux': {
                    'units': 'erg/s/cm^2',
                    'convention': 'observed value, distance-dependent',
                    'alternative': 'luminosity for intrinsic values'
                },
                'metallicity': {
                    'units': 'dimensionless (relative to solar)',
                    'convention': '[Fe/H] = log10(Z/Z_sun) or Z/Z_sun',
                    'alternative': 'Specify definition used'
                },
                'optical_depth': {
                    'units': 'dimensionless',
                    'convention': 'tau > 1 = optically thick',
                    'alternative': 'optical depth (tau) vs optical depth (A_V)'
                },
                'column_density': {
                    'units': 'cm^-2',
                    'convention': 'N = integral(n)dl',
                    'alternative': 'Specify whether along line of sight or projected'
                }
            },
            'ism': {
                'filament_width': {
                    'units': 'pc',
                    'convention': 'FWHM of density profile',
                    'alternative': 'Specify measurement method'
                },
                'Mach_number': {
                    'units': 'dimensionless',
                    'convention': 'M = sigma_sound / c_s',
                    'alternative': 'Specify whether using 1D or 3D velocity dispersion'
                },
                'plasma_beta': {
                    'units': 'dimensionless',
                    'convention': 'beta = P_gas / P_mag',
                    'alternative': 'Specify if using thermal or total pressure'
                }
            }
        }

    def _load_precedents(self) -> Dict[str, List[Dict]]:
        """Load historical precedents database"""
        return {
            'ism': [
                {
                    'claim': 'Filament width is ~0.1 pc across environments',
                    'year': 2011,
                    'authors': 'Arzoumanian et al.',
                    'outcome': 'Confirmed in Herschel Gould Belt survey',
                    'similarity_keywords': ['filament', 'width', 'universal']
                },
                {
                    'claim': 'Turbulence regulates star formation',
                    'year': 2018,
                    'authors': 'Multiple',
                    'outcome': 'Complex relationship with magnetic fields',
                    'similarity_keywords': ['turbulence', 'star formation', 'ISM']
                }
            ],
            'cosmology': [
                {
                    'claim': 'Universe acceleration discovered',
                    'year': 1998,
                    'authors': 'Riess et al., Perlmutter et al.',
                    'outcome': 'Nobel Prize, standard Lambda-CDM model',
                    'similarity_keywords': ['acceleration', 'expansion', 'dark energy']
                }
            ]
        }

    def _load_heuristics(self) -> Dict[str, List[Dict]]:
        """Load domain heuristic rules"""
        return {
            'astrophysics': [
                {
                    'rule': 'Always report uncertainties',
                    'explanation': 'Measurements have errors - conclusions must account for them',
                    'action': 'Add error bars and discuss uncertainty impact'
                },
                {
                    'rule': 'Check order-of-magnitude first',
                    'explanation': 'Most errors are caught by quick sanity checks',
                    'action': 'Verify result is physically reasonable before detailed analysis'
                },
                {
                    'rule': 'Consider selection effects',
                    'explanation': 'Observational biases can create spurious correlations',
                    'action': 'Model detection limits and selection function'
                }
            ],
            'ism': [
                {
                    'rule': 'Magnetic fields complicate simple collapse',
                    'explanation': 'Ambipolar diffusion and support are often important',
                    'action': 'Consider magnetic pressure and ambipolar diffusion timescales'
                },
                {
                    'rule': 'Turbulence is multi-scale',
                    'explanation': 'Single-scale models miss important physics',
                    'action': 'Consider power spectrum and injection scales'
                }
            ]
        }

    def _violates_rule(self, claim: str, rule: Dict) -> bool:
        """Check if claim violates a domain rule"""
        # Simplified - would use NLP in production
        rule_keywords = rule['name'].split('_')
        claim_lower = claim.lower()

        # If claim mentions rule topic, check if it properly addresses it
        if any(keyword in claim_lower for keyword in rule_keywords):
            # Check if claim has proper handling
            if 'conservation' in rule['name']:
                return 'conserved' not in claim_lower and 'balance' not in claim_lower
            elif 'limit' in rule['name']:
                return 'limit' not in claim_lower and 'constrained' not in claim_lower

        return False

    def _is_theory_valid(self, theory: str, domain: str) -> bool:
        """Check if theory is valid for domain"""
        # Simplified - would use theory database in production
        valid_theories = {
            'astrophysics': ['radiative_transfer', 'hydrodynamics', 'mhd',
                           'nucleosynthesis', 'stellar_evolution'],
            'cosmology': ['general_relativity', 'inflation', 'lambda_cdm'],
            'ism': ['mhd_turbulence', 'jeans_instability', 'ambipolar_diffusion']
        }

        domain_theories = valid_theories.get(domain, [])
        return any(theory_concept in theory.lower()
                  for theory_concept in domain_theories)

    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get context around a term"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()

    def _check_term_usage(self, term: str, context: str, domain: str) -> bool:
        """Check if term is used correctly in context"""
        # Simplified - would use NLP in production
        domain_terms = self.terminology_database.get(domain, {})

        if term in domain_terms:
            term_info = domain_terms[term]

            # Check for common misuse patterns
            if 'alternative' in term_info:
                # Check if alternative should have been used instead
                alternative = term_info['alternative']

                # Example: using "luminosity" when "flux" is appropriate
                if 'observed' in context.lower() and term == 'luminosity':
                    return False
                if 'distance' in context.lower() and term == 'luminosity':
                    return False

        return True

    def _compute_claim_similarity(self, claim1: str, claim2: str) -> float:
        """Compute similarity between two claims"""
        # Simplified - would use embeddings in production
        words1 = set(claim1.lower().split())
        words2 = set(claim2.lower().split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _assess_relevance(self, claim: DomainClaim, precedent: Dict) -> float:
        """Assess relevance of precedent to current claim"""
        # Combine similarity with other factors
        base_similarity = self._compute_claim_similarity(claim.claim, precedent['precedent_claim'])

        # Boost if domains match
        if claim.domain == precedent.get('domain', ''):
            base_similarity *= 1.2

        # Boost if recent (last 10 years)
        if precedent.get('year', 0) >= 2014:
            base_similarity *= 1.1

        return min(base_similarity, 1.0)

    def _group_by_quantity(self, claims: List[DomainClaim]) -> Dict[str, List[DomainClaim]]:
        """Group claims by physical quantities they discuss"""
        # Simplified - would use NER in production
        quantity_keywords = {
            'mass': ['mass', 'masses'],
            'temperature': ['temperature', 'temperatures', 't_eff'],
            'luminosity': ['luminosity', 'brightness'],
            'density': ['density', 'densities'],
            'velocity': ['velocity', 'velocities', 'speed'],
            'magnetic_field': ['magnetic', 'b_field', 'b-field']
        }

        grouped = {}

        for claim in claims:
            claim_lower = claim.claim.lower()

            for quantity, keywords in quantity_keywords.items():
                if any(keyword in claim_lower for keyword in keywords):
                    if quantity not in grouped:
                        grouped[quantity] = []
                    grouped[quantity].append(claim)
                    break

        return grouped

    def _triggers_heuristic(self, claim: DomainClaim, analysis: Dict[str, Any],
                          heuristic: Dict) -> bool:
        """Check if claim triggers a heuristic rule"""
        # Simplified - would use rule engine in production
        rule = heuristic['rule'].lower()

        if 'uncertainty' in rule:
            return 'uncertainty' not in claim.claim.lower() and \
                   'error' not in claim.claim.lower()
        elif 'selection' in rule:
            return 'selection' not in claim.claim.lower() and \
                   'bias' not in claim.claim.lower()
        elif 'magnetic' in rule:
            return 'magnetic' not in claim.claim.lower()

        return False

    def _claims_contradict(self, claim1: DomainClaim, claim2: DomainClaim) -> bool:
        """Check if two claims contradict each other"""
        # Simplified - would use NLP in production
        # Look for explicit contradiction indicators
        text1 = claim1.claim.lower()
        text2 = claim2.claim.lower()

        # Check for opposite assertions
        contradiction_pairs = [
            ('increases', 'decreases'),
            ('positive', 'negative'),
            ('correlation', 'anti-correlation'),
            ('greater than', 'less than'),
            ('hotter', 'cooler')
        ]

        for pos, neg in contradiction_pairs:
            if pos in text1 and neg in text2:
                return True
            if neg in text1 and pos in text2:
                return True

        return False


def create_domain_validation_gatekeeper() -> DomainValidationGatekeeper:
    """Factory function for DomainValidationGatekeeper"""
    return DomainValidationGatekeeper()
