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
Hypothesis Critic Engine

Generates adversarial hypotheses, implements falsification testing,
and simulates referee perspective to strengthen claims before submission.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass
class Hypothesis:
    """A scientific hypothesis with metadata"""
    statement: str
    prediction: str
    evidence: List[str]
    confidence: float
    alternatives: List[str]
    falsification_tests: List[Dict[str, Any]]
    referee_concerns: List[str]


@dataclass
class CritiqueResult:
    """Result of hypothesis criticism"""
    original_hypothesis: Hypothesis
    adversarial_alternatives: List[Hypothesis]
    falsification_plan: List[Dict[str, Any]]
    referee_concerns: List[str]
    strength_score: float
    recommended_revisions: List[str]


class HypothesisCriticEngine:
    """
    Adversarial hypothesis generation and criticism that addresses
    peer review concerns about alternative explanations and robustness.
    """

    def __init__(self):
        self.critique_dimensions = [
            "alternative_mechanisms",
            "measurement_bias",
            "selection_effects",
            "sample_size",
            "domain_applicability",
            "theoretical_consistency",
            "reproducibility",
            "novelty_vs_conservatism"
        ]

        self.question_templates = {
            "alternative_mechanisms": [
                "What other physical processes could produce {observation}?",
                "Could {effect} be explained by established theory X?",
                "Have you ruled out Y as the primary driver?"
            ],
            "measurement_bias": [
                "How do measurement uncertainties affect {claim}?",
                "Could systematic errors explain the observed pattern?",
                "What calibration procedures were used?"
            ],
            "selection_effects": [
                "How might selection bias affect your sample?",
                "Are detection limits influencing {pattern}?",
                "Would the relationship hold for a flux-limited sample?"
            ],
            "sample_size": [
                "Is N={n} sufficient to support this claim?",
                "What is the statistical power of your test?",
                "How would larger samples affect your conclusions?"
            ],
            "domain_applicability": [
                "Does this apply beyond {specific_case}?",
                "Are there boundary conditions where this fails?",
                "Is this a universal or local phenomenon?"
            ],
            "theoretical_consistency": [
                "How does this align with {established_theory}?",
                "Does this violate any known physical principles?",
                "What theoretical framework supports this?"
            ],
            "reproducibility": [
                "Could another group reproduce this analysis?",
                "Are all data sources and methods documented?",
                "What information is needed for independent verification?"
            ],
            "novelty_vs_conservatism": [
                "Is this truly novel or consistent with prior work?",
                "How does this build on or challenge existing theory?",
                "Are you claiming too much from limited evidence?"
            ]
        }

    def generate_adversarial_hypotheses(self,
                                       hypothesis: Hypothesis,
                                       context: Dict[str, Any],
                                       n_alternatives: int = 5) -> List[Hypothesis]:
        """
        Generate competing explanations to test robustness of original.

        Addresses peer review: "Alternative mechanisms not considered"
        """
        alternatives = []

        # Category 1: Established theory alternatives
        alt_1 = Hypothesis(
            statement=f"Established physics explains {hypothesis.statement}",
            prediction=f"Standard models predict {hypothesis.prediction}",
            evidence=["established_theory", "literature"],
            confidence=0.7,
            alternatives=[],
            falsification_tests=[],
            referee_concerns=[]
        )
        alternatives.append(alt_1)

        # Category 2: Measurement artifact
        alt_2 = Hypothesis(
            statement=f"Measurement/systematic effects produce observed pattern",
            prediction=f"Pattern disappears with improved calibration",
            evidence=["systematic_error_literature"],
            confidence=0.5,
            alternatives=[],
            falsification_tests=[],
            referee_concerns=[]
        )
        alternatives.append(alt_2)

        # Category 3: Selection bias
        alt_3 = Hypothesis(
            statement=f"Observational selection creates apparent {hypothesis.statement}",
            prediction=f"Pattern strength correlates with selection function",
            evidence=["selection_bias_theory"],
            confidence=0.6,
            alternatives=[],
            falsification_tests=[],
            referee_concerns=[]
        )
        alternatives.append(alt_3)

        # Category 4: Chance/Fluke
        alt_4 = Hypothesis(
            statement=f"Observed pattern is statistical fluctuation",
            prediction=f"Pattern does not repeat in independent samples",
            evidence=["null_hypothesis"],
            confidence=0.3,
            alternatives=[],
            falsification_tests=[],
            referee_concerns=[]
        )
        alternatives.append(alt_4)

        # Category 5: Domain-specific alternative
        if context.get('domain') == 'astrophysics':
            alt_5 = Hypothesis(
                statement=f"Alternative astrophysical mechanism: feedback/heating",
                prediction=f"Pattern correlates with star formation activity",
                evidence=["feedback_literature"],
                confidence=0.6,
                alternatives=[],
                falsification_tests=[],
                referee_concerns=[]
            )
            alternatives.append(alt_5)

        return alternatives[:n_alternatives]

    def design_falsification_tests(self,
                                   hypothesis: Hypothesis,
                                   alternatives: List[Hypothesis],
                                   context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Design experiments to disprove each hypothesis.

        Addresses peer review: "How would you test if this is wrong?"
        """
        tests = []

        # Test 1: Critical prediction test
        tests.append({
            'name': 'critical_prediction_test',
            'hypothesis_to_falsify': hypothesis.statement,
            'method': 'Measure key prediction with independent dataset',
            'outcome_if_false': 'Prediction fails outside uncertainty',
            'outcome_if_true': 'Prediction confirmed within errors',
            'feasibility': 'high',
            'required_data': context.get('required_data', ['independent_observation']),
            'estimated_cost': 'moderate'
        })

        # Test 2: Cross-domain validation
        tests.append({
            'name': 'cross_domain_test',
            'hypothesis_to_falsify': hypothesis.statement,
            'method': f'Test in different {context.get("domain", "astrophysical")} regime',
            'outcome_if_false': 'Effect does not generalize',
            'outcome_if_true': 'Effect robust across regimes',
            'feasibility': 'medium',
            'required_data': ['multi_domain_sample'],
            'estimated_cost': 'high'
        })

        # Test 3: Alternative mechanism test
        for i, alt in enumerate(alternatives):
            tests.append({
                'name': f'alternative_mechanism_test_{i}',
                'hypothesis_to_falsify': alt.statement,
                'method': f'Test prediction unique to alternative: {alt.prediction}',
                'outcome_if_false': 'Alternative prediction fails',
                'outcome_if_true': 'Alternative supported',
                'feasibility': 'variable',
                'required_data': ['discriminating_measurement'],
                'estimated_cost': 'moderate'
            })

        # Test 4: Sample size sensitivity
        tests.append({
            'name': 'sample_size_sensitivity',
            'hypothesis_to_falsify': hypothesis.statement,
            'method': 'Vary sample size to test if effect is robust',
            'outcome_if_false': 'Effect disappears with larger sample',
            'outcome_if_true': 'Effect stable across sample sizes',
            'feasibility': 'high',
            'required_data': ['larger_sample', 'bootstrap_resamples'],
            'estimated_cost': 'low'
        })

        # Test 5: Systematic error test
        tests.append({
            'name': 'systematic_error_test',
            'hypothesis_to_falsify': f'{hypothesis.statement} vs measurement artifact',
            'method': 'Apply different calibration/analysis methods',
            'outcome_if_false': 'Effect depends on analysis choice',
            'outcome_if_true': 'Effect robust to analysis changes',
            'feasibility': 'high',
            'required_data': ['alternative_calibration'],
            'estimated_cost': 'low'
        })

        return tests

    def simulate_referee_perspective(self,
                                    hypothesis: Hypothesis,
                                    context: Dict[str, Any]) -> List[str]:
        """
        Ask "what would a critic attack?" to pre-empt referee concerns.

        Addresses peer review: "Reviewer will likely question X"
        """
        concerns = []

        # Generate concerns from each critique dimension
        for dimension in self.critique_dimensions:
            if dimension in self.question_templates:
                # Select relevant template
                templates = self.question_templates[dimension]
                # Customize template for this hypothesis
                for template in templates:
                    concern = self._customize_template(template, hypothesis, context)
                    concerns.append(concern)

        # Add domain-specific concerns
        if context.get('domain') == 'astrophysics':
            concerns.extend(self._astrophysics_specific_concerns(hypothesis, context))

        return concerns

    def assess_hypothesis_strength(self,
                                  hypothesis: Hypothesis,
                                  alternatives: List[Hypothesis],
                                  falsification_tests: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Score hypothesis strength based on survivability through criticism.
        """
        scores = {}

        # Score 1: Uniqueness vs alternatives
        if alternatives:
            originality_scores = [
                1.0 - alt.confidence for alt in alternatives
            ]
            scores['uniqueness'] = np.mean(originality_scores)
        else:
            scores['uniqueness'] = 0.5  # Neutral if no alternatives

        # Score 2: Testability
        feasible_tests = [t for t in falsification_tests
                         if t.get('feasibility') in ['high', 'medium']]
        scores['testability'] = len(feasible_tests) / max(len(falsification_tests), 1)

        # Score 3: Evidence quality
        if hypothesis.evidence:
            scores['evidence_quality'] = min(len(hypothesis.evidence) / 3, 1.0)
        else:
            scores['evidence_quality'] = 0.0

        # Score 4: Confidence calibration
        scores['confidence_calibration'] = 1.0 - abs(hypothesis.confidence - scores['evidence_quality'])

        # Score 5: Domain appropriateness
        scores['domain_fit'] = 0.7  # Would be computed based on domain knowledge

        # Aggregate strength score
        scores['aggregate'] = np.mean(list(scores.values()))

        return scores

    def generate_revised_hypothesis(self,
                                   original: Hypothesis,
                                   critique: CritiqueResult) -> Hypothesis:
        """
        Strengthen hypothesis by addressing referee concerns.
        """
        # Address top concerns
        addressed_concerns = critique.referee_concerns[:3]

        # Revise statement to be more precise
        revised_statement = original.statement
        for concern in addressed_concerns:
            if 'measurement' in concern.lower():
                revised_statement += " (after accounting for measurement uncertainties)"
            elif 'alternative' in concern.lower():
                revised_statement += " (distinguishing from alternative mechanisms)"
            elif 'sample' in concern.lower():
                revised_statement += " (with current sample size limitations noted)"

        # Add falsification tests to evidence
        enhanced_evidence = original.evidence.copy()
        for test in critique.falsification_plan[:2]:
            enhanced_evidence.append(f"Test: {test['name']}")

        # Adjust confidence based on strength score
        revised_confidence = original.confidence * critique.strength_score

        # Create revised hypothesis
        revised = Hypothesis(
            statement=revised_statement,
            prediction=original.prediction,
            evidence=enhanced_evidence,
            confidence=revised_confidence,
            alternatives=[alt.statement for alt in critique.adversarial_alternatives],
            falsification_tests=critique.falsification_plan,
            referee_concerns=addressed_concerns
        )

        return revised

    def _customize_template(self, template: str, hypothesis: Hypothesis,
                          context: Dict[str, Any]) -> str:
        """Customize question template for specific hypothesis"""
        # Replace placeholders
        custom = template.replace("{observation}", hypothesis.statement)
        custom = custom.replace("{effect}", hypothesis.prediction)
        custom = custom.replace("{claim}", hypothesis.statement)
        custom = custom.replace("{pattern}", hypothesis.statement)
        custom = custom.replace("{specific_case}", context.get('specific_case', 'this case'))
        custom = custom.replace("{established_theory}",
                              context.get('established_theory', 'standard theory'))
        custom = custom.replace("{n}", str(context.get('sample_size', 'N')))

        return custom

    def _astrophysics_specific_concerns(self, hypothesis: Hypothesis,
                                      context: Dict[str, Any]) -> List[str]:
        """Generate astrophysics-domain specific concerns"""
        concerns = []

        # Redshift/cosmological concerns
        if 'redshift' in hypothesis.statement.lower() or 'distance' in hypothesis.statement.lower():
            concerns.append("How do cosmological effects influence your measurements?")
            concerns.append("Have you accounted for evolution with redshift?")

        # Instrumental concerns
        concerns.append("How do telescope/instrument characteristics affect results?")
        concerns.append("Are there wavelength-dependent selection effects?")

        # Physical mechanism concerns
        concerns.append("What physical mechanism drives this relationship?")
        concerns.append("Is this consistent with timescales for physical processes?")

        # Sample concerns
        concerns.append("Is your sample representative of the population?")
        concerns.append("How do flux limits affect your detected objects?")

        return concerns


def create_hypothesis_critic_engine() -> HypothesisCriticEngine:
    """Factory function for HypothesisCriticEngine"""
    return HypothesisCriticEngine()
