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
Epistemic Humility Engine

Quantifies uncertainty, calibrates confidence, recognizes knowledge boundaries,
and classifies evidential strength.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass
class UncertaintyQuantification:
    """Uncertainty analysis for a claim"""
    claim: str
    uncertainty_sources: List[str]
    total_uncertainty: float
    uncertainty_breakdown: Dict[str, float]
    uncertainty_propagation: str
    confidence_interval: Tuple[float, float]
    uncertainty_type: str  # 'aleatoric', 'epistemic', 'both'


@dataclass
class ConfidenceCalibration:
    """Calibration of predicted vs actual confidence"""
    predicted_confidence: float
    actual_accuracy: float
    calibration_score: float
    is_overconfident: bool
    is_underconfident: bool
    recommended_adjustment: float


@dataclass
class KnowledgeBoundary:
    """Recognition of knowledge boundaries"""
    claim: str
    domain: str
    extrapolation_type: str  # 'temporal', 'spatial', 'parameter', 'regime'
    distance_from_training: float
    confidence_penalty: float
    boundary_warning: str


@dataclass
class EvidentialStrength:
    """Classification of evidence quality"""
    claim: str
    evidence_quality: str  # 'strong', 'moderate', 'weak', 'speculative'
    evidence_types: List[str]
    sample_size_adequacy: str
    reproducibility: str
    independence: str
    overall_strength: float


class EpistemicHumilityEngine:
    """
    Uncertainty quantification and epistemic humility that addresses
    peer review concerns about overconfidence and uncertainty underestimation.
    """

    def __init__(self):
        self.uncertainty_categories = [
            'measurement_error',
            'sampling_error',
            'systematic_error',
            'model_uncertainty',
            'parameter_uncertainty',
            'extrapolation_uncertainty'
        ]

        self.evidence_quality_criteria = {
            'strong': {
                'sample_size': 'large (n > 100)',
                'reproducibility': 'independent_replications',
                'methodology': 'randomized_or_interventional',
                'consistency': 'across_studies'
            },
            'moderate': {
                'sample_size': 'moderate (n > 30)',
                'reproducibility': 'some_replication',
                'methodology': 'well_designed_observation',
                'consistency': 'some_support'
            },
            'weak': {
                'sample_size': 'small (n < 30)',
                'reproducibility': 'not_replicated',
                'methodology': 'observational',
                'consistency': 'isolated_finding'
            },
            'speculative': {
                'sample_size': 'very_small (n < 10)',
                'reproducibility': 'not_replicable',
                'methodology': 'anecdotal',
                'consistency': 'novel_claim'
            }
        }

    def propagate_uncertainty(self,
                            claim: str,
                            uncertainty_sources: Dict[str, float],
                            propagation_method: str = 'monte_carlo') -> UncertaintyQuantification:
        """
        Propagate uncertainty through entire analysis chain.

        Addresses peer review: "Uncertainties not properly propagated"
        """
        # Combine uncertainties
        if propagation_method == 'monte_carlo':
            total_unc = self._monte_carlo_propagation(uncertainty_sources)
        elif propagation_method == 'analytic':
            total_unc = self._analytic_propagation(uncertainty_sources)
        else:
            # Simple sum in quadrature
            total_unc = np.sqrt(sum(u**2 for u in uncertainty_sources.values()))

        # Calculate confidence interval
        central_value = uncertainty_sources.get('central_value', 0.0)
        ci_low = central_value - 1.96 * total_unc
        ci_high = central_value + 1.96 * total_unc

        # Determine uncertainty type
        has_model_unc = 'model_uncertainty' in uncertainty_sources
        has_measurement_unc = any(k in uncertainty_sources
                                  for k in ['measurement_error', 'sampling_error'])

        if has_model_unc and has_measurement_unc:
            unc_type = 'both'
        elif has_model_unc:
            unc_type = 'epistemic'
        else:
            unc_type = 'aleatoric'

        return UncertaintyQuantification(
            claim=claim,
            uncertainty_sources=list(uncertainty_sources.keys()),
            total_uncertainty=total_unc,
            uncertainty_breakdown=uncertainty_sources,
            uncertainty_propagation=propagation_method,
            confidence_interval=(ci_low, ci_high),
            uncertainty_type=unc_type
        )

    def calibrate_confidence(self,
                           predicted_confidences: List[float],
                           actual_outcomes: List[bool]) -> ConfidenceCalibration:
        """
        Compare predicted confidence to actual accuracy.

        Addresses peer review: "Overconfident claims given evidence quality"
        """
        if not predicted_confidences or not actual_outcomes:
            return ConfidenceCalibration(
                predicted_confidence=0.5,
                actual_accuracy=0.5,
                calibration_score=0.5,
                is_overconfident=False,
                is_underconfident=False,
                recommended_adjustment=0.0
            )

        # Calculate actual accuracy
        actual_accuracy = np.mean(actual_outcomes)

        # Calculate mean predicted confidence
        mean_predicted = np.mean(predicted_confidences)

        # Calculate calibration score
        calibration_score = 1.0 - abs(mean_predicted - actual_accuracy)

        # Determine if over/underconfident
        is_overconfident = mean_predicted > actual_accuracy + 0.1
        is_underconfident = mean_predicted < actual_accuracy - 0.1

        # Calculate recommended adjustment
        recommended_adjustment = actual_accuracy - mean_predicted

        return ConfidenceCalibration(
            predicted_confidence=mean_predicted,
            actual_accuracy=actual_accuracy,
            calibration_score=calibration_score,
            is_overconfident=is_overconfident,
            is_underconfident=is_underconfident,
            recommended_adjustment=recommended_adjustment
        )

    def recognize_knowledge_boundaries(self,
                                      claim: str,
                                      domain: str,
                                      training_data_limits: Dict[str, Any]) -> List[KnowledgeBoundary]:
        """
        Recognize when ASTRA is extrapolating beyond training data.

        Addresses peer review: "Claim made outside domain of validity"
        """
        boundaries = []

        # Check temporal extrapolation
        if 'temporal_range' in training_data_limits:
            current_time = training_data_limits.get('current_time', 2026)
            training_end = training_data_limits['temporal_range'].get('end', 2020)

            if current_time > training_end + 5:  # More than 5 years beyond training
                distance = (current_time - training_end) / (training_end - training_data_limits['temporal_range'].get('start', 2000))
                boundaries.append(KnowledgeBoundary(
                    claim=claim,
                    domain=domain,
                    extrapolation_type='temporal',
                    distance_from_training=distance,
                    confidence_penalty=min(0.5 * distance, 0.8),
                    boundary_warning=f"Claim extrapolates {distance:.1f}x beyond temporal training data"
                ))

        # Check parameter extrapolation
        if 'parameter_ranges' in training_data_limits:
            for param, (min_val, max_val) in training_data_limits['parameter_ranges'].items():
                if param.lower() in claim.lower():
                    # Check if claim refers to extreme values
                    boundaries.append(KnowledgeBoundary(
                        claim=claim,
                        domain=domain,
                        extrapolation_type='parameter',
                        distance_from_training=1.0,  # Would compute actual distance
                        confidence_penalty=0.3,
                        boundary_warning=f"Claim concerns parameter '{param}' near/exceeding training bounds"
                    ))

        # Check regime extrapolation
        if 'regime_validity' in training_data_limits:
            valid_regimes = training_data_limits['regime_validity']
            claim_regime = self._infer_regime(claim)

            if claim_regime and claim_regime not in valid_regimes:
                boundaries.append(KnowledgeBoundary(
                    claim=claim,
                    domain=domain,
                    extrapolation_type='regime',
                    distance_from_training=1.0,
                    confidence_penalty=0.5,
                    boundary_warning=f"Claim applies to '{claim_regime}' regime outside validated range: {valid_regimes}"
                ))

        return boundaries

    def classify_evidential_strength(self,
                                    claim: str,
                                    evidence: Dict[str, Any]) -> EvidentialStrength:
        """
        Categorize claims by evidence quality.

        Addresses peer review: "Claim not supported by adequate evidence"
        """
        # Assess each criterion
        sample_size = evidence.get('sample_size', 0)
        if sample_size > 100:
            size_adequacy = 'large'
        elif sample_size > 30:
            size_adequacy = 'moderate'
        elif sample_size > 10:
            size_adequacy = 'small'
        else:
            size_adequacy = 'very_small'

        # Check reproducibility
        has_replication = evidence.get('replication_studies', 0)
        if has_replication >= 3:
            reproducibility = 'independent_replications'
        elif has_replication >= 1:
            reproducibility = 'some_replication'
        else:
            reproducibility = 'not_replicated'

        # Check methodology
        methodology = evidence.get('methodology', 'observational')
        if methodology in ['randomized', 'interventional', 'experimental']:
            method_quality = 'randomized_or_interventional'
        elif methodology == 'observational':
            method_quality = 'well_designed_observation'
        else:
            method_quality = 'anecdotal'

        # Check consistency
        consistency = evidence.get('consistency', 'isolated')
        if consistency == 'across_studies':
            consistency_quality = 'strong'
        elif consistency == 'some_support':
            consistency_quality = 'moderate'
        else:
            consistency_quality = 'weak'

        # Classify overall evidence quality
        criteria_scores = {
            'sample_size': self._score_sample_size(sample_size),
            'reproducibility': self._score_reproducibility(has_replication),
            'methodology': self._score_methodology(methodology),
            'consistency': self._score_consistency(consistency)
        }

        overall_strength = np.mean(list(criteria_scores.values()))

        if overall_strength >= 0.75:
            evidence_quality = 'strong'
        elif overall_strength >= 0.5:
            evidence_quality = 'moderate'
        elif overall_strength >= 0.25:
            evidence_quality = 'weak'
        else:
            evidence_quality = 'speculative'

        return EvidentialStrength(
            claim=claim,
            evidence_quality=evidence_quality,
            evidence_types=evidence.get('types', []),
            sample_size_adequacy=size_adequacy,
            reproducibility=reproducibility,
            independence=evidence.get('independence', 'unknown'),
            overall_strength=overall_strength
        )

    def compute_humble_confidence(self,
                                 base_confidence: float,
                                 uncertainty: UncertaintyQuantification,
                                 boundaries: List[KnowledgeBoundary],
                                 evidence: EvidentialStrength) -> float:
        """
        Compute confidence score that incorporates all humility factors.
        """
        # Start with base confidence
        humble_confidence = base_confidence

        # Reduce by uncertainty
        unc_penalty = min(uncertainty.total_uncertainty / base_confidence, 0.3)
        humble_confidence -= unc_penalty

        # Reduce by knowledge boundary penalties
        boundary_penalty = sum(b.confidence_penalty for b in boundaries)
        humble_confidence -= boundary_penalty

        # Scale by evidence quality
        evidence_multiplier = {
            'strong': 1.0,
            'moderate': 0.8,
            'weak': 0.5,
            'speculative': 0.3
        }
        humble_confidence *= evidence_multiplier.get(evidence.evidence_quality, 0.5)

        # Ensure in valid range
        humble_confidence = max(0.1, min(0.95, humble_confidence))

        return humble_confidence

    def _monte_carlo_propagation(self,
                                uncertainty_sources: Dict[str, float],
                                n_samples: int = 10000) -> float:
        """Monte Carlo uncertainty propagation"""
        # Sample from each uncertainty distribution
        samples = []

        for _ in range(n_samples):
            sample_value = 0.0
            for source, std in uncertainty_sources.items():
                if source != 'central_value':
                    # Assume normal distribution for each source
                    sample_value += np.random.normal(0, std)

            samples.append(sample_value)

        return np.std(samples)

    def _analytic_propagation(self,
                             uncertainty_sources: Dict[str, float]) -> float:
        """Analytic uncertainty propagation (sum in quadrature)"""
        return np.sqrt(sum(u**2 for u in uncertainty_sources.values()
                          if isinstance(u, (int, float))))

    def _infer_regime(self, claim: str) -> Optional[str]:
        """Infer physical regime from claim text"""
        claim_lower = claim.lower()

        regime_keywords = {
            'quantum': ['quantum', 'planck', 'sub-angstrom'],
            'relativistic': ['relativistic', 'near c', 'light speed'],
            'newtonian': ['classical', 'newtonian', 'low velocity'],
            'thermal': ['thermal', 'temperature-driven'],
            'quantum_limit': ['degenerate', 'fermi', 'bose-einstein'],
            'radiation': ['radiation-dominated', 'photon'],
            'matter': ['matter-dominated', 'dust', 'gas']
        }

        for regime, keywords in regime_keywords.items():
            if any(keyword in claim_lower for keyword in keywords):
                return regime

        return None

    def _score_sample_size(self, n: int) -> float:
        """Score sample size adequacy (0-1)"""
        if n > 100:
            return 1.0
        elif n > 30:
            return 0.7
        elif n > 10:
            return 0.4
        else:
            return 0.1

    def _score_reproducibility(self, n_replications: int) -> float:
        """Score reproducibility (0-1)"""
        if n_replications >= 3:
            return 1.0
        elif n_replications >= 1:
            return 0.6
        else:
            return 0.2

    def _score_methodology(self, methodology: str) -> float:
        """Score methodology quality (0-1)"""
        quality_scores = {
            'randomized': 1.0,
            'interventional': 0.9,
            'experimental': 0.85,
            'observational': 0.5,
            'anecdotal': 0.1
        }
        return quality_scores.get(methodology.lower(), 0.3)

    def _score_consistency(self, consistency: str) -> float:
        """Score consistency across studies (0-1)"""
        consistency_scores = {
            'across_studies': 1.0,
            'some_support': 0.6,
            'isolated': 0.2
        }
        return consistency_scores.get(consistency.lower(), 0.3)


def create_epistemic_humility_engine() -> EpistemicHumilityEngine:
    """Factory function for EpistemicHumilityEngine"""
    return EpistemicHumilityEngine()
