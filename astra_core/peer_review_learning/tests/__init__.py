"""
Tests for Peer Review Learning Architecture

Comprehensive test suite for all peer review learning modules.
"""

import sys
import pytest
import numpy as np
from datetime import datetime

# Import all modules as package members (works under pytest and as astra_core.peer_review_learning.tests)
from ..causal_validation_framework import CausalValidationFramework, CausalClaim
from ..hypothesis_critic_engine import HypothesisCriticEngine, Hypothesis
from ..statistical_defense_framework import StatisticalDefenseFramework
from ..physics_consistency_sentinel import PhysicsConsistencySentinel, PhysicalClaim
from ..domain_validation_gatekeeper import DomainValidationGatekeeper, DomainClaim
from ..epistemic_humility_engine import EpistemicHumilityEngine
from ..peer_review_simulator import PeerReviewSimulator


class TestCausalValidationFramework:
    """Test causal validation framework"""

    def test_confounder_enumeration(self):
        """Test that confounders are systematically enumerated"""
        framework = CausalValidationFramework()

        confounders = framework.enumerate_confounders(
            cause="turbulence",
            effect="star_formation",
            context={'domain': 'astrophysics', 'variables': ['density', 'temperature']}
        )

        assert len(confounders) > 0
        assert any(c['type'] == 'common_cause' for c in confounders)
        assert any(c['type'] == 'reverse_causation' for c in confounders)

    def test_causal_confidence_scoring(self):
        """Test Hill's criteria scoring"""
        framework = CausalValidationFramework()

        claim = CausalClaim(
            claim="Turbulence causes star formation",
            cause="turbulence",
            effect="star_formation",
            confidence=0.7,
            confounders=[],
            alternative_explanations=[],
            intervention_results={},
            domain_constraints=[]
        )

        evidence = {
            'effect_size': 1.5,
            'replication': [1.2, 1.8, 1.3],
            'temporal_data': {'cause_precedes_effect': True},
            'gradient': {'monotonic': True},
            'theoretical_mechanism': True
        }

        scores = framework.compute_causal_confidence_score(claim, evidence)

        assert 0 <= scores['aggregate'] <= 1
        assert scores['strength'] > 0.5  # Large effect size
        assert scores['temporality'] == 1.0  # Clear temporal ordering


class TestHypothesisCriticEngine:
    """Test hypothesis critic engine"""

    def test_adversarial_hypothesis_generation(self):
        """Test generation of competing explanations"""
        critic = HypothesisCriticEngine()

        hypothesis = Hypothesis(
            statement="Magnetic fields regulate filament width",
            prediction="Width correlates with magnetic field strength",
            evidence=[" observational_data"],
            confidence=0.7,
            alternatives=[],
            falsification_tests=[],
            referee_concerns=[]
        )

        alternatives = critic.generate_adversarial_hypotheses(
            hypothesis,
            context={'domain': 'astrophysics'},
            n_alternatives=3
        )

        assert len(alternatives) == 3
        assert any("measurement" in alt.statement.lower() for alt in alternatives)
        assert any("selection" in alt.statement.lower() for alt in alternatives)

    def test_referee_perspective_simulation(self):
        """Test referee concern generation"""
        critic = HypothesisCriticEngine()

        hypothesis = Hypothesis(
            statement="Filament width is universal",
            prediction="All filaments have width ~0.1 pc",
            evidence=[" Herschel_data"],
            confidence=0.8,
            alternatives=[],
            falsification_tests=[],
            referee_concerns=[]
        )

        concerns = critic.simulate_referee_perspective(
            hypothesis,
            context={'domain': 'astrophysics', 'sample_size': 100}
        )

        assert len(concerns) > 0
        assert any('alternative' in c.lower() for c in concerns)
        assert any('sample' in c.lower() for c in concerns)


class TestStatisticalDefenseFramework:
    """Test statistical defense framework"""

    def test_power_analysis(self):
        """Test sample size calculation"""
        framework = StatisticalDefenseFramework()

        power_analysis = framework.calculate_required_sample_size(
            effect_size=0.5,
            alpha=0.05,
            power=0.8
        )

        assert power_analysis.required_sample_size > 0
        assert 0 < power_analysis.achieved_power <= 1
        assert isinstance(power_analysis.recommendations, list)

    def test_robustness_testing(self):
        """Test robustness to outliers and methods"""
        framework = StatisticalDefenseFramework()

        data = np.random.randn(100)
        data[0] = 10  # Add outlier

        def test_func(x):
            return np.mean(x)

        robustness_results = framework.test_robustness(data, 5.0, test_func)

        assert len(robustness_results) > 0
        assert any(r.test_type == 'outlier_removal' for r in robustness_results)

    def test_multiple_comparisons_tracking(self):
        """Test multiple testing correction"""
        framework = StatisticalDefenseFramework()

        p_values = [0.01, 0.03, 0.001, 0.08, 0.02]
        claims = [f"Claim {i}" for i in range(len(p_values))]

        results = framework.track_multiple_comparisons(p_values, claims)

        assert results['n_tests'] == len(p_values)
        assert 'n_significant_uncorrected' in results
        assert 'n_significant_corrected' in results


class TestPhysicsConsistencySentinel:
    """Test physics consistency sentinel"""

    def test_dimensional_consistency(self):
        """Test dimensional analysis"""
        sentinel = PhysicsConsistencySentinel()

        claim = PhysicalClaim(
            claim="Energy scales with mass",
            equation="E = mc^2",
            variables={'m': (1.0, 'kg'), 'c': (3e8, 'm/s')},
            dimensions={'E': 'M*L^2/T^2', 'm': 'M', 'c': 'L/T'},
            limit_cases={},
            theoretical_bounds={},
            consistency_checks=[]
        )

        violations = sentinel.check_dimensional_consistency(claim)

        # Should have no violations for correct dimensional equation
        critical_violations = [v for v in violations if v.severity == 'critical']
        assert len(critical_violations) == 0

    def test_order_of_magnitude_checking(self):
        """Test sanity checking of physical values"""
        sentinel = PhysicsConsistencySentinel()

        claim = PhysicalClaim(
            claim="Star with temperature 1e20 K",
            equation="",
            variables={'T': (1e20, 'K')},
            dimensions={'T': 'Θ'},
            limit_cases={},
            theoretical_bounds={},
            consistency_checks=[]
        )

        violations = sentinel.check_order_of_magnitude(claim)

        # Should flag unrealistic temperature
        assert len(violations) > 0
        assert any('temperature' in v.description.lower() for v in violations)


class TestDomainValidationGatekeeper:
    """Test domain validation gatekeeper"""

    def test_terminology_consistency(self):
        """Test terminology checking"""
        gatekeeper = DomainValidationGatekeeper()

        text = "The luminosity of the source is 1e40 erg/s at a distance of 10 pc"

        checks = gatekeeper.check_terminology_consistency(text, 'astrophysics')

        assert len(checks) > 0
        assert any(c.term == 'luminosity' for c in checks)

    def test_historical_precedent_finding(self):
        """Test precedent search"""
        gatekeeper = DomainValidationGatekeeper()

        claim = DomainClaim(
            claim="Filament width is ~0.1 pc",
            domain='ism',
            terminology=[],
            theoretical_basis=[],
            observational_support=[],
            consistency_checks=[],
            confidence_by_domain_expert=0.8
        )

        precedents = gatekeeper.find_historical_precedents(claim)

        # Should find Herschel filament width result
        assert len(precedents) > 0
        assert any('Arzoumanian' in str(p.get('authors', '')) for p in precedents)


class TestEpistemicHumilityEngine:
    """Test epistemic humility engine"""

    def test_uncertainty_propagation(self):
        """Test uncertainty propagation through analysis"""
        engine = EpistemicHumilityEngine()

        uncertainty_sources = {
            'measurement_error': 0.1,
            'sampling_error': 0.05,
            'systematic_error': 0.08
        }

        uncertainty = engine.propagate_uncertainty(
            "Test claim",
            uncertainty_sources,
            propagation_method='analytic'
        )

        assert uncertainty.total_uncertainty > 0
        assert uncertainty.confidence_interval[0] < uncertainty.confidence_interval[1]

    def test_confidence_calibration(self):
        """Test confidence calibration"""
        engine = EpistemicHumilityEngine()

        # Overconfident predictions
        predicted = [0.9, 0.85, 0.8]
        actual = [True, False, False]  # Only 1/3 correct

        calibration = engine.calibrate_confidence(predicted, actual)

        assert calibration.is_overconfident
        assert calibration.recommended_adjustment < 0

    def test_evidence_classification(self):
        """Test evidential strength classification"""
        engine = EpistemicHumilityEngine()

        evidence = {
            'sample_size': 200,
            'replication_studies': 3,
            'methodology': 'randomized',
            'consistency': 'across_studies',
            'types': ['experimental', 'observational']
        }

        strength = engine.classify_evidential_strength("Test claim", evidence)

        assert strength.evidence_quality in ['strong', 'moderate', 'weak', 'speculative']
        assert 0 <= strength.overall_strength <= 1


class TestPeerReviewSimulator:
    """Test integrated peer review simulator"""

    def test_full_review_simulation(self):
        """Test complete peer review simulation"""
        simulator = PeerReviewSimulator()

        paper_content = {
            'paper_id': 'test_paper',
            'claims': [
                {
                    'statement': 'Magnetic fields cause filament width to be 0.1 pc',
                    'analysis': {},
                    'equation': 'B_regulates_width'
                }
            ],
            'methods': {
                'sample_size': 50,
                'n_tests': 5
            },
            'results': {
                'p_values': [0.01, 0.03, 0.05, 0.08, 0.02]
            }
        }

        review = simulator.simulate_review(paper_content, domain='astrophysics')

        assert review.readiness_score >= 0
        assert review.overall_verdict in ['accept', 'minor_revisions', 'major_revisions', 'reject']
        assert 'causal_rigor' in review.concerns_by_category
        assert 'statistical_validity' in review.concerns_by_category

    def test_pre_defense_rebuttal(self):
        """Test pre-emptive defense strengthening"""
        simulator = PeerReviewSimulator()

        paper_content = {
            'paper_id': 'test_paper',
            'claims': [
                {'statement': 'Turbulence determines star formation rate'}
            ],
            'methods': {},
            'results': {}
        }

        # Simulate review
        review = simulator.simulate_review(paper_content)

        # Strengthen claims
        rebuttal = simulator.pre_defense_rebuttal(review, paper_content)

        assert len(rebuttal.strengthened_claims) > 0
        assert rebuttal.original_claims is not None


# Integration tests
class TestIntegration:
    """Test integration of all modules"""

    def test_end_to_end_workflow(self):
        """Test complete peer review learning workflow"""
        simulator = PeerReviewSimulator()

        # Create realistic paper content
        paper_content = {
            'paper_id': 'filament_width_paper',
            'claims': [
                {
                    'statement': 'Filament width is regulated by ambipolar diffusion',
                    'analysis': {'confounders': ['turbulence', 'magnetic_field']},
                    'equation': 'width ~ f(ion_neutral_fraction)',
                    'uncertainty': 0.02,
                    'confidence_interval': (0.08, 0.12)
                }
            ],
            'methods': {
                'sample_size': 150,
                'n_tests': 4,
                'multiple_testing_correction': 'fdr',
                'power_analysis': True,
                'replication_studies': 1,
                'code_availability': True
            },
            'results': {
                'effect_sizes': [0.5, 0.3, 0.7, 0.4],
                'uncertainty_propagation': 'monte_carlo'
            }
        }

        # Run review
        review = simulator.simulate_review(paper_content, domain='ism')

        # Verify review structure
        assert review.overall_verdict in ['accept', 'minor_revisions', 'major_revisions', 'reject']
        assert review.readiness_score > 0.5  # Should be relatively high for decent paper

        # Strengthen if needed
        if review.readiness_score < 0.8:
            rebuttal = simulator.pre_defense_rebuttal(review, paper_content)
            assert len(rebuttal.addressed_concerns) > 0


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))