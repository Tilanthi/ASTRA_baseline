"""
Peer Review Learning Architecture - Integration Guide

This guide shows how to integrate the peer review learning modules
into ASTRA's discovery pipeline to improve reasoning, inference,
and discovery capabilities through learning from expert feedback.
"""

# ==========================================
# BASIC USAGE
# ==========================================

from astra_core.peer_review_learning import (
    PeerReviewSimulator,
    CausalValidationFramework,
    HypothesisCriticEngine,
    StatisticalDefenseFramework,
    create_peer_review_memory
)

# Example 1: Simulate peer review before submission
def example_pre_submission_review():
    """Run pre-emptive review to catch issues early"""

    simulator = PeerReviewSimulator()

    # Prepare paper content
    paper_content = {
        'paper_id': 'filament_discovery_001',
        'domain': 'astrophysics',
        'claims': [
            {
                'statement': 'Magnetic fields regulate filament width to 0.1 pc',
                'analysis': {'confounders': ['turbulence', 'density']},
                'methods': ['observation', 'statistical_test']
            }
        ],
        'methods': {
            'sample_size': 150,
            'n_tests': 5,
            'multiple_testing_correction': 'fdr',
            'power_analysis': True
        },
        'results': {
            'p_values': [0.01, 0.03, 0.05, 0.08, 0.02],
            'effect_sizes': [0.5, 0.3, 0.7, 0.4]
        }
    }

    # Run simulated review
    review = simulator.simulate_review(paper_content, domain='ism')

    # Check results
    print(f"Overall verdict: {review.overall_verdict}")
    print(f"Readiness score: {review.readiness_score:.2f}")
    print(f"\nConcerns by category:")
    for category, concerns in review.concerns_by_category.items():
        if concerns:
            print(f"  {category}: {len(concerns)} concerns")
            for concern in concerns[:3]:  # Show top 3
                print(f"    - {concern['concern']}")

    # Strengthen claims if needed
    if review.readiness_score < 0.8:
        rebuttal = simulator.pre_defense_rebuttal(review, paper_content)
        print(f"\nStrengthened claims:")
        for i, claim in enumerate(rebuttal.strengthened_claims):
            print(f"  {i+1}. {claim}")

    return review


# Example 2: Validate causal claims
def example_causal_validation():
    """Ensure causal claims are properly supported"""

    from astra_core.peer_review_learning import CausalClaim

    framework = CausalValidationFramework()

    # Define causal claim
    claim = CausalClaim(
        claim="Ambipolar diffusion sets filament width",
        cause="ion_neutral_fraction",
        effect="filament_width",
        confidence=0.75,
        confounders=['turbulence', 'magnetic_field_strength'],
        alternative_explanations=['purely_hydrodynamic_turbulence'],
        intervention_results={'effect_under_intervention': 0.11},
        domain_constraints=['energy_conservation', 'virial_equilibrium']
    )

    # Enumerate confounders
    confounders = framework.enumerate_confounders(
        cause="ion_neutral_fraction",
        effect="filament_width",
        context={'domain': 'ism', 'variables': ['density', 'temperature']}
    )

    print(f"Found {len(confounders)} potential confounders")
    for confounder in confounders:
        print(f"  - {confounder['type']}: {confounder['explanation']}")

    # Compute causal confidence using Hill's criteria
    evidence = {
        'effect_size': 0.8,
        'replication': [0.75, 0.85, 0.78],
        'temporal_data': {'cause_precedes_effect': True},
        'gradient': {'monotonic': True},
        'theoretical_mechanism': True,
        'supporting_evidence': True
    }

    scores = framework.compute_causal_confidence_score(claim, evidence)

    print(f"\nCausal confidence scores:")
    for criterion, score in scores.items():
        if criterion != 'aggregate':
            print(f"  {criterion}: {score:.2f}")
    print(f"\nAggregate causal confidence: {scores['aggregate']:.2f}")

    return claim, scores


# Example 3: Generate adversarial hypotheses
def example_hypothesis_criticism():
    """Generate competing explanations to test robustness"""

    from astra_core.peer_review_learning import Hypothesis, HypothesisCriticEngine

    critic = HypothesisCriticEngine()

    # Define hypothesis
    hypothesis = Hypothesis(
        statement="Filament width is regulated by ambipolar diffusion",
        prediction="Width correlates with ion-neutral collision timescale",
        evidence=[" Herschel_observations", "MHD_simulations"],
        confidence=0.7,
        alternatives=[],
        falsification_tests=[],
        referee_concerns=[]
    )

    # Generate adversarial alternatives
    alternatives = critic.generate_adversarial_hypotheses(
        hypothesis,
        context={'domain': 'ism'},
        n_alternatives=5
    )

    print(f"Generated {len(alternatives)} adversarial hypotheses:")
    for i, alt in enumerate(alternatives):
        print(f"  {i+1}. {alt.statement}")
        print(f"     Prediction: {alt.prediction}")

    # Simulate referee perspective
    concerns = critic.simulate_referee_perspective(
        hypothesis,
        context={'domain': 'ism', 'sample_size': 150}
    )

    print(f"\nSimulated referee concerns:")
    for concern in concerns[:5]:
        print(f"  - {concern}")

    return alternatives, concerns


# Example 4: Statistical defense
def example_statistical_defense():
    """Build statistical rigor into analysis"""

    from astra_core.peer_review_learning import StatisticalDefenseFramework

    framework = StatisticalDefenseFramework()

    # Power analysis before collecting data
    power_analysis = framework.calculate_required_sample_size(
        effect_size=0.5,  # Medium effect
        alpha=0.05,
        power=0.8
    )

    print(f"Required sample size: {power_analysis.required_sample_size}")
    print(f"Achieved power: {power_analysis.achieved_power:.2f}")
    print(f"Recommendations:")
    for rec in power_analysis.recommendations:
        print(f"  - {rec}")

    # Validate effect size
    effect_validation = framework.validate_effect_size(
        effect_size=0.5,
        domain='astrophysics',
        context={}
    )

    print(f"\nEffect size validation:")
    print(f"  Magnitude: {effect_validation['magnitude']}")
    print(f"  Practically significant: {effect_validation['practically_significant']}")

    return power_analysis, effect_validation


# Example 5: Learn from actual peer review
def example_learn_from_review():
    """Extract lessons from actual referee feedback"""

    memory = create_peer_review_memory()

    # Simulate receiving referee comments
    review_text = """
    The authors claim that magnetic fields regulate filament width,
    but they have not adequately addressed the role of turbulence.
    The sample size of 50 filaments may be insufficient for
    the claimed statistical significance. Alternative mechanisms
    such as pure hydrodynamic turbulence should be considered.
    """

    # Extract structured feedback
    comments = memory.extract_referee_feedback(
        paper_id='filament_paper_001',
        review_text=review_text
    )

    print(f"Extracted {len(comments)} referee comments:")
    for comment in comments:
        print(f"  [{comment.severity}] {comment.category}: {comment.concern}")

    # Recognize patterns
    patterns = memory.recognize_mistake_patterns(comments)
    if patterns:
        print(f"\nRecurring patterns identified:")
        for pattern in patterns:
            print(f"  - {pattern.pattern_name}: {pattern.description}")
            print(f"    Prevention: {pattern.prevention_strategy}")

    return comments, patterns


# ==========================================
# INTEGRATION WITH ASTRA CORE
# ==========================================

def integrate_with_astra_discovery():
    """Example: Integrate peer review learning into ASTRA's discovery pipeline"""

    from astra_core import create_stan_system
    from astra_core.peer_review_learning import PeerReviewSimulator

    # Create ASTRA system
    system = create_stan_system()

    # Add peer review simulator
    simulator = PeerReviewSimulator()

    # When ASTRA generates a discovery candidate
    discovery = {
        'paper_id': 'astra_discovery_001',
        'domain': 'astrophysics',
        'claims': [
            {'statement': 'X causes Y in regime Z'},
            # ... more claims
        ],
        'methods': {...},
        'results': {...}
    }

    # Run pre-submission review
    review = simulator.simulate_review(discovery)

    # Only publish if ready
    if review.readiness_score >= 0.8:
        print("Discovery ready for publication")
        return discovery
    else:
        print("Discovery needs revision")
        # Strengthen claims
        rebuttal = simulator.pre_defense_rebuttal(review, discovery)
        # Use strengthened claims
        return rebuttal.strengthened_claims


# ==========================================
# BATCH PROCESSING
# ==========================================

def batch_review_discoveries(discoveries: list) -> dict:
    """Review multiple discoveries in batch"""

    simulator = PeerReviewSimulator()
    results = {
        'ready': [],
        'needs_revision': [],
        'reject': []
    }

    for discovery in discoveries:
        review = simulator.simulate_review(discovery)

        if review.overall_verdict == 'accept':
            results['ready'].append(discovery)
        elif review.overall_verdict in ['minor_revisions', 'major_revisions']:
            results['needs_revision'].append({
                'discovery': discovery,
                'review': review
            })
        else:
            results['reject'].append({
                'discovery': discovery,
                'review': review
            })

    return results


# ==========================================
# CONFIGURATION
# ==========================================

# Custom domain rules
ISM_DOMAIN_RULES = {
    'Jeans_mass': 'Must exceed Jeans mass for collapse',
    'virial_equilibrium': '2K + U = 0 for equilibrium',
    'sonic_scale': 'Turbulence transitions at sonic scale',
    'ambipolar_diffusion': 'Timescale ~ 1/(ion-neutral collision rate)'
}

# Custom terminology
ISM_TERMINOLOGY = {
    'filament_width': {
        'units': 'pc',
        'convention': 'FWHM of density profile',
        'typical_value': 0.1
    },
    'Mach_number': {
        'units': 'dimensionless',
        'convention': 'M = sigma / c_s',
        'regime': {'subsonic': '< 1', 'supersonic': '> 1'}
    }
}


# ==========================================
# RUN EXAMPLES
# ==========================================

if __name__ == '__main__':
    print("=== Peer Review Learning Architecture Demo ===\n")

    print("Example 1: Pre-Submission Review")
    print("-" * 50)
    review = example_pre_submission_review()
    print()

    print("Example 2: Causal Validation")
    print("-" * 50)
    claim, scores = example_causal_validation()
    print()

    print("Example 3: Hypothesis Criticism")
    print("-" * 50)
    alternatives, concerns = example_hypothesis_criticism()
    print()

    print("Example 4: Statistical Defense")
    print("-" * 50)
    power_analysis, effect_validation = example_statistical_defense()
    print()

    print("Example 5: Learn from Review")
    print("-" * 50)
    comments, patterns = example_learn_from_review()
    print()
