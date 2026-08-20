# Peer Review Learning Architecture for ASTRA

## Overview

This architectural upgrade implements eight major modules designed to improve ASTRA's reasoning, inference, and discovery capabilities by learning from expert peer review feedback. The architecture transforms ASTRA from a pattern discovery system into a **defensible discovery system** that generates claims capable of surviving expert scrutiny.

## Design Philosophy

**Key Insight from Peer Review**: Discoveries must be **defensible before they're valuable**.

The architecture implements this through:
- **Adversarial testing** of every claim
- **Epistemic humility** in uncertainty quantification
- **Domain expertise** deeply embedded in validation
- **Pre-submission review** as standard practice

## Module Descriptions

### 1. Causal Validation Framework
**File**: `causal_validation_framework.py`

**Purpose**: Addresses "correlation vs. causation" concerns by systematically enumerating confounders, implementing do-calculus interventions, and quantifying causal confidence using Hill's criteria.

**Key Features**:
- Systematic confounder enumeration across 6 categories
- Do-calculus intervention simulation
- Causal confidence scoring (Hill's 9 criteria)
- Domain constraint checking
- Dimensional consistency verification

**Example**:
```python
from astra_core.peer_review_learning import CausalValidationFramework

framework = CausalValidationFramework()

# Enumerate confounders
confounders = framework.enumerate_confounders(
    cause="turbulence",
    effect="star_formation",
    context={'domain': 'astrophysics'}
)

# Compute causal confidence
scores = framework.compute_causal_confidence_score(claim, evidence)
```

### 2. Hypothesis Critic Engine
**File**: `hypothesis_critic_engine.py`

**Purpose**: Generates adversarial hypotheses and designs falsification tests to address "alternative mechanisms not considered" concerns.

**Key Features**:
- Adversarial hypothesis generation (5+ alternatives)
- Falsification test design
- Referee perspective simulation
- Hypothesis strength assessment
- Pre-emptive claim strengthening

**Example**:
```python
from astra_core.peer_review_learning import HypothesisCriticEngine

critic = HypothesisCriticEngine()

# Generate competing explanations
alternatives = critic.generate_adversarial_hypotheses(
    hypothesis,
    context={'domain': 'ism'}
)

# Simulate referee concerns
concerns = critic.simulate_referee_perspective(hypothesis, context)
```

### 3. Statistical Defense Framework
**File**: `statistical_defense_framework.py`

**Purpose**: Implements power analysis, robustness testing, and multiple comparisons tracking to address statistical rigor concerns.

**Key Features**:
- Pre-study power analysis
- Outlier sensitivity testing
- Bootstrap robustness validation
- Alternative method comparison
- Multiple testing correction (4 methods)
- Effect size validation
- Assumption checking

**Example**:
```python
from astra_core.peer_review_learning import StatisticalDefenseFramework

framework = StatisticalDefenseFramework()

# Calculate required sample size
power_analysis = framework.calculate_required_sample_size(
    effect_size=0.5,
    alpha=0.05,
    power=0.8
)

# Test robustness
robustness = framework.test_robustness(data, test_result, test_function)
```

### 4. Physics Consistency Sentinel
**File**: `physics_consistency_sentinel.py`

**Purpose**: Enforces dimensional analysis, limit-case validation, and order-of-magnitude sanity checking.

**Key Features**:
- Dimensional consistency checking
- Limit-case validation (zero, infinity)
- Cross-theory consistency verification
- Order-of-magnitude sanity checks
- Physical constraint enforcement
- Conservation law validation

**Example**:
```python
from astra_core.peer_review_learning import PhysicsConsistencySentinel

sentinel = PhysicsConsistencySentinel()

# Check dimensional consistency
violations = sentinel.check_dimensional_consistency(claim)

# Check limit cases
violations.extend(sentinel.check_limit_cases(claim, test_function))

# Sanity check values
violations.extend(sentinel.check_order_of_magnitude(claim))
```

### 5. Peer Review Memory System
**File**: `peer_review_memory.py`

**Purpose**: Stores referee feedback, recognizes recurring mistake patterns, and maintains successful defensive strategies.

**Key Features**:
- SQLite-based persistent memory
- Referee comment extraction and storage
- Mistake pattern recognition
- Defensive strategy database
- Pre-submission review simulation
- Effectiveness tracking

**Example**:
```python
from astra_core.peer_review_learning import create_peer_review_memory

memory = create_peer_review_memory()

# Extract feedback from review
comments = memory.extract_referee_feedback(paper_id, review_text)

# Recognize patterns
patterns = memory.recognize_mistake_patterns(comments)

# Retrieve effective strategies
strategies = memory.retrieve_effective_strategies('causal')
```

### 6. Domain Validation Gatekeeper
**File**: `domain_validation_gatekeeper.py`

**Purpose**: Validates claims against expert domain knowledge, ensures terminology consistency, and finds historical precedents.

**Key Features**:
- Expert knowledge validation
- Terminology consistency checking
- Historical precedent search
- Domain-specific heuristics
- Cross-domain consistency checking

**Example**:
```python
from astra_core.peer_review_learning import DomainValidationGatekeeper

gatekeeper = DomainValidationGatekeeper()

# Validate against domain knowledge
warnings = gatekeeper.validate_against_expert_knowledge(claim)

# Check terminology
terminology_issues = gatekeeper.check_terminology_consistency(text, 'ism')

# Find precedents
precedents = gatekeeper.find_historical_precedents(claim)
```

### 7. Epistemic Humility Engine
**File**: `epistemic_humility_engine.py`

**Purpose**: Quantifies uncertainty, calibrates confidence, recognizes knowledge boundaries, and classifies evidential strength.

**Key Features**:
- Uncertainty propagation (Monte Carlo, analytic)
- Confidence calibration
- Knowledge boundary recognition
- Evidence quality classification
- Humble confidence computation

**Example**:
```python
from astra_core.peer_review_learning import EpistemicHumilityEngine

engine = EpistemicHumilityEngine()

# Propagate uncertainty
uncertainty = engine.propagate_uncertainty(claim, uncertainty_sources)

# Calibrate confidence
calibration = engine.calibrate_confidence(predicted, actual)

# Classify evidence
strength = engine.classify_evidential_strength(claim, evidence)
```

### 8. Peer Review Simulation Module
**File**: `peer_review_simulator.py`

**Purpose**: Integrates all modules to simulate expert peer review before publication and enable pre-emptive defense strengthening.

**Key Features**:
- Comprehensive review simulation
- Category-specific concern analysis
- Readiness scoring
- Verdict determination
- Pre-defensive rebuttal generation
- Referee response estimation

**Example**:
```python
from astra_core.peer_review_learning import PeerReviewSimulator

simulator = PeerReviewSimulator()

# Run simulated review
review = simulator.simulate_review(paper_content, domain='ism')

# Check readiness
print(f"Verdict: {review.overall_verdict}")
print(f"Readiness: {review.readiness_score:.2f}")

# Strengthen claims if needed
if review.readiness_score < 0.8:
    rebuttal = simulator.pre_defense_rebuttal(review, paper_content)
```

## Installation

The peer review learning architecture is integrated into ASTRA's core:

```python
from astra_core.peer_review_learning import (
    PeerReviewSimulator,
    CausalValidationFramework,
    HypothesisCriticEngine,
    StatisticalDefenseFramework,
    PhysicsConsistencySentinel,
    DomainValidationGatekeeper,
    EpistemicHumilityEngine,
    create_peer_review_memory
)
```

## Usage Workflow

### Basic Workflow

1. **Discovery Generation**: ASTRA generates initial discovery/hypothesis
2. **Pre-Submission Review**: Run `PeerReviewSimulator.simulate_review()`
3. **Readiness Assessment**: Check `readiness_score` and `overall_verdict`
4. **Defense Strengthening**: If score < 0.8, run `pre_defense_rebuttal()`
5. **Publication**: Submit only when `ready_to_submit` is True

### Advanced Workflow

```python
# Initialize simulator
simulator = PeerReviewSimulator()

# Prepare discovery content
discovery = {
    'paper_id': 'discovery_001',
    'domain': 'ism',
    'claims': [...],
    'methods': {...},
    'results': {...}
}

# Run comprehensive review
review = simulator.simulate_review(discovery)

# Analyze concerns
for category, concerns in review.concerns_by_category.items():
    if concerns:
        print(f"{category}: {len(concerns)} concerns")
        for concern in concerns:
            print(f"  - {concern['concern']}")
            print(f"    Suggestion: {concern['suggestion']}")

# Strengthen claims
rebuttal = simulator.pre_defense_rebuttal(review, discovery)

# Use strengthened claims
final_claims = rebuttal.strengthened_claims
```

## Testing

Comprehensive test suite in `tests/__init__.py`:

```bash
cd /Users/gjw255/astrodata/SWARM/ASTRA
python -m pytest astra_core/peer_review_learning/tests/ -v
```

## Database Schema

Peer Review Memory uses SQLite with tables:
- `referee_comments`: Extracted reviewer feedback
- `mistake_patterns`: Recurring weakness patterns
- `defensive_strategies`: Successful response strategies
- `pre_submission_reviews`: Pre-emptive review results

Default location: `astra_core/data/peer_review_memory.db`

## Configuration

Customize for specific domains:

```python
# Domain-specific rules
DOMAIN_RULES = {
    'ism': {
        'Jeans_mass': 'Minimum mass for collapse',
        'virial_equilibrium': '2K + U = 0',
        'sonic_scale': 'Turbulence transition scale'
    }
}

# Terminology conventions
TERMINOLOGY = {
    'filament_width': {
        'units': 'pc',
        'convention': 'FWHM of density profile'
    }
}
```

## Performance Considerations

- **Review simulation**: ~1-5 seconds per paper
- **Memory storage**: SQLite handles millions of records
- **Bootstrap robustness**: Configurable iterations (default: 1000)
- **Monte Carlo uncertainty**: Configurable samples (default: 10000)

## Integration with ASTRA Core

The peer review learning architecture integrates with existing ASTRA components:

- **astra_core/reasoning**: Enhanced causal discovery
- **astra_core/domains**: Domain validation
- **astra_core/memory**: Peer review memory integration
- **astra_core/capabilities**: Claim generation with built-in validation

## Future Enhancements

Planned improvements:
1. NLP-based referee comment parsing
2. Automated rebuttal generation
3. Multi-paper consistency checking
4. Real-time literature monitoring
5. Collaborative filtering for strategy recommendation

## Citation

If you use this architecture in your research, please cite:

```bibtex
@software{astra_peer_review_2026,
  author = {White, G.J. and ASTRA Development Team},
  title = {Peer Review Learning Architecture for ASTRA},
  year = {2026},
  note = {Autonomous Scientific Discovery in Astrophysics}
}
```

## License

Part of ASTRA project. See main LICENSE file.

## Contact

For questions or issues, please open a GitHub issue or contact the ASTRA development team.
