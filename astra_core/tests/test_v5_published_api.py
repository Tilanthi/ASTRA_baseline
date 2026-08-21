"""
The published V5.0 discovery API must work in a standalone install.

Why this file exists
--------------------
A user installed a clean clone into an empty folder, ran six analysis scripts
that had previously worked, and they failed. The scripts import the V5.0
capabilities from their published flat paths::

    from astra_core.capabilities.v101_temporal_causal import TemporalFCIDiscovery

The subpackage reorganisation moved every one of those implementations and left
nothing at the old path. On the original machine the imports still resolved --
by picking the modules up from *other ASTRA checkouts on the same disk* -- so
the breakage was invisible there while a fresh clone from GitHub failed. This
repository's own ``tests/test_discovery/test_v5_capabilities.py`` had been
failing for the same reason.

These tests pin the published surface so it cannot silently move again.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest


PUBLISHED = {
    "v101_temporal_causal": ["TemporalFCIDiscovery", "TimeLaggedPAGEdge",
                             "create_temporal_fci_discovery", "create_granger_fci_hybrid"],
    "v102_counterfactual_engine": ["CounterfactualEngine", "Intervention",
                                   "create_counterfactual_engine"],
    "v103_multimodal_evidence": ["MultiModalEvidenceFusion", "EvidenceType",
                                 "create_multimodal_evidence_fusion"],
    "v104_adversarial_discovery": ["AdversarialDiscoverySystem", "DevilsAdvocateAgent",
                                   "create_adversarial_discovery_system"],
    "v105_meta_discovery": ["MetaDiscoveryTransferEngine", "DiscoveryPattern",
                            "create_meta_discovery_transfer_engine"],
    "v106_explainable_causal": ["ExplainableCausalReasoner", "CausalExplanation",
                                "CausalRelationshipType",
                                "create_explainable_causal_reasoner"],
    "v107_discovery_triage": ["DiscoveryTriageSystem", "TriageCategory",
                              "ImpactDimension", "create_discovery_triage_system"],
    "v108_streaming_discovery": ["StreamingDiscoveryEngine", "OnlineCausalDiscovery",
                                 "StreamingAlertSystem",
                                 "create_streaming_discovery_engine"],
}

CANONICAL = {
    "v101_temporal_causal": "causal.temporal_causal",
    "v102_counterfactual_engine": "synthesis.counterfactual_engine",
    "v103_multimodal_evidence": "multimodal.multimodal_evidence",
    "v104_adversarial_discovery": "discovery.adversarial_discovery",
    "v105_meta_discovery": "discovery.meta_discovery",
    "v106_explainable_causal": "causal.explainable_causal",
    "v107_discovery_triage": "discovery.discovery_triage",
    "v108_streaming_discovery": "discovery.streaming_discovery",
}


@pytest.mark.parametrize("module,names", sorted(PUBLISHED.items()))
def test_published_module_path_imports(module, names):
    m = importlib.import_module(f"astra_core.capabilities.{module}")
    missing = [n for n in names if not hasattr(m, n)]
    assert not missing, f"{module} is missing {missing}"


@pytest.mark.parametrize("module", sorted(CANONICAL))
def test_shim_exports_the_same_objects_not_copies(module):
    """A compatibility shim that duplicated the implementation would drift."""
    shim = importlib.import_module(f"astra_core.capabilities.{module}")
    canon = importlib.import_module(f"astra_core.capabilities.{CANONICAL[module]}")
    name = PUBLISHED[module][0]
    assert getattr(shim, name) is getattr(canon, name)


# ---------------------------------------------------------------------------
# The DML estimator was returning garbage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("true_ate", [2.0, 5.0, 0.0, -3.0])
def test_dml_recovers_a_known_average_treatment_effect(true_ate):
    """
    `DoubleMachineLearning.estimate_ate` divided by `mean(residual_treatment)`
    -- the mean of a residual, which is ~0 by construction. Measured before the
    fix, against data with a known ATE and with plain OLS as an independent
    check:

        true +2.0 -> DML  -175.8   (OLS +2.01)
        true +5.0 -> DML -1167.8   (OLS +5.00)
        true -3.0 -> DML   +28.0   (OLS -2.99)

    The correct Robinson/DML score is E[T~ Y~] / E[T~^2].
    """
    from astra_core.capabilities.v102_counterfactual_engine import (
        create_counterfactual_engine,
    )

    rng = np.random.default_rng(42)
    n = 2000
    confounder = rng.normal(0, 1, n)
    treatment = 0.5 * confounder + rng.normal(0, 1, n)
    outcome = true_ate * treatment + 1.0 * confounder + rng.normal(0, 0.3, n)
    data = np.column_stack([treatment, outcome, confounder])

    result = create_counterfactual_engine().comprehensive_counterfactual_analysis(
        data, ["T", "Y", "C"], "T", "Y", ["C"])
    ate = float(result["dml"].ate)

    # independent check: OLS of Y on (T, C) recovers the coefficient exactly
    design = np.column_stack([np.ones(n), treatment, confounder])
    ols = float(np.linalg.lstsq(design, outcome, rcond=None)[0][1])
    assert ols == pytest.approx(true_ate, abs=0.05)

    assert ate == pytest.approx(true_ate, abs=max(0.15 * abs(true_ate), 0.15)), (
        f"DML returned {ate}, truth {true_ate}, OLS {ols}")


def test_dml_is_reproducible():
    """
    The fold shuffle and both nuisance forests were unseeded, so the same data
    gave a different ATE on every run -- unusable for a published figure.
    """
    from astra_core.capabilities.v102_counterfactual_engine import (
        create_counterfactual_engine,
    )

    rng = np.random.default_rng(1)
    n = 800
    c = rng.normal(0, 1, n)
    t = 0.5 * c + rng.normal(0, 1, n)
    y = 2.0 * t + c + rng.normal(0, 0.3, n)
    data = np.column_stack([t, y, c])

    runs = [float(create_counterfactual_engine()
                  .comprehensive_counterfactual_analysis(
                      data, ["T", "Y", "C"], "T", "Y", ["C"])["dml"].ate)
            for _ in range(3)]
    assert len(set(round(v, 9) for v in runs)) == 1, runs


# ---------------------------------------------------------------------------
# Call-site compatibility
# ---------------------------------------------------------------------------

def test_time_lagged_edge_supports_the_published_tuple_access():
    """Published scripts index these edges positionally."""
    from astra_core.capabilities.v101_temporal_causal import TimeLaggedPAGEdge
    from astra_core.capabilities.causal.temporal_causal import EdgeEndpointType

    end = list(EdgeEndpointType)[0]
    edge = TimeLaggedPAGEdge("mass", "sfr", end, end, 2)
    assert edge[0] == "mass" and edge[1] == "sfr" and edge[2] == 2
    source, target, lag = edge
    assert (source, target, lag) == ("mass", "sfr", 2)
    assert edge.source == "mass" and edge.target == "sfr"
    assert isinstance(str(edge), str)          # __str__ referenced self.t_end


def test_temporal_discovery_accepts_max_lag():
    """`max_lag` was a method parameter in the published API."""
    from astra_core.capabilities.v101_temporal_causal import (
        create_temporal_fci_discovery,
    )

    discovery = create_temporal_fci_discovery()
    series = np.random.default_rng(0).normal(size=(200, 3))
    result = discovery.discover_temporal_causal_structure(
        series, ["a", "b", "c"], max_lag=3)
    assert result is not None
    assert discovery.max_lag == 3


def test_intervention_accepts_the_published_argument_names():
    from astra_core.capabilities.v102_counterfactual_engine import Intervention
    from astra_core.capabilities.synthesis.counterfactual_engine import (
        Intervention as Canonical,
    )

    iv = Intervention(variable="var_A", value=1.5, intervention_type="do")
    assert iv.target_variable == "var_A"
    assert iv.intervention_value == 1.5
    assert isinstance(iv, Canonical)


def test_estimate_causal_effect_exists_and_is_correct():
    from astra_core.capabilities.v102_counterfactual_engine import (
        create_counterfactual_engine,
    )

    rng = np.random.default_rng(7)
    n = 1200
    c = rng.normal(0, 1, n)
    a = 0.5 * c + rng.normal(0, 1, n)
    b = 3.0 * a + c + rng.normal(0, 0.3, n)
    data = np.column_stack([a, b, c])

    engine = create_counterfactual_engine()
    effect = engine.estimate_causal_effect(
        data, ["var_A", "var_B", "var_C"],
        {"variable": "var_A", "value": 1.5, "intervention_type": "do"}, "var_B")
    assert effect.effect_estimate == pytest.approx(3.0, abs=0.2)


def test_streaming_result_exposes_the_published_alerts_key():
    """The key was renamed 'alerts' -> 'new_alerts' with no alias, so every
    existing caller silently got nothing."""
    from astra_core.capabilities.v108_streaming_discovery import (
        create_streaming_discovery_engine,
    )

    engine = create_streaming_discovery_engine(variable_names=["x", "y"])
    batch = np.random.default_rng(0).normal(size=(50, 2))
    result = engine.process_stream_batch(batch)
    assert "alerts" in result
    assert result["alerts"] == result["new_alerts"]


def test_temporal_result_supports_the_published_mapping_access():
    """
    `discover_temporal_causal_structure` used to return a plain dict; scripts
    call `.get('temporal_edges')` on the result. Returning a bare object broke
    every one of them.
    """
    from astra_core.capabilities.v101_temporal_causal import (
        create_temporal_fci_discovery,
    )

    discovery = create_temporal_fci_discovery()
    series = np.random.default_rng(0).normal(size=(150, 3))
    result = discovery.discover_temporal_causal_structure(
        series, ["a", "b", "c"], max_lag=3)

    assert isinstance(result.get("temporal_edges", []), list)
    assert "nodes" in result
    assert set(result["nodes"]) == {"a", "b", "c"}
    assert result.get("does_not_exist", "fallback") == "fallback"
    assert len(result.edges) == len(result.get("temporal_edges"))   # both APIs agree
