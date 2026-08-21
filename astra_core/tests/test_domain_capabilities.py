"""
Contract tests for the 75 domain modules.

Background
----------
48 of the 75 domains were byte-identical copies of a 110-line template whose
`process_query` returned an f-string echoing the query, with a hard-coded
``confidence=0.7`` and no computation of any kind. 73 of the 75 contained no
call to `numpy` or `math`.

These tests enforce two properties:

1. **No domain asserts confidence it has not earned.** Confidence must be
   derived from what actually happened (`metadata['provenance']`), never
   written as a literal in `process_query`.

2. **Every declared computation is unit-verified.** Each
   `ComputationalCapability` carries a `self_check` -- a worked example
   computed by hand from the formula it cites -- and it must reproduce it.
   This exists because the most dangerous error when wrapping a numerical
   backend is declaring the wrong output unit: the wrapper still runs, it just
   silently returns a number that is (for example) 1e33 times too large. That
   is precisely the class of defect the August 2026 audit found throughout
   `astro_physics`.
"""

from __future__ import annotations

import pytest

from astra_core.domains import DomainRegistry
from astra_core.domains._computational import (
    ComputationalDomainModule,
    ImplementationStatus,
    Provenance,
    verify_capability,
)


ALL_DOMAINS = sorted(
    d for d in __import__("os").listdir("astra_core/domains")
    if __import__("os").path.isdir(f"astra_core/domains/{d}")
    and d != "__pycache__"
)


@pytest.fixture(scope="module")
def loaded():
    """Every domain, loaded through the registry."""
    reg = DomainRegistry()
    results = reg.auto_load_domains({name: {} for name in ALL_DOMAINS})
    failed = [k for k, v in results.items() if not v]
    assert not failed, f"domains failed to load: {failed}"
    return reg


def _computational_domains(reg):
    return [(n, d) for n, d in reg._domains.items()
            if isinstance(d, ComputationalDomainModule)]


# ---------------------------------------------------------------------------
# 1. every declared computation reproduces its hand-computed reference value
# ---------------------------------------------------------------------------

def test_every_capability_has_a_self_check(loaded):
    """An unverified unit declaration is not acceptable."""
    missing = []
    for name, domain in _computational_domains(loaded):
        for cap in domain.computations:
            if cap.self_check is None:
                missing.append(f"{name}.{cap.name}")
    assert not missing, (
        "capabilities without a self_check (a unit slip here is silent):\n  "
        + "\n  ".join(missing))


def test_every_capability_reproduces_its_reference_value(loaded):
    """
    Run each capability on its declared inputs and compare with the value
    derived by hand from the cited formula.
    """
    failures = []
    checked = 0
    for name, domain in _computational_domains(loaded):
        for cap in domain.computations:
            if cap.self_check is None:
                continue
            checked += 1
            ok, msg = verify_capability(cap)
            if not ok:
                failures.append(f"{name}.{msg}")
    assert checked > 0, "no capabilities were checked"
    assert not failures, "capability self-checks failed:\n  " + "\n  ".join(failures)


def test_every_capability_cites_a_formula(loaded):
    """A number with no traceable formula is not a result."""
    bare = [f"{name}.{cap.name}"
            for name, domain in _computational_domains(loaded)
            for cap in domain.computations
            if not cap.reference.strip()]
    assert not bare, "capabilities with no `reference` formula:\n  " + "\n  ".join(bare)


# ---------------------------------------------------------------------------
# 2. the honesty contract
# ---------------------------------------------------------------------------

def test_no_domain_returns_the_old_template_answer(loaded):
    """
    The template's answer was `f"{description}: Analysis of '{query}'"`.
    No domain may still produce it.
    """
    query = "arbitrary probe query for template detection"
    offenders = []
    for name, domain in loaded._domains.items():
        try:
            result = domain.process_query(query)
        except Exception:                                  # noqa: BLE001
            continue
        answer = getattr(result, "answer", "") or ""
        if f"Analysis of '{query}'" in answer:
            offenders.append(name)
    assert not offenders, (
        "domains still returning the stub template answer:\n  "
        + "\n  ".join(offenders))


def test_domains_without_an_implementation_report_zero_confidence(loaded):
    """
    A domain with no computational backend must say so, not claim 0.7.
    """
    bad = []
    for name, domain in _computational_domains(loaded):
        if domain.implementation_status is not ImplementationStatus.NO_IMPLEMENTATION:
            continue
        result = domain.process_query("some query in this domain")
        if result.confidence != 0.0:
            bad.append(f"{name}: confidence={result.confidence}")
        if result.metadata.get("provenance") != Provenance.NONE.value:
            bad.append(f"{name}: provenance={result.metadata.get('provenance')}")
    assert not bad, "unimplemented domains claiming confidence:\n  " + "\n  ".join(bad)


def test_confidence_is_consistent_with_provenance(loaded):
    """
    Confidence must be a function of provenance, so it cannot be inflated
    independently of what actually happened.
    """
    expected = {
        Provenance.COMPUTED.value: 0.90,
        Provenance.CAPABILITY_AVAILABLE.value: 0.40,
        Provenance.DESCRIPTIVE.value: 0.20,
        Provenance.NONE.value: 0.0,
    }
    bad = []
    for name, domain in _computational_domains(loaded):
        for query in ("compute something with n = 1e4 cm^-3 and T = 10 K",
                      "a query naming nothing in particular"):
            result = domain.process_query(query)
            prov = result.metadata.get("provenance")
            if prov in expected and result.confidence != expected[prov]:
                bad.append(f"{name}: provenance={prov} but "
                           f"confidence={result.confidence}")
    assert not bad, "\n  ".join(bad)


def test_every_domain_declares_its_implementation_status(loaded):
    """`get_status()` must expose how much of the domain is real."""
    missing = [name for name, domain in _computational_domains(loaded)
               if "implementation_status" not in domain.get_status()]
    assert not missing, "\n  ".join(missing)


def test_computed_results_carry_inputs_units_and_reference(loaded):
    """
    A computed answer must be reproducible: it has to say what went in, what
    unit came out, and which formula was used.
    """
    bad = []
    for name, domain in _computational_domains(loaded):
        for cap in domain.computations:
            if cap.self_check is None:
                continue
            inputs, _expected, _rtol = cap.self_check
            query = f"{cap.name.replace('_', ' ')} " + " ".join(
                f"{k} = {v}" for k, v in inputs.items())
            result = domain.process_query(query)
            if result.metadata.get("provenance") != Provenance.COMPUTED.value:
                continue      # parameter phrasing did not round-trip; not this test
            for key in ("inputs", "units", "reference", "value"):
                if key not in result.metadata:
                    bad.append(f"{name}.{cap.name}: metadata missing '{key}'")
    assert not bad, "\n  ".join(bad)


# ---------------------------------------------------------------------------
# 3. the reference implementation, end to end
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query,expected_value,rel", [
    ("What is the Jeans mass for n = 1e4 cm^-3 and T = 10 K?", 2.86841, 1e-3),
    ("jeans length at density 1e4 and temperature 10", 0.211874, 1e-3),
    ("sound speed at T = 10 K", 0.18817, 1e-3),
    ("free-fall time for n = 1e4 cm^-3", 337078.0, 1e-3),
])
def test_statistical_mechanics_computes_correct_values(loaded, query,
                                                       expected_value, rel):
    domain = loaded.get_domain("statistical_mechanics")
    result = domain.process_query(query)
    assert result.metadata["provenance"] == Provenance.COMPUTED.value, result.answer
    assert result.metadata["value"] == pytest.approx(expected_value, rel=rel)
    assert result.confidence == 0.90


def test_missing_parameters_are_reported_not_guessed(loaded):
    """
    Asking for a Jeans mass without supplying n and T must not invent them.
    """
    domain = loaded.get_domain("statistical_mechanics")
    result = domain.process_query("What is the Jeans mass?")
    assert result.metadata["provenance"] == Provenance.CAPABILITY_AVAILABLE.value
    assert result.confidence == 0.40
    assert "did not supply" in result.answer
    assert "density" in result.answer and "temperature" in result.answer


def test_dimensionally_wrong_input_is_rejected(loaded):
    """`T = 10 pc` is a genuine ambiguity and must not be silently accepted."""
    from astra_core.domains._computational import extract_parameters
    params = [("temperature|t", "K", ""), ("density|n", "cm^-3", "")]
    assert extract_parameters("temperature 10 pc", params) == {}
    assert extract_parameters("temperature 10 K", params) == {"temperature": 10.0}


# ---------------------------------------------------------------------------
# 4. curated (hand-written knowledge) domains
# ---------------------------------------------------------------------------

def test_curated_domains_do_not_claim_capabilities_they_did_not_run(loaded):
    """
    The curated domains used to return e.g.
    ``capabilities_used=["light_curve_analysis", "classification"]`` with
    ``confidence=0.91`` when no light curve had been analysed. The text is
    kept; the claim is not.
    """
    from astra_core.domains._computational import Provenance

    probes = {
        "time_domain": "What causes supernovae?",
        "ism": "What is the interstellar medium made of?",
        "cosmology": "What is dark energy?",
        "star_formation": "How do stars form?",
        "agn": "What powers an AGN?",
    }
    bad = []
    for name, query in probes.items():
        domain = loaded.get_domain(name)
        if domain is None:
            continue
        result = domain.process_query(query)
        if result.metadata.get("provenance") != Provenance.DESCRIPTIVE.value:
            bad.append(f"{name}: provenance={result.metadata.get('provenance')}")
        if result.capabilities_used:
            bad.append(f"{name}: claims capabilities_used={result.capabilities_used}")
        if result.confidence != 0.20:
            bad.append(f"{name}: confidence={result.confidence}")
        if "curated_topics" not in result.metadata:
            bad.append(f"{name}: curated topics not recorded")
        if not result.answer.strip():
            bad.append(f"{name}: curated text was lost")
    assert not bad, "\n  ".join(bad)


def test_no_hardcoded_confidence_literals_remain_in_domains():
    """
    194 literal `confidence=0.xx` values were spread through the domain
    modules. Confidence must be derived from provenance, so none may remain.

    AST-based rather than textual, so that prose in docstrings describing the
    old behaviour does not trip it.
    """
    import ast
    import pathlib

    offenders = []
    for path in sorted(pathlib.Path("astra_core/domains").rglob("*.py")):
        if path.name == "_computational.py":
            continue          # defines the provenance -> confidence mapping
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != "confidence":
                    continue
                if isinstance(kw.value, ast.Constant) and \
                        isinstance(kw.value.value, (int, float)):
                    offenders.append(
                        f"{path}:{kw.value.lineno}: confidence={kw.value.value}")
    assert not offenders, (
        "hardcoded confidence literals (confidence must come from provenance):\n  "
        + "\n  ".join(offenders))


def test_process_query_context_is_optional(loaded):
    """
    All 27 curated domains declared `process_query(self, query, context)` with
    context REQUIRED, breaking BaseDomainModule's contract.
    """
    failures = []
    for name, domain in loaded._domains.items():
        try:
            domain.process_query("a probe query")
        except TypeError as exc:
            if "context" in str(exc):
                failures.append(f"{name}: {exc}")
        except Exception:                                  # noqa: BLE001
            pass          # other failures are not this test's business
    assert not failures, "\n  ".join(failures)


# ---------------------------------------------------------------------------
# 5. routing prefers a domain that can actually compute
# ---------------------------------------------------------------------------

def test_routing_prefers_a_domain_that_can_compute(loaded):
    """
    A curated text domain used to outrank one that would return a number:
    "Jeans mass for n = 1e4 cm^-3 and T = 10 K" routed to
    molecular_cloud_collapse (reference text, confidence 0.20) rather than
    statistical_mechanics (M_J = 2.868 Msun, confidence 0.90), because the
    registry looked only at keyword scores.
    """
    from astra_core.domains._computational import Provenance

    quantitative = [
        "What is the Jeans mass for n = 1e4 cm^-3 and T = 10 K?",
        "free-fall time for n = 1e5 cm^-3",
        "sound speed at T = 20 K",
    ]
    for query in quantitative:
        result = loaded.process_query(query)
        assert result.get("confidence") == 0.90, (query, result.get("domain"))
        assert result["metadata"]["provenance"] == Provenance.COMPUTED.value


def test_conceptual_queries_still_reach_the_curated_text(loaded):
    """
    Preferring computation must not strand the reference text: a question with
    no numbers in it should still be answered from curated knowledge.
    """
    from astra_core.domains._computational import Provenance

    result = loaded.process_query("How do molecular clouds collapse?")
    assert result["metadata"]["provenance"] == Provenance.DESCRIPTIVE.value
    assert result.get("confidence") == 0.20
    assert result.get("answer")


def test_can_compute_does_not_run_the_computation(loaded):
    """`can_compute` is a routing probe; it must be cheap and side-effect free."""
    domain = loaded.get_domain("statistical_mechanics")
    assert domain.can_compute("jeans mass for n = 1e4 cm^-3 and T = 10 K")
    assert not domain.can_compute("jeans mass")            # no parameters
    assert not domain.can_compute("what is a galaxy")      # no capability match
