"""Regression tests for domain-registry query routing.

Before the August 2026 re-audit, `DomainRegistry.process_query()` answered
"No suitable domain found for query" for essentially every query, because of
four independent defects:

1. `find_best_domain_for_query` seeded `best_score = min_confidence` and then
   required `score > best_score`, so a domain scoring exactly the documented
   minimum was rejected.
2. `BaseDomainModule.can_handle_query` read `self.config.keywords`, but the
   registry builds domains from `get_default_config()`, which in several modules
   (e.g. `ism`) omits keywords -- those domains scored 0.0 for every query.
3. The score was `matches / len(keywords)`, dividing by the domain's own
   vocabulary size, so a domain with a rich keyword list was penalised.
4. Keyword matching was a bare substring test, so 'rv' matched inside "curve",
   'hi' inside "this" and 'sn' inside "doesn't".
"""

import pytest

from astra_core.domains import DomainRegistry, _keyword_in


DOMAINS = [
    "ism", "star_formation", "exoplanets", "time_domain",
    "cosmology", "mhd", "gravitational_waves", "black_holes",
]


@pytest.fixture(scope="module")
def registry():
    reg = DomainRegistry()
    loaded = reg.auto_load_domains({name: {} for name in DOMAINS})
    assert all(loaded.values()), f"failed to load: {loaded}"
    return reg


def test_keyword_matching_respects_word_boundaries():
    # pre-fix: all four of these were True
    assert not _keyword_in("rv", "supernova light curve classification")
    assert not _keyword_in("hi", "what is this")
    assert not _keyword_in("sn", "it doesn't matter")
    assert not _keyword_in("agn", "magnetic field")
    # genuine matches still work, including multi-word keywords
    assert _keyword_in("supernova", "supernova light curve")
    assert _keyword_in("molecular cloud", "a molecular cloud core")


def test_score_is_not_diluted_by_vocabulary_size(registry):
    """`ism` declares 45 keywords; it must not be punished for that."""
    ism = registry.get_domain("ism")
    score = ism.can_handle_query("molecular cloud filament fragmentation")
    # pre-fix: 3/45 = 0.067, below the 0.1 threshold -> never selected
    assert score >= 0.1


@pytest.mark.parametrize("query,expected", [
    ("molecular cloud filament fragmentation", "ism"),
    ("transiting exoplanet atmosphere", "exoplanets"),
    ("magnetic reconnection dynamo", "mhd"),
    ("initial mass function of young stars", "star_formation"),
    ("binary black hole merger gravitational wave", "gravitational_waves"),
])
def test_queries_route_to_the_right_domain(registry, query, expected):
    result = registry.process_query(query)
    # pre-fix: every one of these returned
    # {'success': False, 'error': 'No suitable domain found for query'}
    assert result.get("success") is not False, result
    assert result.get("domain") == expected, result


def test_off_topic_query_still_matches_nothing(registry):
    result = registry.process_query("what is the capital of France")
    assert result.get("success") is False
    assert "No suitable domain" in result.get("error", "")


def test_threshold_boundary_is_inclusive(registry):
    """A domain scoring exactly `min_confidence` must be selectable."""
    domain = registry.find_best_domain_for_query(
        "molecular cloud filament fragmentation", min_confidence=1.0
    )
    # score saturates at 1.0 for >=3 units of evidence, so an exact-1.0
    # threshold must still select it (pre-fix `>` made this impossible)
    assert domain is not None
    assert domain.config.domain_name == "ism"


def test_top_level_answer_confidence_is_numeric():
    """
    `EnhancedUnifiedSTANSystem` put the orchestrator's boolean `success` flag
    into the `confidence` field, so the documented top-level API returned
    `confidence: True` instead of a number.
    """
    import astra_core

    result = astra_core.create_stan_system().answer(
        "What causes filament fragmentation in molecular clouds?"
    )
    conf = result["confidence"]
    assert not isinstance(conf, bool), "confidence is still a bool"
    assert isinstance(conf, (int, float))
    assert 0.0 <= conf <= 1.0
