"""
Computational domain base class.

Background
----------
48 of the 75 domain modules were byte-identical copies of a 110-line template
whose entire `process_query` was::

    # Simple implementation for now
    result = DomainQueryResult(
        domain_name=...,
        answer=f"{description}: Analysis of '{query}'",
        confidence=0.7, reasoning_trace=[], capabilities_used=[], metadata={})

No computation, no reasoning trace, no capabilities -- and a hard-coded
confidence of 0.7 attached to a string that merely echoes the query back.

Replacing those 48 stubs with 48 hand-written canned-text lookups (which is
what most of the *other* 27 domains do) would not be an improvement: it would
just scale up the same problem, presenting authored prose as if it were
analysis, with an invented confidence attached.

What this module does instead
-----------------------------
1. **Confidence is earned, never asserted.** It is derived from what actually
   happened when the query was processed -- see :class:`Provenance`. A domain
   that computed a number reports high confidence *because a computation ran*;
   a domain with no implementation reports **0.0** and says so.

2. **Real capabilities are callables, not strings.** A domain declares
   :class:`ComputationalCapability` objects that wrap verified functions from
   ``astra_core.astro_physics``. When the query supplies the required
   parameters, the capability is *executed* and the result carries the numbers,
   the units and a literature reference.

3. **Absence is reported, not disguised.** A domain with no computational
   backend returns ``confidence=0.0`` with
   ``implementation_status=NO_IMPLEMENTATION``. That is a useful, honest signal
   to an orchestrator; ``confidence=0.7`` on an echoed string is not.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import BaseDomainModule, DomainQueryResult

logger = logging.getLogger(__name__)


class ImplementationStatus(Enum):
    """How much of a domain is actually implemented."""

    #: Verified numerical routines are wired in and can be executed.
    COMPUTATIONAL = "computational"
    #: Routines exist but have not been validated against ground truth.
    UNVALIDATED = "unvalidated"
    #: Descriptive knowledge only -- no numerical capability.
    DESCRIPTIVE = "descriptive"
    #: Nothing beyond a name and keywords.
    NO_IMPLEMENTATION = "no_implementation"


class Provenance(Enum):
    """
    Where an answer came from. This is what `confidence` is derived from.

    The ordering is deliberate: it is the only thing allowed to set confidence.
    """

    #: A verified numerical routine ran on parameters taken from the query.
    COMPUTED = "computed"
    #: A routine is available but the query did not supply its parameters.
    CAPABILITY_AVAILABLE = "capability_available"
    #: Only a textual description of the domain's scope.
    DESCRIPTIVE = "descriptive"
    #: The domain has nothing to offer for this query.
    NONE = "none"


#: Confidence attached to each provenance. `COMPUTED` is not 1.0 because a
#: correct computation on mis-parsed inputs is still wrong; nothing here is
#: allowed to reach certainty.
_CONFIDENCE = {
    Provenance.COMPUTED: 0.90,
    Provenance.CAPABILITY_AVAILABLE: 0.40,
    Provenance.DESCRIPTIVE: 0.20,
    Provenance.NONE: 0.0,
}


@dataclass
class ComputationalCapability:
    """
    A real, callable computation offered by a domain.

    Attributes:
        name: Short identifier, e.g. ``"jeans_mass"``.
        description: One line describing what it computes.
        function: The callable itself. Must come from a module that has been
            numerically verified; put the verifying test in `test_ref`.
        parameters: Ordered ``(name, unit, description)`` triples. `name` is
            both the keyword passed to `function` and the token looked for in
            the query.
        returns: ``(name, unit)`` of the returned quantity.
        reference: Literature reference for the formula, so a user can check it.
        test_ref: Name of the regression test that pins this function's output.
    """

    name: str
    description: str
    function: Callable[..., Any]
    parameters: Sequence[Tuple[str, str, str]]
    returns: Tuple[str, str]
    reference: str = ""
    test_ref: str = ""
    #: A worked example: ``(inputs, expected_value, relative_tolerance)``.
    #: REQUIRED, and enforced by `test_domain_capabilities.py`, because the
    #: single most dangerous mistake when wrapping a numerical backend is
    #: declaring the wrong output unit -- the wrapper still "works", it just
    #: silently returns a number 1e33 times too large. The reference case is
    #: computed by hand from the formula in `reference` and pins the unit.
    self_check: Optional[Tuple[Dict[str, float], float, float]] = None

    @property
    def required(self) -> List[str]:
        """Canonical parameter names -- the first alias of each declaration.

        `parameters` may declare aliases as "density|n_h2|n"; `extract_parameters`
        returns the canonical name, so this must too or every parameter looks
        missing.
        """
        return [p[0].split("|")[0].strip() for p in self.parameters]

    def signature(self) -> str:
        args = ", ".join(f"{n.split('|')[0].strip()} [{u}]"
                         for n, u, _ in self.parameters)
        return f"{self.name}({args}) -> {self.returns[0]} [{self.returns[1]}]"

    def run(self, **kwargs: Any) -> Any:
        return self.function(**kwargs)


def verify_capability(cap: "ComputationalCapability") -> Tuple[bool, str]:
    """
    Run a capability's declared reference case.

    Returns ``(ok, message)``. A capability without a `self_check` fails: an
    unverified unit declaration is not acceptable in this codebase.
    """
    if cap.self_check is None:
        return False, f"{cap.name}: no self_check declared"
    inputs, expected, rtol = cap.self_check
    try:
        got = cap.run(**inputs)
    except Exception as exc:                               # noqa: BLE001
        return False, f"{cap.name}: raised {type(exc).__name__}: {exc}"
    if not isinstance(got, (int, float)):
        return False, f"{cap.name}: returned {type(got).__name__}, expected a number"
    if expected == 0:
        ok = abs(got) <= rtol
    else:
        ok = abs(got - expected) / abs(expected) <= rtol
    return ok, (f"{cap.name}: got {got:.6g} {cap.returns[1]}, "
                f"expected {expected:.6g} (rtol {rtol:g})")


# --------------------------------------------------------------------------
# Parameter extraction
# --------------------------------------------------------------------------

#: Multiplier to convert a recognised unit to the capability's declared unit.
#: Only unambiguous, dimensionally-checked conversions are listed; anything not
#: here is left alone rather than guessed at.
_UNIT_FACTORS: Dict[str, Dict[str, float]] = {
    "K": {"k": 1.0, "kelvin": 1.0},
    "cm^-3": {"cm^-3": 1.0, "cm-3": 1.0, "/cm3": 1.0, "cm^3": 1.0, "m^-3": 1.0e-6},
    "pc": {"pc": 1.0, "parsec": 1.0, "kpc": 1.0e3, "mpc": 1.0e6},
    "km/s": {"km/s": 1.0, "kms": 1.0, "m/s": 1.0e-3},
    "Msun": {"msun": 1.0, "m_sun": 1.0, "solar mass": 1.0, "solar masses": 1.0},
    "GHz": {"ghz": 1.0, "mhz": 1.0e-3, "khz": 1.0e-6, "hz": 1.0e-9},
    "micron": {"micron": 1.0, "microns": 1.0, "um": 1.0, "µm": 1.0, "mm": 1.0e3,
               "nm": 1.0e-3},
    "G": {"g": 1.0, "gauss": 1.0, "ug": 1.0e-6, "microgauss": 1.0e-6, "mg": 1.0e-3},
    "yr": {"yr": 1.0, "year": 1.0, "years": 1.0, "myr": 1.0e6, "gyr": 1.0e9},
}

_NUMBER = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def extract_parameters(query: str,
                       parameters: Sequence[Tuple[str, str, str]]
                       ) -> Dict[str, float]:
    """
    Pull ``name = value unit`` style quantities out of a free-text query.

    Deliberately conservative: a parameter is only returned when the query
    names it explicitly *and* gives a number. Guessing which bare number in a
    sentence is the temperature is exactly the kind of silent wrongness this
    audit was about, so unmatched parameters are simply absent and the caller
    reports them as missing.

    >>> extract_parameters("jeans mass for n = 1e4 cm^-3 and T = 10 K",
    ...                    [("temperature", "K", ""), ("density", "cm^-3", "")])
    {}

    (``temperature``/``density`` are not literally present; ``T``/``n`` aliases
    must be declared by the capability if they are to be recognised.)
    """
    found: Dict[str, float] = {}
    q = query.lower()
    for name, unit, _desc in parameters:
        # `name` may declare aliases as "temperature|t_kin|t"
        aliases = [a.strip() for a in name.split("|") if a.strip()]
        canonical = aliases[0]
        for alias in aliases:
            pattern = (r"(?<!\w)" + re.escape(alias.lower()) +
                       r"\s*(?:=|:|\bof\b|\bis\b)?\s*(" + _NUMBER + r")\s*"
                       r"([a-zµ_^/\-0-9]*)")
            m = re.search(pattern, q)
            if not m:
                continue
            value = float(m.group(1))
            given_unit = (m.group(2) or "").strip()
            factor = _unit_factor(unit, given_unit)
            if factor is None:
                logger.debug("unrecognised unit %r for %s; skipping",
                             given_unit, canonical)
                continue
            found[canonical] = value * factor
            break
    return found


#: Every token recognised as a unit anywhere in _UNIT_FACTORS. Used to tell a
#: real unit from an ordinary English word that the regex happened to capture
#: ("density 1e4 and temperature 10" must not read "and" as a unit and then
#: silently drop the density).
_ALL_UNIT_TOKENS = {tok for table in _UNIT_FACTORS.values() for tok in table}


def _unit_factor(target_unit: str, given_unit: str) -> Optional[float]:
    """
    Conversion factor from `given_unit` to `target_unit`, or None if the two
    are genuinely incompatible.

    Three cases:
      * no token, or a token that is not a unit at all -> the number is taken
        to be in the capability's own declared unit (factor 1);
      * a recognised unit of the right dimension -> its conversion factor;
      * a recognised unit of the *wrong* dimension -> None, because "T = 10 pc"
        is a genuine ambiguity and must not be silently accepted.
    """
    token = (given_unit or "").strip().lower()
    if not token or token not in _ALL_UNIT_TOKENS:
        return 1.0
    table = _UNIT_FACTORS.get(target_unit)
    if table is None:
        return 1.0 if token == target_unit.lower() else None
    return table.get(token)



def curated_result(domain_name: str, answer: str, topics: Sequence[str],
                   reasoning_trace: Optional[List[str]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> DomainQueryResult:
    """
    Build a result for an answer that is *curated reference text*, not analysis.

    27 of the 75 domains answer from hand-written domain knowledge. That text is
    accurate and worth keeping, but the stubs presented it as though a pipeline
    had run: a literal ``confidence=0.91`` alongside
    ``capabilities_used=["light_curve_analysis", "classification"]`` when no
    light curve had been analysed and nothing had been classified.

    This helper keeps the text and fixes the claim:

    * ``confidence`` is the DESCRIPTIVE constant -- the same for every curated
      answer, because none of them is better evidenced than any other;
    * ``capabilities_used`` is empty, because nothing was executed;
    * the topics the text covers are recorded in
      ``metadata['curated_topics']`` so the information is not lost;
    * ``metadata['provenance'] = 'descriptive'`` marks it as reference text.
    """
    meta: Dict[str, Any] = dict(metadata or {})
    meta.update({
        "provenance": Provenance.DESCRIPTIVE.value,
        "curated_topics": list(topics),
        "computed": False,
        "note": ("Curated reference text for this domain; no numerical "
                 "analysis was performed for this query."),
    })
    return DomainQueryResult(
        domain_name=domain_name,
        answer=answer,
        confidence=_CONFIDENCE[Provenance.DESCRIPTIVE],
        reasoning_trace=list(reasoning_trace or []),
        capabilities_used=[],
        metadata=meta,
    )


# --------------------------------------------------------------------------
# Base class
# --------------------------------------------------------------------------

class ComputationalDomainModule(BaseDomainModule):
    """
    Base for domains that either compute something real or say they cannot.

    Subclasses override :meth:`get_config` and, if they have one,
    :meth:`build_capabilities`. They must not override `process_query` to
    return an authored string with an invented confidence.
    """

    #: Set by subclasses. DESCRIPTIVE/NO_IMPLEMENTATION domains leave
    #: `build_capabilities` returning an empty list.
    implementation_status: ImplementationStatus = \
        ImplementationStatus.NO_IMPLEMENTATION

    #: Optional short note shown to the user for domains with no backend,
    #: explaining what the system *can* do instead.
    referral: str = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._capabilities: Optional[List[ComputationalCapability]] = None

    # -- capability plumbing ------------------------------------------------

    def build_capabilities(self) -> List[ComputationalCapability]:
        """Return the domain's real computations. Default: none."""
        return []

    @property
    def computations(self) -> List[ComputationalCapability]:
        if self._capabilities is None:
            try:
                self._capabilities = list(self.build_capabilities())
            except Exception as exc:                       # noqa: BLE001
                # A broken backend must degrade this domain only, and loudly.
                logger.warning("%s: capabilities unavailable (%s: %s)",
                               type(self).__name__, type(exc).__name__, exc)
                self._capabilities = []
        return self._capabilities

    def get_capabilities(self) -> List[str]:
        caps = [c.name for c in self.computations]
        return caps or list(self.get_config().capabilities or [])

    # -- query processing ---------------------------------------------------

    def process_query(self, query: str,
                      context: Optional[Dict[str, Any]] = None
                      ) -> DomainQueryResult:
        config = self.get_config()
        trace: List[str] = []
        caps = self.computations

        if not caps:
            return self._no_implementation_result(config, query, trace)

        selected = self._select_capability(query, caps)
        if selected is None:
            trace.append(
                f"No capability matched the query; {len(caps)} available.")
            return self._capability_listing_result(
                config, query, caps, trace,
                note="Query did not name any of this domain's computations.")

        trace.append(f"Selected capability: {selected.signature()}")
        params = extract_parameters(query, selected.parameters)
        missing = [p for p in selected.required if p not in params]

        if missing:
            trace.append(
                f"Missing required parameter(s): {', '.join(missing)}")
            return self._capability_listing_result(
                config, query, [selected], trace,
                note=(f"'{selected.name}' can be computed, but the query did "
                      f"not supply: {', '.join(missing)}."))

        try:
            value = selected.run(**params)
        except Exception as exc:                           # noqa: BLE001
            logger.warning("%s.%s raised %s: %s", config.domain_name,
                           selected.name, type(exc).__name__, exc)
            trace.append(f"{selected.name} raised {type(exc).__name__}: {exc}")
            return self._capability_listing_result(
                config, query, [selected], trace,
                note=f"'{selected.name}' failed on the supplied parameters.")

        trace.append(f"Computed {selected.returns[0]} = {value!r}")
        answer = (f"{selected.description}: {selected.returns[0]} = "
                  f"{_format(value)} {selected.returns[1]}")
        if selected.reference:
            answer += f" (formula: {selected.reference})"
        return DomainQueryResult(
            domain_name=config.domain_name,
            answer=answer,
            confidence=_CONFIDENCE[Provenance.COMPUTED],
            reasoning_trace=trace,
            capabilities_used=[selected.name],
            metadata={
                "provenance": Provenance.COMPUTED.value,
                "implementation_status": self.implementation_status.value,
                "computation": selected.name,
                "inputs": params,
                "value": value,
                "units": selected.returns[1],
                "reference": selected.reference,
                "verified_by": selected.test_ref,
            },
        )

    # -- result builders ----------------------------------------------------

    def _select_capability(self, query: str,
                           caps: List[ComputationalCapability]
                           ) -> Optional[ComputationalCapability]:
        """Pick the capability whose name/description the query names."""
        q = query.lower()
        best, best_score = None, 0
        for cap in caps:
            tokens = set(re.split(r"[^a-z0-9]+", cap.name.lower()))
            tokens |= set(re.split(r"[^a-z0-9]+", cap.description.lower()))
            tokens.discard("")
            score = sum(1 for t in tokens
                        if len(t) > 3 and re.search(r"(?<!\w)" + re.escape(t) +
                                                    r"(?!\w)", q))
            if score > best_score:
                best, best_score = cap, score
        return best

    def _capability_listing_result(self, config: Any, query: str,
                                   caps: List[ComputationalCapability],
                                   trace: List[str], note: str
                                   ) -> DomainQueryResult:
        lines = [note, "", "Available computations in this domain:"]
        lines += [f"  - {c.signature()}" for c in caps]
        return DomainQueryResult(
            domain_name=config.domain_name,
            answer="\n".join(lines),
            confidence=_CONFIDENCE[Provenance.CAPABILITY_AVAILABLE],
            reasoning_trace=trace,
            capabilities_used=[],
            metadata={
                "provenance": Provenance.CAPABILITY_AVAILABLE.value,
                "implementation_status": self.implementation_status.value,
                "available": [c.signature() for c in caps],
                "computed": False,
            },
        )

    def _no_implementation_result(self, config: Any, query: str,
                                  trace: List[str]) -> DomainQueryResult:
        trace.append("Domain has no computational implementation.")
        answer = (
            f"The '{config.domain_name}' domain is registered "
            f"({config.description}) but has no computational implementation "
            f"in this codebase, so no analysis of this query was performed."
        )
        if self.referral:
            answer += f" {self.referral}"
        return DomainQueryResult(
            domain_name=config.domain_name,
            answer=answer,
            confidence=_CONFIDENCE[Provenance.NONE],
            reasoning_trace=trace,
            capabilities_used=[],
            metadata={
                "provenance": Provenance.NONE.value,
                "implementation_status":
                    ImplementationStatus.NO_IMPLEMENTATION.value,
                "computed": False,
            },
        )

    # -- status -------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status.update({
            "implementation_status": self.implementation_status.value,
            "n_computations": len(self.computations),
            "computations": [c.signature() for c in self.computations],
        })
        return status


def _format(value: Any) -> str:
    """Format a computed value without pretending to more precision."""
    if isinstance(value, float):
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return f"{value:.4e}"
        return f"{value:.4f}"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}={_format(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format(v) for v in value) + "]"
    return str(value)


__all__ = [
    "ComputationalCapability",
    "verify_capability",
    "ComputationalDomainModule",
    "ImplementationStatus",
    "Provenance",
    "extract_parameters",
    "curated_result",
]
