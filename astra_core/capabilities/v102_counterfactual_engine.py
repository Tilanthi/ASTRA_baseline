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
V102 Counterfactual Engine -- compatibility module.

The V5.0 discovery capabilities were originally published as flat modules at
``astra_core.capabilities.v102_counterfactual_engine``. The subpackage reorganisation moved the
implementation to ``astra_core.capabilities.synthesis.counterfactual_engine`` but left nothing behind
at the old path, so every script written against the published API -- including
this repository's own ``tests/test_discovery/test_v5_capabilities.py`` -- broke
with ``ModuleNotFoundError``.

On a developer machine that also held older ASTRA checkouts the import could
still succeed by picking the module up from a *different tree*, which made the
breakage invisible locally while a clean clone from GitHub failed. This module
re-exports the canonical implementation so that a standalone install behaves
the same way.

There is no separate implementation here: these are the same objects.

    >>> from astra_core.capabilities.v102_counterfactual_engine import CounterfactualEngine
    >>> from astra_core.capabilities.synthesis.counterfactual_engine import CounterfactualEngine as canonical
    >>> CounterfactualEngine is canonical
    True

New code should import from ``astra_core.capabilities.synthesis.counterfactual_engine``.
"""

from __future__ import annotations

from .synthesis.counterfactual_engine import (  # noqa: F401
    CounterfactualEngine,
    Intervention as _CanonicalInterventionBase,
    CausalEffect,
    create_counterfactual_engine,
)

# `Intervention` is re-bound below to a subclass that also accepts the
# published V5.0 argument names. Everything else is the canonical object.
Intervention = _CanonicalInterventionBase

__all__ = [
    "CounterfactualEngine",
    "Intervention",
    "CausalEffect",
    "create_counterfactual_engine",
]


# ---------------------------------------------------------------------------
# Published-API compatibility
# ---------------------------------------------------------------------------
# The V5.0 API documented `Intervention(variable=..., value=..., 
# intervention_type='do')` and `engine.estimate_causal_effect(...)`. The fields
# were later renamed and the method removed, with no alias and no deprecation,
# so every script written against the published API raised TypeError /
# AttributeError. These adapters restore the published spelling; they add no
# new estimation logic, they delegate to the canonical implementation.

from dataclasses import dataclass as _dataclass  # noqa: E402
from typing import Any as _Any, List as _List, Optional as _Opt  # noqa: E402

from .synthesis.counterfactual_engine import (  # noqa: E402
    InterventionType as InterventionType,
)

_CanonicalIntervention = _CanonicalInterventionBase


def _coerce_intervention_type(value):
    """Accept 'do', 'DO', InterventionType.DO, ..."""
    if isinstance(value, InterventionType):
        return value
    if value is None:
        return list(InterventionType)[0]
    text = str(value).strip().lower()
    for member in InterventionType:
        if member.name.lower() == text or str(member.value).lower() == text:
            return member
    return list(InterventionType)[0]


class Intervention(_CanonicalIntervention):
    """
    `Intervention` accepting both the published V5.0 argument names
    (``variable``, ``value``) and the current ones (``target_variable``,
    ``intervention_value``).

    It is a subclass, so ``isinstance(obj, canonical.Intervention)`` still holds
    and it can be passed anywhere the canonical class is expected.
    """

    def __init__(self, variable: _Opt[str] = None, value: _Any = None,
                 intervention_type: _Any = None, *,
                 target_variable: _Opt[str] = None,
                 intervention_value: _Any = None,
                 intervention_id: _Opt[str] = None,
                 original_value: _Any = 0.0,
                 **kwargs: _Any) -> None:
        target = target_variable if target_variable is not None else variable
        if target is None:
            raise TypeError("Intervention requires `variable` (or `target_variable`)")
        new_value = intervention_value if intervention_value is not None else value
        super().__init__(
            intervention_id=intervention_id or f"do({target}={new_value})",
            target_variable=target,
            intervention_type=_coerce_intervention_type(intervention_type),
            original_value=original_value,
            intervention_value=new_value,
            **kwargs,
        )


@_dataclass
class CausalEffectEstimate:
    """Return type of :func:`estimate_causal_effect` (published V5.0 shape)."""
    effect_estimate: float
    confidence_interval: tuple
    treatment: str
    outcome: str
    method: str = "DML (Robinson partialling-out, cross-fitted)"


def estimate_causal_effect(engine, data, variable_names, intervention, outcome_var,
                           covariates: _Opt[_List[str]] = None) -> CausalEffectEstimate:
    """
    Estimate the causal effect of an intervention on `outcome_var`.

    Delegates to :meth:`CounterfactualEngine.comprehensive_counterfactual_analysis`
    and returns its DML estimate. `intervention` may be an :class:`Intervention`
    or the published plain dict ``{'variable': ..., 'value': ...}``.
    """
    if isinstance(intervention, dict):
        treatment = intervention.get("variable") or intervention.get("target_variable")
    else:
        treatment = getattr(intervention, "target_variable", None)
    if treatment is None:
        raise ValueError("could not determine the treatment variable")
    if covariates is None:
        covariates = [v for v in variable_names if v not in (treatment, outcome_var)]
    result = engine.comprehensive_counterfactual_analysis(
        data, list(variable_names), treatment, outcome_var, covariates)
    dml = result["dml"]
    return CausalEffectEstimate(
        effect_estimate=float(dml.ate),
        confidence_interval=tuple(dml.confidence_interval),
        treatment=treatment,
        outcome=outcome_var,
    )


# expose as a bound-style method so `engine.estimate_causal_effect(...)` works
CounterfactualEngine.estimate_causal_effect = estimate_causal_effect  # noqa: E305

__all__ = __all__ + [
    "InterventionType", "CausalEffectEstimate", "estimate_causal_effect",
]
