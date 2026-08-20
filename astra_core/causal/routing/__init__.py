"""
MoE-inspired Routing Module for STAN_IX_ASTRO

This module implements Mixture-of-Experts (MoE) style routing for dynamic
capability selection and conditional computation.

Key components:
- MoECapabilityRouter: Routes tasks to relevant specialized experts
- ConditionalComputationEngine: Orchestrates execution with routing
"""


def _degraded_warn(_module: str, _exc: BaseException) -> None:
    """Log why an optional import degraded instead of failing silently."""
    import logging
    logging.getLogger(__name__).warning(
        "%s unavailable (%s: %s) - dependent names set to None",
        _module, type(_exc).__name__, _exc,
    )


try:
    from .moe_router import (
    MoECapabilityRouter,
    ConditionalComputationEngine,
    TaskType,
    Expert,
    RoutingDecision,
    create_moe_router,
    create_conditional_engine,
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".moe_router", _exc)
    MoECapabilityRouter = ConditionalComputationEngine = TaskType = Expert = RoutingDecision = create_moe_router = create_conditional_engine = None  # degraded: unavailable

__all__ = [
    'MoECapabilityRouter',
    'ConditionalComputationEngine',
    'TaskType',
    'Expert',
    'RoutingDecision',
    'create_moe_router',
    'create_conditional_engine',
]
