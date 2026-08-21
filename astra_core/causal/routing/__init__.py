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
