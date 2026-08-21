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
Dust Formation & Evolution Domain Module for ASTRA

Nucleation, grain growth, sputtering, destruction

STATUS: registered but NOT IMPLEMENTED. This module was one of 48
byte-identical copies of a 110-line template whose `process_query` returned
``f"{description}: Analysis of '{query}'"`` with a hard-coded
``confidence=0.7`` and performed no computation.

Every one of this domain's four declared capabilities was checked against the
codebase during the August 2026 audit follow-up and none has a verified
computational backend:

* **Nucleation / grain growth** - no classical nucleation theory, no
  condensation temperatures, no monomer attachment rates anywhere in
  `astra_core.astro_physics`. `chemical_networks` handles gas-phase UMIST/KIDA
  reactions only (and every rate there was a factor n_H too fast until audit
  item C17); it has no grain-surface network.
* **Sputtering** - `shock_physics.ShockChemistry.grain_sputtering` is a
  threshold look-up table (0.1 Si, 0.05 Fe, 0.1 Mg, 0.2 C times a linear ramp
  in v_s/v_threshold) with no cited source; worker B's audit classed it and
  `h2_survival_fraction` as invented. Wiring it would present fabricated
  yields as a calculation.
* **Destruction** - likewise absent; there is no thermal/inertial sputtering
  or grain-grain shattering model.

Rather than dress any of that up as analysis, the domain reports
``confidence=0.0`` with ``implementation_status=NO_IMPLEMENTATION``, which is
a useful signal to an orchestrator in a way that 0.7 on an echoed query was
not. The verified dust *emission* physics (opacity, modified blackbody,
dust-traced masses) lives in the `dust_grain_physics` domain.

Version: 2.0.0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .. import DomainConfig, register_domain
from .._computational import (
    ComputationalCapability,
    ComputationalDomainModule,
    ImplementationStatus,
)

logger = logging.getLogger(__name__)


class DustFormationDomain(ComputationalDomainModule):
    """
    Dust formation and destruction.

    Registered so that queries in this area are recognised and routed, but no
    numerical capability is implemented: the codebase contains no nucleation,
    grain-growth or sputtering physics that survived verification.
    """

    implementation_status = ImplementationStatus.NO_IMPLEMENTATION
    referral = ("Dust nucleation, grain growth and sputtering are not "
                "modelled anywhere in this codebase; the only grain-sputtering "
                "routine present (shock_physics.ShockChemistry) is an uncited "
                "threshold table and is deliberately not used. For "
                "dust opacity, modified-blackbody emission and dust-traced "
                "masses use the 'dust_grain_physics' domain.")

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="dust_formation",
            version="2.0.0",
            dependencies=[],
            description="Nucleation, grain growth, sputtering, destruction",
            keywords=['dust formation', 'nucleation', 'grain_growth',
                      'sputtering', 'dust_destruction', 'condensation',
                      'grain shattering'],
            capabilities=[],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising dust_formation domain "
                    "(no computational backend)")

    def build_capabilities(self) -> List[ComputationalCapability]:
        """No verified computational backend exists for this domain."""
        return []


def create_dust_formation_domain() -> DustFormationDomain:
    """Create a Dust Formation & Evolution domain instance."""
    return DustFormationDomain()


try:
    register_domain(DustFormationDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
