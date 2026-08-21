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
Dwarf Galaxies Domain Module for ASTRA

Satellite galaxies, tidal streams, galaxy formation, dark matter distribution

STATUS: registered but NOT IMPLEMENTED. This module was one of 48 byte-identical
copies of a 110-line template whose `process_query` returned
``f"{description}: Analysis of '{query}'"`` with a hard-coded
``confidence=0.7`` and performed no computation.

There is no computational backend for this domain anywhere in the codebase, and
inventing authored prose to fill the gap would present writing as analysis. It
therefore reports `confidence=0.0` and
`implementation_status=NO_IMPLEMENTATION`, which is a useful signal to an
orchestrator in a way that 0.7 on an echoed query was not.

Version: 2.0.0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .. import DomainConfig, register_domain
from .._computational import ComputationalCapability, ComputationalDomainModule, ImplementationStatus

logger = logging.getLogger(__name__)


class DwarfGalaxiesDomain(ComputationalDomainModule):
    """
    Dwarf Galaxies.

    Registered so that queries in this area are recognised and routed, but no
    numerical capability is implemented.
    """

    implementation_status = ImplementationStatus.NO_IMPLEMENTATION
    referral = ("No dwarf-galaxy dynamical modelling is present. For general stellar-population quantities see the star_formation domain.")

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="dwarf_galaxies",
            version="2.0.0",
            dependencies=[],
            description="Satellite galaxies, tidal streams, galaxy formation, dark matter distribution",
            keywords=['dwarf galaxy', 'satellite galaxy', 'tidal stream', 'dark matter', 'galaxy formation'],
            capabilities=[],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising %s domain (no computational backend)", "dwarf_galaxies")

    def build_capabilities(self) -> List[ComputationalCapability]:
        """No computational backend exists for this domain."""
        return []


def create_dwarf_galaxies_domain() -> DwarfGalaxiesDomain:
    """Create a Dwarf Galaxies domain instance."""
    return DwarfGalaxiesDomain()


try:
    register_domain(DwarfGalaxiesDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
