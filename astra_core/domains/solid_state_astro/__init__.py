"""
Solid State Astrophysics Domain Module for ASTRA

Lattice vibrations, band structure, grain optical properties

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


class SolidStateAstrophysicsDomain(ComputationalDomainModule):
    """
    Solid State Astrophysics.

    Registered so that queries in this area are recognised and routed, but no
    numerical capability is implemented.
    """

    implementation_status = ImplementationStatus.NO_IMPLEMENTATION
    referral = ("No ice-mantle or grain-surface solid-state modelling is present. For dust opacities see the dust_grain_physics domain.")

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="solid_state_astro",
            version="2.0.0",
            dependencies=[],
            description="Lattice vibrations, band structure, grain optical properties",
            keywords=['solid state', 'lattice vibrations', 'band_structure', 'grain_optics', 'material_properties'],
            capabilities=[],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising %s domain (no computational backend)", "solid_state_astro")

    def build_capabilities(self) -> List[ComputationalCapability]:
        """No computational backend exists for this domain."""
        return []


def create_solid_state_astro_domain() -> SolidStateAstrophysicsDomain:
    """Create a Solid State Astrophysics domain instance."""
    return SolidStateAstrophysicsDomain()


try:
    register_domain(SolidStateAstrophysicsDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
