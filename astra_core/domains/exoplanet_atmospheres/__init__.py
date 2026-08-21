"""
Exoplanet Atmospheres Domain Module for ASTRA

Transmission/emission spectroscopy, atmospheric characterization,
biosignatures.

STATUS: registered but NOT IMPLEMENTED.  This module was one of 48
byte-identical copies of a 110-line template whose `process_query` returned
``f"{description}: Analysis of '{query}'"`` with a hard-coded
``confidence=0.7`` and performed no computation.

Why nothing is wired
--------------------
There are two candidate backends and neither may be used:

1. `astro_physics/next_gen/atmospheric_retrieval.py` (Guillot P-T profiles,
   transmission and emission spectra, chemical equilibrium, nested-sampling
   retrieval) is the only atmospheric code in the tree, and it was **not
   audited** -- it falls inside the 5173 unaudited lines of `next_gen/` listed
   in section (c).7 of the August 2026 physics audit, for which the audit
   projects roughly 25 further undetected defects at the measured density of
   one confirmed bug per 300 audited lines.  Presenting its numbers with the
   0.90 "computed" confidence of this framework would assert a verification
   that does not exist.

2. `astro_physics/exoplanet_transit.py` *was* audited and its BLS was
   repaired in this branch (audit B6.1/B6.2/B6.3), but it detects transits in
   a photometric time series.  It says nothing about an atmosphere, it takes
   a light curve rather than scalar parameters, and it belongs to the
   `exoplanets` domain rather than to this one.

Reporting `confidence=0.0` with `implementation_status=NO_IMPLEMENTATION` is a
useful signal to an orchestrator in a way that 0.7 on an echoed query was not.
The referral below points at what the system can actually do instead.

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


class ExoplanetAtmospheresDomain(ComputationalDomainModule):
    """
    Exoplanet atmospheres.

    Registered so that queries in this area are recognised and routed, but no
    numerical capability is implemented: the only atmospheric backend in the
    codebase is unaudited (see the module docstring).
    """

    implementation_status = ImplementationStatus.NO_IMPLEMENTATION
    referral = (
        "No verified atmospheric model is wired. `astro_physics/next_gen/"
        "atmospheric_retrieval.py` exists but was not covered by the August "
        "2026 physics audit, so its output carries no verification. For "
        "transit DETECTION from a light curve see the exoplanets domain; for "
        "blackbody/modified-blackbody continuum emission see the "
        "infrared_astronomy domain."
    )

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="exoplanet_atmospheres",
            version="2.0.0",
            dependencies=[],
            description=("Transmission/emission spectroscopy, atmospheric "
                         "characterization, biosignatures"),
            keywords=[
                'exoplanet atmosphere', 'transmission spectroscopy',
                'emission spectroscopy', 'biosignature',
                'atmospheric retrieval', 'scale height', 'hot jupiter',
            ],
            capabilities=[],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising exoplanet_atmospheres domain "
                    "(no verified computational backend)")

    def build_capabilities(self) -> List[ComputationalCapability]:
        """No *verified* computational backend exists for this domain."""
        return []


# Factory function
def create_exoplanet_atmospheres_domain() -> ExoplanetAtmospheresDomain:
    """Create an Exoplanet Atmospheres domain instance"""
    return ExoplanetAtmospheresDomain()


# Domain registration
try:
    register_domain(ExoplanetAtmospheresDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
