"""
Stellar Atmospheres Domain Module for ASTRA

Model atmospheres, line formation, abundance analysis

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation.

**Read this before using the domain.** There is no model-atmosphere code in
this codebase: no hydrostatic/radiative-equilibrium solver, no opacity tables,
no line-formation or curve-of-growth machinery and no abundance analysis. What
*is* available and verified is the continuum blackbody layer - the Planck
function of :mod:`astra_core.astro_physics.infrared_submm` (verified exact:
int B_nu dnu = sigma T^4 / pi to 2.2e-16) and the Stefan-Boltzmann relation of
:mod:`astra_core.astro_physics.physics`. Those give the emergent bolometric
flux and specific intensity of a grey atmosphere, i.e. the definition of
T_eff, and nothing more. Three of this domain's four declared capabilities
(`model_atmospheres`, `line_formation`, `abundance_determination`,
`spectral_synthesis`) therefore have no implementation, and the domain does
not pretend otherwise.

Both capabilities below carry a `self_check` recomputed by hand from the
formula in their `reference` field.

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


class StellarAtmospheresDomain(ComputationalDomainModule):
    """
    Continuum (grey/blackbody) atmosphere relations only.

    Backed by `astro_physics.infrared_submm.ModifiedBlackbody.planck_function`
    and `astro_physics.physics.StellarStructureModel.stefan_boltzmann`.
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="stellar_atmospheres",
            version="2.0.0",
            dependencies=[],
            description=("Model atmospheres, line formation, abundance "
                         "analysis"),
            keywords=['stellar atmosphere', 'model atmosphere',
                      'line formation', 'abundance analysis',
                      'spectral_synthesis', 'planck function', 'blackbody',
                      'effective temperature', 'emergent flux',
                      'specific intensity'],
            capabilities=['model_atmospheres', 'line_formation',
                          'abundance_determination', 'spectral_synthesis'],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising stellar_atmospheres domain "
                    "(continuum blackbody relations only)")

    def build_capabilities(self) -> List[ComputationalCapability]:
        import math

        from ...astro_physics.infrared_submm import ModifiedBlackbody
        from ...astro_physics.physics import StellarStructureModel

        # NOTE ON UNITS. `planck_function` takes microns and K and returns
        # B_nu in erg/s/cm^2/Hz/sr (CGS). `stefan_boltzmann` is SI: it takes
        # metres and returns watts, so evaluating it at R = 1 m and dividing
        # by the surface area 4 pi m^2 gives sigma T^4 in W/m^2, converted to
        # erg/s/cm^2 by 1e3 (1 W/m^2 = 1e3 erg/s/cm^2). The self_checks pin
        # both: B_nu(1 um, 1e4 K) = 1.2355e-4 and sigma T^4 at 5772 K =
        # 6.294e10 erg/s/cm^2.
        mbb = ModifiedBlackbody()
        model = StellarStructureModel()

        return [
            ComputationalCapability(
                name="planck_intensity",
                description=("Planck specific intensity B_nu of a blackbody "
                             "photosphere"),
                function=lambda wavelength, temperature: float(
                    mbb.planck_function(wavelength, temperature)),
                parameters=[
                    ("wavelength|lambda", "micron", "wavelength"),
                    ("temperature|t_eff", "K", "photospheric temperature"),
                ],
                returns=("B_nu", "erg/s/cm^2/Hz/sr"),
                reference=("B_nu(T) = 2 h nu^3 / c^2 / [exp(h nu / k T) - 1], "
                           "nu = c / lambda"),
                test_ref="test_domain_capabilities.py",
                self_check=({"wavelength": 1.0, "temperature": 1e4},
                            1.23552995e-4, 1e-3),
            ),
            ComputationalCapability(
                name="emergent_bolometric_flux",
                description=("Emergent bolometric surface flux of a star of "
                             "given effective temperature"),
                function=lambda temperature: float(
                    model.stefan_boltzmann(1.0, temperature)
                    / (4.0 * math.pi)) * 1e3,
                parameters=[("temperature|t_eff", "K",
                             "effective temperature")],
                returns=("F", "erg/s/cm^2"),
                reference=("F = sigma_SB T_eff^4 = L / (4 pi R^2) "
                           "(definition of T_eff)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"temperature": 5772.0}, 6.29385925e10, 1e-3),
            ),
        ]


def create_stellar_atmospheres_domain() -> StellarAtmospheresDomain:
    """Create a Stellar Atmospheres domain instance."""
    return StellarAtmospheresDomain()


try:
    register_domain(StellarAtmospheresDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
