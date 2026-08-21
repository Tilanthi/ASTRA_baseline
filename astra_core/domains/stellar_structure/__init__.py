"""
Stellar Structure Theory Domain Module for ASTRA

Polytropes, equation of state, energy transport, opacity

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation. It now wraps
:class:`astra_core.astro_physics.physics.StellarStructureModel`, whose
Stefan-Boltzmann relation is exact and whose main-sequence mass-luminosity
relation was repaired under audit item H1 (the normalisation constants of the
piecewise fit had all been dropped, producing a factor 2.2e4 discontinuity at
55 Msun: L(55.1 Msun) came out as 55 Lsun instead of 1.76e6 Lsun).

Both capabilities carry a `self_check` recomputed by hand from the formula in
their `reference` field. This is not decoration: the backend works in SI and
takes the mass in *kilograms*, so a wrapper that declared Msun without
converting would silently return a number 1e30 times too large and still
"work".

NOT implemented in this codebase and therefore not offered: Lane-Emden
polytrope integration, equations of state, convective/radiative energy
transport and opacity tables. The published fit itself is discontinuous by
-3.4% / +1.0% / -1.8% at its 0.43 / 2 / 55 Msun break points; that is a
property of the fit, not of the code.

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


class StellarStructureDomain(ComputationalDomainModule):
    """
    Bulk stellar structure relations.

    Backed by `astro_physics.physics.StellarStructureModel`.
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="stellar_structure",
            version="2.0.0",
            dependencies=[],
            description=("Polytropes, equation of state, energy transport, "
                         "opacity"),
            keywords=['stellar structure', 'polytrope', 'equation of state',
                      'energy transport', 'opacity', 'mass-luminosity',
                      'main sequence', 'stefan-boltzmann', 'luminosity',
                      'effective temperature'],
            capabilities=['polytrope_models', 'eos_modeling',
                          'energy_transport', 'opacity_calculations'],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising stellar_structure domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.physics import (
            PhysicalConstants,
            StellarStructureModel,
        )

        # NOTE ON UNITS. `physics.py` is SI end to end:
        # `main_sequence_mass_luminosity` expects kg and returns W, and
        # `stefan_boltzmann` expects metres and returns W. The wrappers below
        # take Msun / Rsun and return Lsun, and both self_checks pin those
        # conversions (M = 10 Msun -> 1.4 * 10^3.5 = 4427.19 Lsun;
        # R = 1 Rsun, T = 5772 K -> 1.002 Lsun).
        model = StellarStructureModel()
        M_SUN = PhysicalConstants.M_sun     # kg
        R_SUN = PhysicalConstants.R_sun     # m
        L_SUN = PhysicalConstants.L_sun     # W

        return [
            ComputationalCapability(
                name="main_sequence_luminosity",
                description=("Main-sequence luminosity from the stellar mass "
                             "(mass-luminosity relation)"),
                function=lambda mass: float(
                    model.main_sequence_mass_luminosity(mass * M_SUN)) / L_SUN,
                parameters=[("mass|m_star", "Msun", "stellar mass")],
                returns=("L", "Lsun"),
                reference=("L/Lsun = 0.23 M^2.3 (M<0.43), M^4 (0.43-2), "
                           "1.4 M^3.5 (2-55), 32000 M (>55) "
                           "(Duric, Advanced Astrophysics, CUP 2004)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"mass": 10.0}, 4427.18872, 1e-3),
            ),
            ComputationalCapability(
                name="stefan_boltzmann_luminosity",
                description=("Bolometric luminosity of a star from its radius "
                             "and effective temperature"),
                function=lambda radius, temperature: float(
                    model.stefan_boltzmann(radius * R_SUN,
                                           temperature)) / L_SUN,
                parameters=[
                    ("radius|r_star", "Rsun", "stellar radius"),
                    ("temperature|t_eff", "K", "effective temperature"),
                ],
                returns=("L", "Lsun"),
                reference="L = 4 pi R^2 sigma_SB T_eff^4",
                test_ref="test_domain_capabilities.py",
                self_check=({"radius": 1.0, "temperature": 5772.0},
                            1.00183834, 1e-3),
            ),
        ]


def create_stellar_structure_domain() -> StellarStructureDomain:
    """Create a Stellar Structure Theory domain instance."""
    return StellarStructureDomain()


try:
    register_domain(StellarStructureDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
