"""
Shock Physics Extended Domain Module for ASTRA

Jump conditions, radiative shocks, collisionless shocks, particle acceleration

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation. It now wraps
:class:`astra_core.astro_physics.shock_physics.RankineHugoniot`, whose jump
conditions conserve all three fluxes to 1e-16 (verified in the August 2026
audit), and the diffusive-shock-acceleration index relation from
`supernova_remnant_physics`, which was likewise verified.

Every capability carries a `self_check` recomputed by hand from the formula in
its `reference` field.

Deliberately NOT wired, because the audit found them wrong or unvalidated and
they have not been repaired: `JShock.critical_velocity_h2_dissociation`
(returns 10.2 km/s while the module's own constant says 25 km/s, B-SH-4),
`CShock.alfven_velocity` (contradicts its own docstring, B-SH-6),
`CShock.maximum_temperature` (~10x too hot, B-SH-5), `CShock.compute`
(profile violates mass conservation by ~2x, B-SH-3), `ShockChemistry`
(`h2_survival_fraction`, `sio_enhancement`, `grain_sputtering` are invented
threshold look-up tables with no cited source) and
`OutflowShockAnalysis.momentum_from_co` (omits helium, B-SH-10). Radiative
(cooling) shock structure is therefore not offered: `cooling_length` uses the
pre-shock density and a hardcoded T = 100 K (B-SH-9).

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


class ShockPhysicsExtendedDomain(ComputationalDomainModule):
    """
    Shock jump conditions and diffusive shock acceleration.

    Backed by `astro_physics.shock_physics.RankineHugoniot` (gamma = 5/3,
    mu = 1.27 atomic gas) and the DSA index relations in
    `astro_physics.supernova_remnant_physics`.
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="shock_physics_extended",
            version="2.0.0",
            dependencies=[],
            description=("Jump conditions, radiative shocks, collisionless "
                         "shocks, particle acceleration"),
            keywords=['shock', 'jump conditions', 'radiative shock',
                      'collisionless shock', 'particle_acceleration',
                      'rankine-hugoniot', 'compression ratio', 'mach number',
                      'post-shock temperature', 'diffusive shock acceleration',
                      'shock velocity'],
            capabilities=['rankine_hugoniot', 'radiative_shocks',
                          'collisionless_shocks',
                          'diffusive_shock_acceleration'],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising shock_physics_extended domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.shock_physics import RankineHugoniot
        from ...astro_physics.supernova_remnant_physics import (
            SynchrotronEmission,
        )

        # NOTE ON UNITS. RankineHugoniot works in CGS: velocities in cm/s,
        # temperatures in K. The wrappers declare km/s, which is what shock
        # papers quote, and each self_check pins that factor 1e5. gamma = 5/3
        # and mu = 1.27 (atomic H + 10% He) are the module defaults.
        KMS = 1.0e5
        rh = RankineHugoniot()
        syn = SynchrotronEmission()

        return [
            ComputationalCapability(
                name="compression_ratio",
                description="Density jump across a shock of given Mach number",
                function=lambda mach: float(rh.compression_ratio(mach)),
                parameters=[("mach|mach_number|m", "dimensionless",
                             "upstream Mach number")],
                returns=("rho_2/rho_1", "dimensionless"),
                reference=("r = (g+1)M^2 / [(g-1)M^2 + 2], g = 5/3 "
                           "(-> 4 for M >> 1)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"mach": 10.0}, 3.88349515, 1e-6),
            ),
            ComputationalCapability(
                name="pressure_jump",
                description="Pressure jump across a shock of given Mach number",
                function=lambda mach: float(rh.pressure_ratio(mach)),
                parameters=[("mach|mach_number|m", "dimensionless",
                             "upstream Mach number")],
                returns=("P_2/P_1", "dimensionless"),
                reference="P_2/P_1 = [2 g M^2 - (g-1)] / (g+1), g = 5/3",
                test_ref="test_domain_capabilities.py",
                self_check=({"mach": 10.0}, 124.75, 1e-6),
            ),
            ComputationalCapability(
                name="temperature_jump",
                description=("Temperature jump across a shock of given Mach "
                             "number"),
                function=lambda mach: float(rh.temperature_ratio(mach)),
                parameters=[("mach|mach_number|m", "dimensionless",
                             "upstream Mach number")],
                returns=("T_2/T_1", "dimensionless"),
                reference="T_2/T_1 = (P_2/P_1) / (rho_2/rho_1), g = 5/3",
                test_ref="test_domain_capabilities.py",
                self_check=({"mach": 10.0}, 32.123125, 1e-6),
            ),
            ComputationalCapability(
                name="post_shock_temperature",
                description=("Post-shock temperature from the full jump "
                             "conditions"),
                function=lambda shock_velocity, pre_shock_temperature:
                    float(rh.post_shock_temperature(shock_velocity * KMS,
                                                    pre_shock_temperature)),
                parameters=[
                    ("shock_velocity|v_shock|v_s", "km/s", "shock velocity"),
                    ("pre_shock_temperature|t_1", "K",
                     "upstream gas temperature"),
                ],
                returns=("T_2", "K"),
                reference=("T_2 = T_1 (P_2/P_1)/(rho_2/rho_1) with "
                           "M = v_s / sqrt(g k T_1 / mu m_H), mu = 1.27"),
                test_ref="test_domain_capabilities.py",
                self_check=({"shock_velocity": 100.0,
                             "pre_shock_temperature": 100.0},
                            288570.013, 1e-3),
            ),
            ComputationalCapability(
                name="strong_shock_temperature",
                description="Post-shock temperature in the strong-shock limit",
                function=lambda shock_velocity:
                    float(rh.strong_shock_temperature(shock_velocity * KMS)),
                parameters=[("shock_velocity|v_shock|v_s", "km/s",
                             "shock velocity")],
                returns=("T_s", "K"),
                reference="T_s = 3 mu m_H v_s^2 / (16 k_B), mu = 1.27",
                test_ref="test_domain_capabilities.py",
                self_check=({"shock_velocity": 100.0}, 288482.515, 1e-3),
            ),
            ComputationalCapability(
                name="dsa_electron_index",
                description=("Diffusive-shock-acceleration particle spectral "
                             "index from the compression ratio"),
                function=lambda compression_ratio:
                    float(syn.electron_spectrum_index(compression_ratio)),
                parameters=[("compression_ratio|r", "dimensionless",
                             "shock compression ratio")],
                returns=("p", "dimensionless"),
                reference=("N(E) ~ E^-p with p = (r+2)/(r-1) "
                           "(Bell 1978; Blandford & Ostriker 1978)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"compression_ratio": 4.0}, 2.0, 1e-9),
            ),
        ]


def create_shock_physics_extended_domain() -> ShockPhysicsExtendedDomain:
    """Create a Shock Physics Extended domain instance."""
    return ShockPhysicsExtendedDomain()


try:
    register_domain(ShockPhysicsExtendedDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
