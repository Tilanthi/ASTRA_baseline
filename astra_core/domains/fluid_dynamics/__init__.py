"""
Fluid Dynamics & Turbulence Domain Module for ASTRA

Navier-Stokes, Reynolds decomposition, turbulence spectra.

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation. It now wraps
:mod:`astra_core.astro_physics.shock_physics` (Rankine-Hugoniot jump
conditions, verified in the August 2026 audit to conserve all three fluxes
to 1e-16) and :mod:`astra_core.astro_physics.turbulence_analysis`
(TurbulenceStatistics, verified against hand values).

Scope of what is wired
----------------------
Scalar flow diagnostics only. NOT wired: `CShock.shock_width` (audit: ~1e7x
too small); the 2-D structure function and `SpectralPCA` of
`turbulence_analysis` (audit H11, H12) -- the 1-D structure function that
did pass verification is exposed by the `signal_processing` domain instead;
`TurbulenceStatistics.reynolds_number`, which honestly returns None unless
an ISM kinematic viscosity is supplied and so cannot carry a self_check.

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


class FluidDynamicsDomain(ComputationalDomainModule):
    """
    Fluid dynamics and turbulence.

    Backed by `astro_physics.shock_physics.RankineHugoniot`,
    `astro_physics.shock_physics.CShock.alfven_velocity` and
    `astro_physics.turbulence_analysis.TurbulenceStatistics`.
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="fluid_dynamics",
            version="2.0.0",
            dependencies=[],
            description=("Navier-Stokes, Reynolds decomposition, turbulence "
                         "spectra"),
            keywords=[
                "fluid dynamics", "navier-stokes", "turbulence", "reynolds",
                "kolmogorov", "shock", "rankine-hugoniot", "compression",
                "mach number", "alfven", "dissipation", "sonic scale",
                "jump conditions",
            ],
            capabilities=[
                "shock_compression_ratio", "post_shock_temperature",
                "alfven_velocity", "sonic_mach_number",
                "turbulent_dissipation_rate", "sonic_scale",
            ],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising fluid_dynamics domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.shock_physics import (
            MU_ATOMIC, MU_MOLECULAR, CShock, RankineHugoniot,
        )
        from ...astro_physics.turbulence_analysis import TurbulenceStatistics

        # NOTE ON UNITS. `shock_physics` is CGS: `strong_shock_temperature`
        # wants cm/s and returns K, `alfven_velocity` wants Gauss and g/cm^3
        # and returns cm/s. `TurbulenceStatistics` is the exception -- it
        # takes and returns km/s and pc (its `sound_speed` divides by 1e5
        # internally). The wrappers below convert to the astronomer-facing
        # units they declare, and each self_check was recomputed by hand with
        # CODATA k_B = 1.380649e-16 and m_H = 1.6735e-24 g; residuals of a
        # few 1e-4 against the module's rounded 1.381e-16 / 1.673e-24 are
        # expected and are what the quoted rtol allows.
        KMS = 1.0e5
        rh = RankineHugoniot(gamma=5.0 / 3.0, mu=MU_ATOMIC)
        cshock = CShock()

        def sonic_mach_number(velocity_dispersion: float,
                              temperature: float) -> float:
            return float(TurbulenceStatistics(temperature_k=temperature)
                         .mach_number(velocity_dispersion))

        def turbulent_dissipation_rate(velocity_dispersion: float,
                                       scale: float) -> float:
            return float(TurbulenceStatistics()
                         .dissipation_rate(velocity_dispersion, scale))

        def sonic_scale(velocity_dispersion: float, driving_scale: float,
                        temperature: float = 10.0) -> float:
            return float(TurbulenceStatistics(temperature_k=temperature)
                         .sonic_scale_pc(velocity_dispersion, driving_scale))

        return [
            ComputationalCapability(
                name="shock_compression_ratio",
                description=("Density compression ratio across a "
                             "hydrodynamic shock"),
                function=lambda mach_number: float(
                    rh.compression_ratio(mach_number)),
                parameters=[("mach_number|mach|m", "dimensionless",
                             "upstream sonic Mach number")],
                returns=("rho_2/rho_1", "dimensionless"),
                reference=("r = (gamma+1) M^2 / ((gamma-1) M^2 + 2), "
                           "gamma = 5/3 (Landau & Lifshitz, Fluid "
                           "Mechanics, sec. 89)"),
                test_ref="test_domain_capabilities.py",
                # M = 10: (8/3)(100) / ((2/3)(100) + 2) = 266.667/68.667
                #       = 3.8834951456
                self_check=({"mach_number": 10.0}, 3.8834951456, 1e-9),
            ),
            ComputationalCapability(
                name="post_shock_temperature",
                description=("Post-shock temperature in the strong-shock "
                             "limit"),
                function=lambda shock_velocity: float(
                    rh.strong_shock_temperature(shock_velocity * KMS)),
                parameters=[("shock_velocity|v_shock|velocity", "km/s",
                             "shock velocity")],
                returns=("T_2", "K"),
                reference=("T_2 = 3 mu m_H v_s^2 / (16 k_B), mu = 1.27 "
                           "(atomic H + 10% He)"),
                test_ref="test_domain_capabilities.py",
                # 3 x 1.27 x 1.6735e-24 x (1e7)^2 / (16 x 1.380649e-16)
                #   = 2.8863e5 K
                self_check=({"shock_velocity": 100.0}, 2.88634e5, 2e-3),
            ),
            ComputationalCapability(
                name="alfven_velocity",
                description="Alfven velocity of magnetised molecular gas",
                function=lambda magnetic_field, density: float(
                    cshock.alfven_velocity(magnetic_field,
                                           density * MU_MOLECULAR * 1.673e-24)
                    / KMS),
                parameters=[
                    ("magnetic_field|b_field|b", "G", "magnetic field"),
                    ("density|n_h2|n", "cm^-3", "H2 number density"),
                ],
                returns=("v_A", "km/s"),
                reference=("v_A = B / sqrt(4 pi rho), rho = n mu m_H with "
                           "mu = 2.37 (molecular H2 + He)"),
                test_ref="test_domain_capabilities.py",
                # B = 100 uG, n = 1e4: rho = 3.9662e-20 g/cm^3,
                # sqrt(4 pi rho) = 7.0598e-10, v_A = 1.41647e5 cm/s
                self_check=({"magnetic_field": 1e-4, "density": 1e4},
                            1.416472, 2e-3),
            ),
            ComputationalCapability(
                name="sonic_mach_number",
                description="Sonic Mach number of turbulent gas",
                function=sonic_mach_number,
                parameters=[
                    ("velocity_dispersion|sigma_v|sigma", "km/s",
                     "1-D non-thermal velocity dispersion"),
                    ("temperature|t_kin|t", "K", "gas kinetic temperature"),
                ],
                returns=("Mach", "dimensionless"),
                reference=("M = sigma / c_s, c_s = sqrt(k_B T / (mu m_H)), "
                           "mu = 2.33"),
                test_ref="test_domain_capabilities.py",
                # c_s(10 K) = 0.188170 km/s -> M = 1/0.188170 = 5.31434
                self_check=({"velocity_dispersion": 1.0,
                             "temperature": 10.0}, 5.314338, 2e-3),
            ),
            ComputationalCapability(
                name="turbulent_dissipation_rate",
                description=("Turbulent energy dissipation rate per unit "
                             "mass"),
                function=turbulent_dissipation_rate,
                parameters=[
                    ("velocity_dispersion|sigma_v|sigma", "km/s",
                     "1-D velocity dispersion at the driving scale"),
                    ("scale|driving_scale|l", "pc", "driving scale"),
                ],
                returns=("epsilon", "erg/g/s"),
                reference="epsilon ~ sigma^3 / L (Kolmogorov 1941)",
                test_ref="test_domain_capabilities.py",
                # (1e5)^3 / 3.0857e18 = 3.24076e-4 erg/g/s
                self_check=({"velocity_dispersion": 1.0, "scale": 1.0},
                            3.240756e-4, 1e-3),
            ),
            ComputationalCapability(
                name="sonic_scale",
                description=("Scale at which supersonic turbulence becomes "
                             "subsonic"),
                function=sonic_scale,
                parameters=[
                    ("velocity_dispersion|sigma_v|sigma", "km/s",
                     "1-D velocity dispersion at the driving scale"),
                    ("driving_scale|scale|l", "pc", "driving scale"),
                ],
                returns=("l_sonic", "pc"),
                reference=("l_s = L M^(-1/p) with sigma(l) = sigma_L (l/L)^p, "
                           "p = 0.5 (Larson 1981); T = 10 K assumed"),
                test_ref="test_domain_capabilities.py",
                # M = 5.31434 at 10 K -> l_s = 1 pc x M^-2 = 0.0354080 pc
                self_check=({"velocity_dispersion": 1.0,
                             "driving_scale": 1.0}, 0.0354080, 2e-3),
            ),
        ]


def create_fluid_dynamics_domain() -> FluidDynamicsDomain:
    """Create a FluidDynamicsDomain instance."""
    return FluidDynamicsDomain()


# Domain registration
try:
    register_domain(FluidDynamicsDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
