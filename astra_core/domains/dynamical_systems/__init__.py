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
Dynamical Systems Theory Domain Module for ASTRA

Chaos theory, stability analysis, bifurcations.

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation. It now wraps
:mod:`astra_core.astro_physics.gravitational_collapse` and reports a
confidence derived from whether a computation actually ran.

Scope of what is wired
----------------------
The three classical dynamical-systems results of self-gravitating
astrophysical flows that were verified against hand values in the August
2026 audit:

* **Toomre Q** -- the linear stability criterion of a differentially
  rotating disc, and the marginal (bifurcation) surface density Q = 1;
* **Bondi accretion** -- the transonic solution that passes through the
  critical point of the steady spherical flow;
* **Shu inside-out collapse** -- the self-similar solution of the
  isothermal collapse problem.

No chaos diagnostics (Lyapunov exponents, Poincare sections, bifurcation
diagrams) are declared: there is no implementation of any of them anywhere
in `astro_physics`, and writing one here would not be wrapping a verified
routine.

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


class DynamicalSystemsDomain(ComputationalDomainModule):
    """
    Dynamical systems: stability, critical points and self-similar
    solutions in self-gravitating gas.

    Backed by `astro_physics.gravitational_collapse`
    (`FragmentationCriterion.toomre`, `AccretionRates`).
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="dynamical_systems",
            version="2.0.0",
            dependencies=[],
            description="Chaos theory, stability analysis, bifurcations",
            keywords=[
                "dynamical systems", "chaos", "stability", "bifurcation",
                "nonlinear_dynamics", "toomre", "linear stability",
                "critical point", "transonic", "bondi", "self-similar",
                "shu", "marginal stability",
            ],
            capabilities=[
                "toomre_q", "toomre_critical_surface_density",
                "bondi_accretion", "shu_accretion_rate",
            ],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising dynamical_systems domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.gravitational_collapse import (
            AccretionRates, FragmentationCriterion, JeansAnalysis,
        )

        # NOTE ON UNITS. `gravitational_collapse` is CGS throughout:
        # `toomre` wants a surface density in g/cm^2 and an angular
        # frequency in rad/s, `bondi_hoyle` and `shu_inside_out` return g/s.
        # The wrappers convert to (Msun/pc^2, km/s/pc, Msun/yr) using the
        # module's own constants, and every self_check was recomputed by
        # hand in CGS with CODATA k_B and m_H (residuals of a few 1e-5
        # against the module's rounded constants are expected).
        M_SUN, PC, YR = 1.989e33, 3.086e18, 3.156e7
        KMS = 1.0e5

        frag = FragmentationCriterion(mu_particle=2.33)
        jeans = JeansAnalysis(number_density=True, mu_particle=2.33)
        accretion = AccretionRates(mu_particle=2.33)

        def toomre_q(surface_density: float, temperature: float,
                     omega: float) -> float:
            return float(frag.toomre(
                surface_density=surface_density * M_SUN / PC ** 2,
                temperature=temperature,
                omega=omega * KMS / PC,
                radius=0.0)['Q'])

        def toomre_critical_surface_density(temperature: float,
                                            omega: float) -> float:
            sigma_cgs = frag.toomre(
                surface_density=1.0, temperature=temperature,
                omega=omega * KMS / PC,
                radius=0.0)['critical_surface_density']
            return float(sigma_cgs * PC ** 2 / M_SUN)

        def bondi_accretion_rate(mass: float, density: float,
                                 temperature: float) -> float:
            mdot = accretion.bondi_hoyle(
                mass_msun=mass, number_density=density,
                sound_speed=jeans.sound_speed(temperature))
            return float(mdot * YR / M_SUN)

        def shu_accretion_rate(temperature: float) -> float:
            return float(accretion.shu_inside_out(temperature) * YR / M_SUN)

        return [
            ComputationalCapability(
                name="toomre_q",
                description=("Toomre Q linear stability parameter of a "
                             "rotating disc"),
                function=toomre_q,
                parameters=[
                    ("surface_density|sigma_gas", "Msun/pc^2",
                     "disc surface density"),
                    ("temperature|t_mid|t", "K", "midplane temperature"),
                    ("omega|angular_frequency", "km/s/pc",
                     "orbital angular frequency"),
                ],
                returns=("Q", "dimensionless"),
                reference=("Q = c_s Omega / (pi G Sigma); Q < 1 is "
                           "gravitationally unstable (Toomre 1964)"),
                test_ref="test_domain_capabilities.py",
                # c_s(10 K) = 18817 cm/s, Omega = 1e5/3.0857e18 = 3.2405e-14,
                # Sigma = 100 Msun/pc^2 = 2.0885e-2 g/cm^2 -> Q = 0.139230
                self_check=({"surface_density": 100.0, "temperature": 10.0,
                             "omega": 1.0}, 0.1392296, 1e-3),
            ),
            ComputationalCapability(
                name="toomre_critical_surface_density",
                description=("Marginally stable (Q = 1) disc surface "
                             "density"),
                function=toomre_critical_surface_density,
                parameters=[
                    ("temperature|t_mid|t", "K", "midplane temperature"),
                    ("omega|angular_frequency", "km/s/pc",
                     "orbital angular frequency"),
                ],
                returns=("Sigma_crit", "Msun/pc^2"),
                reference="Sigma_crit = c_s Omega / (pi G), i.e. Q = 1",
                test_ref="test_domain_capabilities.py",
                # 18817 x 3.2405e-14 / (pi x 6.6743e-8) = 2.9078e-3 g/cm^2
                #   = 13.9230 Msun/pc^2
                self_check=({"temperature": 10.0, "omega": 1.0},
                            13.92296, 1e-3),
            ),
            ComputationalCapability(
                name="bondi_accretion",
                description=("Bondi transonic accretion onto a compact "
                             "object"),
                function=bondi_accretion_rate,
                parameters=[
                    ("mass|m", "Msun", "accreting mass"),
                    ("density|n_h2|n", "cm^-3", "ambient H2 number density"),
                    ("temperature|t_kin|t", "K", "ambient temperature"),
                ],
                returns=("Mdot", "Msun/yr"),
                reference=("Mdot = 4 pi lambda G^2 M^2 rho / c_s^3, "
                           "lambda = 1.12 for an isothermal gas "
                           "(Bondi 1952)"),
                test_ref="test_domain_capabilities.py",
                # M = 1 Msun, rho = 3.8990e-20 g/cm^3, c_s = 18817 cm/s
                #   -> 1.4517e21 g/s x 3.156e7/1.989e33 = 2.30315e-5 Msun/yr
                self_check=({"mass": 1.0, "density": 1e4,
                             "temperature": 10.0}, 2.303153e-5, 1e-3),
            ),
            ComputationalCapability(
                name="shu_accretion_rate",
                description=("Shu inside-out collapse accretion rate "
                             "(self-similar solution)"),
                function=shu_accretion_rate,
                parameters=[("temperature|t_kin|t", "K",
                             "core kinetic temperature")],
                returns=("Mdot", "Msun/yr"),
                reference=("Mdot = 0.975 c_s^3 / G, the singular isothermal "
                           "sphere expansion wave (Shu 1977)"),
                test_ref="test_domain_capabilities.py",
                # 0.975 x (18817)^3 / 6.6743e-8 = 9.7345e19 g/s
                #   = 1.544440e-6 Msun/yr
                self_check=({"temperature": 10.0}, 1.544440e-6, 1e-3),
            ),
        ]


def create_dynamical_systems_domain() -> DynamicalSystemsDomain:
    """Create a DynamicalSystemsDomain instance."""
    return DynamicalSystemsDomain()


# Domain registration
try:
    register_domain(DynamicalSystemsDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
