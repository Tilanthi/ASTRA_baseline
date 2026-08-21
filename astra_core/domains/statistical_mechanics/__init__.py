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
Statistical Mechanics Domain Module for ASTRA

Gravitational statistical mechanics: Jeans criterion, virial balance and
free-fall timescales for self-gravitating gas.

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation. It now wraps
verified routines from :mod:`astra_core.astro_physics.gravitational_collapse`,
which are pinned by regression tests, and reports a confidence derived from
whether a computation actually ran.

Version: 2.0.0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from .. import DomainConfig, register_domain
from .._computational import (
    ComputationalCapability,
    ComputationalDomainModule,
    ImplementationStatus,
)

logger = logging.getLogger(__name__)


class StatisticalMechanicsDomain(ComputationalDomainModule):
    """
    Gravitational statistical mechanics.

    Backed by `astro_physics.gravitational_collapse`, whose Jeans,
    Bonnor-Ebert, Toomre and free-fall routines were verified against
    hand-derived values during the August 2026 audit.
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="statistical_mechanics",
            version="2.0.0",
            dependencies=[],
            description=("Gravitational statistical mechanics: Jeans criterion, "
                         "virial balance, free-fall collapse"),
            keywords=[
                "statistical mechanics", "jeans mass", "jeans length",
                "virial parameter", "virial theorem", "free-fall time",
                "freefall time", "bonnor-ebert", "toomre", "equipartition",
                "boltzmann", "sound speed",
            ],
            capabilities=[
                "jeans_mass", "jeans_length", "sound_speed",
                "virial_parameter", "free_fall_time", "bonnor_ebert_mass",
                "toomre_q",
            ],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising statistical_mechanics domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.gravitational_collapse import (
            JeansAnalysis, VirialAnalysis, FreefallCollapse,
            FragmentationCriterion,
        )

        jeans = JeansAnalysis()
        virial = VirialAnalysis()
        frag = FragmentationCriterion()

        # NOTE ON UNITS. `gravitational_collapse` works in CGS throughout:
        # sound_speed -> cm/s, jeans_length -> cm, jeans_mass -> g. The wrappers
        # below convert to the astronomer-facing units they declare, and every
        # one carries a `self_check` computed by hand from the formula in
        # `reference` so that a unit slip fails a test instead of silently
        # returning a number 1e33 times too large.
        M_SUN = 1.989e33      # g
        PC = 3.0857e18        # cm
        KMS = 1.0e5           # cm/s
        YR = 3.156e7          # s

        return [
            ComputationalCapability(
                name="jeans_mass",
                description="Jeans mass of an isothermal self-gravitating gas",
                function=lambda density, temperature: jeans.jeans_mass(
                    density=density, temperature=temperature) / M_SUN,
                parameters=[
                    ("density|n_h2|n", "cm^-3", "H2 number density"),
                    ("temperature|t_kin|t", "K", "gas kinetic temperature"),
                ],
                returns=("M_J", "Msun"),
                reference="M_J = (pi^(5/2)/6) c_s^3 G^(-3/2) rho^(-1/2)",
                test_ref="test_domain_capabilities.py",
                self_check=({"density": 1e4, "temperature": 10.0}, 2.86841, 1e-3),
            ),
            ComputationalCapability(
                name="jeans_length",
                description="Jeans length of an isothermal self-gravitating gas",
                function=lambda density, temperature: jeans.jeans_length(
                    density=density, temperature=temperature) / PC,
                parameters=[
                    ("density|n_h2|n", "cm^-3", "H2 number density"),
                    ("temperature|t_kin|t", "K", "gas kinetic temperature"),
                ],
                returns=("lambda_J", "pc"),
                reference="lambda_J = c_s sqrt(pi / (G rho))",
                test_ref="test_domain_capabilities.py",
                self_check=({"density": 1e4, "temperature": 10.0}, 0.211874, 1e-3),
            ),
            ComputationalCapability(
                name="sound_speed",
                description="Isothermal sound speed",
                function=lambda temperature: jeans.sound_speed(
                    temperature=temperature) / KMS,
                parameters=[("temperature|t_kin|t", "K", "gas temperature")],
                returns=("c_s", "km/s"),
                reference="c_s = sqrt(k_B T / (mu m_H)), mu = 2.33",
                test_ref="test_domain_capabilities.py",
                self_check=({"temperature": 10.0}, 0.18817, 1e-3),
            ),
            ComputationalCapability(
                name="virial_parameter",
                description="Virial parameter of a cloud or clump",
                # NB the backend adds the thermal sound speed in quadrature,
                # so `velocity_dispersion` here is the NON-thermal component.
                function=lambda mass, radius, velocity_dispersion,
                temperature=10.0: virial.virial_parameter(
                    mass_msun=mass, radius_pc=radius,
                    temperature=temperature,
                    sigma_nt_km_s=velocity_dispersion),
                parameters=[
                    ("mass|m", "Msun", "cloud mass"),
                    ("radius|r", "pc", "cloud radius"),
                    ("velocity_dispersion|sigma_nt|sigma", "km/s",
                     "non-thermal 1-D velocity dispersion"),
                ],
                returns=("alpha_vir", "dimensionless"),
                reference="alpha_vir = 5 sigma^2 R / (G M)",
                test_ref="test_domain_capabilities.py",
                self_check=({"mass": 1000.0, "radius": 1.0,
                             "velocity_dispersion": 1.0}, 1.20336, 1e-3),
            ),
            ComputationalCapability(
                name="free_fall_time",
                description="Gravitational free-fall time",
                function=lambda density: np.sqrt(
                    3.0 * np.pi / (32.0 * 6.67430e-8
                                   * density * 2.33 * 1.6735e-24)) / YR,
                parameters=[("density|n_h2|n", "cm^-3", "H2 number density")],
                returns=("t_ff", "yr"),
                reference="t_ff = sqrt(3 pi / (32 G rho)), rho = n mu m_H",
                test_ref="test_domain_capabilities.py",
                self_check=({"density": 1e4}, 337078.0, 1e-3),
            ),
            ComputationalCapability(
                name="bonnor_ebert_mass",
                description="Bonnor-Ebert critical mass of a pressure-confined core",
                function=lambda density: frag.bonnor_ebert_mass(
                    surface_number_density=density) / M_SUN,
                parameters=[("density|n_h2|n", "cm^-3",
                             "surface H2 number density")],
                returns=("M_BE", "Msun"),
                reference="M_BE = 1.182 c_s^3 / (G^(3/2) rho_S^(1/2)), c_s at 10 K",
                test_ref="test_domain_capabilities.py",
                self_check=({"density": 1e4}, 1.16288, 1e-3),
            ),
        ]


def create_statistical_mechanics_domain() -> StatisticalMechanicsDomain:
    """Create a StatisticalMechanicsDomain instance."""
    return StatisticalMechanicsDomain()


try:
    register_domain(StatisticalMechanicsDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
