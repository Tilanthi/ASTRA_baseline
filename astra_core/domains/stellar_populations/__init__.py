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
Stellar Populations Domain Module for ASTRA

Population synthesis, isochrones, stellar evolution models

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation. It now wraps
:class:`astra_core.astro_physics.star_formation.InitialMassFunction`, the one
part of that module the August 2026 audit verified numerically (the
inverse-CDF sampler reproduces the analytic IMF to 0.05%).

The capabilities are the deterministic moments of the IMF, not the sampler:
a stochastic draw cannot be pinned by a self_check, whereas <m> can be, and
each expected value below was obtained by integrating the module's own
segmented power laws analytically by hand.

Deliberately NOT wired: the `StellarEvolution` prescriptions
(`main_sequence_lifetime`, `main_sequence_radius`, `remnant_mass`,
`wind_mass_loss_rate`). Their own docstring calls them "approximate", the
sub-solar lifetime branch and the black-hole fallback fraction are explicitly
labelled heuristics with no cited source, and the audit did not verify them;
`star_formation`'s iron yields carry an in-code AUDIT-FLAG for being uncapped
(1.87 Msun of Fe from a 100 Msun star). Isochrones, spectral synthesis and
colour-magnitude fitting are not implemented in this codebase at all.

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


class StellarPopulationsDomain(ComputationalDomainModule):
    """
    Initial mass function moments.

    Backed by `astro_physics.star_formation.InitialMassFunction`.
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="stellar_populations",
            version="2.0.0",
            dependencies=[],
            description=("Population synthesis, isochrones, stellar evolution "
                         "models"),
            keywords=['stellar population', 'population synthesis',
                      'isochrone', 'stellar evolution', 'imf',
                      'initial mass function', 'kroupa', 'salpeter',
                      'mean stellar mass', 'cluster mass'],
            capabilities=['population_synthesis', 'isochrone_fitting',
                          'stellar_evolution_tracks', 'imf_modeling'],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising stellar_populations domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.star_formation import InitialMassFunction

        # NOTE ON UNITS. The IMF works in Msun throughout; nothing is
        # converted here. The self_check values are the analytic ratios
        # int m xi(m) dm / int xi(m) dm over the quoted range, evaluated by
        # hand from the module's own segmented power laws:
        #   Kroupa   0.08-120 : 0.5795 Msun
        #   Salpeter 0.1-100  : 0.3514 Msun
        def _mean(form: str, m_min: float, m_max: float) -> float:
            return float(InitialMassFunction(form=form, m_min=m_min,
                                             m_max=m_max).mean_mass())

        return [
            ComputationalCapability(
                name="kroupa_mean_stellar_mass",
                description=("Mean stellar mass of a Kroupa (2001) initial "
                             "mass function"),
                function=lambda m_min, m_max: _mean('kroupa', m_min, m_max),
                parameters=[
                    ("m_min|lower_mass", "Msun", "lower mass limit"),
                    ("m_max|upper_mass", "Msun", "upper mass limit"),
                ],
                returns=("<m>", "Msun"),
                reference=("<m> = int m xi dm / int xi dm with xi ~ m^-1.3 "
                           "(0.08-0.5) and m^-2.3 (>0.5), continuous at 0.5 "
                           "(Kroupa 2001)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"m_min": 0.08, "m_max": 120.0}, 0.5794712, 1e-3),
            ),
            ComputationalCapability(
                name="salpeter_mean_stellar_mass",
                description=("Mean stellar mass of a Salpeter (1955) initial "
                             "mass function"),
                function=lambda m_min, m_max: _mean('salpeter', m_min, m_max),
                parameters=[
                    ("m_min|lower_mass", "Msun", "lower mass limit"),
                    ("m_max|upper_mass", "Msun", "upper mass limit"),
                ],
                returns=("<m>", "Msun"),
                reference=("<m> = int m^-1.35 dm / int m^-2.35 dm "
                           "(Salpeter 1955, dN/dM ~ M^-2.35)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"m_min": 0.1, "m_max": 100.0}, 0.35136878, 1e-3),
            ),
            ComputationalCapability(
                name="cluster_birth_mass",
                description=("Expected total birth mass of N stars drawn from "
                             "a Kroupa IMF over 0.08-120 Msun"),
                function=lambda n_stars: float(
                    InitialMassFunction(form='kroupa', m_min=0.08,
                                        m_max=120.0)
                    .total_mass_to_n_stars(n_stars)),
                parameters=[("n_stars|n", "dimensionless",
                             "number of stars drawn")],
                returns=("M_total", "Msun"),
                reference="M_total = N <m>, <m> = 0.5795 Msun (Kroupa 2001)",
                test_ref="test_domain_capabilities.py",
                self_check=({"n_stars": 1000.0}, 579.4712, 1e-3),
            ),
        ]


def create_stellar_populations_domain() -> StellarPopulationsDomain:
    """Create a Stellar Populations domain instance."""
    return StellarPopulationsDomain()


try:
    register_domain(StellarPopulationsDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
