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
Atomic Physics Domain Module for ASTRA

Energy levels, transitions, collisional excitation, ionization

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation. It now wraps
:mod:`astra_core.astro_physics.spectral_line_analysis` (the LTE column-density
relation of Mangum & Shirley 2015 eq. 80 and the brightness-temperature
conversion J(T); the column density was wrong by a factor 1.4388e-5 before
audit item C14) and the hydrogen recombination coefficient of
:mod:`astra_core.astro_physics.hii_region_physics`.

Every capability carries a `self_check` recomputed by hand from the formula in
its `reference` field.

Deliberately NOT wired: `ColumnDensityCalculator.h2_from_13co` and
`c18o_column`, which use E_u/k = 5.87 K for a level whose energy is 5.289 K
and an inverted Boltzmann sign (audit B2.3/B2.8, flagged in-code and not
repaired); the fabricated hyperfine-component tables in the same module
(B2.4/B2.5); and the synthesised collisional rate coefficients in
`spectroscopic_databases` (~14x low for CO 1-0 at 10 K, with the Boltzmann
factor applied to a downward rate). Consequently no collisional-excitation or
ionization-balance capability is offered: this codebase contains no verified
electron-impact excitation rates, no Saha solver and no photoionization
cross-sections.

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


class AtomicPhysicsDomain(ComputationalDomainModule):
    """
    Level populations and transition diagnostics.

    Backed by `astro_physics.spectral_line_analysis` and
    `astro_physics.hii_region_physics`.
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="atomic_physics",
            version="2.0.0",
            dependencies=[],
            description=("Energy levels, transitions, collisional excitation, "
                         "ionization"),
            keywords=['atomic physics', 'energy_levels', 'transitions',
                      'collisional_excitation', 'ionization',
                      'column density', 'lte', 'level populations',
                      'brightness temperature', 'recombination'],
            capabilities=['atomic_structure', 'transition_rates',
                          'collisional_processes', 'ionization_balance'],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising atomic_physics domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.hii_region_physics import (
            RecombinationCoefficients,
        )
        from ...astro_physics.spectral_line_analysis import (
            ColumnDensityCalculator,
            OpticalDepthCorrector,
        )

        # NOTE ON UNITS. `lte_column` takes the integrated intensity in
        # K km/s and the frequency in GHz and returns cm^-2 (it converts to
        # CGS internally); `j_nu` takes GHz and returns K. No conversion is
        # applied here, and the self_checks pin that: the CO 1-0 reference
        # case returns 9.28e15 cm^-2, not the 1.335e11 the pre-C14 code gave.
        cdc = ColumnDensityCalculator()
        recomb = RecombinationCoefficients()

        return [
            ComputationalCapability(
                name="lte_column_density",
                description=("Optically thin LTE column density from an "
                             "integrated line intensity"),
                function=lambda integrated_intensity, frequency, einstein_a,
                t_ex, e_upper, g_upper, partition_function: float(
                    cdc.lte_column(integrated_intensity, frequency,
                                   einstein_a, t_ex, e_upper, g_upper,
                                   1.0, partition_function)),
                parameters=[
                    ("integrated_intensity|w", "K km/s",
                     "velocity-integrated main-beam brightness temperature"),
                    ("frequency|nu", "GHz", "line rest frequency"),
                    ("einstein_a|a_ul", "s^-1", "Einstein A coefficient"),
                    ("t_ex|excitation_temperature", "K",
                     "excitation temperature"),
                    ("e_upper|e_u", "K", "upper level energy E_u/k"),
                    ("g_upper|g_u", "dimensionless",
                     "upper level degeneracy"),
                    ("partition_function|q_rot", "dimensionless",
                     "partition function at T_ex"),
                ],
                returns=("N_tot", "cm^-2"),
                reference=("N = [8 pi nu^3/(c^3 A_ul)] (Q/g_u) exp(E_u/kT_ex) "
                           "[exp(h nu/kT_ex)-1]^-1 W / [J(T_ex)-J(T_bg)] "
                           "(Mangum & Shirley 2015, eq. 80)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"integrated_intensity": 10.0,
                             "frequency": 115.2712, "einstein_a": 7.203e-8,
                             "t_ex": 10.0, "e_upper": 5.53, "g_upper": 3.0,
                             "partition_function": 3.968},
                            9.28124163e15, 1e-3),
            ),
            ComputationalCapability(
                name="brightness_temperature",
                description=("Rayleigh-Jeans equivalent brightness "
                             "temperature J(T) of a transition"),
                function=lambda temperature, frequency: float(
                    OpticalDepthCorrector.j_nu(temperature, frequency)),
                parameters=[
                    ("temperature|t_ex", "K", "excitation temperature"),
                    ("frequency|nu", "GHz", "line rest frequency"),
                ],
                returns=("J(T)", "K"),
                reference="J(T) = (h nu / k) / [exp(h nu / k T) - 1]",
                test_ref="test_domain_capabilities.py",
                self_check=({"temperature": 10.0, "frequency": 115.2712},
                            7.48767455, 1e-3),
            ),
            ComputationalCapability(
                name="case_a_recombination_rate",
                description=("Case A hydrogen radiative recombination "
                             "coefficient alpha_A(T)"),
                function=lambda temperature: float(
                    recomb.alpha_A(temperature)),
                parameters=[("temperature|t_e", "K",
                             "electron temperature")],
                returns=("alpha_A", "cm^3/s"),
                reference=("alpha_A = 4.18e-13 (T/1e4)^-0.72 "
                           "(Osterbrock & Ferland 2006)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"temperature": 5000.0}, 6.8852209e-13, 1e-3),
            ),
        ]


def create_atomic_physics_domain() -> AtomicPhysicsDomain:
    """Create a Atomic Physics domain instance."""
    return AtomicPhysicsDomain()


try:
    register_domain(AtomicPhysicsDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
