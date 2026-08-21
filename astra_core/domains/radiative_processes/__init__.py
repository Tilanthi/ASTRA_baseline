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
Radiative Processes Domain Module for ASTRA

Bremsstrahlung, synchrotron, Compton scattering, line processes

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation. It now wraps the
synchrotron and thermal-bremsstrahlung routines of
:mod:`astra_core.astro_physics.supernova_remnant_physics` (synchrotron cooling
time, DSA spectral-index relation and free-free emissivity were verified in
the August 2026 audit) and the fine-structure line luminosity of
:mod:`astra_core.astro_physics.infrared_submm` (verified to 0.3% against the
standard 1.04e-3 S dV nu_GHz D^2 relation).

Every capability carries a `self_check` recomputed by hand from the formula in
its `reference` field.

Deliberately NOT wired: `SynchrotronEmission.critical_frequency`, which is
exactly a factor 2 below its own docstring and below the standard
nu_c = 4.2e6 B gamma^2 (audit B-SNR-5, still unrepaired);
`SynchrotronEmission.surface_brightness` (an unnormalised dimensional estimate
with a hardcoded 1% acceleration efficiency); and
`XRayThermalEmission.cooling_function`, whose absolute normalisation could not
be established offline and which carries an in-code AUDIT-FLAG saying it must
not be used quantitatively. Compton/inverse-Compton scattering has no
implementation in this codebase at all, so no Compton capability is offered.

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


class RadiativeProcessesDomain(ComputationalDomainModule):
    """
    Continuum and line radiation mechanisms.

    Backed by `astro_physics.supernova_remnant_physics` (synchrotron cooling,
    radio spectral index, thermal bremsstrahlung emissivity) and
    `astro_physics.infrared_submm` (fine-structure line luminosity).
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="radiative_processes",
            version="2.0.0",
            dependencies=[],
            description=("Bremsstrahlung, synchrotron, Compton scattering, "
                         "line processes"),
            keywords=['bremsstrahlung', 'synchrotron', 'compton',
                      'radiation_mechanisms', 'emission_processes',
                      'free-free', 'cooling time', 'spectral index',
                      'fine-structure line', 'cii', 'line cooling'],
            capabilities=['bremsstrahlung_emission', 'synchrotron_emission',
                          'compton_scattering', 'line_emission'],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising radiative_processes domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.infrared_submm import LineCooling
        from ...astro_physics.supernova_remnant_physics import (
            SynchrotronEmission,
            XRayThermalEmission,
        )

        # NOTE ON UNITS. `cooling_time` returns seconds for an electron
        # energy in erg and B in gauss; `bremsstrahlung_emissivity` returns
        # erg/s/cm^3; `line_luminosity` returns Lsun for a flux in Jy km/s and
        # a distance in Mpc. The only conversion applied here is s -> yr, and
        # its self_check pins it.
        YR = 3.155693e7        # s

        syn = SynchrotronEmission()
        xray = XRayThermalEmission()
        lines = LineCooling()

        return [
            ComputationalCapability(
                name="synchrotron_cooling_time",
                description=("Synchrotron cooling time of a relativistic "
                             "electron"),
                function=lambda electron_energy, magnetic_field: float(
                    syn.cooling_time(electron_energy, magnetic_field)) / YR,
                parameters=[
                    ("electron_energy|energy|e_electron", "erg",
                     "electron total energy"),
                    ("magnetic_field|b_field", "G", "magnetic field strength"),
                ],
                returns=("t_cool", "yr"),
                reference=("t_cool = 3 m_e c / (4 sigma_T U_B gamma) "
                           "= 6 pi m_e c / (sigma_T B^2 gamma), "
                           "U_B = B^2/8pi (Rybicki & Lightman 1979)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"electron_energy": 8.18710578e-3,
                             "magnetic_field": 1e-4}, 245207.577, 1e-3),
            ),
            ComputationalCapability(
                name="synchrotron_spectral_index",
                description=("Radio spectral index from the electron energy "
                             "spectrum index"),
                function=lambda electron_index: float(
                    syn.radio_spectral_index(electron_index)),
                parameters=[("electron_index|p", "dimensionless",
                             "electron spectrum index p in N(E) ~ E^-p")],
                returns=("alpha", "dimensionless"),
                reference="S_nu ~ nu^-alpha with alpha = (p - 1)/2",
                test_ref="test_domain_capabilities.py",
                self_check=({"electron_index": 2.2}, 0.6, 1e-9),
            ),
            ComputationalCapability(
                name="bremsstrahlung_emissivity",
                description=("Thermal free-free volume emissivity of an "
                             "ionized plasma"),
                function=lambda temperature, n_e: float(
                    xray.bremsstrahlung_emissivity(temperature, n_e)),
                parameters=[
                    ("temperature|t_e", "K", "plasma temperature"),
                    ("n_e|electron_density", "cm^-3", "electron density"),
                ],
                returns=("eps_ff", "erg/s/cm^3"),
                reference=("eps_ff = 1.4e-27 T^0.5 n_e^2 g_ff, g_ff = 1.2 "
                           "(Rybicki & Lightman 1979, eq. 5.15b)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"temperature": 1e7, "n_e": 1.0},
                            5.31262647e-24, 1e-3),
            ),
            ComputationalCapability(
                name="cii_line_luminosity",
                description=("[CII] 158 micron line luminosity from the "
                             "velocity-integrated line flux"),
                function=lambda integrated_flux, distance: float(
                    lines.line_luminosity('CII_158', integrated_flux,
                                          distance)),
                parameters=[
                    ("integrated_flux|s_int", "Jy km/s",
                     "velocity-integrated line flux"),
                    ("distance|d_l", "Mpc", "luminosity distance"),
                ],
                returns=("L([CII])", "Lsun"),
                reference=("L = 4 pi D^2 S_int nu / c, nu = c/157.741 um "
                           "(equivalent to 1.04e-3 S dV nu_GHz D_Mpc^2 Lsun)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"integrated_flux": 100.0, "distance": 10.0},
                            19815.0164, 1e-3),
            ),
        ]


def create_radiative_processes_domain() -> RadiativeProcessesDomain:
    """Create a Radiative Processes domain instance."""
    return RadiativeProcessesDomain()


try:
    register_domain(RadiativeProcessesDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
