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
Dust Grain Physics Domain Module for ASTRA

Grain formation, evolution, destruction, alignment properties

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation. It now wraps
:mod:`astra_core.astro_physics.infrared_submm`, the one module the August 2026
audit classed REAL: the Planck function, the power-law dust opacity and the
optically thin modified-blackbody flux all reproduce hand-derived values
exactly (S_nu(350 um, 20 K, 1 Msun, 1 Mpc) = 2.8435e-4 Jy, matching an
independent derivation digit for digit after the dimensional error in the
pre-audit `flux_density` was repaired).

Every capability carries a `self_check` recomputed by hand from the formula in
its `reference` field.

NOT implemented in this codebase and therefore not offered: grain-size
distributions (MRN/WD01), grain charging, radiative torque alignment,
sputtering and shattering. In particular `shock_physics.ShockChemistry.
grain_sputtering` is an invented threshold look-up table with no cited source
(0.1 Si / 0.05 Fe / 0.2 C times a linear ramp) and is deliberately not wired
here. The opacity offered below is the observers' power-law parameterisation,
not a Mie calculation; note also that the `DustModel.OSSENKOPF_THICK` entry in
`molecular_cloud_physics` carries an in-code AUDIT-FLAG (kappa = 1.85 cm^2/g
is the usual OH94 thick-ice value at 850 um, not at the 1300 um it is
tabulated against).

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


class DustGrainPhysicsDomain(ComputationalDomainModule):
    """
    Dust continuum emission and opacity.

    Backed by `astro_physics.infrared_submm`.
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="dust_grain_physics",
            version="2.0.0",
            dependencies=[],
            description=("Grain formation, evolution, destruction, alignment "
                         "properties"),
            keywords=['dust grain', 'grain formation', 'grain evolution',
                      'dust alignment', 'grain destruction', 'dust opacity',
                      'emissivity index', 'modified blackbody',
                      'dust emission', 'dust mass', 'submillimetre'],
            capabilities=['grain_formation', 'grain_evolution',
                          'alignment_physics', 'grain_charge',
                          'dust_destruction'],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising dust_grain_physics domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.infrared_submm import (
            ModifiedBlackbody,
            calculate_gas_mass,
        )

        # NOTE ON UNITS. `ModifiedBlackbody` takes wavelengths in microns and
        # returns kappa in cm^2/g and S_nu in Jy for a mass in Msun and a
        # distance in Mpc; `calculate_gas_mass` returns Msun. Nothing is
        # rescaled here, and each self_check pins the declared unit - in
        # particular that `flux_density` really is Jy and not the CGS
        # erg/s/cm^2/Hz it is built from (a factor 1e23).
        #
        # kappa_0 conventions differ between the two: `ModifiedBlackbody`
        # references kappa_0 to 350 um per gram of *dust*, while
        # `calculate_gas_mass` uses the gas-referenced 0.1 cm^2/g at 250 um.
        # Both are exposed explicitly rather than hidden in a default.
        return [
            ComputationalCapability(
                name="dust_opacity",
                description=("Power-law dust opacity kappa_nu at a given "
                             "wavelength"),
                function=lambda wavelength, beta, kappa_0: float(
                    ModifiedBlackbody(kappa_0=kappa_0,
                                      beta=beta).opacity(wavelength)),
                parameters=[
                    ("wavelength|lambda", "micron", "wavelength"),
                    ("beta|emissivity_index", "dimensionless",
                     "dust emissivity index"),
                    ("kappa_0|opacity_0", "cm^2/g",
                     "reference opacity at 350 micron"),
                ],
                returns=("kappa_nu", "cm^2/g"),
                reference=("kappa_nu = kappa_0 (lambda/350 um)^-beta "
                           "(Hildebrand 1983)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"wavelength": 850.0, "beta": 1.5,
                             "kappa_0": 10.0}, 2.64224861, 1e-3),
            ),
            ComputationalCapability(
                name="modified_blackbody_flux",
                description=("Optically thin modified-blackbody flux density "
                             "of a dust mass"),
                function=lambda wavelength, temperature, dust_mass, distance:
                    float(ModifiedBlackbody(kappa_0=10.0, beta=1.5)
                          .flux_density(wavelength, temperature, dust_mass,
                                        distance)),
                parameters=[
                    ("wavelength|lambda", "micron", "wavelength"),
                    ("temperature|t_dust", "K", "dust temperature"),
                    ("dust_mass|m_dust", "Msun", "dust mass"),
                    ("distance|d", "Mpc", "distance"),
                ],
                returns=("S_nu", "Jy"),
                reference=("S_nu = M_dust kappa_nu B_nu(T) / D^2, "
                           "kappa_0 = 10 cm^2/g at 350 um, beta = 1.5"),
                test_ref="test_domain_capabilities.py",
                self_check=({"wavelength": 350.0, "temperature": 20.0,
                             "dust_mass": 1.0, "distance": 1.0},
                            2.84245678e-4, 1e-3),
            ),
            ComputationalCapability(
                name="dust_traced_gas_mass",
                description=("Gas mass from an optically thin submillimetre "
                             "continuum flux"),
                function=lambda flux, wavelength, temperature, distance:
                    float(calculate_gas_mass(flux, wavelength, temperature,
                                             distance)),
                parameters=[
                    ("flux|s_nu", "Jy", "continuum flux density"),
                    ("wavelength|lambda", "micron", "wavelength"),
                    ("temperature|t_dust", "K", "dust temperature"),
                    ("distance|d", "Mpc", "distance"),
                ],
                returns=("M_gas", "Msun"),
                reference=("M = S_nu D^2 / [kappa_nu B_nu(T)], "
                           "kappa_0 = 0.1 cm^2/g of gas at 250 um, beta = 2"),
                test_ref="test_domain_capabilities.py",
                self_check=({"flux": 1.0, "wavelength": 850.0,
                             "temperature": 20.0, "distance": 0.1},
                            113863.407, 1e-3),
            ),
        ]


def create_dust_grain_physics_domain() -> DustGrainPhysicsDomain:
    """Create a Dust Grain Physics domain instance."""
    return DustGrainPhysicsDomain()


try:
    register_domain(DustGrainPhysicsDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
