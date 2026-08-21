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
Accretion Disk Theory Domain Module for ASTRA

Thin/thick disks, magnetorotational instability, jets.

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation.  It now wraps the
routines of :mod:`astra_core.astro_physics.gravitational_collapse` that the
August 2026 physics audit classified REAL and verified exactly ("Jeans /
Bonnor-Ebert / Bondi / Shu / Toomre all verified exact").

NOT wired, deliberately (see `08_domains_batch_C.md`):
  * `next_gen/disk_physics.py` is an 18-line STUB -- imports and a docstring
    promising "viscous evolution, dust dynamics, gap opening", containing
    none of it.  There is therefore no Shakura-Sunyaev alpha-disk, no MRI
    growth rate and no jet-launching calculation anywhere in this codebase,
    and none is invented here.
  * `physics.GalaxyDynamicsModel.circular_velocity` works in the SI constant
    set of `physics.py` while everything else in this domain is CGS; mixing
    the two is the defect class (audit M7) this framework exists to prevent.

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


class AccretionDiskTheoryDomain(ComputationalDomainModule):
    """
    Accretion and disk self-gravity.

    Backed by `astro_physics.gravitational_collapse` (Toomre stability,
    Bondi-Hoyle and Shu inside-out accretion rates).
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="accretion_disk_theory",
            version="2.0.0",
            dependencies=[],
            description="Thin/thick disks, magnetorotational instability, jets",
            keywords=[
                'accretion disk', 'thin disk', 'thick disk', 'mri',
                'jet_formation',
                # extensions matching the wired computations
                'toomre', 'toomre q', 'disk stability',
                'gravitational instability',
                'bondi', 'bondi-hoyle', 'accretion rate', 'shu',
                'inside-out collapse', 'protostellar accretion',
            ],
            capabilities=[
                'toomre_q', 'bondi_accretion_rate',
                'shu_inside_out_accretion_rate',
            ],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising accretion_disk_theory domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.gravitational_collapse import (
            AccretionRates, FragmentationCriterion, M_sun, year,
        )

        # UNITS.  `gravitational_collapse` is CGS throughout.
        #   `FragmentationCriterion.toomre` takes Sigma in g/cm^2, T in K,
        #   Omega in rad/s and a (reported-only) radius in cm, and returns a
        #   dict whose 'Q' is dimensionless.
        #   `AccretionRates.bondi_hoyle` takes M in Msun, a NUMBER density in
        #   cm^-3 and speeds in cm/s, and returns g/s -- the wrappers below
        #   convert to Msun/yr.
        #   `shu_inside_out` returns g/s as well.
        # Every self_check was recomputed from the cited formula in
        # `astra_baseline_audit_aug2026/repro/batchC_hand_values.py`.
        frag = FragmentationCriterion(mu_particle=2.33)
        acc = AccretionRates(mu_particle=2.33)

        return [
            ComputationalCapability(
                name="toomre_q",
                description=("Toomre stability parameter of a rotating "
                             "gaseous disk"),
                function=lambda surface_density, temperature, omega: float(
                    frag.toomre(surface_density=surface_density,
                                temperature=temperature, omega=omega,
                                radius=1.0)['Q']),
                parameters=[
                    ("surface_density|sigma_gas|sigma", "g/cm^2",
                     "disk surface density"),
                    ("temperature|t_mid|t", "K", "midplane temperature"),
                    ("omega|angular_frequency", "rad/s",
                     "orbital angular frequency"),
                ],
                returns=("Q", "dimensionless"),
                reference=("Q = c_s Omega / (pi G Sigma) with "
                           "c_s = sqrt(k_B T / (mu m_H)), mu = 2.33 "
                           "(Toomre 1964); the disk is unstable for Q < 1"),
                test_ref="test_domain_capabilities.py",
                self_check=({"surface_density": 100.0, "temperature": 100.0,
                             "omega": 1e-9}, 2.8379522, 1e-5),
            ),
            ComputationalCapability(
                name="bondi_accretion_rate",
                description=("Bondi-Hoyle accretion rate onto a compact "
                             "object moving through ambient gas"),
                function=lambda mass, density, sound_speed,
                relative_velocity=0.0: acc.bondi_hoyle(
                    mass_msun=mass, number_density=density,
                    sound_speed=sound_speed * 1e5,
                    relative_velocity=relative_velocity * 1e5,
                    lambda_bondi=1.12) * year / M_sun,
                parameters=[
                    ("mass|m_star|m", "Msun", "accretor mass"),
                    ("density|n_h2|n", "cm^-3", "ambient number density"),
                    ("sound_speed|c_s", "km/s", "ambient sound speed"),
                ],
                returns=("Mdot", "Msun/yr"),
                reference=("Mdot = 4 pi lambda G^2 M^2 rho / "
                           "(v^2 + c_s^2)^(3/2), lambda = 1.12 "
                           "(Bondi 1952; Bondi & Hoyle 1944); "
                           "rho = n mu m_H with mu = 2.33"),
                test_ref="test_domain_capabilities.py",
                self_check=({"mass": 1.0, "density": 1e4,
                             "sound_speed": 0.2}, 1.9186398e-5, 1e-5),
            ),
            ComputationalCapability(
                name="shu_inside_out_accretion_rate",
                description=("Shu inside-out collapse accretion rate of a "
                             "singular isothermal sphere"),
                function=lambda temperature: acc.shu_inside_out(
                    temperature=temperature) * year / M_sun,
                parameters=[("temperature|t_kin|t", "K",
                             "envelope temperature")],
                returns=("Mdot", "Msun/yr"),
                reference=("Mdot = 0.975 c_s^3 / G with "
                           "c_s = sqrt(k_B T / (mu m_H)), mu = 2.33 "
                           "(Shu 1977)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"temperature": 10.0}, 1.5443441e-6, 1e-5),
            ),
        ]


# Factory function
def create_accretion_disk_theory_domain() -> AccretionDiskTheoryDomain:
    """Create an Accretion Disk Theory domain instance"""
    return AccretionDiskTheoryDomain()


# Domain registration
try:
    register_domain(AccretionDiskTheoryDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
