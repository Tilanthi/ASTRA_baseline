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
Galaxy Clusters Domain Module for ASTRA

Intracluster medium, galaxy evolution in clusters, cluster cosmology.

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation.  It now wraps the
linear growth factor of
:mod:`astra_core.astro_physics.next_gen.cosmological_context` (rewritten in
this branch as an exact quadrature after audit B-COS-2 found the growth
*index* being used as a multiplicative growth *factor*), the Dutton & Maccio
concentration-mass relation, the virial-mass estimator, and the thermal
bremsstrahlung emissivity from
:mod:`astra_core.astro_physics.supernova_remnant_physics` (verified by the
audit).

NOT wired, deliberately (see `08_domains_batch_C.md`):
  * `HaloMassFunction.press_schechter` / `sheth_tormen` / `tinker` /
    `cumulative_number_density`.  The f(nu) normalisations are correct (the
    audit verified them) but they all rest on ``sigma(M) = sigma_8
    (M/M_8)^(-1/6)``, a toy power law, and its M_8 drops the h^2 that
    RHO_CRIT_0 carries -- an UNFIXED defect of the same class as audit H4.
    An absolute cluster number density from that would look like a
    prediction and is not one.
  * `CGMModel._virial_radius` -- ``rho_crit = RHO_CRIT_0 * 1e-9`` likewise
    drops the h^2 (and any E(z)), so R_vir is ~1.27x too small at h = 0.7 and
    T_vir ~1.6x too high.  UNFIXED; reported rather than wired.
  * Sunyaev-Zel'dovich signal and cluster lensing: no implementation here
    (the lensing module's own SIE/NFW defects were repaired in this branch,
    but that is the `gravitational_lensing` domain's business, not this one).

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


class GalaxyClustersDomain(ComputationalDomainModule):
    """
    Galaxy clusters: growth of structure, halo structure and ICM emission.
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="galaxy_clusters",
            version="2.0.0",
            dependencies=[],
            description=("Intracluster medium, galaxy evolution in clusters, "
                         "cluster cosmology"),
            keywords=[
                'galaxy cluster', 'intracluster medium', 'icm',
                'cluster cosmology', 'sunyaev-zeldovich',
                # extensions matching the wired computations
                'growth factor', 'linear growth', 'structure formation',
                'concentration', 'nfw', 'virial mass', 'velocity dispersion',
                'bremsstrahlung', 'free-free', 'x-ray emissivity',
            ],
            capabilities=[
                'linear_growth_factor', 'nfw_concentration',
                'virial_mass_from_velocity', 'bremsstrahlung_emissivity',
            ],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising galaxy_clusters domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.next_gen.cosmological_context import (
            CosmologyParams, GalaxyHaloConnection, HaloMassFunction,
            virial_mass_from_velocity,
        )
        from ...astro_physics.supernova_remnant_physics import (
            XRayThermalEmission,
        )

        # UNITS.  `cosmological_context` works in astronomer units:
        # masses in Msun, distances in Mpc/kpc, velocities in km/s; the
        # growth factor and concentration are dimensionless.
        # `XRayThermalEmission.bremsstrahlung_emissivity` is CGS and returns
        # erg/s/cm^3 for T in K and n_e in cm^-3.
        # `virial_mass_from_velocity` is FIXED at the module's fiducial
        # cosmology (H0 = 70, Om = 0.3, OL = 0.7) -- it reads the module
        # globals, not `self.cosmo` -- which is documented in the reference
        # string below rather than hidden.
        # Every self_check was recomputed from the cited formula in
        # `astra_baseline_audit_aug2026/repro/batchC_hand_values.py`.
        cosmo = CosmologyParams()          # H0 = 70, Om = 0.3, OL = 0.7
        hmf = HaloMassFunction(cosmo)
        ghc = GalaxyHaloConnection(cosmo)
        xray = XRayThermalEmission()

        return [
            ComputationalCapability(
                name="linear_growth_factor",
                description=("Linear density-perturbation growth factor of a "
                             "flat LCDM universe"),
                function=lambda redshift: float(hmf.growth_factor(z=redshift)),
                parameters=[("redshift|z", "dimensionless", "redshift")],
                returns=("D(z)", "dimensionless"),
                reference=("D(a) prop. E(a) Int_0^a da'/(a' E(a'))^3, "
                           "normalised to D(0) = 1, with "
                           "E(a) = sqrt(Om a^-3 + OL) for Om = 0.3, OL = 0.7 "
                           "(Heath 1977; Peebles 1980 sec. 11)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"redshift": 2.0}, 0.4214457, 1e-4),
            ),
            ComputationalCapability(
                name="nfw_concentration",
                description=("NFW halo concentration from the "
                             "concentration-mass relation"),
                function=lambda halo_mass, redshift: float(
                    ghc.concentration_mass_relation(M_halo=halo_mass,
                                                    z=redshift)),
                parameters=[
                    ("halo_mass|m_halo|m_vir", "Msun/h",
                     "halo mass M_200c in h^-1 Msun"),
                    ("redshift|z", "dimensionless", "redshift"),
                ],
                returns=("c_200", "dimensionless"),
                reference=("log10 c = a + b (log10 M - 12) with "
                           "a = 0.520 + 0.385 exp(-0.617 z^1.21), "
                           "b = -0.101 + 0.026 z (Dutton & Maccio 2014 eq. 8; "
                           "M_200c in units of 1e12 h^-1 Msun)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"halo_mass": 1e14, "redshift": 0.0},
                            5.0466130, 1e-5),
            ),
            ComputationalCapability(
                name="virial_mass_from_velocity",
                description=("Halo virial mass estimated from the circular "
                             "velocity"),
                function=lambda circular_velocity, redshift: float(
                    virial_mass_from_velocity(v_c=circular_velocity,
                                              z=redshift)),
                parameters=[
                    ("circular_velocity|v_circ|v_c", "km/s",
                     "circular velocity at the virial radius"),
                    ("redshift|z", "dimensionless", "redshift"),
                ],
                returns=("M_vir", "Msun"),
                reference=("M_vir = v_c^3 / (10 G H(z)) (Mo, Mao & White 1998 "
                           "eq. 2), evaluated at the module's fiducial "
                           "H0 = 70 km/s/Mpc, Om = 0.3, OL = 0.7"),
                test_ref="test_domain_capabilities.py",
                self_check=({"circular_velocity": 1000.0, "redshift": 0.0},
                            3.3207146e14, 1e-5),
            ),
            ComputationalCapability(
                name="bremsstrahlung_emissivity",
                description=("Thermal free-free volume emissivity of the "
                             "intracluster plasma"),
                function=lambda temperature, electron_density: xray.
                bremsstrahlung_emissivity(temperature=temperature,
                                          n_e=electron_density),
                parameters=[
                    ("temperature|t_x|t", "K", "plasma temperature"),
                    ("electron_density|n_e", "cm^-3", "electron density"),
                ],
                returns=("epsilon_ff", "erg/s/cm^3"),
                reference=("eps_ff = 1.4e-27 T^(1/2) n_e n_i g_ff with "
                           "n_i = n_e and a mean Gaunt factor g_ff = 1.2 "
                           "(Rybicki & Lightman 1979 eq. 5.15b)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"temperature": 1e8, "electron_density": 1e-3},
                            1.68e-29, 1e-9),
            ),
        ]


# Factory function
def create_galaxy_clusters_domain() -> GalaxyClustersDomain:
    """Create a Galaxy Clusters domain instance"""
    return GalaxyClustersDomain()


# Domain registration
try:
    register_domain(GalaxyClustersDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
