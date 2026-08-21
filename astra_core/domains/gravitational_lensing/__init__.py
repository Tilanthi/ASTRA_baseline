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
Gravitational Lensing Domain Module for ASTRA

Strong lensing, weak lensing, microlensing, lensing-based cosmology,
time-delay cosmography.

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation. It now wraps the
flat-LambdaCDM distance layer and the mass profiles of
:mod:`astra_core.astro_physics.advanced_lensing`, and reports a confidence
derived from whether a computation actually ran.

Scope of what is wired
----------------------
Only the parts of `advanced_lensing` that were independently verified are
exposed: the comoving/angular-diameter distances, `D_ds`, `Sigma_crit` and
the (post-audit-fix) NFW `rho_crit` normalisation.  The SIE/NFW convergence
and deflection fields are self-consistent again after audit fixes C6-C8, but
they are field-valued rather than scalar and are not exposed as scalar
capabilities here.  `TimeDelayCosmography.infer_H0` is NOT wired: audit H5
(the returned "km/s/Mpc" uncertainty is in days) is still open.

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


class GravitationalLensingDomain(ComputationalDomainModule):
    """
    Gravitational lensing.

    Backed by `astro_physics.advanced_lensing`, whose flat-LambdaCDM
    distances and critical surface density were reproduced to 5 digits by
    independent quadrature during the August 2026 audit, and whose NFW
    virial radius was corrected (audit H4: the h^2 in rho_crit,0 had been
    dropped, making r_200 for a 1e14 Msun halo 681 kpc instead of 864 kpc).
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="gravitational_lensing",
            version="2.0.0",
            dependencies=[],
            description=("Strong lensing, weak lensing, microlensing, "
                         "lensing-based cosmology, time-delay cosmography"),
            keywords=[
                "gravitational lensing", "strong lensing", "weak lensing",
                "microlensing", "cosmology", "einstein radius",
                "critical surface density", "sigma_crit", "convergence",
                "time delay distance", "nfw", "virial radius", "lens",
            ],
            capabilities=[
                "critical_surface_density", "angular_diameter_distance",
                "time_delay_distance", "einstein_radius_sis",
                "nfw_virial_radius",
            ],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising gravitational_lensing domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.advanced_lensing import Cosmology, NFWProfile

        # Default flat LambdaCDM: H0 = 70, Omega_m = 0.3, Omega_L = 0.7.
        cosmo = Cosmology()

        # NOTE ON UNITS. `advanced_lensing` works in (km/s, Mpc, Msun):
        # `angular_diameter_distance` -> Mpc, `sigma_crit` -> Msun/kpc^2,
        # `time_delay_distance` -> Mpc, `NFWProfile.r_200` -> kpc. Those are
        # already the astronomer-facing units declared below, so no rescaling
        # is applied -- but every capability still carries a `self_check`
        # recomputed by hand (independent quadrature of 1/E(z), CODATA c and
        # G) so that a change of unit in the backend fails a test.
        ARCSEC_PER_RAD = 206264.806
        C_KM_S = 299792.458

        def einstein_radius_sis(velocity_dispersion: float,
                                z_lens: float, z_source: float) -> float:
            """theta_E = 4 pi (sigma_v/c)^2 D_ds/D_s, in arcsec."""
            d_s = cosmo.angular_diameter_distance(z_source)
            d_ds = cosmo.D_ds(z_lens, z_source)
            theta_rad = (4.0 * np.pi * (velocity_dispersion / C_KM_S) ** 2
                         * d_ds / d_s)
            return float(theta_rad * ARCSEC_PER_RAD)

        def nfw_virial_radius(m_200: float, z_lens: float) -> float:
            """r_200 in kpc. Independent of the source redshift; a source is
            still required to build the profile, so one is placed behind the
            lens at z_lens + 1."""
            profile = NFWProfile(M_200=m_200, c=5.0, z_lens=z_lens,
                                 z_source=z_lens + 1.0, cosmo=cosmo)
            return float(profile.r_200)

        return [
            ComputationalCapability(
                name="critical_surface_density",
                description="Critical surface density for strong lensing",
                function=lambda z_lens, z_source: float(
                    cosmo.sigma_crit(z_lens, z_source)),
                parameters=[
                    ("z_lens|z_l|zl", "dimensionless", "lens redshift"),
                    ("z_source|z_s|zs", "dimensionless", "source redshift"),
                ],
                returns=("Sigma_crit", "Msun/kpc^2"),
                reference=("Sigma_crit = c^2/(4 pi G) * D_s/(D_d D_ds) "
                           "(Schneider, Ehlers & Falco 1992)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"z_lens": 0.5, "z_source": 2.0},
                            2.07862e9, 1e-3),
            ),
            ComputationalCapability(
                name="angular_diameter_distance",
                description="Angular diameter distance in flat LambdaCDM",
                function=lambda redshift: float(
                    cosmo.angular_diameter_distance(redshift)),
                parameters=[("redshift|z", "dimensionless", "redshift")],
                returns=("D_A", "Mpc"),
                reference=("D_A = (c/H0)/(1+z) int_0^z dz'/E(z'), "
                           "E = sqrt(Om (1+z)^3 + OL); H0 = 70, Om = 0.3"),
                test_ref="test_domain_capabilities.py",
                self_check=({"redshift": 0.5}, 1259.084, 1e-3),
            ),
            ComputationalCapability(
                name="time_delay_distance",
                description="Time-delay distance for time-delay cosmography",
                function=lambda z_lens, z_source: float(
                    cosmo.time_delay_distance(z_lens, z_source)),
                parameters=[
                    ("z_lens|z_l|zl", "dimensionless", "lens redshift"),
                    ("z_source|z_s|zs", "dimensionless", "source redshift"),
                ],
                returns=("D_dt", "Mpc"),
                reference="D_dt = (1+z_d) D_d D_s / D_ds (Refsdal 1964)",
                test_ref="test_domain_capabilities.py",
                self_check=({"z_lens": 0.5, "z_source": 2.0},
                            2972.384, 1e-3),
            ),
            ComputationalCapability(
                name="einstein_radius_sis",
                description=("Einstein radius of a singular isothermal "
                             "sphere lens"),
                function=einstein_radius_sis,
                parameters=[
                    ("velocity_dispersion|sigma_v|sigma", "km/s",
                     "1-D stellar velocity dispersion of the lens"),
                    ("z_lens|z_l|zl", "dimensionless", "lens redshift"),
                    ("z_source|z_s|zs", "dimensionless", "source redshift"),
                ],
                returns=("theta_E", "arcsec"),
                reference=("theta_E = 4 pi (sigma_v/c)^2 D_ds/D_s "
                           "(Narayan & Bartelmann 1996, eq. 22)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"velocity_dispersion": 250.0, "z_lens": 0.5,
                             "z_source": 2.0}, 1.145288, 1e-3),
            ),
            ComputationalCapability(
                name="nfw_virial_radius",
                description="Virial radius r_200 of an NFW dark matter halo",
                function=nfw_virial_radius,
                parameters=[
                    ("m_200|mass|m", "Msun", "halo mass within r_200"),
                    ("z_lens|redshift|z", "dimensionless", "halo redshift"),
                ],
                returns=("r_200", "kpc"),
                reference=("r_200 = [3 M_200 / (4 pi 200 rho_crit(z))]^(1/3), "
                           "rho_crit(z) = 2.775e11 h^2 E(z)^2 Msun/Mpc^3"),
                test_ref="test_domain_capabilities.py",
                self_check=({"m_200": 1e14, "z_lens": 0.3}, 864.418, 1e-3),
            ),
        ]


def create_gravitational_lensing_domain() -> GravitationalLensingDomain:
    """Create a GravitationalLensingDomain instance."""
    return GravitationalLensingDomain()


# Domain registration
try:
    register_domain(GravitationalLensingDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
