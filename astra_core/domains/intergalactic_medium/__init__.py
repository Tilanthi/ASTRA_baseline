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
Intergalactic Medium Domain Module for ASTRA

IGM heating/cooling, reionization, quasar absorption lines.

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation.  It now wraps

  * the flat-LCDM distance and lookback-time integrals of
    :mod:`astra_core.astro_physics.physics` (the audit reproduced these
    "to 5+ digits" against independent quadrature), and
  * `ReionizationModel.optical_depth_reionization` from
    :mod:`astra_core.astro_physics.next_gen.cosmological_context`, which was
    wrong by 1.7e40 (audit C11: both the Msun -> g and the Mpc^3 -> cm^3
    conversions were missing) and was repaired in this branch.

All of them are evaluated at the H0 = 70, Om = 0.3, OL = 0.7 cosmology that
`cosmological_context` uses, so that the distances and tau_reion this domain
reports are mutually consistent; `physics.CosmologicalModel` would otherwise
default to Planck 2018 (H0 = 67.4, Om = 0.315).

NOT wired, deliberately (see `08_domains_batch_C.md`):
  * `ReionizationModel.uv_background` -- a four-branch piecewise "simplified
    HM12 fit" that is discontinuous at every knot (J_21 jumps 0.5 -> 0.2 at
    z = 3) and whose constants appear in no cited table.
  * `CGMModel.column_density` -- built on `_virial_radius`, which drops the
    h^2 carried by RHO_CRIT_0 (unfixed, same class as audit H4), and on a
    hardcoded central density n_0 = 1e-3 cm^-3.
  * Lyman-alpha forest statistics: nothing in this codebase computes them.

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


class IntergalacticMediumDomain(ComputationalDomainModule):
    """
    Intergalactic medium: cosmological distances and reionization optical
    depth.
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="intergalactic_medium",
            version="2.0.0",
            dependencies=[],
            description=("IGM heating/cooling, reionization, quasar "
                         "absorption lines"),
            keywords=[
                'igm', 'intergalactic medium', 'reionization', 'lya forest',
                'quasar absorption',
                # extensions matching the wired computations
                'optical depth', 'thomson scattering', 'comoving distance',
                'luminosity distance', 'angular diameter distance',
                'lookback time', 'cosmological distance',
            ],
            capabilities=[
                'comoving_distance', 'luminosity_distance',
                'angular_diameter_distance', 'lookback_time',
                'reionization_optical_depth',
            ],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising intergalactic_medium domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.physics import CosmologicalModel
        from ...astro_physics.next_gen.cosmological_context import (
            CosmologyParams, ReionizationModel,
        )

        # UNITS.  `CosmologicalModel` returns distances in Mpc and lookback
        # time in Gyr; it is the one module in `astro_physics` that works in
        # SI internally (audit M7), so nothing CGS may be mixed in here.
        # `optical_depth_reionization` is dimensionless.
        # Each self_check was recomputed from the cited formula in
        # `astra_baseline_audit_aug2026/repro/batchC_hand_values.py` with an
        # independently written trapezoid quadrature.
        cosmo = CosmologicalModel(H0=70.0, Omega_m=0.3, Omega_L=0.7)
        reion = ReionizationModel(CosmologyParams())   # same 70/0.3/0.7

        return [
            ComputationalCapability(
                name="comoving_distance",
                description="Line-of-sight comoving distance to a redshift",
                function=lambda redshift: float(
                    cosmo.comoving_distance(z=redshift)),
                parameters=[("redshift|z", "dimensionless", "redshift")],
                returns=("D_C", "Mpc"),
                reference=("D_C = (c/H0) Int_0^z dz'/E(z'), "
                           "E(z) = sqrt(Om (1+z)^3 + OL), flat LCDM with "
                           "H0 = 70 km/s/Mpc, Om = 0.3, OL = 0.7 "
                           "(Hogg 1999 eq. 15)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"redshift": 1.0}, 3303.8288, 1e-5),
            ),
            ComputationalCapability(
                name="luminosity_distance",
                description="Luminosity distance to a redshift",
                function=lambda redshift: float(
                    cosmo.luminosity_distance(z=redshift)),
                parameters=[("redshift|z", "dimensionless", "redshift")],
                returns=("D_L", "Mpc"),
                reference=("D_L = (1+z) D_C, flat LCDM with H0 = 70, "
                           "Om = 0.3, OL = 0.7 (Hogg 1999 eq. 21)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"redshift": 1.0}, 6607.6576, 1e-5),
            ),
            ComputationalCapability(
                name="angular_diameter_distance",
                description="Angular diameter distance to a redshift",
                function=lambda redshift: float(
                    cosmo.angular_diameter_distance(z=redshift)),
                parameters=[("redshift|z", "dimensionless", "redshift")],
                returns=("D_A", "Mpc"),
                reference=("D_A = D_C / (1+z), flat LCDM with H0 = 70, "
                           "Om = 0.3, OL = 0.7 (Hogg 1999 eq. 18)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"redshift": 1.0}, 1651.9144, 1e-5),
            ),
            ComputationalCapability(
                name="lookback_time",
                description="Lookback time to a redshift",
                function=lambda redshift: float(
                    cosmo.lookback_time(z=redshift)),
                parameters=[("redshift|z", "dimensionless", "redshift")],
                returns=("t_L", "Gyr"),
                reference=("t_L = (1/H0) Int_0^z dz'/((1+z') E(z')), flat "
                           "LCDM with H0 = 70, Om = 0.3, OL = 0.7; the "
                           "backend converts km/s/Mpc to 1/Gyr with 1.022e-3 "
                           "rather than 1.02271e-3, which is a +0.07% offset "
                           "(audit, section (e))"),
                test_ref="test_domain_capabilities.py",
                self_check=({"redshift": 1.0}, 7.7207133, 1e-5),
            ),
            ComputationalCapability(
                name="reionization_optical_depth",
                description=("Thomson-scattering optical depth of the "
                             "reionized intergalactic medium"),
                function=lambda z_max: float(
                    reion.optical_depth_reionization(z_max=z_max)),
                parameters=[("z_max|z_maximum", "dimensionless",
                             "upper redshift limit of the integration")],
                returns=("tau_e", "dimensionless"),
                reference=("tau = Int sigma_T n_e(z) c |dt/dz| dz with "
                           "n_e = 0.875 n_nucleon (1+z)^3 Q_HII(z) and a tanh "
                           "ionisation history centred on z_re = 7.5 of width "
                           "0.5; Planck 2018 measure 0.054"),
                test_ref="test_domain_capabilities.py",
                self_check=({"z_max": 30.0}, 0.0534785, 1e-4),
            ),
        ]


# Factory function
def create_intergalactic_medium_domain() -> IntergalacticMediumDomain:
    """Create an Intergalactic Medium domain instance"""
    return IntergalacticMediumDomain()


# Domain registration
try:
    register_domain(IntergalacticMediumDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
