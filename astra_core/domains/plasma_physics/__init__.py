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
Plasma Physics Domain Module for ASTRA

Kinetic theory, waves, instabilities, collisionless processes.

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation.  It now wraps the
routines of :mod:`astra_core.astro_physics.shock_physics` that the August 2026
physics audit verified numerically (Rankine-Hugoniot jump conditions, "all
three fluxes conserved to 1e-16") together with the C-shock width, whose
1e7-times-too-small ion-neutral coupling rate was repaired in this branch
(audit B-SH-2).

NOT wired, deliberately (see `08_domains_batch_C.md`):
  * `CShock.maximum_temperature` -- it hardcodes v_drift = 0.5 v_s with the
    in-code comment "Approximate"; the drift is a property of the solution,
    not a constant, so the number would be an invention.
  * `ShockChemistry` -- the SiO/H2O enhancement factors are tabulated fits
    with no cited source and were not audited.
  * kinetic theory, plasma waves and instability growth rates: no numerical
    implementation exists anywhere in this codebase.

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


class PlasmaPhysicsDomain(ComputationalDomainModule):
    """
    Plasma physics of astrophysical shocks in partially ionised gas.

    Backed by `astro_physics.shock_physics`.
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="plasma_physics",
            version="2.0.0",
            dependencies=[],
            description=("Kinetic theory, waves, instabilities, collisionless "
                         "processes"),
            keywords=[
                'plasma', 'kinetic theory', 'plasma waves', 'instabilities',
                'collisionless',
                # extensions matching the wired computations
                'rankine-hugoniot', 'compression ratio', 'shock jump',
                'post-shock temperature', 'alfven velocity', 'alfven speed',
                'c-shock', 'ion-neutral', 'ambipolar',
            ],
            capabilities=[
                'compression_ratio', 'post_shock_temperature',
                'alfven_velocity', 'c_shock_width',
            ],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising plasma_physics domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.shock_physics import (
            CShock, RankineHugoniot, MU_ATOMIC, MU_MOLECULAR, M_PROTON,
        )

        # UNITS.  `shock_physics` is CGS throughout: velocities in cm/s,
        # densities as MASS densities in g/cm^3, magnetic field in gauss,
        # lengths in cm.  The wrappers below present km/s, cm^-3 and
        # microgauss, converting n -> rho = n mu m_H with the SAME mu the
        # backend class was constructed with (MU_ATOMIC = 1.27 for the
        # Rankine-Hugoniot solver, MU_MOLECULAR = 2.37 for the C-shock).
        # Each self_check was recomputed from the cited formula in
        # `astra_baseline_audit_aug2026/repro/batchC_hand_values.py`.
        rh = RankineHugoniot(gamma=5.0 / 3.0, mu=MU_ATOMIC)
        cshock = CShock(mu_neutral=MU_MOLECULAR)

        def _rho_molecular(n: float) -> float:
            return n * MU_MOLECULAR * M_PROTON

        return [
            ComputationalCapability(
                name="compression_ratio",
                description="Rankine-Hugoniot compression ratio across a shock front",
                function=lambda mach: rh.compression_ratio(mach=mach),
                parameters=[("mach|mach_number|m", "dimensionless",
                             "upstream sonic Mach number")],
                returns=("rho_2/rho_1", "dimensionless"),
                reference=("r = (gamma+1) M^2 / ((gamma-1) M^2 + 2), "
                           "gamma = 5/3 (Landau & Lifshitz, Fluid Mechanics "
                           "sec. 85)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"mach": 10.0}, 3.8834951, 1e-6),
            ),
            ComputationalCapability(
                name="post_shock_temperature",
                description=("Immediate post-shock temperature in the strong-"
                             "shock limit"),
                function=lambda shock_velocity: rh.strong_shock_temperature(
                    shock_velocity=shock_velocity * 1e5),
                parameters=[("shock_velocity|v_shock|v_s", "km/s",
                             "shock velocity")],
                returns=("T_2", "K"),
                reference=("T_2 = 3 mu m_H v_s^2 / (16 k_B), mu = 1.27 "
                           "(atomic gas with 10% He)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"shock_velocity": 100.0}, 288474.38, 1e-5),
            ),
            ComputationalCapability(
                name="alfven_velocity",
                description="Alfven velocity of a magnetised molecular gas",
                function=lambda magnetic_field, density: cshock.alfven_velocity(
                    magnetic_field=magnetic_field,
                    density=_rho_molecular(density)) / 1e5,
                parameters=[
                    ("magnetic_field|b_field|b", "G", "magnetic field strength"),
                    ("density|n_h2|n", "cm^-3", "total particle number density"),
                ],
                returns=("v_A", "km/s"),
                reference=("v_A = B / sqrt(4 pi rho), rho = n mu m_H with "
                           "mu = 2.37 (molecular gas with 10% He)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"magnetic_field": 1e-4, "density": 1e4},
                            1.4166838, 1e-6),
            ),
            ComputationalCapability(
                name="c_shock_width",
                description=("Thickness of a magnetically-mediated C-type "
                             "shock in partially ionised gas"),
                function=lambda shock_velocity, magnetic_field, density,
                ionization_fraction: cshock.shock_width(
                    shock_velocity=shock_velocity * 1e5,
                    magnetic_field=magnetic_field,
                    density=_rho_molecular(density),
                    ionization_fraction=ionization_fraction),
                parameters=[
                    ("shock_velocity|v_shock|v_s", "km/s", "shock velocity"),
                    ("magnetic_field|b_field|b", "G", "pre-shock field strength"),
                    ("density|n_h2|n", "cm^-3", "pre-shock number density"),
                    ("ionization_fraction|x_i", "dimensionless",
                     "fractional ionisation"),
                ],
                returns=("L_shock", "cm"),
                reference=("L = v_s / nu_ni, nu_ni = <sigma v>_in x_i n_tot "
                           "with <sigma v>_in = 2e-9 cm^3/s (Draine 1980 "
                           "eq. 1.3; Draine & McKee 1993)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"shock_velocity": 10.0, "magnetic_field": 1e-4,
                             "density": 1e4, "ionization_fraction": 1e-7},
                            5.0e17, 1e-9),
            ),
        ]


# Factory function
def create_plasma_physics_domain() -> PlasmaPhysicsDomain:
    """Create a Plasma Physics domain instance"""
    return PlasmaPhysicsDomain()


# Domain registration
try:
    register_domain(PlasmaPhysicsDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
