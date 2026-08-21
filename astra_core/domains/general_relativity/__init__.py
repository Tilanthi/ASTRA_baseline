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
General Relativity Applications Domain Module for ASTRA

Metric theory, black hole solutions, cosmological perturbations.

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation. It now wraps the
`schwarzschild_metric` model of
:class:`astra_core.physics.UnifiedPhysicsEngine` and the flat-LambdaCDM
distance layer of :mod:`astra_core.astro_physics.advanced_lensing`, and
reports a confidence derived from whether a computation actually ran.

Scope of what is wired
----------------------
The Schwarzschild exterior solution (g_tt = 1 - r_s/r, verified here
against 2GM/c^2 = 2.954 km for 1 Msun), the gravitational redshift that
follows from it, the Einstein light-deflection angle (1.7516 arcsec at the
solar limb) and the FLRW angular-diameter distance (reproduced to 5 digits
by independent quadrature during the August 2026 audit).

Implementation note: `UnifiedPhysicsEngine.compute` is called with
``enforce_constraints=False``. With the default ``True`` the engine
*subtracts a constraint penalty from the returned physical value*
(`astra_core/physics/__init__.py:412-437`), so a model value and an
optimisation objective share one return channel. The registered constraints
happen not to fire for these parameters, but a physical quantity must not
depend on that.

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


class GeneralRelativityDomain(ComputationalDomainModule):
    """
    General relativity: the Schwarzschild solution and FLRW distances.

    Backed by `astra_core.physics.UnifiedPhysicsEngine` and
    `astra_core.astro_physics.advanced_lensing.Cosmology`.
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="general_relativity",
            version="2.0.0",
            dependencies=[],
            description=("Metric theory, black hole solutions, "
                         "cosmological perturbations"),
            keywords=[
                "general relativity", "metric", "black hole",
                "cosmological perturbation", "einstein_equation",
                "schwarzschild", "event horizon", "gravitational redshift",
                "light deflection", "geodesic", "flrw", "curvature",
            ],
            capabilities=[
                "schwarzschild_radius", "metric_g_tt",
                "gravitational_redshift", "light_deflection_angle",
                "angular_diameter_distance",
            ],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising general_relativity domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.advanced_lensing import Cosmology
        from ...physics import UnifiedPhysicsEngine

        engine = UnifiedPhysicsEngine()
        cosmo = Cosmology()

        # NOTE ON UNITS. `UnifiedPhysicsEngine` is CGS: its
        # `schwarzschild_metric` model wants a mass in GRAMS and a radius in
        # CENTIMETRES and returns the dimensionless g_tt, so the wrappers
        # convert from (Msun, km). `Cosmology.angular_diameter_distance`
        # returns Mpc. The constants below are the engine's own, so that a
        # capability and the model it calls cannot drift apart.
        G = engine.constants['G']
        C = engine.constants['c']
        M_SUN = engine.constants['M_sun']
        KM = 1.0e5              # cm
        ARCSEC_PER_RAD = 206264.806

        def _g_tt(mass: float, radius: float) -> float:
            result = engine.compute(
                'schwarzschild_metric',
                {'mass': mass * M_SUN, 'radius': radius * KM},
                enforce_constraints=False)
            return float(result.value)

        def schwarzschild_radius(mass: float) -> float:
            return float(2.0 * G * mass * M_SUN / C ** 2 / KM)

        def gravitational_redshift(mass: float, radius: float) -> float:
            g_tt = _g_tt(mass, radius)
            if g_tt <= 0.0:
                raise ValueError("radius is inside the event horizon")
            return float(1.0 / np.sqrt(g_tt) - 1.0)

        def light_deflection_angle(mass: float,
                                   impact_parameter: float) -> float:
            alpha_rad = (4.0 * G * mass * M_SUN
                         / (C ** 2 * impact_parameter * KM))
            return float(alpha_rad * ARCSEC_PER_RAD)

        return [
            ComputationalCapability(
                name="schwarzschild_radius",
                description="Schwarzschild radius of a non-rotating mass",
                function=schwarzschild_radius,
                parameters=[("mass|m", "Msun", "gravitating mass")],
                returns=("r_s", "km"),
                reference="r_s = 2 G M / c^2 (Schwarzschild 1916)",
                test_ref="test_domain_capabilities.py",
                # 2 x 6.6743e-8 x 1.98892e33 / (2.99792458e10)^2
                #   = 2.95401e5 cm = 2.954008 km
                self_check=({"mass": 1.0}, 2.954008, 1e-3),
            ),
            ComputationalCapability(
                name="metric_g_tt",
                description=("Time-time component of the Schwarzschild "
                             "metric"),
                function=_g_tt,
                parameters=[
                    ("mass|m", "Msun", "gravitating mass"),
                    ("radius|r", "km", "areal radius"),
                ],
                returns=("-g_tt", "dimensionless"),
                reference=("g_tt = -(1 - r_s/r), r_s = 2GM/c^2; the model "
                           "returns the factor (1 - r_s/r)"),
                test_ref="test_domain_capabilities.py",
                # 1 - 2.954008/10 = 0.7045992
                self_check=({"mass": 1.0, "radius": 10.0}, 0.7045992, 1e-3),
            ),
            ComputationalCapability(
                name="gravitational_redshift",
                description=("Gravitational redshift of light emitted at a "
                             "given radius"),
                function=gravitational_redshift,
                parameters=[
                    ("mass|m", "Msun", "gravitating mass"),
                    ("radius|r", "km", "emission radius"),
                ],
                returns=("z", "dimensionless"),
                reference="1 + z = (1 - r_s/r)^(-1/2) (Einstein 1916)",
                test_ref="test_domain_capabilities.py",
                # 1/sqrt(0.7045992) - 1 = 0.1913213
                self_check=({"mass": 1.0, "radius": 10.0},
                            0.1913213, 1e-3),
            ),
            ComputationalCapability(
                name="light_deflection_angle",
                description=("General-relativistic deflection of light by a "
                             "point mass"),
                function=light_deflection_angle,
                parameters=[
                    ("mass|m", "Msun", "deflecting mass"),
                    ("impact_parameter|b", "km", "impact parameter"),
                ],
                returns=("alpha", "arcsec"),
                reference=("alpha = 4 G M / (c^2 b) = 2 r_s / b "
                           "(Einstein 1916; Dyson, Eddington & Davidson "
                           "1920)"),
                test_ref="test_domain_capabilities.py",
                # Solar limb: 2 x 2.95401e5 cm / 6.957e10 cm = 8.4924e-6 rad
                #   = 1.75164 arcsec
                self_check=({"mass": 1.0, "impact_parameter": 695700.0},
                            1.751640, 1e-3),
            ),
            ComputationalCapability(
                name="angular_diameter_distance",
                description=("Angular diameter distance in the flat FLRW "
                             "metric"),
                function=lambda redshift: float(
                    cosmo.angular_diameter_distance(redshift)),
                parameters=[("redshift|z", "dimensionless", "redshift")],
                returns=("D_A", "Mpc"),
                reference=("D_A = (c/H0)/(1+z) int_0^z dz'/E(z'), "
                           "E = sqrt(Om (1+z)^3 + OL); H0 = 70, Om = 0.3"),
                test_ref="test_domain_capabilities.py",
                self_check=({"redshift": 1.0}, 1651.914, 1e-3),
            ),
        ]


def create_general_relativity_domain() -> GeneralRelativityDomain:
    """Create a GeneralRelativityDomain instance."""
    return GeneralRelativityDomain()


# Domain registration
try:
    register_domain(GeneralRelativityDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
