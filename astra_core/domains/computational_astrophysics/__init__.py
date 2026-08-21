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
Computational Astrophysics Domain Module for ASTRA

Numerical simulations, hydrodynamics, N-body, radiative transfer codes.

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation. It now wraps
:mod:`astra_core.astro_physics.sph_gas_dynamics` and reports a confidence
derived from whether a computation actually ran.

Scope of what is wired
----------------------
The SPH interpolation kernel and the direct-summation gravity solver -- the
two parts of `sph_gas_dynamics` that passed numerical verification in the
August 2026 audit: integral W d^3r = 1.00000000 for all three kernels at
h = 1 by adaptive quadrature, W(0) = 1/(pi h^3) exactly, compact support
exact at 2h, analytic dW/dr matching central differences to <= 2e-10, and
two-body acceleration 1.39389205e-11 cm/s^2 = GM/r^2 exact with W = -GM^2/r
to 7 digits.

NOT wired: `SPHSimulation.compute_forces` (audit H9 -- the force kernel is
hardcoded to the cubic spline whatever `kernel_type` says, and the
advertised artificial viscosity is not implemented); `SPHKernel.gaussian`
(audit H10 -- truncated at 2h, giving rho/rho_true = 0.952); `FilamentFinder`
(audit H13 -- no connected-component labelling and, without `skimage`, no
thinning at all).

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


class ComputationalAstrophysicsDomain(ComputationalDomainModule):
    """
    Computational astrophysics: SPH interpolation and N-body gravity.

    Backed by `astro_physics.sph_gas_dynamics`.
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="computational_astrophysics",
            version="2.0.0",
            dependencies=[],
            description=("Numerical simulations, hydrodynamics, N-body, "
                         "radiative transfer codes"),
            keywords=[
                "computational", "simulation", "numerical", "hydrodynamics",
                "n-body", "radiative transfer", "sph",
                "smoothed particle hydrodynamics", "kernel",
                "smoothing length", "gravity solver", "softening",
                "potential energy",
            ],
            capabilities=[
                "sph_kernel", "sph_kernel_gradient", "nbody_acceleration",
                "gravitational_potential_energy",
            ],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising computational_astrophysics domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.sph_gas_dynamics import GravitySolver, SPHKernel

        # NOTE ON UNITS. `SPHKernel` is scale-free: W has the dimensions of
        # 1/length^3 in whatever length unit r and h are given in, so pc in
        # -> pc^-3 out (the module's own docstrings say cm, which is the same
        # function). `GravitySolver` is strictly CGS: positions in cm, masses
        # in g, accelerations in cm/s^2 and energies in erg; the wrappers
        # convert from (Msun, pc) and the self_checks are GM/r^2 and -GM^2/r
        # evaluated by hand in CGS.
        PC = 3.086e18       # cm
        M_SUN = 1.989e33    # g

        def sph_kernel(r: float, h: float) -> float:
            return float(SPHKernel.cubic_spline(np.asarray(float(r)),
                                                float(h)))

        def sph_kernel_gradient(r: float, h: float) -> float:
            return float(SPHKernel.cubic_spline_derivative(
                np.asarray(float(r)), float(h)))

        def _two_body(mass: float, separation: float):
            # Softening 1e-6 pc, i.e. 1e-6 of the reference separation, so
            # the Plummer-softened force is the Newtonian one to 1e-12.
            solver = GravitySolver(softening_pc=1.0e-6)
            pos = np.array([[0.0, 0.0, 0.0],
                            [separation * PC, 0.0, 0.0]])
            m = np.array([mass * M_SUN, mass * M_SUN])
            return solver, pos, m

        def nbody_acceleration(mass: float, separation: float) -> float:
            solver, pos, m = _two_body(mass, separation)
            return float(np.linalg.norm(solver.accelerations(pos, m)[0]))

        def gravitational_potential_energy(mass: float,
                                           separation: float) -> float:
            solver, pos, m = _two_body(mass, separation)
            return float(solver.potential_energy(pos, m))

        return [
            ComputationalCapability(
                name="sph_kernel",
                description="Cubic-spline SPH smoothing kernel W(r, h)",
                function=sph_kernel,
                parameters=[
                    ("r|radius|separation", "pc", "distance from particle"),
                    ("h|smoothing_length", "pc", "smoothing length"),
                ],
                returns=("W", "pc^-3"),
                reference=("W = (1/(pi h^3)) [1 - 1.5 q^2 + 0.75 q^3] for "
                           "q <= 1, (1/(4 pi h^3))(2-q)^3 for 1 < q <= 2, "
                           "q = r/h (Monaghan & Lattanzio 1985)"),
                test_ref="test_domain_capabilities.py",
                # q = 0.5: (1 - 1.5(0.25) + 0.75(0.125))/pi
                #        = 0.71875/pi = 0.2287852307
                self_check=({"r": 0.5, "h": 1.0}, 0.2287852307, 1e-9),
            ),
            ComputationalCapability(
                name="sph_kernel_gradient",
                description=("Radial derivative dW/dr of the cubic-spline "
                             "SPH kernel"),
                function=sph_kernel_gradient,
                parameters=[
                    ("r|radius|separation", "pc", "distance from particle"),
                    ("h|smoothing_length", "pc", "smoothing length"),
                ],
                returns=("dW/dr", "pc^-4"),
                reference=("dW/dr = (1/(pi h^4)) [-3q + 2.25 q^2] for q <= 1, "
                           "-(3/(4 pi h^4))(2-q)^2 for 1 < q <= 2"),
                test_ref="test_domain_capabilities.py",
                # q = 0.5: (-1.5 + 0.5625)/pi = -0.9375/pi = -0.298415518
                self_check=({"r": 0.5, "h": 1.0}, -0.298415518, 1e-9),
            ),
            ComputationalCapability(
                name="nbody_acceleration",
                description=("Direct-summation gravitational acceleration "
                             "of one of two equal masses"),
                function=nbody_acceleration,
                parameters=[
                    ("mass|m", "Msun", "mass of each of the two bodies"),
                    ("separation|distance|r", "pc", "separation"),
                ],
                returns=("|a|", "cm/s^2"),
                reference="a = G M / r^2 (Plummer-softened, eps = 1e-6 pc)",
                test_ref="test_domain_capabilities.py",
                # 6.674e-8 * 1.989e33 / (3.086e18)^2 = 1.3938921e-11 cm/s^2
                self_check=({"mass": 1.0, "separation": 1.0},
                            1.3938921e-11, 1e-6),
            ),
            ComputationalCapability(
                name="gravitational_potential_energy",
                description=("Gravitational potential energy of a two-body "
                             "system"),
                function=gravitational_potential_energy,
                parameters=[
                    ("mass|m", "Msun", "mass of each of the two bodies"),
                    ("separation|distance|r", "pc", "separation"),
                ],
                returns=("W", "erg"),
                reference="W = -G M^2 / r (Plummer-softened, eps = 1e-6 pc)",
                test_ref="test_domain_capabilities.py",
                # -6.674e-8 * (1.989e33)^2 / 3.086e18 = -8.5557847e+40 erg
                self_check=({"mass": 1.0, "separation": 1.0},
                            -8.5557847e40, 1e-6),
            ),
        ]


def create_computational_astrophysics_domain() -> ComputationalAstrophysicsDomain:
    """Create a ComputationalAstrophysicsDomain instance."""
    return ComputationalAstrophysicsDomain()


# Domain registration
try:
    register_domain(ComputationalAstrophysicsDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
