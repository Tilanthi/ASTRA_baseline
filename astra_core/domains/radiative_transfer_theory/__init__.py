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
Radiative Transfer Theory Domain Module for ASTRA

Formal solutions, Monte Carlo methods, line formation, polarization

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation. It now wraps
routines from :mod:`astra_core.astro_physics.radiative_transfer`, whose escape
probabilities, uniform-slab dust transfer and line-profile integrals were
verified against independently derived ground truth during the August 2026
audit (the LVG/Osterbrock escape probabilities reproduce the asymptotic limits
beta*tau -> 1.5 / 1.0 / 1/3 exactly, and the small-tau series were repaired
under audit items H6/H7).

Every capability carries a `self_check` recomputed by hand from the formula in
its `reference` field, so a unit slip in this wrapper fails a test instead of
returning a silently wrong number.

NOT implemented anywhere in this codebase, and therefore not offered here:
Monte Carlo radiative transfer, polarized transfer / Stokes parameters, and
frequency-redistribution (non-grey) line transfer. The multi-level statistical
equilibrium solver in the same backend *is* verified but needs structured
molecular data (level energies, collision-rate matrices) rather than the
scalar parameters a text query can supply, so it is not exposed as a
capability here.

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


class RadiativeTransferTheoryDomain(ComputationalDomainModule):
    """
    Radiative transfer theory.

    Backed by `astro_physics.radiative_transfer`: escape probabilities for
    three geometries, the uniform-slab formal solution I = B(1 - e^-tau), and
    the Gaussian line-profile integral.
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="radiative_transfer_theory",
            version="2.0.0",
            dependencies=[],
            description=("Formal solutions, Monte Carlo methods, line "
                         "formation, polarization"),
            keywords=['radiative transfer', 'monte carlo', 'line formation',
                      'polarization', 'formal_solution', 'escape probability',
                      'optical depth', 'sobolev', 'lvg', 'source function',
                      'brightness temperature'],
            capabilities=['rt_formal_solutions', 'monte_carlo_methods',
                          'line_formation', 'polarized_radiation'],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising radiative_transfer_theory domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.radiative_transfer import (
            DustContinuumRT,
            EscapeGeometry,
            LineProfileSynthesizer,
            escape_probability,
        )

        # NOTE ON UNITS. `radiative_transfer` is CGS throughout: b_nu and
        # intensity are erg/s/cm^2/Hz/sr, LineProfileSynthesizer works in cm/s
        # and returns K cm/s. The wrappers below declare GHz, km/s and K km/s,
        # which is what an astronomer types, and each self_check pins that
        # conversion.
        KMS = 1.0e5      # cm/s per km/s

        def _slab_intensity(frequency: float, temperature: float,
                            surface_density: float) -> float:
            # kappa_0 = 0.1 cm^2/g at 250 um, beta = 2 (the module default,
            # the canonical Milky Way gas-referenced submm opacity).
            rt = DustContinuumRT(kappa_0=0.1, beta=2.0, lambda_0_um=250.0)
            return float(rt.intensity(frequency * 1e9, temperature,
                                      surface_density))

        def _line_w(t_ex: float, tau: float, t_line: float,
                    sigma: float) -> float:
            lps = LineProfileSynthesizer()
            return float(lps.integrated_intensity(t_ex, tau, t_line,
                                                  sigma * KMS)) / KMS

        return [
            ComputationalCapability(
                name="escape_probability_lvg",
                description=("Sobolev/LVG photon escape probability for an "
                             "expanding sphere"),
                function=lambda tau: float(escape_probability(
                    tau, EscapeGeometry.EXPANDING_SPHERE)),
                parameters=[("tau|optical_depth", "dimensionless",
                             "line-centre optical depth")],
                returns=("beta", "dimensionless"),
                reference="beta = (1 - exp(-tau)) / tau (Sobolev 1960)",
                test_ref="test_domain_capabilities.py",
                self_check=({"tau": 1.0}, 0.632120559, 1e-3),
            ),
            ComputationalCapability(
                name="escape_probability_uniform_sphere",
                description=("Osterbrock uniform-sphere photon escape "
                             "probability"),
                function=lambda tau: float(escape_probability(
                    tau, EscapeGeometry.UNIFORM_SPHERE)),
                parameters=[("tau|optical_depth", "dimensionless",
                             "line-centre optical depth")],
                returns=("beta", "dimensionless"),
                reference=("beta = 1.5/tau [1 - 2/tau^2 + (2/tau + 2/tau^2) "
                           "exp(-tau)] (Osterbrock 1989, App. 2; RADEX)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"tau": 1.0}, 0.707276647, 1e-3),
            ),
            ComputationalCapability(
                name="escape_probability_slab",
                description=("Plane-parallel slab photon escape probability"),
                function=lambda tau: float(escape_probability(
                    tau, EscapeGeometry.PLANE_PARALLEL)),
                parameters=[("tau|optical_depth", "dimensionless",
                             "line-centre optical depth")],
                returns=("beta", "dimensionless"),
                reference="beta = (1 - exp(-3 tau)) / (3 tau)",
                test_ref="test_domain_capabilities.py",
                self_check=({"tau": 1.0}, 0.316737644, 1e-3),
            ),
            ComputationalCapability(
                name="slab_emergent_intensity",
                description=("Formal solution for a uniform isothermal dust "
                             "slab"),
                function=_slab_intensity,
                parameters=[
                    ("frequency|nu", "GHz", "observing frequency"),
                    ("temperature|t_dust", "K", "dust temperature"),
                    ("surface_density|sigma_gas", "g/cm^2",
                     "gas+dust mass surface density"),
                ],
                returns=("I_nu", "erg/s/cm^2/Hz/sr"),
                reference=("I_nu = B_nu(T) [1 - exp(-tau_nu)], "
                           "tau_nu = kappa_0 (nu/nu_0)^2 Sigma, "
                           "kappa_0 = 0.1 cm^2/g at 250 um"),
                test_ref="test_domain_capabilities.py",
                self_check=({"frequency": 300.0, "temperature": 30.0,
                             "surface_density": 50.0}, 1.73672351e-13, 1e-3),
            ),
            ComputationalCapability(
                name="line_integrated_intensity",
                description=("Velocity-integrated brightness temperature of a "
                             "Gaussian line"),
                function=_line_w,
                parameters=[
                    ("t_ex|excitation_temperature", "K",
                     "excitation temperature"),
                    ("tau|optical_depth", "dimensionless",
                     "line-centre optical depth"),
                    ("t_line|t0", "K",
                     "h nu / k of the transition"),
                    ("sigma|linewidth", "km/s",
                     "1-D velocity dispersion (thermal + turbulent)"),
                ],
                returns=("W", "K km/s"),
                reference=("W = [J(T_ex) - J(T_bg)] (1 - e^-tau) sigma "
                           "sqrt(2 pi), J(T) = T_0/(exp(T_0/T) - 1)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"t_ex": 20.0, "tau": 1.0, "t_line": 5.53214,
                             "sigma": 1.0}, 26.1831605, 1e-3),
            ),
        ]


def create_radiative_transfer_theory_domain() -> RadiativeTransferTheoryDomain:
    """Create a Radiative Transfer Theory domain instance."""
    return RadiativeTransferTheoryDomain()


try:
    register_domain(RadiativeTransferTheoryDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
