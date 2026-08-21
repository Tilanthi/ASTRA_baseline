"""
Numerical Methods Domain Module for ASTRA

Finite difference/volume, spectral methods, particle methods.

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation. It now wraps
:mod:`astra_core.astro_physics.multiscale_coupling` and reports a confidence
derived from whether a computation actually ran.

Scope of what is wired
----------------------
The two routines of `multiscale_coupling` that carry the module's CRITICAL
audit fixes and are now pinned by regression tests:

* `HierarchicalRefinement.level_for_density` -- the Truelove et al. (1997)
  Jeans-resolution criterion, which had been **inverted** (audit C1): the
  code computed lambda_J/(N_J dx) instead of (N_J dx)/lambda_J, so the
  densest, 43x-under-resolved cell was assigned refinement level 0 while a
  well-resolved diffuse cell got 3 levels.
* `MultiScaleSimulation._upwind_step` -- donor-cell advection, which was
  anti-diffusive for v < 0 (audit C2): a delta function grew to
  max|f| = 1.85e6 in 40 steps at v = -0.5, dt = 0.5.

NOT wired: `ScaleCoupler`/`MultiScaleSimulation.step_coarse` mass
conservation (audit H2 -- the reported 1e-16 "conservation" is produced by
an explicit rescale, so it is an identity, not a property of the scheme);
`CoolingFunction` (an unvalidated hand-fitted table); the sub-grid
star-formation and feedback models (audit H3, H14 open).

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


class NumericalMethodsDomain(ComputationalDomainModule):
    """
    Numerical methods for astrophysical fluid dynamics.

    Backed by `astro_physics.multiscale_coupling` (AMR criterion, donor-cell
    advection) and `astro_physics.gravitational_collapse` (Jeans length).
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="numerical_methods",
            version="2.0.0",
            dependencies=[],
            description=("Finite difference/volume, spectral methods, "
                         "particle methods"),
            keywords=[
                "numerical methods", "finite_difference", "finite_volume",
                "spectral_methods", "particle_methods", "amr",
                "adaptive mesh refinement", "truelove", "jeans resolution",
                "advection", "upwind", "donor cell", "courant", "cfl",
                "stability", "resolution",
            ],
            capabilities=[
                "jeans_refinement_level", "truelove_cell_size",
                "donor_cell_amplitude_ratio",
            ],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising numerical_methods domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.gravitational_collapse import JeansAnalysis
        from ...astro_physics.multiscale_coupling import (
            HierarchicalRefinement, MultiScaleSimulation,
        )

        jeans = JeansAnalysis(number_density=True, mu_particle=2.33)

        # NOTE ON UNITS. `HierarchicalRefinement` is CGS throughout: it wants
        # a sound speed in cm/s, a cell size in cm and a MASS density in
        # g/cm^3, and returns a numpy integer (converted to `int` here, since
        # np.int64 is not a Python int and would fail the capability
        # contract). `JeansAnalysis.jeans_length` returns cm. The wrappers
        # take the astronomer-facing (cm^-3, K, pc) and convert.
        PC = 3.0857e18        # cm
        MU, M_H = 2.33, 1.674e-24
        N_JEANS = 4.0         # Truelove et al. (1997)
        MAX_LEVEL = 20        # high enough not to clip the reference case

        def jeans_refinement_level(density: float, temperature: float,
                                   cell_size: float) -> float:
            refiner = HierarchicalRefinement(
                sound_speed=jeans.sound_speed(temperature),
                dx=cell_size * PC, n_jeans=N_JEANS, max_level=MAX_LEVEL)
            return int(refiner.level_for_density(
                np.array(density * MU * M_H)))

        def truelove_cell_size(density: float, temperature: float) -> float:
            return float(jeans.jeans_length(density=density,
                                            temperature=temperature)
                         / N_JEANS / PC)

        def donor_cell_amplitude_ratio(velocity: float,
                                       n_steps: float) -> float:
            """Peak amplitude of a top-hat after `n_steps` donor-cell steps,
            relative to its initial peak. `velocity` is in cells per step, so
            |velocity| is the Courant number; a TVD scheme returns <= 1."""
            field = np.zeros((32, 1, 1))
            field[16, 0, 0] = 1.0
            peak0 = field.max()
            v = np.array([float(velocity), 0.0, 0.0])
            for _ in range(int(round(n_steps))):
                field = MultiScaleSimulation._upwind_step(field, v, 1.0)
            return float(np.max(np.abs(field)) / peak0)

        return [
            ComputationalCapability(
                name="jeans_refinement_level",
                description=("AMR refinement level required by the Truelove "
                             "Jeans-resolution criterion"),
                function=jeans_refinement_level,
                parameters=[
                    ("density|n_h2|n", "cm^-3", "H2 number density"),
                    ("temperature|t_kin|t", "K", "gas temperature"),
                    ("cell_size|dx", "pc", "base-grid cell size"),
                ],
                returns=("level", "refinement levels"),
                reference=("level = ceil(log2(N_J dx / lambda_J)), N_J = 4, "
                           "lambda_J = c_s sqrt(pi/(G rho)) "
                           "(Truelove et al. 1997)"),
                test_ref="test_domain_capabilities.py",
                # c_s(10 K) = 0.18817 km/s, lambda_J(1e6 cm^-3) = 0.0211843 pc,
                # ratio = 4 x 0.1 / 0.0211843 = 18.88, log2 = 4.24 -> level 5.
                self_check=({"density": 1e6, "temperature": 10.0,
                             "cell_size": 0.1}, 5.0, 1e-9),
            ),
            ComputationalCapability(
                name="truelove_cell_size",
                description=("Largest cell size that resolves the Jeans "
                             "length to the Truelove standard"),
                function=truelove_cell_size,
                parameters=[
                    ("density|n_h2|n", "cm^-3", "H2 number density"),
                    ("temperature|t_kin|t", "K", "gas temperature"),
                ],
                returns=("dx_max", "pc"),
                reference=("dx <= lambda_J / 4, lambda_J = c_s sqrt(pi/(G rho)) "
                           "(Truelove et al. 1997)"),
                test_ref="test_domain_capabilities.py",
                # lambda_J(1e4 cm^-3, 10 K) = 0.211874 pc; /4 = 0.0529685 pc.
                self_check=({"density": 1e4, "temperature": 10.0},
                            0.0529685, 1e-3),
            ),
            ComputationalCapability(
                name="donor_cell_amplitude_ratio",
                description=("Peak amplitude growth of a top-hat under "
                             "donor-cell (upwind) advection"),
                function=donor_cell_amplitude_ratio,
                parameters=[
                    ("velocity|v", "cells/step",
                     "advection velocity; |v| is the Courant number"),
                    ("n_steps|steps", "count", "number of time steps"),
                ],
                returns=("max|f|/max|f_0|", "dimensionless"),
                reference=("f_i^{n+1} = f_i - v dt (f_i - f_{i-1}) for v > 0, "
                           "f_i - v dt (f_{i+1} - f_i) for v < 0; TVD for "
                           "|v| dt <= 1 (Courant, Friedrichs & Lewy 1928)"),
                test_ref="test_domain_capabilities.py",
                # At |v| dt = 1 the donor-cell update is an exact shift
                # (f_i^{n+1} = f_{i+1} for v = -1), so the peak is preserved
                # exactly: the ratio is 1.0 for any number of steps. With the
                # pre-audit sign error this case grew without bound.
                self_check=({"velocity": -1.0, "n_steps": 8.0}, 1.0, 1e-12),
            ),
        ]


def create_numerical_methods_domain() -> NumericalMethodsDomain:
    """Create a NumericalMethodsDomain instance."""
    return NumericalMethodsDomain()


# Domain registration
try:
    register_domain(NumericalMethodsDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
