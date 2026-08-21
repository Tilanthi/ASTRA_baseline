"""
Interferometry Domain Module for ASTRA

Optical/IR interferometry, VLBI, aperture synthesis.

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation. It now wraps
:mod:`astra_core.astro_physics.interferometry` and reports a confidence
derived from whether a computation actually ran.

Scope of what is wired
----------------------
The array/uv geometry layer only. Its central defect (audit C12: `c_light`
is CGS in this module, so `c_light/freq_Hz` was a wavelength in centimetres
divided into baselines in metres, making every uv coordinate 100.6x too
small) and the ENU->equatorial baseline rotation (audit B5.5) have been
fixed and are pinned by the reference cases below. The imaging layer
(`Imager`, `CLEANDeconvolver`, `SelfCalibrator`, `VisibilityModeler`) is
image-valued rather than scalar and is not exposed as scalar capabilities.

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


class InterferometryDomain(ComputationalDomainModule):
    """
    Interferometry: aperture synthesis and uv-plane geometry.

    Backed by `astro_physics.interferometry`. Every capability here is
    checked against a hand value computed from lambda = c/nu with CODATA
    c = 2.99792458e10 cm/s, which is exactly the check that exposed audit
    C12 (uv coordinates 100.6x too small).
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="interferometry",
            version="2.0.0",
            dependencies=[],
            description=("Optical/IR interferometry, VLBI, aperture "
                         "synthesis"),
            keywords=[
                "interferometry", "vlbi", "aperture synthesis",
                "optical interferometry", "radio interferometry",
                "uv coverage", "baseline", "angular resolution",
                "largest angular scale", "synthesised beam", "visibility",
            ],
            capabilities=[
                "angular_resolution", "largest_angular_scale",
                "uv_distance",
            ],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising interferometry domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.interferometry import (
            Antenna, ArrayConfiguration, UVSimulator,
        )

        # NOTE ON UNITS. `interferometry.ArrayConfiguration` stores antenna
        # positions in METRES and its own `c_light` is CGS (cm/s);
        # `angular_resolution` and `largest_angular_scale` return ARCSEC and
        # `UVSimulator.generate_coverage` returns uv coordinates in
        # WAVELENGTHS. Frequencies are passed to the backend in Hz, so the
        # GHz declared below is converted here. Each self_check is
        # lambda = c/nu evaluated by hand, which is the check that caught
        # audit C12.
        GHZ_TO_HZ = 1.0e9

        def _array(baselines_m) -> ArrayConfiguration:
            """Linear east-west array with the requested spacings."""
            array = ArrayConfiguration("query", latitude=0.0)
            array.add_antenna(Antenna(name="a0", x=0.0, y=0.0, z=0.0,
                                      diameter=12.0))
            for i, b in enumerate(baselines_m, start=1):
                array.add_antenna(Antenna(name=f"a{i}", x=float(b), y=0.0,
                                          z=0.0, diameter=12.0))
            return array

        def angular_resolution(max_baseline: float,
                               frequency: float) -> float:
            return float(_array([max_baseline])
                         .angular_resolution(frequency * GHZ_TO_HZ))

        def largest_angular_scale(min_baseline: float,
                                  frequency: float) -> float:
            # A second, longer baseline is added so that `min_baseline` is
            # genuinely the shortest spacing of a real array.
            return float(_array([min_baseline, 10.0 * min_baseline])
                         .largest_angular_scale(frequency * GHZ_TO_HZ))

        def uv_distance(baseline: float, frequency: float) -> float:
            """uv radius of an east-west baseline observing at transit
            (hour angle 0, declination 0), in wavelengths."""
            coverage = UVSimulator(_array([baseline])).generate_coverage(
                source_dec=0.0, freq_Hz=frequency * GHZ_TO_HZ,
                hour_angles=np.array([0.0]))
            return float(np.max(np.hypot(coverage.u, coverage.v)))

        return [
            ComputationalCapability(
                name="angular_resolution",
                description=("Synthesised-beam angular resolution of an "
                             "interferometer"),
                function=angular_resolution,
                parameters=[
                    ("max_baseline|baseline|b_max", "m",
                     "longest projected baseline"),
                    ("frequency|freq|nu", "GHz", "observing frequency"),
                ],
                returns=("theta_res", "arcsec"),
                reference=("theta = lambda / B_max (Thompson, Moran & "
                           "Swenson, Interferometry and Synthesis in Radio "
                           "Astronomy, ch. 5)"),
                test_ref="test_domain_capabilities.py",
                # lambda(230 GHz) = 0.1303445 cm; 0.1303445/1e5 rad
                # = 0.268855 arcsec.
                self_check=({"max_baseline": 1000.0, "frequency": 230.0},
                            0.268855, 1e-3),
            ),
            ComputationalCapability(
                name="largest_angular_scale",
                description=("Largest angular scale recoverable by an "
                             "interferometer"),
                function=largest_angular_scale,
                parameters=[
                    ("min_baseline|shortest_baseline|b_min", "m",
                     "shortest projected baseline"),
                    ("frequency|freq|nu", "GHz", "observing frequency"),
                ],
                returns=("theta_LAS", "arcsec"),
                reference="theta_LAS = lambda / B_min (TMS ch. 5)",
                test_ref="test_domain_capabilities.py",
                # lambda(230 GHz)/(100 m) = 2.68855 arcsec.
                self_check=({"min_baseline": 100.0, "frequency": 230.0},
                            2.68855, 1e-3),
            ),
            ComputationalCapability(
                name="uv_distance",
                description=("uv-plane radius sampled by a baseline at "
                             "transit"),
                function=uv_distance,
                parameters=[
                    ("baseline|b", "m", "physical baseline length"),
                    ("frequency|freq|nu", "GHz", "observing frequency"),
                ],
                returns=("uv_radius", "wavelengths"),
                reference=("|u| = B/lambda for an east-west baseline at "
                           "hour angle 0, declination 0 (TMS eq. 4.1)"),
                test_ref="test_domain_capabilities.py",
                # 1000 m / 1.3034455e-3 m = 767197 wavelengths.
                self_check=({"baseline": 1000.0, "frequency": 230.0},
                            767197.4, 1e-3),
            ),
        ]


def create_interferometry_domain() -> InterferometryDomain:
    """Create an InterferometryDomain instance."""
    return InterferometryDomain()


# Domain registration
try:
    register_domain(InterferometryDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
