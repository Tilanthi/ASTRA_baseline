"""
Compact Object Binaries Domain Module for ASTRA

Binary evolution, tidal disruption events, mergers.

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation.  It now wraps

  * `KeplerianFitter` from :mod:`astra_core.astro_physics.radial_velocity` --
    the audit measured the Newton-Raphson Kepler solver's residual at 8.9e-16
    up to e = 0.95, and the radial-velocity model was corrected in this branch
    (audit B7.1: the sign of omega was flipped and the constant K e cos omega
    term was missing), and
  * `RelativisticPhysics.chirp_mass` / `schwarzschild_radius` from
    :mod:`astra_core.physics.relativistic_physics`.

**Provenance warning.** `relativistic_physics.py` was NOT in the scope of the
August 2026 physics audit.  Only these two routines are used, both are
single-line closed-form expressions, both were read in full and both are
pinned here by a self_check recomputed by hand.  Nothing else from that module
is wired, and the rest of it carries no verification claim.

NOT wired, deliberately (see `08_domains_batch_C.md`):
  * the m sin i / spectroscopic mass function.  The verified constant 28.406
    appears only *inline* inside `RVDetector.detect_planets`; there is no
    callable to wrap, and re-deriving it here would be new physics rather
    than a wrapper.
  * `RVPeriodogram` / `RVDetector` -- the audit's items B7.4/B7.5 (a
    self-normalised periodogram whose peak power is always 1.0, a
    trials-free FAP, and an unbounded Nelder-Mead over the period) are still
    OPEN: the module's own example is handed a 4.166 d alias for a true 10 d
    orbit and reports it as a planet.
  * `multi_messenger/` gravitational-wave inspiral code -- explicitly not
    audited (audit section (c).7).

Version: 2.0.0
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List

import numpy as np

from .. import DomainConfig, register_domain
from .._computational import (
    ComputationalCapability,
    ComputationalDomainModule,
    ImplementationStatus,
)

logger = logging.getLogger(__name__)


class CompactObjectBinariesDomain(ComputationalDomainModule):
    """
    Compact-object binaries: orbital kinematics and merger-relevant scales.
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="compact_binaries",
            version="2.0.0",
            dependencies=[],
            description="Binary evolution, tidal disruption events, mergers",
            keywords=[
                'compact binary', 'binary evolution', 'tidal disruption',
                'merger', 'ns-ns', 'bh-ns',
                # extensions matching the wired computations
                'kepler equation', 'eccentric anomaly', 'radial velocity',
                'orbit', 'spectroscopic binary', 'chirp mass',
                'schwarzschild radius', 'gravitational radius',
            ],
            capabilities=[
                'eccentric_anomaly', 'orbital_radial_velocity', 'chirp_mass',
                'schwarzschild_radius',
            ],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising compact_binaries domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.radial_velocity import KeplerianFitter
        from ...physics.relativistic_physics import RelativisticPhysics

        # UNITS.
        #   `KeplerianFitter._solve_kepler` and `.rv_model` are unit-agnostic
        #   in the sense that K comes back in whatever unit it went in (the
        #   module's own convention is m/s) and angles are RADIANS.  The
        #   wrappers below take omega in DEGREES and convert, and drive
        #   `rv_model` at a chosen orbital PHASE by setting period = 1 and
        #   T0 = 0 so that the mean anomaly is 2 pi phase.
        #   `RelativisticPhysics` is CGS: `chirp_mass` takes and returns
        #   GRAMS and `schwarzschild_radius` takes grams and returns CM, so
        #   the wrappers convert Msun -> g and cm -> km.  Declaring "Msun"
        #   while returning grams is precisely the 1e33 unit slip these
        #   self_checks exist to catch.
        # Every self_check was recomputed from the cited formula in
        # `astra_baseline_audit_aug2026/repro/batchC_hand_values.py`, whose
        # Kepler solver is written out again independently.
        fitter = KeplerianFitter()
        M_SUN_G = 1.989e33          # g -- the CGS constant set used by
        #                             relativistic_physics (G = 6.674e-8,
        #                             c = 2.998e10)

        def _eccentric_anomaly(mean_anomaly: float, eccentricity: float
                               ) -> float:
            return float(fitter._solve_kepler(
                np.atleast_1d(float(mean_anomaly)), float(eccentricity))[0])

        def _rv(semi_amplitude: float, eccentricity: float, omega: float,
                mean_anomaly: float) -> float:
            # period = 1 and T0 = 0 => M = 2 pi t; drive it at t = M/(2 pi)
            t = np.atleast_1d(float(mean_anomaly) / (2.0 * math.pi))
            return float(fitter.rv_model(t, {
                'period': 1.0, 'K': semi_amplitude, 'ecc': eccentricity,
                'omega': math.radians(omega), 'T0': 0.0, 'gamma': 0.0})[0])

        return [
            ComputationalCapability(
                name="eccentric_anomaly",
                description=("Eccentric anomaly of a Keplerian orbit from "
                             "the mean anomaly"),
                function=_eccentric_anomaly,
                parameters=[
                    ("mean_anomaly|m_anom", "rad", "mean anomaly"),
                    ("eccentricity|ecc|e", "dimensionless",
                     "orbital eccentricity"),
                ],
                returns=("E", "rad"),
                reference=("Kepler's equation M = E - e sin E, solved by "
                           "Newton-Raphson (Murray & Dermott 1999 eq. 2.52)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"mean_anomaly": 1.0, "eccentricity": 0.3},
                            1.2880913, 1e-6),
            ),
            ComputationalCapability(
                name="orbital_radial_velocity",
                description=("Stellar reflex radial velocity at a given "
                             "mean anomaly of an eccentric orbit"),
                function=_rv,
                parameters=[
                    ("semi_amplitude|k_amp|k", "m/s",
                     "radial-velocity semi-amplitude"),
                    ("eccentricity|ecc|e", "dimensionless",
                     "orbital eccentricity"),
                    ("omega|arg_periastron", "deg",
                     "argument of periastron"),
                    ("mean_anomaly|m_anom", "rad", "mean anomaly"),
                ],
                returns=("v_r", "m/s"),
                reference=("v_r = K [cos(f + omega) + e cos omega], f from "
                           "the eccentric anomaly (Murray & Correia 2010 "
                           "eq. 65; Lovis & Fischer 2010 eq. 12); systemic "
                           "velocity gamma = 0"),
                test_ref="test_domain_capabilities.py",
                self_check=({"semi_amplitude": 50.0, "eccentricity": 0.3,
                             "omega": 45.0, "mean_anomaly": 1.0},
                            -25.551445, 1e-6),
            ),
            ComputationalCapability(
                name="chirp_mass",
                description=("Chirp mass of a compact-object binary from its "
                             "component masses"),
                function=lambda mass_1, mass_2: float(
                    RelativisticPhysics.chirp_mass(mass1=mass_1 * M_SUN_G,
                                                   mass2=mass_2 * M_SUN_G)
                ) / M_SUN_G,
                parameters=[
                    ("mass_1|m_1", "Msun", "primary mass"),
                    ("mass_2|m_2", "Msun", "secondary mass"),
                ],
                returns=("M_chirp", "Msun"),
                reference=("M_chirp = (m1 m2)^(3/5) / (m1 + m2)^(1/5) "
                           "(Peters & Mathews 1963); 1.219 Msun for a "
                           "1.4 + 1.4 Msun neutron-star binary"),
                test_ref="test_domain_capabilities.py",
                self_check=({"mass_1": 1.4, "mass_2": 1.4}, 1.2187708, 1e-6),
            ),
            ComputationalCapability(
                name="schwarzschild_radius",
                description="Schwarzschild radius of a compact object",
                function=lambda mass: float(
                    RelativisticPhysics.schwarzschild_radius(
                        mass=mass * M_SUN_G)) / 1e5,
                parameters=[("mass|m", "Msun", "object mass")],
                returns=("R_s", "km"),
                reference=("R_s = 2 G M / c^2; 2.954 km for 1 Msun with "
                           "G = 6.674e-8 cgs and c = 2.998e10 cm/s"),
                test_ref="test_domain_capabilities.py",
                self_check=({"mass": 1.0}, 2.9538451, 1e-6),
            ),
        ]


# Factory function
def create_compact_binaries_domain() -> CompactObjectBinariesDomain:
    """Create a Compact Object Binaries domain instance"""
    return CompactObjectBinariesDomain()


# Domain registration
try:
    register_domain(CompactObjectBinariesDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
