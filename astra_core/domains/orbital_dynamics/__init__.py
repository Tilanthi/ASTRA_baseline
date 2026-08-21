"""
Orbital Dynamics Domain Module for ASTRA

3-body problems, resonances, secular evolution, scattering.

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation. It now wraps
:mod:`astra_core.astro_physics.radial_velocity` and reports a confidence
derived from whether a computation actually ran.

Scope of what is wired
----------------------
The Keplerian two-body machinery, which is the part of `radial_velocity`
that passed verification in the August 2026 audit: the Newton-Raphson
solution of Kepler's equation has a residual max|E - e sin E - M| = 8.9e-16
up to e = 0.95, and the stellar reflex-velocity model was corrected in audit
fix B7.1 (it had computed cos(f - omega) instead of cos(f + omega) and
dropped the constant e cos(omega) term, so fitting a true e = 0.3,
omega = 45 deg orbit returned e = 0.245, omega = 297.5 deg).

`semi_major_axis` and `radial_velocity_semi_amplitude` are NOT wrappers:
Kepler's third law is not implemented anywhere in `astro_physics` as a
callable, and the K <-> m_p sin(i) relation exists only as three inline
lines inside `RVDetector.detect_planets` (`radial_velocity.py:373-377`).
Both are therefore written out here from the cited textbook form with exact
constants; the second reproduces 28.41 m/s for 1 M_Jup on a 1-yr orbit
around 1 M_sun, the value the audit derived independently (28.406 m/s)
while checking the module's hardcoded 28.4.

NOT wired: `RVPeriodogram` (audit B7.4 -- the periodogram is normalised to
its own maximum, so the peak power is always 1.0, and `_estimate_fap`
ignores both of its arguments) and `KeplerianFitter.fit` (audit B7.5 --
unbounded Nelder-Mead over the period).

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


class OrbitalDynamicsDomain(ComputationalDomainModule):
    """
    Orbital dynamics: Keplerian orbits and reflex motion.

    Backed by `astro_physics.radial_velocity.KeplerianFitter`.
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="orbital_dynamics",
            version="2.0.0",
            dependencies=[],
            description=("3-body problems, resonances, secular evolution, "
                         "scattering"),
            keywords=[
                "orbital dynamics", "three_body", "resonance",
                "secular_evolution", "scattering", "kepler",
                "eccentric anomaly", "semi-major axis", "orbit",
                "radial velocity", "reflex motion", "periastron",
            ],
            capabilities=[
                "eccentric_anomaly", "semi_major_axis",
                "radial_velocity_semi_amplitude",
                "radial_velocity_at_periastron",
            ],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising orbital_dynamics domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.radial_velocity import KeplerianFitter

        # NOTE ON UNITS. `KeplerianFitter` is unit-agnostic in time (the
        # period and the epoch simply have to share a unit) and returns the
        # radial velocity in whatever unit K is given in; angles are in
        # RADIANS. The two closed-form capabilities below are evaluated in
        # CGS from exact constants and converted to AU and m/s.
        G = 6.67430e-8         # cm^3 g^-1 s^-2  (CODATA 2018)
        M_SUN = 1.98892e33     # g
        M_JUP = 1.89813e30     # g
        AU = 1.495979e13       # cm
        YR = 3.15576e7         # s  (Julian year)

        fitter = KeplerianFitter()

        def eccentric_anomaly(mean_anomaly: float,
                              eccentricity: float) -> float:
            return float(fitter._solve_kepler(
                np.array([float(mean_anomaly)]), float(eccentricity))[0])

        def semi_major_axis(period: float, total_mass: float) -> float:
            p = period * YR
            return float((G * total_mass * M_SUN * p ** 2
                          / (4.0 * np.pi ** 2)) ** (1.0 / 3.0) / AU)

        def radial_velocity_semi_amplitude(planet_mass: float,
                                           stellar_mass: float,
                                           period: float,
                                           eccentricity: float) -> float:
            m_p = planet_mass * M_JUP
            m_tot = stellar_mass * M_SUN + m_p
            p = period * YR
            k_cgs = ((2.0 * np.pi * G / p) ** (1.0 / 3.0) * m_p
                     / m_tot ** (2.0 / 3.0)
                     / np.sqrt(1.0 - eccentricity ** 2))
            return float(k_cgs / 100.0)          # cm/s -> m/s

        def radial_velocity_at_periastron(k: float, eccentricity: float,
                                          omega: float) -> float:
            return float(fitter.rv_model(
                np.array([0.0]),
                {'period': 1.0, 'K': k, 'ecc': eccentricity,
                 'omega': omega, 'T0': 0.0, 'gamma': 0.0})[0])

        return [
            ComputationalCapability(
                name="eccentric_anomaly",
                description=("Eccentric anomaly from Kepler's equation "
                             "M = E - e sin E"),
                function=eccentric_anomaly,
                parameters=[
                    ("mean_anomaly|m_anomaly", "rad", "mean anomaly"),
                    ("eccentricity|ecc|e", "dimensionless", "eccentricity"),
                ],
                returns=("E", "rad"),
                reference="M = E - e sin E, solved by Newton-Raphson (Kepler 1609)",
                test_ref="test_domain_capabilities.py",
                # Independent Brent root of E - 0.3 sin E - 1 = 0:
                # E = 1.2880913132 rad (residual 2e-13).
                self_check=({"mean_anomaly": 1.0, "eccentricity": 0.3},
                            1.2880913132, 1e-9),
            ),
            ComputationalCapability(
                name="semi_major_axis",
                description=("Orbital semi-major axis from Kepler's third "
                             "law"),
                function=semi_major_axis,
                parameters=[
                    ("period|p", "yr", "orbital period"),
                    ("total_mass|mass|m", "Msun",
                     "total mass of the two bodies"),
                ],
                returns=("a", "AU"),
                reference=("a = [G (M1+M2) P^2 / (4 pi^2)]^(1/3) "
                           "(Kepler 1619; Newton 1687)"),
                test_ref="test_domain_capabilities.py",
                # P = 1 Julian yr, M = 1 Msun: a = 1.49609e13 cm
                #   = 1.000073 AU (the 7e-5 excess over 1 AU is the Julian
                #   vs sidereal year).
                self_check=({"period": 1.0, "total_mass": 1.0},
                            1.0000727, 1e-6),
            ),
            ComputationalCapability(
                name="radial_velocity_semi_amplitude",
                description=("Stellar radial-velocity semi-amplitude "
                             "induced by a planet"),
                function=radial_velocity_semi_amplitude,
                parameters=[
                    ("planet_mass|m_planet|m_p", "M_Jup",
                     "planet mass times sin(i)"),
                    ("stellar_mass|m_star", "Msun", "stellar mass"),
                    ("period|p", "yr", "orbital period"),
                    ("eccentricity|ecc|e", "dimensionless", "eccentricity"),
                ],
                returns=("K", "m/s"),
                reference=("K = (2 pi G / P)^(1/3) m_p sin i / "
                           "[(M_* + m_p)^(2/3) sqrt(1 - e^2)] "
                           "(Lovis & Fischer 2010, eq. 14)"),
                test_ref="test_domain_capabilities.py",
                # 1 M_Jup, 1 Msun, P = 1 yr, e = 0 -> 28.4096 m/s; the
                # audit's independent value for the same case is 28.406 m/s.
                self_check=({"planet_mass": 1.0, "stellar_mass": 1.0,
                             "period": 1.0, "eccentricity": 0.0},
                            28.4096, 1e-3),
            ),
            ComputationalCapability(
                name="radial_velocity_at_periastron",
                description=("Stellar radial velocity at periastron "
                             "passage"),
                function=radial_velocity_at_periastron,
                parameters=[
                    ("k|semi_amplitude", "m/s",
                     "radial-velocity semi-amplitude"),
                    ("eccentricity|ecc|e", "dimensionless", "eccentricity"),
                    ("omega|argument_of_periastron", "rad",
                     "argument of periastron"),
                ],
                returns=("v_r", "m/s"),
                reference=("v_r = K [cos(f + omega) + e cos omega]; at "
                           "periastron f = 0, so v_r = K (1 + e) cos omega "
                           "(Murray & Correia 2010, eq. 65)"),
                test_ref="test_domain_capabilities.py",
                # K = 50, e = 0.3, omega = pi/4:
                #   50 x 1.3 x 0.70710678 = 45.9619407771 m/s
                self_check=({"k": 50.0, "eccentricity": 0.3,
                             "omega": np.pi / 4.0}, 45.9619407771, 1e-9),
            ),
        ]


def create_orbital_dynamics_domain() -> OrbitalDynamicsDomain:
    """Create an OrbitalDynamicsDomain instance."""
    return OrbitalDynamicsDomain()


# Domain registration
try:
    register_domain(OrbitalDynamicsDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
