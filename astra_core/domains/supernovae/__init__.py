"""
Supernovae Domain Module for ASTRA

All types, nucleosynthesis, remnants, cosmology applications.

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation.  It now wraps the
routines of :mod:`astra_core.astro_physics.supernova_remnant_physics` that the
August 2026 audit verified numerically (Sedov-Taylor radius, velocity and
post-shock temperature; the diffusive-shock-acceleration index relations)
together with the exact Sedov energy partition, whose swapped kinetic/thermal
split was repaired in this branch (audit B-SNR-2).

NOT wired, deliberately (see `08_domains_batch_C.md`):
  * `SynchrotronEmission.luminosity_1ghz` / `surface_brightness` -- repaired
    for the W -> erg and spurious-4pi errors (B-SNR-4) but the underlying
    equipartition electron normalisation is an assumption of the model, not a
    measured quantity, and it was not independently validated.
  * `SNREvolution.transition_sedov_to_snowplow` / `_snowplow_to_merger` --
    order-of-magnitude scaling laws with no cited source in-code.
  * `XRayThermalEmission.cooling_function` -- the 1e4 K discontinuity was
    repaired (B-SNR-6) but the piecewise fit itself is uncited.
  * SN classification, light curves and Type Ia cosmology: no numerical
    implementation exists in this codebase (`next_gen/transient_science.
    fit_sn_ia` is a truncated function body that returns None, audit M5).

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


class SupernovaeDomain(ComputationalDomainModule):
    """
    Supernova blast-wave dynamics and remnant evolution.

    Backed by `astro_physics.supernova_remnant_physics`.
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="supernovae",
            version="2.0.0",
            dependencies=[],
            description=("All types, nucleosynthesis, remnants, cosmology "
                         "applications"),
            keywords=[
                'supernova', 'sn ia', 'core collapse', 'nucleosynthesis',
                'sn cosmology',
                # extensions matching the wired computations
                'supernova remnant', 'snr', 'sedov', 'sedov-taylor',
                'blastwave', 'blast wave', 'shock radius',
                'diffusive shock acceleration', 'energy partition',
            ],
            capabilities=[
                'sedov_radius', 'sedov_velocity', 'post_shock_temperature',
                'sedov_kinetic_energy_fraction', 'dsa_electron_index',
                'dsa_radio_spectral_index',
            ],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising supernovae domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.supernova_remnant_physics import (
            SedovTaylorBlastwave, SynchrotronEmission,
            sedov_similarity_profiles,
            MU_NEUTRAL, M_PROTON, PC, YR, MU_IONIZED,
        )

        # UNITS.  `supernova_remnant_physics` is CGS throughout.
        # `SedovTaylorBlastwave.radius/velocity` take a MASS density in
        # g/cm^3 (note: `solve()` takes a NUMBER density and converts with
        # MU_NEUTRAL = 1.4 -- the two entry points differ, which is exactly
        # the kind of slip these self_checks exist to catch), a time in
        # seconds and an energy in erg, and return cm and cm/s.
        # `post_shock_temperature` takes cm/s and returns K with the default
        # mu = MU_IONIZED = 0.6.  Every self_check below was recomputed from
        # the cited formula in
        # `astra_baseline_audit_aug2026/repro/batchC_hand_values.py`; the
        # energy partition there is integrated with an independently written
        # RK4 solver rather than with scipy's.
        sedov = SedovTaylorBlastwave(gamma=5.0 / 3.0)
        synch = SynchrotronEmission()

        def _rho(n_ambient: float) -> float:
            return n_ambient * MU_NEUTRAL * M_PROTON

        return [
            ComputationalCapability(
                name="sedov_radius",
                description="Sedov-Taylor blast-wave radius of a remnant",
                function=lambda energy, density, age: sedov.radius(
                    energy=energy, density=_rho(density),
                    time=age * YR) / PC,
                parameters=[
                    ("energy|e_sn|e", "erg", "explosion energy"),
                    ("density|n_0|n", "cm^-3", "ambient number density"),
                    ("age|time|t", "yr", "time since explosion"),
                ],
                returns=("R_shock", "pc"),
                reference=("R = xi_0 (E t^2 / rho)^(1/5), xi_0 = 1.15167 for "
                           "gamma = 5/3, rho = n mu m_H with mu = 1.4 "
                           "(Sedov 1959; Taylor 1950)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"energy": 1e51, "density": 1.0, "age": 1000.0},
                            4.984965, 1e-5),
            ),
            ComputationalCapability(
                name="sedov_velocity",
                description="Sedov-Taylor blast-wave expansion speed",
                function=lambda energy, density, age: sedov.velocity(
                    energy=energy, density=_rho(density),
                    time=age * YR) / 1e5,
                parameters=[
                    ("energy|e_sn|e", "erg", "explosion energy"),
                    ("density|n_0|n", "cm^-3", "ambient number density"),
                    ("age|time|t", "yr", "time since explosion"),
                ],
                returns=("v_shock", "km/s"),
                reference="v = dR/dt = (2/5) R/t for the Sedov solution",
                test_ref="test_domain_capabilities.py",
                self_check=({"energy": 1e51, "density": 1.0, "age": 1000.0},
                            1949.7595, 1e-5),
            ),
            ComputationalCapability(
                name="post_shock_temperature",
                description=("Immediate post-shock temperature behind a "
                             "remnant's forward shock"),
                function=lambda shock_velocity: sedov.post_shock_temperature(
                    velocity=shock_velocity * 1e5, mu=MU_IONIZED),
                parameters=[("shock_velocity|v_shock|v_s", "km/s",
                             "forward-shock velocity")],
                returns=("T_s", "K"),
                reference=("T_s = 3 mu m_H v_s^2 / (16 k_B), mu = 0.6 "
                           "(fully ionised solar composition)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"shock_velocity": 1000.0}, 1.3628711e7, 1e-5),
            ),
            ComputationalCapability(
                name="sedov_kinetic_energy_fraction",
                description=("Fraction of the explosion energy that is "
                             "kinetic in the Sedov similarity solution"),
                function=lambda gamma: float(
                    sedov_similarity_profiles(gamma)['f_kinetic']),
                parameters=[("gamma|adiabatic_index", "dimensionless",
                             "adiabatic index of the shocked gas")],
                returns=("E_kin/E", "dimensionless"),
                reference=("E_kin/E = Int_0^1 (G V^2/2) xi^2 dxi / Int_0^1 "
                           "(G V^2/2 + P/(gamma-1)) xi^2 dxi over the Sedov "
                           "similarity solution (Landau & Lifshitz, Fluid "
                           "Mechanics sec. 106); 0.283 for gamma = 5/3"),
                test_ref="test_domain_capabilities.py",
                self_check=({"gamma": 5.0 / 3.0}, 0.2827555, 1e-4),
            ),
            ComputationalCapability(
                name="dsa_electron_index",
                description=("Diffusive-shock-acceleration electron energy "
                             "spectral index from the compression ratio"),
                function=lambda compression_ratio: synch.
                electron_spectrum_index(compression_ratio=compression_ratio),
                parameters=[("compression_ratio|r", "dimensionless",
                             "shock compression ratio")],
                returns=("p", "dimensionless"),
                reference=("N(E) ~ E^-p with p = (r+2)/(r-1) (Bell 1978; "
                           "Blandford & Ostriker 1978); p = 2 for r = 4"),
                test_ref="test_domain_capabilities.py",
                self_check=({"compression_ratio": 4.0}, 2.0, 1e-12),
            ),
            ComputationalCapability(
                name="dsa_radio_spectral_index",
                description=("Synchrotron radio spectral index from the "
                             "relativistic electron energy index"),
                function=lambda electron_index: synch.radio_spectral_index(
                    electron_index=electron_index),
                parameters=[("electron_index|p", "dimensionless",
                             "electron energy spectral index")],
                returns=("alpha", "dimensionless"),
                reference=("S_nu ~ nu^-alpha with alpha = (p-1)/2; "
                           "alpha = 0.5 for p = 2"),
                test_ref="test_domain_capabilities.py",
                self_check=({"electron_index": 2.0}, 0.5, 1e-12),
            ),
        ]


# Factory function
def create_supernovae_domain() -> SupernovaeDomain:
    """Create a Supernovae domain instance"""
    return SupernovaeDomain()


# Domain registration
try:
    register_domain(SupernovaeDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
