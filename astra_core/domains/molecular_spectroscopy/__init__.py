"""
Molecular Spectroscopy Domain Module for ASTRA

Rotational/vibrational spectra, Einstein coefficients, line shapes

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation. It now wraps the
linear-rotor routines of
:mod:`astra_core.astro_physics.spectroscopic_databases`, which were rebuilt
under audit item C3 (the pre-audit Einstein A coefficients were 3.7e14 times
too large for CO and 27 times too small for HCN/HCO+/N2H+) and C4b (the
hardcoded rotational partition functions were wrong by -15% to +43%).

Every capability carries a `self_check` recomputed by hand from the formula in
its `reference` field.

Deliberately NOT wired, because the audit found the data fabricated rather
than measured: `CDMSDatabase.query_lines` / `LAMDADatabase` line *catalogues*
(the frequencies are computed from fitted constants, not retrieved, and are
now labelled `database="synthetic"`), and the synthesised collision-rate
coefficients (~14x low for CO 1-0 at 10 K, and Boltzmann-suppressed in the
wrong direction). Vibrational spectra and pressure/collisional line-shape
models have no implementation in this codebase at all.

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


class MolecularSpectroscopyDomain(ComputationalDomainModule):
    """
    Molecular spectroscopy of linear rotors.

    Backed by `astro_physics.spectroscopic_databases` (rest frequencies,
    Einstein A coefficients, rotational partition functions) and
    `astro_physics.radiative_transfer` (thermal line width).
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="molecular_spectroscopy",
            version="2.0.0",
            dependencies=[],
            description=("Rotational/vibrational spectra, Einstein "
                         "coefficients, line shapes"),
            keywords=['molecular spectroscopy', 'rotational', 'vibrational',
                      'einstein_coefficients', 'line_profiles',
                      'einstein a', 'partition function', 'linear rotor',
                      'rest frequency', 'centrifugal distortion',
                      'thermal linewidth', 'doppler broadening'],
            capabilities=['rotational_spectra', 'vibrational_spectra',
                          'einstein_coefficients', 'line_shape_models'],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising molecular_spectroscopy domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.radiative_transfer import LineProfileSynthesizer
        from ...astro_physics.spectroscopic_databases import (
            linear_rotor_einstein_a,
            linear_rotor_frequency,
            linear_rotor_partition_function,
        )

        # NOTE ON UNITS. The backend takes B, D and nu in MHz and returns MHz
        # and s^-1; `thermal_sigma` returns cm/s. The wrappers declare GHz and
        # km/s where that is the natural observational unit, and each
        # self_check pins the conversion (the CO J=1-0 reference values are
        # 115271.2018 MHz and A = 7.20e-8 s^-1).
        KMS = 1.0e5

        return [
            ComputationalCapability(
                name="rotational_line_frequency",
                description=("Rest frequency of a linear-rotor J -> J-1 "
                             "transition including centrifugal distortion"),
                function=lambda j_upper, b_const, d_const:
                    float(linear_rotor_frequency(int(j_upper), b_const,
                                                 d_const)),
                parameters=[
                    ("j_upper|j", "dimensionless",
                     "upper rotational quantum number"),
                    ("b_const|rotational_constant", "MHz",
                     "effective rotational constant B"),
                    ("d_const|distortion_constant", "MHz",
                     "centrifugal distortion constant D"),
                ],
                returns=("nu", "MHz"),
                reference=("nu(J -> J-1) = 2 B J - 4 D J^3 "
                           "(Townes & Schawlow 1955)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"j_upper": 1, "b_const": 57635.9681,
                             "d_const": 0.18350435}, 115271.202, 1e-6),
            ),
            ComputationalCapability(
                name="einstein_a_coefficient",
                description=("Einstein A coefficient of a linear-rotor "
                             "J -> J-1 electric dipole transition"),
                function=lambda frequency, dipole_moment, j_upper:
                    float(linear_rotor_einstein_a(frequency * 1e3,
                                                  dipole_moment,
                                                  int(j_upper))),
                parameters=[
                    ("frequency|nu", "GHz", "rest frequency of the line"),
                    ("dipole_moment|mu", "Debye",
                     "permanent electric dipole moment"),
                    ("j_upper|j", "dimensionless",
                     "upper rotational quantum number"),
                ],
                returns=("A_ul", "s^-1"),
                reference=("A = 64 pi^4 nu^3 mu^2 J / [3 h c^3 (2J+1)] "
                           "(Mangum & Shirley 2015, eqs. 10-11)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"frequency": 115.2712018,
                             "dipole_moment": 0.11011,
                             "j_upper": 1}, 7.20501361e-08, 1e-3),
            ),
            ComputationalCapability(
                name="rotational_partition_function",
                description=("Rotational partition function of a linear "
                             "molecule"),
                function=lambda temperature, b_const:
                    float(linear_rotor_partition_function(temperature,
                                                          b_const)),
                parameters=[
                    ("temperature|t_ex|t_kin", "K", "excitation temperature"),
                    ("b_const|rotational_constant", "MHz",
                     "effective rotational constant B"),
                ],
                returns=("Q_rot", "dimensionless"),
                reference=("Q_rot(T) = sum_J (2J+1) exp(-h B J(J+1)/kT), "
                           "no nuclear-spin factor (g_u = 2J+1 convention)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"temperature": 300.0, "b_const": 57635.9681},
                            108.790282, 1e-3),
            ),
            ComputationalCapability(
                name="thermal_linewidth",
                description=("Thermal (Doppler) 1-D velocity dispersion of a "
                             "molecular line"),
                function=lambda temperature, molecular_mass:
                    float(LineProfileSynthesizer.thermal_sigma(
                        temperature, molecular_mass)) / KMS,
                parameters=[
                    ("temperature|t_kin", "K", "kinetic temperature"),
                    ("molecular_mass|mass_amu", "amu",
                     "molecular mass (28 = CO)"),
                ],
                returns=("sigma_thermal", "km/s"),
                reference="sigma = sqrt(k_B T / m), FWHM = 2 sqrt(2 ln 2) sigma",
                test_ref="test_domain_capabilities.py",
                self_check=({"temperature": 10.0, "molecular_mass": 28.0},
                            0.0544926686, 1e-3),
            ),
        ]


def create_molecular_spectroscopy_domain() -> MolecularSpectroscopyDomain:
    """Create a Molecular Spectroscopy domain instance."""
    return MolecularSpectroscopyDomain()


try:
    register_domain(MolecularSpectroscopyDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
