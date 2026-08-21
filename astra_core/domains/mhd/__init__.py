"""
Magnetohydrodynamics Domain Module for ASTRA

Magnetic field evolution, dynamo theory, reconnection.

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation.  It now wraps the
routines of :mod:`astra_core.astro_physics.turbulence_analysis` that the
August 2026 physics audit verified numerically -- Davis-Chandrasekhar-Fermi
field strength, sonic Mach number, turbulent dissipation rate and the sonic
scale -- and reports a confidence derived from whether a computation actually
ran.

NOT wired, deliberately (see `08_domains_batch_C.md`):
  * `StructureFunctionAnalysis.compute_2d` -- audit H11, the lag vector is
    truncated to integers and S_2(1) is 11x low.  Unfixed in this tree.
  * `SpectralPCA` -- audit H12, subtracts a global scalar mean.  Unfixed.
  * dynamo and reconnection: no numerical implementation exists anywhere in
    this codebase, so nothing is offered for them rather than authored prose.

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


class MagnetohydrodynamicsDomain(ComputationalDomainModule):
    """
    Magnetohydrodynamics of the magnetised, turbulent ISM.

    Backed by `astro_physics.turbulence_analysis`, whose DCF, sonic Mach
    number, dissipation rate and sonic-scale routines were verified against
    hand-derived values during the August 2026 audit (the audit's section (e)
    lists "DCF B_pos ... matches my hand value to 4 decimals" and "moments,
    sonic Mach number, eps = sigma^3/L, sonic scale L M^{-1/p} all correct").
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="mhd",
            version="2.0.0",
            dependencies=[],
            description="Magnetic field evolution, dynamo theory, reconnection",
            keywords=[
                'mhd', 'magnetohydrodynamics', 'magnetic field', 'dynamo',
                'reconnection',
                # extensions matching the wired computations
                'davis-chandrasekhar-fermi', 'dcf', 'polarization dispersion',
                'field strength', 'sonic mach number', 'turbulence',
                'dissipation rate', 'sonic scale',
            ],
            capabilities=[
                'dcf_field_strength', 'sonic_mach_number',
                'turbulent_dissipation_rate', 'sonic_scale',
            ],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising mhd domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.turbulence_analysis import (
            DavisChandrasekharFermi, TurbulenceStatistics,
        )

        # UNITS.  `DavisChandrasekharFermi.field_strength` takes sigma_theta in
        # DEGREES, sigma_los in km/s and a NUMBER density in cm^-3 (it forms
        # rho = n mu m_H internally with mu = 2.33 and m_H = 1.673e-24 g) and
        # returns a dict; 'B_pos_G' is in gauss and 'B_pos_uG' in microgauss.
        # `TurbulenceStatistics` takes km/s and pc and returns km/s, a pure
        # number, erg/g/s and pc respectively.  Every self_check below was
        # recomputed from the cited formula in
        # `astra_baseline_audit_aug2026/repro/batchC_hand_values.py`, which
        # imports nothing from this package.
        dcf = DavisChandrasekharFermi(correction_factor=0.5)

        def _stats(temperature: float) -> TurbulenceStatistics:
            return TurbulenceStatistics(temperature_k=temperature,
                                        mean_molecular_weight=2.33)

        return [
            ComputationalCapability(
                name="dcf_field_strength",
                description=("Davis-Chandrasekhar-Fermi plane-of-sky magnetic "
                             "field strength from polarization angle scatter"),
                function=lambda sigma_theta, sigma_los, density: dcf.
                field_strength(sigma_theta_deg=sigma_theta,
                               sigma_los_kms=sigma_los,
                               volume_density_cm3=density)['B_pos_uG'],
                parameters=[
                    ("sigma_theta|sigma_pol|polarization_dispersion", "deg",
                     "polarization position-angle dispersion"),
                    ("sigma_los|sigma_v|sigma", "km/s",
                     "line-of-sight velocity dispersion"),
                    ("density|n_h2|n", "cm^-3", "total particle number density"),
                ],
                returns=("B_pos", "microgauss"),
                reference=("B_pos = f sqrt(4 pi rho) sigma_los / sigma_theta, "
                           "f = 0.5, rho = n mu m_H (Davis 1951; Chandrasekhar "
                           "& Fermi 1953; Ostriker, Stone & Gammie 2001)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"sigma_theta": 10.0, "sigma_los": 0.5,
                             "density": 1e4}, 100.2521, 1e-4),
            ),
            ComputationalCapability(
                name="sonic_mach_number",
                description="Sonic Mach number of a turbulent molecular cloud",
                function=lambda velocity_dispersion, temperature: _stats(
                    temperature).mach_number(sigma_kms=velocity_dispersion),
                parameters=[
                    ("velocity_dispersion|sigma_v|sigma", "km/s",
                     "1-D velocity dispersion"),
                    ("temperature|t_kin|t", "K", "gas kinetic temperature"),
                ],
                returns=("M_s", "dimensionless"),
                reference=("M_s = sigma / c_s, c_s = sqrt(k_B T / (mu m_H)), "
                           "mu = 2.33"),
                test_ref="test_domain_capabilities.py",
                self_check=({"velocity_dispersion": 2.0, "temperature": 10.0},
                            10.62574, 1e-4),
            ),
            ComputationalCapability(
                name="turbulent_dissipation_rate",
                description=("Turbulent kinetic energy dissipation rate per "
                             "unit mass"),
                function=lambda velocity_dispersion, scale: _stats(
                    10.0).dissipation_rate(sigma_kms=velocity_dispersion,
                                           scale_pc=scale),
                parameters=[
                    ("velocity_dispersion|sigma_v|sigma", "km/s",
                     "1-D velocity dispersion"),
                    ("scale|driving_scale|length", "pc", "outer/driving scale"),
                ],
                returns=("epsilon", "erg/g/s"),
                reference="epsilon = sigma^3 / L (Kolmogorov scaling)",
                test_ref="test_domain_capabilities.py",
                self_check=({"velocity_dispersion": 2.0, "scale": 1.0},
                            2.592353e-3, 1e-4),
            ),
            ComputationalCapability(
                name="sonic_scale",
                description=("Scale at which supersonic turbulence becomes "
                             "subsonic"),
                function=lambda velocity_dispersion, driving_scale,
                temperature: _stats(temperature).sonic_scale_pc(
                    sigma_kms=velocity_dispersion,
                    driving_scale_pc=driving_scale,
                    linewidth_slope=0.5),
                parameters=[
                    ("velocity_dispersion|sigma_v|sigma", "km/s",
                     "1-D velocity dispersion at the driving scale"),
                    ("driving_scale|scale|length", "pc", "driving scale"),
                    ("temperature|t_kin|t", "K", "gas kinetic temperature"),
                ],
                returns=("l_s", "pc"),
                reference=("l_s = L M_s^(-1/p) from sigma(l) = sigma_L (l/L)^p "
                           "with p = 0.5 (Larson linewidth-size law)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"velocity_dispersion": 2.0, "driving_scale": 1.0,
                             "temperature": 10.0}, 8.856902e-3, 1e-4),
            ),
        ]


# Factory function
def create_mhd_domain() -> MagnetohydrodynamicsDomain:
    """Create a Magnetohydrodynamics domain instance"""
    return MagnetohydrodynamicsDomain()


# Domain registration
try:
    register_domain(MagnetohydrodynamicsDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
