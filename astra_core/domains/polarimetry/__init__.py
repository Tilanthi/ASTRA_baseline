"""
Polarimetry Domain Module for ASTRA

Magnetic field measurements, synchrotron radiation, dust alignment.

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation. It now wraps
:mod:`astra_core.astro_physics.turbulence_analysis` (Davis-Chandrasekhar-
Fermi) and the synchrotron layer of
:mod:`astra_core.astro_physics.supernova_remnant_physics`, and reports a
confidence derived from whether a computation actually ran.

Scope of what is wired
----------------------
The routines that reproduced independent hand values in the August 2026
audit: the DCF plane-of-sky field strength (200.50 uG against a hand value
of 200.53 uG), the diffusive-shock-acceleration index relations
(p = (r+2)/(r-1), alpha = (p-1)/2, exact at r = 2.5, 3, 4, 6) and the
synchrotron cooling time (7.7384e12 s at B = 100 uG, gamma = 1e4, against
3 m_e c / (4 sigma_T U_B gamma) and the classic 7.74e8/(B^2 gamma)).

NOT wired: `DavisChandrasekharFermi.polarization_dispersion` and
`HistogramRelativeOrientations.analyze` are correct but take an angle array
and an image respectively, so they cannot be driven from a query;
`SynchrotronEmission.surface_brightness` is excluded (audit B-SNR-12: it is
dimensionally wrong -- erg/cm^2 rather than erg/s/cm^2/Hz/sr -- and uses
the wrong frequency exponent).

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


class PolarimetryDomain(ComputationalDomainModule):
    """
    Polarimetry: magnetic field diagnostics.

    Backed by `astro_physics.turbulence_analysis.DavisChandrasekharFermi`
    and `astro_physics.supernova_remnant_physics.SynchrotronEmission`.
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="polarimetry",
            version="2.0.0",
            dependencies=[],
            description=("Magnetic field measurements, synchrotron "
                         "radiation, dust alignment"),
            keywords=[
                "polarimetry", "polarization", "magnetic field",
                "synchrotron", "dust alignment", "davis-chandrasekhar-fermi",
                "dcf", "position angle", "spectral index", "cooling time",
                "plane of sky",
            ],
            capabilities=[
                "dcf_field_strength", "synchrotron_spectral_index",
                "synchrotron_cooling_time",
            ],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising polarimetry domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.supernova_remnant_physics import (
            C_LIGHT, M_ELECTRON, SynchrotronEmission,
        )
        from ...astro_physics.turbulence_analysis import (
            DavisChandrasekharFermi,
        )

        # NOTE ON UNITS. `DavisChandrasekharFermi.field_strength` takes
        # DEGREES, km/s and cm^-3 and returns a dict carrying both Gauss and
        # microgauss -- the microgauss entry is used, so the declared unit is
        # the one actually returned. `SynchrotronEmission.cooling_time` is
        # CGS: it takes an electron ENERGY IN ERG (not a Lorentz factor) and
        # a field in Gauss, and returns SECONDS; the wrapper converts
        # gamma -> E = gamma m_e c^2 and s -> yr.
        YR = 3.156e7          # s
        dcf = DavisChandrasekharFermi(correction_factor=0.5)
        synch = SynchrotronEmission()

        def dcf_field_strength(sigma_theta: float, sigma_v: float,
                               density: float) -> float:
            return float(dcf.field_strength(
                sigma_theta_deg=sigma_theta, sigma_los_kms=sigma_v,
                volume_density_cm3=density)['B_pos_uG'])

        def synchrotron_spectral_index(compression_ratio: float) -> float:
            p = synch.electron_spectrum_index(compression_ratio)
            return float(synch.radio_spectral_index(p))

        def synchrotron_cooling_time(magnetic_field: float,
                                     lorentz_factor: float) -> float:
            energy = lorentz_factor * M_ELECTRON * C_LIGHT ** 2
            return float(synch.cooling_time(energy, magnetic_field) / YR)

        return [
            ComputationalCapability(
                name="dcf_field_strength",
                description=("Plane-of-sky magnetic field from the "
                             "Davis-Chandrasekhar-Fermi method"),
                function=dcf_field_strength,
                parameters=[
                    ("sigma_theta|angle_dispersion", "deg",
                     "polarization position-angle dispersion"),
                    ("sigma_v|sigma_los|sigma", "km/s",
                     "line-of-sight velocity dispersion"),
                    ("density|n_h2|n", "cm^-3", "H2 number density"),
                ],
                returns=("B_pos", "microgauss"),
                reference=("B_pos = f sqrt(4 pi rho) sigma_v / sigma_theta, "
                           "f = 0.5, rho = n mu m_H with mu = 2.33 "
                           "(Davis 1951; Chandrasekhar & Fermi 1953; "
                           "Ostriker, Stone & Gammie 2001)"),
                test_ref="test_domain_capabilities.py",
                # rho = 3.8990e-20 g/cm^3, sqrt(4 pi rho) = 7.0004e-10,
                # sigma_theta = 0.1745329 rad:
                # 0.5 x 7.0004e-10 x 1e5 / 0.1745329 = 2.00534e-4 G
                self_check=({"sigma_theta": 10.0, "sigma_v": 1.0,
                             "density": 1e4}, 200.534, 1e-3),
            ),
            ComputationalCapability(
                name="synchrotron_spectral_index",
                description=("Synchrotron spectral index from the shock "
                             "compression ratio"),
                function=synchrotron_spectral_index,
                parameters=[("compression_ratio|r", "dimensionless",
                             "shock density compression ratio")],
                returns=("alpha", "dimensionless"),
                reference=("p = (r+2)/(r-1), alpha = (p-1)/2 = 1.5/(r-1) "
                           "for S_nu ~ nu^-alpha (Bell 1978; Blandford & "
                           "Ostriker 1978)"),
                test_ref="test_domain_capabilities.py",
                # r = 4 (strong adiabatic shock): 1.5/3 = 0.5 exactly.
                self_check=({"compression_ratio": 4.0}, 0.5, 1e-12),
            ),
            ComputationalCapability(
                name="synchrotron_cooling_time",
                description=("Synchrotron cooling time of a relativistic "
                             "electron"),
                function=synchrotron_cooling_time,
                parameters=[
                    ("magnetic_field|b_field|b", "G", "magnetic field"),
                    ("lorentz_factor|gamma", "dimensionless",
                     "electron Lorentz factor"),
                ],
                returns=("t_cool", "yr"),
                reference=("t_cool = 3 m_e c / (4 sigma_T U_B gamma), "
                           "U_B = B^2/(8 pi) (Rybicki & Lightman 1979, "
                           "eq. 6.7b)"),
                test_ref="test_domain_capabilities.py",
                # B = 100 uG, gamma = 1e4: U_B = 3.97887e-10 erg/cm^3,
                # t = 7.73800e12 s = 245184 yr.
                self_check=({"magnetic_field": 1e-4,
                             "lorentz_factor": 1e4}, 245184.0, 1e-3),
            ),
        ]


def create_polarimetry_domain() -> PolarimetryDomain:
    """Create a PolarimetryDomain instance."""
    return PolarimetryDomain()


# Domain registration
try:
    register_domain(PolarimetryDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
