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
Photoionization Domain Module for ASTRA

Cross-sections, chemistry networks, PDRs/XDRs

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation. It now wraps
:mod:`astra_core.astro_physics.hii_region_physics`, whose Stroemgren radius,
Case A/B recombination coefficients, Case B Balmer decrement (2.860 / 0.469 /
0.259) and L(Hbeta)/Q_H = 4.78e-13 were all verified exactly during the
August 2026 audit.

Every capability carries a `self_check` recomputed by hand from the formula in
its `reference` field.

Deliberately NOT wired: `ionizing_photon_rate` (now a log-linear interpolation
of Martins, Schaerer & Hillier 2005 Table 1 rather than the pre-audit fit that
was 713x high at 50 kK - correct, but it is tabulated data rather than a
formula, so this wrapper cannot verify it against an independent hand
derivation), and `NebularDiagnosticsCalculator.oxygen_abundance` /
`oiii_temperature` (repaired under B-HII-2/B-HII-3 but not independently
re-derived here). Photoionization *cross-sections* (Verner-style fits),
photodissociation rates and PDR/XDR chemistry networks are not implemented in
this codebase; the `PDRInterface` in `radiative_transfer` provides only
heating-rate and field-conversion constants, not a network.

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


class PhotoionizationDomain(ComputationalDomainModule):
    """
    Photoionized nebulae: Stroemgren spheres and Case B recombination.

    Backed by `astro_physics.hii_region_physics`.
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="photoionization",
            version="2.0.0",
            dependencies=[],
            description="Cross-sections, chemistry networks, PDRs/XDRs",
            keywords=['photoionization', 'photodissociation', 'cross_section',
                      'pdr', 'xdr', 'stromgren', 'stroemgren',
                      'recombination', 'case b', 'balmer decrement',
                      'hii region', 'ionizing photons', 'h beta'],
            capabilities=['photoionization_cross_sections',
                          'photodissociation_rates', 'pdr_modeling',
                          'xdr_modeling'],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising photoionization domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.hii_region_physics import (
            RecombinationCoefficients,
            RecombinationLines,
            StromgrenSphere,
        )

        # NOTE ON UNITS. `hii_region_physics` is CGS: `stromgren_radius`
        # returns cm and `recombination_time` returns s. The wrappers below
        # convert to pc and yr, and every self_check pins the conversion.
        PC = 3.0856775814913673e18     # cm
        YR = 3.155693e7                # s

        sphere = StromgrenSphere()
        recomb = RecombinationCoefficients()

        return [
            ComputationalCapability(
                name="stromgren_radius",
                description=("Stroemgren radius of an ionization-bounded HII "
                             "region"),
                function=lambda q_h, n_e, temperature=1e4: float(
                    sphere.stromgren_radius(q_h, n_e, temperature)) / PC,
                parameters=[
                    ("q_h|ionizing_photon_rate", "s^-1",
                     "rate of H-ionizing photons"),
                    ("n_e|electron_density", "cm^-3", "electron density"),
                    ("temperature|t_e", "K", "electron temperature"),
                ],
                returns=("R_S", "pc"),
                reference=("R_S = [3 Q_H / (4 pi alpha_B(T) n_e^2)]^(1/3) "
                           "(Stroemgren 1939)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"q_h": 1e49, "n_e": 100.0, "temperature": 1e4},
                            3.1539393, 1e-3),
            ),
            ComputationalCapability(
                name="case_b_recombination_rate",
                description=("Case B hydrogen recombination coefficient "
                             "alpha_B(T)"),
                function=lambda temperature: float(recomb.alpha_B(temperature)),
                parameters=[("temperature|t_e", "K",
                             "electron temperature")],
                returns=("alpha_B", "cm^3/s"),
                reference=("alpha_B = 2.59e-13 t4^(-0.833 - 0.034 ln t4), "
                           "t4 = T/1e4 (Osterbrock & Ferland 2006)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"temperature": 8000.0}, 3.1137989e-13, 1e-3),
            ),
            ComputationalCapability(
                name="recombination_time",
                description="Recombination timescale of ionized hydrogen",
                function=lambda n_e, temperature=1e4: float(
                    sphere.recombination_time(n_e, temperature)) / YR,
                parameters=[
                    ("n_e|electron_density", "cm^-3", "electron density"),
                    ("temperature|t_e", "K", "electron temperature"),
                ],
                returns=("t_rec", "yr"),
                reference="t_rec = 1 / (alpha_B(T) n_e)",
                test_ref="test_domain_capabilities.py",
                self_check=({"n_e": 100.0, "temperature": 1e4},
                            1223.50427, 1e-3),
            ),
            ComputationalCapability(
                name="balmer_decrement",
                description=("Case B H-alpha / H-beta intensity ratio at the "
                             "given electron temperature"),
                function=lambda temperature: float(
                    RecombinationLines(temperature).emissivity_ratio(
                        'Halpha', 'Hbeta')),
                parameters=[("temperature|t_e", "K",
                             "electron temperature")],
                returns=("I(Ha)/I(Hb)", "dimensionless"),
                reference=("j(Ha)/j(Hb) = alpha_eff(Ha) nu_Ha / "
                           "[alpha_eff(Hb) nu_Hb], Case B "
                           "(Osterbrock & Ferland 2006, table 4.2)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"temperature": 1e4}, 2.86026642, 1e-3),
            ),
            ComputationalCapability(
                name="hbeta_luminosity",
                description=("H-beta luminosity of an ionization-bounded "
                             "nebula from its ionizing photon rate"),
                function=lambda q_h, temperature=1e4: float(
                    RecombinationLines(temperature).hbeta_luminosity(q_h)),
                parameters=[
                    ("q_h|ionizing_photon_rate", "s^-1",
                     "rate of H-ionizing photons"),
                    ("temperature|t_e", "K", "electron temperature"),
                ],
                returns=("L(Hbeta)", "erg/s"),
                reference=("L(Hb) = Q_H [alpha_eff(Hb)/alpha_B] h nu_Hb "
                           "= 4.78e-13 Q_H at 1e4 K"),
                test_ref="test_domain_capabilities.py",
                self_check=({"q_h": 1e49, "temperature": 1e4},
                            4.78043232e36, 1e-3),
            ),
        ]


def create_photoionization_domain() -> PhotoionizationDomain:
    """Create a Photoionization domain instance."""
    return PhotoionizationDomain()


try:
    register_domain(PhotoionizationDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
