"""
Nuclear Astrophysics Domain Module for ASTRA

Nuclear reaction rates, nucleosynthesis pathways, explosive burning.

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation.

`astra_core.physics.nuclear_astro` was **not** covered by the August 2026
physics audit, so it was read line by line before anything here was wired, and
only the semi-empirical mass formula survived that reading.  The rest of the
module is left unwired on purpose -- see `08_domains_batch_C.md` for the
per-routine reasoning, and in brief:

  * `nuclear_reaction_rate` -- the body says "This is a simplified
    placeholder" and returns
    ``screening * exp(-E_coul/kT) / (m_red T)^(2/3)``.  That is not the Gamow
    rate: the tunnelling factor exp(-3 E_G^(1/3)/(kT)^(1/3)) is absent
    entirely, `S(E)` never appears, and the result is not in cm^3/s as the
    docstring claims.  Wiring it would attach a formula citation to a number
    that the formula does not produce.
  * `pp_chain_energy`, `cno_cycle_energy`, `triple_alpha_rate` -- power-law /
    exponential fits whose leading coefficients (2.4e6, 8e27, 1e8) have no
    stated source; the code itself labels them "approximate" and "Very
    approximate", and T8^40 is a local expansion valid only near T8 = 1.
  * `r_process_path`, `supernova_yields` -- dictionaries of hardcoded
    abundances/yields multiplied by a scale factor.  There is no
    nucleosynthesis calculation of any kind behind them.
  * `nuclear_matter_eos`, `tov_equation` -- ``P = K dRho/(rho_0 * 1.6e-6)`` is
    not dimensionally a pressure, and the "TOV solution" is a hand-written
    scaling relation anchored on 1.4 M_sun / 12 km, not an integration.
  * `mass_defect` -- returns the right number (BE/931.5 amu) but only because
    a grams-minus-amu subtraction cancels; the intermediate `mass_nucleus` is
    dimensionally meaningless.  Flagged rather than wired.

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


class NuclearAstrophysicsDomain(ComputationalDomainModule):
    """
    Nuclear astrophysics.

    Only the Bethe-Weizsaecker semi-empirical mass formula in
    `astra_core.physics.nuclear_astro` is wired: it is a closed-form
    expression with standard published coefficients that can be, and has
    been, reproduced by hand.  Everything else in that module is a
    placeholder, an uncited fit or a hardcoded table (see the module
    docstring), so this domain reports far fewer capabilities than its
    original `capabilities` list advertised.
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="nuclear_astrophysics",
            version="2.0.0",
            dependencies=[],
            description=("Nuclear reaction rates, nucleosynthesis pathways, "
                         "explosive burning"),
            keywords=[
                'nuclear astrophysics', 'nuclear reactions', 'nucleosynthesis',
                'explosive_burning', 'reaction_rates',
                # extensions matching the wired computations
                'binding energy', 'semi-empirical mass formula',
                'bethe-weizsacker', 'liquid drop model', 'nuclear stability',
            ],
            capabilities=[
                'binding_energy', 'binding_energy_per_nucleon',
            ],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising nuclear_astrophysics domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...physics.nuclear_astro import NuclearAstrophysics

        # UNITS.  `binding_energy` returns MeV (total, not per nucleon) and
        # takes dimensionless A and Z.  Both self_checks were recomputed from
        # the Bethe-Weizsaecker expression written out again in
        # `astra_baseline_audit_aug2026/repro/batchC_hand_values.py`.
        # NOTE: the SEMF is a *model*.  B(Fe-56) = 495.38 MeV here against the
        # measured 492.25 MeV (0.6%); the self_check pins the formula and its
        # units, not the experimental value.
        semf = NuclearAstrophysics.binding_energy

        return [
            ComputationalCapability(
                name="binding_energy",
                description=("Nuclear binding energy from the semi-empirical "
                             "mass formula"),
                function=lambda mass_number, atomic_number: float(
                    semf(mass_number=int(mass_number),
                         atomic_number=int(atomic_number))),
                parameters=[
                    ("mass_number|a", "dimensionless", "mass number A"),
                    ("atomic_number|z", "dimensionless", "atomic number Z"),
                ],
                returns=("B", "MeV"),
                reference=("B = a_v A - a_s A^(2/3) - a_c Z(Z-1)/A^(1/3) "
                           "- a_a (A-2Z)^2/A + delta, with "
                           "(a_v,a_s,a_c,a_a,a_p) = "
                           "(15.75, 17.8, 0.711, 23.7, 11.18) MeV "
                           "(Bethe & Weizsaecker liquid-drop model; "
                           "Wapstra coefficients, Krane 1988 tab. 3.2)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"mass_number": 56.0, "atomic_number": 26.0},
                            495.38370, 1e-6),
            ),
            ComputationalCapability(
                name="binding_energy_per_nucleon",
                description=("Binding energy per nucleon from the "
                             "semi-empirical mass formula"),
                function=lambda mass_number, atomic_number: float(
                    semf(mass_number=int(mass_number),
                         atomic_number=int(atomic_number))) / mass_number,
                parameters=[
                    ("mass_number|a", "dimensionless", "mass number A"),
                    ("atomic_number|z", "dimensionless", "atomic number Z"),
                ],
                returns=("B/A", "MeV/nucleon"),
                reference=("B/A with B from the semi-empirical mass formula "
                           "(coefficients as in `binding_energy`); the curve "
                           "peaks near A = 56-62"),
                test_ref="test_domain_capabilities.py",
                self_check=({"mass_number": 56.0, "atomic_number": 26.0},
                            8.8461375, 1e-6),
            ),
        ]


# Factory function
def create_nuclear_astrophysics_domain() -> NuclearAstrophysicsDomain:
    """Create a Nuclear Astrophysics domain instance"""
    return NuclearAstrophysicsDomain()


# Domain registration
try:
    register_domain(NuclearAstrophysicsDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
