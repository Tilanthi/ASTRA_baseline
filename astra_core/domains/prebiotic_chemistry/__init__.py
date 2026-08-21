"""
Prebiotic Chemistry Domain Module for ASTRA

Complex organics in space, ISM chemistry, delivery to planets.

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation.  It now wraps the
rate-coefficient forms of :mod:`astra_core.astro_physics.chemical_networks`,
which the August 2026 audit checked and found correct ("UMIST/KIDA rate forms
correct"), plus the Polanyi-Wigner thermal desorption rate that governs when
an ice mantle returns its complex organics to the gas phase.

NOT wired, deliberately (see `08_domains_batch_C.md`):
  * `ChemistrySolver.solve` / `.equilibrium`.  The factor-n_H error that made
    every reported timescale wrong (audit C17) was repaired in this branch and
    the t_eval overflow (H16) with it, but the solver returns an abundance
    vector for a whole network, not a scalar, and cannot be driven from a
    free-text query.
  * `Reaction.rate_coefficient` for `COSMIC_RAY_PHOTODISSOCIATION` -- audit
    H15 is still OPEN: the documented gamma/(1-omega) grain-albedo factor
    (~1e3-1e4) is missing from the return because `omega` is not carried by
    `Reaction`.  Those rates are orders of magnitude low and are not offered.
  * `GrainChemistry.freeze_out_timescale` -- its grain abundance
    (n_grain = 1.3e-12) disagrees with the 1e-12 used by the
    `GRAIN_ADSORPTION` branch of `Reaction.rate_coefficient` in the same file,
    and neither is sourced. Reported rather than wired.
  * COM/prebiotic synthesis networks: the module's two standard networks are
    a minimal C-O network and an N network; there is no glycine, sugar or
    amino-acid chemistry anywhere in this codebase, and none is invented here.

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


class PrebioticChemistryDomain(ComputationalDomainModule):
    """
    Interstellar chemistry underlying complex-organic (prebiotic) molecule
    formation: gas-phase rate coefficients and ice-mantle desorption.
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="prebiotic_chemistry",
            version="2.0.0",
            dependencies=[],
            description=("Complex organics in space, ISM chemistry, delivery "
                         "to planets"),
            keywords=[
                'prebiotic', 'organic chemistry', 'complex molecules',
                'life_ingredients', 'delivery_mechanisms',
                # extensions matching the wired computations
                'rate coefficient', 'arrhenius', 'kooij', 'umist', 'kida',
                'ion-molecule', 'neutral-neutral',
                'dissociative recombination', 'thermal desorption',
                'ice mantle', 'binding energy', 'sublimation',
            ],
            capabilities=[
                'arrhenius_rate_coefficient',
                'dissociative_recombination_rate',
                'thermal_desorption_rate',
            ],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising prebiotic_chemistry domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.chemical_networks import Reaction, ReactionType

        # UNITS.  `Reaction.rate_coefficient` returns cm^3/s for two-body
        # reactions and s^-1 for unary ones -- which of the two depends on
        # the ReactionType, so each wrapper below fixes the type and declares
        # the corresponding unit rather than leaving it to the caller.
        # `alpha` carries the units of the result; `gamma` is in KELVIN (an
        # activation energy E_a/k_B for gas-phase reactions, a binding energy
        # E_b/k_B for desorption).  NOTE: the backend CLIPS T to the
        # reaction's `temperature_range`, default (10, 41000) K.
        # Every self_check was recomputed from the cited formula in
        # `astra_baseline_audit_aug2026/repro/batchC_hand_values.py`.

        def _rate(rtype: ReactionType, alpha: float, beta: float,
                  gamma: float, temperature: float) -> float:
            rxn = Reaction(reactants=[], products=[], reaction_type=rtype,
                           alpha=alpha, beta=beta, gamma=gamma)
            return float(rxn.rate_coefficient(T=temperature))

        return [
            ComputationalCapability(
                name="arrhenius_rate_coefficient",
                description=("Modified-Arrhenius (Kooij) gas-phase reaction "
                             "rate coefficient"),
                function=lambda alpha, beta, gamma, temperature: _rate(
                    ReactionType.NEUTRAL_NEUTRAL, alpha, beta, gamma,
                    temperature),
                parameters=[
                    ("alpha|a_rate", "cm^3/s",
                     "rate coefficient at 300 K with beta = gamma = 0"),
                    ("beta|b_rate", "dimensionless", "temperature exponent"),
                    ("gamma|e_activation", "K",
                     "activation energy divided by k_B"),
                    ("temperature|t_kin|t", "K", "gas kinetic temperature"),
                ],
                returns=("k", "cm^3/s"),
                reference=("k = alpha (T/300)^beta exp(-gamma/T) "
                           "(UMIST: McElroy et al. 2013; KIDA: Wakelam et al. "
                           "2012); T is clipped to the reaction's validity "
                           "range, by default 10-41000 K"),
                test_ref="test_domain_capabilities.py",
                self_check=({"alpha": 1e-10, "beta": 0.5, "gamma": 100.0,
                             "temperature": 300.0}, 7.1653131e-11, 1e-6),
            ),
            ComputationalCapability(
                name="dissociative_recombination_rate",
                description=("Dissociative recombination rate coefficient of "
                             "a molecular ion with an electron"),
                function=lambda alpha, beta, temperature: _rate(
                    ReactionType.DISSOCIATIVE_RECOMBINATION, alpha, beta,
                    0.0, temperature),
                parameters=[
                    ("alpha|a_rate", "cm^3/s", "rate coefficient at 300 K"),
                    ("beta|b_rate", "dimensionless", "temperature exponent"),
                    ("temperature|t_kin|t", "K", "electron temperature"),
                ],
                returns=("k_DR", "cm^3/s"),
                reference=("k = alpha (T/300)^beta with no activation "
                           "barrier (UMIST/KIDA dissociative-recombination "
                           "form); beta is typically about -0.7"),
                test_ref="test_domain_capabilities.py",
                self_check=({"alpha": 2.4e-7, "beta": -0.69,
                             "temperature": 30.0}, 1.1754692e-6, 1e-6),
            ),
            ComputationalCapability(
                name="thermal_desorption_rate",
                description=("First-order thermal sublimation rate of a "
                             "species from an icy grain mantle"),
                function=lambda binding_energy, dust_temperature: _rate(
                    ReactionType.GRAIN_DESORPTION, 0.0, 0.0, binding_energy,
                    dust_temperature),
                parameters=[
                    ("binding_energy|e_bind|e_b", "K",
                     "binding energy divided by k_B"),
                    ("dust_temperature|t_dust|t_d", "K", "dust temperature"),
                ],
                returns=("k_des", "1/s"),
                reference=("k = nu_0 exp(-E_b/T_d) with nu_0 = 1e12 s^-1 "
                           "(Polanyi-Wigner first-order desorption; Hasegawa, "
                           "Herbst & Leung 1992); E_b/k_B = 1150 K for CO and "
                           "5700 K for H2O in this module's species table"),
                test_ref="test_domain_capabilities.py",
                self_check=({"binding_energy": 1150.0,
                             "dust_temperature": 20.0}, 1.0667614e-13, 1e-6),
            ),
        ]


# Factory function
def create_prebiotic_chemistry_domain() -> PrebioticChemistryDomain:
    """Create a Prebiotic Chemistry domain instance"""
    return PrebioticChemistryDomain()


# Domain registration
try:
    register_domain(PrebioticChemistryDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
