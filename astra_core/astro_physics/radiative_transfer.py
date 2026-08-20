#!/usr/bin/env python3
"""
Radiative Transfer Solver Module for ASTRO-SWARM
=================================================

Comprehensive radiative transfer calculations for molecular line and
continuum emission in astrophysical environments.

Capabilities:
1. Non-LTE molecular excitation (RADEX-like statistical equilibrium)
2. Escape probability methods (uniform sphere, expanding sphere, slab)
3. Multi-layer radiative transfer
4. Line profile synthesis with turbulence
5. Dust continuum + line combination
6. PDR interface layer

Key References:
- van der Tak et al. 2007 (RADEX)
- Sobolev 1960 (escape probability)
- Rybicki & Lightman 1979 (RT fundamentals)
- Draine 2011 (ISM physics)

Author: Claude Code (ASTRO-SWARM)
Date: 2024-11
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from abc import ABC, abstractmethod
from scipy.integrate import quad, odeint, solve_ivp
from scipy.optimize import fsolve, brentq
from scipy.interpolate import interp1d
from scipy.special import expn
import warnings

# Physical Constants (CGS)
k_B = 1.38e-16          # Boltzmann constant (erg/K)
h_planck = 6.626e-27    # Planck constant (erg s)
c_light = 2.998e10      # Speed of light (cm/s)
m_H = 1.67e-24          # Hydrogen mass (g)
T_CMB = 2.7255          # CMB temperature (K)


# =============================================================================
# ESCAPE PROBABILITY GEOMETRIES
# =============================================================================

class EscapeGeometry(Enum):
    """Geometry for escape probability calculation"""
    UNIFORM_SPHERE = "uniform_sphere"
    EXPANDING_SPHERE = "expanding_sphere"  # LVG/Sobolev
    PLANE_PARALLEL = "plane_parallel"      # Slab geometry
    STATIC_SPHERE = "static_sphere"


def escape_probability(tau: float, geometry: EscapeGeometry) -> float:
    """
    Calculate photon escape probability for given optical depth and geometry.

    Parameters
    ----------
    tau : float
        Line-center optical depth
    geometry : EscapeGeometry
        Geometry assumption

    Returns
    -------
    beta : float
        Escape probability (0 to 1)
    """
    tau = np.abs(tau)

    if tau < 1e-10:
        return 1.0

    if geometry in (EscapeGeometry.UNIFORM_SPHERE, EscapeGeometry.STATIC_SPHERE):
        # Uniform (static, homogeneous) sphere, Osterbrock (1989) Appendix 2 --
        # the same expression RADEX uses for its "uniform sphere" geometry:
        #
        #   beta = 1.5/tau * [1 - 2/tau^2 + (2/tau + 2/tau^2) exp(-tau)]
        #
        # FIX(audit H7): the small-tau branch used the LVG series 1 - tau/2 +
        # tau^2/6. Expanding the expression above gives
        #   beta = 1 - 3 tau/8 + tau^2/10 - tau^3/120 + O(tau^4),
        # so the old branch was discontinuous by 1.16e-3 across tau = 0.01
        # (0.9950663 vs the exact 0.9962973 at tau = 0.0099).
        # The series is needed only to avoid catastrophic cancellation as
        # tau -> 0 (the 2/tau^2 terms).
        # FIX(audit, LOW): the tau > 50 branch returned 1.5/tau, discontinuous
        # by 8e-4 relative at tau = 50; the full expression has no cancellation
        # problem at large tau, so it is now used everywhere above tau = 0.01.
        if tau < 0.01:
            return 1.0 - 3.0*tau/8.0 + tau**2/10.0 - tau**3/120.0
        return 1.5/tau * (1.0 - 2.0/tau**2 + (2.0/tau + 2.0/tau**2) * np.exp(-tau))

    elif geometry == EscapeGeometry.EXPANDING_SPHERE:
        # LVG/Sobolev approximation: beta = (1 - exp(-tau)) / tau
        if tau < 0.01:
            return 1.0 - tau/2.0 + tau**2/6.0
        else:
            return (1.0 - np.exp(-tau)) / tau

    elif geometry == EscapeGeometry.PLANE_PARALLEL:
        # Plane-parallel slab: beta = (1 - exp(-3*tau)) / (3*tau)
        if tau < 0.01:
            return 1.0 - 1.5*tau + 1.5*tau**2
        else:
            return (1.0 - np.exp(-3.0*tau)) / (3.0*tau)

    # FIX(audit H6): STATIC_SPHERE is handled together with UNIFORM_SPHERE above.
    # It previously returned the uniform-sphere expression below tau = 50 and a
    # Doppler-core asymptote 1/(tau sqrt(ln(tau/sqrt(pi)))) above it, which is a
    # different physical assumption: beta jumped by x2.74 at tau = 50 (0.0300 ->
    # 0.0109) and diverged from the sphere result thereafter (x3.78 at tau=1000).
    # The static homogeneous sphere IS the Osterbrock/RADEX uniform-sphere case
    # for a frequency-independent (line-centre) optical depth, so the two are
    # now identical and continuous.
    # AUDIT-FLAG: a genuinely frequency-dependent static-sphere treatment (with
    # a Doppler profile and partial frequency redistribution, for which
    # beta ~ 1/(tau sqrt(pi ln tau)) at large tau; Osterbrock 1962, Hummer 1968)
    # is NOT implemented here and cannot be obtained by patching a branch onto
    # the grey formula -- it requires frequency redistribution in the solver.

    return 1.0


# =============================================================================
# MOLECULAR LEVEL POPULATIONS
# =============================================================================

@dataclass
class MolecularLevel:
    """Properties of a molecular energy level"""
    J: int                      # Rotational quantum number
    energy: float               # Energy above ground (K)
    weight: float               # Statistical weight (2J+1 for linear)


@dataclass
class CollisionRates:
    """Collision rate coefficients"""
    partner: str                # Collision partner (H2, He, e-, H)
    temperatures: np.ndarray    # Temperature grid (K)
    rates: np.ndarray          # Rate coefficients (cm³/s), shape (n_temps, n_trans)
    transitions: List[Tuple[int, int]]  # (upper, lower) level indices


@dataclass
class MolecularData:
    """Complete molecular data for RT calculations"""
    name: str
    levels: List[MolecularLevel]
    einstein_A: np.ndarray      # Einstein A coefficients (s⁻¹)
    frequencies: np.ndarray     # Transition frequencies (Hz)
    collision_rates: List[CollisionRates]
    # FIX(audit H8): (upper, lower) level indices for each entry of
    # einstein_A/frequencies, in the same order. LAMDA files list only the
    # allowed transitions, so this mapping cannot be inferred in general.
    # If left None the solver accepts only unambiguous layouts (all-pairs
    # packed order, or an adjacent dipole ladder) and otherwise raises.
    radiative_transitions: Optional[List[Tuple[int, int]]] = None


# =============================================================================
# STATISTICAL EQUILIBRIUM SOLVER
# (re-implemented 2026-08; the original body was lost to file truncation
#  before the August 2026 audit. Standard escape-probability method as in
#  RADEX/van der Tak et al. 2007, built on the surviving data structures.)
# =============================================================================

@dataclass
class ExcitationResult:
    """Level populations and derived line properties"""
    populations: np.ndarray              # Fractional level populations (n_levels,)
    optical_depths: np.ndarray           # Line-center optical depths (n_trans,)
    excitation_temperatures: np.ndarray  # Tex per transition (K)
    rayleigh_jeans_Tb: np.ndarray        # Line brightness temperature Tb (K)
    converged: bool
    n_iterations: int


class StatisticalEquilibriumSolver:
    """
    Solve the statistical equilibrium rate equations for molecular level
    populations using the (large velocity gradient) escape probability
    method.

    The rate matrix for each level i is

        dn_i/dt = sum_j [ n_j (C_ji + A_ji beta_ji + B_ji I_nu,j beta_ji)
                        - n_i (C_ij + A_ij beta_ij + B_ij I_nu,ij beta_ij) ]

    with stimulated emission/absorption driven by the CMB and escape
    probabilities beta = escape_probability(tau) computed self-consistently
    from the current populations. Populations are normalised to sum to one;
    LTE is the starting guess.
    """

    def __init__(self, molecular_data: MolecularData,
                 geometry: EscapeGeometry = EscapeGeometry.EXPANDING_SPHERE,
                 background_temperature: float = T_CMB):
        self.data = molecular_data
        self.geometry = geometry
        self.T_bg = background_temperature
        self.n_levels = len(molecular_data.levels)
        self._index_transitions()

    # ------------------------------------------------------------------ setup
    def _index_transitions(self):
        """Build flat radiative and collisional transition tables."""
        A = np.atleast_1d(np.asarray(self.data.einstein_A, dtype=float))
        nu = np.atleast_1d(np.asarray(self.data.frequencies, dtype=float))
        pairs = self._radiative_pairs(len(A))
        if len(nu) != len(A):
            raise ValueError(
                f"einstein_A has {len(A)} entries but frequencies has {len(nu)}")
        # Radiative transitions: (upper, lower, A_ul, freq)
        self.radiative = []
        for k, (up, low) in enumerate(pairs):
            if A[k] > 0:
                self.radiative.append((int(up), int(low), float(A[k]), float(nu[k])))
        # Collisional rates interpolated to T_kin, flattened per partner
        self._coll_cache = {}

    def _radiative_pairs(self, n_trans: int) -> List[Tuple[int, int]]:
        """
        Map each entry of einstein_A/frequencies onto its (upper, lower) levels.

        FIX(audit H8): the old `_radiative_index` *assumed* the all-pairs packed
        ordering k = up(up-1)/2 + low for every input. Real LAMDA files list only
        the allowed transitions, so for a 3-level linear rotor with
        einstein_A = [A_10, A_21] the solver silently assigned A_21 and nu_21 to
        the dipole-FORBIDDEN 2->0 transition and dropped the real 2->1 line.
        Nothing warned. The mapping is now either taken from the data
        (`MolecularData.radiative_transitions`) or inferred only when the array
        length makes the ordering unambiguous; anything else raises.

        Accepted layouts:
          * explicit `radiative_transitions` [(up, low), ...]  -- preferred;
          * n_trans == n(n-1)/2 : all-pairs packed order, k = up(up-1)/2 + low
            (zero-pad the forbidden entries; the solver skips A <= 0);
          * n_trans == n-1      : adjacent ladder, k -> (k+1, k), the LAMDA
            layout for a linear rotor with dipole selection rule dJ = 1.
        """
        n = self.n_levels
        explicit = getattr(self.data, 'radiative_transitions', None)
        if explicit is not None:
            pairs = [(int(u), int(l)) for u, l in explicit]
            if len(pairs) != n_trans:
                raise ValueError(
                    f"radiative_transitions has {len(pairs)} entries but "
                    f"einstein_A has {n_trans}")
            for up, low in pairs:
                if not (0 <= low < up < n):
                    raise ValueError(
                        f"invalid radiative transition ({up}, {low}) for "
                        f"{n} levels")
            return pairs

        n_packed = n * (n - 1) // 2
        if n_trans == n_packed:
            return [(up, low) for up in range(n) for low in range(up)]
        if n_trans == n - 1:
            return [(k + 1, k) for k in range(n - 1)]
        raise ValueError(
            f"cannot infer the (upper, lower) level pairs for {n_trans} "
            f"radiative transitions among {n} levels: supply "
            f"MolecularData.radiative_transitions explicitly (expected "
            f"{n_packed} for all-pairs packed order or {n - 1} for an "
            f"adjacent dipole ladder)")

    def _collisional_rates(self, T_kin: float) -> np.ndarray:
        """Total collisional rate matrix C[i, j] (s^-1 * n) at T_kin."""
        if T_kin in self._coll_cache:
            return self._coll_cache[T_kin]
        C = np.zeros((self.n_levels, self.n_levels))
        for cr in self.data.collision_rates:
            temps = np.asarray(cr.temperatures)
            for t_i, (up, low) in enumerate(cr.transitions):
                k_ul = np.interp(T_kin, temps, np.asarray(cr.rates)[:, t_i])
                # Detailed balance: k_lu = k_ul * (g_u/g_l) * exp(-dE/kT)
                g_up = self.data.levels[up].weight
                g_low = self.data.levels[low].weight
                dE = self.data.levels[up].energy - self.data.levels[low].energy
                k_lu = k_ul * (g_up / g_low) * np.exp(-dE / T_kin)
                C[up, low] += k_ul
                C[low, up] += k_lu
        self._coll_cache[T_kin] = C
        return C

    # ------------------------------------------------------------------ solve
    def optical_depth(self, populations: np.ndarray, up: int, low: int,
                      A_ul: float, freq: float, N_total: float,
                      dv: float) -> float:
        """
        Line-center optical depth for the LVG approximation

        tau = (c^3 / 8 pi nu^3) * A_ul * N_u / dv * (exp(h nu / k Tex) - 1)
        """
        N_u = populations[up] * N_total                   # upper-level column (cm^-2)
        c = c_light
        dE = (self.data.levels[up].energy - self.data.levels[low].energy)  # K
        if populations[low] <= 0 or populations[up] <= 0 or dE <= 0:
            return 0.0
        Tex = dE / np.log(populations[low] * self.data.levels[up].weight
                          / (populations[up] * self.data.levels[low].weight))
        tau = (c ** 3 / (8 * np.pi * freq ** 3)) * A_ul * N_u / dv * np.expm1(dE / Tex)
        return float(tau)

    def solve(self, T_kin: float, n_H2: float, N_mol: float, dv: float = 1.0,
              n_iterations: int = 100, tolerance: float = 1e-4) -> ExcitationResult:
        """
        Solve for level populations.

        Parameters
        ----------
        T_kin : kinetic temperature (K)
        n_H2 : molecular hydrogen density (cm^-3)
        N_mol : molecular column density (cm^-2); used to scale tau
        dv : velocity width / gradient for the LVG tau (km/s)
        """
        h, c, kB = h_planck, c_light, k_B
        levels = self.data.levels
        n = self.n_levels
        dv_cms = dv * 1e5  # km/s -> cm/s

        # LTE starting guess
        boltz = np.array([np.exp(-lv.energy / T_kin) * lv.weight for lv in levels])
        pop = boltz / boltz.sum()

        C = self._collisional_rates(T_kin)          # cm^3/s

        taus = np.zeros(len(self.radiative))
        converged = False
        it = 0
        for it in range(1, n_iterations + 1):
            # Steady-state rows:  sum_j R_{j->i} n_j  -  n_i sum_j R_{i->j} = 0
            M = np.zeros((n, n))
            for t_i, (up, low, A_ul, freq) in enumerate(self.radiative):
                dE = levels[up].energy - levels[low].energy      # K
                tau = self.optical_depth(pop, up, low, A_ul, freq, N_mol, dv_cms)
                taus[t_i] = tau
                beta = max(escape_probability(tau, self.geometry), 1e-8)
                # CMB photon occupation number at this frequency
                eta = 1.0 / np.expm1(dE / self.T_bg) if dE > 0 else 0.0
                g_up, g_low = levels[up].weight, levels[low].weight
                rate_dn = A_ul * beta * (1.0 + eta)              # spontaneous + stimulated
                rate_up = A_ul * (g_up / g_low) * eta * beta     # CMB absorption
                M[low, up] += rate_dn                            # up -> low inflow
                M[up, up] -= rate_dn                             # up loses molecules
                M[up, low] += rate_up                            # low -> up inflow
                M[low, low] -= rate_up                           # low loses molecules
            for i in range(n):
                for j in range(n):
                    if i != j and C[i, j] > 0:
                        M[j, i] += C[i, j] * n_H2     # collisional i -> j inflow
                        M[i, i] -= C[i, j] * n_H2     # i loses molecules
            M += np.eye(n) * 1e-30                    # isolated levels stay solvable
            # Replace one row with normalisation sum(pop) = 1
            M[-1, :] = 1.0
            rhs = np.zeros(n)
            rhs[-1] = 1.0
            try:
                new_pop = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                break
            if np.all(new_pop >= -1e-10) and np.isfinite(new_pop).all():
                new_pop = np.clip(new_pop, 0, None)
                new_pop /= new_pop.sum()
                delta = np.max(np.abs(new_pop - pop))
                pop = new_pop
                if delta < tolerance:
                    converged = True
                    break
            else:
                # damp towards previous solution
                pop = 0.5 * (pop + np.clip(new_pop, 0, None) / max(np.clip(new_pop, 0, None).sum(), 1e-30))

        # Derived quantities per transition
        Tex = np.zeros(len(self.radiative))
        Tb = np.zeros(len(self.radiative))
        for t_i, (up, low, A_ul, freq) in enumerate(self.radiative):
            dE = levels[up].energy - levels[low].energy
            with np.errstate(divide='ignore', invalid='ignore'):
                tex = (dE / np.log(pop[low] * levels[up].weight
                                  / (pop[up] * levels[low].weight))
                       if pop[up] > 0 and pop[low] > 0 else np.nan)
                Tex[t_i] = tex
                tau = taus[t_i]
                # Rayleigh-Jeans brightness temperature
                T0 = dE  # h nu / k in K
                if np.isfinite(tex) and tex > 0 and tau > 0:
                    source_fn = T0 / (np.exp(dE / tex) - 1.0)
                    bg = T0 / (np.exp(dE / self.T_bg) - 1.0)
                    Tb[t_i] = (source_fn - bg) * (1.0 - np.exp(-tau))
                else:
                    Tb[t_i] = 0.0

        return ExcitationResult(
            populations=pop,
            optical_depths=taus,
            excitation_temperatures=Tex,
            rayleigh_jeans_Tb=Tb,
            converged=converged,
            n_iterations=it,
        )


# =============================================================================
# LINE PROFILE SYNTHESIS
# =============================================================================

class LineProfileSynthesizer:
    """
    Velocity-resolved line profiles from excitation results.

    The line-center brightness temperature

        T_b,0 = [J(T_ex) - J(T_bg)] * (1 - exp(-tau))

    with J(T) = T0/(exp(T0/T) - 1) and T0 = h nu / k, is broadened by
    a Gaussian with the total width

        sigma_tot^2 = k T_kin / (m mol) + sigma_turb^2

    (thermal + non-thermal broadening added in quadrature).
    """

    def __init__(self, molecule_mass_amu: float = 28.0,
                 T_bg: float = T_CMB):
        """
        Args:
            molecule_mass_amu: molecular mass (amu); 28 = CO
            T_bg: background radiation temperature (K)
        """
        self.m_mass = molecule_mass_amu * 1.66054e-24   # g
        self.T_bg = T_bg

    @staticmethod
    def thermal_sigma(T_kin: float, molecule_mass_amu: float) -> float:
        """1-D thermal velocity dispersion (cm/s)."""
        m = molecule_mass_amu * 1.66054e-24
        return float(np.sqrt(k_B * T_kin / m))

    @classmethod
    def total_sigma(cls, T_kin: float, molecule_mass_amu: float,
                    sigma_turb_cms: float) -> float:
        """Thermal + turbulent broadening in quadrature (cm/s)."""
        return float(np.sqrt(cls.thermal_sigma(T_kin, molecule_mass_amu) ** 2
                             + sigma_turb_cms ** 2))

    def profile(self, velocities_cms: np.ndarray, T_ex: float, tau: float,
                T0_K: float, sigma_cms: float) -> np.ndarray:
        """
        Brightness-temperature profile T_b(v) (K).

        Args:
            velocities_cms: velocity grid (cm/s)
            T_ex: excitation temperature (K)
            tau: line-center optical depth
            T0_K: h nu / k of the transition (K)
            sigma_cms: total 1-D width (cm/s)

        Returns:
            T_b(v) with the same shape
        """
        v = np.asarray(velocities_cms, dtype=float)
        J = lambda T: T0_K / np.expm1(T0_K / T) if T > 0 else 0.0
        Tb0 = (J(T_ex) - J(self.T_bg)) * (1.0 - np.exp(-tau))
        return Tb0 * np.exp(-v ** 2 / (2.0 * sigma_cms ** 2))

    def integrated_intensity(self, T_ex: float, tau: float, T0_K: float,
                             sigma_cms: float) -> float:
        """Integral of T_b over velocity, T_b,0 sigma sqrt(2 pi) (K cm/s)."""
        J = lambda T: T0_K / np.expm1(T0_K / T) if T > 0 else 0.0
        Tb0 = (J(T_ex) - J(self.T_bg)) * (1.0 - np.exp(-tau))
        return float(Tb0 * sigma_cms * np.sqrt(2.0 * np.pi))


# =============================================================================
# DUST CONTINUUM RADIATIVE TRANSFER
# =============================================================================

class DustContinuumRT:
    """
    1-D dust continuum radiative transfer for a uniform slab:

        I_nu = B_nu(T_d) (1 - e^{-tau_nu}),  tau_nu = kappa_0 (nu/nu0)^beta Sigma

    covering the optically thin (I ~ B tau) and thick (I -> B) limits
    exactly, plus colour-temperature inversion by root finding.
    """

    def __init__(self, kappa_0: float = 0.1, beta: float = 2.0,
                 lambda_0_um: float = 250.0):
        """
        Args:
            kappa_0: opacity at lambda_0 (cm^2/g of gas+dust)
        """
        self.kappa_0 = kappa_0
        self.beta = beta
        self.nu_0 = c_light / (lambda_0_um * 1e-4)

    def opacity_nu(self, nu_hz: float) -> float:
        """kappa_nu = kappa_0 (nu/nu0)^beta (cm^2/g)."""
        return self.kappa_0 * (nu_hz / self.nu_0) ** self.beta

    def b_nu(self, nu_hz: float, temperature: float) -> float:
        """Planck B_nu (erg/s/cm^2/Hz/sr)."""
        x = h_planck * nu_hz / (k_B * temperature)
        return 2.0 * h_planck * nu_hz ** 3 / c_light ** 2 / np.expm1(x)

    def intensity(self, nu_hz: float, temperature: float,
                  surface_density: float) -> float:
        """
        Emergent specific intensity (erg/s/cm^2/Hz/sr).

        Args:
            surface_density: gas+dust mass surface density (g/cm^2)
        """
        tau = self.opacity_nu(nu_hz) * surface_density
        return self.b_nu(nu_hz, temperature) * (-np.expm1(-tau))

    def color_temperature(self, nu1: float, nu2: float, I1: float,
                          I2: float) -> float:
        """Temperature for which the model matches I1/I2 (optically
        thin; the ratio is tau-independent only if thick at both
        bands, in which case it is exactly B_nu ratio)."""
        ratio = I1 / I2

        def f(t):
            tau1 = 1.0
            tau2 = (self.opacity_nu(nu2) / self.opacity_nu(nu1))
            return ((self.b_nu(nu1, t) * tau1)
                    / (self.b_nu(nu2, t) * tau2) - ratio)

        return float(brentq(f, 3.0, 2000.0))

    def surface_density_from_intensity(self, nu_hz: float,
                                       temperature: float,
                                       intensity: float) -> float:
        """Invert I_nu = B_nu(1 - e^-tau) for the surface density."""
        b = self.b_nu(nu_hz, temperature)
        frac = np.clip(intensity / b, 0.0, 1.0 - 1e-12)
        tau = -np.log(1.0 - frac)
        return float(tau / self.opacity_nu(nu_hz))


# =============================================================================
# PDR INTERFACE
# =============================================================================

class PDRInterface:
    """
    Photon-dominated-region quantities.

    Includes only relations with well-established constants:

    - Photoelectric heating:  Gamma_PE = 1.3e-24 eps G0 n_H erg/cm^3/s
      (eps ~ 0.05 for the warm neutral medium; Draine 2011).
    - Habing/Draine FUV field equivalences: G0 = 1 <-> 1.6e-3 erg/s/cm^2
      (Habing 1968) <-> 2.7e-3 erg/s/cm^2 (Draine 1978).
    - Gas-dust extinction relation N_H = 1.87e21 A_V cm^-2 (Bohlin et
      al. 1978), with FUV attenuation approximated by tau_FUV ~ 2.5 A_V.
    - H2 grain-surface formation rate R n(H) n(HI), R = 3e-17 cm^3/s.

    The thermal-balance solver is generic: any user-supplied cooling
    function Lambda(T) can be passed, avoiding unverifiable fitted
    cooling coefficients.
    """

    HABING_ERG = 1.6e-3       # erg/s/cm^2 at G0 = 1
    DRAINE_ERG = 2.7e-3       # erg/s/cm^2 at G0 = 1
    N_H_PER_AV = 1.87e21      # cm^-2 mag^-1 (Bohlin et al. 1978)
    FUV_CROSS_SECTION = 2.5 * 1.87e21  # cm^-2 per mag * ~2.5 FUV/optical
    R_H2_FORMATION = 3e-17    # cm^3/s

    # fine-structure transition constants (exact level spacings:
    # T0 = 1.4388 cm K / lambda_cm; A-values from the literature)
    CII_158 = {'T0_K': 91.21, 'A_ul_s': 2.3e-6, 'g_up': 4, 'g_low': 2}
    OI_63 = {'T0_K': 227.72, 'A_ul_s': 8.9e-5, 'g_up': 5, 'g_low': 3}

    def photoelectric_heating(self, G0: float, n_H: float,
                              efficiency: float = 0.05) -> float:
        """Gamma_PE (erg/cm^3/s)."""
        return 1.3e-24 * efficiency * G0 * n_H

    @classmethod
    def habing_to_G0(cls, intensity_erg: float) -> float:
        return float(intensity_erg / cls.HABING_ERG)

    @classmethod
    def draine_to_G0(cls, intensity_erg: float) -> float:
        return float(intensity_erg / cls.DRAINE_ERG)

    @classmethod
    def attenuated_G0(cls, G0: float, A_V: float) -> float:
        """FUV field after dust extinction, G0 e^{-tau_FUV}."""
        return float(G0 * np.exp(-2.5 * A_V))

    @classmethod
    def A_V_from_N_H(cls, N_H: float) -> float:
        return float(N_H / cls.N_H_PER_AV)

    def h2_formation_rate(self, n_H: float, f_HI: float = 1.0) -> float:
        """H2 formation rate per volume (cm^-3 s^-1)."""
        return self.R_H2_FORMATION * n_H * (f_HI * n_H)

    @staticmethod
    def thermal_balance(heating: float, cooling_fn, T_low: float = 5.0,
                        T_high: float = 1e4) -> Dict[str, Any]:
        """
        Solve heating = cooling(T) for the equilibrium temperature.

        Args:
            heating: volumetric heating rate (erg/cm^3/s)
            cooling_fn: callable Lambda(T) -> erg/cm^3/s
            T_low, T_high: bracketing range for the root (K)

        Returns:
            dict with 'T_eq_K' (or None) and the residual at the root
        """
        f = lambda t: cooling_fn(t) - heating
        try:
            if f(T_low) * f(T_high) > 0:
                return {'T_eq_K': None,
                        'residual': float(f((T_low + T_high) / 2)),
                        'note': 'no sign change in bracket'}
            T_eq = brentq(f, T_low, T_high, xtol=1e-6)
            return {'T_eq_K': float(T_eq),
                    'residual': float(f(T_eq)), 'note': 'converged'}
        except (ValueError, RuntimeError) as exc:
            return {'T_eq_K': None, 'residual': np.nan, 'note': str(exc)}
