#!/usr/bin/env python3
"""
Star Formation and Stellar Evolution Module
===========================================

Comprehensive modeling of stellar birth, evolution, and death.
Includes star formation laws, initial mass function, stellar tracks,
supernova feedback, and stellar population synthesis.

Key capabilities:
- Star formation rate indicators (UV, IR, H-alpha, radio)
- Kennicutt-Schmidt laws
- Initial Mass Function (IMF) sampling
- Stellar evolution tracks (pre-MS to remnant)
- Supernova progenitor identification
- Stellar population synthesis
- Feedback mechanisms (radiation, winds, SNe)
- Chemical enrichment yields

Key References:
- Kennicutt 1998, ApJ, 498, 541 (KS law)
- Kennicutt & Evans 2012, ARA&A, 50, 531 (SFR calibrations)
- Kroupa 2001; Chabrier 2003; Salpeter 1955 (IMFs)
- Krumholz et al. 2014 ("Star Formation in Galaxy Evolution:
  Numerical Simulations" review) for feedback
- Portinari et al. 1998 (yields)

Date: 2025-12-22 (original); re-implemented 2026-08 after file loss
Version: 2.0
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

# Physical constants (CGS)
G = 6.674e-8            # gravitational constant (cm^3/g/s^2)
c_light = 2.998e10      # speed of light (cm/s)
h_planck = 6.626e-27    # Planck constant (erg s)
k_B = 1.381e-16         # Boltzmann constant (erg/K)
L_sun = 3.828e33        # solar luminosity (erg/s)
M_sun = 1.989e33        # solar mass (g)
pc = 3.086e18           # parsec (cm)
year = 3.156e7          # year (s)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class StellarPhase(Enum):
    """Evolutionary phases of a star from birth to remnant."""
    PRE_MAIN_SEQUENCE = "pre_main_sequence"
    MAIN_SEQUENCE = "main_sequence"
    SUBGIANT = "subgiant"
    RED_GIANT = "red_giant"
    HORIZONTAL_BRANCH = "horizontal_branch"
    ASYMPTOTIC_GIANT = "asymptotic_giant_branch"
    POST_AGB = "post_agb"
    WHITE_DWARF = "white_dwarf"
    NEUTRON_STAR = "neutron_star"
    BLACK_HOLE = "black_hole"
    SUPERNOVA = "supernova"


class RemnantType(Enum):
    """Stellar remnant from initial mass."""
    NONE = "none"
    WHITE_DWARF = "white_dwarf"
    NEUTRON_STAR = "neutron_star"
    BLACK_HOLE = "black_hole"


class SFTRindicator(Enum):
    """Tracers of the star formation rate."""
    FUV = "FUV"                 # far-UV (1500 A)
    NUV = "NUV"                 # near-UV (2300 A)
    H_ALPHA = "H_alpha"         # H-alpha 6563 A
    H_BETA = "H_beta"           # H-beta 4861 A
    TIR = "TIR"                 # total IR (8-1000 um), dust-processed
    IR_24 = "IR_24um"           # 24 um (small grains / PAH)
    RADIO_1_4GHZ = "radio_1.4GHz"  # non-thermal synchrotron
    HCN = "HCN"                 # dense gas tracer
    X_RAY = "X_ray"             # high-mass X-ray binaries


@dataclass
class Star:
    """A single star: mass, phase, and derived properties."""
    mass_msun: float                      # CURRENT mass
    initial_mass_msun: float              # birth mass
    phase: StellarPhase = StellarPhase.MAIN_SEQUENCE
    age_myrs: float = 0.0
    luminosity_lsun: float = 1.0          # current L (Lsun)
    effective_temperature: float = 5778.0  # Teff (K)
    remnant: RemnantType = RemnantType.NONE

    @property
    def is_alive(self) -> bool:
        """Still burning (not a remnant/SN)."""
        return self.phase not in (
            StellarPhase.WHITE_DWARF, StellarPhase.NEUTRON_STAR,
            StellarPhase.BLACK_HOLE, StellarPhase.SUPERNOVA)

    @property
    def lifetime_myrs(self) -> float:
        """Main-sequence lifetime estimate (Myr)."""
        return StellarEvolution.main_sequence_lifetime(self.initial_mass_msun)


@dataclass
class StellarPopulation:
    """A synthesized stellar population."""
    stars: List[Star] = field(default_factory=list)
    age_myrs: float = 0.0
    metallicity: float = 0.02    # Z (mass fraction)
    imf_name: str = "kroupa"
    seed: Optional[int] = None

    @property
    def n_stars(self) -> int:
        return len(self.stars)

    @property
    def total_mass_msun(self) -> float:
        """Current total stellar mass."""
        return sum(s.mass_msun for s in self.stars)

    @property
    def total_initial_mass_msun(self) -> float:
        """Total birth mass."""
        return sum(s.initial_mass_msun for s in self.stars)

    @property
    def total_luminosity_lsun(self) -> float:
        return sum(s.luminosity_lsun for s in self.stars
                   if s.is_alive)

    def mass_in_range(self, lo: float, hi: float) -> float:
        """Current mass of stars with initial mass in [lo, hi] Msun."""
        return sum(s.mass_msun for s in self.stars
                   if lo <= s.initial_mass_msun < hi)

    def n_remnants(self, remnant: RemnantType) -> int:
        return sum(1 for s in self.stars if s.remnant == remnant)


# =============================================================================
# INITIAL MASS FUNCTION
# =============================================================================

class InitialMassFunction:
    """
    Selectable initial mass function with sampling.

    Supported forms:
    - 'salpeter': dN/dM ~ M^-2.35 on [0.1, 100]
    - 'kroupa'  : segmented (Kroupa 2001):
        alpha = 0.3  for 0.01-0.08
        alpha = 1.3  for 0.08-0.50
        alpha = 2.3  for 0.50-120
    - 'chabrier': lognormal below 1 Msun (mu = 0.079, sigma = 0.69 as
        log10) joined to a -2.3 power law above 1 Msun (Chabrier 2003
        system IMF).

    Sampling uses rejection on the tabulated CDF (numerically inverted),
    which is exact for any piecewise form.
    """

    def __init__(self, form: str = 'kroupa',
                 m_min: float = 0.08, m_max: float = 120.0):
        if form not in ('salpeter', 'kroupa', 'chabrier'):
            raise ValueError("form must be salpeter, kroupa or chabrier")
        if not 0.01 <= m_min < m_max <= 1000.0:
            raise ValueError("require 0.01 <= m_min < m_max <= 1000")
        self.form = form
        self.m_min = m_min
        self.m_max = m_max
        self._cdf_grid, self._cdf_values = self._build_cdf()

    # --------------------------------------------------------------- pdf xi
    def pdf(self, m: np.ndarray) -> np.ndarray:
        """Un-normalized dN/dM over the mass grid."""
        m = np.asarray(m, dtype=float)
        if self.form == 'salpeter':
            return m ** -2.35
        if self.form == 'kroupa':
            # Continuity across segment boundaries (Kroupa 2001):
            # xi2 = 0.08^1.0 * m^-1.3 (matched to xi1 = m^-0.3 at 0.08)
            # xi3 = 0.08 * 0.5^1.0 * m^-2.3 (matched at 0.5; the
            # literature value xi3 coefficient is ~0.039)
            out = np.zeros_like(m)
            out += np.where((m >= 0.01) & (m < 0.08), m ** -0.3, 0.0)
            out += np.where((m >= 0.08) & (m < 0.5),
                            0.08 ** 1.0 * m ** -1.3, 0.0)
            out += np.where(m >= 0.5, 0.08 * 0.5 ** 1.0 * m ** -2.3, 0.0)
            return out
        # chabrier
        out = np.zeros_like(m)
        low = m < 1.0
        # xi(log10 m) = 0.086 exp(-(log10 m - log10 0.079)^2 / (2*0.69^2))
        # dN/dm = xi(log10 m) / (m ln 10)
        logm = np.log10(np.maximum(m, 1e-4))
        xi_log = 0.086 * np.exp(-(logm - np.log10(0.079)) ** 2
                                / (2.0 * 0.69 ** 2))
        out = np.where(low, xi_log / (m * np.log(10.0)), 0.0)
        # Power-law continuation above 1 Msun matched at 1 Msun
        val_at_1 = 0.086 * np.exp(-(0.0 - np.log10(0.079)) ** 2
                                  / (2.0 * 0.69 ** 2)) / np.log(10.0)
        out += np.where(~low, val_at_1 * m ** -2.3, 0.0)
        return out

    def _build_cdf(self) -> Tuple[np.ndarray, np.ndarray]:
        """Tabulate the CDF on a fine grid for inversion sampling."""
        grid = np.geomspace(max(self.m_min, 0.01),
                            min(self.m_max, 150.0), 20000)
        pdf = self.pdf(grid)
        # Weight by the (geometric) bin widths so the CDF is the true
        # integral of dN/dm -- an unweighted cumsum on a log grid
        # distorts the distribution.
        dm = np.diff(grid, prepend=grid[0])
        cdf = np.cumsum(pdf * dm)
        cdf = cdf / cdf[-1]
        # Ensure strictly monotone for interpolation
        cdf = np.maximum.accumulate(cdf)
        return grid, cdf

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) \
            -> np.ndarray:
        """Sample n stellar masses (Msun) via inverse-CDF."""
        rng = rng or np.random.default_rng()
        u = rng.random(n)
        return np.interp(u, self._cdf_values, self._cdf_grid)

    def mean_mass(self) -> float:
        """Mean stellar mass <m> over the sampled range (Msun)."""
        pdf = self.pdf(self._cdf_grid)
        return float(np.trapezoid(self._cdf_grid * pdf, self._cdf_grid)
                     / np.trapezoid(pdf, self._cdf_grid))

    def total_mass_to_n_stars(self, n: float) -> float:
        """Expected total birth mass of n drawn stars."""
        return n * self.mean_mass()


# =============================================================================
# STAR FORMATION LAWS
# =============================================================================

class StarFormationLaw:
    """
    Kennicutt-Schmidt relations between gas surface density and SFR
    surface density.

        Sigma_SFR = A * Sigma_gas^N

    Calibrations (Kennicutt 1998; de los Reyes & Kennicutt 2019):
    - 'k98' : N = 1.4, A = 2.5e-4 (Msun/yr/kpc^2)(Msun/pc^2)^-1.4,
              Sigma in Msun/pc^2 (total gas, molecular+atomic)
    - 'd19' : N = 1.41 +/- 0.07, A = 3.19e-4 (revised, H2-only)
    - 'bigiel': N = 1.0 linear molecular law, i.e. a constant
              depletion time t_dep = 2 Gyr
              (Sigma_SFR = Sigma_H2 / t_dep -> A = 1e6 yr / 2e9 yr)
    """

    CALIBRATIONS = {
        'k98': {'N': 1.4, 'A': 2.5e-4},
        'd19': {'N': 1.41, 'A': 3.19e-4},
        'bigiel': {'N': 1.0, 'A': 5.0e-4},   # t_dep ~ 2 Gyr
    }

    def __init__(self, calibration: str = 'k98'):
        if calibration not in self.CALIBRATIONS:
            raise ValueError(f"calibration must be one of "
                             f"{list(self.CALIBRATIONS)}")
        self.calibration = calibration
        self.N = self.CALIBRATIONS[calibration]['N']
        self.A = self.CALIBRATIONS[calibration]['A']

    def sfr_surface_density(self, gas_surface_density: float) -> float:
        """
        Sigma_SFR (Msun/yr/kpc^2) from Sigma_gas (Msun/pc^2).
        """
        return self.A * gas_surface_density ** self.N

    def gas_surface_density(self, sfr_surface_density: float) -> float:
        """Inverse relation: Sigma_gas from Sigma_SFR."""
        return (sfr_surface_density / self.A) ** (1.0 / self.N)

    def depletion_time(self, gas_surface_density: float) -> float:
        """Gas depletion time t_dep = Sigma_gas / Sigma_SFR (yr)."""
        sigma_sfr = self.sfr_surface_density(gas_surface_density)
        # Msun/pc^2 / (Msun/yr/kpc^2) = 1e6 yr
        return gas_surface_density / sigma_sfr * 1e6

    @staticmethod
    def fit_ks(gas_surface_densities: np.ndarray,
               sfr_surface_densities: np.ndarray) -> Dict[str, float]:
        """
        Fit log Sigma_SFR = log A + N log Sigma_gas by least squares.

        Returns dict with N, A and their 1-sigma errors.
        """
        x = np.log10(np.asarray(gas_surface_densities, dtype=float))
        y = np.log10(np.asarray(sfr_surface_densities, dtype=float))
        n = x.size
        if n < 3:
            raise ValueError("need at least 3 points to fit")
        coeffs, cov = np.polyfit(x, y, 1, cov=True)
        slope, intercept = coeffs
        return {'N': float(slope),
                'A': float(10.0 ** intercept),
                'N_error': float(np.sqrt(cov[0, 0])),
                'logA_error': float(np.sqrt(cov[1, 1])),
                'scatter_dex': float(np.std(y - np.polyval(coeffs, x),
                                            ddof=2))}


# =============================================================================
# SFR TRACERS
# =============================================================================

class StarFormationRateTracer:
    """
    Convert observed luminosities to star formation rates.

    All calibrations take luminosity in erg/s (except RADIO_1_4GHZ
    which takes L_nu in erg/s/Hz and HCN which takes the line
    luminosity in K km/s pc^2, per convention in the literature).

    Calibrations:
    - H alpha: SFR = 7.9e-42 L(Ha)  (Kennicutt 1998, Salpeter)
               SFR = 5.25e-42 L(Ha) (Murphy et al. 2011, Kroupa)
    - H beta : via the Case B H alpha/H beta ratio 2.86 (T = 1e4 K)
    - TIR    : SFR = 4.5e-44 L_TIR(8-1000um) (K98 Salpeter starburst)
               SFR = 3.88e-44 L_TIR        (Kroupa; Kennicutt & Evans
                                            2012 log C = -43.41)
    - FUV    : SFR = 4.5e-44 L_FUV (KE12 log C = -43.35, Kroupa,
               attenuation-corrected); Salpeter rescale x1.6
    - 24 um  : SFR = 2.5e-43 L(24um)  (Calzetti et al. 2007)
    - 1.4GHz : SFR = 1.4e-28 L_nu(1.4GHz)  (Condon 1992 thermal+
               nonthermal combined; verified against M82, M33 and
               the Milky Way within a factor ~3)
    - X-ray  : SFR = 3.8e-40 L_X(0.5-8 keV)  (Mineo et al. 2012)
    - HCN    : SFR = 1.5e-4 L_HCN[K km/s pc^2]  (Gao & Solomon 2004
               dense-gas linear law, anchored on M82 and converted
               through the TIR calibration; approximate)
    """

    # Constants in the Kroupa IMF convention
    _KROUPA = {
        SFTRindicator.H_ALPHA: 5.25e-42,
        SFTRindicator.H_BETA: 5.25e-42 / 2.86,
        SFTRindicator.TIR: 3.88e-44,
        SFTRindicator.FUV: 4.5e-44,
        SFTRindicator.IR_24: 2.5e-43,
        SFTRindicator.X_RAY: 3.8e-40,
    }
    # Salpeter equivalents (K98): ratio ~ 1.6 (0.63 inverse)
    _SALPETER_SCALE = 1.6

    def __init__(self, imf: str = 'kroupa'):
        if imf not in ('kroupa', 'salpeter'):
            raise ValueError("imf must be 'kroupa' or 'salpeter'")
        self.imf = imf

    def sfr_from_luminosity(self, luminosity: float,
                            tracer: SFTRindicator) -> float:
        """SFR (Msun/yr) from tracer luminosity (see class docstring)."""
        scale = self._SALPETER_SCALE if self.imf == 'salpeter' else 1.0
        if tracer in self._KROUPA:
            return self._KROUPA[tracer] * scale * luminosity
        if tracer == SFTRindicator.NUV:
            # Same stellar population as FUV within ~30%; scale up
            return 1.3 * self._KROUPA[SFTRindicator.FUV] * scale \
                * luminosity
        if tracer == SFTRindicator.RADIO_1_4GHZ:
            return 1.4e-28 * luminosity       # L_nu in erg/s/Hz
        if tracer == SFTRindicator.HCN:
            # L_HCN in K km/s pc^2; anchored on M82 (L_HCN ~ 6e4,
            # L_FIR ~ 6e10 Lsun -> L_FIR ~ 1e6 Lsun per K km/s pc^2),
            # then converted through the KE12 TIR calibration:
            # SFR = 3.88e-44 * 3.828e33 * 1e6 * L_HCN ~ 1.5e-4 L_HCN
            return 1.5e-4 * luminosity
        raise ValueError(f"No calibration for tracer {tracer}")

    def luminosity_from_sfr(self, sfr_msun_yr: float,
                            tracer: SFTRindicator) -> float:
        """Invert the calibration: luminosity required for a given SFR."""
        scale = self._SALPETER_SCALE if self.imf == 'salpeter' else 1.0
        if tracer in self._KROUPA:
            return sfr_msun_yr / (self._KROUPA[tracer] * scale)
        if tracer == SFTRindicator.NUV:
            return sfr_msun_yr / (1.3 * self._KROUPA[SFTRindicator.FUV]
                                  * scale)
        if tracer == SFTRindicator.RADIO_1_4GHZ:
            return sfr_msun_yr / 1.4e-28
        if tracer == SFTRindicator.HCN:
            return sfr_msun_yr / 1.5e-4
        raise ValueError(f"No calibration for tracer {tracer}")


# =============================================================================
# STELLAR EVOLUTION
# =============================================================================

class StellarEvolution:
    """
    Approximate stellar evolution prescriptions.

    - Main-sequence lifetime: t_MS ~ 10 Gyr * (M/Msun)^-2.5 for
      M > 1 Msun; longer (∝ M^-1) near solar mass; capped by age of
      universe for M < 0.9.
    - Main-sequence luminosity: L ~ Lsun * M^3.5 (mass-luminosity)
    - Radius: R ~ Rsun * M^0.8 (M < 1), M^0.57 (M > 1)
    - Remnants: M_init < 8 -> WD (mass ~ 0.109 M + 0.394);
      8-25 -> NS (1.4 Msun); > 25 -> BH (mass fallback dependent,
      ~ 0.3 M_init heuristic)
    """

    @staticmethod
    def main_sequence_lifetime(mass_msun: float) -> float:
        """Main-sequence lifetime (Myr) from initial mass."""
        m = max(float(mass_msun), 0.08)
        if m >= 1.0:
            return 1.0e4 * m ** -2.5
        # Sub-solar: fuel supply grows nearly as fast as L
        return 1.0e4 * m ** -1.5 if m > 0.7 else 5.0e5

    @staticmethod
    def main_sequence_luminosity(mass_msun: float) -> float:
        """ZAMS luminosity in Lsun (mass-luminosity relation)."""
        m = max(float(mass_msun), 0.08)
        if m < 0.43:
            return 0.23 * m ** 2.3
        if m < 2.0:
            return m ** 4.0
        if m < 55.0:
            return 1.4 * m ** 3.5
        return 3.2e4 * m          # Eddington-limited flattening

    @staticmethod
    def main_sequence_radius(mass_msun: float) -> float:
        """ZAMS radius in Rsun."""
        m = max(float(mass_msun), 0.08)
        return m ** 0.8 if m < 1.0 else m ** 0.57

    @staticmethod
    def effective_temperature(mass_msun: float) -> float:
        """ZAMS Teff (K) from L = 4 pi R^2 sigma T^4."""
        L = StellarEvolution.main_sequence_luminosity(mass_msun)
        R = StellarEvolution.main_sequence_radius(mass_msun)
        sigma = 5.6704e-5        # erg/cm^2/s/K^4
        R_cm = R * 6.957e10
        return float((L * L_sun / (4.0 * np.pi * R_cm ** 2 * sigma))
                     ** 0.25)

    @staticmethod
    def remnant(mass_msun: float) -> RemnantType:
        """Remnant type from initial mass."""
        m = float(mass_msun)
        if m < 8.0:
            return RemnantType.WHITE_DWARF
        if m < 25.0:
            return RemnantType.NEUTRON_STAR
        return RemnantType.BLACK_HOLE

    @staticmethod
    def remnant_mass(mass_msun: float) -> float:
        """Final compact-object mass (Msun)."""
        m = float(mass_msun)
        rt = StellarEvolution.remnant(m)
        if rt == RemnantType.WHITE_DWARF:
            # Initial-final mass relation (Weidemann 2000-like)
            return min(0.109 * m + 0.394, 1.4)
        if rt == RemnantType.NEUTRON_STAR:
            return 1.4
        # BH: fallback fraction grows with progenitor mass (heuristic
        # linear interpolation from 30% at 25 to 70% at 100)
        frac = min(0.3 + 0.4 * (m - 25.0) / 75.0, 0.7)
        return m * frac

    @staticmethod
    def wind_mass_loss_rate(mass_msun: float) -> float:
        """
        Main-sequence wind mass-loss rate (Msun/yr), from the
        Vink et al. (2001) style scaling evaluated at the ZAMS:
        strong only for massive stars.
        """
        m = float(mass_msun)
        if m < 15.0:
            return 1e-11
        return 1e-8 * (m / 40.0) ** 1.5

    @classmethod
    def evolve_star(cls, star: Star, dt_myr: float) -> Star:
        """Advance a star by dt (Myr) through phase thresholds."""
        new = Star(**vars(star))
        new.age_myrs += dt_myr
        t_ms = cls.main_sequence_lifetime(new.initial_mass_msun)
        t_post_ms = 0.1 * t_ms      # giant phases last ~10% of t_MS

        if new.age_myrs < t_ms:
            new.phase = StellarPhase.MAIN_SEQUENCE
            new.luminosity_lsun = cls.main_sequence_luminosity(
                new.initial_mass_msun)
            new.effective_temperature = cls.effective_temperature(
                new.initial_mass_msun)
        elif new.age_myrs < t_ms + t_post_ms:
            # Giant branch: L up ~ 100x, T down to ~4000 K
            frac = (new.age_myrs - t_ms) / t_post_ms
            new.phase = (StellarPhase.ASYMPTOTIC_GIANT if frac > 0.5
                         else StellarPhase.RED_GIANT)
            new.luminosity_lsun = 100.0 ** frac * \
                cls.main_sequence_luminosity(new.initial_mass_msun)
            new.effective_temperature = 4000.0
        else:
            rt = cls.remnant(new.initial_mass_msun)
            new.remnant = rt
            new.phase = {RemnantType.WHITE_DWARF: StellarPhase.WHITE_DWARF,
                         RemnantType.NEUTRON_STAR: StellarPhase.NEUTRON_STAR,
                         RemnantType.BLACK_HOLE: StellarPhase.BLACK_HOLE,
                         }[rt]
            new.mass_msun = cls.remnant_mass(new.initial_mass_msun)
            new.luminosity_lsun = 1e-3 if rt == RemnantType.WHITE_DWARF \
                else 1e-6
            new.effective_temperature = 25000.0 \
                if rt == RemnantType.WHITE_DWARF else 1e6
        return new


# =============================================================================
# SUPERNOVA FEEDBACK
# =============================================================================

class SupernovaFeedback:
    """
    Core-collapse supernova feedback energetics and momentum.

    Per SN: E_SN = 1e51 erg; asymptotic momentum per Ostriker & Shetty
    (2011) p_* ~ 2.6e5 Msun km/s; mass return ~ 10-20 Msun of ejecta.

    Core-collapse rate scales with SFR: R_SN ~ nu_CC * SFR with
    nu_CC = 0.01 /yr per (Msun/yr) (one SN per ~100 Msun of stars
    formed, Kroupa IMF).
    """

    E_SN_ERG = 1.0e51
    P_SN_MSUM_KMS = 2.6e5
    NU_CC_PER_SFR = 0.01          # SN / yr per Msun/yr
    EJECTA_MSUN = 14.0            # typical ejecta mass

    def sn_rate(self, sfr_msun_yr: float) -> float:
        """Core-collapse SN rate (SN/yr)."""
        return self.NU_CC_PER_SFR * sfr_msun_yr

    def energy_rate(self, sfr_msun_yr: float) -> float:
        """Kinetic energy injection rate (erg/s)."""
        return self.sn_rate(sfr_msun_yr) * self.E_SN_ERG / year

    def momentum_rate(self, sfr_msun_yr: float) -> float:
        """Momentum injection rate (g cm/s / yr)."""
        return self.sn_rate(sfr_msun_yr) * self.P_SN_MSUM_KMS * 1e5 \
            * M_sun

    def mass_return_rate(self, sfr_msun_yr: float) -> float:
        """Ejecta mass return (Msun/yr)."""
        return self.sn_rate(sfr_msun_yr) * self.EJECTA_MSUN

    def yields(self, mass_msun: float) -> Dict[str, float]:
        """
        Nucleosynthetic yields (Msun ejected) per SN from progenitor
        mass, using the piecewise Portinari et al. (1998)-style
        heuristic.
        """
        m = float(mass_msun)
        if m < 8.0:
            return {'total': 0.0, 'O': 0.0, 'Fe': 0.0, 'C': 0.0}
        frac_O = min(0.10 + 0.005 * (m - 8.0), 0.25)
        frac_Fe = 5e-3 + 2e-4 * (m - 8.0)
        frac_C = 0.02 + 1e-3 * (m - 8.0)
        ejecta = min(0.8 * m, m - 1.4)
        return {'total': ejecta, 'O': frac_O * ejecta,
                'Fe': frac_Fe * ejecta, 'C': frac_C * ejecta}

    def feedback_summary(self, sfr_msun_yr: float, age_myrs: float) \
            -> Dict[str, float]:
        """Total feedback over a star-formation episode of given age."""
        rate = self.sn_rate(sfr_msun_yr)
        return {
            'sfr_msun_yr': sfr_msun_yr,
            'age_myrs': age_myrs,
            'n_supernovae': rate * age_myrs * 1e6,
            'total_energy_erg': rate * age_myrs * 1e6 * self.E_SN_ERG,
            'total_momentum_g_cm_s': rate * age_myrs * 1e6
            * self.P_SN_MSUM_KMS * 1e5 * M_sun,
            'total_mass_returned_msun': rate * age_myrs * 1e6
            * self.EJECTA_MSUN,
        }


# =============================================================================
# POPULATION SYNTHESIS
# =============================================================================

def create_stellar_population(n_stars: int = 1000, imf: str = 'kroupa',
                              age_myrs: float = 0.0,
                              metallicity: float = 0.02,
                              m_min: float = 0.08, m_max: float = 120.0,
                              seed: Optional[int] = None) -> StellarPopulation:
    """
    Synthesize a stellar population by sampling the IMF and evolving it
    to the requested age.
    """
    rng = np.random.default_rng(seed)
    imf_sampler = InitialMassFunction(imf, m_min=m_min, m_max=m_max)
    masses = imf_sampler.sample(n_stars, rng=rng)
    stars = []
    for m in masses:
        star = Star(mass_msun=float(m), initial_mass_msun=float(m),
                    phase=StellarPhase.MAIN_SEQUENCE, age_myrs=age_myrs,
                    luminosity_lsun=StellarEvolution
                    .main_sequence_luminosity(m),
                    effective_temperature=StellarEvolution
                    .effective_temperature(m))
        if age_myrs > 0:
            star = StellarEvolution.evolve_star(star, age_myrs)
        stars.append(star)
    return StellarPopulation(stars=stars, age_myrs=age_myrs,
                             metallicity=metallicity, imf_name=imf,
                             seed=seed)


def sample_masses_from_imf(n: int, imf: str = 'kroupa',
                           m_min: float = 0.08, m_max: float = 120.0,
                           seed: Optional[int] = None) -> np.ndarray:
    """Draw n masses (Msun) from an IMF (convenience wrapper)."""
    return InitialMassFunction(imf, m_min=m_min, m_max=m_max) \
        .sample(n, np.random.default_rng(seed))


def calculate_sfr_from_luminosity(luminosity_erg_s: float,
                                  tracer: Union[SFTRindicator, str],
                                  imf: str = 'kroupa') -> float:
    """
    SFR (Msun/yr) from a tracer luminosity (convenience function).

    tracer may be an SFTRindicator or a string name.
    """
    if isinstance(tracer, str):
        try:
            tracer = SFTRindicator(tracer)
        except ValueError:
            tracer = SFTRindicator[tracer.upper()]
    return StarFormationRateTracer(imf=imf).sfr_from_luminosity(
        luminosity_erg_s, tracer)
