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
HII Region Physics

This module provides comprehensive physics for ionized hydrogen regions
(HII regions) including:
- Strömgren sphere calculations
- Ionization equilibrium and structure
- Nebular emission line diagnostics (T_e, n_e from line ratios)
- Recombination cascades and line emissivities
- Free-free (Bremsstrahlung) continuum emission
- Photoionization modeling interface

Physical constants in CGS units throughout.

Date: 2025-12-11
Version: 43.0
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum, auto

# Physical constants (CGS)
C_LIGHT = 2.998e10       # Speed of light (cm/s)
H_PLANCK = 6.626e-27     # Planck constant (erg s)
K_BOLTZMANN = 1.381e-16  # Boltzmann constant (erg/K)
M_ELECTRON = 9.109e-28   # Electron mass (g)
M_PROTON = 1.673e-24     # Proton mass (g)
E_CHARGE = 4.803e-10     # Electron charge (esu)
M_SUN = 1.989e33         # Solar mass (g)
L_SUN = 3.828e33         # Solar luminosity (erg/s)
PC = 3.086e18            # Parsec (cm)
YEAR = 3.156e7           # Year (seconds)

# Ionization energies (eV)
CHI_H = 13.6             # Hydrogen ionization potential
CHI_HE = 24.6            # Helium first ionization
CHI_HE2 = 54.4           # Helium second ionization

# Conversion factors
EV_TO_ERG = 1.602e-12    # eV to erg
EV_TO_K = 11604.5        # eV to Kelvin


class HIIRegionType(Enum):
    """Types of HII regions."""
    ULTRACOMPACT = auto()     # UC HII, < 0.1 pc, n_e > 10^4
    COMPACT = auto()          # Compact, 0.1-1 pc
    CLASSICAL = auto()        # Classical, 1-10 pc
    GIANT = auto()            # Giant HII, 10-100 pc
    SUPERGIANT = auto()       # Supergiant, > 100 pc


class IonizationState(Enum):
    """Ionization-bounded vs density-bounded."""
    IONIZATION_BOUNDED = auto()  # All ionizing photons absorbed
    DENSITY_BOUNDED = auto()     # Ionizing photons escape


@dataclass
class StromgrenParameters:
    """Strömgren sphere parameters."""
    radius: float               # Strömgren radius (cm)
    volume: float               # Ionized volume (cm³)
    ionizing_photon_rate: float  # Q_H (photons/s)
    electron_density: float     # n_e (cm⁻³)
    recombination_rate: float   # α_B (cm³/s)
    recombination_time: float   # t_rec (s)
    ionization_state: IonizationState
    filling_factor: float       # Volume filling factor

    @property
    def radius_pc(self) -> float:
        """Radius in parsecs."""
        return self.radius / PC

    @property
    def recombination_time_yr(self) -> float:
        """Recombination time in years."""
        return self.recombination_time / YEAR


@dataclass
class NebularDiagnostics:
    """Nebular diagnostic results."""
    electron_temperature: float  # T_e (K)
    electron_density: float     # n_e (cm⁻³)
    ionic_abundances: Dict[str, float]  # Ion abundances relative to H
    ionization_correction: Dict[str, float]  # ICFs for total abundances
    total_abundances: Dict[str, float]  # Total element abundances

    # Diagnostic line ratios used
    oiii_ratio: Optional[float]  # [OIII] 4363/5007 (T_e diagnostic)
    nii_ratio: Optional[float]   # [NII] 5755/6583 (T_e diagnostic)
    sii_ratio: Optional[float]   # [SII] 6717/6731 (n_e diagnostic)
    oii_ratio: Optional[float]   # [OII] 3726/3729 (n_e diagnostic)


@dataclass
class RecombinationSpectrum:
    """Hydrogen recombination line spectrum."""
    line_name: str
    wavelength: float           # Wavelength (Angstrom)
    upper_level: int            # Upper principal quantum number
    lower_level: int            # Lower principal quantum number
    emissivity: float           # j_line (erg cm³/s)
    intensity_ratio: float      # Relative to Hβ

    @property
    def wavelength_micron(self) -> float:
        """Wavelength in microns."""
        return self.wavelength / 1e4


@dataclass
class FreeFreeEmission:
    """Free-free (Bremsstrahlung) emission."""
    emission_measure: float     # EM = ∫n_e n_i dl (cm⁻⁶ pc)
    brightness_temperature: float  # T_b at reference frequency (K)
    flux_density: float         # S_ν at reference frequency (Jy)
    spectral_index: float       # α where S_ν ∝ ν^α
    optical_depth: float        # τ_ff at reference frequency
    turnover_frequency: float   # ν where τ = 1 (Hz)


class RecombinationCoefficients:
    """
    Hydrogen recombination coefficients.

    Case A: All Lyman photons escape
    Case B: Lyman photons trapped (optically thick to Lyman series)
    """

    def __init__(self):
        """Initialize recombination coefficient calculator."""
        pass

    def alpha_A(self, temperature: float) -> float:
        """
        Case A total recombination coefficient.

        Includes recombinations to all levels including ground state.

        Args:
            temperature: Electron temperature (K)

        Returns:
            α_A (cm³/s)
        """
        # Fit from Osterbrock & Ferland (2006)
        t4 = temperature / 1e4
        return 4.18e-13 * t4**(-0.72)

    def alpha_B(self, temperature: float) -> float:
        """
        Case B total recombination coefficient.

        Excludes recombinations to ground state (Lyman photons trapped).

        Args:
            temperature: Electron temperature (K)

        Returns:
            α_B (cm³/s)
        """
        # Fit from Osterbrock & Ferland (2006)
        t4 = temperature / 1e4
        return 2.59e-13 * t4**(-0.833 - 0.034 * math.log(t4))

    def alpha_eff_hbeta(self, temperature: float) -> float:
        """
        Effective recombination coefficient for Hβ.

        Args:
            temperature: Electron temperature (K)

        Returns:
            α_eff(Hβ) (cm³/s)
        """
        t4 = temperature / 1e4
        return 3.03e-14 * t4**(-0.874)

    def alpha_eff_halpha(self, temperature: float) -> float:
        """
        Effective recombination coefficient for Hα.

        Args:
            temperature: Electron temperature (K)

        Returns:
            α_eff(Hα) (cm³/s)
        """
        t4 = temperature / 1e4
        return 1.17e-13 * t4**(-0.942)


class StromgrenSphere:
    """
    Strömgren sphere calculations for HII regions.

    The Strömgren sphere is the ionized region around a hot star
    where ionizations balance recombinations.

    R_s = (3 Q_H / 4π α_B n²)^(1/3)
    """

    def __init__(self):
        """Initialize Strömgren sphere calculator."""
        self.recomb = RecombinationCoefficients()

    def ionizing_photon_rate(self, spectral_type: str = None,
                             luminosity: float = None,
                             temperature: float = None) -> float:
        """
        Ionizing photon rate Q_H for given stellar parameters.

        Args:
            spectral_type: Spectral type (O3-B2)
            luminosity: Stellar luminosity (erg/s)
            temperature: Stellar effective temperature (K)

        Returns:
            Q_H (photons/s)
        """
        # Ionizing photon rates for O/B stars (Martins+ 2005, Sternberg+ 2003)
        q_h_table = {
            'O3V': 1e50, 'O4V': 6e49, 'O5V': 4e49, 'O6V': 2e49,
            'O7V': 1e49, 'O8V': 5e48, 'O9V': 2e48, 'O9.5V': 1e48,
            'B0V': 3e47, 'B0.5V': 1e47, 'B1V': 3e46, 'B2V': 5e45
        }

        if spectral_type and spectral_type in q_h_table:
            return q_h_table[spectral_type]

        if temperature:
            # FIX(audit B-HII-1): the old fit, 10**(3e-4*T_eff + 39) with a hard
            # 0 below 30 000 K, was unbounded and unphysical -- Q_H(50 kK) = 1e54
            # photons/s (~5e9 Lsun of Lyman continuum from one star), a factor
            # 713 too high at 44 850 K, and discontinuous 0 -> 1e48 at 30 000 K.
            # Replaced by log-linear interpolation of the Martins, Schaerer &
            # Hillier (2005, A&A 436, 1049) Table 1 luminosity-class-V grid
            # (theoretical Teff scale), which is the calibration this module
            # already cites.
            return self._q_h_from_teff(float(temperature))

        if luminosity:
            # Rough estimate assuming O star
            return luminosity / (H_PLANCK * C_LIGHT / (912e-8))  # 912 Å photon

        return 1e49  # Default O6V-like star

    # Martins, Schaerer & Hillier (2005), A&A 436, 1049, Table 1:
    # Galactic O dwarfs (luminosity class V), theoretical Teff scale.
    # (T_eff [K], log10 Q0 [s^-1]) for spectral types O3V ... O9.5V.
    _MSH05_DWARF_TEFF_LOGQ0 = (
        (30488, 47.56), (31524, 47.90), (32522, 48.10), (33383, 48.29),
        (34419, 48.44), (35531, 48.63), (36826, 48.80), (38151, 48.96),
        (40062, 49.11), (41540, 49.26), (43419, 49.47), (44616, 49.63),
    )

    @classmethod
    def _q_h_from_teff(cls, temperature: float) -> float:
        """
        Q_H [photons/s] for a main-sequence (class V) star of given T_eff.

        Linear interpolation of log10 Q0 in T_eff over the Martins et al. (2005)
        O-dwarf grid (30 488 - 44 616 K). Outside that range the value is a
        log-linear extrapolation using the slope of the two nearest grid points
        and a warning is issued: Q_H falls extremely steeply for B stars and
        rises into a poorly calibrated regime above O3V, so extrapolated values
        should be treated as order-of-magnitude only.
        """
        grid = cls._MSH05_DWARF_TEFF_LOGQ0
        t_lo, q_lo = grid[0]
        t_hi, q_hi = grid[-1]

        if temperature < t_lo:
            slope = (grid[1][1] - grid[0][1]) / (grid[1][0] - grid[0][0])
            warnings.warn(
                f"T_eff = {temperature:.0f} K is below the Martins et al. (2005) "
                f"O-star grid ({t_lo} K); Q_H is extrapolated and is an "
                f"order-of-magnitude estimate only.", RuntimeWarning)
            return 10.0 ** (q_lo + slope * (temperature - t_lo))
        if temperature > t_hi:
            slope = (grid[-1][1] - grid[-2][1]) / (grid[-1][0] - grid[-2][0])
            warnings.warn(
                f"T_eff = {temperature:.0f} K is above the Martins et al. (2005) "
                f"O-star grid ({t_hi} K); Q_H is extrapolated and is an "
                f"order-of-magnitude estimate only.", RuntimeWarning)
            return 10.0 ** (q_hi + slope * (temperature - t_hi))

        for (t0, q0), (t1, q1) in zip(grid[:-1], grid[1:]):
            if t0 <= temperature <= t1:
                frac = (temperature - t0) / (t1 - t0)
                return 10.0 ** (q0 + frac * (q1 - q0))
        raise RuntimeError("unreachable: T_eff bracketing failed")

    def stromgren_radius(self, q_h: float, n_e: float,
                         temperature: float = 1e4,
                         filling_factor: float = 1.0) -> float:
        """
        Calculate Strömgren radius.

        R_s = (3 Q_H / 4π α_B n² f)^(1/3)

        Args:
            q_h: Ionizing photon rate (photons/s)
            n_e: Electron density (cm⁻³)
            temperature: Electron temperature (K)
            filling_factor: Volume filling factor

        Returns:
            Strömgren radius (cm)
        """
        alpha_b = self.recomb.alpha_B(temperature)
        n_eff = n_e * math.sqrt(filling_factor)
        return (3 * q_h / (4 * math.pi * alpha_b * n_eff**2))**(1/3)

    def recombination_time(self, n_e: float, temperature: float = 1e4) -> float:
        """
        Recombination timescale.

        t_rec = 1 / (α_B n_e)

        Args:
            n_e: Electron density (cm⁻³)
            temperature: Electron temperature (K)

        Returns:
            Recombination time (s)
        """
        alpha_b = self.recomb.alpha_B(temperature)
        return 1 / (alpha_b * n_e)

    def expansion_time(self, r_s: float, n_e: float,
                       temperature: float = 1e4) -> float:
        """
        Expansion timescale for HII region.

        t_exp ~ R_s / c_i where c_i is ionized gas sound speed

        Args:
            r_s: Strömgren radius (cm)
            n_e: Electron density (cm⁻³)
            temperature: Electron temperature (K)

        Returns:
            Expansion time (s)
        """
        # Ionized gas sound speed (including He contribution)
        mu_ion = 0.62
        c_i = math.sqrt(K_BOLTZMANN * temperature / (mu_ion * M_PROTON))
        return r_s / c_i

    def compute(self, q_h: float, n_0: float, temperature: float = 1e4,
                filling_factor: float = 1.0,
                ambient_radius: float = None) -> StromgrenParameters:
        """
        Compute complete Strömgren sphere parameters.

        Args:
            q_h: Ionizing photon rate (photons/s)
            n_0: Ambient density (cm⁻³)
            temperature: Electron temperature (K)
            filling_factor: Volume filling factor
            ambient_radius: Available radius (cm), if limited

        Returns:
            StromgrenParameters
        """
        # Electron density (assuming fully ionized H + 10% He)
        n_e = 1.1 * n_0

        # Strömgren radius
        r_s = self.stromgren_radius(q_h, n_e, temperature, filling_factor)

        # Check if ionization-bounded or density-bounded
        if ambient_radius and r_s > ambient_radius:
            ionization_state = IonizationState.DENSITY_BOUNDED
            r_s = ambient_radius
        else:
            ionization_state = IonizationState.IONIZATION_BOUNDED

        # Volume
        volume = (4/3) * math.pi * r_s**3 * filling_factor

        # Timescales
        alpha_b = self.recomb.alpha_B(temperature)
        t_rec = self.recombination_time(n_e, temperature)

        return StromgrenParameters(
            radius=r_s,
            volume=volume,
            ionizing_photon_rate=q_h,
            electron_density=n_e,
            recombination_rate=alpha_b,
            recombination_time=t_rec,
            ionization_state=ionization_state,
            filling_factor=filling_factor
        )


class IonizationEquilibrium:
    """
    Ionization equilibrium calculations.

    Solves photoionization-recombination balance for H and He.
    """

    def __init__(self):
        """Initialize ionization equilibrium solver."""
        self.recomb = RecombinationCoefficients()

    def hydrogen_ionization_fraction(self, ionization_parameter: float,
                                     temperature: float = 1e4) -> float:
        """
        Hydrogen ionization fraction x = n(H+)/n(H).

        Args:
            ionization_parameter: U = Q_H / (4π r² n c)
            temperature: Electron temperature (K)

        Returns:
            Ionization fraction (0-1)
        """
        # For typical HII region conditions, H is nearly fully ionized
        if ionization_parameter > 1e-4:
            return 0.9999
        elif ionization_parameter > 1e-6:
            return 0.99
        else:
            return ionization_parameter * 1e4

    def ionization_parameter(self, q_h: float, n_h: float,
                             radius: float) -> float:
        """
        Dimensionless ionization parameter.

        U = Q_H / (4π r² n_H c)

        Args:
            q_h: Ionizing photon rate (photons/s)
            n_h: Hydrogen density (cm⁻³)
            radius: Distance from source (cm)

        Returns:
            Ionization parameter U
        """
        return q_h / (4 * math.pi * radius**2 * n_h * C_LIGHT)

    def ionization_front_thickness(self, n_h: float,
                                   temperature: float = 1e4) -> float:
        """
        Thickness of ionization front.

        Δr ~ 1 / (n_H σ_H) where σ_H is H photoionization cross-section

        Args:
            n_h: Hydrogen density (cm⁻³)
            temperature: Temperature (K)

        Returns:
            Front thickness (cm)
        """
        # H photoionization cross-section at threshold
        sigma_h = 6.3e-18  # cm²
        return 1 / (n_h * sigma_h)


class NebularDiagnosticsCalculator:
    """
    Nebular emission line diagnostics.

    Derives T_e and n_e from collisionally excited line ratios.
    """

    def __init__(self):
        """Initialize nebular diagnostics calculator."""
        pass

    def oiii_temperature(self, ratio_4363_5007: float) -> float:
        """
        Electron temperature from [OIII] 4363/5007 ratio.

        This ratio is sensitive to T_e because 4363 comes from
        a higher energy level than 4959+5007.

        Args:
            ratio_4363_5007: [OIII] 4363 / ([OIII] 4959 + 5007)

        Returns:
            Electron temperature (K)
        """
        # FIX(audit B-HII-3): the previous anchor "R = 0.0100 at T_e = 1e4 K
        # => C = 0.275" was wrong by 2.17x and biased T_e low by 18-45%.
        #
        # The auroral line 4363 A (upper level 1S0) and the nebular lines
        # 4959+5007 A (upper 1D2) are both collisionally excited from the
        # ground term, so R = I(4363)/I(4959+5007) = C exp(-Delta E / k T_e).
        # Osterbrock & Ferland (2006), eq. 5.4, give the low-density limit
        #
        #     j(4959)+j(5007)         exp(3.29e4 / T_e)
        #     ---------------  = 7.90 -----------------
        #        j(4363)              1 + 4.5e-4 n_e/sqrt(T_e)
        #
        # i.e. C = 1/7.90 = 0.12658 and Delta E / k = 3.29e4 K, so that
        # R(1e4 K) = 4.72e-3 (NOT 1.00e-2). Inverting the low-density limit,
        #
        #     T_e = 32900 / ln(0.12658 / R)   [K]
        #
        # Valid for n_e << 10^4 cm^-3 (the density term above is neglected);
        # at higher densities collisional de-excitation raises R and this
        # formula overestimates T_e.
        # Before/after at R = 0.0100: 9 998 K -> 12 961 K.
        if ratio_4363_5007 <= 0.0:
            raise ValueError("ratio must be positive")
        t_e = 32900.0 / math.log((1.0 / 7.90) / ratio_4363_5007)
        if not (4000.0 < t_e < 50000.0):
            raise ValueError(
                f"T_e = {t_e:.0f} K outside the valid range of the "
                f"[O III] diagnostic (4000-50000 K); check the ratio")
        return t_e

    def oxygen_abundance(self, t3: float, t2: float,
                         o2plus_over_hbeta: float,
                         oplus_over_hbeta: float,
                         n_e: float = 100.0) -> float:
        """
        Total oxygen abundance O/H = (O+ + O++)/H+ from ionic line ratios.

        Args:
            t3: Temperature (K) in the O++ zone (from [O III])
            t2: Temperature (K) in the O+ zone (from [O II])
            o2plus_over_hbeta: I(4959+5007)/I(H beta)   (ratio, H beta = 1)
            oplus_over_hbeta: I(3727+3729)/I(H beta)    (ratio, H beta = 1)
            n_e: electron density (cm^-3), used only in the [O II] density
                 correction term; negligible for n_e << 1e3 cm^-3

        Returns:
            O/H by number (linear, NOT 12 + log10)
        """
        # FIX(audit B-HII-2): the previous expression
        #   O++/H+ = R3 * 4.5e-3 * t3^0.5 / (1 + 2.2e-2 t3)
        # had NO exp(Delta E / k T_e) Boltzmann factor at all, so the derived
        # abundance *rose* with T_e instead of falling steeply, and was ~231x
        # too high (O/H = 0.048 for a normal H II region, i.e. 12+log O/H = 10.7).
        #
        # Replaced with the standard direct-method ionic abundance relations,
        # Izotov et al. (2006, A&A 448, 955) eqs. (3) and (4):
        #
        #  12 + log(O+ /H+) = log(I3727/IHb) + 5.961 + 1.676/t2 - 0.40 log t2
        #                     - 0.034 t2 + log(1 + 1.35 x)
        #  12 + log(O++/H+) = log(I4959+5007/IHb) + 6.200 + 1.251/t3
        #                     - 0.55 log t3 - 0.014 t3
        #  with t = T_e/1e4 and x = 1e-4 n_e t2^-0.5.
        #
        # The 1.676/t2 and 1.251/t3 terms are the exp(Delta E/kT) dependence in
        # log form; they make O/H fall by ~5x from t = 0.7 to t = 2.0, as it must.
        # Before/after at t3 = t2 = 1, R3 = 5, R2 = 2: 0.04804 -> 2.18e-4.
        if t3 <= 0 or t2 <= 0:
            raise ValueError("temperatures must be positive")
        if o2plus_over_hbeta <= 0 or oplus_over_hbeta <= 0:
            raise ValueError("line ratios must be positive")

        t3_4 = t3 / 1e4
        t2_4 = t2 / 1e4
        x = 1e-4 * n_e * t2_4 ** -0.5

        log_o_plus = (math.log10(oplus_over_hbeta) + 5.961 + 1.676 / t2_4
                      - 0.40 * math.log10(t2_4) - 0.034 * t2_4
                      + math.log10(1.0 + 1.35 * x) - 12.0)
        log_o_2plus = (math.log10(o2plus_over_hbeta) + 6.200 + 1.251 / t3_4
                       - 0.55 * math.log10(t3_4) - 0.014 * t3_4 - 12.0)

        return 10.0 ** log_o_plus + 10.0 ** log_o_2plus


# =============================================================================
# RECOMBINATION LINES
# =============================================================================

class RecombinationLines:
    """
    Hydrogen recombination line emission (Case B).

    Case B assumes the nebula is optically thick to Lyman-series
    photons (they are reabsorbed on the spot), so recombinations
    directly to the ground state do not produce observable photons.

    Reference emissivities: Hummer & Storey (1987); fits from
    Osterbrock & Ferland (2006), table 4.2.
    """

    # Case B effective recombination coefficients (cm^3/s) at Te = 1e4 K
    # and their temperature exponents: alpha_eff(T) = a0 * (T/1e4)^beta
    # alpha_eff(line, T) = a0 * (T/1e4)^beta, in PHOTONS cm^3/s:
    # j(line) = n_e n_p alpha_eff h nu. The coefficients reproduce the
    # canonical Case B energy decrement at Te = 1e4 K:
    # I(Ha)/I(Hb) = 2.86, I(Hg)/I(Hb) = 0.469, I(Hd)/I(Hb) = 0.259.
    _LINE_COEFFICIENTS = {
        'Halpha': (1.17e-13, -0.942),   # 6563 A, alpha_eff(Ha)/alpha_B ~ 0.45
        'Hbeta':  (3.03e-14, -0.874),   # 4861 A
        'Hgamma': (1.27e-14, -0.885),   # 4340 A
        'Hdelta': (6.62e-15, -0.900),   # 4102 A
    }

    # Line wavelengths (cm)
    _WAVELENGTHS_CM = {
        'Halpha': 6562.8e-8,
        'Hbeta': 4861.3e-8,
        'Hgamma': 4340.5e-8,
        'Hdelta': 4101.7e-8,
    }

    # Photon energies (erg)
    _PHOTON_ENERGY = {
        name: H_PLANCK * C_LIGHT / wl
        for name, wl in _WAVELENGTHS_CM.items()
    }

    def __init__(self, temperature: float = 1e4):
        """
        Args:
            temperature: Electron temperature (K)
        """
        self.temperature = temperature
        self.recomb = RecombinationCoefficients()

    def alpha_eff(self, line: str) -> float:
        """Effective recombination coefficient (cm^3/s) for a line."""
        if line not in self._LINE_COEFFICIENTS:
            raise ValueError(f"unknown line: {line}")
        a0, beta = self._LINE_COEFFICIENTS[line]
        t4 = self.temperature / 1e4
        return a0 * t4 ** beta

    def emissivity(self, line: str, n_e: float, n_p: float) -> float:
        """
        Volume emissivity j (erg cm^-3 s^-1).

        j = n_e n_p alpha_eff(T) h nu
        """
        return (n_e * n_p * self.alpha_eff(line)
                * self._PHOTON_ENERGY[line])

    def emissivity_ratio(self, line1: str, line2: str) -> float:
        """
        Case B energy emissivity ratio j(line1)/j(line2), i.e. the
        observed line-intensity ratio (Halpha/Hbeta = 2.86 at 1e4 K).
        """
        return (self.alpha_eff(line1) * self._PHOTON_ENERGY[line1]
                / (self.alpha_eff(line2) * self._PHOTON_ENERGY[line2]))

    def balmer_decrement(self) -> Dict[str, float]:
        """
        Case B Balmer decrement relative to H beta.

        At Te = 1e4 K: Halpha/Hbeta = 2.86, Hgamma/Hbeta = 0.469,
        Hdelta/Hbeta = 0.259 (Osterbrock & Ferland 2006, table 4.2).
        """
        out = {}
        for name in ('Halpha', 'Hgamma', 'Hdelta'):
            out[name] = self.emissivity_ratio(name, 'Hbeta')
        return out

    def luminosity(self, line: str, q_h: float,
                   filling_factor: float = 1.0) -> float:
        """
        Total line luminosity (erg/s) of an ionization-bounded HII
        region with ionizing photon rate q_h.

        In ionization equilibrium every Case B recombination cascades
        through the Balmer series, so the emission rate of photons in
        a given line is q_h * alpha_eff(line)/alpha_B.
        """
        alpha_b = self.recomb.alpha_B(self.temperature)
        rate = q_h * self.alpha_eff(line) / alpha_b   # photons/s
        return rate * self._PHOTON_ENERGY[line]

    def hbeta_luminosity(self, q_h: float) -> float:
        """L(H beta) (erg/s) for ionizing rate q_h (photons/s)."""
        return self.luminosity('Hbeta', q_h)

    def ionizing_rate_from_hbeta(self, l_hbeta: float) -> float:
        """
        Invert L(H beta) to Q_H (photons/s), Case B, Te = 1e4 K:
        L(Hb) = Q_H * alpha_eff(Hb)/alpha_B * h nu_Hb.
        """
        alpha_b = self.recomb.alpha_B(self.temperature)
        rate = (l_hbeta / self._PHOTON_ENERGY['Hbeta']
                * alpha_b / self.alpha_eff('Hbeta'))
        return rate


# =============================================================================
# MODULE-LEVEL STRÖMGREN RADIUS
# =============================================================================

def stromgren_radius(q_h: float, n_e: float, temperature: float = 1e4,
                     filling_factor: float = 1.0) -> float:
    """
    Strömgren radius (cm) for ionization-bounded sphere.

    R_s = (3 Q_H / (4 pi alpha_B n_e^2 f))^(1/3)

    Args:
        q_h: Ionizing photon rate (photons/s)
        n_e: Electron density (cm^-3)
        temperature: Electron temperature (K)
        filling_factor: Volume filling factor

    Returns:
        Strömgren radius (cm)
    """
    sphere = StromgrenSphere()
    return sphere.stromgren_radius(q_h, n_e, temperature, filling_factor)


# =============================================================================
# FACTORY
# =============================================================================

def get_diagnostics_calculator() -> NebularDiagnosticsCalculator:
    """Factory: nebular diagnostics calculator (T_e from line ratios)."""
    return NebularDiagnosticsCalculator()
