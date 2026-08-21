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
Infrared and Submillimeter Astronomy Module

Comprehensive analysis of infrared and submillimeter observations.
Supports data from Spitzer, Herschel, JWST, SOFIA, ALMA, NOEMA, JCMT.

Key capabilities:
- Dust emission modeling (modified blackbody)
- SED fitting across IR/submm
- PAH feature analysis
- Spectral energy distributions
- Color-color diagrams
- Redshift estimation from submm
- Cold dust temperature
- Gas mass from dust emission
- Line cooling calculations

Date: 2025-12-22
Version: 1.0
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy import constants
from scipy.optimize import curve_fit
import warnings

# Physical constants (CGS)
H_PLANCK = 6.626e-27  # erg s
K_BOLTZMANN = 1.381e-16  # erg/K
C_LIGHT = 2.998e10  # cm/s
M_H = 1.673e-24  # g
PC = 3.086e18  # cm
JANSKY = 1e-23  # erg/s/cm^2/Hz
L_SUN = 3.828e33  # erg/s
M_SUN = 1.989e33  # g


class IRBand(Enum):
    """Infrared and submillimeter bands"""
    # Near-IR
    IRAC_3_6 = "irac_3_6"  # 3.6 microns
    IRAC_4_5 = "irac_4_5"  # 4.5 microns
    IRAC_5_8 = "irac_5_8"  # 5.8 microns
    IRAC_8_0 = "irac_8_0"  # 8.0 microns
    # Mid-IR
    WISE_12 = "wise_12"  # 12 microns
    WISE_22 = "wise_22"  # 22 microns
    WISE_24 = "wise_24"  # 24 microns
    MIPS_24 = "mips_24"  # 24 microns
    # Far-IR
    PACS_70 = "pacs_70"  # 70 microns
    PACS_100 = "pacs_100"  # 100 microns
    PACS_160 = "pacs_160"  # 160 microns
    SPIRE_250 = "spire_250"  # 250 microns
    SPIRE_350 = "spire_350"  # 350 microns
    SPIRE_500 = "spire_500"  # 500 microns
    # Submillimeter
    SCUBA_450 = "scuba_450"  # 450 microns
    SCUBA_850 = "scuba_850"  # 850 microns
    # ALMA bands
    ALMA_BAND3 = "alma_band3"  # 3 mm (100 GHz)
    ALMA_BAND6 = "alma_band6"  # 1 mm (230 GHz)
    ALMA_BAND7 = "alma_band7"  # 0.87 mm (345 GHz)


class PAHFeature(Enum):
    """Polycyclic Aromatic Hydrocarbon features"""
    PAH_3_3 = "pah_3_3"  # 3.3 microns
    PAH_6_2 = "pah_6_2"  # 6.2 microns
    PAH_7_7 = "pah_7_7"  # 7.7 microns
    PAH_8_6 = "pah_8_6"  # 8.6 microns
    PAH_11_3 = "pah_11_3"  # 11.3 microns
    PAH_12_7 = "pah_12_7"  # 12.7 microns


@dataclass
class IRPhotometry:
    """Infrared/submillimeter photometry point"""
    band: Union[IRBand, str]
    wavelength: float  # microns
    flux: float  # Jy
    flux_err: float = 0.0
    frequency: float = 0.0  # Hz (calculated from wavelength)
    facility: str = ""

    def __post_init__(self):
        if self.frequency == 0:
            # Convert wavelength to frequency
            lam_cm = self.wavelength * 1e-4  # microns to cm
            self.frequency = C_LIGHT / lam_cm


@dataclass
class PAHSpectrum:
    """PAH emission spectrum"""
    features: Dict[PAHFeature, float] = field(default_factory=dict)
    continuum: Dict[str, float] = field(default_factory=dict)
    feature_ratios: Dict[str, float] = field(default_factory=dict)


@dataclass
class DustProperties:
    """Dust properties from SED fitting"""
    temperature: float  # K
    mass: float  # Msun
    beta: float = 1.5  # Emissivity index
    luminosity: float = 0.0  # Lsun
    power: float = 0.0  # erg/s


class ModifiedBlackbody:
    """
    Modified blackbody dust emission model.

    I_nu = tau_nu * B_nu(T)
    tau_nu = kappa_nu * (M_dust / D^2)
    kappa_nu = kappa_0 * (nu/nu_0)^beta

    Where:
    - B_nu is Planck function
    - kappa_nu is dust opacity
    - beta is emissivity index (1.5-2.0 typical)
    """

    def __init__(self, kappa_0: float = 10.0, beta: float = 1.5,
                 distance_mpc: float = 1.0):
        """
        Initialize modified blackbody.

        Args:
            kappa_0: Reference opacity at lambda_0 (cm^2/g)
            beta: Emissivity index
            distance_mpc: default distance used by fit()
        """
        self.kappa_0 = kappa_0  # at 350 microns
        self.lambda_0 = 350.0  # microns
        self.beta = beta
        self.distance = distance_mpc

    def planck_function(self, wavelength: float, temperature: float) -> float:
        """
        Planck function B_nu(T) at the frequency of the given
        wavelength.

        Args:
            wavelength: Wavelength (microns)
            temperature: Temperature (K)

        Returns:
            B_nu in erg/s/cm^2/Hz/sr
        """
        lam_cm = wavelength * 1e-4  # microns to cm
        nu = C_LIGHT / lam_cm

        x = H_PLANCK * nu / (K_BOLTZMANN * temperature)
        if x > 100:               # avoid overflow
            return 0.0

        return 2.0 * H_PLANCK * nu ** 3 / C_LIGHT ** 2 / np.expm1(x)

    def opacity(self, wavelength: float) -> float:
        """
        Dust opacity kappa_nu = kappa_0 (lambda/lambda_0)^(-beta)
        (cm^2/g of whatever the kappa_0 convention references).

        Args:
            wavelength: Wavelength (microns)

        Returns:
            Opacity (cm^2/g)
        """
        return self.kappa_0 * (wavelength / self.lambda_0) ** (-self.beta)

    def flux_density(self, wavelength: float, temperature: float,
                    dust_mass: float, distance: float = 1.0,
                    beta: Optional[float] = None) -> float:
        """
        Optically thin modified-blackbody flux density

            S_nu = M_dust * kappa_nu * B_nu(T) / D^2

        (Replaces a version that divided a per-cm B_lambda flux by the
        per-Hz Jansky unit - dimensionally incorrect.)

        Args:
            wavelength: Wavelength (microns)
            temperature: Dust temperature (K)
            dust_mass: Dust mass (Msun)
            distance: Distance (Mpc)
            beta: override of the emissivity index

        Returns:
            Flux density (Jy)
        """
        beta = self.beta if beta is None else beta
        kappa = self.kappa_0 * (wavelength / self.lambda_0) ** (-beta)
        b_nu = self.planck_function(wavelength, temperature)
        dist_cm = distance * 1e6 * PC            # Mpc to cm
        mass_g = dust_mass * M_SUN

        flux_nu = mass_g * kappa * b_nu / dist_cm ** 2   # erg/s/cm^2/Hz
        return flux_nu / JANSKY

    def fit(self, wavelengths: np.ndarray, fluxes: np.ndarray,
           flux_errs: np.ndarray = None) -> Dict[str, Any]:
        """
        Fit modified blackbody to photometry.

        Args:
            wavelengths: Wavelengths (microns)
            fluxes: Flux densities (Jy)
            flux_errs: Flux uncertainties (Jy)

        Returns:
            Fit results (temperature, mass, beta)
        """
        if flux_errs is None:
            flux_errs = np.ones_like(fluxes) * 0.1 * np.mean(fluxes)

        p0 = [20.0, max(np.median(fluxes), 1e-3), 1.5]
        sigma = np.asarray(flux_errs, dtype=float)
        sigma = np.where(sigma > 0, sigma, 0.1 * np.mean(fluxes))

        def model(lam_um, temperature, mass_msun, beta):
            return np.array([self.flux_density(float(l), temperature,
                                               mass_msun, self.distance,
                                               beta) for l in lam_um])

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(
                    model, wavelengths, fluxes, p0=p0, sigma=sigma,
                    bounds=([5.0, 1e-8, 0.5], [200.0, 1e12, 3.5]),
                    maxfev=20000)
            perr = np.sqrt(np.diag(pcov))
            resid = (fluxes - model(wavelengths, *popt)) / sigma
            chi2 = float(np.sum(resid ** 2))
            success = True
        except (RuntimeError, ValueError) as exc:
            popt, perr = p0, [np.inf] * 3
            chi2, success = np.inf, False

        return {
            'temperature': float(popt[0]),
            'temperature_err': float(perr[0]),
            'dust_mass_msun': float(popt[1]),
            'dust_mass_err': float(perr[1]),
            'beta': float(popt[2]),
            'beta_err': float(perr[2]),
            'chi2': chi2,
            'dof': max(len(fluxes) - 3, 1),
            'success': success,
        }


# =============================================================================
# IR COLOR ANALYSIS
# =============================================================================

class IRColorAnalysis:
    """
    Far-IR colour temperatures from flux ratios.

    For an MBB, the ratio of fluxes at two wavelengths determines the
    temperature (weakly dependent on beta); solved here by Brent
    root-finding on the model ratio, which is exact for the model.
    """

    def __init__(self, beta: float = 1.5, kappa_0: float = 10.0):
        self.mbb = ModifiedBlackbody(kappa_0=kappa_0, beta=beta)

    def color_temperature(self, lam1: float, lam2: float,
                          flux1: float, flux2: float) -> Dict[str, float]:
        """
        Temperature from the flux ratio F(lam1)/F(lam2).

        Returns:
            dict with 'temperature_K', 'flux_ratio', 'lam1', 'lam2'
        """
        from scipy.optimize import brentq

        ratio = flux1 / flux2

        def f(t):
            m = self.mbb
            return (m.flux_density(lam1, t, 1.0, 1.0)
                    / m.flux_density(lam2, t, 1.0, 1.0) - ratio)

        t = brentq(f, 3.0, 500.0)
        return {'temperature_K': float(t), 'flux_ratio': float(ratio),
                'lam1': lam1, 'lam2': lam2}

    def color_table(self, lam1: float = 70.0, lam2: float = 160.0,
                    temperatures: Optional[np.ndarray] = None
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ratio R(lam1/lam2) as a function of temperature."""
        temps = (np.linspace(10, 60, 51) if temperatures is None
                 else np.asarray(temperatures, dtype=float))
        ratios = np.array([
            self.mbb.flux_density(lam1, t, 1.0, 1.0)
            / self.mbb.flux_density(lam2, t, 1.0, 1.0) for t in temps])
        return temps, ratios


# =============================================================================
# SUBMILLIMETER ANALYSIS
# =============================================================================

class SubmillimeterAnalysis:
    """
    Submillimetre/rayleigh-Jeans slope analysis.

    On the Rayleigh-Jeans side the MBB gives S_nu ~ nu^(2+beta), so
    the slope of log S vs log nu directly yields the emissivity index.
    """

    def __init__(self, distance_mpc: float = 1.0):
        self.distance = distance_mpc

    def emissivity_index(self, wavelengths: np.ndarray,
                         fluxes: np.ndarray,
                         flux_errs: Optional[np.ndarray] = None
                         ) -> Dict[str, float]:
        """
        Fit a power law S_nu = A nu^alpha longward of 250 microns.

        Returns:
            dict with 'alpha', 'beta' (= alpha - 2) and errors
        """
        wavelengths = np.asarray(wavelengths, dtype=float)
        sel = wavelengths >= 250.0
        if sel.sum() < 3:
            return {'alpha': np.nan, 'alpha_err': np.inf,
                    'beta': np.nan, 'beta_err': np.inf, 'n_points': int(sel.sum())}

        lam = wavelengths[sel]
        f = np.asarray(fluxes, dtype=float)[sel]
        nu = C_LIGHT / (lam * 1e-4)

        if flux_errs is None:
            err = np.full_like(f, 0.1 * np.mean(f))
        else:
            err = np.asarray(flux_errs, dtype=float)[sel]
        w = 1.0 / err ** 2

        coeffs = np.polyfit(np.log10(nu), np.log10(f), 1, w=w)
        alpha = float(coeffs[0])
        # slope error for weighted least squares
        var = 1.0 / np.sum(w * (np.log10(nu) - np.average(np.log10(nu),
                                                          weights=w)) ** 2)
        return {'alpha': alpha, 'alpha_err': float(np.sqrt(var)),
                'beta': alpha - 2.0, 'beta_err': float(np.sqrt(var)),
                'n_points': int(sel.sum())}


# =============================================================================
# LINE COOLING
# =============================================================================

class LineCooling:
    """
    Far-IR fine-structure line luminosities and cooling fractions.

    Converts integrated line flux to line luminosity (via nu L_nu) and
    reports the line-to-continuum ratio. Includes the [CII] 158 and
    [OI] 63 micron lines.
    """

    LINE_WAVELENGTHS_UM = {'CII_158': 157.741, 'OI_63': 63.184,
                           'OI_145': 145.525, 'OIII_88': 88.356,
                           'NII_122': 121.898, 'NII_205': 205.178}

    def line_luminosity(self, line: str, integrated_flux_jy_kms: float,
                        distance_mpc: float) -> float:
        """
        Line luminosity from the velocity-integrated line flux.

        Exact first-principles form: for a narrow line the frequency
        integral of the flux density is int S_nu dnu
            = int S_nu (nu/c) dv = S_int * nu / c
        with S_int the velocity integral in Jy km/s, and the line
        energy luminosity is

            L = 4 pi D^2 * int S_nu dnu.

        Args:
            line: key of LINE_WAVELENGTHS_UM
            integrated_flux_jy_kms: velocity-integrated flux (Jy km/s)
            distance_mpc: luminosity distance (Mpc)

        Returns:
            Line luminosity (Lsun)
        """
        lam_um = self.LINE_WAVELENGTHS_UM[line]
        nu = C_LIGHT / (lam_um * 1e-4)          # Hz
        d_cm = distance_mpc * 1e6 * PC

        # integral over frequency of S_nu (erg/s/cm^2): the velocity
        # integral in Jy km/s converts with dnu = nu dv/c. The energy
        # luminosity is then simply L = 4 pi D^2 int S_nu dnu (the
        # integral is already the bolometric line flux).
        s_int = integrated_flux_jy_kms * 1e5 * JANSKY * nu / C_LIGHT
        l_total = 4.0 * np.pi * d_cm ** 2 * s_int
        return float(l_total / L_SUN)

    def cooling_fraction(self, line_luminosity_lsun: float,
                         total_ir_luminosity_lsun: float) -> float:
        """Line luminosity as a fraction of L_TIR."""
        return float(line_luminosity_lsun / total_ir_luminosity_lsun)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def fit_dust_sed(wavelengths: np.ndarray, fluxes: np.ndarray,
                 flux_errs: Optional[np.ndarray] = None,
                 distance_mpc: float = 1.0, beta: float = 1.5,
                 kappa_0: float = 10.0) -> Dict[str, Any]:
    """Fit an MBB to photometry (wrapper around ModifiedBlackbody.fit)."""
    mbb = ModifiedBlackbody(kappa_0=kappa_0, beta=beta)
    mbb.distance = distance_mpc
    return mbb.fit(wavelengths, fluxes, flux_errs)


def calculate_gas_mass(flux_jy: float, wavelength_um: float,
                       temperature_k: float, distance_mpc: float,
                       kappa_0: float = 0.1, beta: float = 2.0,
                       lambda_0_um: float = 250.0) -> float:
    """
    Gas (dust-traced) mass from a submillimetre flux measurement:

        M = S_nu D^2 / (kappa_nu B_nu(T))

    Args:
        kappa_0: gas-referenced opacity at lambda_0 (cm^2/g of gas;
            the canonical Milky Way value at 250 um is ~0.1)

    Returns:
        Gas mass (Msun)
    """
    mbb = ModifiedBlackbody(kappa_0=kappa_0, beta=beta)
    mbb.lambda_0 = lambda_0_um
    b_nu = mbb.planck_function(wavelength_um, temperature_k)
    kappa = mbb.opacity(wavelength_um)
    d_cm = distance_mpc * 1e6 * PC
    mass_g = flux_jy * JANSKY * d_cm ** 2 / (kappa * b_nu)
    return float(mass_g / M_SUN)


def get_ir_color(flux1: float, flux2: float, lam1: float = 70.0,
                 lam2: float = 160.0, beta: float = 1.5) -> Dict[str, float]:
    """Far-IR colour and colour temperature from two fluxes."""
    ca = IRColorAnalysis(beta=beta)
    return ca.color_temperature(lam1, lam2, flux1, flux2)
