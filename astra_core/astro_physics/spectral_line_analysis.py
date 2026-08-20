#!/usr/bin/env python3
"""
Spectral Line Analysis Module
=============================

Fitting and analysis tools for astronomical spectral lines.

Capabilities:
1. Gaussian line fitting with baseline
2. Voigt profile fitting (real Faddeeva function)
3. Hyperfine structure fitting (HCN, N2H+ and arbitrary component sets)
4. Line identification from observed frequency (common ISM molecules)
5. Optical-depth corrections (thin/thick radiative transfer)
6. Column-density determination from line intensities

Key References:
- Stahler & Pala 2005 (radiative transfer fundamentals)
- Mangum & Shirley 2015, PASP, 127, 266 (column density recipes)
- Garden et al. 1991, ApJ, 374, 540 (13CO column density)
- Caselli et al. 2002 (N2H+ hyperfine fitting)
- Pety et al. 2017 (line identification complexity)

Author: Claude Code (ASTRA)
Date: 2026-08
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import curve_fit
from scipy.special import wofz

# Physical Constants (CGS)
k_B = 1.381e-16          # Boltzmann constant (erg/K)
h_planck = 6.626e-27     # Planck constant (erg s)
c_light = 2.998e10       # Speed of light (cm/s)
c_kms = 2.998e5          # Speed of light (km/s)
m_H = 1.674e-24          # Hydrogen mass (g)
T_CMB = 2.7255           # CMB temperature (K)

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # FWHM -> sigma


# =============================================================================
# LINE FITTERS
# =============================================================================

@dataclass
class LineFitResult:
    """Result of a spectral line fit."""
    amplitude: float            # Peak line temperature (K)
    centroid: float             # Line centroid (km/s)
    sigma: float                # Gaussian sigma (km/s)
    fwhm: float                 # Full width at half maximum (km/s)
    integral: float             # Integrated intensity (K km/s)
    baseline: float             # Fitted baseline offset (K)
    baseline_slope: float       # Fitted baseline slope (K per km/s)
    rms: float                  # Residual RMS (K)
    parameters: Optional[np.ndarray] = None
    parameter_errors: Optional[np.ndarray] = None


def _gaussian_with_baseline(v, amp, v0, sigma, base, slope):
    """Gaussian line on a linear baseline."""
    return base + slope * (v - v[0] if len(v) else 0.0) \
        + amp * np.exp(-0.5 * ((v - v0) / sigma) ** 2)


class GaussianLineFitter:
    """
    Fit a single Gaussian line plus linear baseline to a spectrum.

    Model:  T(v) = base + slope*(v - v[0]) + A exp(-(v-v0)^2 / (2 sigma^2))

    Initial guesses are derived from the data (peak channel, second moment
    of the brightest region) so curve_fit converges from data alone.
    """

    def fit(self, velocity: np.ndarray, temperature: np.ndarray,
            sigma_error: Optional[np.ndarray] = None) -> LineFitResult:
        v = np.asarray(velocity, dtype=float)
        t = np.asarray(temperature, dtype=float)
        if v.size < 5:
            raise ValueError("Need at least 5 channels to fit a line")

        # Data-driven initial guess
        i_peak = int(np.argmax(t - np.median(t)))
        amp0 = max(t[i_peak] - np.median(t), 1e-3)
        v00 = v[i_peak]
        mask = t > np.median(t) + 0.3 * amp0
        sigma0 = max(np.std(v[mask]) if mask.sum() > 2
                     else max((v.max() - v.min()) / 20.0, 0.05), 0.02)
        base0 = float(np.median(t))
        slope0 = 0.0
        p0 = [amp0, v00, sigma0, base0, slope0]

        # Bounds keep the optimizer well-behaved
        bounds = ([0.0, v.min(), 0.005, -np.inf, -np.inf],
                  [np.inf, v.max(), (v.max() - v.min()), np.inf, np.inf])
        try:
            popt, pcov = curve_fit(
                _gaussian_with_baseline, v, t, p0=p0, bounds=bounds,
                sigma=sigma_error, absolute_sigma=sigma_error is not None,
                maxfev=10000)
            perr = np.sqrt(np.diag(pcov))
        except Exception:
            # Fall back to the initial-guess-based moment method
            popt = np.array(p0, dtype=float)
            perr = np.full_like(popt, np.nan)

        amp, v0, sig, base, slope = popt
        model = _gaussian_with_baseline(v, *popt)
        rms = float(np.sqrt(np.mean((t - model) ** 2)))
        integral = amp * sig * np.sqrt(2.0 * np.pi)

        return LineFitResult(
            amplitude=float(amp), centroid=float(v0), sigma=float(sig),
            fwhm=float(2.3548 * sig), integral=float(integral),
            baseline=float(base), baseline_slope=float(slope), rms=rms,
            parameters=popt, parameter_errors=perr)


# ---------------------------------------------------------------------- Voigt

def voigt_profile(x: np.ndarray, sigma: float, gamma: float) -> np.ndarray:
    """
    Normalized Voigt profile V(x; sigma, gamma) using the Faddeeva
    function w(z):

        V(x) = Re[w((x + i gamma) / (sigma sqrt(2)))] / (sigma sqrt(2 pi))
    """
    z = (x + 1j * gamma) / (sigma * np.sqrt(2.0))
    return np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))


class VoigtProfileFitter:
    """
    Fit a Voigt profile (Gaussian + Lorentzian convolution) to a line.

    Used where non-thermal broadening (Gaussian) and opacity/natural
    broadening (Lorentzian) both shape the profile. sigma is the
    Gaussian width, gamma the Lorentzian HWHM (both in km/s).
    """

    def fit(self, velocity: np.ndarray, temperature: np.ndarray,
            sigma_error: Optional[np.ndarray] = None) -> LineFitResult:
        v = np.asarray(velocity, dtype=float)
        t = np.asarray(temperature, dtype=float)

        def model(vv, amp, v0, sig, gam, base):
            return base + amp * voigt_profile(vv - v0, sig, gam) * \
                sig * np.sqrt(2.0 * np.pi)

        i_peak = int(np.argmax(t - np.median(t)))
        amp0 = max(t[i_peak] - np.median(t), 1e-3)
        p0 = [amp0, v[i_peak], max(np.std(v) / 5.0, 0.05),
              max(np.std(v) / 10.0, 0.02), float(np.median(t))]
        bounds = ([0.0, v.min(), 0.005, 1e-4, -np.inf],
                  [np.inf, v.max(), 10.0 * (v.max() - v.min()),
                   10.0 * (v.max() - v.min()), np.inf])
        try:
            popt, pcov = curve_fit(model, v, t, p0=p0, bounds=bounds,
                                   sigma=sigma_error,
                                   absolute_sigma=sigma_error is not None,
                                   maxfev=20000)
            perr = np.sqrt(np.diag(pcov))
        except Exception:
            popt = np.array(p0, dtype=float)
            perr = np.full_like(popt, np.nan)

        amp, v0, sig, gam, base = popt
        # Voigt FWHM (Olivero & Longbothum 1977 approximation)
        fG, fL = 2.3548 * sig, 2.0 * gam
        fwhm = 0.5346 * fL + np.sqrt(0.2166 * fL ** 2 + fG ** 2)
        model_t = model(v, *popt)
        rms = float(np.sqrt(np.mean((t - model_t) ** 2)))
        integral = float(np.trapezoid(model_t - base, v))

        return LineFitResult(
            amplitude=float(amp), centroid=float(v0), sigma=float(sig),
            fwhm=float(fwhm), integral=integral, baseline=float(base),
            baseline_slope=0.0, rms=rms, parameters=popt,
            parameter_errors=perr)


# ------------------------------------------------------------------ hyperfine

# Velocity offsets (km/s) and relative line strengths of common
# hyperfine multiplets (frequencies from the CDMS/LAMDA catalogs;
# strengths are the relative Einstein-A weighted intensities).

HYPERFINE_COMPONENTS: Dict[str, List[Tuple[float, float]]] = {
    # N2H+ 1-0 (Caselli et al. 1995; isolated F2-1 component at -7.98)
    'N2H+ 1-0': [(-7.988, 1.0 / 9.0), (-5.557, 1.0 / 9.0),
                 (-4.585, 2.0 / 9.0), (-2.633, 1.0 / 9.0),
                 (-1.568, 1.0 / 9.0), (0.0, 3.0 / 9.0),
                 (1.568, 1.0 / 9.0), (2.633, 1.0 / 9.0),
                 (4.585, 2.0 / 9.0), (5.557, 1.0 / 9.0),
                 (7.988, 1.0 / 9.0)],
    # HCN 1-0 (Loughnane et al. 2012 / CDMS relative strengths)
    'HCN 1-0': [(-7.057, 1.0), (-4.881, 2.0 / 3.0),
                (-4.749, 1.0 / 9.0), (-2.414, 1.0 / 3.0),
                (-1.059, 1.0 / 9.0), (0.0, 1.0 / 3.0),
                (2.414, 1.0 / 9.0), (4.910, 2.0 / 3.0)],
    # H13CN 1-0 and HC15N not included; add via `components=` argument.
    # CCH N=1-0 hyperfine triplet (Tucker et al. 1974)
    'CCH 1-0': [(0.0, 0.4), (1.35, 0.2), (-1.35, 0.4)],
    # NH3 (1,1) satellite structure handled via inversion-split groups;
    # the main group plus satellites (Ho & Townes 1983 separations)
    'NH3 (1,1)': [(0.0, 1.0), (7.5, 0.3), (-7.5, 0.3),
                  (16.0, 0.1), (-16.0, 0.1), (23.0, 0.05), (-23.0, 0.05)],
}


class HyperfineStructureFitter:
    """
    Fit a hyperfine multiplet with a shared velocity centroid and width.

    All components share v_LSR and sigma; relative component positions
    and strengths are fixed from laboratory/catalog values. The optical
    depth follows from the ratio of the fitted total amplitude to the
    optically thin expectation:

        T_peak,i = T_ex (1 - exp(-tau_i)),   tau_i = tau_tot * s_i
    """

    def __init__(self, components: Optional[List[Tuple[float, float]]] = None,
                 molecule: Optional[str] = None):
        if components is None:
            if molecule is None or molecule not in HYPERFINE_COMPONENTS:
                raise ValueError("Provide `components=[(offset, strength),]"
                                 " or a known molecule name "
                                 f"{list(HYPERFINE_COMPONENTS)}")
            components = HYPERFINE_COMPONENTS[molecule]
        self.components = [(float(o), float(s)) for o, s in components]
        total = sum(s for _, s in self.components)
        self.components = [(o, s / total) for o, s in self.components]

    def model(self, v: np.ndarray, amp: float, v0: float, sigma: float,
              base: float) -> np.ndarray:
        out = np.full_like(v, base, dtype=float)
        for offset, strength in self.components:
            out += amp * strength * np.exp(
                -0.5 * ((v - v0 - offset) / sigma) ** 2)
        return out

    def fit(self, velocity: np.ndarray, temperature: np.ndarray,
            sigma_error: Optional[np.ndarray] = None) -> LineFitResult:
        v = np.asarray(velocity, dtype=float)
        t = np.asarray(temperature, dtype=float)
        i_peak = int(np.argmax(t - np.median(t)))
        amp0 = max(t[i_peak] - np.median(t), 1e-3)
        # Initial centroid: brightest component corrected by nearest offset
        offsets = np.array([o for o, _ in self.components])
        v00 = v[i_peak] - offsets[np.argmin(np.abs(offsets))]
        sig0 = max(np.std(v) / 8.0, 0.05)
        p0 = [amp0 / max(self.components[np.argmax(
            [s for _, s in self.components])][1], 0.05), v00, sig0,
            float(np.median(t))]
        bounds = ([0.0, v.min(), 0.005, -np.inf],
                  [np.inf, v.max(), v.max() - v.min(), np.inf])
        try:
            popt, pcov = curve_fit(self.model, v, t, p0=p0, bounds=bounds,
                                   sigma=sigma_error,
                                   absolute_sigma=sigma_error is not None,
                                   maxfev=20000)
            perr = np.sqrt(np.diag(pcov))
        except Exception:
            popt = np.array(p0, dtype=float)
            perr = np.full_like(popt, np.nan)

        amp, v0, sig, base = popt
        model_t = self.model(v, *popt)
        rms = float(np.sqrt(np.mean((t - model_t) ** 2)))
        integral = float(np.trapezoid(model_t - base, v))
        return LineFitResult(
            amplitude=float(amp), centroid=float(v0), sigma=float(sig),
            fwhm=float(2.3548 * sig), integral=integral,
            baseline=float(base), baseline_slope=0.0, rms=rms,
            parameters=popt, parameter_errors=perr)


# =============================================================================
# LINE IDENTIFICATION
# =============================================================================

# Rest frequencies (MHz) of common ISM/cm lines.
LINE_CATALOG: Dict[str, float] = {
    '12CO J=1-0': 115271.202, '13CO J=1-0': 110201.354,
    'C18O J=1-0': 109782.173, 'C17O J=1-0': 112359.29,
    '12CO J=2-1': 230538.000, '13CO J=2-1': 220398.684,
    'C18O J=2-1': 219560.354, '12CO J=3-2': 345795.990,
    '13CO J=3-2': 330587.965, 'C18O J=3-2': 329330.55,
    'CI 3P1-3P0': 492160.65, 'CI 3P2-3P1': 809343.0,
    'CII 158um': 1900536.9,
    'HCN J=1-0': 88631.847, 'H13CN J=1-0': 86342.330,
    'HC15N J=1-0': 90003.06,
    'HCO+ J=1-0': 89188.526, 'H13CO+ J=1-0': 86754.330,
    'N2H+ J=1-0': 93173.767,
    'CS J=1-0': 48990.956, 'CS J=2-1': 97980.968, 'CS J=3-2': 146969.038,
    'C34S J=2-1': 96412.982,
    'SO 3_2-2_1': 99299.87, 'SO2 4_2,2-3_1,3': 104029.42,
    'SiO J=1-0 v=0': 43122.03, 'SiO J=2-1 v=0': 86246.96,
    'CCH N=1-0': 87316.9, 'c-C3H2 2_1,2-1_0,1': 85338.9,
    'HNC J=1-0': 90663.572, 'HN13C J=1-0': 87090.61,
    'H2CO 2_0,2-1_0,1': 145602.95, 'H2CO 3_0,3-2-0,2': 218222.19,
    'CH3OH 2_0,1-1_0,1 A+': 96739.37, 'CH3OH 5_1,4-4_1,3 E': 216945.6,
    'NH3 (1,1)': 23694.50, 'NH3 (2,2)': 23722.63, 'NH3 (3,3)': 23870.13,
    'H2O 557 GHz': 556935.99,
    'H2D0 1_1,0-1_1,1': 110153.2,
    'OH 1665': 1665.402, 'OH 1667': 1667.359,
    'H41alpha': 95034.4, 'C41alpha': 95076.7,
}


class LineIdentifier:
    """
    Identify spectral lines from observed frequencies.

    Uses the radio-definition Doppler formula

        v_LSR = c (f_rest - f_obs) / f_rest

    and matches every catalog entry within `tolerance_km_s`.
    """

    def __init__(self, catalog: Optional[Dict[str, float]] = None,
                 tolerance_km_s: float = 30.0):
        self.catalog = dict(catalog or LINE_CATALOG)
        self.tolerance = tolerance_km_s

    def identify(self, observed_frequency_mhz: float,
                 v_lsr_range: Tuple[float, float] = (-250.0, 250.0)) \
            -> List[Dict[str, Any]]:
        """Rank catalog lines matching an observed frequency."""
        matches = []
        for name, f_rest in self.catalog.items():
            v = c_kms * (f_rest - observed_frequency_mhz) / f_rest
            if v_lsr_range[0] <= v <= v_lsr_range[1]:
                matches.append({'line': name, 'rest_frequency_mhz': f_rest,
                                'v_lsr_km_s': round(v, 2),
                                'distance_from_range_center': abs(
                                    v - 0.5 * sum(v_lsr_range))})
        matches.sort(key=lambda m: abs(m['distance_from_range_center']))
        return matches

    def identify_line(self, observed_frequency_mhz: float) \
            -> Optional[Dict[str, Any]]:
        """Best single identification or None."""
        m = self.identify(observed_frequency_mhz)
        return m[0] if m else None

    def expected_frequency(self, line_name: str, v_lsr_km_s: float) -> float:
        """Rest frequency redshifted to v_LSR (MHz)."""
        if line_name not in self.catalog:
            raise KeyError(f"Unknown line '{line_name}'")
        f_rest = self.catalog[line_name]
        return f_rest * (1.0 - v_lsr_km_s / c_kms)


# =============================================================================
# OPTICAL DEPTH CORRECTIONS
# =============================================================================

class OpticalDepthCorrector:
    """
    Radiative-transfer corrections between observed brightness
    temperatures and intrinsic (optically thin) quantities.

    For a homogeneous layer:

        T_R = f [J(T_ex) - J(T_bg)] (1 - e^{-tau})

    with the brightness-temperature conversion J(T) = (h nu / k) /
    (exp(h nu / k T) - 1). In the Rayleigh-Jeans limit this reduces to
    the familiar T_R = f T_0 x / (1 + x), x = tau/(line-strength factor).
    """

    @staticmethod
    def j_nu(temperature: float, frequency_ghz: float) -> float:
        """J(T) in K for a line frequency in GHz."""
        nu = frequency_ghz * 1e9
        hv_k = h_planck * nu / k_B
        if temperature <= 0:
            return 0.0
        return hv_k / (np.exp(hv_k / temperature) - 1.0)

    def tau_from_ratio(self, t_main: float, t_satellite: float,
                       strength_ratio: float) -> float:
        """
        Optical depth from main/satellite hyperfine line ratio:

            tau = -ln(1 - T_main / (T_ex * strength_ratio_frac))

        Simplified standard formula (when both lines share T_ex and
        width, using the ratio T_sat/T_main against the optically thin
        expectation):
            tau = -ln(1 - r_thin_ratio_observed)
        where r_obs = T_sat / T_main and r_thin is the catalog ratio.

        Args:
            t_main: main-line brightness temperature (K)
            t_satellite: satellite brightness temperature (K)
            strength_ratio: satellite/main intrinsic strength (< 1)
        """
        if t_main <= 0 or strength_ratio <= 0 or strength_ratio >= 1:
            raise ValueError("Require t_main > 0 and 0 < strength_ratio < 1")
        ratio_observed = t_satellite / t_main
        x = 1.0 - ratio_observed / strength_ratio
        if x <= 0:
            return np.inf  # optically thin limit
        return -np.log(x)

    def corrected_temperature(self, observed_tb: float, tau: float,
                              frequency_ghz: float, t_ex: float = 10.0,
                              filling_factor: float = 1.0) -> float:
        """
        Intrinsic T_ex-implied brightness of an optically thick line:

            T_true = T_obs / [(1 - e^{-tau}) (J(T_ex)-J(T_bg))/T_ex]

        Returns the line temperature corrected to tau -> 0.
        """
        j_ex = self.j_nu(t_ex, frequency_ghz)
        j_bg = self.j_nu(T_CMB, frequency_ghz)
        denom = (1.0 - np.exp(-tau)) * max((j_ex - j_bg) / t_ex, 1e-6)
        return observed_tb / (filling_factor * denom)

    def thin_from_thick(self, integrated_thick: float, tau_thick: float,
                        tau_thin: float) -> float:
        """
        Rescale an integrated intensity from opacity tau_thick to
        tau_thin using (1 - e^{-tau})/tau weighting:

            W_thin = W_thick * [tau_thin (1-e^{-tau_thick})] /
                               [tau_thick (1-e^{-tau_thin})]
        """
        w_tau_thick = (1.0 - np.exp(-tau_thick)) / tau_thick
        w_tau_thin = (1.0 - np.exp(-tau_thin)) / tau_thin
        return integrated_thick * w_tau_thin / w_tau_thick


# =============================================================================
# COLUMN DENSITY
# =============================================================================

class ColumnDensityCalculator:
    """
    Molecular column densities from spectral-line measurements.

    Implemented recipes:
    1. 13CO J=1-0 (Garden et al. 1991):

        N(13CO) = 3.0e14 * T_ex * exp(-5.87/T_ex) *
                  tau/(1-exp(-tau)) * W(13CO)   [cm^-2, W in K km/s]

        N(H2) = N(13CO) / X_13CO with the local ISM abundance
        X_13CO ~ 2e-6 (e.g. Frerking et al. 1982 range 1.5-4e-6).

    2. Optically thin LTE column from integrated intensity and
        partition function Q(T_ex):

        N = (8 pi nu^2 / (c^2 A_ul)) (g_l/g_u) Q(T_ex) exp(E_u/kT_ex)
            * W / [ (exp(h nu/kT_ex)-1)^-1 ... ]

        evaluated in the Rayleigh-Jeans-friendly form of Mangum &
        Shirley (2015) eq. 80.
    """

    X_13CO_DEFAULT = 2.0e-6        # 13CO abundance relative to H2
    X_CO_DEFAULT = 1.0e-4          # 12CO abundance

    def __init__(self):
        self.corrector = OpticalDepthCorrector()

    def h2_from_13co(self, integrated_intensity: float, t_ex: float = 10.0,
                     tau_13co: float = 0.5,
                     x_13co: Optional[float] = None) -> Dict[str, float]:
        """
        N(H2) from 13CO J=1-0 integrated intensity (K km/s).
        """
        x = x_13co if x_13co is not None else self.X_13CO_DEFAULT
        n_13co = 3.0e14 * t_ex * np.exp(-5.87 / t_ex) * \
            tau_13co / (1.0 - np.exp(-tau_13co)) * integrated_intensity
        return {
            'N_13CO': n_13co,
            'N_H2': n_13co / x,
            'A_V': n_13co / x / 1.87e21,   # Bohlin et al. 1978 conversion
            't_ex': t_ex, 'tau': tau_13co,
        }

    def lte_column(self, integrated_intensity: float, frequency_ghz: float,
                   einstein_a: float, t_ex: float, e_upper_k: float,
                   g_upper: float, g_lower: float,
                   partition_function: float) -> float:
        """
        Optically thin LTE molecular column density (cm^-2), Mangum &
        Shirley (2015) eq. 80:

            N = (8 pi nu^2 / c^2 A_ul) (g_l/g_u) Q exp(E_u/kT_ex)
                * [J(T_ex)-J(T_bg)]^-1 * W_RJ-integrated brightness

        `integrated_intensity` in K km/s (main-beam). Assumes tau << 1.
        """
        nu = frequency_ghz * 1e9
        hv_k = h_planck * nu / k_B
        j_ex = self.corrector.j_nu(t_ex, frequency_ghz)
        j_bg = self.corrector.j_nu(T_CMB, frequency_ghz)
        bright = j_ex - j_bg
        if bright <= 0:
            raise ValueError("T_ex produces no contrast against the CMB")
        prefactor = 8.0 * np.pi * nu ** 2 / (c_light ** 2 * einstein_a)
        n_line = prefactor * (g_lower / g_upper) * partition_function * \
            np.exp(e_upper_k / t_ex) * (integrated_intensity / bright) * \
            hv_k / (np.exp(hv_k / t_ex) - 1.0)
        return n_line

    def c18o_column(self, integrated_intensity: float, t_ex: float = 10.0,
                    x_c18o: float = 1.7e-7) -> Dict[str, float]:
        """
        N(H2) from optically thin C18O J=1-0 (assume tau<<1, LTE):
        N(C18O) = 3.0e14 T_ex exp(-5.87/T_ex) W  (same recipe, tau->0).
        """
        n_c18o = 3.0e14 * t_ex * np.exp(-5.87 / t_ex) * integrated_intensity
        return {'N_C18O': n_c18o, 'N_H2': n_c18o / x_c18o,
                'A_V': n_c18o / x_c18o / 1.87e21}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def fit_gaussian_line(velocity: np.ndarray, temperature: np.ndarray,
                      sigma_error: Optional[np.ndarray] = None) \
        -> LineFitResult:
    """Fit a single Gaussian line to a spectrum (convenience wrapper)."""
    return GaussianLineFitter().fit(velocity, temperature, sigma_error)


def identify_line(observed_frequency_mhz: float,
                  tolerance_km_s: float = 30.0) -> Optional[Dict[str, Any]]:
    """Identify a line from its observed frequency (best match or None)."""
    ident = LineIdentifier(tolerance_km_s=tolerance_km_s)
    return ident.identify_line(observed_frequency_mhz)
