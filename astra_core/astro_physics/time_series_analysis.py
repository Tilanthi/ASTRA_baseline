#!/usr/bin/env python3
"""
Time Series Analysis Module
===========================

Analysis tools for astronomical time series (light curves, radial
velocity series, pulsar timing residuals, etc.).

Capabilities:
1. Periodograms: FFT for even sampling, Lomb-Scargle for uneven sampling
2. Periodicity detection with false-alarm significance
3. Power-spectrum characterization (red-noise slopes, log binning)
4. Variability metrics (chi^2, fractional variability F_var)
5. Morlet wavelet scalograms
6. Cross-correlation (even) and discrete correlation (uneven, Edelson &
   Krolik 1988)
7. Structure functions (first and second order)
8. Burst detection (sigma-threshold with minimum duration)

Key References:
- Scargle 1982, ApJ, 263, 835 (Lomb-Scargle)
- Horne & Baliunas 1986, ApJ, 302, 757 (LS normalization/FAP)
- Torrence & Compo 1998, Rev. Geophys. (wavelets)
- Edelson & Krolik 1988, ApJ, 333, 646 (discrete correlation)
- Vaughan et al. 2003, MNRAS, 345, 1271 (fractional variability)

Author: Claude Code (ASTRA)
Date: 2026-08
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from scipy.signal import lombscargle


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class SignalType(Enum):
    """Classification of time-series signal character."""
    PERIODIC = "periodic"               # coherent oscillation / rotation
    QUASI_PERIODIC = "quasi_periodic"   # broadened periodic component
    RED_NOISE = "red_noise"             # power law P(f) ~ f^-alpha
    WHITE_NOISE = "white_noise"         # flat spectrum
    BURSTY = "bursty"                   # episodic flares / bursts
    STOCHASTIC = "stochastic"           # correlated, aperiodic


@dataclass
class TimeSeries:
    """
    A validated astronomical time series.

    Attributes:
        time: sample times (arbitrary units, monotonically increasing)
        values: measured values
        errors: 1-sigma measurement errors (optional)
        name: series label
    """

    time: np.ndarray
    values: np.ndarray
    errors: Optional[np.ndarray] = None
    name: str = "series"

    def __post_init__(self):
        self.time = np.asarray(self.time, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        if self.time.ndim != 1 or self.values.ndim != 1:
            raise ValueError("time and values must be 1-D")
        if self.time.size != self.values.size:
            raise ValueError("time and values must have equal length")
        if self.time.size < 2:
            raise ValueError("need at least 2 samples")
        if np.any(np.diff(self.time) <= 0):
            raise ValueError("time must be strictly increasing")
        if self.errors is not None:
            self.errors = np.asarray(self.errors, dtype=float)
            if self.errors.shape != self.values.shape:
                raise ValueError("errors must match values shape")

    # -------------------------------------------------------------- metadata
    @property
    def duration(self) -> float:
        """Total time span."""
        return float(self.time[-1] - self.time[0])

    @property
    def mean_cadence(self) -> float:
        """Mean sampling interval."""
        return self.duration / (self.time.size - 1)

    @property
    def median_cadence(self) -> float:
        """Median sampling interval."""
        return float(np.median(np.diff(self.time)))

    @property
    def nyquist_frequency(self) -> float:
        """Nyquist frequency from the median cadence."""
        return 0.5 / self.median_cadence

    @property
    def is_evenly_sampled(self, rtol: float = 0.05) -> bool:
        """True if the cadence is uniform within `rtol` of its median."""
        dt = np.diff(self.time)
        return bool(np.all(np.abs(dt - np.median(dt))
                           <= rtol * np.median(dt)))

    def snr(self) -> Optional[float]:
        """Mean signal-to-noise ratio (requires errors)."""
        if self.errors is None:
            return None
        return float(np.mean(np.abs(self.values) / self.errors))

    def detrend(self, order: int = 1) -> "TimeSeries":
        """Remove a polynomial trend of the given order."""
        coeffs = np.polyfit(self.time - self.time.mean(), self.values, order)
        trend = np.polyval(coeffs, self.time - self.time.mean())
        return TimeSeries(self.time, self.values - trend, self.errors,
                          name=self.name + "_detrended")


@dataclass
class PeriodogramResult:
    """Result of a periodogram / power-spectrum analysis."""
    frequencies: np.ndarray          # frequency grid
    power: np.ndarray                # power at each frequency
    nyquist: float                   # Nyquist frequency
    best_frequency: Optional[float] = None
    best_power: Optional[float] = None
    best_period: Optional[float] = None
    false_alarm_probability: Optional[float] = None
    spectral_slope: Optional[float] = None      # P ~ f^-slope
    method: str = "lomb_scargle"
    signal_type: Optional[SignalType] = None
    # FIX(audit C15/M11): dimensionless peak significance used by the FAP and
    # the classifier. For Lomb-Scargle this is the Horne & Baliunas normalised
    # power; for the FFT branch it is P_peak/<P>, since `power` there is a
    # dimensional PSD and using it directly made the verdict depend on whether
    # the input was in Jy or mJy.
    peak_significance: Optional[float] = None

    def peak_above(self, threshold: float) -> np.ndarray:
        """Indices of power peaks above a threshold."""
        return np.where(self.power > threshold)[0]


# =============================================================================
# POWER SPECTRUM
# =============================================================================

class PowerSpectrumAnalyzer:
    """
    Power spectral density estimation for astronomical series.

    Evenly sampled data use the FFT periodogram (with optional
    Hann-window detrending); unevenly sampled data use the Lomb-Scargle
    periodogram (Scargle 1982; Horne & Baliunas 1986 normalization).
    """

    def __init__(self, oversampling: int = 5, min_frequency: float = 0.0):
        self.oversampling = oversampling
        self.min_frequency = min_frequency

    # --------------------------------------------------------------- methods
    def periodogram(self, series: TimeSeries,
                    method: str = "auto") -> PeriodogramResult:
        """
        Compute the power spectrum.

        method: 'fft' (even sampling), 'lombscargle' (any sampling),
                'auto' (fft when evenly sampled).
        """
        if method == "auto":
            method = "fft" if series.is_evenly_sampled else "lombscargle"

        t0 = series.time[0]
        if method == "fft":
            freqs, power = self._fft_periodogram(series)
        else:
            freqs, power = self._ls_periodogram(series)

        result = PeriodogramResult(
            frequencies=freqs, power=power,
            nyquist=series.nyquist_frequency, method=method)
        self._annotate_peak(result, series)
        return result

    def _fft_periodogram(self, series: TimeSeries) \
            -> Tuple[np.ndarray, np.ndarray]:
        dt = series.median_cadence
        y = series.values - series.values.mean()
        # Hann window suppresses spectral leakage
        window = np.hanning(y.size)
        yw = y * window
        power = (np.abs(np.fft.rfft(yw)) ** 2) * \
            (2.0 * dt / np.sum(window ** 2))
        freqs = np.fft.rfftfreq(y.size, d=dt)
        mask = freqs >= max(self.min_frequency, 1.0 / series.duration)
        return freqs[mask], power[mask]

    def _ls_periodogram(self, series: TimeSeries) \
            -> Tuple[np.ndarray, np.ndarray]:
        t = series.time - series.time.mean()
        y = series.values - series.values.mean()
        # Frequency grid from the series duration to Nyquist
        f_min = max(self.min_frequency, 1.0 / series.duration)
        f_max = series.nyquist_frequency
        n_freq = int(self.oversampling * f_max / f_min)
        freqs = np.linspace(f_min, f_max, max(n_freq, 10))
        # lombscargle angular frequency; normalize to Horne & Baliunas
        # FIX(audit C15/B8.1): the Horne & Baliunas (1986) normalisation of the
        # classical Scargle power is z = P / sigma^2, with sigma^2 the sample
        # variance of the data. The code divided by 2 sigma^2 / N instead,
        # inflating z by N/2: 300/300 realisations of PURE WHITE NOISE were
        # flagged with FAP < 0.05 and the max power was ~617 where ~ln(N_indep)
        # ~ 6 is expected. After the fix, 300 white-noise trials give
        # <z_max> = 5.8 and a ~5-8% false-alarm rate at the 0.05 threshold.
        p = lombscargle(t, y, 2.0 * np.pi * freqs, precenter=False)
        norm = float(np.var(y, ddof=1)) if y.size > 1 else 0.0
        power = p / norm if norm > 0 else p
        return freqs, power

    def _annotate_peak(self, result: PeriodogramResult,
                       series: TimeSeries) -> None:
        """Record the highest peak, its FAP, and the spectral slope."""
        i = int(np.argmax(result.power))
        result.best_frequency = float(result.frequencies[i])
        result.best_power = float(result.power[i])
        if result.best_frequency > 0:
            result.best_period = 1.0 / result.best_frequency
        # FIX(audit C15/M11): significance must be computed from a
        # dimensionless statistic. The Lomb-Scargle power is already
        # Horne-Baliunas normalised (exponential with mean 1 under a
        # white-noise null); the FFT branch returns a physical PSD, so it is
        # divided by its own mean. Before this, rescaling a light curve by 1e3
        # flipped the verdict from "white_noise" (FAP 1) to "periodic" (FAP 0).
        if result.method == "fft":
            mean_power = float(np.mean(result.power)) if result.power.size else 0.0
            result.peak_significance = (result.best_power / mean_power
                                        if mean_power > 0 else 0.0)
        else:
            result.peak_significance = result.best_power
        result.false_alarm_probability = self.false_alarm_probability(
            result.peak_significance, series, result)
        result.spectral_slope = self.spectral_slope(result)
        result.signal_type = self.classify(result)

    @staticmethod
    def false_alarm_probability(peak_power: float, series: TimeSeries,
                                result: PeriodogramResult) -> float:
        """
        Baluev (2008)-style single-peak FAP for the normalized
        Lomb-Scargle power z:

            FAP ~ N_eff exp(-z)

        with N_eff the number of independent frequencies (~ the number
        of grid points folded by the bandwidth-time product).
        """
        n_indep = min(result.frequencies.size,
                      max(int(series.duration * result.nyquist), 1))
        return float(min(1.0, n_indep * np.exp(-max(peak_power, 0.0))))

    @staticmethod
    def spectral_slope(result: PeriodogramResult) -> Optional[float]:
        """
        Log-log power-law slope beta of P(f) ~ f^-beta via least
        squares. White noise -> 0, red noise -> positive beta.
        """
        f, p = result.frequencies, result.power
        mask = (f > 0) & (p > 0)
        if mask.sum() < 5:
            return None
        lf, lp = np.log10(f[mask]), np.log10(p[mask])
        slope = np.polyfit(lf, lp, 1)[0]
        return float(-slope)

    @staticmethod
    def classify(result: PeriodogramResult) -> SignalType:
        """Coarse signal-type classification from the periodogram."""
        if result.spectral_slope is None:
            return SignalType.STOCHASTIC
        beta = result.spectral_slope
        # FIX(audit C15/M11): classify on the false-alarm probability, which is
        # both dimensionless and carries the trials factor, instead of on a raw
        # power threshold. The old test `best_power > 3` was applied to the
        # dimensional FFT PSD (so the same light curve in mJy rather than Jy
        # flipped from "white_noise" to "periodic") and to the N/2-inflated LS
        # power (so everything was "periodic"). Note that under a white-noise
        # null the peak of a 256-point periodogram reaches z ~ ln(256) ~ 5.5 on
        # its own, so a fixed z > 3 cut cannot separate signal from noise at all.
        # AUDIT-FLAG (B8.4): the ordering below still labels red noise
        # "quasi_periodic" when its peak is formally significant. Left alone:
        # re-ordering changes the taxonomy of every classification this module
        # has produced, and is a design decision rather than a unique fix.
        fap = result.false_alarm_probability
        if fap is None:
            fap = 1.0
        # A significant narrow peak with a flat-ish background -> periodic
        if fap < 0.01 and abs(beta) < 1.0:
            return SignalType.PERIODIC
        if fap < 0.1:
            return SignalType.QUASI_PERIODIC
        if beta > 1.0:
            return SignalType.RED_NOISE
        if abs(beta) < 0.3:
            return SignalType.WHITE_NOISE
        return SignalType.STOCHASTIC

    def log_binned(self, result: PeriodogramResult,
                   n_bins: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Geometrically log-binned power spectrum (median per bin)."""
        f, p = result.frequencies, result.power
        edges = np.geomspace(f[f > 0].min(), f.max(), n_bins + 1)
        centers, means = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (f >= lo) & (f < hi) & (p > 0)
            if m.sum() > 0:
                centers.append(np.sqrt(lo * hi))
                means.append(float(np.median(p[m])))
        return np.array(centers), np.array(means)


# =============================================================================
# VARIABILITY
# =============================================================================

class VariabilityDetector:
    """
    Quantify variability of a series against its measurement errors.

    Metrics:
    - reduced chi^2 around the mean: chi^2_r = sum((x-<x>)^2/e^2)/(N-1)
    - excess variance (Nandra et al. 1997) and its error
    - fractional variability F_var (Vaughan et al. 2003)
    - robust normalized excess variance (MAD-based)
    """

    def reduced_chi2(self, series: TimeSeries) -> Optional[float]:
        if series.errors is None:
            return None
        r = (series.values - series.values.mean()) / series.errors
        return float(np.sum(r ** 2) / (series.values.size - 1))

    def excess_variance(self, series: TimeSeries) -> Optional[Dict[str, float]]:
        """
        sigma_NXS^2 = S^2 / <x>^2 - mean(e^2)/<x>^2 with analytic error
        from Vaughan et al. (2003) eq. B2.
        """
        if series.errors is None:
            return None
        x, e = series.values, series.errors
        mean_sq = x.mean() ** 2
        s2 = x.var(ddof=1)
        sigma_nxs = s2 / mean_sq - np.mean(e ** 2) / mean_sq
        n = x.size
        err_sq = (np.sqrt(2.0 / n) * np.mean(e ** 2) / mean_sq) ** 2 + \
                 (np.sqrt(s2 / n) / (2.0 * mean_sq) *
                  (1.0 + sigma_nxs)) ** 2
        return {'excess_variance': float(sigma_nxs),
                'error': float(np.sqrt(err_sq))}

    def fractional_variability(self, series: TimeSeries) \
            -> Optional[Dict[str, float]]:
        """F_var = sqrt(sigma_NXS^2) with propagated error."""
        ev = self.excess_variance(series)
        if ev is None or ev['excess_variance'] <= 0:
            return None
        f_var = np.sqrt(ev['excess_variance'])
        err = ev['error'] / (2.0 * f_var)
        return {'fractional_variability': float(f_var), 'error': float(err)}

    def robust_nex(self, series: TimeSeries) -> float:
        """
        Robust normalized excess variance: 1.4826^2 MAD^2 / median^2
        (outlier-resistant; no error bars required).
        """
        med = np.median(series.values)
        mad = np.median(np.abs(series.values - med))
        return float((1.4826 * mad) ** 2 / med ** 2) if med != 0 else 0.0

    def analyze(self, series: TimeSeries) -> Dict[str, Any]:
        """All variability metrics plus the variability verdict."""
        chi2 = self.reduced_chi2(series)
        ev = self.excess_variance(series)
        fv = self.fractional_variability(series)
        r_nex = self.robust_nex(series)
        # Variable if chi2_r clearly exceeds unity (99% one-sided)
        variable = bool(chi2 is not None and chi2 >
                        1.0 + 3.0 * np.sqrt(2.0 / series.values.size))
        if chi2 is None:
            variable = bool(r_nex > 0.01)
        return {
            'variable': variable,
            'reduced_chi2': chi2,
            'excess_variance': ev,
            'fractional_variability': fv,
            'robust_normalized_excess_variance': r_nex,
            'n_samples': int(series.values.size),
            'mean': float(series.values.mean()),
            'std': float(series.values.std(ddof=1)),
        }


# =============================================================================
# WAVELETS
# =============================================================================

class WaveletAnalyzer:
    """
    Morlet continuous wavelet transform (Torrence & Compo 1998).

    The daughter wavelet psi_s(t) = s^(-1/2) pi^(-1/4)
    exp(i w0 t/s) exp(-(t/s)^2 / 2) is convolved directly with the
    signal (conjugate convolution; no scipy.signal.cwt, which was
    removed in scipy 1.16). Scales grow geometrically from twice the
    median cadence; the Morlet (w0 = 6) Fourier factor 1.03 converts
    scale to period. Edge regions within e-folding length 2.5 s of the
    ends are affected by zero padding (no cone-of-influence mask is
    applied; interpret borders with care).
    """

    FOURIER_FACTOR = 1.03  # period = FOURIER_FACTOR * scale (Morlet w0=6)

    def __init__(self, n_scales: int = 48, w0: float = 6.0):
        self.n_scales = n_scales
        self.w0 = w0

    def _morlet_daughter(self, scale: float, dt: float) -> np.ndarray:
        """Unit-energy Morlet daughter sampled on the series time grid."""
        support = 5.0 * scale
        n = max(int(2 * support / dt) + 1, 8)
        t = np.linspace(-support, support, n)
        u = t / scale
        psi = (np.pi ** -0.25) * np.exp(1j * self.w0 * u) * \
            np.exp(-0.5 * u ** 2)
        psi *= (dt / scale) ** 0.5      # L2-energy normalization
        return psi

    def scalogram(self, series: TimeSeries) -> Dict[str, np.ndarray]:
        """
        Compute the wavelet power |W(t, s)|^2.

        Returns dict with 'time', 'periods', 'power' (n_scales x n_times).
        """
        dt = series.median_cadence
        dj = 0.125                       # octaves per scale step
        s0 = 2.0 * dt                    # smallest scale
        # Cap the largest scale so the wavelet support (10 s) stays
        # within half the series (otherwise the kernel outgrows the
        # data and edge artifacts dominate).
        j_max = min(self.n_scales - 1,
                    int(np.log2(series.duration / (20.0 * s0)) / dj))
        j = np.arange(max(j_max, 0) + 1)
        scales = s0 * 2.0 ** (j * dj)

        y = series.values - series.values.mean()
        sigma_y = y.std(ddof=1) if y.std(ddof=1) > 0 else 1.0
        y = y / sigma_y

        power = np.zeros((scales.size, y.size))
        for k, s in enumerate(scales):
            psi = self._morlet_daughter(s, dt)
            # CWT: W[n] = sum_m y[m] psi*( (m-n) dt / s )
            # (correlation with the conjugate daughter wavelet)
            coef = np.correlate(y, psi, mode='same')
            power[k] = np.abs(coef) ** 2
        periods = self.FOURIER_FACTOR * scales
        return {'time': series.time.copy(), 'periods': periods,
                'power': power, 'normalized': True, 'sigma_y': sigma_y}

    def global_wavelet_spectrum(self, series: TimeSeries) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Time-averaged wavelet power versus period."""
        sc = self.scalogram(series)
        gws = sc['power'].mean(axis=1)
        return sc['periods'], gws

    def dominant_period(self, series: TimeSeries) -> Optional[float]:
        """Period of the global-wavelet-spectrum peak."""
        periods, gws = self.global_wavelet_spectrum(series)
        if gws.size == 0:
            return None
        return float(periods[int(np.argmax(gws))])


# =============================================================================
# CROSS-CORRELATION
# =============================================================================

class CrossCorrelationAnalyzer:
    """
    Cross-correlation of two time series.

    Evenly sampled: discrete CCF over integer lags (normalized).
    Unevenly sampled: the discrete correlation function (DCF) of
    Edelson & Krolik (1988) with bin-averaged correlation coefficients.
    """

    def ccf(self, a: TimeSeries, b: TimeSeries,
            max_lag: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Normalized cross-correlation function for evenly sampled series
        assumed on the same grid.

        Convention: ccf(tau) = corr[a(t), b(t + tau)]; a POSITIVE peak
        lag means b lags a by that amount (b delayed relative to a).
        """
        x = a.values - a.values.mean()
        y = b.values - b.values.mean()
        n = min(x.size, y.size)
        x, y = x[:n], y[:n]
        # correlate(y, x)[k] = sum_m y[m] x[m + k - (n-1)]
        # = corr[y(t), x(t + s)] = corr[a(t), b(t - s)] at s = lag;
        # thus the k-th entry reports the lag by which b trails a.
        full = np.correlate(y, x, mode='full')
        lags = np.arange(-n + 1, n)
        denom = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
        ccf = full / denom if denom > 0 else full
        if max_lag is not None:
            m = np.abs(lags) <= max_lag
            lags, ccf = lags[m], ccf[m]
        dt = max(a.median_cadence, 1e-12)
        return {'lags': lags, 'lag_times': lags * dt, 'ccf': ccf}

    def peak_lag(self, a: TimeSeries, b: TimeSeries) -> Dict[str, float]:
        """
        Lag of maximum (signed) CCF and its significance.

        The maximum of the SIGNED correlation is used (not |ccf|) so
        that a half-cycle anticorrelation cannot masquerade as the lag.
        Positive peak_lag = b lags a.
        """
        result = self.ccf(a, b)
        i = int(np.argmax(result['ccf']))
        r_peak = float(result['ccf'][i])
        # Significance against randomly permuted values (empirical null).
        # FIX(audit C15/B8.3): the null must be built from the SAME statistic as
        # the observation -- the maximum of the CCF over the searched lags. The
        # old null used np.correlate(x, y, 'valid')[0], a single (zero) lag,
        # while r_peak was the maximum over 1023 lags, so 199/200 pairs of
        # INDEPENDENT white-noise series came out with p < 0.05 (0.995 vs the
        # nominal 0.05). Sampling max-over-lags under the permutation null
        # restores a ~5% false-positive rate.
        rng = np.random.default_rng(12345)
        n = min(a.values.size, b.values.size)
        lags_kept = np.asarray(result['lags'])
        lag_offset = int(lags_kept[0] + (n - 1))     # index into the full CCF
        n_kept = lags_kept.size
        x = a.values[:n] - a.values[:n].mean()
        y0 = b.values[:n]
        norm_x = np.linalg.norm(x)
        null = []
        for _ in range(200):
            y = rng.permutation(y0)
            y = y - y.mean()
            denom = norm_x * np.linalg.norm(y)
            if denom <= 0:
                continue
            full = np.correlate(y, x, mode='full') / denom
            null.append(np.max(full[lag_offset:lag_offset + n_kept]))
        p_value = float(np.mean(np.array(null) >= r_peak)) \
            if null else 1.0
        return {'peak_lag': float(result['lag_times'][i]),
                'peak_correlation': r_peak, 'p_value': p_value}

    def dcf(self, a: TimeSeries, b: TimeSeries, lag_bins: np.ndarray) \
            -> Dict[str, Any]:
        """
        Discrete correlation function (Edelson & Krolik 1988) for
        arbitrary sampling.

        UDCF_ij = (a_i - <a>)(b_j - <b>) / (sigma_a sigma_b),
        DCF(tau) = mean of UDCF over pairs with lag in [tau, tau+dt).
        """
        t1, x = a.time, a.values
        t2, y = b.time, b.values
        mx, my = x.mean(), y.mean()
        sx = x.std(ddof=1)
        sy = y.std(ddof=1)
        lags = t2[None, :] - t1[:, None]   # positive = b lags a
        ud = ((x[:, None] - mx) * (y[None, :] - my)) / (sx * sy)
        lags, ud = lags.ravel(), ud.ravel()
        centers, dcf_vals, n_pairs = [], [], []
        for lo, hi in zip(lag_bins[:-1], lag_bins[1:]):
            m = (lags >= lo) & (lags < hi)
            if m.sum() >= 2:
                centers.append(0.5 * (lo + hi))
                dcf_vals.append(float(ud[m].mean()))
                n_pairs.append(int(m.sum()))
        return {'lag_centers': np.array(centers),
                'dcf': np.array(dcf_vals),
                'n_pairs': np.array(n_pairs)}


# =============================================================================
# BURSTS
# =============================================================================

class BurstDetector:
    """
    Episodic-burst detection via sigma-thresholding with hysteresis.

    A burst is a run of >= min_duration consecutive samples at least
    `threshold` robust sigmas (1.4826*MAD units) above the running
    baseline. Reports start/end times, peak, fluence (integral of the
    baseline-subtracted signal), and rise time.
    """

    def __init__(self, threshold: float = 5.0, min_duration: int = 3,
                 baseline_halfwindow: Optional[int] = None):
        self.threshold = threshold
        self.min_duration = max(min_duration, 1)
        self.baseline_halfwindow = baseline_halfwindow

    def _baseline(self, series: TimeSeries) -> np.ndarray:
        if self.baseline_halfwindow is None:
            med = np.median(series.values)
            return np.full_like(series.values, med)
        t, v = series.time, series.values
        base = np.empty_like(v)
        for i in range(v.size):
            m = np.abs(t - t[i]) <= self.baseline_halfwindow * \
                series.median_cadence
            base[i] = np.median(v[m]) if m.sum() else med_global(v)
        return base

    def detect(self, series: TimeSeries) -> Dict[str, Any]:
        base = self._baseline(series)
        resid = series.values - base
        mad = np.median(np.abs(resid - np.median(resid)))
        sigma = 1.4826 * mad if mad > 0 else resid.std(ddof=1)
        active = resid > self.threshold * sigma

        bursts = []
        i = 0
        while i < active.size:
            if active[i]:
                j = i
                while j + 1 < active.size and active[j + 1]:
                    j += 1
                if (j - i + 1) >= self.min_duration:
                    sl = slice(i, j + 1)
                    peak_i = i + int(np.argmax(resid[sl]))
                    fluence = float(np.trapezoid(resid[sl], series.time[sl]))
                    t_rise = series.time[peak_i] - series.time[i]
                    bursts.append({
                        'start': float(series.time[i]),
                        'end': float(series.time[j]),
                        'duration': float(series.time[j] - series.time[i]),
                        'peak_time': float(series.time[peak_i]),
                        'peak_value': float(series.values[peak_i]),
                        'peak_sigma': float(resid[peak_i] / sigma),
                        'fluence': fluence,
                        'rise_time': float(t_rise),
                        'n_samples': int(j - i + 1),
                    })
                i = j + 1
            else:
                i += 1
        return {
            'n_bursts': len(bursts),
            'bursts': bursts,
            'burst_rate': len(bursts) / series.duration
            if series.duration > 0 else 0.0,
            'threshold_sigma': self.threshold,
            'robust_sigma': float(sigma),
        }


def med_global(values: np.ndarray) -> float:
    return float(np.median(values))


# =============================================================================
# STRUCTURE FUNCTION
# =============================================================================

def compute_structure_function(time: np.ndarray, values: np.ndarray,
                               order: int = 2,
                               n_bins: int = 15) -> Dict[str, np.ndarray]:
    """
    Structure function SF(tau) = < |x(t+tau) - x(t)|^order > binned in
    lag. For a random walk SF_2 ~ tau^1; for white noise SF_2 ~ tau^0
    at lags beyond the correlation time.

    Lags are binned geometrically (standard practice: small lags are
    far better sampled than large lags).

    Returns dict with 'lag_centers', 'sf', 'sf_error' (standard error).
    """
    t = np.asarray(time, dtype=float)
    x = np.asarray(values, dtype=float)
    lag_matrix = t[None, :] - t[:, None]
    diff = np.abs(x[None, :] - x[:, None]) ** order
    iu = np.triu_indices(t.size, k=1)
    lags, diffs = lag_matrix[iu], diff[iu]
    lags = np.abs(lags)
    positive = lags[lags > 0]

    edges = np.geomspace(positive.min(), positive.max(), n_bins + 1)
    centers, sf, err = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (lags >= lo) & (lags < hi)
        if m.sum() >= 3:
            centers.append(np.sqrt(lo * hi))
            sf.append(float(diffs[m].mean()))
            err.append(float(diffs[m].std(ddof=1) / np.sqrt(m.sum())))
    return {'lag_centers': np.array(centers), 'sf': np.array(sf),
            'sf_error': np.array(err), 'order': order}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_power_spectrum(time: np.ndarray, values: np.ndarray,
                           errors: Optional[np.ndarray] = None,
                           method: str = "auto") -> PeriodogramResult:
    """Full periodogram analysis of a raw (t, x) series."""
    series = TimeSeries(time, values, errors)
    return PowerSpectrumAnalyzer().periodogram(series, method)


def detect_periodicity(time: np.ndarray, values: np.ndarray,
                       fap_threshold: float = 0.05) -> Dict[str, Any]:
    """
    Detect the best period with false-alarm significance.

    Returns dict with best period/power/FAP and the significance
    verdict at the given FAP threshold.
    """
    result = analyze_power_spectrum(time, values)
    return {
        'significant': bool(result.false_alarm_probability is not None
                            and result.false_alarm_probability
                            < fap_threshold),
        'best_period': result.best_period,
        'best_frequency': result.best_frequency,
        'best_power': result.best_power,
        'false_alarm_probability': result.false_alarm_probability,
        'signal_type': result.signal_type.value if result.signal_type
        else None,
        'spectral_slope': result.spectral_slope,
    }


def cross_correlate_series(t1, x1, t2=None, x2=None, max_lag: int = None):
    """
    Cross-correlate two series. If t2/x2 omitted, computes the
    autocorrelation of the first series.
    """
    a = TimeSeries(t1, x1)
    if x2 is None:
        b = a
    else:
        b = TimeSeries(t2, x2)
    return CrossCorrelationAnalyzer().ccf(a, b, max_lag)
