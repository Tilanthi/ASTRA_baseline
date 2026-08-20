#!/usr/bin/env python3
"""
MHD & Turbulence Analysis Tools for ASTRO-SWARM
================================================

Analysis tools for magnetohydrodynamic simulations and
turbulent ISM observations.

Capabilities:
1. Structure function analysis
2. Power spectrum computation
3. Velocity Channel Analysis (VCA)
4. Velocity Coordinate Spectrum (VCS)
5. Principal Component Analysis for spectral cubes
6. Davis-Chandrasekhar-Fermi magnetic field estimation
7. Histogram of Relative Orientations (HRO)
8. Turbulence statistics (Mach number, sonic scale)

Key References:
- Lazarian & Pogosyan 2000 (VCA/VCS)
- Heyer & Brunt 2004 (structure functions)
- Davis 1951, Chandrasekhar & Fermi 1953 (DCF)
- Soler et al. 2013 (HRO)
- Brunt & Heyer 2002 (PCA)

Author: Claude Code (ASTRO-SWARM)
Date: 2024-11
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from enum import Enum
from scipy.fft import fft, fft2, fftn, fftfreq, fftshift
from scipy.ndimage import gaussian_filter, sobel, uniform_filter
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr
from scipy.interpolate import interp1d
import warnings


# =============================================================================
# STRUCTURE FUNCTIONS
# =============================================================================

@dataclass
class StructureFunctionResult:
    """Result from structure function analysis"""
    lags: np.ndarray            # Spatial lags
    S_p: np.ndarray             # Structure function values
    order: int                  # Order p
    slope: float                # Power-law slope
    slope_err: float            # Slope uncertainty
    fit_range: Tuple[float, float]


class StructureFunctionAnalysis:
    """
    Spatial structure function analysis for turbulence characterization.

    The structure function of order p is:
    S_p(l) = <|v(x+l) - v(x)|^p>

    For Kolmogorov turbulence: S_2(l) ∝ l^(2/3)
    For Burgers turbulence: S_2(l) ∝ l
    """

    def __init__(self):
        pass

    def compute_1d(self, data: np.ndarray, order: int = 2,
                  max_lag: Optional[int] = None) -> StructureFunctionResult:
        """
        Compute 1D structure function.

        Parameters
        ----------
        data : np.ndarray
            1D data array
        order : int
            Structure function order
        max_lag : int, optional
            Maximum lag (default: N/4)

        Returns
        -------
        StructureFunctionResult
        """
        n = len(data)
        if max_lag is None:
            max_lag = n // 4

        lags = np.arange(1, max_lag + 1)
        S_p = np.zeros(len(lags))

        for i, lag in enumerate(lags):
            diff = np.abs(data[lag:] - data[:-lag])**order
            S_p[i] = np.mean(diff)

        # Fit power law
        slope, slope_err, fit_range = self._fit_power_law(lags, S_p)

        return StructureFunctionResult(
            lags=lags,
            S_p=S_p,
            order=order,
            slope=slope,
            slope_err=slope_err,
            fit_range=fit_range
        )

    def compute_2d(self, data: np.ndarray, order: int = 2,
                  max_lag: Optional[int] = None,
                  n_angles: int = 36) -> StructureFunctionResult:
        """
        Compute 2D structure function (azimuthally averaged).

        Parameters
        ----------
        data : np.ndarray
            2D data array
        order : int
            Structure function order
        max_lag : int, optional
            Maximum lag
        n_angles : int
            Number of angles for averaging

        Returns
        -------
        StructureFunctionResult
        """
        ny, nx = data.shape
        if max_lag is None:
            max_lag = min(nx, ny) // 4

        lags = np.arange(1, max_lag + 1)
        S_p = np.zeros(len(lags))

        # FIX(audit H11): the previous implementation stepped through n_angles
        # angles and truncated the displacement with dx = int(lag*cos(theta)),
        # dy = int(lag*sin(theta)). At lag = 1 that gives (0, 0) for 32 of the 36
        # default angles, so S_2(1) was the mean of 32 exact zeros and 4 real
        # values -- 11x too low (3.894e-4 vs a direct 4.494e-3), and the fitted
        # slope was biased (1.577 vs 1.513).
        # We now average over the *integer lattice* displacements that actually
        # fall in the annulus |d| in [lag-0.5, lag+0.5), which is the standard
        # radially-binned estimator, and weight by the number of pixel pairs.
        offsets_by_lag = self._annulus_offsets(lags, nx, ny, n_angles)

        for i, lag in enumerate(lags):
            total = 0.0
            count = 0
            for dx, dy in offsets_by_lag[i]:
                # overlapping region of data and data shifted by (dx, dy)
                y1 = slice(max(0, dy), ny + min(0, dy))
                y0 = slice(max(0, -dy), ny + min(0, -dy))
                x1 = slice(max(0, dx), nx + min(0, dx))
                x0 = slice(max(0, -dx), nx + min(0, -dx))
                diff = data[y1, x1] - data[y0, x0]
                if diff.size:
                    total += float(np.sum(np.abs(diff) ** order))
                    count += diff.size
            if count:
                S_p[i] = total / count

        # Fit power law
        slope, slope_err, fit_range = self._fit_power_law(lags, S_p)

        return StructureFunctionResult(
            lags=lags,
            S_p=S_p,
            order=order,
            slope=slope,
            slope_err=slope_err,
            fit_range=fit_range
        )

    @staticmethod
    def _annulus_offsets(lags: np.ndarray, nx: int, ny: int,
                        n_angles: int) -> List[List[Tuple[int, int]]]:
        """
        Integer lattice displacements in each lag annulus, |d| in [l-0.5, l+0.5).

        Only half of the plane is kept (dx > 0, or dx == 0 and dy > 0) because
        (dx, dy) and (-dx, -dy) sample exactly the same pixel pairs.

        `n_angles` is retained from the previous API and now acts as a cap: if an
        annulus contains more than `n_angles` distinct half-plane displacements,
        they are subsampled uniformly in position angle so the azimuthal average
        stays isotropic while the cost stays bounded. (Setting n_angles very
        small degrades the isotropy of the average; the default 36 is fine for
        the usual max_lag = N/4.)
        """
        max_lag = int(lags[-1])
        out: List[List[Tuple[int, int]]] = []
        # Pre-bin all candidate displacements once.
        buckets: Dict[int, List[Tuple[float, int, int]]] = {int(l): [] for l in lags}
        rng = range(-max_lag, max_lag + 1)
        for dy in rng:
            if abs(dy) >= ny:
                continue
            for dx in rng:
                if abs(dx) >= nx:
                    continue
                if dx < 0 or (dx == 0 and dy <= 0):
                    continue                      # keep one of each +/- pair
                r = np.hypot(dx, dy)
                lag_bin = int(np.floor(r + 0.5))   # annulus [l-0.5, l+0.5)
                if lag_bin in buckets:
                    buckets[lag_bin].append((float(np.arctan2(dy, dx)), dx, dy))
        for lag in lags:
            entries = sorted(buckets[int(lag)])
            if n_angles and len(entries) > n_angles:
                idx = np.linspace(0, len(entries) - 1, n_angles).round().astype(int)
                entries = [entries[j] for j in sorted(set(idx.tolist()))]
            out.append([(dx, dy) for _, dx, dy in entries])
        return out

    def velocity_structure_function(self, centroid_velocity: np.ndarray,
                                   pixel_scale: float,
                                   order: int = 2) -> StructureFunctionResult:
        """
        Compute structure function from centroid velocity map.

        Parameters
        ----------
        centroid_velocity : np.ndarray
            2D velocity centroid map (km/s)
        pixel_scale : float
            Pixel size (pc or arcsec)
        order : int
            Structure function order

        Returns
        -------
        StructureFunctionResult
        """
        result = self.compute_2d(centroid_velocity, order=order)

        # Convert lags to physical units
        result.lags = result.lags * pixel_scale

        return result

    def _fit_power_law(self, x: np.ndarray, y: np.ndarray,
                      fit_fraction: float = 0.5) -> Tuple[float, float, Tuple]:
        """Fit power law to structure function"""
        # Use middle portion for fit
        n = len(x)
        start = int(n * 0.1)
        end = int(n * fit_fraction)

        if end <= start:
            return 0.0, np.inf, (x[0], x[-1])

        log_x = np.log10(x[start:end])
        log_y = np.log10(y[start:end] + 1e-30)

        # Remove invalid values
        valid = np.isfinite(log_x) & np.isfinite(log_y) \
            & (y[start:end] > 0)
        if valid.sum() < 3:
            return 0.0, np.inf, (float(x[0]), float(x[-1]))

        lx, ly = log_x[valid], log_y[valid]
        coeffs, cov = np.polyfit(lx, ly, 1, cov=True)
        slope = float(coeffs[0])
        slope_err = float(np.sqrt(max(cov[0, 0], 0.0)))
        fit_range = (float(x[start]), float(x[end - 1]))

        return slope, slope_err, fit_range


# =============================================================================
# POWER SPECTRUM
# =============================================================================

class PowerSpectrumAnalysis:
    """
    Azimuthally averaged power spectrum of 2-D fields.

    P(k) = <|F(k)|^2> over annuli in k-space; for isotropic
    turbulence P(k) ~ k^gamma.
    """

    def __init__(self, pixel_scale: float = 1.0):
        """pixel_scale: size of one pixel (pc or arcsec)."""
        self.pixel_scale = pixel_scale

    def power_spectrum_2d(self, data: np.ndarray,
                          n_bins: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            (k, P(k)) with k in 1/pixel_scale units.
        """
        data = np.asarray(data, dtype=float)
        ny, nx = data.shape
        data = data - data.mean()

        fk = fftshift(fft2(data))
        power = np.abs(fk) ** 2

        kx = fftshift(fftfreq(nx, d=self.pixel_scale))
        ky = fftshift(fftfreq(ny, d=self.pixel_scale))
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX ** 2 + KY ** 2)

        kmax = K.max()
        edges = np.linspace(0.0, kmax, n_bins + 1)
        kk, pp = [], []
        for i in range(1, n_bins):
            sel = (K >= edges[i]) & (K < edges[i + 1])
            if sel.sum() < 4:
                continue
            kk.append(K[sel].mean())
            pp.append(power[sel].mean())
        return np.array(kk), np.array(pp)

    @staticmethod
    def fit_slope(k: np.ndarray, p: np.ndarray,
                  k_range: Optional[Tuple[float, float]] = None
                  ) -> Tuple[float, float]:
        """Least-squares slope of log P vs log k and its error."""
        if k_range is not None:
            sel = (k >= k_range[0]) & (k <= k_range[1])
        else:
            sel = np.ones(len(k), dtype=bool)
        sel &= (k > 0) & (p > 0)
        if sel.sum() < 3:
            return 0.0, np.inf
        lx, ly = np.log10(k[sel]), np.log10(p[sel])
        coeffs, cov = np.polyfit(lx, ly, 1, cov=True)
        return float(coeffs[0]), float(np.sqrt(cov[0, 0]))


# =============================================================================
# VELOCITY FIELD ANALYSIS
# =============================================================================

class VelocityAnalysis:
    """
    Moment analysis of position-position-velocity cubes.

    Centroid velocity v_c = sum(d v)/sum(d) and the intensity-weighted
    line-of-sight velocity dispersion

        sigma_los^2 = sum(d (v - v_c)^2)/sum(d)

    both computed channel-by-channel with the velocity axis last.
    """

    def __init__(self, velocity_axis: np.ndarray):
        """velocity_axis: channel velocities (km/s), shape (n_ch,)."""
        self.v = np.asarray(velocity_axis, dtype=float)

    def centroid(self, cube: np.ndarray) -> np.ndarray:
        """First moment (km/s)."""
        cube = np.asarray(cube, dtype=float)
        w = cube.sum(axis=-1)
        return np.divide((cube * self.v).sum(axis=-1), w,
                         out=np.zeros_like(w), where=w > 0)

    def dispersion(self, cube: np.ndarray) -> np.ndarray:
        """Second-moment velocity dispersion (km/s)."""
        cube = np.asarray(cube, dtype=float)
        w = cube.sum(axis=-1)
        m1 = self.centroid(cube)
        m2 = np.divide((cube * self.v ** 2).sum(axis=-1), w,
                       out=np.zeros_like(w), where=w > 0)
        var = np.clip(m2 - m1 ** 2, 0.0, None)
        return np.sqrt(var)

    def stats(self, cube: np.ndarray) -> Dict[str, float]:
        """Global statistics over the whole cube."""
        cube = np.asarray(cube, dtype=float)
        prof = cube.sum(axis=(0, 1))
        norm = prof.sum()
        m1 = float((prof * self.v).sum() / norm)
        m2 = float((prof * self.v ** 2).sum() / norm)
        return {'mean_velocity_kms': m1,
                'sigma_los_kms': float(np.sqrt(max(m2 - m1 ** 2, 0.0))),
                'peak_temperature': float(cube.max()),
                'total_emission': float(norm)}


# =============================================================================
# SPECTRAL PCA
# =============================================================================

class SpectralPCA:
    """
    Principal component analysis of a PPV cube (Heyer & Brunt 2002).

    The cube is reshaped to (n_pixels, n_channels); SVD of the
    channel-space matrix gives the "principal spectra" (eigenvectors)
    and the score maps (projections) of their spatial distribution.
    """

    def __init__(self, n_components: int = 5):
        self.n_components = n_components

    def fit(self, cube: np.ndarray) -> Dict[str, Any]:
        """
        Args:
            cube: (nx, ny, nch) PPV cube

        Returns:
            dict with 'explained_variance_ratio', 'principal_spectra'
            (n_components, nch), 'score_maps' (n_components, nx, ny)
        """
        cube = np.asarray(cube, dtype=float)
        nx, ny, nch = cube.shape
        # FIX(audit H12): centre each spectral channel on its own mean, as PCA
        # requires. Subtracting the single global scalar `cube.mean()` leaves the
        # mean spectrum in the data, so PC1 was simply the mean spectrum:
        # on a cube = mean spectrum + 1% noise the old code reported
        # explained_variance_ratio = [0.99915, 5e-5, 5e-5]; correctly centred it
        # is [0.0558, 0.0517, 0.0494]. `total_variance` below was already
        # computed on the properly centred matrix, so the two outputs used to be
        # mutually inconsistent.
        X = cube.reshape(nx * ny, nch)
        X = X - X.mean(axis=0, keepdims=True)

        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        eig = S ** 2
        ratio = eig / eig.sum()
        n = min(self.n_components, len(ratio))

        spectra = Vt[:n]                      # (n, nch)
        scores = (X @ Vt[:n].T).reshape(nx, ny, n)
        return {
            'explained_variance_ratio': ratio[:n],
            'principal_spectra': spectra,
            'score_maps': np.moveaxis(scores, -1, 0),
            'total_variance': float(X.var(axis=0).sum()),
        }


# =============================================================================
# DAVIS-CHANDRASEKHAR-FERMI METHOD
# =============================================================================

class DavisChandrasekharFermi:
    """
    Plane-of-sky magnetic field strength from polarization and
    spectroscopic data:

        B_pos = f * sqrt(4 pi rho) * sigma_los / sigma_theta

    where sigma_theta is the polarization-angle dispersion (radians),
    sigma_los the line-of-sight velocity dispersion, rho the mass
    density and f ~ 0.5 the correction factor for isotropic turbulence
    (Ostriker, Stone & Gammie 2001).
    """

    def __init__(self, correction_factor: float = 0.5):
        self.f = correction_factor

    def field_strength(self, sigma_theta_deg: float, sigma_los_kms: float,
                       volume_density_cm3: float,
                       mean_molecular_weight: float = 2.33
                       ) -> Dict[str, float]:
        """
        Args:
            sigma_theta_deg: polarization angle dispersion (degrees)
            sigma_los_kms: LOS velocity dispersion (km/s)
            volume_density_cm3: total particle number density (cm^-3)

        Returns:
            dict with B_pos (G and microgauss), rho, sigma_theta (rad)
        """
        rho = volume_density_cm3 * mean_molecular_weight * 1.673e-24
        sigma_theta = np.radians(sigma_theta_deg)
        sigma_los = sigma_los_kms * 1e5
        b = self.f * np.sqrt(4.0 * np.pi * rho) * sigma_los / sigma_theta
        return {'B_pos_G': float(b), 'B_pos_uG': float(b * 1e6),
                'rho_g_cm3': float(rho),
                'sigma_theta_rad': float(sigma_theta)}

    @staticmethod
    def polarization_dispersion(angles_deg: np.ndarray) -> float:
        """
        Dispersion of angles accounting for the 180-degree ambiguity:
        the circular dispersion of doubled angles, halved.
        """
        a = 2.0 * np.radians(np.asarray(angles_deg, dtype=float))
        x, y = np.cos(a).mean(), np.sin(a).mean()
        r = np.hypot(x, y)                    # resultant length
        sigma2 = np.sqrt(max(-2.0 * np.log(max(r, 1e-12)), 0.0))
        return float(np.degrees(sigma2 / 2.0))


# =============================================================================
# HISTOGRAM OF RELATIVE ORIENTATIONS
# =============================================================================

class HistogramRelativeOrientations:
    """
    Relative orientation between intensity gradients and the magnetic
    field projected on the sky (Soler et al. 2013).

    Computes the angle between the image gradient and a given field
    direction and histograms it over [0, 90]; the alignment ratio
    contrasts the populations parallel and perpendicular to the field.
    """

    def __init__(self, field_angle_deg: float, n_bins: int = 18):
        """
        Args:
            field_angle_deg: position angle of the plane-of-sky B
                field (degrees east of north)
            n_bins: histogram bins over [0, 90]
        """
        self.field_angle_deg = float(field_angle_deg)
        self.n_bins = n_bins

    def analyze(self, image: np.ndarray,
                noise_threshold: float = 0.2) -> Dict[str, Any]:
        """
        Args:
            image: 2-D intensity map
            noise_threshold: fraction of the max gradient below which
                gradients are ignored

        Returns:
            dict with 'histogram', 'bin_centers', 'alignment_ratio',
            'mean_relative_angle', 'n_pixels'
        """
        image = np.asarray(image, dtype=float)
        gy, gx = np.gradient(image)
        mag = np.hypot(gx, gy)

        # gradient direction in sky convention (east of north):
        # north = -y, east = +x  ->  PA = atan2(gx, -gy)
        grad_pa = np.degrees(np.arctan2(gx, -gy))

        rel = (grad_pa - self.field_angle_deg) % 180.0
        rel = np.minimum(rel, 180.0 - rel)        # fold to [0, 90]

        valid = mag > noise_threshold * np.nanmax(mag)
        rel_v = rel[valid]

        counts, edges = np.histogram(rel_v, bins=self.n_bins,
                                     range=(0.0, 90.0))
        centers = 0.5 * (edges[:-1] + edges[1:])

        par = counts[centers <= 10.0].sum()
        perp = counts[centers >= 80.0].sum()
        denom = par + perp
        ratio = float((par - perp) / denom) if denom > 0 else 0.0

        return {'histogram': counts, 'bin_centers': centers,
                'alignment_ratio': ratio,
                'mean_relative_angle': float(rel_v.mean()),
                'n_pixels': int(valid.sum())}


# =============================================================================
# TURBULENCE STATISTICS
# =============================================================================

class TurbulenceStatistics:
    """
    Derived turbulence quantities: sonic Mach number, dissipation
    rate per unit mass and the sonic scale.

    The sonic scale follows from the linewidth-size law
    sigma(l) = sigma_L (l/L)^p: setting sigma(l_s) = c_s gives

        l_s = L * M^{-1/p},   M = sigma_L / c_s.
    """

    K_B = 1.381e-16   # erg/K
    M_H = 1.673e-24   # g

    def __init__(self, temperature_k: float = 10.0,
                 mean_molecular_weight: float = 2.33):
        self.T = float(temperature_k)
        self.mu = float(mean_molecular_weight)

    def sound_speed(self) -> float:
        """Isothermal sound speed (km/s)."""
        return float(np.sqrt(self.K_B * self.T / (self.mu * self.M_H))
                     / 1e5)

    def mach_number(self, sigma_kms: float) -> float:
        """Sonic Mach number."""
        return float(sigma_kms / self.sound_speed())

    def dissipation_rate(self, sigma_kms: float, scale_pc: float) -> float:
        """
        Turbulent energy dissipation rate per unit mass,
        epsilon ~ sigma^3 / L (erg/g/s).
        """
        return float((sigma_kms * 1e5) ** 3 / (scale_pc * 3.086e18))

    def reynolds_number(self, sigma_kms: float, scale_pc: float,
                        kinematic_viscosity_cm2_s: Optional[float] = None
                        ) -> Optional[float]:
        """
        Re = sigma L / nu. The ISM kinematic viscosity is uncertain
        (molecular-viscosity estimates span orders of magnitude), so
        it must be supplied explicitly; returns None otherwise.
        """
        if kinematic_viscosity_cm2_s is None:
            return None
        return float(sigma_kms * 1e5 * scale_pc * 3.086e18
                     / kinematic_viscosity_cm2_s)

    def sonic_scale_pc(self, sigma_kms: float, driving_scale_pc: float,
                       linewidth_slope: float = 0.5) -> float:
        """Scale at which turbulence becomes subsonic (pc)."""
        mach = self.mach_number(sigma_kms)
        if mach <= 1.0:
            return driving_scale_pc
        return float(driving_scale_pc * mach ** (-1.0 / linewidth_slope))

    def summary(self, sigma_kms: float, driving_scale_pc: float,
                linewidth_slope: float = 0.5) -> Dict[str, float]:
        return {
            'sound_speed_kms': self.sound_speed(),
            'mach_number': self.mach_number(sigma_kms),
            'dissipation_rate_erg_g_s':
                self.dissipation_rate(sigma_kms, driving_scale_pc),
            'sonic_scale_pc': self.sonic_scale_pc(
                sigma_kms, driving_scale_pc, linewidth_slope),
        }
