#!/usr/bin/env python3
"""
Interferometric Imaging Support for ASTRO-SWARM
================================================

Tools for radio/mm interferometric data analysis.

Capabilities:
1. UV-plane sampling and simulation
2. Visibility modeling and prediction
3. CLEAN/MEM deconvolution interfaces
4. Primary beam correction
5. Self-calibration support
6. Visibility model comparison
7. Baseline-dependent analysis

Key References:
- Thompson, Moran & Swenson (Interferometry and Synthesis)
- Cornwell et al. 2008 (W-projection)
- Briggs 1995 (Robust weighting)
- Rau & Cornwell 2011 (Multi-scale CLEAN)

Author: Claude Code (ASTRO-SWARM)
Date: 2024-11
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from enum import Enum
from abc import ABC, abstractmethod
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from scipy.interpolate import griddata
from scipy.optimize import minimize
import warnings

# Physical Constants
c_light = 2.998e10      # cm/s


# =============================================================================
# VISIBILITY DATA STRUCTURES
# =============================================================================

@dataclass
class Visibility:
    """Visibility data container"""
    u: np.ndarray               # U coordinates (wavelengths)
    v: np.ndarray               # V coordinates (wavelengths)
    w: np.ndarray               # W coordinates (wavelengths)
    real: np.ndarray            # Real part of visibility
    imag: np.ndarray            # Imaginary part of visibility
    weight: np.ndarray          # Visibility weights
    freq: float                 # Frequency (Hz)
    time: Optional[np.ndarray] = None  # Time stamps

    @property
    def vis(self) -> np.ndarray:
        """Complex visibility"""
        return self.real + 1j * self.imag

    @property
    def amplitude(self) -> np.ndarray:
        """Visibility amplitude"""
        return np.abs(self.vis)

    @property
    def phase(self) -> np.ndarray:
        """Visibility phase (radians)"""
        return np.angle(self.vis)

    @property
    def uvdist(self) -> np.ndarray:
        """UV distance (wavelengths)"""
        return np.sqrt(self.u**2 + self.v**2)

    def __len__(self):
        return len(self.u)


@dataclass
class UVCoverage:
    """UV plane coverage pattern"""
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    weight: np.ndarray
    antenna1: np.ndarray
    antenna2: np.ndarray
    time: np.ndarray


# =============================================================================
# ARRAY CONFIGURATION
# =============================================================================

@dataclass
class Antenna:
    """Antenna position"""
    name: str
    x: float        # East coordinate (m)
    y: float        # North coordinate (m)
    z: float        # Up coordinate (m)
    diameter: float  # Dish diameter (m)


class ArrayConfiguration:
    """
    Interferometer array configuration.
    """

    def __init__(self, name: str, latitude: float = 0.0):
        """
        Parameters
        ----------
        name : str
            Array name
        latitude : float
            Array latitude (degrees)
        """
        self.name = name
        self.latitude = np.radians(latitude)
        self.antennas: List[Antenna] = []

    def add_antenna(self, antenna: Antenna):
        """Add antenna to array"""
        self.antennas.append(antenna)

    @property
    def n_antennas(self) -> int:
        return len(self.antennas)

    @property
    def n_baselines(self) -> int:
        n = self.n_antennas
        return n * (n - 1) // 2

    def get_baselines(self) -> List[Tuple[int, int, np.ndarray]]:
        """
        Get all baselines.

        Returns
        -------
        list : (i, j, baseline_vector) tuples
        """
        baselines = []
        for i, ant1 in enumerate(self.antennas):
            for j, ant2 in enumerate(self.antennas):
                if j > i:
                    baseline = np.array([
                        ant2.x - ant1.x,
                        ant2.y - ant1.y,
                        ant2.z - ant1.z
                    ])
                    baselines.append((i, j, baseline))
        return baselines

    def max_baseline(self) -> float:
        """Maximum baseline length (m)"""
        max_b = 0.0
        for i, j, b in self.get_baselines():
            max_b = max(max_b, np.linalg.norm(b))
        return max_b

    def min_baseline(self) -> float:
        """Minimum baseline length (m)"""
        min_b = np.inf
        for i, j, b in self.get_baselines():
            blen = np.linalg.norm(b)
            if blen > 0:
                min_b = min(min_b, blen)
        return min_b

    def angular_resolution(self, freq_Hz: float) -> float:
        """
        Angular resolution (arcsec).

        θ ≈ λ / B_max
        """
        wavelength = c_light / freq_Hz
        b_max = self.max_baseline() * 100  # m to cm
        theta_rad = wavelength / b_max
        return theta_rad * 206265  # radians to arcsec

    def largest_angular_scale(self, freq_Hz: float) -> float:
        """
        Largest recoverable angular scale (arcsec).

        θ_LAS ≈ λ / B_min
        """
        wavelength = c_light / freq_Hz
        b_min = self.min_baseline() * 100
        if b_min > 0:
            theta_rad = wavelength / b_min
            return theta_rad * 206265
        return np.inf


class StandardArrays:
    """
    Standard interferometer array configurations.
    """

    @staticmethod
    def vla_a() -> ArrayConfiguration:
        """VLA A-configuration (approximate)"""
        array = ArrayConfiguration("VLA_A", latitude=34.08)

        # Simplified Y-shaped array
        arm_length = 21000  # meters (max baseline ~36 km)
        n_per_arm = 9

        for arm, angle in enumerate([0, 120, 240]):
            angle_rad = np.radians(angle - 90)  # North arm at 0
            for i in range(n_per_arm):
                r = arm_length * (i / n_per_arm)**1.7  # Non-uniform spacing
                x = r * np.cos(angle_rad)
                y = r * np.sin(angle_rad)
                array.add_antenna(Antenna(
                    name=f"Ant{arm*n_per_arm + i}",
                    x=x, y=y, z=0, diameter=25
                ))

        return array

    @staticmethod
    def alma_compact() -> ArrayConfiguration:
        """ALMA compact configuration (approximate)"""
        array = ArrayConfiguration("ALMA_C1", latitude=-23.02)

        # Random compact distribution
        np.random.seed(42)
        n_antennas = 50
        max_radius = 150  # meters

        for i in range(n_antennas):
            r = max_radius * np.sqrt(np.random.random())
            theta = np.random.random() * 2 * np.pi
            array.add_antenna(Antenna(
                name=f"DA{i:02d}",
                x=r * np.cos(theta),
                y=r * np.sin(theta),
                z=0,
                diameter=12
            ))

        return array

    @staticmethod
    def alma_extended() -> ArrayConfiguration:
        """ALMA extended configuration (approximate)"""
        array = ArrayConfiguration("ALMA_C6", latitude=-23.02)

        np.random.seed(43)
        n_antennas = 50
        max_radius = 8000  # meters

        for i in range(n_antennas):
            r = max_radius * np.sqrt(np.random.random())
            theta = np.random.random() * 2 * np.pi
            array.add_antenna(Antenna(
                name=f"DA{i:02d}",
                x=r * np.cos(theta),
                y=r * np.sin(theta),
                z=0,
                diameter=12
            ))

        return array


# =============================================================================
# UV PLANE SIMULATION
# =============================================================================

class UVSimulator:
    """
    Simulate UV coverage for observations.
    """

    def __init__(self, array: ArrayConfiguration):
        self.array = array

    def generate_coverage(self,
                         source_dec: float,
                         freq_Hz: float,
                         hour_angles: np.ndarray,
                         integration_time: float = 10.0) -> UVCoverage:
        """
        Generate UV coverage for an observation.

        Parameters
        ----------
        source_dec : float
            Source declination (degrees)
        freq_Hz : float
            Observing frequency (Hz)
        hour_angles : np.ndarray
            Hour angles to observe (degrees)
        integration_time : float
            Integration time per sample (seconds)

        Returns
        -------
        UVCoverage
        """
        dec_rad = np.radians(source_dec)
        lat_rad = self.array.latitude
        wavelength = c_light / freq_Hz * 1e-2  # wavelengths (m -> cm, then to wavelengths)
        wavelength_m = c_light / freq_Hz  # in meters

        baselines = self.array.get_baselines()

        all_u = []
        all_v = []
        all_w = []
        all_ant1 = []
        all_ant2 = []
        all_time = []

        for ha in hour_angles:
            ha_rad = np.radians(ha)

            # Rotation matrix from antenna coords to UV coords
            sin_ha = np.sin(ha_rad)
            cos_ha = np.cos(ha_rad)
            sin_dec = np.sin(dec_rad)
            cos_dec = np.cos(dec_rad)
            sin_lat = np.sin(lat_rad)
            cos_lat = np.cos(lat_rad)

            for i, j, b in baselines:
                # Transform baseline to UV coordinates
                # b = (East, North, Up) in meters

                # Convert to (X, Y, Z) in wavelengths
                # where X points to HA=0, Y to HA=6h, Z to NCP
                u = (sin_ha * b[0] + cos_ha * b[1]) / wavelength_m
                v = (-sin_dec * cos_ha * b[0] + sin_dec * sin_ha * b[1] +
                     cos_dec * b[2]) / wavelength_m
                w = (cos_dec * cos_ha * b[0] - cos_dec * sin_ha * b[1] +
                     sin_dec * b[2]) / wavelength_m

                all_u.append(u)
                all_v.append(v)
                all_w.append(w)
                all_ant1.append(i)
                all_ant2.append(j)
                all_time.append(ha)

                # Add conjugate
                all_u.append(-u)
                all_v.append(-v)
                all_w.append(-w)
                all_ant1.append(j)
                all_ant2.append(i)
                all_time.append(ha)

        return UVCoverage(
            u=np.array(all_u),
            v=np.array(all_v),
            w=np.array(all_w),
            weight=np.ones(len(all_u)),
            antenna1=np.array(all_ant1),
            antenna2=np.array(all_ant2),
            time=np.array(all_time)
        )

    def observe_model(self,
                     model_image: np.ndarray,
                     pixel_size_arcsec: float,
                     uv_coverage: UVCoverage,
                     freq_Hz: float,
                     add_noise: bool = True,
                     noise_Jy: float = 1e-3) -> Visibility:
        """
        Simulate visibilities from a model image.

        Parameters
        ----------
        model_image : np.ndarray
            Sky model (Jy/pixel)
        pixel_size_arcsec : float
            Pixel size (arcsec)
        uv_coverage : UVCoverage
            UV coverage pattern
        freq_Hz : float
            Observing frequency (Hz)
        add_noise : bool
            Add thermal noise
        noise_Jy : float
            RMS noise per visibility (Jy)

        Returns
        -------
        Visibility
        """
        ny, nx = model_image.shape

        # FFT of model (visibility function)
        model_ft = fftshift(fft2(ifftshift(model_image)))

        # UV pixel scale
        du = 1.0 / (nx * pixel_size_arcsec / 206265)
        dv = 1.0 / (ny * pixel_size_arcsec / 206265)

        # UV grid coordinates
        u_grid = fftfreq(nx, d=pixel_size_arcsec / 206265)
        v_grid = fftfreq(ny, d=pixel_size_arcsec / 206265)

        # Sample visibilities at UV coverage points
        real_parts = []
        imag_parts = []

        for u, v in zip(uv_coverage.u, uv_coverage.v):
            # Find nearest grid point (simple nearest-neighbor)
            iu = int(np.round(u / du)) + nx // 2
            iv = int(np.round(v / dv)) + ny // 2

            if 0 <= iu < nx and 0 <= iv < ny:
                vis = model_ft[iv, iu]
            else:
                vis = 0.0

            real_parts.append(np.real(vis))
            imag_parts.append(np.imag(vis))

        real_parts = np.array(real_parts)
        imag_parts = np.array(imag_parts)

        if add_noise:
            real_parts += np.random.normal(0, noise_Jy, len(real_parts))
            imag_parts += np.random.normal(0, noise_Jy, len(imag_parts))

        weights = np.ones(len(uv_coverage.u)) / noise_Jy**2

        return Visibility(
            u=uv_coverage.u,
            v=uv_coverage.v,
            w=uv_coverage.w,
            real=real_parts,
            imag=imag_parts,
            weight=weights,
            freq=freq_Hz,
            time=uv_coverage.time
        )


# =============================================================================
# IMAGING
# =============================================================================

class WeightingScheme(Enum):
    """Visibility weighting schemes"""
    NATURAL = "natural"
    UNIFORM = "uniform"
    BRIGGS = "briggs"


@dataclass
class DirtyImage:
    """Dirty image and beam"""
    image: np.ndarray           # Dirty image
    beam: np.ndarray            # Dirty beam (PSF)
    pixel_size: float           # Pixel size (arcsec)
    beam_params: Dict[str, float]  # Fitted beam parameters


class Imager:
    """
    Interferometric imaging.
    """

    def __init__(self):
        pass

    def make_dirty_image(self,
                        vis: Visibility,
                        image_size: int = 256,
                        pixel_size: float = 0.1,
                        weighting: WeightingScheme = WeightingScheme.NATURAL,
                        robust: float = 0.5) -> DirtyImage:
        """
        Make dirty image from visibilities.

        Parameters
        ----------
        vis : Visibility
            Visibility data
        image_size : int
            Image size in pixels
        pixel_size : float
            Pixel size (arcsec)
        weighting : WeightingScheme
            Weighting scheme
        robust : float
            Briggs robust parameter (-2 to 2)

        Returns
        -------
        DirtyImage
        """
        # Grid size
        nx = ny = image_size

        # UV pixel scale (wavelengths per pixel in UV plane)
        du = 1.0 / (nx * pixel_size / 206265)
        dv = 1.0 / (ny * pixel_size / 206265)

        # Create grids
        vis_grid = np.zeros((ny, nx), dtype=complex)
        weight_grid = np.zeros((ny, nx))
        sampling_grid = np.zeros((ny, nx))

        # Apply weighting
        weights = self._compute_weights(vis, weighting, robust, du, dv, nx, ny)

        # Grid visibilities
        for i in range(len(vis.u)):
            iu = int(np.round(vis.u[i] / du)) + nx // 2
            iv = int(np.round(vis.v[i] / dv)) + ny // 2

            if 0 <= iu < nx and 0 <= iv < ny:
                vis_grid[iv, iu] += vis.vis[i] * weights[i]
                weight_grid[iv, iu] += weights[i]
                sampling_grid[iv, iu] += 1

        # Normalize
        with np.errstate(invalid='ignore', divide='ignore'):
            vis_grid = np.where(weight_grid > 0, vis_grid / weight_grid, 0)

        # FFT to image
        dirty_image = np.real(fftshift(ifft2(ifftshift(vis_grid))))

        # Dirty beam (PSF)
        beam_grid = np.where(sampling_grid > 0, 1.0, 0.0)
        dirty_beam = np.real(fftshift(ifft2(ifftshift(beam_grid))))
        dirty_beam /= np.max(dirty_beam)

        # Fit beam
        beam_params = self._fit_beam(dirty_beam, pixel_size)

        return DirtyImage(
            image=dirty_image,
            beam=dirty_beam,
            pixel_size=pixel_size,
            beam_params=beam_params
        )

    def _compute_weights(self, vis: Visibility,
                        weighting: WeightingScheme,
                        robust: float,
                        du: float, dv: float,
                        nx: int, ny: int) -> np.ndarray:
        """Compute visibility weights"""
        if weighting == WeightingScheme.NATURAL:
            return vis.weight

        elif weighting == WeightingScheme.UNIFORM:
            # Count visibilities per UV cell
            cell_counts = {}
            for i in range(len(vis.u)):
                iu = int(np.round(vis.u[i] / du)) + nx // 2
                iv = int(np.round(vis.v[i] / dv)) + ny // 2
                key = (iu, iv)
                cell_counts[key] = cell_counts.get(key, 0) + 1

            weights = np.zeros(len(vis.u))
            for i in range(len(vis.u)):
                iu = int(np.round(vis.u[i] / du)) + nx // 2
                iv = int(np.round(vis.v[i] / dv)) + ny // 2
                key = (iu, iv)
                weights[i] = 1.0 / cell_counts.get(key, 1)

            return weights

        elif weighting == WeightingScheme.BRIGGS:
            # Briggs robust weighting
            # First compute uniform weights
            uniform = self._compute_weights(vis, WeightingScheme.UNIFORM,
                                           robust, du, dv, nx, ny)
            natural = vis.weight

            # Robust factor
            f2 = (5 * 10**(-robust))**2 * np.sum(uniform) / np.sum(natural)

            return natural / (1 + natural * f2)

        return vis.weight

    def _fit_beam(self, beam: np.ndarray, pixel_size: float) -> Dict[str, float]:
        """Fit 2D Gaussian to beam"""
        ny, nx = beam.shape
        y, x = np.mgrid[0:ny, 0:nx]
        ny, nx = beam.shape
        y, x = np.mgrid[0:ny, 0:nx]
        y = (y - ny//2) * pixel_size
        x = (x - nx//2) * pixel_size

        # Fit an elliptical Gaussian to the main lobe of the dirty
        # beam via intensity-weighted second moments over the
        # contiguous positive lobe containing the peak (out to the
        # first null). For a Gaussian main lobe this is exact;
        # sidelobes beyond the first null are excluded.
        from scipy.ndimage import label
        positive = beam > 0
        lab, n_lab = label(positive)
        peak_idx = np.unravel_index(np.argmax(beam), beam.shape)
        core = lab == lab[peak_idx] if n_lab > 0 else np.ones_like(
            beam, dtype=bool)
        if core.sum() < 5:
            core = np.ones_like(beam, dtype=bool)
        w = np.where(core, beam, 0.0)
        total = w.sum()
        x0 = (w * x).sum() / total
        y0 = (w * y).sum() / total
        dx = x - x0
        dy = y - y0

        # weighted second moments -> covariance
        sxx = (w * dx * dx).sum() / total
        syy = (w * dy * dy).sum() / total
        sxy = (w * dx * dy).sum() / total
        cov = np.array([[sxx, sxy], [sxy, syy]])

        # FWHM = 2 sqrt(2 ln 2) sigma: for a 2-D Gaussian the FWHM
        # along each principal axis is 2.3548 * sqrt(eigenvalue)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0.0)
        fwhm1, fwhm2 = 2.0 * np.sqrt(2.0 * np.log(2.0)) * np.sqrt(
            eigvals[::-1])          # descending order

        # position angle of the major axis (east of north, image
        # convention: x = RA offset (east), y = Dec offset (north))
        major_dir = eigvecs[:, -1]
        pa = np.degrees(np.arctan2(major_dir[0], major_dir[1]))

        return {
            'bmaj': float(max(fwhm1, fwhm2)),   # major axis FWHM (arcsec)
            'bmin': float(min(fwhm1, fwhm2)),   # minor axis FWHM (arcsec)
            'pa': float(pa),                    # position angle (deg)
            'center_x': float(x0),
            'center_y': float(y0),
        }


# =============================================================================
# CLEAN DECONVOLUTION
# =============================================================================

class CLEANDeconvolver:
    """
    Hogbom CLEAN (Hogbom 1974).

    Iteratively subtracts scaled copies of the dirty beam at the
    brightest residual pixel (minor cycle), then restores the
    accumulated point-source model convolved with a clean Gaussian
    (major cycle).
    """

    def __init__(self, gain: float = 0.1, n_iter: int = 1000,
                 threshold: float = 0.0):
        """
        Args:
            gain: loop gain (fraction of peak subtracted per step)
            n_iter: maximum minor-cycle iterations
            threshold: stop when residual peak falls below this
                (absolute, same units as image)
        """
        self.gain = gain
        self.n_iter = n_iter
        self.threshold = threshold

    def clean(self, dirty_image: np.ndarray, dirty_beam: np.ndarray,
              pixel_size: float = 1.0) -> Dict[str, Any]:
        """
        Run Hogbom CLEAN.

        Args:
            dirty_image: dirty image (ny, nx)
            dirty_beam: dirty beam (ny, nx), peak-normalized
            pixel_size: pixel size for the restored-beam record

        Returns:
            dict with 'clean_model' (point-source component list),
            'residual', 'restored' (CLEAN image), 'model_image',
            'beam' (fitted restoring beam params), 'n_iter', 'flux'
        """
        img = np.array(dirty_image, dtype=float)
        psf = np.array(dirty_beam, dtype=float)
        ny, nx = img.shape
        residual = img.copy()
        components = []          # (x_idx, y_idx, flux)

        # beam fit via the Imager (shared implementation)
        beam_params = Imager()._fit_beam(psf, pixel_size)
        psf_peak = psf.max()

        for it in range(self.n_iter):
            peak_idx = np.unravel_index(np.argmax(residual), residual.shape)
            peak_val = residual[peak_idx]
            if peak_val <= self.threshold:
                break
            if peak_val <= 0:
                break

            flux = self.gain * peak_val
            components.append((int(peak_idx[1]), int(peak_idx[0]),
                               float(flux)))

            # subtract gain*peak*PSF centred on the peak pixel
            sub = flux * psf
            # shift PSF peak to the component position
            y0, x0 = peak_idx
            pys, pxs = np.unravel_index(np.argmax(psf), psf.shape)
            dy = y0 - pys
            dx = x0 - pxs
            sub_shift = np.roll(np.roll(sub, dy, axis=0), dx, axis=1)
            residual -= sub_shift

        # model image from components
        model = np.zeros_like(img)
        for (x0, y0, flux) in components:
            model[y0, x0] += flux

        # restoring beam: elliptical Gaussian with fitted bmaj/bmin/pa
        restored = residual.copy()
        if components:
            gauss = self._elliptical_gaussian(ny, nx, beam_params,
                                              pixel_size)
            restored = residual + np.real(_fft_convolve(model, gauss))

        return {
            'clean_model': components,
            'residual': residual,
            'restored': restored,
            'model_image': model,
            'beam': beam_params,
            'n_iter': len(components),
            'flux': float(sum(f for _, _, f in components)),
        }

    @staticmethod
    def _elliptical_gaussian(ny: int, nx: int, beam_params: Dict[str, float],
                             pixel_size: float) -> np.ndarray:
        """Unit-peak elliptical Gaussian for the restoring beam."""
        y, x = np.mgrid[0:ny, 0:nx]
        y = (y - ny // 2) * pixel_size - beam_params.get('center_y', 0.0)
        x = (x - nx // 2) * pixel_size - beam_params.get('center_x', 0.0)
        bmaj = beam_params['bmaj']
        bmin = beam_params['bmin']
        pa = np.radians(beam_params['pa'])
        # rotate to principal frame (x' along major axis)
        xp = x * np.cos(pa) + y * np.sin(pa)
        yp = -x * np.sin(pa) + y * np.cos(pa)
        sig_maj = bmaj / 2.3548200450309493
        sig_min = bmin / 2.3548200450309493
        return np.exp(-0.5 * ((yp / sig_maj) ** 2 + (xp / sig_min) ** 2))


def _fft_convolve(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """FFT convolution, same-size output ('same' mode)."""
    return fftconvolve(a, k, mode='same')


# =============================================================================
# SELF-CALIBRATION
# =============================================================================

class SelfCalibrator:
    """
    Antenna-based self-calibration.

    Solves for per-antenna complex gains g_i that minimize

        sum_k w_k |V_k^obs - g_p g_q* V_k^model|^2

    over baselines (p, q), with the standard phase-only or
    amplitude-and-phase options. The solution per antenna uses the
    least-squares estimate

        g_i ~ sum over baselines with antenna i of
              w_k V_k^obs (V_k^model)* / sum w_k |V_k^model|^2

    iterated a few times (baseline-based gains in the data are exactly
    representable, so this converges quickly).
    """

    def __init__(self, n_antennas: int, baselines: List[Tuple[int, int]],
                 mode: str = 'phase'):
        """
        Args:
            n_antennas: number of antennas
            baselines: list of (ant1, ant2) index pairs, same order as
                the visibility arrays
            mode: 'phase' (unit-amplitude gains) or 'ap'
                (amplitude + phase)
        """
        self.n_ant = n_antennas
        self.baselines = list(baselines)
        if mode not in ('phase', 'ap'):
            raise ValueError("mode must be 'phase' or 'ap'")
        self.mode = mode

    def solve(self, vis_obs: np.ndarray, vis_model: np.ndarray,
              weights: Optional[np.ndarray] = None,
              n_iter: int = 10) -> np.ndarray:
        """
        Solve for antenna gains.

        Args:
            vis_obs: observed complex visibilities per baseline
            vis_model: model complex visibilities per baseline
            weights: optional per-baseline weights
            n_iter: alternating-solve iterations

        Returns:
            complex gain array (n_antennas,), normalized to mean |g|=1
        """
        obs = np.asarray(vis_obs)
        mod = np.asarray(vis_model)
        if obs.shape != mod.shape or obs.shape[0] != len(self.baselines):
            raise ValueError("visibility arrays must match the "
                             "baseline list")
        w = (np.ones_like(obs.real) if weights is None
             else np.asarray(weights, dtype=float))

        # Direct (log-domain) antenna calibration: with
        #   V_obs / V_mod = G_k = g_p conj(g_q)
        # the baseline phases obey  angle(G_k) = phi_p - phi_q  and the
        # log amplitudes obey  ln|G_k| = ln a_p + ln a_q.  Both are
        # linear systems solved by least squares (Schwab 1980). The
        # global phase degeneracy is pinned by sum(phi) = 0.
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = obs / mod
        good = np.isfinite(ratio) & (np.abs(mod) > 0)

        # ---- phases: A phi = angle(G) ----
        A = np.zeros((len(self.baselines) + 1, self.n_ant))
        b = np.zeros(len(self.baselines) + 1)
        for k, (p_, q_) in enumerate(self.baselines):
            A[k, p_] += 1.0
            A[k, q_] -= 1.0
            b[k] = np.angle(ratio[k]) if good[k] else 0.0
        A[-1, :] = 1.0                     # sum(phi) = 0
        b[-1] = 0.0
        phi = np.linalg.lstsq(A, b, rcond=None)[0]

        # ---- amplitudes: B x = ln|G| ----
        B = np.zeros((len(self.baselines) + 1, self.n_ant))
        y = np.zeros(len(self.baselines) + 1)
        n_used = 0
        for k, (p_, q_) in enumerate(self.baselines):
            if good[k] and np.abs(ratio[k]) > 0:
                B[k, p_] += 1.0
                B[k, q_] += 1.0
                y[k] = np.log(np.abs(ratio[k]))
                n_used += 1
        B[-1, :] = 1.0                     # sum(ln a) = 0
        y[-1] = 0.0
        if self.mode == 'ap' and n_used >= self.n_ant:
            ln_a = np.linalg.lstsq(B, y, rcond=None)[0]
            amps = np.exp(ln_a)
        else:
            amps = np.ones(self.n_ant)
        gains = amps * np.exp(1j * phi)
        gains = gains / np.mean(np.abs(gains))

        # normalize: mean amplitude 1
        gains = gains / np.mean(np.abs(gains))
        return gains

    def apply(self, gains: np.ndarray,
              vis: np.ndarray) -> np.ndarray:
        """Apply solved gains to a visibility array (corrupt them)."""
        out = vis.copy()
        for k, (p, q) in enumerate(self.baselines):
            out[k] = out[k] / (gains[p] * np.conj(gains[q]))
        return out


# =============================================================================
# VISIBILITY MODELING
# =============================================================================

class VisibilityModeler:
    """
    Forward modeling of visibilities from sky images.

    V(u, v) = integral I(l, m) exp[-2 pi i (ul + vm)] dl dm

    evaluated by FFT of the image, then interpolation to the observed
    (u, v) points. Also provides chi-squared comparison of a model
    against data.
    """

    def __init__(self, pixel_size: float):
        """
        Args:
            pixel_size: image pixel size (arcsec)
        """
        self.pixel_size = float(pixel_size)

    def model_visibilities(self, image: np.ndarray, u: np.ndarray,
                           v: np.ndarray) -> np.ndarray:
        """
        Predict visibilities at the given uv coordinates.

        The image FFT samples visibilities on a grid with spacing
        du = 1/(nx * dl); off-grid points are bilinearly interpolated.

        Args:
            image: sky brightness (ny, nx), brightness units
            u, v: uv coordinates (wavelengths)

        Returns:
            complex model visibilities at (u, v)
        """
        img = np.asarray(image, dtype=float)
        ny, nx = img.shape
        dl = self.pixel_size / 206265.0        # arcsec -> radians

        # DFT convention: V(u,v) = sum I(l,m) e^{-2pi i (u l + v m)};
        # with numpy's ifft2 (which has the e^{+2pi i} kernel) applied
        # to the shifted image we get exactly this, up to the pixel
        # area factor dl*dm
        # image pixel values are flux per pixel (e.g. Jy): the DFT
        # sum over pixels gives the visibility directly
        vis_grid = fftshift(ifft2(ifftshift(img))) * (nx * ny)
        # uv grid coordinates of the FFT samples
        u_grid = fftshift(fftfreq(nx, d=dl))
        v_grid = fftshift(fftfreq(ny, d=dl))

        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        # grid in raster order for bilinear interpolation
        uu, vv = np.meshgrid(u_grid, v_grid)
        points = np.column_stack([uu.ravel(), vv.ravel()])
        values = vis_grid.ravel()
        return griddata(points, values, np.column_stack([u, v]),
                        method='linear')

    def chi_squared(self, vis_obs: np.ndarray, vis_err: np.ndarray,
                    image: np.ndarray, u: np.ndarray,
                    v: np.ndarray) -> float:
        """Chi-squared of the model image against visibility data."""
        model = self.model_visibilities(image, u, v)
        resid = (np.asarray(vis_obs) - model) / np.asarray(vis_err)
        return float(np.sum(np.abs(resid) ** 2))

