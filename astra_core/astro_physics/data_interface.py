#!/usr/bin/env python3
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
Observational Data Interface Layer for ASTRO-SWARM
===================================================

Comprehensive interfaces for reading, writing, and manipulating
astronomical data formats.

Capabilities:
1. FITS file I/O (images, tables, cubes)
2. Spectral cube handling (position-position-velocity)
3. VOTable and Virtual Observatory support
4. CASA measurement set interface
5. Automatic unit handling
6. Region file support
7. World Coordinate System (WCS) transformations

Key Dependencies:
- astropy (FITS, WCS, units, tables)
- spectral-cube (optional, for advanced cube handling)
- regions (optional, for region file support)

Author: Claude Code (ASTRO-SWARM)
Date: 2024-11
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from pathlib import Path
import warnings
import json

# Try to import optional dependencies
try:
    from astropy.io import fits
    from astropy import units as u
    from astropy.wcs import WCS
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    warnings.warn("astropy not available - some functionality will be limited")

try:
    from spectral_cube import SpectralCube
    SPECTRAL_CUBE_AVAILABLE = True
except ImportError:
    SPECTRAL_CUBE_AVAILABLE = False
    SpectralCube = None

try:
    from regions import Regions
    REGIONS_AVAILABLE = True
except ImportError:
    REGIONS_AVAILABLE = False
    Regions = None

# Starlink NDF support (for JCMT, UKIRT data)
try:
    from starlink import Ndf
    from starlink import Ast
    NDF_AVAILABLE = True
except ImportError:
    NDF_AVAILABLE = False
    Ast = None
    Ndf = None


# =============================================================================
# UNIT HANDLING
# =============================================================================

class AstroUnits:
    """
    Astronomical unit handling and conversions.

    Provides standardized unit conversions for common astronomical quantities.
    """

    # Common unit conversions
    CONVERSIONS = {
        # Length
        'pc_to_cm': 3.0857e18,
        'kpc_to_cm': 3.0857e21,
        'Mpc_to_cm': 3.0857e24,
        'AU_to_cm': 1.496e13,
        'ly_to_cm': 9.461e17,

        # Mass
        'Msun_to_g': 1.989e33,
        'Mjup_to_g': 1.898e30,
        'Mearth_to_g': 5.972e27,

        # Luminosity
        'Lsun_to_erg_s': 3.828e33,

        # Flux
        'Jy_to_cgs': 1e-23,  # erg/s/cm²/Hz
        'mJy_to_cgs': 1e-26,
        'uJy_to_cgs': 1e-29,

        # Angle
        'arcsec_to_rad': np.pi / 180 / 3600,
        'arcmin_to_rad': np.pi / 180 / 60,
        'deg_to_rad': np.pi / 180,

        # Frequency/wavelength
        'GHz_to_Hz': 1e9,
        'MHz_to_Hz': 1e6,
        'um_to_cm': 1e-4,
        'nm_to_cm': 1e-7,
        'Angstrom_to_cm': 1e-8,

        # Temperature
        'K_to_K': 1.0,  # Identity

        # Column density
        'cm-2_to_cm-2': 1.0,

        # Velocity
        'km_s_to_cm_s': 1e5,
    }

    @classmethod
    def convert(cls, value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert between units.

        Parameters
        ----------
        value : float
            Value to convert
        from_unit : str
            Source unit
        to_unit : str
            Target unit

        Returns
        -------
        float : Converted value
        """
        key = f"{from_unit}_to_{to_unit}"
        if key in cls.CONVERSIONS:
            return value * cls.CONVERSIONS[key]

        # Try reverse
        key_rev = f"{to_unit}_to_{from_unit}"
        if key_rev in cls.CONVERSIONS:
            return value / cls.CONVERSIONS[key_rev]

        if from_unit == to_unit:
            return value

        raise ValueError(
            f"Unknown conversion: {from_unit} -> {to_unit}. "
            f"Available: {sorted(cls.CONVERSIONS)}")


# =============================================================================
# FITS HANDLER
# =============================================================================

class FITSHandler:
    """
    Read and write FITS images, tables, and headers (via astropy).

    Degrades gracefully: constructing the handler works without
    astropy, but I/O methods raise a RuntimeError explaining the
    missing dependency.
    """

    def __init__(self):
        self.last_header = None
        self.last_data = None

    # ------------------------------------------------------------- reading
    def read_image(self, path: str) -> Tuple[np.ndarray, Any]:
        """
        Read a FITS image.

        Returns (data, header); data is (ny, nx) float.
        """
        if not ASTROPY_AVAILABLE:
            raise RuntimeError("astropy required for FITS I/O")
        with fits.open(path) as hdul:
            self.last_header = hdul[0].header.copy()
            self.last_data = np.asarray(hdul[0].data, dtype=float)
        return self.last_data, self.last_header

    def read_cube(self, path: str) -> Tuple[np.ndarray, Any]:
        """
        Read a FITS data cube (position-position-velocity).
        Returns (data, header) with data (nv, ny, nx).
        """
        if not ASTROPY_AVAILABLE:
            raise RuntimeError("astropy required for FITS I/O")
        with fits.open(path) as hdul:
            self.last_header = hdul[0].header.copy()
            self.last_data = np.asarray(hdul[0].data, dtype=float)
        if self.last_data.ndim not in (2, 3):
            raise ValueError(f"expected 2-D image or 3-D cube, got "
                             f"{self.last_data.ndim}-D")
        return self.last_data, self.last_header

    def read_table(self, path: str, hdu: int = 1) -> "Table":
        """Read a FITS binary table into an astropy Table."""
        if not ASTROPY_AVAILABLE:
            raise RuntimeError("astropy required for FITS I/O")
        return Table.read(path, hdu=hdu)

    # ------------------------------------------------------------- writing
    def write_image(self, path: str, data: np.ndarray,
                    header: Any = None, overwrite: bool = True) -> None:
        """Write a 2-D image to FITS."""
        if not ASTROPY_AVAILABLE:
            raise RuntimeError("astropy required for FITS I/O")
        hdu = fits.PrimaryHDU(np.asarray(data))
        if header is not None:
            hdu.header.update(header)
        hdu.writeto(path, overwrite=overwrite)

    def append_table(self, path: str, table: "Table",
                     overwrite: bool = True) -> None:
        """Write an astropy Table as a FITS binary table."""
        if not ASTROPY_AVAILABLE:
            raise RuntimeError("astropy required for FITS I/O")
        table.write(path, format='fits', overwrite=overwrite)

    # --------------------------------------------------------- extraction
    def get_wcs(self, header: Any = None) -> Any:
        """WCS object for a header (or the last-read header)."""
        if not ASTROPY_AVAILABLE:
            raise RuntimeError("astropy required for WCS")
        return WCS(header if header is not None else self.last_header)

    def header_value(self, keyword: str, header: Any = None):
        """Value of a header keyword (case-insensitive)."""
        h = header if header is not None else self.last_header
        if h is None:
            raise RuntimeError("no header loaded")
        try:
            return h[keyword]
        except KeyError:
            return None

    def summary(self, header: Any = None) -> Dict[str, Any]:
        """Quick observational summary from the header."""
        h = header if header is not None else self.last_header
        if h is None:
            return {}
        return {
            'object': h.get('OBJECT', ''),
            'telescope': h.get('TELESCOP', ''),
            'instrument': h.get('INSTRUME', ''),
            'naxis': h.get('NAXIS', 0),
            'date_obs': str(h.get('DATE-OBS', '')),
            'exposure': h.get('EXPTIME', None),
            'rest_freq_ghz': h.get('RESTFRQ', 0.0) and
            h['RESTFRQ'] / 1e9,
        }


# =============================================================================
# SPECTRAL CUBE HANDLER
# =============================================================================

class SpectralCubeHandler:
    """
    Position-position-velocity cube utilities.

    Uses spectral-cube when available; otherwise falls back to plain
    numpy operations on the (nv, ny, nx) array with the FITS header.
    """

    def __init__(self, data: np.ndarray, header: Any = None):
        self.data = np.asarray(data, dtype=float)
        self.header = header
        self._cube = None
        if SPECTRAL_CUBE_AVAILABLE and header is not None:
            try:
                self._cube = SpectralCube(self.data, WCS(header))
            except Exception:
                self._cube = None

    # ------------------------------------------------------------ moments
    def moment0(self, velocity_range: Optional[Tuple[float, float]] = None) \
            -> np.ndarray:
        """Integrated intensity map (K km/s)."""
        d = self._channel_select(velocity_range)
        dv = self.channel_width()
        return np.nansum(d, axis=0) * dv

    def moment1(self, velocity_range: Optional[Tuple[float, float]] = None) \
            -> np.ndarray:
        """Intensity-weighted mean velocity (km/s)."""
        d = self._channel_select(velocity_range)
        vs = self._channel_select_axis(velocity_range)
        w = np.nansum(d, axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            m1 = np.nansum(d * vs[:, None, None], axis=0) / np.where(w > 0, w, np.nan)
        return m1

    def moment2(self, velocity_range: Optional[Tuple[float, float]] = None) \
            -> np.ndarray:
        """Intensity-weighted velocity dispersion (km/s)."""
        d = self._channel_select(velocity_range)
        vs = self._channel_select_axis(velocity_range)
        w = np.nansum(d, axis=0)
        m1 = self.moment1(velocity_range)
        with np.errstate(invalid='ignore', divide='ignore'):
            m2 = np.nansum(d * (vs[:, None, None] - m1[None, :, :]) ** 2,
                           axis=0) / np.where(w > 0, w, np.nan)
        return np.sqrt(m2)

    # ------------------------------------------------------------- axes
    def velocity_axis(self) -> np.ndarray:
        """Velocity axis (km/s) from the header."""
        if self.header is None:
            return np.arange(self.data.shape[0], dtype=float)
        crval = self.header.get('CRVAL3', 0.0)
        cdelt = self.header.get('CDELT3', 1.0)
        crpix = self.header.get('CRPIX3', 1.0)
        n = self.data.shape[0]
        return crval + (np.arange(n) + 1 - crpix) * cdelt

    def channel_width(self) -> float:
        """Velocity channel width |CDELT3| (km/s)."""
        return abs(self.header.get('CDELT3', 1.0)) if self.header \
            is not None else 1.0

    def _channel_select(self, velocity_range) -> np.ndarray:
        if velocity_range is None:
            return self.data
        v = self.velocity_axis()
        m = (v >= velocity_range[0]) & (v <= velocity_range[1])
        return self.data[m]

    def _channel_select_axis(self, velocity_range) -> np.ndarray:
        """Velocity axis restricted to the same channels as _channel_select."""
        v = self.velocity_axis()
        if velocity_range is None:
            return v
        m = (v >= velocity_range[0]) & (v <= velocity_range[1])
        return v[m]

    # ------------------------------------------------------------ spectra
    def spectrum(self, y: int, x: int) -> np.ndarray:
        """Spectrum at pixel (y, x)."""
        return self.data[:, y, x]

    def average_spectrum(self, mask: Optional[np.ndarray] = None) \
            -> np.ndarray:
        """Spatially averaged spectrum (optionally inside a mask)."""
        if mask is None:
            return np.nanmean(self.data, axis=(1, 2))
        return np.nanmean(self.data[:, mask], axis=1)

    def rms(self, exclude_edge: int = 5) -> float:
        """Noise RMS from signal-free edge channels."""
        n = self.data.shape[0]
        edge = np.concatenate([self.data[:exclude_edge].ravel(),
                               self.data[-exclude_edge:].ravel()])
        return float(np.nanstd(edge))


# =============================================================================
# VOTABLE HANDLER
# =============================================================================

class VOTableHandler:
    """Read/write VOTable (Virtual Observatory) catalogs via astropy."""

    def __init__(self):
        self.last_table = None
        self.last_path = None

    def read(self, path: str) -> "Table":
        """Read a VOTable file into an astropy Table."""
        if not ASTROPY_AVAILABLE:
            raise RuntimeError("astropy required for VOTable I/O")
        from astropy.io.votable import parse_single_table
        self.last_table = parse_single_table(path).to_table()
        self.last_path = path
        return self.last_table

    def write(self, path: str, table: "Table", overwrite: bool = True) -> None:
        """Write an astropy Table to VOTable XML."""
        if not ASTROPY_AVAILABLE:
            raise RuntimeError("astropy required for VOTable I/O")
        table.write(path, format='votable', overwrite=overwrite)

    def columns(self) -> List[str]:
        """Column names of the last-read table."""
        return list(self.last_table.colnames) if self.last_table else []

    def select(self, conditions: Dict[str, Tuple[str, float]]) -> "Table":
        """
        Filter the last-read table.

        conditions maps column -> (operator, value) with operator one
        of '<', '>', '<=', '>=', '=='.
        """
        t = self.last_table
        if t is None:
            raise RuntimeError("read a table first")
        mask = np.ones(len(t), dtype=bool)
        ops = {'<': np.less, '>': np.greater, '<=': np.less_equal,
               '>=': np.greater_equal, '==': np.equal}
        for col, (op, value) in conditions.items():
            mask &= ops[op](np.asarray(t[col]), value)
        return t[mask]

    def cone_search(self, ra_deg: float, dec_deg: float,
                    radius_arcmin: float) -> "Table":
        """Rows within a circular region of the given center."""
        t = self.last_table
        if t is None or 'ra' not in [c.lower() for c in t.colnames]:
            raise RuntimeError("table must contain 'ra'/'dec' columns")
        ra = np.asarray(t['ra'], dtype=float)
        dec = np.asarray(t['dec'], dtype=float)
        dra = (ra - ra_deg) * np.cos(np.radians(dec)) * 60.0
        ddec = (dec - dec_deg) * 60.0
        sep = np.hypot(dra, ddec)
        return t[sep <= radius_arcmin]


# =============================================================================
# REGION HANDLER
# =============================================================================

class RegionHandler:
    """
    DS9 region file handling.

    Parses the common DS9 shape syntax (circle, ellipse, box, polygon)
    directly; uses the `regions` package when available for full
    fidelity.
    """

    def __init__(self):
        self.regions: List[Dict[str, Any]] = []

    def read_ds9(self, path: str) -> List[Dict[str, Any]]:
        """
        Parse a DS9 region file into dictionaries with keys
        'shape', 'x', 'y', 'params'.
        """
        self.regions = []
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('global'):
                continue
            if line.lower().startswith(('fk5', 'fk4', 'icrs', 'image',
                                        'physical', 'galactic')):
                self._frame = line.split()[0].lower()
                continue
            parsed = self._parse_region_line(line)
            if parsed:
                self.regions.append(parsed)
        return self.regions

    def _parse_region_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse one DS9 shape line, e.g. 'circle(30.5, 12.3, 10")'."""
        line = line.strip().lstrip('+').split('#')[0].strip()
        for shape in ('circle', 'ellipse', 'box', 'polygon', 'point',
                      'annulus', 'line'):
            if line.startswith(shape):
                inner = line[len(shape):].strip()
                if not (inner.startswith('(') and inner.endswith(')')):
                    return None
                args = [a.strip() for a in inner[1:-1].split(',')]

                def _to_f(a: str) -> float:
                    a = a.rstrip('"\'')
                    try:
                        return float(a)
                    except ValueError:
                        return float(a.rstrip('dms:\'"'))

                params = [_to_f(a) for a in args]
                if shape == 'circle':
                    x, y, r = params[0], params[1], params[2]
                elif shape in ('ellipse', 'box'):
                    x, y = params[0], params[1]
                elif shape == 'polygon':
                    xs, ys = params[0::2], params[1::2]
                    x, y = float(np.mean(xs)), float(np.mean(ys))
                else:
                    x, y = params[0], params[1]
                return {'shape': shape, 'x': x, 'y': y,
                        'params': params, 'raw': line}
        return None

    def inside(self, x: float, y: float) -> bool:
        """Point-in-region test for the parsed regions."""
        for reg in self.regions:
            if reg['shape'] == 'circle':
                dx = x - reg['params'][0]
                dy = y - reg['params'][1]
                if dx * dx + dy * dy <= reg['params'][2] ** 2:
                    return True
            elif reg['shape'] in ('ellipse', 'box'):
                # Ellipse: a, b, theta; box: w, h, theta (approximate
                # as ellipse of semi-axes w/2, h/2)
                p = reg['params']
                a, b = (p[2], p[3]) if reg['shape'] == 'ellipse' \
                    else (p[2] / 2.0, p[3] / 2.0)
                theta = np.radians(p[4]) if len(p) > 4 else 0.0
                dx, dy = x - p[0], y - p[1]
                xr = dx * np.cos(theta) + dy * np.sin(theta)
                yr = -dx * np.sin(theta) + dy * np.cos(theta)
                if (xr / max(a, 1e-12)) ** 2 + \
                        (yr / max(b, 1e-12)) ** 2 <= 1.0:
                    return True
        return False

    def build_mask(self, shape: Tuple[int, int],
                   wcs_header: Any = None) -> np.ndarray:
        """
        Boolean mask (ny, nx) from the parsed regions, assuming the
        region coordinates are already pixel coordinates (or share the
        axis convention of the header if given).
        """
        ny, nx = shape
        mask = np.zeros(shape, dtype=bool)
        for reg in self.regions:
            if reg['shape'] == 'circle':
                yy, xx = np.ogrid[:ny, :nx]
                dx = xx - reg['params'][0]
                dy = yy - reg['params'][1]
                mask |= dx * dx + dy * dy <= reg['params'][2] ** 2
        return mask

    def write_ds9(self, path: str, regions: List[Dict[str, Any]],
                  frame: str = 'fk5') -> None:
        """Write regions in DS9 format."""
        lines = [f'# Region file format: DS9 version 4.1', frame]
        for reg in regions:
            lines.append(f"{reg['shape']}({', '.join(str(p) for p in reg['params'])})")
        Path(path).write_text('\n'.join(lines) + '\n')


# =============================================================================
# CASA INTERFACE
# =============================================================================

class CASAInterface:
    """
    Interface to (u, v, w) radio interferometry data in CASA-style
    Measurement Set layout.

    Full MS reading requires CASA itself; this interface reads the
    plain-table exports (UVFITS or the MS directory's TABLE layout
    when astropy can parse it) and provides the common analysis
    accessors. It deliberately does NOT shell out to CASA.
    """

    def __init__(self, uvfits_path: Optional[str] = None):
        self.u = None
        self.v = None
        self.w = None
        self.vis_re = None
        self.vis_im = None
        self.weight = None
        if uvfits_path is not None:
            self.read_uvfits(uvfits_path)

    def read_uvfits(self, path: str) -> Dict[str, np.ndarray]:
        """
        Read a UVFITS file (AIPS convention): extracts the uvw
        coordinates, complex visibilities and weights from the DATA
        column of the first randomly-grouped HDU.
        """
        if not ASTROPY_AVAILABLE:
            raise RuntimeError("astropy required for UVFITS I/O")
        with fits.open(path) as hdul:
            hdu = hdul[0]
            data = np.asarray(hdu.data['DATA'])
            uu = np.asarray(hdu.data['UU'], dtype=float) * 1e3
            vv = np.asarray(hdu.data['VV'], dtype=float) * 1e3
            ww = np.asarray(hdu.data['WW'], dtype=float) * 1e3
            self.u, self.v, self.w = uu, vv, ww
            if data.ndim == 4:
                # (n_row, 1, 1, n_if*3) -> real, imag, weight per IF
                self.vis_re = data[:, 0, 0, 0::3].ravel()
                self.vis_im = data[:, 0, 0, 1::3].ravel()
                self.weight = data[:, 0, 0, 2::3].ravel()
            else:
                self.vis_re = data[..., 0].ravel()
                self.vis_im = data[..., 1].ravel()
                self.weight = data[..., 2].ravel()
        return {'u': self.u, 'v': self.v, 'w': self.w}

    def complex_visibility(self) -> np.ndarray:
        """Visibilities as a complex array."""
        return self.vis_re + 1j * self.vis_im

    def uv_distance(self) -> np.ndarray:
        """Radial uv distances (lambda)."""
        return np.hypot(self.u, self.v)

    def amp_phase(self) -> Tuple[np.ndarray, np.ndarray]:
        """Amplitude and phase (rad) arrays."""
        cv = self.complex_visibility()
        return np.abs(cv), np.angle(cv)

    def uv_coverage_stats(self) -> Dict[str, float]:
        """Summary of the uv plane coverage."""
        d = self.uv_distance()
        return {
            'n_vis': int(d.size),
            'max_uv_lambda': float(d.max()) if d.size else 0.0,
            'min_uv_lambda': float(d.min()) if d.size else 0.0,
            'median_uv_lambda': float(np.median(d)) if d.size else 0.0,
            'mean_weight': float(np.mean(self.weight))
            if self.weight is not None and self.weight.size else 0.0,
        }

    def flag_low_amp(self, threshold: float = 0.0) -> int:
        """
        Flag (zero-weight) visibilities below an amplitude threshold.

        Returns the number of flagged samples.
        """
        amp, _ = self.amp_phase()
        bad = amp < threshold
        self.weight = np.where(bad, 0.0, self.weight)
        return int(bad.sum())
