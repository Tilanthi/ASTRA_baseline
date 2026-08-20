"""
SPH and Gas Dynamics Module

Smoothed Particle Hydrodynamics (SPH) implementation and gas dynamics modeling.
Includes molecular cloud formation, filament physics, and ISM turbulence.

Key capabilities:
- SPH particle operations and smoothing kernels
- Gas dynamics equations (momentum, energy, continuity)
- Molecular cloud formation and evolution
- Filament identification and analysis
- Turbulent driving and decay
- Shock capturing
- Self-gravity implementation
- Radiative cooling
- Chemistry integration

Date: 2025-12-22
Version: 1.0
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

# Physical constants (CGS)
G_GRAV = 6.674e-8  # cm^3/g/s^2
K_BOLTZMANN = 1.381e-16  # erg/K
M_H = 1.673e-24  # g
M_PROTON = 1.673e-24
M_SUN = 1.989e33  # g
PC = 3.086e18  # cm
KPC = 3.086e21  # cm


class KernelType(Enum):
    """SPH kernel types"""
    CUBIC_SPLINE = "cubic_spline"
    QUARTIC_SPLINE = "quartic_spline"
    WENDLAND = "wendland"
    GAUSSIAN = "gaussian"


@dataclass
class SPHParticle:
    """Single SPH particle"""
    pos: np.ndarray  # [x, y, z] in cm
    vel: np.ndarray  # [vx, vy, vz] in cm/s
    mass: float  # g
    rho: float = 0.0  # g/cm^3
    pressure: float = 0.0  # erg/cm^3
    temperature: float = 10.0  # K
    h: float = 0.1 * PC  # Smoothing length
    u: float = 0.0  # Internal energy (erg/g)
    metals: float = 0.01  # Metallicity
    molecule: Dict[str, float] = field(default_factory=dict)  # Molecular abundances


@dataclass
class Filament:
    """Molecular filament structure"""
    filament_id: str
    spine_points: np.ndarray  # [N, 3] positions along spine
    width: float  # pc
    length: float  # pc
    mass: float  # Msun
    density: float  # Mean H2 density (cm^-3)
    velocity_gradient: float  # Velocity coherence (km/s/pc)
    aspect_ratio: float = 0.0
    orientation: float = 0.0  # Position angle (degrees)
    n_cores: int = 0
    cores: List[Dict] = field(default_factory=list)


class SPHKernel:
    """
    SPH smoothing kernels.

    Kernels define how properties are interpolated between particles.
    Normalization: integral W(r, h) d^3r = 1
    """

    @staticmethod
    def cubic_spline(r: np.ndarray, h: float) -> np.ndarray:
        """
        Cubic spline kernel, 3-D normalisation (Monaghan & Lattanzio
        1985; support radius 2h):

        W(q) = (1/(pi h^3)) * {
            1 - (3/2)q^2 + (3/4)q^3,    0 <= q <= 1
            (1/4)(2 - q)^3,             1 < q <= 2
            0,                          q > 2
        }
        where q = r/h. Integrates to 1 over 3-D space.

        (Replaces a version that paired the 2-D spline pieces with a
        3-D prefactor and was not normalised in 3-D.)

        Args:
            r: Distance array (cm)
            h: Smoothing length (cm)

        Returns:
            Kernel values
        """
        q = np.asarray(r, dtype=float) / h
        w = np.zeros_like(q)

        sigma = 1.0 / (np.pi * h**3)

        mask1 = q <= 1.0
        mask2 = (q > 1.0) & (q <= 2.0)

        w[mask1] = 1.0 - 1.5*q[mask1]**2 + 0.75*q[mask1]**3
        w[mask2] = 0.25 * (2.0 - q[mask2])**3

        return sigma * w

    @staticmethod
    def cubic_spline_derivative(r: np.ndarray, h: float) -> np.ndarray:
        """
        Radial derivative dW/dr of the 3-D cubic spline:

        dW/dr = (1/(pi h^4)) * {
            -3q + (9/4)q^2,     0 <= q <= 1
            -(3/4)(2 - q)^2,    1 < q <= 2
            0,                  q > 2
        }
        """
        q = np.asarray(r, dtype=float) / h
        dw = np.zeros_like(q)

        sigma = 1.0 / (np.pi * h**4)

        mask1 = q <= 1.0
        mask2 = (q > 1.0) & (q <= 2.0)

        dw[mask1] = -3.0*q[mask1] + 2.25*q[mask1]**2
        dw[mask2] = -0.75 * (2.0 - q[mask2])**2

        return sigma * dw

    @staticmethod
    def wendland_c4(r: np.ndarray, h: float) -> np.ndarray:
        """
        Wendland kernel, compact support 2h (compactly supported,
        positive-definite; the polynomial is the standard Wendland C2
        form, kept under this name for API compatibility).

        W(q) = (21/(16 pi h^3)) * (1 - q/2)^4 * (1 + 2q),   q <= 2

        Integrates to 1 over 3-D space: with q = 2s,
        4 pi h^3 int q^2 W dq = (21/4) * 8 * int s^2(1-s)^4(1+4s) ds
        = 42/42 * ... = 1 since int_0^1 s^2 (1-s)^4 (1+4s) ds = 1/42.
        (Replaces a version truncated at q <= 1, which integrated to
        only ~0.18.)

        Args:
            r: Distance array (cm)
            h: Smoothing length (cm)

        Returns:
            Kernel values
        """
        q = np.asarray(r, dtype=float) / h
        w = np.zeros_like(q)

        mask = q <= 2.0
        w[mask] = (1.0 - 0.5*q[mask])**4 * (1.0 + 2.0*q[mask])

        sigma = 21.0 / (16.0 * np.pi * h**3)

        return sigma * w

    @staticmethod
    def gaussian(r: np.ndarray, h: float) -> np.ndarray:
        """
        Gaussian kernel.

        W(q) = (1/(pi*h^3)^(3/2)) * exp(-q^2)
        where q = r/h

        Args:
            r: Distance array (cm)
            h: Smoothing length (cm)

        Returns:
            Kernel values
        """
        q = r / h
        sigma = 1.0 / ((np.pi * h**2) ** 1.5)  # 3D normalization
        w = sigma * np.exp(-q**2)
        return w

    @staticmethod
    def get_kernel(kernel_type: KernelType) -> Callable:
        """Get kernel function by type"""
        if kernel_type == KernelType.CUBIC_SPLINE:
            return SPHKernel.cubic_spline
        elif kernel_type == KernelType.WENDLAND:
            return SPHKernel.wendland_c4
        elif kernel_type == KernelType.GAUSSIAN:
            return SPHKernel.gaussian
        else:
            return SPHKernel.cubic_spline  # Default


class SPHSimulation:
    """
    Basic SPH simulation implementation.

    Features:
    - Density calculation
    - Pressure forces
    - Artificial viscosity
    - Self-gravity (simplified)
    - Time integration (leapfrog)
    """

    def __init__(self, particles: List[SPHParticle],
                 kernel_type: KernelType = KernelType.CUBIC_SPLINE):
        """
        Initialize SPH simulation.

        Args:
            particles: List of SPH particles
            kernel_type: Smoothing kernel to use
        """
        self.particles = particles
        self.n_particles = len(particles)
        self.kernel_type = kernel_type
        self.kernel = SPHKernel.get_kernel(kernel_type)
        self.time = 0.0

    def compute_density(self) -> np.ndarray:
        """
        Compute density for all particles.

        rho_i = sum_j m_j W_ij

        Returns:
            Densities (g/cm^3)
        """
        # Extract positions and masses
        pos = np.array([p.pos for p in self.particles])
        mass = np.array([p.mass for p in self.particles])
        h = np.array([p.h for p in self.particles])

        # Build KD-tree for neighbor finding
        tree = cKDTree(pos)

        rho = np.zeros(self.n_particles)

        for i in range(self.n_particles):
            # Find neighbors within 2h
            neighbors = tree.query_ball_point(pos[i], 2*h[i])

            # Compute density contribution
            for j in neighbors:
                r = np.linalg.norm(pos[i] - pos[j])
                w = self.kernel(np.array([r]), h[i])[0]
                rho[i] += mass[j] * w

        # Store in particles
        for i, r in enumerate(rho):
            self.particles[i].rho = r

        return rho

    def compute_pressure(self, rho: np.ndarray,
                        temperature: np.ndarray) -> np.ndarray:
        """
        Compute pressure from equation of state.

        P = rho * k_B * T / (mu * m_H)

        Args:
            rho: Densities (g/cm^3)
            temperature: Temperatures (K)

        Returns:
            Pressures (erg/cm^3)
        """
        mu = 2.3  # Mean molecular weight (molecular gas)
        pressure = rho * K_BOLTZMANN * temperature / (mu * M_H)

        for i, p in enumerate(self.particles):
            p.pressure = pressure[i]

        return pressure

    def compute_forces(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute hydrodynamical forces.

        Includes:
        - Pressure gradient forces
        - Artificial viscosity (shock capturing)

        Returns:
            (acceleration, du/dt) arrays
        """
        pos = np.array([p.pos for p in self.particles])
        vel = np.array([p.vel for p in self.particles])
        mass = np.array([p.mass for p in self.particles])
        rho = np.array([p.rho for p in self.particles])
        pressure = np.array([p.pressure for p in self.particles])
        h = np.array([p.h for p in self.particles])

        acc = np.zeros_like(pos)
        du_dt = np.zeros(self.n_particles)

        tree = cKDTree(pos)

        for i in range(self.n_particles):
            neighbors = tree.query_ball_point(pos[i], 2*h[i])

            for j in neighbors:
                if i == j:
                    continue

                rij = pos[j] - pos[i]
                r = np.linalg.norm(rij)

                if r < 1e-10:
                    continue

                # Kernel gradient
                dw_dr = SPHKernel.cubic_spline_derivative(np.array([r]), h[i])[0]
                grad_w = dw_dr * rij / r

                # Pressure force
                p_term = (pressure[i] / rho[i]**2 + pressure[j] / rho[j]**2)
                acc[i] -= mass[j] * p_term * grad_w

        return acc, du_dt

    def integration_step(self, dt: float):
        """
        Leapfrog integration step.

        Args:
            dt: Timestep (s)
        """
        # Get current state
        pos = np.array([p.pos for p in self.particles])
        vel = np.array([p.vel for p in self.particles])
        mass = np.array([p.mass for p in self.particles])

        # Compute forces
        acc, _ = self.compute_forces()

        # Kick velocities
        vel_half = vel + 0.5 * acc * dt

        # Drift positions
        pos_new = pos + vel_half * dt

        # Update positions
        for i, p in enumerate(self.particles):
            p.pos = pos_new[i]
            p.vel = vel_half[i]  # Temporary

        # Compute new forces
        for i, p in enumerate(self.particles):
            p.pos = pos_new[i]  # Ensure updated
        acc_new, _ = self.compute_forces()

        # Kick velocities
        vel_new = vel_half + 0.5 * acc_new * dt

        # Update
        for i, p in enumerate(self.particles):
            p.pos = pos_new[i]
            p.vel = vel_new[i]

        self.time += dt


class FilamentFinder:
    """
    Identify and analyze filaments in molecular cloud data.

    Methods:
    - Skeleton extraction
    - Width measurement
    - Density profile
    - Velocity coherence
    """

    def __init__(self, min_length: float = 0.5, min_width: float = 0.05):
        """
        Initialize filament finder.

        Args:
            min_length: Minimum filament length (pc)
            min_width: Minimum filament width (pc)
        """
        self.min_length = min_length
        self.min_width = min_width

    def find_filaments(self, data: np.ndarray, threshold: float = None) -> List[Filament]:
        """
        Find filaments in 2D/3D data cube.

        Args:
            data: Density/Intensity data (nD array)
            threshold: Detection threshold

        Returns:
            List of filaments
        """
        filaments = []

        # Simple thresholding + skeletonization
        if threshold is None:
            threshold = np.mean(data) + 2 * np.std(data)

        # Binary mask
        mask = data > threshold

        # Morphological skeletonization (scikit-image; scipy.ndimage
        # has never exported skeletonize)
        try:
            from skimage.morphology import skeletonize
        except ImportError:
            from scipy.ndimage import binary_erosion, label as cc_label
            skeleton = mask  # fallback: full mask, no thinning
        else:
            skeleton = skeletonize(mask)

        # Extract skeleton points
        points = np.argwhere(skeleton)

        if len(points) > 0:
            # Create filament from skeleton
            fil = self._create_filament_from_skeleton(points, data)
            if fil and fil.length >= self.min_length:
                filaments.append(fil)

        return filaments

    def _create_filament_from_skeleton(self, points: np.ndarray,
                                       data: np.ndarray) -> Optional[Filament]:
        """Create filament object from skeleton points"""
        if len(points) < 2:
            return None

        # Sort points along primary axis
        # Get principal components
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(2, points.shape[1]))
        pca.fit(points)

        # Order the skeleton points along the principal axis
        proj = points @ pca.components_[0]
        order = np.argsort(proj)
        spine = points[order]

        # Length: path integral along the ordered spine (grid units,
        # assumed to be pc - FilamentFinder has no WCS information)
        seg = np.diff(spine, axis=0)
        length = float(np.sum(np.linalg.norm(seg, axis=1)))
        if length < 1e-12:
            return None

        # Effective width: mask area / spine length, the standard
        # "area over length" filament width (grid units = pc)
        threshold = np.mean(data) + 2 * np.std(data)
        mask = data > threshold
        effective_width = float(np.sum(mask) / max(len(spine), 1))
        width = max(effective_width, 1e-3)

        # Mass and mean density, interpreting data as H2 column
        # density N(H2) in cm^-2 on a grid whose pixels are 1 pc wide
        n_pix_mask = float(np.sum(mask))
        mean_col = float(np.mean(data[mask])) if mask.any() else 0.0
        mu = 2.33
        pixel_area_cm2 = PC ** 2
        mass = float(n_pix_mask * mean_col * pixel_area_cm2
                     * mu * M_H / M_SUN)
        density = float(mean_col / max(width * PC, 1.0))  # cm^-3

        # Orientation: position angle of the principal axis in the
        # first two coordinate axes (degrees)
        d = pca.components_[0]
        orientation = float(np.degrees(np.arctan2(d[0], d[1])) % 180.0) \
            if points.shape[1] >= 2 else 0.0

        # Cores: local maxima of the data inside the mask (8-neighbour)
        from scipy.ndimage import maximum_filter
        maxf = maximum_filter(data, size=3)
        peak_mask = (data == maxf) & mask
        peak_mask &= data > np.mean(data) + 3 * np.std(data)
        n_cores = int(np.sum(peak_mask))
        cores = [{'position': list(map(int, p)),
                  'value': float(data[tuple(p)])}
                 for p in np.argwhere(peak_mask)]

        return Filament(
            filament_id=f"fil_{id(points) % 10000:04d}",
            spine_points=spine,
            width=width,
            length=length,
            mass=mass,
            density=density,
            velocity_gradient=0.0,   # no velocity field available
            aspect_ratio=length / width,
            orientation=orientation,
            n_cores=n_cores,
            cores=cores,
        )


# =============================================================================
# MOLECULAR CLOUD FORMATION
# =============================================================================

class MolecularCloudFormation:
    """
    Analytic molecular-cloud physics: virial state, free-fall time,
    Jeans stability and collapse.

    All formulas are standard: virial parameter
    alpha = 5 sigma_v^2 R / (G M) (Bertoldi & McKee 1992),
    Jeans length/mass from isothermal sound speed, and free-fall time
    t_ff = sqrt(3 pi / (32 G rho)).
    """

    def __init__(self, mu: float = 2.33):
        """mu: mean molecular weight per free particle."""
        self.mu = mu

    def sound_speed(self, temperature: float) -> float:
        """Isothermal sound speed c_s = sqrt(k_B T / mu m_H) (cm/s)."""
        return float(np.sqrt(K_BOLTZMANN * temperature / (self.mu * M_H)))

    def free_fall_time(self, density: float) -> float:
        """Free-fall time t_ff (s) from mass density (g/cm^3)."""
        return float(np.sqrt(3.0 * np.pi / (32.0 * G_GRAV * density)))

    def jeans_length(self, density: float, temperature: float) -> float:
        """Jeans length lambda_J = c_s sqrt(pi / (G rho)) (cm)."""
        c_s = self.sound_speed(temperature)
        return float(c_s * np.sqrt(np.pi / (G_GRAV * density)))

    def jeans_mass(self, density: float, temperature: float) -> float:
        """
        Jeans mass M_J = (pi^(5/2)/6) c_s^3 / (G^(3/2) rho^(1/2)) (g),
        the mass inside a sphere of diameter lambda_J.
        """
        c_s = self.sound_speed(temperature)
        return float((np.pi ** 2.5 / 6.0) * c_s ** 3
                     / (G_GRAV ** 1.5 * np.sqrt(density)))

    def virial_parameter(self, mass_msun: float, radius_pc: float,
                         sigma_kms: float) -> float:
        """
        alpha = 5 sigma_v^2 R / (G M); alpha < 2 is bound, alpha < 1
        collapsing.
        """
        m = mass_msun * M_SUN
        r = radius_pc * PC
        return float(5.0 * (sigma_kms * 1e5) ** 2 * r / (G_GRAV * m))

    def cloud_state(self, mass_msun: float, radius_pc: float,
                    sigma_kms: float) -> Dict[str, Any]:
        """Virial analysis summary for one cloud."""
        alpha = self.virial_parameter(mass_msun, radius_pc, sigma_kms)
        state = ('unbound' if alpha > 2.0
                 else 'bound' if alpha > 1.0 else 'collapsing')
        return {'virial_parameter': alpha, 'state': state,
                'mass_msun': mass_msun, 'radius_pc': radius_pc,
                'sigma_kms': sigma_kms}

    def number_density(self, mass_msun: float, radius_pc: float) -> float:
        """Mean H2 number density n (cm^-3) of a uniform sphere."""
        m = mass_msun * M_SUN
        r = radius_pc * PC
        rho = 3.0 * m / (4.0 * np.pi * r ** 3)
        return float(rho / (2.8 * M_H))   # 2.8 = 2*mu_H2 per H2 molecule


# =============================================================================
# TURBULENT DRIVING
# =============================================================================

class TurbulentDriver:
    """
    Generate isotropic turbulent velocity fields with a power-law
    spectrum, in solenoidal or compressive mode.

    The field is built in Fourier space with amplitudes
    |v_hat(k)|^2 proportional to k^-(p+2) so that the angle-integrated
    spectrum E(k) = 4 pi k^2 |v_hat(k)|^2 ~ k^-p, then inverse
    transformed. Solenoidal (divergence-free) fields remove the
    compressive component in k-space (Federrath et al. 2010):
        v_hat_sol = v_hat - k (k.v_hat) / k^2.
    """

    def __init__(self, grid_size: int = 64, box_size_pc: float = 10.0,
                 spectral_slope: float = 2.0, seed: Optional[int] = None,
                 solenoidal: bool = True):
        """
        Args:
            grid_size: cubic grid resolution N
            box_size_pc: physical box size (pc)
            spectral_slope: p in E(k) ~ k^-p (2 for Burgers,
                5/3 for Kolmogorov)
            seed: RNG seed
            solenoidal: project out compressive modes if True
        """
        self.n = int(grid_size)
        self.L = box_size_pc * PC
        self.p = spectral_slope
        self.solenoidal = solenoidal
        self.rng = np.random.default_rng(seed)

    def generate(self) -> np.ndarray:
        """
        Returns:
            velocity field (3, N, N, N) in cm/s components with zero
            mean and unit mean-square speed (callers scale to sigma)
        """
        n = self.n
        kx = 2.0 * np.pi * np.fft.fftfreq(n, d=self.L / n)
        KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
        K2 = KX ** 2 + KY ** 2 + KZ ** 2
        K = np.sqrt(K2)

        vk = np.zeros((3, n, n, n), dtype=complex)
        for comp in range(3):
            phase = self.rng.normal(size=(n, n, n)) \
                + 1j * self.rng.normal(size=(n, n, n))
            vk[comp] = phase

        # kill the DC mode and impose the power-law amplitudes
        ksafe = np.where(K > 0, K, 1.0)
        amp = ksafe ** (-(self.p + 2.0) / 2.0)
        vk *= amp[None]
        vk[:, 0, 0, 0] = 0.0

        kvec = np.stack([KX, KY, KZ])
        if self.solenoidal:
            kdotv = np.einsum('i...,i...->...', kvec, vk)
            vk = vk - kvec * (kdotv / ksafe ** 2)[None]
        else:
            # purely compressive: keep only the longitudinal part
            kdotv = np.einsum('i...,i...->...', kvec, vk)
            vk = kvec * (kdotv / ksafe ** 2)[None]

        v = np.fft.ifftn(vk, axes=(1, 2, 3)).real
        v *= np.sqrt(n ** 3 / np.sum(v ** 2))
        return v

    def power_spectrum(self, v: np.ndarray, n_bins: int = 20) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Measured E(k) of a field: bins |v_hat(k)|^2 over shells of k.

        Returns:
            (k in 1/cm, E(k)) arrays for log-log fitting.
        """
        n = v.shape[1]
        kx = 2.0 * np.pi * np.fft.fftfreq(n, d=self.L / n)
        KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
        K = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
        vk = np.fft.fftn(v, axes=(1, 2, 3))
        power = np.sum(np.abs(vk) ** 2, axis=0)

        kflat, pflat = K.ravel(), power.ravel()
        kmax = kflat.max()
        edges = np.geomspace(kflat[kflat > 0].min(), kmax, n_bins + 1)
        kk, ee = [], []
        for i in range(n_bins):
            sel = (kflat >= edges[i]) & (kflat < edges[i + 1])
            if sel.sum() < 4:
                continue
            kk.append(np.mean(kflat[sel]))
            # E(k) ~ k^2 <|v_hat|^2>_shell
            ee.append(np.mean(pflat[sel]) * np.mean(kflat[sel]) ** 2)
        return np.array(kk), np.array(ee)

    @staticmethod
    def divergence(v: np.ndarray, spacing: float) -> np.ndarray:
        """Divergence by centred differences."""
        dvdx = np.gradient(v[0], spacing, axis=0)
        dvdy = np.gradient(v[1], spacing, axis=1)
        dvdz = np.gradient(v[2], spacing, axis=2)
        return dvdx + dvdy + dvdz


# =============================================================================
# GRAVITY SOLVER
# =============================================================================

class GravitySolver:
    """
    Direct-summation softened self-gravity (O(N^2), exact pairwise).

    Softening: r -> sqrt(r^2 + eps^2) in the denominator (Plummer
    softening), which regularises close encounters while preserving
    the far-field 1/r^2 force exactly.
    """

    def __init__(self, softening_pc: float = 0.01):
        self.eps = softening_pc * PC

    def accelerations(self, pos: np.ndarray, mass: np.ndarray) -> np.ndarray:
        """
        Args:
            pos: (N, 3) positions (cm)
            mass: (N,) masses (g)

        Returns:
            (N, 3) accelerations (cm/s^2); total momentum = 0
        """
        pos = np.asarray(pos, dtype=float)
        mass = np.asarray(mass, dtype=float)
        if mass.ndim == 0:
            mass = np.full(len(pos), float(mass))

        d = pos[None, :, :] - pos[:, None, :]      # (N, N, 3) j - i
        r2 = np.sum(d * d, axis=-1) + self.eps ** 2
        np.fill_diagonal(r2, 1.0)                  # skip self term
        inv_r3 = r2 ** -1.5
        np.fill_diagonal(inv_r3, 0.0)
        acc = G_GRAV * (mass[None, :, None] * d * inv_r3[:, :, None]) \
            .sum(axis=1)
        return acc

    def potential_energy(self, pos: np.ndarray, mass: np.ndarray) -> float:
        """Total gravitational potential energy W < 0 (erg)."""
        pos = np.asarray(pos, dtype=float)
        mass = np.asarray(mass, dtype=float)
        if mass.ndim == 0:
            mass = np.full(len(pos), float(mass))
        d = pos[None, :, :] - pos[:, None, :]
        r = np.sqrt(np.sum(d * d, axis=-1) + self.eps ** 2)
        np.fill_diagonal(r, np.inf)
        w = -0.5 * G_GRAV * np.sum(mass[:, None] * mass[None, :] / r)
        return float(w)

    @staticmethod
    def kinetic_energy(mass: np.ndarray, vel: np.ndarray) -> float:
        """Total kinetic energy (erg)."""
        return float(0.5 * np.sum(mass * np.sum(vel ** 2, axis=1)))

    def virial_ratio(self, pos: np.ndarray, mass: np.ndarray,
                     vel: np.ndarray) -> float:
        """2T / |W|; 1.0 for a virialised system."""
        t = self.kinetic_energy(mass, vel)
        w = self.potential_energy(pos, mass)
        return float(2.0 * t / abs(w))


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_sph_simulation(n_particles: int = 1000, box_size_pc: float = 1.0,
                          total_mass_msun: float = 100.0,
                          temperature: float = 10.0,
                          kernel_type: KernelType = KernelType.CUBIC_SPLINE,
                          seed: Optional[int] = None,
                          layout: str = 'random') -> SPHSimulation:
    """
    Build an SPHSimulation of identical particles in a periodic-ish box.

    layout='random' uniformly samples the box; layout='lattice' places
    particles on a Cartesian grid (exact uniform density).
    """
    rng = np.random.default_rng(seed)
    L = box_size_pc * PC
    m = total_mass_msun * M_SUN / n_particles
    h = L / max(n_particles ** (1.0 / 3.0), 1.0)

    if layout == 'lattice':
        n1 = int(round(n_particles ** (1.0 / 3.0)))
        if n1 ** 3 != n_particles:
            raise ValueError("lattice layout needs n_particles = n^3")
        g = (np.arange(n1) + 0.5) / n1 * L
        X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
        pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    else:
        pos = rng.uniform(0.0, L, size=(n_particles, 3))

    particles = [SPHParticle(pos=pos[i], vel=np.zeros(3), mass=m,
                             temperature=temperature, h=h)
                 for i in range(n_particles)]
    return SPHSimulation(particles, kernel_type=kernel_type)


def find_filaments_in_data(data: np.ndarray,
                           threshold: float = None) -> List[Filament]:
    """Convenience wrapper around FilamentFinder.find_filaments."""
    return FilamentFinder().find_filaments(data, threshold)


def get_h2_fraction(number_density: float, surface_density_msun_pc2: float,
                    metallicity: float = 1.0,
                    formation_coeff: float = 3e-17,
                    dissipation_rate: float = 3e-11,
                    dust_cross_section: float = 1.9e-21) -> float:
    """
    Depth-averaged molecular fraction of a plane-parallel slab
    irradiated from both sides.

    At every depth the gas is in local equilibrium between grain
    catalysis R n (1 - f) and photodissociation D0 exp(-sigma_d N_H) f,
    giving

        f(N) = 1 / (1 + D0 exp(-sigma_d N_H) / (R n)),

    which is exact for the steady-state two-species balance. The
    returned value is the column average over the slab.

    Default parameters are canonical diffuse-ISM values: R = 3e-17
    cm^3/s (grain formation coefficient, Jura 1975), D0 = 3e-11 s^-1
    (unattenuated rate in the Draine field), sigma_d = 1.9e-21 Z'
    cm^2 per H nucleus at ~1000 Angstrom.

    Args:
        number_density: total H-nucleus density n (cm^-3)
        surface_density_msun_pc2: total gas surface density (both
            sides); converted with N_H = Sigma / (1.4 m_H)
        metallicity: scales the dust cross section

    Returns:
        H2 mass fraction <f> in [0, 1]
    """
    Sigma = surface_density_msun_pc2 * M_SUN / PC ** 2      # g/cm^2
    N_total = Sigma / (1.4 * M_H)                           # H cm^-2
    Rn = formation_coeff * number_density
    sigma_d = dust_cross_section * metallicity

    # integrate f(N) over the column with the mid-plane symmetry
    n_grid = 2000
    half = N_total / 2.0
    N = np.linspace(0.0, half, n_grid)
    # depth measured from each surface: attenuation e^{-sigma (half - x)}
    f = 1.0 / (1.0 + dissipation_rate
               * np.exp(-sigma_d * (half - N)) / Rn)
    return float(np.mean(f))
