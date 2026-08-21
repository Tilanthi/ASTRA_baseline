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
Multi-Scale Coupling Framework for Astrophysical Simulations

Provides tools for connecting simulations across different scales:
- Zoom-in simulation techniques
- Sub-grid physics models
- Feedback prescriptions (stellar, AGN)
- Scale bridging and interpolation
- Hierarchical refinement

Applications:
- Galaxy formation with ISM physics
- Star formation in molecular clouds
- Protoplanetary disk structure
- AGN feedback and outflows
- Cosmological zoom simulations
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable, Union
from enum import Enum
from abc import ABC, abstractmethod
import json


# Physical constants (CGS)
G_GRAV = 6.67430e-8  # cm^3/g/s^2
C_LIGHT = 2.99792458e10  # cm/s
K_BOLTZMANN = 1.380649e-16  # erg/K
M_PROTON = 1.6726219e-24  # g
M_SUN = 1.989e33  # g
PC = 3.086e18  # cm
KPC = 3.086e21  # cm
MPC = 3.086e24  # cm
YR = 3.156e7  # s
MYR = 3.156e13  # s
GYR = 3.156e16  # s


class ScaleLevel(Enum):
    """Hierarchical scale levels."""
    COSMOLOGICAL = "cosmological"  # > 10 Mpc
    CLUSTER = "cluster"  # 1-10 Mpc
    GALAXY = "galaxy"  # 10-100 kpc
    CGM = "cgm"  # 10-300 kpc
    DISK = "disk"  # 1-30 kpc
    GMC = "gmc"  # 1-100 pc
    CLUMP = "clump"  # 0.1-1 pc
    CORE = "core"  # 0.01-0.1 pc
    PROTOSTELLAR = "protostellar"  # < 0.01 pc


@dataclass
class ScaleBoundary:
    """Defines the interface between simulation scales."""
    upper_scale: ScaleLevel
    lower_scale: ScaleLevel
    resolution_ratio: float  # Refinement factor
    overlap_fraction: float  # Fractional overlap region
    coupling_vars: List[str]  # Variables to transfer
    interpolation_order: int = 2
    conservation_enforced: bool = True


@dataclass
class SubGridModel(ABC):
    """Abstract base class for sub-grid physics models."""
    name: str
    scale_below: float  # pc, resolution below which model applies
    parameters: Dict[str, float] = field(default_factory=dict)

    @abstractmethod
    def compute(self, local_properties: Dict[str, Any]) -> Dict[str, Any]:
        """Compute sub-grid physics contribution."""
        pass


class TurbulentPressureModel(SubGridModel):
    """
    Sub-grid turbulent pressure support.

    Adds effective pressure from unresolved turbulent motions.
    """

    def __init__(self, sigma_turb: float = 1.0, scale_below: float = 1.0):
        """
        Parameters
        ----------
        sigma_turb : float
            Turbulent velocity dispersion at reference scale (km/s)
        scale_below : float
            Scale below which model applies (pc)
        """
        self.name = "turbulent_pressure"
        self.scale_below = scale_below
        self.parameters = {
            "sigma_turb": sigma_turb,
            "reference_scale": 1.0,  # pc
            "power_law_index": 0.5  # Larson relation
        }

    def compute(self, local_properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute turbulent pressure contribution.

        Parameters
        ----------
        local_properties : dict
            Must contain 'density' (g/cm^3) and 'cell_size' (pc)

        Returns
        -------
        dict
            Contains 'P_turb' (erg/cm^3), 'sigma_eff' (km/s)
        """
        rho = local_properties['density']
        cell_size = local_properties.get('cell_size', self.scale_below)

        # Turbulent velocity from Larson relation
        sigma_ref = self.parameters['sigma_turb'] * 1e5  # cm/s
        l_ref = self.parameters['reference_scale'] * PC
        alpha = self.parameters['power_law_index']

        sigma_turb = sigma_ref * (cell_size * PC / l_ref)**alpha

        # Turbulent pressure
        P_turb = rho * sigma_turb**2

        return {
            'P_turb': P_turb,
            'sigma_eff': sigma_turb / 1e5,  # km/s
            'turbulent_energy': 0.5 * rho * sigma_turb**2
        }


class StarFormationModel(SubGridModel):
    """
    Sub-grid star formation prescription.

    Implements various star formation recipes.
    """

    def __init__(self, efficiency: float = 0.01, density_threshold: float = 100.0,
                 scale_below: float = 10.0, recipe: str = "schmidt"):
        """
        Parameters
        ----------
        efficiency : float
            Star formation efficiency per free-fall time
        density_threshold : float
            Density threshold in H/cm^3
        scale_below : float
            Scale below which model applies (pc)
        recipe : str
            Star formation recipe ("schmidt", "krumholz", "hopkins")
        """
        self.name = "star_formation"
        self.scale_below = scale_below
        self.recipe = recipe
        self.parameters = {
            "efficiency": efficiency,
            "n_threshold": density_threshold,
            "schmidt_index": 1.5,
            "virial_parameter_crit": 2.0
        }

    def compute(self, local_properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute star formation rate.

        Parameters
        ----------
        local_properties : dict
            Must contain 'density', 'temperature', optional 'velocity_dispersion'

        Returns
        -------
        dict
            Contains 'sfr_density' (M_sun/yr/pc^3), 'stellar_mass_formed'
        """
        rho = local_properties['density']
        T = local_properties.get('temperature', 10.0)

        # Number density
        n_H = rho / (1.4 * M_PROTON)
        n_threshold = self.parameters['n_threshold']

        if n_H < n_threshold:
            return {'sfr_density': 0.0, 'stellar_mass_formed': 0.0}

        # Free-fall time
        t_ff = np.sqrt(3 * np.pi / (32 * G_GRAV * rho))

        if self.recipe == "schmidt":
            # Simple Schmidt law
            epsilon = self.parameters['efficiency']
            sfr_vol = epsilon * rho / t_ff

        elif self.recipe == "krumholz":
            # Krumholz & McKee (2005) turbulence-regulated SF
            sigma = local_properties.get('velocity_dispersion', 1.0) * 1e5  # cm/s
            alpha_vir = self.parameters['virial_parameter_crit']

            # Virial parameter
            cell_size = local_properties.get('cell_size', 1.0) * PC
            alpha = 5 * sigma**2 * cell_size / (G_GRAV * rho * cell_size**3)

            # Efficiency depends on virial state
            epsilon = self.parameters['efficiency'] * np.exp(-alpha_vir / alpha)
            sfr_vol = epsilon * rho / t_ff

        elif self.recipe == "hopkins":
            # Hopkins (2013) multi-freefall
            sigma = local_properties.get('velocity_dispersion', 1.0) * 1e5
            mach = sigma / np.sqrt(K_BOLTZMANN * T / M_PROTON)

            # Multi-freefall model
            s_crit = np.log(1 + 0.5 * mach**2)
            epsilon = 0.5 * (1 + np.tanh((np.log(n_H / n_threshold) - s_crit) / 0.5))
            sfr_vol = epsilon * rho / t_ff

        else:
            sfr_vol = 0.0

        # Convert to M_sun/yr/pc^3
        sfr_density = sfr_vol / M_SUN * YR * PC**3

        # Mass formed in timestep
        dt = local_properties.get('timestep', 1e4 * YR)
        mass_formed = sfr_vol * local_properties.get('cell_volume', PC**3) * dt / M_SUN

        return {
            'sfr_density': sfr_density,
            'stellar_mass_formed': mass_formed,
            'free_fall_time': t_ff / YR  # years
        }


class StellarFeedbackModel(SubGridModel):
    """
    Sub-grid stellar feedback prescription.

    Includes:
    - Supernova energy/momentum injection
    - Stellar winds
    - Radiation pressure
    - Photoionization heating
    """

    def __init__(self, scale_below: float = 100.0,
                 sn_energy: float = 1e51,
                 coupling_efficiency: float = 0.1):
        """
        Parameters
        ----------
        scale_below : float
            Scale below which feedback is sub-grid (pc)
        sn_energy : float
            Energy per supernova (erg)
        coupling_efficiency : float
            Fraction of energy coupled to ISM
        """
        self.name = "stellar_feedback"
        self.scale_below = scale_below
        self.parameters = {
            "E_sn": sn_energy,
            "eta_couple": coupling_efficiency,
            "sn_rate_per_100msun": 1.0,  # per 100 M_sun of SF
            "wind_mass_loading": 0.3,
            "wind_velocity": 30.0,  # km/s
            "ionizing_photon_rate": 1e49,  # per M_sun of young stars
            "radiation_pressure_boost": 2.0
        }

    def compute(self, local_properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute stellar feedback.

        Parameters
        ----------
        local_properties : dict
            Must contain 'stellar_mass' or 'sfr', 'density'

        Returns
        -------
        dict
            Energy, momentum, and mass injection rates
        """
        # Star formation rate or recent stellar mass
        sfr = local_properties.get('sfr', 0.0)  # M_sun/yr
        young_stars = local_properties.get('young_stellar_mass', sfr * 10e6)  # M_sun
        rho = local_properties['density']

        # Supernova feedback
        sn_rate = self.parameters['sn_rate_per_100msun'] * sfr / 100.0  # per year
        E_sn = self.parameters['E_sn']
        eta = self.parameters['eta_couple']

        # Energy injection rate
        E_dot_sn = eta * sn_rate * E_sn / YR  # erg/s

        # AUDIT-FLAG (H3, NOT FIXED - outside this worker's scope):
        # p_dot_sn below is PER YEAR (sn_rate is /yr) while p_dot_wind
        # and p_dot_rad are PER SECOND, and the three are summed.  With
        # SFR = 1 Msun/yr, n_H = 43: p_dot_sn = 3.663e41 vs
        # p_dot_wind = 5.672e31 and p_dot_rad = 2.555e33, so the SN term
        # is overstated by 3.156e7 and the sum is 100% SN.  The
        # `heating_rate` two lines further down is also dimensionally
        # meaningless (a global photon rate times a local number
        # density times two magic constants).
        # Momentum injection (Sedov-Taylor terminal momentum)
        n_H = rho / (1.4 * M_PROTON)
        p_terminal = 3e5 * M_SUN * 1e5 * (E_sn / 1e51)**0.93 * (n_H / 1.0)**(-0.13)
        p_dot_sn = sn_rate * p_terminal  # g cm/s /yr

        # Stellar winds
        wind_mass_rate = self.parameters['wind_mass_loading'] * sfr * M_SUN / YR  # g/s
        wind_velocity = self.parameters['wind_velocity'] * 1e5  # cm/s
        p_dot_wind = wind_mass_rate * wind_velocity

        # Radiation pressure
        L_young = 1e3 * young_stars * 3.83e33  # erg/s, young stellar luminosity
        boost = self.parameters['radiation_pressure_boost']
        p_dot_rad = boost * L_young / C_LIGHT

        # Photoionization heating
        Q_ion = self.parameters['ionizing_photon_rate'] * young_stars  # photons/s
        heating_rate = 1e-13 * n_H * Q_ion * 2e-11  # erg/s (approximate)

        return {
            'energy_injection_rate': E_dot_sn + 0.5 * wind_mass_rate * wind_velocity**2,
            'momentum_injection_rate': p_dot_sn + p_dot_wind + p_dot_rad,
            'mass_injection_rate': wind_mass_rate,
            'heating_rate': heating_rate,
            'supernova_rate': sn_rate,
            'ionizing_luminosity': Q_ion
        }


class AGNFeedbackModel(SubGridModel):
    """
    Sub-grid AGN feedback prescription.

    Implements:
    - Quasar/radiative mode
    - Kinetic/jet mode
    - Radiation pressure on dust
    """

    def __init__(self, scale_below: float = 1000.0,
                 radiative_efficiency: float = 0.1,
                 coupling_efficiency: float = 0.05):
        """
        Parameters
        ----------
        scale_below : float
            Scale below which AGN is sub-grid (pc)
        radiative_efficiency : float
            Radiative efficiency of accretion
        coupling_efficiency : float
            Fraction of energy coupled to surrounding gas
        """
        self.name = "agn_feedback"
        self.scale_below = scale_below
        self.parameters = {
            "epsilon_r": radiative_efficiency,
            "epsilon_f": coupling_efficiency,
            "eddington_factor_crit": 0.01,  # Below this, jet mode
            "jet_efficiency": 0.1,
            "jet_velocity": 10000.0,  # km/s
            "momentum_boost": 20.0  # IR momentum boost
        }

    def compute(self, local_properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute AGN feedback.

        Parameters
        ----------
        local_properties : dict
            Must contain 'bh_mass' (M_sun), 'accretion_rate' (M_sun/yr)

        Returns
        -------
        dict
            Energy and momentum injection rates
        """
        M_bh = local_properties.get('bh_mass', 1e6) * M_SUN  # g
        Mdot = local_properties.get('accretion_rate', 0.01) * M_SUN / YR  # g/s

        # Eddington luminosity and rate
        L_edd = 4 * np.pi * G_GRAV * M_bh * M_PROTON * C_LIGHT / 6.65e-25
        Mdot_edd = L_edd / (self.parameters['epsilon_r'] * C_LIGHT**2)

        f_edd = Mdot / Mdot_edd

        # Bolometric luminosity
        L_bol = self.parameters['epsilon_r'] * Mdot * C_LIGHT**2

        if f_edd > self.parameters['eddington_factor_crit']:
            # Quasar/radiative mode
            E_dot = self.parameters['epsilon_f'] * L_bol

            # Radiation pressure on dust (momentum boost)
            boost = self.parameters['momentum_boost']
            p_dot = boost * L_bol / C_LIGHT

            mode = "quasar"

        else:
            # Jet/kinetic mode
            E_dot = self.parameters['jet_efficiency'] * Mdot * C_LIGHT**2
            v_jet = self.parameters['jet_velocity'] * 1e5  # cm/s
            p_dot = E_dot / v_jet

            mode = "jet"

        return {
            'energy_injection_rate': E_dot,
            'momentum_injection_rate': p_dot,
            'luminosity': L_bol,
            'eddington_ratio': f_edd,
            'feedback_mode': mode
        }


class CoolingFunction:
    """
    Gas cooling function for ISM/CGM physics.

    Includes:
    - Atomic line cooling
    - Metal cooling
    - Molecular cooling
    - Dust cooling
    """

    def __init__(self, metallicity: float = 1.0, redshift: float = 0.0):
        """
        Parameters
        ----------
        metallicity : float
            Metallicity in solar units
        redshift : float
            Redshift for CMB floor
        """
        self.metallicity = metallicity
        self.redshift = redshift
        self.T_cmb = 2.725 * (1 + redshift)

        # Cooling table temperatures
        self._T_table = np.logspace(1, 9, 100)
        self._build_cooling_table()

    def _build_cooling_table(self):
        """Build interpolation table for cooling function."""
        T = self._T_table

        # Primordial cooling (H, He)
        Lambda_prim = np.zeros_like(T)

        # Collisional ionization
        Lambda_prim += 1.27e-21 * np.sqrt(T) * np.exp(-157809 / T) / (1 + np.sqrt(T / 1e5))

        # Recombination
        Lambda_prim += 8.7e-27 * np.sqrt(T) * (T / 1e3)**(-0.2) / (1 + (T / 1e6)**0.7)

        # Collisional excitation
        Lambda_prim += 7.5e-19 * np.exp(-118348 / T) / (1 + np.sqrt(T / 1e5))

        # Bremsstrahlung
        Lambda_prim += 1.42e-27 * np.sqrt(T)

        # Metal cooling (approximate CIE)
        Lambda_metal = np.zeros_like(T)

        # Low-T metal lines (OI, CII, etc.)
        Lambda_metal += 2e-26 * (T / 1e4)**0.5 * np.exp(-92 / T)

        # High-T metal lines
        Lambda_metal += 1e-22 * np.exp(-(np.log10(T) - 5.25)**2 / 0.5)

        self._Lambda_prim = Lambda_prim
        self._Lambda_metal = Lambda_metal

    def cooling_rate(self, temperature: float, density: float) -> float:
        """
        Compute cooling rate.

        Parameters
        ----------
        temperature : float
            Gas temperature in K
        density : float
            Gas density in g/cm^3

        Returns
        -------
        float
            Cooling rate in erg/cm^3/s
        """
        # Clamp temperature
        T = np.clip(temperature, 10, 1e9)

        # Interpolate cooling function
        Lambda_prim = np.interp(T, self._T_table, self._Lambda_prim)
        Lambda_metal = np.interp(T, self._T_table, self._Lambda_metal)

        Lambda_total = Lambda_prim + self.metallicity * Lambda_metal

        # Number density
        n_H = density / (1.4 * M_PROTON)

        # Cooling rate
        cool_rate = n_H**2 * Lambda_total

        # CMB floor
        if T < self.T_cmb:
            cool_rate = 0.0

        return cool_rate

    def cooling_time(self, temperature: float, density: float) -> float:
        """
        Compute cooling time.

        Parameters
        ----------
        temperature : float
            Gas temperature in K
        density : float
            Gas density in g/cm^3

        Returns
        -------
        float
            Cooling time in years
        """
        cool_rate = self.cooling_rate(temperature, density)

        if cool_rate <= 0:
            return np.inf

        # Thermal energy
        n_H = density / (1.4 * M_PROTON)
        E_thermal = 1.5 * n_H * K_BOLTZMANN * temperature

        t_cool = E_thermal / cool_rate

        return t_cool / YR


class ZoomRegion:
    """
    Defines a zoom-in simulation region.

    Handles:
    - Lagrangian region selection
    - Boundary conditions
    - Resolution hierarchy
    """

    def __init__(self, center: np.ndarray, radius: float,
                 base_resolution: float, max_refinement: int = 5):
        """
        Parameters
        ----------
        center : ndarray
            Center coordinates (kpc)
        radius : float
            Region radius (kpc)
        base_resolution : float
            Base resolution (pc)
        max_refinement : int
            Maximum refinement levels
        """
        self.center = center
        self.radius = radius
        self.base_resolution = base_resolution
        self.max_refinement = max_refinement

        self.refinement_levels = []
        self._setup_refinement_hierarchy()

    def _setup_refinement_hierarchy(self):
        """Set up nested refinement regions."""
        current_radius = self.radius
        current_res = self.base_resolution

        for level in range(self.max_refinement + 1):
            self.refinement_levels.append({
                'level': level,
                'radius': current_radius,
                'resolution': current_res,
                'cell_mass': None  # Set based on density
            })
            current_radius /= 2
            current_res /= 2

    def get_refinement_level(self, position: np.ndarray) -> int:
        """
        Determine refinement level at position.

        Parameters
        ----------
        position : ndarray
            Position coordinates (kpc)

        Returns
        -------
        int
            Refinement level (0 = coarsest)
        """
        r = np.linalg.norm(position - self.center)

        for level_info in reversed(self.refinement_levels):
            if r <= level_info['radius']:
                return level_info['level']

        return 0

    def get_resolution(self, position: np.ndarray) -> float:
        """Get spatial resolution at position."""
        level = self.get_refinement_level(position)
        return self.refinement_levels[level]['resolution']


class BoundaryConditionHandler:
    """
    Handles boundary conditions between simulation scales.

    Supports:
    - Fixed boundary
    - Periodic boundary
    - Outflow boundary
    - Inflowing boundary from larger scale
    """

    def __init__(self, boundary_type: str = "outflow"):
        """
        Parameters
        ----------
        boundary_type : str
            Type of boundary ("fixed", "periodic", "outflow", "inflow")
        """
        self.boundary_type = boundary_type
        self.inflow_conditions: Dict[str, Any] = {}

    def set_inflow_conditions(self, density: float, velocity: np.ndarray,
                              temperature: float, metallicity: float = 1.0):
        """
        Set inflow boundary conditions from larger scale.

        Parameters
        ----------
        density : float
            Inflow density (g/cm^3)
        velocity : ndarray
            Inflow velocity vector (cm/s)
        temperature : float
            Inflow temperature (K)
        metallicity : float
            Inflow metallicity (solar)
        """
        self.inflow_conditions = {
            'density': density,
            'velocity': velocity,
            'temperature': temperature,
            'metallicity': metallicity
        }

    def apply_boundary(self, field: np.ndarray, axis: int,
                       side: str) -> np.ndarray:
        """
        Apply boundary conditions to field.

        Parameters
        ----------
        field : ndarray
            Field values
        axis : int
            Axis for boundary (0, 1, or 2)
        side : str
            Side of boundary ("lower" or "upper")

        Returns
        -------
        ndarray
            Field with boundary applied
        """
        if self.boundary_type == "periodic":
            # Wrap around
            if side == "lower":
                field = np.roll(field, 1, axis=axis)
            else:
                field = np.roll(field, -1, axis=axis)

        elif self.boundary_type == "outflow":
            # Zero gradient
            slices = [slice(None)] * field.ndim
            if side == "lower":
                slices[axis] = 0
                field[tuple(slices)] = field[tuple([slice(None) if i != axis else 1
                                                    for i in range(field.ndim)])]
            else:
                slices[axis] = -1
                field[tuple(slices)] = field[tuple([slice(None) if i != axis else -2
                                                    for i in range(field.ndim)])]

        elif self.boundary_type == "fixed":
            # Keep boundary values fixed
            pass

        return field


class ScaleCoupler:
    """
    Couples simulations at different scales.

    Handles:
    - Downsampling from large to small scale
    - Upsampling/feedback from small to large scale
    - Conservation enforcement
    """

    def __init__(self, upper_scale: ScaleLevel, lower_scale: ScaleLevel,
                 refinement_factor: int = 4):
        """
        Parameters
        ----------
        upper_scale : ScaleLevel
            Coarser scale level
        lower_scale : ScaleLevel
            Finer scale level
        refinement_factor : int
            Resolution ratio between scales
        """
        self.upper_scale = upper_scale
        self.lower_scale = lower_scale
        self.refinement_factor = refinement_factor

        self.conserved_quantities = ['mass', 'momentum', 'energy']

    def downsample(self, coarse_field: np.ndarray,
                   interpolation: str = "linear") -> np.ndarray:
        """
        Interpolate from coarse to fine grid.

        Parameters
        ----------
        coarse_field : ndarray
            Field on coarse grid
        interpolation : str
            Interpolation method ("nearest", "linear", "cubic")

        Returns
        -------
        ndarray
            Field interpolated to fine grid
        """
        from scipy import ndimage

        factor = self.refinement_factor

        if interpolation == "nearest":
            fine_field = np.repeat(np.repeat(np.repeat(
                coarse_field, factor, axis=0), factor, axis=1), factor, axis=2)

        elif interpolation == "linear":
            fine_field = ndimage.zoom(coarse_field, factor, order=1)

        elif interpolation == "cubic":
            fine_field = ndimage.zoom(coarse_field, factor, order=3)

        else:
            fine_field = ndimage.zoom(coarse_field, factor, order=1)

        return fine_field

    def upsample(self, fine_field: np.ndarray,
                 operation: str = "average") -> np.ndarray:
        """
        Coarsen from fine to coarse grid.

        Parameters
        ----------
        fine_field : ndarray
            Field on fine grid
        operation : str
            Coarsening operation ("average", "sum", "max")

        Returns
        -------
        ndarray
            Field on coarse grid
        """
        factor = self.refinement_factor
        shape = fine_field.shape

        # Reshape for block reduction
        new_shape = (shape[0] // factor, factor,
                     shape[1] // factor, factor,
                     shape[2] // factor, factor)

        reshaped = fine_field.reshape(new_shape)

        if operation == "average":
            coarse_field = reshaped.mean(axis=(1, 3, 5))
        elif operation == "sum":
            coarse_field = reshaped.sum(axis=(1, 3, 5))
        elif operation == "max":
            coarse_field = reshaped.max(axis=(1, 3, 5))
        else:
            coarse_field = reshaped.mean(axis=(1, 3, 5))

        return coarse_field

# =============================================================================
# MULTI-SCALE SIMULATION
# =============================================================================

class MultiScaleSimulation:
    """
    A hierarchy of grid levels coupled by ScaleCoupler transfers.

    Each level carries a 3-D field advected with a donor-cell (upwind)
    scheme; finer levels are sub-cycled so that every level satisfies
    the same CFL condition locally:

        dt_level = dt_coarsest / refinement_factor^(level)

    After each coarse step the levels exchange data: the coarse level
    provides boundary/interior values to the finer level (downsample),
    and the finer level feeds its averaged state back (upsample with
    mass conservation via the 'sum' operation on extensive quantities).
    """

    def __init__(self, levels: List[ScaleLevel],
                 coarse_shape: Tuple[int, int, int] = (16, 16, 16),
                 refinement_factor: int = 2,
                 n_levels: int = 2,
                 velocity: Optional[np.ndarray] = None,
                 cfl: float = 0.3):
        """
        Args:
            levels: scale levels from coarse to fine (length n_levels)
            coarse_shape: grid shape of the coarsest level
            refinement_factor: resolution ratio between levels
            n_levels: number of levels in the hierarchy
            velocity: constant advection velocity (3,) in cells/step
                of the coarse grid
            cfl: CFL number for the sub-cycled time steps
        """
        self.levels = levels
        self.rf = int(refinement_factor)
        self.n_levels = int(n_levels)
        self.cfl = cfl
        self.velocity = (np.zeros(3) if velocity is None
                         else np.asarray(velocity, dtype=float))

        # per-level fields and couplers
        self.fields: List[np.ndarray] = []
        shape = tuple(coarse_shape)
        for _ in range(self.n_levels):
            self.fields.append(np.zeros(shape, dtype=float))
            shape = tuple(s * self.rf for s in shape)

        self.couplers = [
            ScaleCoupler(levels[i], levels[i + 1], self.rf)
            for i in range(self.n_levels - 1)
        ]

        self.time = 0.0
        self.step_counter = 0
        self.n_substeps = self.rf ** (self.n_levels - 1)

    # ------------------------------------------------------------------
    def set_initial_condition(self, field_coarse: np.ndarray) -> None:
        """Set the coarsest-level field; finer levels are filled by
        cell repetition (np.repeat), which preserves the mass and the
        first moment exactly - unlike spline interpolation, which is
        not periodic-aware and shifts the centre of mass when the
        field has power near the grid edges."""
        self.fields[0] = np.asarray(field_coarse, dtype=float)
        for i in range(self.n_levels - 1):
            f = self.fields[i]
            fine = f
            for axis in range(3):
                fine = np.repeat(fine, self.rf, axis=axis)
            self.fields[i + 1] = fine

    # ------------------------------------------------------------------
    @staticmethod
    def _upwind_step(field: np.ndarray, velocity: np.ndarray,
                     dt_cells: float) -> np.ndarray:
        """
        One donor-cell advection step on a unit-spaced periodic grid.
        dt_cells is the step in units of cells (velocity is cells/time).

        Upwind (donor-cell) discretisation of df/dt + v df/dx = 0:

            v > 0:  f_i^{n+1} = f_i - v dt (f_i     - f_{i-1})
            v < 0:  f_i^{n+1} = f_i - v dt (f_{i+1} - f_i    )

        Both branches are the *backward*/forward difference taken on
        the upwind side, and both are TVD for |v| dt <= 1.
        """
        f = field
        for axis, v in enumerate(velocity):
            if v == 0.0:
                continue
            # FIX(audit C2): for v < 0 the donor-cell difference is
            # (f_{i+1} - f_i), not (f_i - f_{i+1}).  The old code used
            # `f - src` in both branches, which flips the sign of the
            # gradient for v<0 and makes the scheme anti-diffusive
            # (a delta function grew to max|f| = 1.85e6 in 40 steps at
            # v = -0.5, dt = 0.5).
            if v > 0.0:
                upwind = np.roll(f, 1, axis=axis)    # f_{i-1}
                dfdx = f - upwind
            else:
                downwind = np.roll(f, -1, axis=axis)  # f_{i+1}
                dfdx = downwind - f
            f = f - v * dt_cells * dfdx
        return f

    # ------------------------------------------------------------------
    def step_coarse(self) -> None:
        """
        Advance the hierarchy by one coarse step, sub-cycling finer
        levels and exchanging data at every sub-step boundary.
        """
        dt_c = self.cfl / max(np.max(np.abs(self.velocity)), 1e-12)
        dt_c = min(dt_c, 0.5)

        # advance the coarsest level once
        self.fields[0] = self._upwind_step(self.fields[0],
                                           self.velocity, dt_c)

        # sub-cycle the finer levels: one coarse cell spans rf^i cells
        # of level i, so the velocity in level-i cells is v*rf^i while
        # the sub-step is dt_c/rf^i - identical per-cell CFL everywhere
        if self.n_levels > 1:
            for i in range(1, self.n_levels):
                n_sub = self.rf ** i
                v_i = self.velocity * self.rf ** i
                dt_i = dt_c / n_sub
                for _ in range(n_sub):
                    self.fields[i] = self._upwind_step(
                        self.fields[i], v_i, dt_i)

            # feedback finest -> coarsest, synchronised at the coarse
            # step boundary; conserve mass by rescaling to the sum the
            # upwind coarse solution already produced
            coarse_from_fine = self.fields[self.n_levels - 1]
            for i in range(self.n_levels - 2, -1, -1):
                coarse_from_fine = self.couplers[i].upsample(
                    coarse_from_fine, operation='average')
            src_sum = self.fields[0].sum()
            new_sum = coarse_from_fine.sum()
            if new_sum > 0:
                # AUDIT-FLAG (H2, NOT FIXED - outside this worker's
                # scope): this explicit rescale FORCES the reported
                # `mass_error` to ~1e-16.  It is an algebraic identity,
                # not a property of the scheme, and must not be quoted
                # as validation of conservation (Appendix E does).
                coarse_from_fine *= src_sum / new_sum
            self.fields[0] = coarse_from_fine

        self.time += dt_c
        self.step_counter += 1

    # ------------------------------------------------------------------
    def run(self, n_coarse_steps: int = 10) -> Dict[str, Any]:
        """Run the hierarchy for a number of coarse steps."""
        mass0 = self.fields[0].sum()
        for _ in range(n_coarse_steps):
            self.step_coarse()
        return {
            'time': self.time,
            'steps': self.step_counter,
            'mass_initial': float(mass0),
            'mass_final': float(self.fields[0].sum()),
            'mass_error': float((self.fields[0].sum() - mass0) / mass0)
            if mass0 != 0 else 0.0,
        }


# =============================================================================
# HIERARCHICAL REFINEMENT
# =============================================================================

class HierarchicalRefinement:
    """
    Adaptive refinement map from a physical resolution criterion.

    Uses the Truelove et al. (1997) Jeans-resolution rule: a cell must
    be smaller than a quarter of the local Jeans length,

        lambda_J = c_s sqrt(pi / (G rho)),

    so the required refinement level at a cell is

        level = ceil(log2( (N_J * dx_base) / lambda_J ))
                clipped to [0, max_level]

    i.e. refinement is demanded where the *base* cell is too coarse
    (dx_base > lambda_J / N_J).  Halving dx `level` times then gives
    dx_base / 2**level <= lambda_J / N_J, which is precisely the
    criterion.  Cells that already resolve the Jeans length get
    level 0.  (Before the audit fix the reciprocal ratio was used,
    so the criterion was inverted - see FIX(audit C1).)
    """

    G = 6.674e-8           # cm^3 g^-1 s^-2

    def __init__(self, sound_speed: float, dx: float,
                 n_jeans: float = 4.0, max_level: int = 4):
        """
        Args:
            sound_speed: sound speed (cm/s) assumed uniform
            dx: base-grid cell size (cm)
            n_jeans: required number of cells per Jeans length
            max_level: maximum refinement level (0 = base grid)
        """
        self.c_s = float(sound_speed)
        self.dx = float(dx)
        self.n_jeans = float(n_jeans)
        self.max_level = int(max_level)

    # ------------------------------------------------------------------
    def level_for_density(self, density: np.ndarray) -> np.ndarray:
        """
        Required refinement level per cell for a density field.

        Args:
            density: number density (cm^-3) or mass density (g/cm^3);
                assumed mass density with mu=2.33 if values are large
                enough that interpretation matters - pass mass density.

        Returns:
            integer level array, same shape, in [0, max_level]
        """
        rho = np.asarray(density, dtype=float)
        lambda_J = self.c_s * np.sqrt(np.pi / (self.G * rho))

        # FIX(audit C1): Truelove criterion was inverted - the required
        # level is set by how many times the base cell must be halved to
        # satisfy dx <= lambda_J / N_J, i.e. ratio = N_J*dx / lambda_J,
        # not its reciprocal.
        # dx needed = lambda_J / n_jeans; each level halves dx
        ratio = (self.n_jeans * self.dx) / lambda_J
        level = np.ceil(np.log2(np.maximum(ratio, 1.0)))
        return np.clip(level, 0, self.max_level).astype(int)

    # ------------------------------------------------------------------
    def refinement_map(self, density: np.ndarray) -> Dict[str, Any]:
        """
        Full refinement statistics for a density field.

        Returns:
            dict with 'levels', 'n_cells', 'n_refined',
            'fraction_refined', 'effective_resolution',
            'max_jeans_ratio'
        """
        rho = np.asarray(density, dtype=float)
        levels = self.level_for_density(rho)
        lambda_J = self.c_s * np.sqrt(np.pi / (self.G * rho))
        jeans_ratio = lambda_J / self.dx
        return {
            'levels': levels,
            'n_cells': int(rho.size),
            'n_refined': int((levels > 0).sum()),
            'fraction_refined': float((levels > 0).mean()),
            'effective_resolution': {
                f"level_{lv}": int(self.dx / 2 ** lv)
                for lv in range(int(levels.max()) + 1)
            },
            'max_jeans_ratio': float(jeans_ratio.max()),
        }

    # ------------------------------------------------------------------
    def build_refined_regions(self, density: np.ndarray) \
            -> List[Dict[str, Any]]:
        """
        List the refined regions as level, mask pairs (one entry per
        level present above the base grid).
        """
        levels = self.level_for_density(density)
        out = []
        for lv in range(1, int(levels.max()) + 1):
            mask = levels == lv
            if mask.any():
                out.append({
                    'level': lv,
                    'dx': self.dx / 2 ** lv,
                    'mask': mask,
                    'n_cells': int(mask.sum()),
                })
        return out
