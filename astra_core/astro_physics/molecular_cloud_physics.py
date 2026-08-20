
"""
Documentation for multi_scale_inference module.

This module provides multi_scale_inference capabilities for STAN.
Enhanced through self-evolution cycle 84.
"""

#!/usr/bin/env python3
"""
Molecular Cloud and Dust Physics Module for ASTRO-SWARM
========================================================

Comprehensive physics for analyzing galactic molecular clouds and
interstellar dust, integrated with stigmergic swarm intelligence.

Physical Models:
1. Molecular Line Spectroscopy (CO, HCN, N2H+, NH3, CS, etc.)
2. Dust Emission and Extinction
3. Column Density and Mass Estimation
4. Temperature and Density Structure
5. Turbulence and Kinematics
6. Magnetic Field Tracers
7. Chemical Abundances

Key References:
- Hildebrand 1983 (dust opacity)
- Ossenkopf & Henning 1994 (dust models)
- Kauffmann et al. 2008 (mass-size relation)
- Lada et al. 2010 (star formation thresholds)
- Planck Collaboration (dust properties)

Author: Claude Code (ASTRO-SWARM)
Date: 2024-11
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from abc import ABC, abstractmethod
import warnings

# Physical Constants (CGS)
k_B = 1.38e-16          # Boltzmann constant (erg/K)
h_planck = 6.626e-27    # Planck constant (erg s)
c_light = 2.998e10      # Speed of light (cm/s)
m_H = 1.67e-24          # Hydrogen mass (g)
m_H2 = 2 * m_H          # H2 mass (g)
G_cgs = 6.67e-8         # Gravitational constant
pc_to_cm = 3.086e18     # parsec to cm
Msun_to_g = 1.989e33    # Solar mass to grams
AU_to_cm = 1.496e13     # AU to cm
Jy_to_cgs = 1e-23       # Jansky to erg/s/cm²/Hz


# =============================================================================
# MOLECULAR LINE DATABASE
# =============================================================================

@dataclass
class MolecularTransition:
    """Properties of a molecular line transition"""
    molecule: str
    transition: str          # e.g., "J=1-0", "J=2-1"
    rest_frequency: float    # GHz
    upper_energy: float      # K (E_u/k_B)
    einstein_A: float        # s⁻¹
    critical_density: float  # cm⁻³
    abundance_typical: float # X = n(mol)/n(H2)
    tracer_of: str          # What physical condition it traces
    optical_depth_typical: str  # "thin", "moderate", "thick"


class MolecularLineDatabase:
    """
    Database of molecular transitions for ISM spectroscopy

    Organized by molecule with key transitions for cloud analysis.
    """

    TRANSITIONS = {
        # Carbon Monoxide - most common tracer
        "12CO_1-0": MolecularTransition(
            molecule="12CO", transition="J=1-0",
            rest_frequency=115.271, upper_energy=5.53,
            einstein_A=7.203e-8, critical_density=2.2e3,
            abundance_typical=1e-4, tracer_of="total_molecular_gas",
            optical_depth_typical="thick"
        ),
        "12CO_2-1": MolecularTransition(
            molecule="12CO", transition="J=2-1",
            rest_frequency=230.538, upper_energy=16.60,
            einstein_A=6.910e-7, critical_density=1.1e4,
            abundance_typical=1e-4, tracer_of="warm_molecular_gas",
            optical_depth_typical="thick"
        ),
        "13CO_1-0": MolecularTransition(
            molecule="13CO", transition="J=1-0",
            rest_frequency=110.201, upper_energy=5.29,
            einstein_A=6.294e-8, critical_density=2.0e3,
            abundance_typical=1.4e-6, tracer_of="column_density",
            optical_depth_typical="moderate"
        ),
        "C18O_1-0": MolecularTransition(
            molecule="C18O", transition="J=1-0",
            rest_frequency=109.782, upper_energy=5.27,
            einstein_A=6.266e-8, critical_density=2.0e3,
            abundance_typical=1.7e-7, tracer_of="column_density_optically_thin",
            optical_depth_typical="thin"
        ),

        # Dense gas tracers
        "HCN_1-0": MolecularTransition(
            molecule="HCN", transition="J=1-0",
            rest_frequency=88.632, upper_energy=4.25,
            einstein_A=2.407e-5, critical_density=2.6e6,
            abundance_typical=2e-8, tracer_of="dense_gas",
            optical_depth_typical="moderate"
        ),
        "HCO+_1-0": MolecularTransition(
            molecule="HCO+", transition="J=1-0",
            rest_frequency=89.189, upper_energy=4.28,
            einstein_A=4.187e-5, critical_density=1.6e5,
            abundance_typical=1e-9, tracer_of="dense_ionized_gas",
            optical_depth_typical="moderate"
        ),
        "N2H+_1-0": MolecularTransition(
            molecule="N2H+", transition="J=1-0",
            rest_frequency=93.173, upper_energy=4.47,
            einstein_A=3.628e-5, critical_density=1.4e5,
            abundance_typical=5e-10, tracer_of="cold_dense_cores",
            optical_depth_typical="thin"
        ),
        "CS_2-1": MolecularTransition(
            molecule="CS", transition="J=2-1",
            rest_frequency=97.981, upper_energy=7.05,
            einstein_A=1.679e-5, critical_density=4.3e5,
            abundance_typical=1e-9, tracer_of="dense_gas",
            optical_depth_typical="moderate"
        ),

        # Temperature tracers
        "NH3_1,1": MolecularTransition(
            molecule="NH3", transition="(1,1)",
            rest_frequency=23.694, upper_energy=23.4,
            einstein_A=1.712e-7, critical_density=1.8e3,
            abundance_typical=3e-8, tracer_of="kinetic_temperature",
            optical_depth_typical="moderate"
        ),
        "NH3_2,2": MolecularTransition(
            molecule="NH3", transition="(2,2)",
            rest_frequency=23.723, upper_energy=64.9,
            einstein_A=2.239e-7, critical_density=2.1e3,
            abundance_typical=3e-8, tracer_of="kinetic_temperature",
            optical_depth_typical="moderate"
        ),

        # Shock/outflow tracers
        "SiO_2-1": MolecularTransition(
            molecule="SiO", transition="J=2-1",
            rest_frequency=86.847, upper_energy=6.25,
            einstein_A=2.927e-5, critical_density=3.4e5,
            abundance_typical=1e-12, tracer_of="shocks_outflows",
            optical_depth_typical="thin"
        ),
        "CH3OH_2-1": MolecularTransition(
            molecule="CH3OH", transition="2(0)-1(0)A+",
            rest_frequency=96.741, upper_energy=12.5,
            einstein_A=3.407e-6, critical_density=1.1e5,
            abundance_typical=1e-9, tracer_of="hot_cores_shocks",
            optical_depth_typical="thin"
        ),

        # Atomic fine structure
        "CI_1-0": MolecularTransition(
            molecule="CI", transition="3P1-3P0",
            rest_frequency=492.161, upper_energy=23.6,
            einstein_A=7.880e-8, critical_density=5e2,
            abundance_typical=1e-5, tracer_of="PDR_interface",
            optical_depth_typical="moderate"
        ),
        "CII_158um": MolecularTransition(
            molecule="CII", transition="2P3/2-2P1/2",
            rest_frequency=1900.537, upper_energy=91.2,
            einstein_A=2.300e-6, critical_density=2.8e3,
            abundance_typical=1.4e-4, tracer_of="PDR_ionized_carbon",
            optical_depth_typical="thick"
        ),
    }

    @classmethod
    def get_transition(cls, name: str) -> MolecularTransition:
        """Get transition by name"""
        return cls.TRANSITIONS.get(name)

    @classmethod
    def get_dense_gas_tracers(cls) -> List[MolecularTransition]:
        """Get transitions that trace dense gas (n > 10⁴ cm⁻³)"""
        return [t for t in cls.TRANSITIONS.values()
                if t.critical_density > 1e4]

    @classmethod
    def get_temperature_tracers(cls) -> List[MolecularTransition]:
        """Get transitions useful for temperature measurement"""
        return [t for t in cls.TRANSITIONS.values()
                if t.tracer_of == "kinetic_temperature"]


# =============================================================================
# DUST MODELS
# =============================================================================

class DustModel(Enum):
    """Standard dust models for the ISM"""
    MRN = "mrn"                    # Mathis, Rumpl, Nordsieck 1977
    WD01 = "wd01"                  # Weingartner & Draine 2001
    OSSENKOPF_THIN = "oh94_thin"   # Ossenkopf & Henning 1994 (thin ice)
    OSSENKOPF_THICK = "oh94_thick" # Ossenkopf & Henning 1994 (thick ice)
    PLANCK = "planck"              # Planck all-sky dust model


@dataclass
class DustProperties:
    """
    Physical properties of interstellar dust

    Key quantities:
    - κ_ν: Dust opacity (cm²/g of dust)
    - β: Spectral index of dust emissivity
    - Gas-to-dust ratio
    - Size distribution parameters
    """
    model: str

    # Opacity at reference wavelength
    kappa_ref: float        # cm²/g at reference λ
    lambda_ref: float       # Reference wavelength (μm)

    # Spectral index
    beta: float             # κ_ν ∝ ν^β
    beta_uncertainty: float

    # Gas-to-dust ratio
    gas_to_dust: float      # M_gas / M_dust

    # Temperature range of applicability
    T_min: float            # K
    T_max: float            # K

    # Size distribution (MRN: dn/da ∝ a^(-3.5))
    a_min: float            # Minimum grain size (μm)
    a_max: float            # Maximum grain size (μm)
    size_index: float       # Power law index

    # Composition
    silicate_fraction: float  # Mass fraction silicates
    carbon_fraction: float    # Mass fraction carbonaceous

    def kappa_nu(self, wavelength_um: float) -> float:
        """
        Dust opacity at given wavelength

        κ_ν = κ_ref × (λ_ref / λ)^β
        """
        return self.kappa_ref * (self.lambda_ref / wavelength_um) ** self.beta

    def kappa_nu_frequency(self, freq_GHz: float) -> float:
        """Dust opacity at given frequency"""
        wavelength_um = c_light / (freq_GHz * 1e9) * 1e4  # cm to μm
        return self.kappa_nu(wavelength_um)


class DustModelLibrary:
    """
    Library of standard dust models for different environments
    """

    MODELS = {
        DustModel.MRN: DustProperties(
            model="MRN (Mathis+ 1977)",
            kappa_ref=10.0, lambda_ref=250.0,
            beta=2.0, beta_uncertainty=0.2,
            gas_to_dust=100.0,
            T_min=10, T_max=50,
            a_min=0.005, a_max=0.25, size_index=-3.5,
            silicate_fraction=0.53, carbon_fraction=0.47
        ),
        DustModel.WD01: DustProperties(
            model="Weingartner & Draine 2001",
            kappa_ref=4.0, lambda_ref=250.0,
            beta=1.8, beta_uncertainty=0.15,
            gas_to_dust=124.0,
            T_min=15, T_max=100,
            a_min=0.00035, a_max=0.25, size_index=-3.5,
            silicate_fraction=0.55, carbon_fraction=0.45
        ),
        DustModel.OSSENKOPF_THIN: DustProperties(
            model="Ossenkopf & Henning 1994 (thin ice)",
            kappa_ref=5.0, lambda_ref=250.0,
            beta=1.8, beta_uncertainty=0.1,
            gas_to_dust=100.0,
            T_min=10, T_max=30,
            a_min=0.005, a_max=0.25, size_index=-3.5,
            silicate_fraction=0.50, carbon_fraction=0.30
        ),
        DustModel.OSSENKOPF_THICK: DustProperties(
            model="Ossenkopf & Henning 1994 (thick ice)",
            # AUDIT-FLAG (B-MC, "suspicious"): 1.85 cm^2/g is the value
            # most often quoted for OH94 thick-ice grains at 850 um, not
            # at 1300 um.  If that is what was meant, lambda_ref should
            # be 850 and kappa(850) would be 1.85 rather than the 3.50
            # this table gives - a factor 1.89 at mm wavelengths.  NOT
            # changed: unlike the Planck entry (B-MC-5) there is no
            # in-code statement of intent to settle it, and OH94 do
            # tabulate kappa at 1.3 mm as well.  Resolve against the
            # OH94 table before using this model quantitatively.
            kappa_ref=1.85, lambda_ref=1300.0,  # 1.3mm reference
            beta=1.5, beta_uncertainty=0.1,
            gas_to_dust=100.0,
            T_min=10, T_max=25,
            a_min=0.01, a_max=1.0, size_index=-3.0,  # Grain growth
            silicate_fraction=0.40, carbon_fraction=0.25
        ),
        DustModel.PLANCK: DustProperties(
            model="Planck Collaboration",
            # FIX(audit B-MC-5): `lambda_ref` is declared in MICRONS
            # (see DustProperties) but held 353, the *frequency* in GHz
            # named in the trailing comment.  kappa_nu therefore treated
            # 353 as a wavelength, making kappa(850 um) 0.22158 instead
            # of the defining 0.92 cm^2/g - an error of 4.152x - while
            # kappa_nu(353 um) spuriously returned exactly 0.92.
            # 353 GHz = c/353 GHz = 849.3 um, so the reference
            # wavelength is 850 um.
            kappa_ref=0.92, lambda_ref=850.0,  # 0.92 cm^2/g at 850 um (353 GHz)
            beta=1.62, beta_uncertainty=0.10,
            gas_to_dust=136.0,
            T_min=14, T_max=30,
            a_min=0.001, a_max=0.5, size_index=-3.5,
            silicate_fraction=0.50, carbon_fraction=0.50
        ),
    }

    @classmethod
    def get_model(cls, model: DustModel) -> DustProperties:
        return cls.MODELS[model]

    @classmethod
    def get_dense_cloud_model(cls) -> DustProperties:
        """Get model appropriate for dense molecular clouds"""
        return cls.MODELS[DustModel.OSSENKOPF_THICK]

    @classmethod
    def get_diffuse_ism_model(cls) -> DustProperties:
        """Get model appropriate for diffuse ISM"""
        return cls.MODELS[DustModel.WD01]


# =============================================================================
# MOLECULAR CLOUD PHYSICS ENGINE
# =============================================================================

@dataclass
class CloudSpectralLine:
    """Observed spectral line properties"""
    transition: str
    peak_temperature: float      # K (T_mb or T_A*)
    velocity_centroid: float     # km/s (LSR)
    line_width: float            # km/s (FWHM)
    integrated_intensity: float  # K km/s
    rms_noise: float            # K
    optical_depth: Optional[float] = None
    excitation_temp: Optional[float] = None


@dataclass
class DustSED:
    """Observed dust spectral energy distribution"""
    wavelengths: np.ndarray      # μm
    fluxes: np.ndarray           # Jy
    flux_errors: np.ndarray      # Jy
    beam_sizes: np.ndarray       # arcsec (FWHM)
    distance_pc: float           # Distance to source


@dataclass
class MolecularCloudProperties:
    """Derived physical properties of a molecular cloud"""
    # Basic properties
    name: str
    distance_pc: float

    # Size and structure
    angular_size_arcmin: float
    physical_size_pc: float
    aspect_ratio: float

    # Mass estimates
    mass_dust_msun: float
    mass_virial_msun: float
    mass_lte_msun: float
    mass_uncertainty_factor: float

    # Column density
    N_H2_peak: float             # cm⁻²
    N_H2_mean: float             # cm⁻²
    A_V_peak: float              # mag

    # Temperature
    T_dust: float                # K
    T_dust_uncertainty: float    # K
    T_kinetic: float             # K (from NH3 or other)
    T_excitation: float          # K (from CO)

    # Density
    n_H2_mean: float             # cm⁻³
    n_H2_peak: float             # cm⁻³
    volume_filling_factor: float

    # Kinematics
    v_lsr: float                 # km/s
    sigma_v: float               # km/s (1D velocity dispersion)
    sigma_nt: float              # km/s (non-thermal)
    mach_number: float           # σ_nt / c_s

    # Stability
    virial_parameter: float      # α_vir = 2K/|W|
    jeans_mass: float            # M_J in M_sun
    bonnor_ebert_mass: float     # M_BE in M_sun
    is_gravitationally_bound: bool

    # Chemistry
    CO_abundance: float          # X_CO = N(CO)/N(H2)
    depletion_factor: float      # CO depletion onto grains
    ionization_fraction: float   # n(e)/n(H2)

    # Star formation
    dense_gas_fraction: float    # M(n>10⁴)/M_total
    star_formation_rate: float   # M_sun/yr
    star_formation_efficiency: float

    # Magnetic field (if measured)
    B_field_strength: Optional[float] = None  # μG
    mass_to_flux_ratio: Optional[float] = None

    # Metadata
    analysis_method: str = ""
    references: List[str] = field(default_factory=list)


class MolecularCloudPhysicsEngine:
    """
    Physics engine for molecular cloud analysis

    Provides forward models and inference methods for:
    1. Spectral line analysis
    2. Dust continuum analysis
    3. Mass estimation
    4. Stability analysis
    5. Star formation diagnostics
    """

    # Standard conversion factors
    X_CO = 2.0e20  # cm⁻² / (K km/s) - Milky Way average
    N_H2_to_AV = 9.4e20  # cm⁻² per mag (Bohlin+ 1978)

    def __init__(self, dust_model: DustModel = DustModel.OSSENKOPF_THICK):
        self.dust = DustModelLibrary.get_model(dust_model)
        self.lines = MolecularLineDatabase()

    # =========================================================================
    # SPECTRAL LINE ANALYSIS
    # =========================================================================

    def analyze_co_isotopologues(self,
                                  co12: CloudSpectralLine,
                                  co13: CloudSpectralLine,
                                  c18o: Optional[CloudSpectralLine] = None
                                  ) -> Dict[str, float]:
        """
        Analyze CO isotopologue ratios to get optical depth and column density

        Method: Use 12CO/13CO ratio to estimate τ(12CO), then derive N(H2)

        Returns:
            Dictionary with optical depth, excitation temp, column density
        """
        # Standard isotope ratios (Galactocentric gradient)
        R_12_13 = 77.0  # 12C/13C ratio (local ISM)
        R_16_18 = 560.0  # 16O/18O ratio

        # Estimate optical depth from line ratio
        # T_12 / T_13 = (1 - exp(-τ_12)) / (1 - exp(-τ_13))
        # With τ_13 = τ_12 / R

        T_ratio = co12.peak_temperature / (co13.peak_temperature + 1e-10)

        # Iterative solution for τ_12
        tau_12 = self._solve_optical_depth(T_ratio, R_12_13)

        # Excitation temperature from 12CO (assuming fills beam)
        T_ex = self._brightness_to_tex(co12.peak_temperature,
                                        self.lines.get_transition("12CO_1-0").rest_frequency)

        # Column density from 13CO (optically thinner)
        tau_13 = tau_12 / R_12_13
        N_13CO = self._column_density_lte(
            co13.integrated_intensity,
            T_ex,
            self.lines.get_transition("13CO_1-0"),
            tau_13
        )

        # Convert to H2 column density
        X_13CO = 1.4e-6  # 13CO/H2 abundance
        N_H2 = N_13CO / X_13CO

        # If C18O available, use for better constraint
        if c18o is not None:
            N_C18O = self._column_density_lte(
                c18o.integrated_intensity,
                T_ex,
                self.lines.get_transition("C18O_1-0"),
                optical_depth=0.1  # Assume optically thin
            )
            X_C18O = 1.7e-7
            N_H2_c18o = N_C18O / X_C18O
            # Average the two estimates
            N_H2 = (N_H2 + N_H2_c18o) / 2

        return {
            'tau_12CO': tau_12,
            'tau_13CO': tau_13,
            'T_ex': T_ex,
            'N_13CO': N_13CO,
            'N_H2': N_H2,
            'A_V': N_H2 / self.N_H2_to_AV
        }

    def analyze_nh3_temperature(self,
                                nh3_11: CloudSpectralLine,
                                nh3_22: CloudSpectralLine) -> Dict[str, float]:
        """
        Derive kinetic temperature from NH3 (1,1) and (2,2) lines

        The rotational temperature is derived from the line ratio,
        then converted to kinetic temperature.

        Returns:
            Dictionary with T_rot, T_kin, optical depth, column density
        """
        # Energy difference between (2,2) and (1,1) levels
        Delta_E = 41.5  # K

        # Line ratio gives rotational temperature
        # Assuming both lines optically thin
        ratio = nh3_22.integrated_intensity / (nh3_11.integrated_intensity + 1e-10)

        # T_rot from Boltzmann equation
        # Includes statistical weights g(2,2)/g(1,1) = 5/3
        #
        # FIX(audit B-MC-4): for ratio >= 5/3 the argument of the log
        # reaches 1, T_rot flips sign, and the old `max(T_rot, 8.0)`
        # clamp turned that into a silent 8.0 K - a x49 discontinuity
        # (ratio 1.50 -> 393.9 K, ratio 1.67 -> 8.0 K).  A ratio at or
        # above the statistical-weight limit means the (2,2)/(1,1)
        # populations are inverted and no LTE T_rot exists; say so.
        x = ratio * 3.0 / 5.0
        if not np.isfinite(x) or x <= 0.0 or x >= 1.0:
            warnings.warn(
                f"NH3 (2,2)/(1,1) ratio {ratio:.3f} implies "
                "I(2,2)*3/5 >= I(1,1): the level populations are "
                "inverted and no LTE rotational temperature exists.",
                RuntimeWarning, stacklevel=2)
            T_rot = float('nan')
        else:
            T_rot = -Delta_E / np.log(x)

        # Convert to kinetic temperature (Ho & Townes 1983; the form
        # and constants below follow Tafalla et al. 2004, A&A 416, 191)
        #
        #   T_kin = T_rot / (1 - (T_rot/T_0) ln[1 + 1.1 exp(-15.7/T_rot)])
        #
        # FIX(audit B-MC-3): the denominator vanishes at T_rot = 67.3 K
        # and is NEGATIVE above it, and there was no guard: an observed
        # ratio of only 0.90 gave T_kin = -44834 K, and ratio 1.00 gave
        # -331.65 K.  The approximation itself is calibrated for
        # T_kin <~ 20-25 K, so beyond the pole it carries no
        # information: return NaN and warn rather than a negative
        # temperature.
        T_0 = Delta_E          # 41.5 K, consistent with the E_u table
        if np.isnan(T_rot):
            T_kin = float('nan')
            denom = float('nan')
        else:
            denom = 1.0 - (T_rot / T_0) * np.log1p(
                1.1 * np.exp(-15.7 / T_rot))
            if denom <= 0.0:
                warnings.warn(
                    f"NH3 T_rot = {T_rot:.1f} K is at or beyond the pole "
                    "of the Ho & Townes / Tafalla T_rot->T_kin relation "
                    "(T_rot ~ 66 K); T_kin is undefined there.",
                    RuntimeWarning, stacklevel=2)
                T_kin = float('nan')
            else:
                T_kin = T_rot / denom

        # Optical depth from hyperfine structure (if resolved)
        # AUDIT-FLAG (B-MC-7): tau_11 is hardcoded to 1.0 even though
        # T_rot above assumes both lines are optically thin; the two
        # assumptions are mutually inconsistent (a fixed 1.58x factor
        # on N_NH3).  Left as-is: recovering tau needs the hyperfine
        # satellite ratios, which this API does not receive.
        tau_11 = 1.0

        # Column density
        # AUDIT-FLAG: the 1.6e13 coefficient could not be reconciled
        # with a standard NH3(1,1) column-density formula and is left
        # unchanged, unverified.
        if np.isnan(T_rot):
            N_NH3 = float('nan')
        else:
            N_NH3 = 1.6e13 * T_rot * nh3_11.integrated_intensity / \
                (1 - np.exp(-tau_11))

        return {
            'T_rot': T_rot,
            'T_kin': T_kin,
            'tau_11': tau_11,
            'N_NH3': N_NH3,
            'N_H2': N_NH3 / 3e-8  # Assuming standard abundance
        }

    def analyze_dense_gas(self,
                          hcn: CloudSpectralLine,
                          hcop: Optional[CloudSpectralLine] = None,
                          n2hp: Optional[CloudSpectralLine] = None) -> Dict[str, float]:
        """
        Analyze dense gas tracers (HCN, HCO+, N2H+)

        These molecules have high critical densities and trace
        gas suitable for star formation (n > 10⁴ cm⁻³).
        """
        results = {}

        # HCN analysis
        hcn_trans = self.lines.get_transition("HCN_1-0")

        # Integrated intensity gives dense gas mass proxy
        # L_HCN ∝ M_dense (Gao & Solomon 2004)
        results['I_HCN'] = hcn.integrated_intensity
        results['n_crit_HCN'] = hcn_trans.critical_density

        if hcop is not None:
            # HCN/HCO+ ratio sensitive to ionization and chemistry
            ratio_hcn_hcop = hcn.integrated_intensity / (hcop.integrated_intensity + 1e-10)
            results['HCN_HCO+_ratio'] = ratio_hcn_hcop

            # High ratio (>1) suggests XDR or high CR ionization
            # Low ratio (<1) suggests PDR or high ionization fraction
            if ratio_hcn_hcop > 2:
                results['chemistry_indicator'] = "XDR_or_high_CR"
            elif ratio_hcn_hcop < 0.5:
                results['chemistry_indicator'] = "PDR_dominated"
            else:
                results['chemistry_indicator'] = "normal"

        if n2hp is not None:
            # N2H+ survives in cold, CO-depleted cores
            results['I_N2H+'] = n2hp.integrated_intensity

            # N2H+/HCN ratio traces CO depletion
            ratio_n2hp_hcn = n2hp.integrated_intensity / (hcn.integrated_intensity + 1e-10)
            results['N2H+_HCN_ratio'] = ratio_n2hp_hcn

            if ratio_n2hp_hcn > 1:
                results['CO_depletion'] = "significant"
            else:
                results['CO_depletion'] = "moderate"

        return results

    # =========================================================================
    # DUST CONTINUUM ANALYSIS
    # =========================================================================

    def fit_dust_sed(self, sed: DustSED,
                     T_range: Tuple[float, float] = (8, 50),
                     beta_range: Tuple[float, float] = (1.0, 2.5)
                     ) -> Dict[str, float]:
        """
        Fit modified blackbody to dust SED

        S_ν = Ω × B_ν(T_d) × (1 - exp(-τ_ν))

        For optically thin: S_ν ∝ ν^β × B_ν(T_d)

        The physical (optically thin) model actually fitted is

            S_nu = M_dust kappa_nu B_nu(T_d) / D^2

        with kappa_nu = kappa_ref (lambda_ref/lambda)^beta taken from
        the module's dust model but with `beta` free.  Fitted
        parameters are (T_d, beta, M_dust); the gas mass follows from
        the model's gas-to-dust ratio and the beam-averaged H2 column
        from N(H2) = M_gas / (mu_H2 m_H Omega D^2).

        FIX(audit C9 / B-MC-2): the file was truncated inside this
        function.  It imported `curve_fit`, never called it, and fell
        off the end returning `None` while the docstring promised
        T_dust/beta/column density/mass.

        Assumes the fluxes have been convolved to a common resolution;
        the largest supplied beam is used for the solid angle.

        Returns:
            dict with T_dust, beta, M_dust, M_gas, N_H2, their
            1-sigma errors, reduced chi2 and `success`.
        """
        from scipy.optimize import curve_fit

        wl_um = np.atleast_1d(np.asarray(sed.wavelengths, dtype=float))
        flux_jy = np.atleast_1d(np.asarray(sed.fluxes, dtype=float))
        flux_err = np.atleast_1d(np.asarray(sed.flux_errors, dtype=float))

        if wl_um.size < 3:
            raise ValueError(
                "fit_dust_sed needs at least 3 photometric points to "
                f"constrain (T_d, beta, mass); got {wl_um.size}.")
        if np.any(flux_err <= 0):
            raise ValueError("fit_dust_sed requires positive flux errors")

        # Convert to frequency
        freq_hz = c_light / (wl_um * 1e-4)  # μm to cm to Hz

        distance_cm = sed.distance_pc * pc_to_cm

        # Modified blackbody model (optically thin), S_nu in Jy
        def mod_bb(nu, T_d, beta, m_dust_msun):
            lam_um = c_light / np.asarray(nu, dtype=float) * 1e4
            kappa = self.dust.kappa_ref * \
                (self.dust.lambda_ref / lam_um) ** beta      # cm^2/g
            B_nu = self._planck_function(nu, T_d)            # cgs
            s_cgs = (m_dust_msun * Msun_to_g) * kappa * B_nu / distance_cm ** 2
            return s_cgs / Jy_to_cgs                          # Jy

        # Initial guess: 15 K, the model's own beta, mass scaled from
        # the longest-wavelength point at that temperature.
        t0 = float(np.clip(15.0, *T_range))
        b0 = float(np.clip(self.dust.beta, *beta_range))
        i_long = int(np.argmax(wl_um))
        unit_flux = float(mod_bb(freq_hz[i_long], t0, b0, 1.0))
        m0 = flux_jy[i_long] / unit_flux if unit_flux > 0 else 1.0
        m0 = max(m0, 1e-12)

        bounds = ([T_range[0], beta_range[0], 0.0],
                  [T_range[1], beta_range[1], np.inf])

        try:
            popt, pcov = curve_fit(
                mod_bb, freq_hz, flux_jy, p0=[t0, b0, m0],
                sigma=flux_err, absolute_sigma=True,
                bounds=bounds, maxfev=20000)
        except Exception as exc:                      # pragma: no cover
            raise RuntimeError(f"dust SED fit failed: {exc}") from exc

        t_dust, beta_fit, m_dust = (float(v) for v in popt)
        perr = np.sqrt(np.diag(pcov))

        resid = (mod_bb(freq_hz, *popt) - flux_jy) / flux_err
        dof = max(wl_um.size - 3, 1)
        chi2_red = float(np.sum(resid ** 2) / dof)

        m_gas = m_dust * self.dust.gas_to_dust

        # Beam-averaged H2 column.  Omega for a Gaussian beam of
        # FWHM theta is pi theta^2 / (4 ln 2).
        theta_arcsec = float(np.max(np.atleast_1d(sed.beam_sizes)))
        theta_rad = theta_arcsec / 206265.0
        omega_sr = np.pi * theta_rad ** 2 / (4.0 * np.log(2.0))
        area_cm2 = omega_sr * distance_cm ** 2
        mu_h2 = 2.8            # mean mass per H2 in units of m_H
        n_h2 = (m_gas * Msun_to_g) / (mu_h2 * m_H * area_cm2)

        return {
            'T_dust': t_dust,
            'T_dust_err': float(perr[0]),
            'beta': beta_fit,
            'beta_err': float(perr[1]),
            'M_dust': m_dust,
            'M_dust_err': float(perr[2]),
            'M_gas': m_gas,
            'N_H2': float(n_h2),
            'A_V': float(n_h2 / self.N_H2_to_AV),
            'chi2_reduced': chi2_red,
            'beam_arcsec': theta_arcsec,
            'success': True,
        }

    # =========================================================================
    # LOW-LEVEL HELPERS
    #
    # FIX(audit C9 / B-MC-1): `_solve_optical_depth`, `_brightness_to_tex`,
    # `_column_density_lte` and `_planck_function` were called by
    # `analyze_co_isotopologues` and `fit_dust_sed` but never defined -
    # the file ended mid-function at line 640, so every call raised
    # AttributeError.  They are re-implemented here from first
    # principles and validated in
    # astra_core/tests/test_physics_regressions.py.
    # =========================================================================

    @staticmethod
    def _planck_function(nu, temperature: float):
        """
        Planck function B_nu(T) in erg s^-1 cm^-2 Hz^-1 sr^-1.

            B_nu = (2 h nu^3 / c^2) / (exp(h nu / k T) - 1)
        """
        nu = np.asarray(nu, dtype=float)
        if temperature <= 0:
            return np.zeros_like(nu)
        x = h_planck * nu / (k_B * float(temperature))
        # expm1 keeps the Rayleigh-Jeans limit accurate for x << 1
        return 2.0 * h_planck * nu ** 3 / c_light ** 2 / np.expm1(x)

    @staticmethod
    def _j_nu(temperature: float, frequency_ghz: float) -> float:
        """Radiation temperature J(T) = (h nu/k)/(exp(h nu/kT)-1), in K."""
        if temperature <= 0:
            return 0.0
        hnu_k = h_planck * frequency_ghz * 1e9 / k_B
        return float(hnu_k / np.expm1(hnu_k / float(temperature)))

    @staticmethod
    def _solve_optical_depth(t_ratio: float, isotope_ratio: float,
                             tau_max: float = 100.0) -> float:
        """
        Optical depth of the abundant isotopologue from the peak
        brightness ratio of two lines that share T_ex:

            T_12 / T_13 = (1 - exp(-tau_12)) / (1 - exp(-tau_12/R))

        The right-hand side falls monotonically from R (tau -> 0) to 1
        (tau -> inf), so the inversion is unique for 1 < T_ratio < R.
        Outside that range the data are consistent with the thin
        (T_ratio >= R) or saturated (T_ratio <= 1) limit and 0 or
        `tau_max` is returned.
        """
        from scipy.optimize import brentq

        r = float(isotope_ratio)
        ratio = float(t_ratio)
        if r <= 1.0:
            raise ValueError("isotope_ratio must exceed 1")
        if ratio >= r:
            return 0.0
        if ratio <= 1.0:
            return float(tau_max)

        def f(tau):
            return (-np.expm1(-tau)) / (-np.expm1(-tau / r)) - ratio

        if f(tau_max) > 0:
            return float(tau_max)
        return float(brentq(f, 1e-8, tau_max, xtol=1e-10, rtol=8.9e-16))

    def _brightness_to_tex(self, peak_temperature: float,
                           frequency_ghz: float,
                           t_bg: float = 2.725) -> float:
        """
        Excitation temperature of an optically thick, beam-filling
        line from its peak main-beam brightness temperature:

            T_mb = [J(T_ex) - J(T_bg)](1 - e^-tau) -> J(T_ex) - J(T_bg)
            T_ex = (h nu/k) / ln[1 + (h nu/k)/(T_mb + J(T_bg))]

        i.e. the standard 12CO thermometer (Pineda et al. 2008).
        """
        hnu_k = h_planck * frequency_ghz * 1e9 / k_B
        j_bg = self._j_nu(t_bg, frequency_ghz)
        denom = float(peak_temperature) + j_bg
        if denom <= 0:
            raise ValueError("peak_temperature + J(T_bg) must be positive")
        return float(hnu_k / np.log1p(hnu_k / denom))

    def _column_density_lte(self, integrated_intensity: float,
                            t_ex: float,
                            transition: MolecularTransition,
                            optical_depth: float = 0.0,
                            t_bg: float = 2.725) -> float:
        """
        Total LTE column density (cm^-2) of a linear rotor from the
        integrated intensity of one of its J -> J-1 lines.

            N_tot = (8 pi nu^3 / (c^3 A_ul)) (Q/g_u) exp(E_u/k T_ex)
                    [exp(h nu/k T_ex) - 1]^-1
                    Int T_R dv / [J(T_ex) - J(T_bg)]
                    * tau / (1 - e^-tau)

        (Mangum & Shirley 2015 eq. 80 plus the usual opacity
        correction, which tends to 1 as tau -> 0.)  The rotational
        constant is recovered from the line itself, B = nu_J/(2J), and
        Q_rot is summed explicitly; g_u = 2 J_u + 1.

        Only linear-rotor "J=Ju-Jl" transitions are supported - NH3 and
        [C II] need their own partition functions and are rejected
        rather than silently mis-evaluated.
        """
        label = transition.transition.replace(" ", "")
        if not label.startswith("J="):
            raise NotImplementedError(
                "_column_density_lte supports linear-rotor J transitions "
                f"only; got '{transition.transition}' for "
                f"{transition.molecule}.")
        try:
            j_up, j_low = (int(v) for v in label[2:].split("-"))
        except ValueError as exc:
            raise NotImplementedError(
                "cannot parse rotational quantum numbers from "
                f"'{transition.transition}'") from exc
        if j_low != j_up - 1:
            raise NotImplementedError(
                "only dipole J -> J-1 transitions are supported")

        nu = transition.rest_frequency * 1e9            # Hz
        b_hz = nu / (2.0 * j_up)                        # rigid-rotor B
        g_u = 2 * j_up + 1

        # Rotational partition function, sum_J (2J+1) exp(-h B J(J+1)/kT)
        j_arr = np.arange(0, 200, dtype=float)
        e_k = (h_planck * b_hz / k_B) * j_arr * (j_arr + 1.0)
        q_rot = float(np.sum((2 * j_arr + 1.0) * np.exp(-e_k / t_ex)))

        j_ex = self._j_nu(t_ex, transition.rest_frequency)
        j_bg = self._j_nu(t_bg, transition.rest_frequency)
        bright = j_ex - j_bg
        if bright <= 0:
            raise ValueError("T_ex gives no contrast against the background")

        hnu_k = h_planck * nu / k_B
        w_cgs = float(integrated_intensity) * 1e5       # K km/s -> K cm/s

        n_tot = (8.0 * np.pi * nu ** 3 /
                 (c_light ** 3 * transition.einstein_A))
        n_tot *= q_rot / g_u
        n_tot *= np.exp(transition.upper_energy / t_ex)
        n_tot /= np.expm1(hnu_k / t_ex)
        n_tot *= w_cgs / bright

        tau = float(optical_depth)
        if tau > 1e-6:
            n_tot *= tau / (-np.expm1(-tau))

        return float(n_tot)

    # =========================================================================
    # PUBLIC ANALYTICS LOST TO THE SAME TRUNCATION
    #
    # FIX(audit C9, extension): besides the four private helpers named in
    # the audit, the truncation at line 640 also removed eight PUBLIC
    # methods that `molecular_cloud_agents.py` calls -
    # `column_density_from_submm`, `virial_mass`, `virial_parameter`,
    # `jeans_mass`, `bonnor_ebert_mass`, `mach_number`,
    # `star_formation_threshold` and `dense_gas_fraction` - so
    # StructureAgent / DustAgent / StarFormationAgent all raised
    # AttributeError.  The seven with unambiguous textbook definitions
    # are restored below and covered by
    # astra_core/tests/test_physics_regressions.py;
    # `dense_gas_fraction` raises NotImplementedError because its call
    # signature does not carry the emitting area needed to turn
    # I(HCN) [K km/s] into L'(HCN) [K km/s pc^2].
    # =========================================================================

    # Mean molecular weight per H2 molecule including helium
    # (Kauffmann et al. 2008): rho = mu_H2 m_H n(H2)
    MU_H2 = 2.8
    # Mean mass per free particle in molecular gas
    MU_PARTICLE = 2.33
    # Lada, Lombardi & Alves (2010): star formation is confined to
    # A_V >~ 8 mag (Sigma ~ 116 Msun/pc^2)
    A_V_SF_THRESHOLD = 8.0

    def sound_speed(self, temperature: float) -> float:
        """Isothermal sound speed c_s = sqrt(k T / (mu m_H)), cm/s."""
        return float(np.sqrt(k_B * temperature /
                             (self.MU_PARTICLE * m_H)))

    def virial_mass(self, radius_pc: float, sigma_v: float) -> float:
        """
        Virial mass of a uniform sphere, M_vir = 5 sigma^2 R / G.

        Args:
            radius_pc: radius (pc)
            sigma_v: one-dimensional velocity dispersion (km/s)

        Returns:
            Virial mass (M_sun).  Equivalent to the familiar
            M_vir = 210 R dV^2 with dV the FWHM linewidth
            (5/G in these units is 1162.5; 1162.5/2.355^2 = 209.7).
        """
        sigma_cgs = float(sigma_v) * 1e5
        r_cgs = float(radius_pc) * pc_to_cm
        return 5.0 * sigma_cgs ** 2 * r_cgs / G_cgs / Msun_to_g

    def virial_parameter(self, mass_msun: float, radius_pc: float,
                         sigma_v: float) -> float:
        """
        Bertoldi & McKee (1992) virial parameter

            alpha_vir = 5 sigma^2 R / (G M) = M_vir / M,

        with alpha_vir = 1 marking virial equilibrium and ~2 the
        boundary of gravitational boundedness.
        """
        if mass_msun <= 0:
            raise ValueError("mass_msun must be positive")
        return self.virial_mass(radius_pc, sigma_v) / float(mass_msun)

    def jeans_mass(self, temperature: float, n_h2: float) -> float:
        """
        Thermal Jeans mass of a uniform medium,

            M_J = (pi^(5/2)/6) c_s^3 / (G^(3/2) rho^(1/2))

        (identical to (pi/6) lambda_J^3 rho).

        Args:
            temperature: kinetic temperature (K)
            n_h2: H2 number density (cm^-3)

        Returns:
            Jeans mass (M_sun)
        """
        c_s = self.sound_speed(temperature)
        rho = float(n_h2) * self.MU_H2 * m_H
        m_j = (np.pi ** 2.5 / 6.0) * c_s ** 3 / (G_cgs ** 1.5 * np.sqrt(rho))
        return float(m_j / Msun_to_g)

    def bonnor_ebert_mass(self, temperature: float,
                          pressure_over_k: float) -> float:
        """
        Critical Bonnor-Ebert mass of a pressure-confined isothermal
        sphere,

            M_BE = 1.18 c_s^4 / (G^(3/2) P_ext^(1/2)).

        Args:
            temperature: gas temperature (K)
            pressure_over_k: external pressure expressed as P/k_B in
                K cm^-3 (the usual ISM unit; typical values 1e3-1e6).
                This is the reading implied by the callers' default of
                1e4 - as a pressure in dyne/cm^2 that would be ~1e16
                times the ISM value.

        Returns:
            Bonnor-Ebert mass (M_sun)
        """
        c_s = self.sound_speed(temperature)
        p_ext = float(pressure_over_k) * k_B          # dyne/cm^2
        if p_ext <= 0:
            raise ValueError("pressure_over_k must be positive")
        m_be = 1.18 * c_s ** 4 / (G_cgs ** 1.5 * np.sqrt(p_ext))
        return float(m_be / Msun_to_g)

    def mach_number(self, sigma_v: float, temperature: float) -> float:
        """
        One-dimensional turbulent Mach number, M = sigma_1D / c_s.

        Args:
            sigma_v: 1-D velocity dispersion (km/s)
            temperature: kinetic temperature (K)
        """
        return float(sigma_v) * 1e5 / self.sound_speed(temperature)

    def column_density_from_submm(self, flux_jy: float,
                                  wavelength_um: float,
                                  t_dust: float,
                                  beam_arcsec: float,
                                  distance_pc: float) -> Dict[str, float]:
        """
        Single-band (sub)millimetre column density and beam mass for
        optically thin dust emission (Hildebrand 1983):

            N(H2)  = S_nu / [Omega_beam mu_H2 m_H kappa_nu B_nu(T_d)]
            M_beam = S_nu D^2 / [kappa_nu B_nu(T_d)] * gas-to-dust

        with Omega_beam = pi theta^2 / (4 ln 2) for a Gaussian beam of
        FWHM theta, and kappa_nu from the engine's dust model (per gram
        of DUST, hence the gas-to-dust factor).

        Args:
            flux_jy: flux density in the beam (Jy)
            wavelength_um: observing wavelength (um)
            t_dust: assumed dust temperature (K)
            beam_arcsec: beam FWHM (arcsec)
            distance_pc: source distance (pc)
        """
        nu = c_light / (float(wavelength_um) * 1e-4)      # Hz
        kappa = self.dust.kappa_nu(float(wavelength_um))  # cm^2 / g dust
        b_nu = float(self._planck_function(np.array(nu), t_dust))
        if b_nu <= 0:
            raise ValueError("t_dust must be positive")

        s_cgs = float(flux_jy) * Jy_to_cgs                # erg/s/cm^2/Hz
        theta_rad = float(beam_arcsec) / 206265.0
        omega = np.pi * theta_rad ** 2 / (4.0 * np.log(2.0))   # sr
        d_cm = float(distance_pc) * pc_to_cm

        # Dust surface density -> gas surface density -> N(H2)
        sigma_dust = s_cgs / (omega * kappa * b_nu)       # g/cm^2 of dust
        sigma_gas = sigma_dust * self.dust.gas_to_dust
        n_h2 = sigma_gas / (self.MU_H2 * m_H)

        m_beam = sigma_gas * omega * d_cm ** 2 / Msun_to_g

        return {
            'N_H2': float(n_h2),
            'A_V': float(n_h2 / self.N_H2_to_AV),
            'M_beam': float(m_beam),
            'T_dust': float(t_dust),
            'kappa_nu': float(kappa),
            'beam_sr': float(omega),
        }

    def star_formation_threshold(self, a_v: float) -> Dict[str, Any]:
        """
        Lada, Lombardi & Alves (2010) extinction threshold for star
        formation: essentially all young stellar objects lie above
        A_V ~ 8 mag (Sigma ~ 116 Msun/pc^2, A_K ~ 0.8 mag).

        Args:
            a_v: visual extinction (mag)

        Returns:
            dict with the threshold, the ratio to it and a boolean
        """
        a_v = float(a_v)
        return {
            'A_V': a_v,
            'A_V_threshold': self.A_V_SF_THRESHOLD,
            'ratio': a_v / self.A_V_SF_THRESHOLD,
            'above_threshold': bool(a_v >= self.A_V_SF_THRESHOLD),
            'N_H2_threshold': self.A_V_SF_THRESHOLD * self.N_H2_to_AV,
            'reference': 'Lada, Lombardi & Alves 2010, ApJ 724, 687',
        }

    def dense_gas_fraction(self, mass_total_msun: float,
                           i_hcn: float,
                           distance_pc: float) -> float:
        """
        Not implemented - see FIX(audit C9, extension) above.

        The Gao & Solomon (2004) calibration is
        M_dense = alpha_HCN L'(HCN) with alpha_HCN ~ 10 Msun
        (K km/s pc^2)^-1, but converting the *intensity* I(HCN)
        [K km/s] supplied here into the *luminosity* L'(HCN)
        [K km/s pc^2] requires the emitting solid angle (a beam or
        source size), which this signature does not provide.  Rather
        than invent one, refuse.
        """
        raise NotImplementedError(
            "dense_gas_fraction needs the HCN emitting area (beam or "
            "source solid angle) to convert I(HCN) [K km/s] into "
            "L'(HCN) [K km/s pc^2] before applying the Gao & Solomon "
            "(2004) alpha_HCN ~ 10 Msun/(K km/s pc^2) calibration. "
            "Pass a beam/source size and use "
            "M_dense = 10 * I_HCN * area_pc2 explicitly.")
