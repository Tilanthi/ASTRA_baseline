#!/usr/bin/env python3
"""
Gravitational Collapse Module
=============================

Analytic and semi-analytic tools for the gravitational collapse of
molecular clouds and cores.

Capabilities:
1. Jeans analysis (thermal, turbulent, and effective support)
2. Virial analysis (Bertoldi & McKee 1992 virial parameter, energy budget)
3. Pressure-free (homologous) free-fall collapse integration
4. Fragmentation criteria (Jeans, Bonnor-Ebert, critical column density)
5. Accretion rates (Bondi-Hoyle, Shu inside-out, free-fall)

Key References:
- Jeans 1902
- Bonnor 1956; Ebert 1955
- Bertoldi & McKee 1992, ApJ, 395, 140
- Shu 1977, ApJ, 214, 488
- Bondi 1952, MNRAS, 112, 195
- Krumholz & McKee 2005, ApJ, 630, 250 (fragmentation criterion)

Author: Claude Code (ASTRA)
Date: 2026-08
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

# Physical Constants (CGS)
G = 6.674e-8            # Gravitational constant (cm^3 / g / s^2)
k_B = 1.381e-16         # Boltzmann constant (erg / K)
m_H = 1.674e-24         # Hydrogen atom mass (g)
mu = 2.33               # Mean molecular weight per particle in molecular gas
pc = 3.086e18           # Parsec (cm)
M_sun = 1.989e33        # Solar mass (g)
year = 3.156e7          # Year (s)


# =============================================================================
# JEANS ANALYSIS
# =============================================================================

class JeansAnalysis:
    """
    Jeans (gravitational versus pressure support) analysis.

    Thermal Jeans length:

        lambda_J = c_s sqrt(pi / (G rho))

    Jeans mass (uniform sphere of diameter lambda_J):

        M_J = (pi^(5/2) / 6) c_s^3 / (G^(3/2) rho^(1/2))

    Turbulent and magnetic support enter through an effective sound speed

        c_eff^2 = c_s^2 + sigma_turb^2 + v_A^2

    All input velocities in cm/s, densities in g/cm^3 (or particles/cm^3
    with `number_density=True` and molecular weight `mu_particle`).
    """

    def __init__(self, number_density: bool = True, mu_particle: float = 2.33):
        self.use_number_density = number_density
        self.mu_particle = mu_particle

    # ---------------------------------------------------------------- helpers
    def _mass_density(self, n_or_rho: float) -> float:
        """Convert input density to mass density rho (g/cm^3)."""
        if self.use_number_density:
            return n_or_rho * self.mu_particle * m_H
        return n_or_rho

    def sound_speed(self, temperature: float) -> float:
        """Isothermal sound speed c_s = sqrt(k_B T / (mu m_H)) in cm/s."""
        return np.sqrt(k_B * temperature / (self.mu_particle * m_H))

    # ---------------------------------------------------------------- scalars
    def jeans_length(self, density: float, temperature: float = 10.0,
                     turbulence: float = 0.0, alfven: float = 0.0) -> float:
        """
        Jeans length in cm. `turbulence` and `alfven` are 1-D velocity
        dispersions (cm/s) added in quadrature to the thermal sound speed.
        """
        rho = self._mass_density(density)
        c_eff = np.sqrt(self.sound_speed(temperature) ** 2
                        + turbulence ** 2 + alfven ** 2)
        return c_eff * np.sqrt(np.pi / (G * rho))

    def jeans_mass(self, density: float, temperature: float = 10.0,
                   turbulence: float = 0.0, alfven: float = 0.0) -> float:
        """Jeans mass in g."""
        rho = self._mass_density(density)
        c_eff = np.sqrt(self.sound_speed(temperature) ** 2
                        + turbulence ** 2 + alfven ** 2)
        return (np.pi ** 2.5 / 6.0) * c_eff ** 3 / (G ** 1.5 * np.sqrt(rho))

    def jeans_number_density(self, radius_pc: float, temperature: float = 10.0,
                             turbulence: float = 0.0) -> float:
        """
        Number density at which a region of the given radius becomes
        Jeans unstable (M > M_J for a sphere of this radius).
        """
        radius = radius_pc * pc
        c_eff = np.sqrt(self.sound_speed(temperature) ** 2 + turbulence ** 2)
        # M_sphere > M_J  =>  rho > 15 c_eff^2 / (4 pi G R^2)
        rho_crit = 15.0 * c_eff ** 2 / (4.0 * np.pi * G * radius ** 2)
        return rho_crit / (self.mu_particle * m_H)

    # ---------------------------------------------------------------- summary
    def analyze(self, density: float, temperature: float = 10.0,
                turbulence: float = 0.0, size_pc: Optional[float] = None,
                mass_msun: Optional[float] = None) -> Dict[str, Any]:
        """
        Full Jeans analysis of a region.

        Args:
            density: number density (cm^-3) or mass density (g/cm^3)
            temperature: kinetic temperature (K)
            turbulence: non-thermal velocity dispersion (cm/s)
            size_pc: region diameter (pc); if given with mass, the
                Jeans ratio M/M_J is reported
            mass_msun: region mass (Msun)

        Returns:
            Dictionary with lambda_J, M_J, t_ff and (if size and mass
            given) the stability verdict.
        """
        rho = self._mass_density(density)
        result = {
            'jeans_length_pc': self.jeans_length(density, temperature,
                                                 turbulence) / pc,
            'jeans_mass_msun': self.jeans_mass(density, temperature,
                                               turbulence) / M_sun,
            'jeans_length_thermal_pc': self.jeans_length(density,
                                                         temperature) / pc,
            'jeans_mass_thermal_msun': self.jeans_mass(density,
                                                       temperature) / M_sun,
            'sound_speed_cm_s': self.sound_speed(temperature),
            'freefall_time_yr': np.sqrt(3.0 * np.pi / (32.0 * G * rho)) / year,
        }
        if size_pc is not None and mass_msun is not None:
            m_jeans = self.jeans_mass(density, temperature, turbulence) / M_sun
            result['mass_msun'] = mass_msun
            result['size_pc'] = size_pc
            result['jeans_ratio'] = mass_msun / m_jeans
            result['jeans_unstable'] = bool(mass_msun > m_jeans)
            result['resolved_jeans_length'] = bool(
                size_pc * pc > self.jeans_length(density, temperature,
                                                 turbulence))
        return result


# =============================================================================
# VIRIAL ANALYSIS
# =============================================================================

@dataclass
class VirialTerms:
    """Individual energy terms of the scalar virial theorem (erg)."""
    gravitational: float = 0.0
    thermal: float = 0.0
    turbulent: float = 0.0
    magnetic: float = 0.0
    external_pressure: float = 0.0
    total: float = 0.0


class VirialAnalysis:
    """
    Scalar virial analysis of a self-gravitating cloud.

    The virial parameter (Bertoldi & McKee 1992) for a spherical cloud

        alpha = 5 sigma^2 R / (G M)

    where sigma^2 = c_s^2 + sigma_NT^2 is the total (thermal + non-thermal)
    1-D velocity dispersion. alpha < 2 roughly indicates a cloud bound by
    self-gravity (for a rho ~ r^-1 sphere, alpha = 2 M_VIR/M).

    Energy budget (Mac Low 1999; McKee & Zweibel 1992):

        W   = -3/5 a G M^2 / R          (a = 1 uniform, 1.5 r^-1)
        E_th  = 3/2 M c_s^2
        E_turb= 3/2 M sigma_NT^2
        E_mag = (1 - epsilon_M) B^2 R^3 / 6   (epsilon_M ~ 1/3 mass-to-flux)
        S     = 4 pi R^3 P_ext
    """

    STRUCTURE_FACTORS = {'uniform': 1.0, 'r^-1': 1.5, 'r^-2': 2.5}

    def __init__(self, structure: str = 'uniform'):
        if structure not in self.STRUCTURE_FACTORS:
            raise ValueError(f"structure must be one of "
                             f"{list(self.STRUCTURE_FACTORS)}")
        self.structure = structure
        self.a = self.STRUCTURE_FACTORS[structure]

    def virial_parameter(self, mass_msun: float, radius_pc: float,
                         temperature: float = 10.0,
                         sigma_nt_km_s: float = 0.0,
                         mu_particle: float = 2.33) -> float:
        """Dimensionless virial parameter alpha = 5 sigma^2 R / (G M)."""
        M = mass_msun * M_sun
        R = radius_pc * pc
        c_s = np.sqrt(k_B * temperature / (mu_particle * m_H))
        sigma = np.sqrt(c_s ** 2 + (sigma_nt_km_s * 1e5) ** 2)
        return 5.0 * sigma ** 2 * R / (G * M)

    def virial_mass(self, radius_pc: float, temperature: float = 10.0,
                    sigma_nt_km_s: float = 0.0,
                    mu_particle: float = 2.33) -> float:
        """
        Virial mass (Msun) assuming alpha = 2 (marginal binding):

            M_vir = 5 sigma^2 R / (2 G)
        """
        R = radius_pc * pc
        c_s = np.sqrt(k_B * temperature / (mu_particle * m_H))
        sigma = np.sqrt(c_s ** 2 + (sigma_nt_km_s * 1e5) ** 2)
        return 5.0 * sigma ** 2 * R / (2.0 * G) / M_sun

    def energy_budget(self, mass_msun: float, radius_pc: float,
                      temperature: float = 10.0, sigma_nt_km_s: float = 0.0,
                      b_field_microG: float = 0.0,
                      p_ext_cgs: float = 0.0,
                      mu_particle: float = 2.33) -> VirialTerms:
        """All scalar virial energy terms in erg."""
        M = mass_msun * M_sun
        R = radius_pc * pc
        c_s = np.sqrt(k_B * temperature / (mu_particle * m_H))
        sigma_nt = sigma_nt_km_s * 1e5
        B = b_field_microG * 1e-6  # Gauss

        W = -3.0 / 5.0 * self.a * G * M ** 2 / R
        E_th = 1.5 * M * c_s ** 2
        E_turb = 1.5 * M * sigma_nt ** 2
        # Mean-square field within the cloud; (1 - eps_M) accounts for the
        # supporting (non-turbulent) fraction (McKee & Zweibel 1992).
        E_mag = (1.0 - 1.0 / 3.0) * B ** 2 * R ** 3 / 6.0
        S = 4.0 * np.pi * R ** 3 * p_ext_cgs

        return VirialTerms(gravitational=W, thermal=E_th, turbulent=E_turb,
                           magnetic=E_mag, external_pressure=S,
                           total=W + E_th + E_turb + E_mag + S)

    def analyze(self, mass_msun: float, radius_pc: float,
                temperature: float = 10.0, sigma_nt_km_s: float = 0.0,
                b_field_microG: float = 0.0, p_ext_cgs: float = 0.0) -> Dict[str, Any]:
        """Full virial summary including the binding verdict."""
        alpha = self.virial_parameter(mass_msun, radius_pc, temperature,
                                      sigma_nt_km_s)
        terms = self.energy_budget(mass_msun, radius_pc, temperature,
                                   sigma_nt_km_s, b_field_microG, p_ext_cgs)
        return {
            'virial_parameter': alpha,
            'virial_mass_msun': self.virial_mass(radius_pc, temperature,
                                                 sigma_nt_km_s),
            'gravitationally_bound': bool(terms.total < 0),
            'virial_equilibrium': bool(abs(2.0 * (terms.thermal
                                                  + terms.turbulent
                                                  + terms.magnetic
                                                  + terms.external_pressure)
                                           + terms.gravitational)
                                       < 0.2 * abs(terms.gravitational)),
            'energy_terms_erg': {
                'gravitational': terms.gravitational,
                'thermal': terms.thermal,
                'turbulent': terms.turbulent,
                'magnetic': terms.magnetic,
                'external_pressure': terms.external_pressure,
            },
            'structure_factor_a': self.a,
        }


# =============================================================================
# FREE-FALL COLLAPSE
# =============================================================================

class FreefallCollapse:
    """
    Pressure-free (homologous) gravitational collapse.

    The exact solution for a uniform-density sphere collapsing under its
    own gravity obeys

        d rho / dt = sqrt(32 pi G / 3) rho^(3/2)   (after free-fall onset)

    with the free-fall time

        t_ff = sqrt(3 pi / (32 G rho_0)).

    The ODE integrates to the closed-form solution
    rho(t) = rho_0 (1 - t/t_ff)^(-2), used directly (the singularity at
    t = t_ff truncates physical application).
    """

    def __init__(self, initial_number_density: float = 1e4,
                 mu_particle: float = 2.33):
        self.rho0 = initial_number_density * mu_particle * m_H
        self.mu_particle = mu_particle
        self.t_ff = np.sqrt(3.0 * np.pi / (32.0 * G * self.rho0))

    def _drho_dt(self, t, rho):
        return np.sqrt(32.0 * np.pi * G / 3.0) * rho ** 1.5

    def density_at_time(self, time_years: np.ndarray,
                        final_density_ratio: float = 1e6) -> np.ndarray:
        """
        Density (g/cm^3) at each time. Times beyond t_ff return nan.
        """
        t = np.atleast_1d(np.asarray(time_years, dtype=float)) * year
        # The ODE integrates to the closed-form solution
        # rho(t) = rho0 (1 - t/t_ff)^(-2); use it directly.
        out = np.full_like(t, np.nan, dtype=float)
        mask = t < self.t_ff
        out[mask] = self.rho0 * (1.0 - t[mask] / self.t_ff) ** (-2)
        return out

    def number_density_at_time(self, time_years) -> np.ndarray:
        """Number density n (cm^-3) at each time."""
        return self.density_at_time(time_years) / (self.mu_particle * m_H)

    def analyze(self, final_density_ratio: float = 1e6,
                n_samples: int = 100) -> Dict[str, Any]:
        """
        Collapse history from rho0 to rho0 * final_density_ratio.

        Returns dict with times, densities, t_ff, central time to reach
        each stage, and the instant collapse rate.
        """
        t_stop = self.t_ff * (1.0 - final_density_ratio ** -0.5)
        times = np.linspace(0.0, t_stop, n_samples)
        rho = self.rho0 * (1.0 - times / self.t_ff) ** (-2)
        # n at each time and d(log n)/dt
        n = rho / (self.mu_particle * m_H)
        dlog = np.gradient(np.log10(n), times / year)
        return {
            't_ff_yr': self.t_ff / year,
            'times_yr': times / year,
            'number_density': n,
            'density_g_cm3': rho,
            'dlog10n_dt_per_yr': dlog,
            'final_n_cm3': n[-1],
        }

    def collapse_time_to_ratio(self, density_ratio: float) -> float:
        """Time (yr) for density to grow by `density_ratio` from rho0."""
        if density_ratio < 1.0:
            raise ValueError("density_ratio must be >= 1")
        return self.t_ff * (1.0 - density_ratio ** -0.5) / year


# =============================================================================
# FRAGMENTATION CRITERION
# =============================================================================

class FragmentationCriterion:
    """
    Criteria for gravitational fragmentation of a cloud / core / disk.

    Implemented criteria:
    1. Jeans: M > M_J (thermal or turbulent Jeans mass)
    2. Bonnor-Ebert: M > M_BE = 1.182 sigma_T^3 / (G^1.5 rho_S^0.5)
       (isothermal sphere at the critical surface density)
    3. Critical column density: the Bonnor-Ebert column
       Sigma_BE = 1.57 sigma_T sqrt(rho_S / G) (~107 Msun/pc^2 at
       n_S = 1e4 cm^-3, T = 10 K), i.e. the observed ~100 Msun/pc^2
       threshold for self-gravitating molecular gas.
    4. Disk fragmentation (Toomre Q < 1 with cooling ts < 3 Omega^-1)
    """

    def __init__(self, mu_particle: float = 2.33):
        self.mu = mu_particle

    def bonnor_ebert_mass(self, surface_number_density: float) -> float:
        """
        Critical Bonnor-Ebert mass (g) for external (edge) number density
        n_S (cm^-3):

            M_BE = 1.182 sigma_T^3 / (G^1.5 rho_S^0.5)

        with sigma_T = c_s (10 K isothermal).
        """
        rho_S = surface_number_density * self.mu * m_H
        c_s = np.sqrt(k_B * 10.0 / (self.mu * m_H))
        return 1.182 * c_s ** 3 / (G ** 1.5 * np.sqrt(rho_S))

    def critical_column_density(self, edge_number_density: float,
                                temperature: float = 10.0,
                                velocity_dispersion: float = None) -> float:
        """
        Critical column density Sigma_crit (g/cm^2) for self-gravity,
        derived from the critical Bonnor-Ebert sphere:

            Sigma_BE = M_BE / (pi R_BE^2)
                     = 1.57 sigma_T sqrt(rho_S / G)

        where rho_S is the edge (external) density. Evaluates to
        ~107 Msun/pc^2 for n_S = 1e4 cm^-3, T = 10 K, consistent with
        the observed ~100 Msun/pc^2 threshold for self-gravitating
        molecular gas (Lada et al. 2010).
        """
        if velocity_dispersion is None:
            velocity_dispersion = np.sqrt(k_B * temperature
                                          / (self.mu * m_H))
        rho_S = edge_number_density * self.mu * m_H
        return 1.57 * velocity_dispersion * np.sqrt(rho_S / G)

    def jeans_fragmentation(self, mass_msun: float, density: float,
                            temperature: float = 10.0,
                            turbulence_km_s: float = 0.0) -> Dict[str, Any]:
        """Jeans mass-size fragmentation verdict for a core."""
        analyzer = JeansAnalysis()
        c_s = analyzer.sound_speed(temperature)
        sigma_eff = np.sqrt(c_s ** 2 + (turbulence_km_s * 1e5) ** 2)
        rho = density * self.mu * m_H
        m_jeans = (np.pi ** 2.5 / 6.0) * sigma_eff ** 3 / (G ** 1.5
                                                           * np.sqrt(rho))
        return {
            'criterion': 'jeans',
            'jeans_mass_msun': m_jeans / M_sun,
            'mass_msun': mass_msun,
            'fragments': bool(mass_msun * M_sun > m_jeans),
            'ratio': mass_msun * M_sun / m_jeans,
        }

    def toomre(self, surface_density: float, temperature: float,
               omega: float, radius: float) -> Dict[str, Any]:
        """
        Toomre Q for a rotating disk:

            Q = c_s Omega / (pi G Sigma)

        Args:
            surface_density: disk surface density (g/cm^2)
            temperature: midplane temperature (K)
            omega: angular frequency (rad/s)
            radius: radius (cm) -- reported only

        Fragmentation requires Q < 1 (and rapid cooling).
        """
        c_s = np.sqrt(k_B * temperature / (self.mu * m_H))
        Q = c_s * omega / (np.pi * G * surface_density)
        return {
            'criterion': 'toomre',
            'Q': Q,
            'gravitationally_unstable': bool(Q < 1.0),
            'critical_surface_density': c_s * omega / (np.pi * G),
            'radius_cm': radius,
        }

    def evaluate(self, mass_msun: float, density: float,
                 temperature: float = 10.0,
                 turbulence_km_s: float = 0.0) -> Dict[str, Any]:
        """Combined fragmentation report (Jeans + BE + column density)."""
        jeans = self.jeans_fragmentation(mass_msun, density, temperature,
                                         turbulence_km_s)
        v_disp = turbulence_km_s * 1e5 if turbulence_km_s > 0 else None
        sigma_crit = self.critical_column_density(density, temperature,
                                                  v_disp)
        return {
            **jeans,
            'bonnor_ebert_mass_msun': self.bonnor_ebert_mass(
                density) / M_sun,
            'critical_column_density_g_cm2': sigma_crit,
            'critical_column_density_msun_pc2': sigma_crit * (pc ** 2) / M_sun,
        }


# =============================================================================
# ACCRETION RATES
# =============================================================================

class AccretionRates:
    """
    Analytic accretion-rate models.

    1. Bondi / Bondi-Hoyle (Bondi 1952):

           Mdot = 4 pi lambda G^2 M^2 rho / (v^2 + c_s^2)^(3/2)

       with lambda = e^(3/2)/4 = 1.12 for the standard Bondi solution.

    2. Shu (1977) inside-out collapse:

           Mdot = 0.975 c_s^3 / G   (constant, singular isothermal sphere)

    3. Free-fall accretion onto a central object from a uniform core:

           Mdot_ff(t) = dM(< r_ff)/dt computed from the collapse solution.
    """

    def __init__(self, mu_particle: float = 2.33):
        self.mu = mu_particle

    def bondi_hoyle(self, mass_msun: float, number_density: float,
                    sound_speed: float, relative_velocity: float = 0.0,
                    lambda_bondi: float = 1.12) -> float:
        """
        Bondi-Hoyle accretion rate (g/s).

        Args:
            mass_msun: central mass
            number_density: ambient density (cm^-3)
            sound_speed: ambient sound speed (cm/s)
            relative_velocity: object-gas bulk velocity (cm/s)
            lambda_bondi: Bondi eigenvalue (1.12 standard; 4 for rotation)
        """
        M = mass_msun * M_sun
        rho = number_density * self.mu * m_H
        denom = (relative_velocity ** 2 + sound_speed ** 2) ** 1.5
        return 4.0 * np.pi * lambda_bondi * G ** 2 * M ** 2 * rho / denom

    def shu_inside_out(self, temperature: float = 10.0) -> float:
        """Shu (1977) inside-out collapse accretion rate (g/s)."""
        c_s = np.sqrt(k_B * temperature / (self.mu * m_H))
        return 0.975 * c_s ** 3 / G

    def shu_collapse_time(self, mass_envelope_msun: float,
                          temperature: float = 10.0) -> float:
        """Time (yr) for the inside-out front to consume the envelope."""
        return mass_envelope_msun * M_sun / self.shu_inside_out(temperature) / year

    def freefall_accretion(self, mass_msun: float, radius_pc: float) -> float:
        """
        Instantaneous free-fall accretion rate (g/s) if the gas at
        radius R loses all pressure support:

            Mdot ~ M / t_ff
        """
        rho = 3.0 * mass_msun * M_sun / (4.0 * np.pi * (radius_pc * pc) ** 3)
        t_ff = np.sqrt(3.0 * np.pi / (32.0 * G * rho))
        return mass_msun * M_sun / t_ff

    def analyze(self, mass_msun: float, number_density: float,
                temperature: float = 10.0,
                relative_velocity_km_s: float = 0.0,
                radius_pc: float = 0.1) -> Dict[str, float]:
        """All accretion models (g/s and Msun/yr)."""
        c_s = np.sqrt(k_B * temperature / (self.mu * m_H))
        bh = self.bondi_hoyle(mass_msun, number_density, c_s,
                              relative_velocity_km_s * 1e5)
        shu = self.shu_inside_out(temperature)
        ff = self.freefall_accretion(mass_msun, radius_pc)
        return {
            'bondi_hoyle_g_s': bh,
            'bondi_hoyle_msun_yr': bh * year / M_sun,
            'shu_g_s': shu,
            'shu_msun_yr': shu * year / M_sun,
            'freefall_g_s': ff,
            'freefall_msun_yr': ff * year / M_sun,
            'sound_speed_cm_s': c_s,
        }


# =============================================================================
# FACTORIES
# =============================================================================

def get_jeans_analyzer(mu_particle: float = 2.33) -> JeansAnalysis:
    """Factory: Jeans analysis with mean molecular weight mu."""
    return JeansAnalysis(number_density=True, mu_particle=mu_particle)


def get_virial_analyzer(structure: str = 'uniform') -> VirialAnalysis:
    """Factory: virial analysis for a chosen density structure."""
    return VirialAnalysis(structure=structure)


def get_jeans_analysis() -> JeansAnalysis:
    """Alias factory kept for backward compatibility."""
    return get_jeans_analyzer()
