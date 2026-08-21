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
Astrochemical Database Interfaces for Spectroscopy

Provides unified access to major spectroscopic databases:
- CDMS (Cologne Database for Molecular Spectroscopy)
- JPL (Jet Propulsion Laboratory Molecular Spectroscopy)
- LAMDA (Leiden Atomic and Molecular Database)
- Splatalogue (Database for Astronomical Spectroscopy)
- HITRAN (High-Resolution Transmission Molecular Absorption)

Features:
- Line transition queries by frequency, molecule, or energy
- Collision rate coefficient retrieval
- Partition function calculations
- Einstein coefficient computations
- Local caching for offline use
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union
from enum import Enum
import json
import os
import warnings
from abc import ABC, abstractmethod


# Physical constants
H_PLANCK = 6.62607015e-27  # erg s
K_BOLTZMANN = 1.380649e-16  # erg/K
C_LIGHT = 2.99792458e10  # cm/s
AMU = 1.6605390666e-24  # g
DEBYE = 1.0e-18  # esu cm per Debye


# =============================================================================
# Linear-rotor spectroscopy helpers        FIX(audit C3)
# =============================================================================
#
# Einstein A coefficient for an electric-dipole rotational transition
# J -> J-1 of a linear molecule in its ground vibrational state:
#
#     A_{J,J-1} = 64 pi^4 nu^3 / (3 h c^3) * |mu_{J,J-1}|^2
#     |mu_{J,J-1}|^2 = mu^2 * S(J -> J-1) / g_u = mu^2 * J / (2J + 1)
#
# with the Hoenl-London line strength S(J -> J-1) = J and g_u = 2J+1
# (Townes & Schawlow 1955, ch. 1 & 4; Mangum & Shirley 2015, PASP 127,
# 266, eqs. 10-11).  Evaluated with nu in Hz and mu in esu cm the
# prefactor 64 pi^4 / (3 h c^3) = 1.16397e-2, i.e. the familiar
# practical form  A = 1.1640e-11 nu_GHz^3 mu_D^2 J/(2J+1).
#
# The pre-audit code used  A = 3.497e-8 * nu_MHz^3 * J/(J+1), which is
# both dimensionally meaningless (nu in MHz) and uses the wrong
# degeneracy factor J/(J+1); it gave A(CO 1-0) = 2.678e7 s^-1 against
# the true 7.203e-8 s^-1 (3.7e14 too large).  The generators for
# HCN/HCO+/N2H+ used A0*(J/3)**3, which places the *literature 1-0*
# value at J = 3 and therefore makes A(1-0) 27x too small.

_EINSTEIN_A_PREFACTOR = 64.0 * np.pi ** 4 / (3.0 * H_PLANCK * C_LIGHT ** 3)


def linear_rotor_einstein_a(frequency_mhz: float, dipole_debye: float,
                            j_upper: int) -> float:
    """
    Einstein A coefficient (s^-1) for the J -> J-1 transition of a
    linear molecule with permanent dipole moment `dipole_debye`.

    Validated against LAMDA/CDMS reference values:
        CO   1-0  7.205e-8  (LAMDA 7.203e-8)
        CO   3-2  2.501e-6  (LAMDA 2.497e-6)
        HCN  1-0  2.407e-5  (LAMDA 2.407e-5)
        HCO+ 1-0  4.187e-5  (LAMDA 4.187e-5)
        N2H+ 1-0  3.628e-5  (LAMDA 3.628e-5)
    """
    nu_hz = float(frequency_mhz) * 1.0e6
    mu_esu = float(dipole_debye) * DEBYE
    j = float(j_upper)
    return (_EINSTEIN_A_PREFACTOR * nu_hz ** 3 * mu_esu ** 2
            * j / (2.0 * j + 1.0))


def linear_rotor_frequency(j_upper: int, b_const: float, d_const: float,
                           h_const: float = 0.0) -> float:
    """
    Rest frequency (MHz) of the J -> J-1 transition of a linear rotor,

        E(J)/h = B J(J+1) - D [J(J+1)]^2 + H [J(J+1)]^3
        nu(J -> J-1) = 2 B J - 4 D J^3 + H {[J(J+1)]^3 - [(J-1)J]^3}

    All constants in MHz.  Omitting the centrifugal-distortion term
    (the pre-audit behaviour for HCN/HCO+/N2H+) shifts the lines by
    1.1 km/s already at J = 1 and by up to ~96 km/s at J = 9.
    """
    j = float(j_upper)
    x_u = j * (j + 1.0)
    x_l = (j - 1.0) * j
    return (2.0 * b_const * j - 4.0 * d_const * j ** 3
            + h_const * (x_u ** 3 - x_l ** 3))


# Effective rotational constants (MHz) and permanent dipole moments (D).
#
# B, D, H were obtained by an unweighted least-squares fit of the
# expression above to the *measured* CDMS catalogue line lists
# (https://cdms.astro.uni-koeln.de/classic/entries/, tags c028503 CO,
# c027501 HCN, c029507 HCO+, c029506 N2H+, hyperfine-collapsed line
# centres) over exactly the J range each generator emits.  `freq_rms`
# is the resulting maximum |model - CDMS| deviation over that range and
# is used as the reported frequency uncertainty - these frequencies are
# *computed*, not retrieved, and the uncertainty must say so.
#
# Dipole moments: CO 0.11011 D (Muenter 1975, J. Mol. Spectrosc. 55,
# 490), HCN 2.985 D (Ebenstein & Muenter 1984, JCP 80, 3989),
# HCO+ 3.90 D and N2H+ 3.40 D (values adopted by LAMDA / Botschwina
# ab-initio).  Each reproduces the tabulated LAMDA A(1-0) to <0.5%.
LINEAR_ROTOR_CONSTANTS: Dict[str, Dict[str, float]] = {
    "CO":   {"B": 57635.9681,  "D": 0.18350435, "H": 1.7057e-07,
             "mu": 0.11011, "j_max": 14, "freq_rms": 0.0064},
    "HCN":  {"B": 44315.97554, "D": 0.08722134, "H": 1.1567e-07,
             "mu": 2.985,   "j_max": 9,  "freq_rms": 0.0048},
    "HCO+": {"B": 44594.42819, "D": 0.08282424, "H": 1.6751e-08,
             "mu": 3.90,    "j_max": 9,  "freq_rms": 0.0061},
    "N2H+": {"B": 46586.87449, "D": 0.08793416, "H": -2.7383e-07,
             "mu": 3.40,    "j_max": 7,  "freq_rms": 0.0044},
}

# Provenance label for every line/rate this module *computes* rather
# than retrieves.  FIX(audit rule 3): the previous code stamped
# database="CDMS"/"LAMDA" on invented A-values and collision rates.
SYNTHETIC_PROVENANCE = "synthetic"


def linear_rotor_partition_function(temperature: float, b_const: float,
                                    j_max: int = 200) -> float:
    """
    Rotational partition function of a linear molecule,

        Q_rot(T) = sum_J (2J+1) exp(-h B J(J+1) / k T)

    with `b_const` in MHz.  No nuclear-spin degeneracy factor is
    included, so this is consistent with the g_u = 2J+1 convention used
    for the transitions in this module (any spin factor cancels between
    Q and g_u in a column density).  Reproduces the classical limit
    kT/(hB) + 1/3 to <0.1% for kT >> hB.

    FIX(audit C4b): the hardcoded `Qrot` tables previously returned by
    `_get_molecule_properties` were wrong by -15% to +43% (CO: 156 at
    300 K against the correct 108.8) and directly scaled every LTE
    column density.
    """
    j = np.arange(0, int(j_max) + 1, dtype=float)
    e_k = (H_PLANCK * b_const * 1e6 / K_BOLTZMANN) * j * (j + 1.0)  # K
    return float(np.sum((2.0 * j + 1.0) * np.exp(-e_k / float(temperature))))


def _synthetic_freq_uncertainty(j_upper: int,
                                consts: Dict[str, float]) -> float:
    """
    Honest uncertainty (MHz) on a *computed* rotational frequency.

    Inside the J range the constants were fitted over this is the
    measured max |model - CDMS| residual.  Beyond it the leading
    neglected higher-order distortion term grows roughly as J^6, so the
    residual is scaled by (J/J_fit)^6 - a deliberately conservative
    bound (verified >= the actual CDMS deviation for CO up to J = 40,
    0.49 MHz, and HCN up to J = 29, 3.15 MHz).
    """
    j_fit = float(consts["j_max"])
    base = float(consts["freq_rms"])
    if j_upper <= j_fit:
        return base
    return base * (float(j_upper) / j_fit) ** 6


class DatabaseType(Enum):
    """Supported spectroscopic databases."""
    CDMS = "cdms"
    JPL = "jpl"
    LAMDA = "lamda"
    SPLATALOGUE = "splatalogue"
    HITRAN = "hitran"


@dataclass
class SpectralLine:
    """Represents a single spectral line transition."""
    frequency: float  # MHz
    frequency_uncertainty: float  # MHz
    intensity: float  # log10(nm^2 MHz) at 300K for CDMS/JPL
    einstein_a: float  # s^-1
    upper_energy: float  # cm^-1 or K
    upper_degeneracy: int
    lower_energy: float  # cm^-1 or K
    lower_degeneracy: int
    quantum_numbers_upper: str
    quantum_numbers_lower: str
    molecule: str
    isotopologue: str = ""
    database: str = ""
    tag: int = 0  # CDMS/JPL molecule tag

    @property
    def wavelength_um(self) -> float:
        """Wavelength in micrometers."""
        return C_LIGHT / (self.frequency * 1e6) * 1e4

    @property
    def wavelength_cm(self) -> float:
        """Wavelength in centimeters."""
        return C_LIGHT / (self.frequency * 1e6)

    @property
    def wavenumber(self) -> float:
        """Wavenumber in cm^-1."""
        return self.frequency * 1e6 / C_LIGHT

    def einstein_b_ul(self) -> float:
        """Einstein B coefficient for stimulated emission."""
        nu = self.frequency * 1e6  # Hz
        return self.einstein_a * C_LIGHT**2 / (2 * H_PLANCK * nu**3)

    def einstein_b_lu(self) -> float:
        """Einstein B coefficient for absorption."""
        return self.einstein_b_ul() * self.upper_degeneracy / self.lower_degeneracy


@dataclass
class CollisionPartner:
    """Collision rate data for a specific partner."""
    partner: str  # e.g., "H2", "He", "e-", "H"
    temperatures: np.ndarray  # K
    rate_coefficients: np.ndarray  # cm^3/s, shape (n_transitions, n_temps)
    transitions: List[Tuple[int, int]]  # (upper, lower) level indices


@dataclass
class MoleculeData:
    """Complete spectroscopic and collisional data for a molecule."""
    name: str
    formula: str
    mass: float  # amu
    symmetry: str  # "linear", "symmetric_top", "asymmetric_top"
    dipole_moment: float  # Debye
    energy_levels: np.ndarray  # cm^-1
    level_degeneracies: np.ndarray
    level_quantum_numbers: List[str]
    transitions: List[SpectralLine]
    collision_partners: Dict[str, CollisionPartner] = field(default_factory=dict)
    partition_function_temps: np.ndarray = field(default_factory=lambda: np.array([]))
    partition_function_values: np.ndarray = field(default_factory=lambda: np.array([]))

    def partition_function(self, temperature: float) -> float:
        """
        Interpolate partition function at given temperature.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin

        Returns
        -------
        float
            Partition function value
        """
        if len(self.partition_function_temps) == 0:
            # Calculate from energy levels
            return np.sum(self.level_degeneracies *
                         np.exp(-self.energy_levels * H_PLANCK * C_LIGHT /
                               (K_BOLTZMANN * temperature)))

        return np.interp(temperature,
                        self.partition_function_temps,
                        self.partition_function_values)

    def column_density_from_line(self, line: SpectralLine,
                                  integrated_intensity: float,
                                  temperature: float,
                                  t_background: float = 0.0) -> float:
        """
        LTE column density from an optically thin integrated intensity.

        Derivation (Goldsmith & Langer 1999, ApJ 517, 209, eq. 2;
        Mangum & Shirley 2015, PASP 127, 266, eq. 80).  For an
        optically thin line the emergent intensity integrates to

            Int I_nu dnu = (h nu / 4 pi) A_ul N_u

        and, with the Rayleigh-Jeans brightness temperature
        T_R = (c^2 / 2 k nu^2) I_nu and dnu = (nu/c) dv,

            Int I_nu dnu = (2 k nu^3 / c^3) Int T_R dv

        so that

            N_u = 8 pi k nu^2 / (h c^3 A_ul) * Int T_R dv          (*)

        and N_tot = N_u (Q/g_u) exp(E_u / k T_ex).

        FIX(audit C4): the pre-audit code used
        `8 pi nu^3 / (c^3 A) * W * 1e5`, i.e. it was missing the factor
        k/(h nu) = 1/5.532 at 115 GHz and left a residual unit of K.

        Parameters
        ----------
        line : SpectralLine
            The spectral line used
        integrated_intensity : float
            Integrated intensity Int T_R dv in K km/s
        temperature : float
            Excitation temperature in K
        t_background : float, optional
            Background temperature in K.  If > 0, the input intensity
            is taken to be background-subtracted, i.e.
            T_R = (J(T_ex) - J(T_bg))(1 - exp(-tau)), and (*) is
            multiplied by J(T_ex)/(J(T_ex) - J(T_bg)).  Default 0
            corresponds to the plain form (*).  Pass 2.725 for the
            usual observer (CMB-subtracted) convention.

        Returns
        -------
        float
            Total column density in cm^-2
        """
        nu = line.frequency * 1e6  # Hz
        Q = self.partition_function(temperature)

        # Upper level population
        g_u = line.upper_degeneracy
        E_u = line.upper_energy  # cm^-1

        w_cgs = integrated_intensity * 1e5  # K km/s -> K cm/s

        # FIX(audit C4): N_u = 8 pi k nu^2 W / (h c^3 A)
        N_u = (8.0 * np.pi * K_BOLTZMANN * nu ** 2 * w_cgs
               / (H_PLANCK * C_LIGHT ** 3 * line.einstein_a))

        if t_background > 0.0:
            hnu_k = H_PLANCK * nu / K_BOLTZMANN
            j_ex = hnu_k / np.expm1(hnu_k / temperature)
            j_bg = hnu_k / np.expm1(hnu_k / t_background)
            if j_ex <= j_bg:
                raise ValueError(
                    "Excitation temperature does not exceed the "
                    "background: the line is in absorption and the "
                    "thin-emission inversion does not apply.")
            N_u *= j_ex / (j_ex - j_bg)

        # Total column density
        N_total = N_u * Q / g_u * np.exp(E_u * H_PLANCK * C_LIGHT / (K_BOLTZMANN * temperature))

        return N_total


class SpectroscopyDatabase(ABC):
    """Abstract base class for spectroscopic databases."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize database interface.

        Parameters
        ----------
        cache_dir : str, optional
            Directory for caching downloaded data
        """
        self.cache_dir = cache_dir or os.path.expanduser("~/.astro_swarm/spectroscopy_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._molecule_cache: Dict[str, MoleculeData] = {}

    @abstractmethod
    def query_lines(self, freq_min: float, freq_max: float,
                   molecule: Optional[str] = None,
                   intensity_threshold: Optional[float] = None) -> List[SpectralLine]:
        """Query spectral lines in frequency range."""
        pass

    @abstractmethod
    def get_molecule(self, molecule: str) -> MoleculeData:
        """Get complete molecule data."""
        pass

    def _load_cache(self, key: str) -> Optional[Dict]:
        """Load cached data."""
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None

    def _save_cache(self, key: str, data: Dict):
        """Save data to cache."""
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        with open(cache_file, 'w') as f:
            json.dump(data, f)


class CDMSDatabase(SpectroscopyDatabase):
    """
    Cologne Database for Molecular Spectroscopy interface.

    CDMS contains spectroscopic data for molecules of astrophysical
    interest, particularly for radio and submillimeter astronomy.
    """

    BASE_URL = "https://cdms.astro.uni-koeln.de"

    # Common molecule tags
    MOLECULE_TAGS = {
        "CO": 28001,
        "13CO": 29001,
        "C18O": 30001,
        "C17O": 29002,
        "HCN": 27001,
        "HNC": 27002,
        "HCO+": 29003,
        "N2H+": 29004,
        "CS": 44001,
        "SO": 48001,
        "SiO": 44002,
        "H2O": 18003,
        "HDO": 19002,
        "NH3": 17002,
        "H2CO": 30004,
        "CH3OH": 32003,
        "HNCO": 43002,
        "CH3CN": 41001,
        "HC3N": 51001,
        "C2H": 25001,
        "CN": 26001,
        "NO": 30006,
        "OH": 17001,
    }

    def query_lines(self, freq_min: float, freq_max: float,
                   molecule: Optional[str] = None,
                   intensity_threshold: Optional[float] = None) -> List[SpectralLine]:
        """
        Query CDMS for spectral lines.

        Parameters
        ----------
        freq_min : float
            Minimum frequency in MHz
        freq_max : float
            Maximum frequency in MHz
        molecule : str, optional
            Molecule name to filter (e.g., "CO", "HCN")
        intensity_threshold : float, optional
            Minimum log intensity (CDMS units)

        Returns
        -------
        List[SpectralLine]
            List of matching spectral lines
        """
        lines = []

        # Check cache first
        cache_key = f"cdms_{freq_min}_{freq_max}_{molecule}"
        cached = self._load_cache(cache_key)

        if cached is not None:
            for line_data in cached:
                lines.append(SpectralLine(**line_data))
            return lines

        # In production, this would make HTTP requests to CDMS
        # Here we provide a local simulation with common lines
        lines = self._get_simulated_lines(freq_min, freq_max, molecule)

        if intensity_threshold is not None:
            lines = [l for l in lines if l.intensity >= intensity_threshold]

        # Cache results
        self._save_cache(cache_key, [vars(l) for l in lines])

        return lines

    def _get_simulated_lines(self, freq_min: float, freq_max: float,
                            molecule: Optional[str]) -> List[SpectralLine]:
        """Generate simulated line data for common molecules."""
        all_lines = []

        # CO rotational transitions
        co_lines = self._generate_co_lines()
        all_lines.extend(co_lines)

        # HCN lines
        hcn_lines = self._generate_hcn_lines()
        all_lines.extend(hcn_lines)

        # HCO+ lines
        hcop_lines = self._generate_hcop_lines()
        all_lines.extend(hcop_lines)

        # N2H+ lines
        n2hp_lines = self._generate_n2hp_lines()
        all_lines.extend(n2hp_lines)

        # Filter by frequency range
        filtered = [l for l in all_lines
                   if freq_min <= l.frequency <= freq_max]

        # Filter by molecule if specified
        if molecule is not None:
            filtered = [l for l in filtered
                       if l.molecule.upper() == molecule.upper()]

        return filtered

    def _generate_co_lines(self) -> List[SpectralLine]:
        """
        Generate the CO rotational ladder from rotational constants.

        NOTE: these lines are *computed*, not retrieved from CDMS - see
        `LINEAR_ROTOR_CONSTANTS`.  Frequencies reproduce the CDMS entry
        to <0.007 MHz, Einstein A coefficients reproduce LAMDA to <0.5%,
        but the `intensity` field remains a placeholder.
        """
        lines = []
        c = LINEAR_ROTOR_CONSTANTS["CO"]
        B, D, H = c["B"], c["D"], c["H"]

        for J in range(1, 15):
            # FIX(audit C3): explicit rigid-rotor + distortion model
            freq = linear_rotor_frequency(J, B, D, H)  # MHz
            E_upper = B * J * (J + 1) / 29979.2458  # cm^-1
            E_lower = B * (J - 1) * J / 29979.2458  # cm^-1

            # FIX(audit C3): A = 64 pi^4 nu^3 mu^2 J / (3 h c^3 (2J+1));
            # was 3.497e-8*nu_MHz^3*J/(J+1) -> 3.7e14x too large.
            A = linear_rotor_einstein_a(freq, c["mu"], J)  # s^-1

            lines.append(SpectralLine(
                frequency=freq,
                frequency_uncertainty=c["freq_rms"],
                intensity=-4.0 + 0.5 * np.log10(J),
                einstein_a=A,
                upper_energy=E_upper,
                upper_degeneracy=2 * J + 1,
                lower_energy=E_lower,
                lower_degeneracy=2 * (J - 1) + 1,
                quantum_numbers_upper=f"J={J}",
                quantum_numbers_lower=f"J={J-1}",
                molecule="CO",
                isotopologue="12C16O",
                database=SYNTHETIC_PROVENANCE,
                tag=28001
            ))

        return lines

    def _generate_hcn_lines(self) -> List[SpectralLine]:
        """
        Generate the HCN rotational ladder from rotational constants.

        Computed, not retrieved - see `_generate_co_lines`.
        """
        lines = []
        c = LINEAR_ROTOR_CONSTANTS["HCN"]
        B, D, H = c["B"], c["D"], c["H"]

        for J in range(1, 10):
            # FIX(audit C3): centrifugal distortion was omitted entirely
            # (2*B*J), shifting HCN 1-0 by 1.18 km/s and 9-8 by 96 km/s.
            freq = linear_rotor_frequency(J, B, D, H)
            E_upper = B * J * (J + 1) / 29979.2458
            E_lower = B * (J - 1) * J / 29979.2458
            # FIX(audit C3): was 2.4e-5*(J/3)**3, i.e. the literature
            # A(1-0) placed at J=3 -> A(1-0) 27x too small.
            A = linear_rotor_einstein_a(freq, c["mu"], J)

            lines.append(SpectralLine(
                frequency=freq,
                frequency_uncertainty=c["freq_rms"],
                intensity=-3.5,
                einstein_a=A,
                upper_energy=E_upper,
                upper_degeneracy=2 * J + 1,
                lower_energy=E_lower,
                lower_degeneracy=2 * (J - 1) + 1,
                quantum_numbers_upper=f"J={J}",
                quantum_numbers_lower=f"J={J-1}",
                molecule="HCN",
                isotopologue="H12C14N",
                database=SYNTHETIC_PROVENANCE,
                tag=27001
            ))

        return lines

    def _generate_hcop_lines(self) -> List[SpectralLine]:
        """
        Generate the HCO+ rotational ladder from rotational constants.

        Computed, not retrieved - see `_generate_co_lines`.
        """
        lines = []
        c = LINEAR_ROTOR_CONSTANTS["HCO+"]
        B, D, H = c["B"], c["D"], c["H"]

        for J in range(1, 10):
            # FIX(audit C3): distortion term restored (1.08 km/s at J=1)
            freq = linear_rotor_frequency(J, B, D, H)
            E_upper = B * J * (J + 1) / 29979.2458
            E_lower = B * (J - 1) * J / 29979.2458
            # FIX(audit C3): was 4.2e-5*(J/3)**3 -> A(1-0) 27x too small
            A = linear_rotor_einstein_a(freq, c["mu"], J)

            lines.append(SpectralLine(
                frequency=freq,
                frequency_uncertainty=c["freq_rms"],
                intensity=-3.8,
                einstein_a=A,
                upper_energy=E_upper,
                upper_degeneracy=2 * J + 1,
                lower_energy=E_lower,
                lower_degeneracy=2 * (J - 1) + 1,
                quantum_numbers_upper=f"J={J}",
                quantum_numbers_lower=f"J={J-1}",
                molecule="HCO+",
                isotopologue="H12C16O+",
                database=SYNTHETIC_PROVENANCE,
                tag=29003
            ))

        return lines

    def _generate_n2hp_lines(self) -> List[SpectralLine]:
        """
        Generate the N2H+ rotational ladder from rotational constants.

        Computed, not retrieved - see `_generate_co_lines`.  These are
        the hyperfine-collapsed line centres (N2H+ 1-0 is split into
        seven resolvable hyperfine groups spanning ~10 km/s).
        """
        lines = []
        c = LINEAR_ROTOR_CONSTANTS["N2H+"]
        B, D, H = c["B"], c["D"], c["H"]

        for J in range(1, 8):
            # FIX(audit C3): distortion term restored (1.08 km/s at J=1)
            freq = linear_rotor_frequency(J, B, D, H)
            E_upper = B * J * (J + 1) / 29979.2458
            E_lower = B * (J - 1) * J / 29979.2458
            # FIX(audit C3): was 3.6e-5*(J/3)**3 -> A(1-0) 27x too small
            A = linear_rotor_einstein_a(freq, c["mu"], J)

            lines.append(SpectralLine(
                frequency=freq,
                frequency_uncertainty=c["freq_rms"],
                intensity=-4.0,
                einstein_a=A,
                upper_energy=E_upper,
                upper_degeneracy=2 * J + 1,
                lower_energy=E_lower,
                lower_degeneracy=2 * (J - 1) + 1,
                quantum_numbers_upper=f"J={J}",
                quantum_numbers_lower=f"J={J-1}",
                molecule="N2H+",
                isotopologue="14N2H+",
                database=SYNTHETIC_PROVENANCE,
                tag=29004
            ))

        return lines

    def get_molecule(self, molecule: str) -> MoleculeData:
        """
        Get complete molecule data from CDMS.

        Parameters
        ----------
        molecule : str
            Molecule name (e.g., "CO", "HCN")

        Returns
        -------
        MoleculeData
            Complete spectroscopic data
        """
        if molecule in self._molecule_cache:
            return self._molecule_cache[molecule]

        # Query all lines for molecule
        lines = self.query_lines(0, 3e6, molecule=molecule)

        # Extract energy levels
        energies = set()
        for line in lines:
            energies.add(line.upper_energy)
            energies.add(line.lower_energy)

        energy_levels = np.array(sorted(energies))

        # Get molecule properties
        props = self._get_molecule_properties(molecule)

        mol_data = MoleculeData(
            name=molecule,
            formula=props['formula'],
            mass=props['mass'],
            symmetry=props['symmetry'],
            dipole_moment=props['dipole'],
            energy_levels=energy_levels,
            level_degeneracies=np.ones(len(energy_levels), dtype=int),
            level_quantum_numbers=[f"E={e:.3f}" for e in energy_levels],
            transitions=lines,
            partition_function_temps=self.QROT_TEMPS.copy(),
            partition_function_values=props['Qrot']
        )

        self._molecule_cache[molecule] = mol_data
        return mol_data

    # CDMS-style partition-function temperature grid (K)
    QROT_TEMPS = np.array([9.375, 18.75, 37.5, 75, 150, 225, 300, 500, 1000])

    def _get_molecule_properties(self, molecule: str) -> Dict[str, Any]:
        """
        Basic molecule properties.

        FIX(audit C4b): the `Qrot` arrays used to be hardcoded and did
        not correspond to any consistent formula - CO was 1.434x the
        true rotational partition function, HCN/HCO+/N2H+ 0.848x.  They
        are now evaluated from the module's own rotational constants
        with `linear_rotor_partition_function`, so Q and the g_u = 2J+1
        line degeneracies are mutually consistent by construction.
        """
        properties = {
            "CO":   {"formula": "CO",   "mass": 28.0},
            "HCN":  {"formula": "HCN",  "mass": 27.0},
            "HCO+": {"formula": "HCO+", "mass": 29.0},
            "N2H+": {"formula": "N2H+", "mass": 29.0},
        }

        if molecule in properties:
            const = LINEAR_ROTOR_CONSTANTS[molecule]
            props = dict(properties[molecule])
            props["symmetry"] = "linear"
            props["dipole"] = const["mu"]
            props["Qrot"] = np.array([
                linear_rotor_partition_function(t, const["B"])
                for t in self.QROT_TEMPS])
            return props

        # Unknown molecule: generic 30 GHz linear rotor placeholder
        return {
            "formula": molecule,
            "mass": 30.0,
            "symmetry": "linear",
            "dipole": 1.0,
            "Qrot": np.array([
                linear_rotor_partition_function(t, 30000.0)
                for t in self.QROT_TEMPS]),
        }


class JPLDatabase(SpectroscopyDatabase):
    """
    JPL Molecular Spectroscopy Database interface.

    Complementary to CDMS, with some unique molecules.
    """

    BASE_URL = "https://spec.jpl.nasa.gov"

    def query_lines(self, freq_min: float, freq_max: float,
                   molecule: Optional[str] = None,
                   intensity_threshold: Optional[float] = None) -> List[SpectralLine]:
        """Query JPL catalog for spectral lines."""
        # Similar implementation to CDMS
        # JPL format is nearly identical
        lines = []

        # In production, would query JPL API
        # For now, delegate to CDMS-style simulation

        return lines

    def get_molecule(self, molecule: str) -> MoleculeData:
        """Get molecule data from JPL catalog."""
        if molecule in self._molecule_cache:
            return self._molecule_cache[molecule]

        lines = self.query_lines(0, 3e6, molecule=molecule)

        energies = set()
        for line in lines:
            energies.add(line.upper_energy)
            energies.add(line.lower_energy)

        mol_data = MoleculeData(
            name=molecule,
            formula=molecule,
            mass=30.0,
            symmetry="linear",
            dipole_moment=1.0,
            energy_levels=np.array(sorted(energies)) if energies else np.array([0.0]),
            level_degeneracies=np.ones(max(len(energies), 1), dtype=int),
            level_quantum_numbers=[],
            transitions=lines
        )

        self._molecule_cache[molecule] = mol_data
        return mol_data


class LAMDADatabase(SpectroscopyDatabase):
    """
    Leiden Atomic and Molecular Database interface.

    Primary source for collision rate coefficients needed
    for non-LTE radiative transfer calculations.
    """

    BASE_URL = "https://home.strw.leidenuniv.nl/~moldata"

    # Available collision partners
    COLLISION_PARTNERS = ["H2", "para-H2", "ortho-H2", "He", "H", "e-"]

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(cache_dir)
        self._collision_data: Dict[str, Dict[str, CollisionPartner]] = {}

    def query_lines(self, freq_min: float, freq_max: float,
                   molecule: Optional[str] = None,
                   intensity_threshold: Optional[float] = None) -> List[SpectralLine]:
        """Query LAMDA for radiative transitions."""
        # LAMDA primarily provides collision rates, not line lists
        # Return transitions from molecule data
        if molecule is None:
            return []

        mol_data = self.get_molecule(molecule)

        lines = [l for l in mol_data.transitions
                if freq_min <= l.frequency <= freq_max]

        return lines

    def get_molecule(self, molecule: str) -> MoleculeData:
        """
        Get complete LAMDA data for a molecule.

        Includes collision rate coefficients.
        """
        if molecule in self._molecule_cache:
            return self._molecule_cache[molecule]

        # Load or generate LAMDA-format data
        mol_data = self._load_lamda_file(molecule)

        self._molecule_cache[molecule] = mol_data
        return mol_data

    def _load_lamda_file(self, molecule: str) -> MoleculeData:
        """Load LAMDA datafile format."""
        # In production, would download from LAMDA website
        # Here we generate simulated data

        if molecule.upper() == "CO":
            return self._generate_co_lamda()
        elif molecule.upper() == "HCN":
            return self._generate_hcn_lamda()
        else:
            return self._generate_generic_lamda(molecule)

    def _generate_co_lamda(self) -> MoleculeData:
        """
        Build a LAMDA-shaped CO dataset from rotational constants.

        WARNING: this is *not* a LAMDA file.  Level energies, line
        frequencies and Einstein A coefficients are computed from
        B/D/H and the CO dipole moment (accurate: A(1-0) matches LAMDA
        to 0.03%), but the H2 collision rate coefficients are an
        order-of-magnitude placeholder, not the Yang et al. (2010)
        LAMDA data.  Provenance is reported as "synthetic".
        """
        n_levels = 41
        c = LINEAR_ROTOR_CONSTANTS["CO"]
        B = c["B"] * 1e-3  # GHz

        # Energy levels
        J_values = np.arange(n_levels)
        energies = B * J_values * (J_values + 1) / 0.0299792458  # cm^-1
        degeneracies = 2 * J_values + 1

        # Transitions and Einstein A
        transitions = []
        for J in range(1, n_levels):
            # FIX(audit C3): distortion term + correct A coefficient.
            # Was freq = 2*B*J*1000 and A = 3.497e-8*(freq/1000)**3*
            # J/(J+1), giving A(CO 1-0) = 2.68e-2 s^-1 - i.e. the CDMS
            # and LAMDA generators disagreed with each other by 1e9.
            freq = linear_rotor_frequency(J, c["B"], c["D"], c["H"])  # MHz
            A = linear_rotor_einstein_a(freq, c["mu"], J)

            transitions.append(SpectralLine(
                frequency=freq,
                frequency_uncertainty=_synthetic_freq_uncertainty(J, c),
                intensity=-4.0,
                einstein_a=A,
                upper_energy=energies[J],
                upper_degeneracy=int(degeneracies[J]),
                lower_energy=energies[J-1],
                lower_degeneracy=int(degeneracies[J-1]),
                quantum_numbers_upper=f"J={J}",
                quantum_numbers_lower=f"J={J-1}",
                molecule="CO",
                database=SYNTHETIC_PROVENANCE
            ))

        # Collision rates with H2
        temps = np.array([10, 20, 30, 50, 70, 100, 150, 200, 300, 500, 1000, 2000])
        n_trans = len(transitions)

        # FIX(audit c.2): these are DOWNWARD (de-excitation) rate
        # coefficients.  The old code multiplied them by
        # exp(-E_u/(1.4 T)), i.e. applied a Boltzmann suppression to a
        # downward rate - backwards; detailed balance puts that factor
        # on the *upward* rate.  It made k(CO 1-0, 10 K) = 2.4e-12,
        # 14x below the LAMDA value ~3.3e-11.  The remaining
        # 1e-11 (T/100)^0.5 scaling is an ORDER-OF-MAGNITUDE PLACEHOLDER
        # with no molecule-specific information; it is flagged as
        # synthetic and warned about in get_collision_rates().
        rates = np.zeros((n_trans, len(temps)))
        for i in range(n_trans):
            rates[i] = 1e-11 * (temps / 100.0) ** 0.5

        collision_h2 = CollisionPartner(
            partner="H2",
            temperatures=temps,
            rate_coefficients=rates,
            transitions=[(i+1, i) for i in range(n_trans)]
        )

        return MoleculeData(
            name="CO",
            formula="CO",
            mass=28.0,
            symmetry="linear",
            dipole_moment=0.112,
            energy_levels=energies,
            level_degeneracies=degeneracies.astype(int),
            level_quantum_numbers=[f"J={J}" for J in J_values],
            transitions=transitions,
            collision_partners={"H2": collision_h2},
            partition_function_temps=temps,
            partition_function_values=np.array([sum(degeneracies * np.exp(-energies * 1.4388 / T))
                                                for T in temps])
        )

    def _generate_hcn_lamda(self) -> MoleculeData:
        """
        Build a LAMDA-shaped HCN dataset from rotational constants.

        WARNING: computed, not retrieved - see `_generate_co_lamda`.
        """
        n_levels = 30
        c = LINEAR_ROTOR_CONSTANTS["HCN"]
        B = c["B"] * 1e-3  # GHz

        J_values = np.arange(n_levels)
        energies = B * J_values * (J_values + 1) / 0.0299792458
        degeneracies = 2 * J_values + 1

        transitions = []
        for J in range(1, n_levels):
            # FIX(audit C3): distortion term + correct Einstein A
            freq = linear_rotor_frequency(J, c["B"], c["D"], c["H"])
            A = linear_rotor_einstein_a(freq, c["mu"], J)

            transitions.append(SpectralLine(
                frequency=freq,
                frequency_uncertainty=_synthetic_freq_uncertainty(J, c),
                intensity=-3.5,
                einstein_a=A,
                upper_energy=energies[J],
                upper_degeneracy=int(degeneracies[J]),
                lower_energy=energies[J-1],
                lower_degeneracy=int(degeneracies[J-1]),
                quantum_numbers_upper=f"J={J}",
                quantum_numbers_lower=f"J={J-1}",
                molecule="HCN",
                database=SYNTHETIC_PROVENANCE
            ))

        temps = np.array([10, 20, 30, 50, 70, 100, 150, 200, 300, 500])
        n_trans = len(transitions)
        rates = np.zeros((n_trans, len(temps)))

        for i in range(n_trans):
            rates[i] = 2e-11 * (temps / 100)**0.5

        collision_h2 = CollisionPartner(
            partner="H2",
            temperatures=temps,
            rate_coefficients=rates,
            transitions=[(i+1, i) for i in range(n_trans)]
        )

        return MoleculeData(
            name="HCN",
            formula="HCN",
            mass=27.0,
            symmetry="linear",
            dipole_moment=2.985,
            energy_levels=energies,
            level_degeneracies=degeneracies.astype(int),
            level_quantum_numbers=[f"J={J}" for J in J_values],
            transitions=transitions,
            collision_partners={"H2": collision_h2}
        )

    def _generate_generic_lamda(self, molecule: str) -> MoleculeData:
        """
        Placeholder linear-rotor dataset for an unrecognised molecule.

        WARNING: nothing here is measured.  B = 30 GHz and mu = 1 D are
        arbitrary stand-ins; only the *internal consistency* between
        level energies, frequencies and Einstein A coefficients is
        guaranteed.  Never treat the numbers as spectroscopy.
        """
        warnings.warn(
            f"No spectroscopic data for '{molecule}': returning a "
            "generic linear rotor with B = 30 GHz and mu = 1 D. "
            "Frequencies and Einstein A coefficients are placeholders, "
            "not measurements.",
            RuntimeWarning, stacklevel=3)

        n_levels = 20
        b_mhz = 30000.0  # MHz, generic placeholder
        mu_generic = 1.0  # Debye, generic placeholder
        B = b_mhz * 1e-3  # GHz

        J_values = np.arange(n_levels)
        energies = B * J_values * (J_values + 1) / 0.0299792458
        degeneracies = 2 * J_values + 1

        transitions = []
        for J in range(1, n_levels):
            freq = 2 * b_mhz * J
            # FIX(audit C3): A = 1e-5*J was an invented linear ramp;
            # use the same rigid-rotor expression as everywhere else so
            # the placeholder is at least internally consistent.
            A = linear_rotor_einstein_a(freq, mu_generic, J)

            transitions.append(SpectralLine(
                frequency=freq,
                frequency_uncertainty=np.nan,  # unknown: not measured
                intensity=-4.0,
                einstein_a=A,
                upper_energy=energies[J],
                upper_degeneracy=int(degeneracies[J]),
                lower_energy=energies[J-1],
                lower_degeneracy=int(degeneracies[J-1]),
                quantum_numbers_upper=f"J={J}",
                quantum_numbers_lower=f"J={J-1}",
                molecule=molecule,
                database=SYNTHETIC_PROVENANCE
            ))

        return MoleculeData(
            name=molecule,
            formula=molecule,
            mass=30.0,
            symmetry="linear",
            dipole_moment=1.0,
            energy_levels=energies,
            level_degeneracies=degeneracies.astype(int),
            level_quantum_numbers=[f"J={J}" for J in J_values],
            transitions=transitions
        )

    def get_collision_rates(self, molecule: str, partner: str,
                           upper: int, lower: int,
                           temperature: float) -> float:
        """
        Get a *synthetic* collisional de-excitation rate coefficient.

        WARNING (audit rule 3): this class does not read LAMDA files.
        The returned coefficient comes from the order-of-magnitude
        scaling k ~ 1e-11 (T/100 K)^0.5 cm^3/s and carries no
        molecule-, transition- or partner-specific information.  For
        quantitative excitation modelling load a real LAMDA datafile.

        Parameters
        ----------
        molecule : str
            Molecule name
        partner : str
            Collision partner (e.g., "H2", "He")
        upper : int
            Upper level index
        lower : int
            Lower level index
        temperature : float
            Temperature in K

        Returns
        -------
        float
            Rate coefficient in cm^3/s (synthetic placeholder)
        """
        warnings.warn(
            "LAMDADatabase.get_collision_rates returns a synthetic "
            "order-of-magnitude placeholder, not LAMDA data.",
            RuntimeWarning, stacklevel=2)
        mol_data = self.get_molecule(molecule)

        if partner not in mol_data.collision_partners:
            raise ValueError(f"No collision data for {molecule}-{partner}")

        coll = mol_data.collision_partners[partner]

        # Find transition index
        trans_idx = None
        for i, (u, l) in enumerate(coll.transitions):
            if u == upper and l == lower:
                trans_idx = i
                break

        if trans_idx is None:
            raise ValueError(f"No data for transition {upper}->{lower}")

        # Interpolate in temperature
        rate = np.interp(temperature, coll.temperatures,
                        coll.rate_coefficients[trans_idx])

        return rate


class SplatalogueInterface:
    """
    Splatalogue database interface.

    Provides unified access to CDMS, JPL, and additional
    laboratory spectroscopy databases.
    """

    BASE_URL = "https://splatalogue.online"

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.astro_swarm/splatalogue_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Internal database instances
        self._cdms = CDMSDatabase(cache_dir)
        self._jpl = JPLDatabase(cache_dir)

    def search(self, freq_min: float, freq_max: float,
               species: Optional[List[str]] = None,
               energy_max: Optional[float] = None,
               line_list: Optional[List[str]] = None,
               include_atmospheric: bool = False) -> List[SpectralLine]:
        """
        Search Splatalogue for spectral lines.

        Parameters
        ----------
        freq_min : float
            Minimum frequency in MHz
        freq_max : float
            Maximum frequency in MHz
        species : list of str, optional
            Species to include (e.g., ["CO", "HCN"])
        energy_max : float, optional
            Maximum upper state energy in K
        line_list : list of str, optional
            Line lists to query (e.g., ["CDMS", "JPL"])
        include_atmospheric : bool
            Include atmospheric/telluric lines

        Returns
        -------
        List[SpectralLine]
            Matching spectral lines
        """
        all_lines = []

        # Query CDMS
        if line_list is None or "CDMS" in line_list:
            for mol in (species or [None]):
                cdms_lines = self._cdms.query_lines(freq_min, freq_max, molecule=mol)
                all_lines.extend(cdms_lines)

        # Query JPL
        if line_list is None or "JPL" in line_list:
            for mol in (species or [None]):
                jpl_lines = self._jpl.query_lines(freq_min, freq_max, molecule=mol)
                all_lines.extend(jpl_lines)

        # Filter by energy
        if energy_max is not None:
            all_lines = [l for l in all_lines
                        if l.upper_energy * 1.4388 <= energy_max]

        # Remove duplicates (same frequency within tolerance)
        unique_lines = []
        for line in sorted(all_lines, key=lambda x: x.frequency):
            if not unique_lines or abs(line.frequency - unique_lines[-1].frequency) > 0.1:
                unique_lines.append(line)

        return unique_lines

    def identify_lines(self, frequencies: np.ndarray,
                       tolerance: float = 1.0,
                       max_energy: float = 500) -> List[List[SpectralLine]]:
        """
        Identify lines from observed frequencies.

        Parameters
        ----------
        frequencies : ndarray
            Observed frequencies in MHz
        tolerance : float
            Frequency tolerance in MHz
        max_energy : float
            Maximum upper state energy in K

        Returns
        -------
        List[List[SpectralLine]]
            Possible identifications for each input frequency
        """
        identifications = []

        for freq in frequencies:
            matches = self.search(freq - tolerance, freq + tolerance,
                                energy_max=max_energy)
            identifications.append(matches)

        return identifications


class HITRANDatabase(SpectroscopyDatabase):
    """
    HITRAN database interface for high-resolution molecular absorption.

    Primarily for atmospheric studies but useful for exoplanet
    and protoplanetary disk atmospheres.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(cache_dir)
        # HITRAN molecule IDs
        self.molecule_ids = {
            "H2O": 1, "CO2": 2, "O3": 3, "N2O": 4, "CO": 5,
            "CH4": 6, "O2": 7, "NO": 8, "SO2": 9, "NO2": 10,
            "NH3": 11, "HNO3": 12, "OH": 13, "HF": 14, "HCl": 15,
            "HBr": 16, "HI": 17, "ClO": 18, "OCS": 19, "H2CO": 20
        }

    def query_lines(self, freq_min: float, freq_max: float,
                   molecule: Optional[str] = None,
                   intensity_threshold: Optional[float] = None) -> List[SpectralLine]:
        """
        Query HITRAN for spectral lines.

        Note: HITRAN uses wavenumber (cm^-1) natively.
        """
        # Convert MHz to cm^-1
        wn_min = freq_min * 1e6 / C_LIGHT
        wn_max = freq_max * 1e6 / C_LIGHT

        # In production, query HITRAN API (HAPI)
        lines = []

        return lines

    def get_molecule(self, molecule: str) -> MoleculeData:
        """Get molecule data from HITRAN."""
        return MoleculeData(
            name=molecule,
            formula=molecule,
            mass=30.0,
            symmetry="linear",
            dipole_moment=1.0,
            energy_levels=np.array([0.0]),
            level_degeneracies=np.array([1]),
            level_quantum_numbers=[],
            transitions=[]
        )


class UnifiedSpectroscopyQuery:
    """
    Unified interface to all spectroscopy databases.

    Provides convenient methods for common spectroscopic queries
    with automatic database selection and result merging.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cdms = CDMSDatabase(cache_dir)
        self.jpl = JPLDatabase(cache_dir)
        self.lamda = LAMDADatabase(cache_dir)
        self.splatalogue = SplatalogueInterface(cache_dir)
        self.hitran = HITRANDatabase(cache_dir)

    def get_line_list(self, freq_min: float, freq_max: float,
                      molecules: Optional[List[str]] = None,
                      databases: Optional[List[str]] = None) -> List[SpectralLine]:
        """
        Get comprehensive line list from multiple databases.

        Parameters
        ----------
        freq_min : float
            Minimum frequency in MHz
        freq_max : float
            Maximum frequency in MHz
        molecules : list of str, optional
            Molecules to include
        databases : list of str, optional
            Databases to query ("cdms", "jpl", "lamda", "hitran")

        Returns
        -------
        List[SpectralLine]
            Merged line list
        """
        all_lines = []
        db_map = {
            "cdms": self.cdms,
            "jpl": self.jpl,
            "lamda": self.lamda,
            "hitran": self.hitran
        }

        databases = databases or ["cdms", "jpl", "lamda"]

        for db_name in databases:
            if db_name not in db_map:
                continue
            db = db_map[db_name]

            for mol in (molecules or [None]):
                lines = db.query_lines(freq_min, freq_max, molecule=mol)
                all_lines.extend(lines)

        # Sort by frequency
        all_lines.sort(key=lambda x: x.frequency)

        return all_lines

    def get_collision_data(self, molecule: str,
                           partner: str = "H2") -> CollisionPartner:
        """
        Get collision rate data for radiative transfer.

        Parameters
        ----------
        molecule : str
            Molecule name
        partner : str
            Collision partner

        Returns
        -------
        CollisionPartner
            Collision rate data
        """
        mol_data = self.lamda.get_molecule(molecule)

        if partner not in mol_data.collision_partners:
            raise ValueError(f"No collision data for {molecule}-{partner}")

        return mol_data.collision_partners[partner]

    def estimate_column_density(self, molecule: str,
                                line_freq: float,
                                integrated_intensity: float,
                                temperature: float) -> float:
        """
        Estimate column density from single line.

        Parameters
        ----------
        molecule : str
            Molecule name
        line_freq : float
            Line frequency in MHz
        integrated_intensity : float
            Integrated intensity in K km/s
        temperature : float
            Excitation temperature in K

        Returns
        -------
        float
            Estimated column density in cm^-2
        """
        mol_data = self.lamda.get_molecule(molecule)

        # Find matching line
        best_line = None
        best_diff = float('inf')

        for line in mol_data.transitions:
            diff = abs(line.frequency - line_freq)
            if diff < best_diff:
                best_diff = diff
                best_line = line

        if best_line is None or best_diff > 1.0:  # 1 MHz tolerance
            raise ValueError(f"No line found at {line_freq} MHz for {molecule}")

        return mol_data.column_density_from_line(best_line, integrated_intensity, temperature)


# Convenience functions

def query_cdms(freq_min: float, freq_max: float,
               molecule: Optional[str] = None) -> List[SpectralLine]:
    """Query CDMS database."""
    db = CDMSDatabase()
    return db.query_lines(freq_min, freq_max, molecule)


def query_splatalogue(freq_min: float, freq_max: float,
                      species: Optional[List[str]] = None) -> List[SpectralLine]:
    """Query Splatalogue database."""
    spl = SplatalogueInterface()
    return spl.search(freq_min, freq_max, species=species)


def get_lamda_molecule(molecule: str) -> MoleculeData:
    """Get molecule data from LAMDA."""
    db = LAMDADatabase()
    return db.get_molecule(molecule)


def identify_line(frequency: float, tolerance: float = 1.0) -> List[SpectralLine]:
    """Identify spectral line at given frequency."""
    spl = SplatalogueInterface()
    return spl.search(frequency - tolerance, frequency + tolerance)
