"""Numerical regression tests for the August-2026 physics audit fixes.

Every test in this file guards one confirmed defect from
`02_physics_audit.md` / `B_findings.md` / `C_findings.md`.  Each asserts
the corrected value against ground truth that is derived here
independently of the module under test (closed-form algebra, an
independent quadrature, a self-consistent round trip, or a published
reference value quoted in the docstring).  The PRE-FIX wrong value is
recorded in a comment beside every assertion so the regression is
obvious if anything reverts.

Bug IDs used below are the audit's:

  C1  Truelove AMR criterion inverted            multiscale_coupling
  C2  donor-cell advection sign-flipped, v<0     multiscale_coupling
  C3  Einstein A / degeneracy / distortion       spectroscopic_databases
  C4  column_density_from_line missing k/h nu    spectroscopic_databases
  C4b hardcoded partition functions wrong        spectroscopic_databases
  C6  gamma() differentiates kappa not psi       advanced_lensing
  C7  potentials inconsistent with deflections   advanced_lensing
  C8  SIE kappa inconsistent with SIE alpha      advanced_lensing
  C9  module truncated, 4 helpers missing        molecular_cloud_physics
  C10 jeans_mass_thermal does not exist          shock_physics
  C11 tau_reion off by 1.7e40                    cosmological_context
  C13 tau_from_ratio returns inf always          spectral_line_analysis
  C17 chemistry rates a factor n_H too fast      chemical_networks
  C14 lte_column wrong by 1.4384e-05             spectral_line_analysis
  H4  NFW rho_crit missing h^2                   advanced_lensing
  H5  infer_H0 error bar in days                 advanced_lensing
  H16 solve() crashes for large t_final          chemical_networks
  B-SNR-1/2/3/4/6  Sedov profile, energy split,
      phase-boundary jump, 1 GHz luminosity,
      cooling-curve discontinuities              supernova_remnant_physics
  B-SH-2  C-shock width ~1e7x too small          shock_physics
  B-MC-3/4/5  NH3 pole, T_rot clamp, Planck
      dust reference wavelength                  molecular_cloud_physics
  B-COS-2 growth factor used the growth index    cosmological_context
"""

import math
import warnings

import numpy as np
import pytest

from astra_core.astro_physics import (
    advanced_lensing,
    chemical_networks,
    molecular_cloud_physics,
    multiscale_coupling,
    shock_physics,
    spectral_line_analysis,
    spectroscopic_databases,
    supernova_remnant_physics,
)
from astra_core.astro_physics.next_gen import cosmological_context


# CODATA / IAU constants used for the independent ground truth below.
H_PLANCK = 6.62607015e-27      # erg s
K_B = 1.380649e-16             # erg / K
C_CGS = 2.99792458e10          # cm / s
PC_CM = 3.0857e18              # cm
M_SUN_G = 1.989e33             # g
M_H = 1.6726e-24               # g
G_CGS = 6.674e-8               # cm^3 g^-1 s^-2


# =============================================================================
# C1 - Truelove Jeans refinement criterion (multiscale_coupling)
# =============================================================================

def test_c1_truelove_refinement_not_inverted():
    """Refinement level must RISE with density.

    Ground truth: a cell needs dx <= lambda_J / N_J, so the number of
    halvings required is ceil(log2(N_J dx / lambda_J)), clipped at 0.

    Pre-fix the code used the reciprocal ratio, giving level 3 to the
    well-resolved n = 1e2 cell and level 0 to the 43x under-resolved
    n = 1e8 cell.
    """
    c_s, dx, n_jeans, max_level = 2.0e4, 0.1 * 3.086e18, 4.0, 6
    ref = multiscale_coupling.HierarchicalRefinement(
        c_s, dx, n_jeans=n_jeans, max_level=max_level)

    mu = 2.33
    densities = np.array([1e2, 1e3, 1e4, 1e6, 1e8]) * mu * M_H
    levels = ref.level_for_density(densities)

    expected = []
    for rho in densities:
        lam_j = c_s * math.sqrt(math.pi / (ref.G * rho))
        expected.append(int(min(max(math.ceil(
            math.log2(max(n_jeans * dx / lam_j, 1.0))), 0), max_level)))

    # expected == [0, 0, 1, 5, 6];  pre-fix the code returned
    # [3, 1, 0, 0, 0] - the ordering was exactly inverted.
    assert list(levels) == expected == [0, 0, 1, 5, 6]
    assert np.all(np.diff(levels) >= 0), "level must be monotonic in density"


# =============================================================================
# C2 - donor-cell advection for v < 0 (multiscale_coupling)
# =============================================================================

def _advect(field, velocity, dt, n_steps):
    step = multiscale_coupling.MultiScaleSimulation._upwind_step
    out = field.copy()
    for _ in range(n_steps):
        out = step(out, np.asarray(velocity, dtype=float), dt)
    return out


def test_c2_donor_cell_stable_for_negative_velocity():
    """Upwind advection must be TVD (non-amplifying) for v < 0 too.

    Ground truth: for |v| dt <= 1 the donor-cell scheme is a convex
    combination of neighbouring values, so max|f| can never grow and
    sum(f) is conserved exactly on a periodic grid.

    Pre-fix, a delta function at v = -0.5, dt = 0.5 grew as
    1.25 / 3.05 / 18.6 / 791 / 1.85e6 after 1/5/10/20/40 steps.
    """
    f0 = np.zeros((16, 1, 1))
    f0[8] = 1.0

    for n_steps in (1, 5, 10, 20, 40):
        f = _advect(f0, [-0.5, 0.0, 0.0], 0.5, n_steps)
        assert f.max() <= 1.0 + 1e-12, f"amplified after {n_steps} steps"
        assert f.min() >= -1e-15
        assert f.sum() == pytest.approx(1.0, rel=1e-12)

    # And v -> -v must mirror exactly (the scheme has no built-in bias).
    f_neg = _advect(f0, [-0.5, 0.0, 0.0], 0.5, 40)
    f_pos = _advect(f0, [+0.5, 0.0, 0.0], 0.5, 40)
    assert f_neg.max() == pytest.approx(f_pos.max(), rel=1e-14)


# =============================================================================
# C3 - Einstein A coefficients, degeneracies, centrifugal distortion
# =============================================================================

def _rigid_rotor_a(nu_mhz, mu_debye, j_up):
    """A = 64 pi^4 nu^3 mu^2 J / (3 h c^3 (2J+1)), all CGS.

    Independent re-derivation of Townes & Schawlow (1955) ch. 1 with the
    Hoenl-London factor S(J -> J-1) = J.
    """
    nu = nu_mhz * 1e6
    mu = mu_debye * 1e-18
    return (64.0 * math.pi ** 4 * nu ** 3 * mu ** 2 * j_up
            / (3.0 * H_PLANCK * C_CGS ** 3 * (2 * j_up + 1)))


# Measured CDMS line centres (hyperfine-collapsed) and the tabulated
# LAMDA Einstein A coefficients for the same transitions.
CDMS_REFERENCE = {
    # molecule: {J_up: (nu_MHz, A_s^-1 or None)}
    "CO":   {1: (115271.2018, 7.203e-08), 2: (230538.0000, 6.910e-07),
             3: (345795.9899, 2.497e-06), 5: (576267.9305, 1.221e-05)},
    "HCN":  {1: (88631.6022, 2.407e-05)},
    "HCO+": {1: (89188.5247, 4.187e-05)},
    "N2H+": {1: (93173.3977, 3.628e-05)},
}


@pytest.fixture(scope="module")
def cdms_lines(tmp_path_factory):
    cache = tmp_path_factory.mktemp("cdms_cache")
    db = spectroscopic_databases.CDMSDatabase(cache_dir=str(cache))
    out = {}
    for mol in CDMS_REFERENCE:
        lines = db.query_lines(0.0, 3e6, molecule=mol)
        out[mol] = {int(l.quantum_numbers_upper.split("=")[1]): l
                    for l in lines}
    return out


@pytest.mark.parametrize("molecule", sorted(CDMS_REFERENCE))
def test_c3_line_frequencies_match_cdms(cdms_lines, molecule):
    """Generated frequencies must reproduce the measured CDMS values.

    Pre-fix, HCN/HCO+/N2H+ omitted the centrifugal-distortion term
    (nu = 2BJ), which is 0.35 MHz = 1.1 km/s off already at J = 1 and
    up to 254 MHz = 96 km/s at J = 9, while the code advertised a
    0.005 MHz frequency uncertainty.
    """
    for j_up, (nu_ref, _) in CDMS_REFERENCE[molecule].items():
        line = cdms_lines[molecule][j_up]
        assert line.frequency == pytest.approx(nu_ref, abs=0.01), (
            f"{molecule} J={j_up}->{j_up - 1}")
        # The quoted uncertainty must actually cover the model error.
        assert abs(line.frequency - nu_ref) <= line.frequency_uncertainty * 2


@pytest.mark.parametrize("molecule", sorted(CDMS_REFERENCE))
def test_c3_einstein_a_matches_reference(cdms_lines, molecule):
    """Einstein A must match LAMDA to <1%, not exceed it by 4e14.

    Pre-fix: A(CO 1-0) = 2.678e+07 s^-1 against the true 7.203e-08
    (3.72e14 too large), because A = 3.497e-8 nu_MHz^3 J/(J+1) used the
    frequency in MHz and the wrong degeneracy factor.  HCN/HCO+/N2H+
    used A0 (J/3)**3, which puts the literature 1-0 value at J = 3 and
    so makes A(1-0) 27x too SMALL.
    """
    for j_up, (_, a_ref) in CDMS_REFERENCE[molecule].items():
        line = cdms_lines[molecule][j_up]
        assert line.einstein_a == pytest.approx(a_ref, rel=0.01), (
            f"{molecule} J={j_up}->{j_up - 1}")


def test_c3_degeneracy_factor_is_j_over_2j_plus_1(cdms_lines):
    """A(J)/A(1) must follow nu^3 J/(2J+1), not J/(J+1)."""
    co = cdms_lines["CO"]
    for j_up in (2, 3, 5):
        ratio_code = co[j_up].einstein_a / co[1].einstein_a
        ratio_true = ((co[j_up].frequency / co[1].frequency) ** 3
                      * (j_up / (2 * j_up + 1)) / (1 / 3))
        assert ratio_code == pytest.approx(ratio_true, rel=1e-10)


def test_c3_helper_matches_independent_derivation():
    a_code = spectroscopic_databases.linear_rotor_einstein_a(
        115271.2018, 0.11011, 1)
    assert a_code == pytest.approx(_rigid_rotor_a(115271.2018, 0.11011, 1),
                                   rel=1e-12)
    # ... and the independent derivation reproduces LAMDA.
    assert a_code == pytest.approx(7.203e-08, rel=0.005)


def test_c3_cdms_and_lamda_generators_agree(cdms_lines, tmp_path):
    """The two 'databases' must not disagree by 1e9 for the same line.

    Pre-fix, CDMSDatabase gave A(CO 1-0) = 2.68e7 s^-1 and
    LAMDADatabase gave 2.68e-2 s^-1 - a factor 1e9 apart.
    """
    lam = spectroscopic_databases.LAMDADatabase(cache_dir=str(tmp_path))
    co = lam.get_molecule("CO")
    a_lamda = [t for t in co.transitions
               if t.quantum_numbers_upper == "J=1"][0].einstein_a
    assert a_lamda == pytest.approx(cdms_lines["CO"][1].einstein_a, rel=1e-9)


def test_c3_provenance_is_honest(cdms_lines, tmp_path):
    """Computed spectroscopy must not be stamped 'CDMS' / 'LAMDA'."""
    assert cdms_lines["CO"][1].database == "synthetic"
    lam = spectroscopic_databases.LAMDADatabase(cache_dir=str(tmp_path))
    assert lam.get_molecule("CO").transitions[0].database == "synthetic"


def test_c3_synthetic_collision_rates_warn(tmp_path):
    """Invented collision rates must announce themselves."""
    lam = spectroscopic_databases.LAMDADatabase(cache_dir=str(tmp_path))
    with pytest.warns(RuntimeWarning, match="synthetic"):
        k = lam.get_collision_rates("CO", "H2", 1, 0, 10.0)
    # Pre-fix a Boltzmann factor exp(-E_u/1.4T) suppressed this DOWNWARD
    # rate, giving 2.4e-12 - 14x below the LAMDA value ~3.3e-11.
    assert 1e-12 < k < 1e-10


# =============================================================================
# C4 / C4b - LTE column density and partition functions
# =============================================================================

def _q_rot_linear(temperature, b_mhz, j_max=200):
    j = np.arange(j_max + 1, dtype=float)
    e_k = (H_PLANCK * b_mhz * 1e6 / K_B) * j * (j + 1.0)
    return float(np.sum((2 * j + 1.0) * np.exp(-e_k / temperature)))


def test_c4b_partition_function_matches_classical_limit():
    """Q_rot(T) -> kT/(hB) + 1/3 for kT >> hB.

    Pre-fix, _get_molecule_properties returned hardcoded tables that
    were 1.434x (CO) and 0.848x (HCN/HCO+/N2H+) the true Q, e.g.
    Q(CO, 300 K) = 156 against the correct 108.79.
    """
    b_co = spectroscopic_databases.LINEAR_ROTOR_CONSTANTS["CO"]["B"]
    q = spectroscopic_databases.linear_rotor_partition_function(300.0, b_co)
    classical = 300.0 / (H_PLANCK * b_co * 1e6 / K_B) + 1.0 / 3.0
    assert q == pytest.approx(classical, rel=1e-4)
    assert q == pytest.approx(108.79, rel=1e-3)


def test_c4_column_density_from_line(tmp_path):
    """N_u = 8 pi k nu^2 W / (h c^3 A), then N_tot = N_u (Q/g_u) e^(Eu/kT).

    Pre-fix the code used 8 pi nu^3/(c^3 A) - missing k/(h nu) = 1/5.532
    at 115 GHz - and, combined with the broken Einstein A, returned
    23.4 cm^-2 for 10 K km/s of CO 1-0 at T_ex = 20 K.
    """
    db = spectroscopic_databases.CDMSDatabase(cache_dir=str(tmp_path))
    mol = db.get_molecule("CO")
    line = [l for l in mol.transitions if l.quantum_numbers_upper == "J=1"][0]

    w_kkms, t_ex = 10.0, 20.0
    n_code = mol.column_density_from_line(line, w_kkms, t_ex)

    # Independent ground truth, built from published constants only.
    nu = 115271.2018e6
    a_ul = 7.203e-08
    n_u = (8 * math.pi * K_B * nu ** 2 * w_kkms * 1e5
           / (H_PLANCK * C_CGS ** 3 * a_ul))
    q = _q_rot_linear(t_ex, 57635.968)
    n_true = n_u * q / 3.0 * math.exp(5.5321 / t_ex)

    assert n_true == pytest.approx(1.19e16, rel=0.02)     # sanity
    assert n_code == pytest.approx(n_true, rel=0.01)      # pre-fix: 23.4

    # The optional CMB-subtracted convention multiplies by
    # J(T_ex)/(J(T_ex)-J(T_bg)) = 1.0507 at 115 GHz, 20 K.
    n_cmb = mol.column_density_from_line(line, w_kkms, t_ex, t_background=2.725)
    hnu_k = H_PLANCK * nu / K_B
    j_ex = hnu_k / math.expm1(hnu_k / t_ex)
    j_bg = hnu_k / math.expm1(hnu_k / 2.725)
    assert n_cmb / n_code == pytest.approx(j_ex / (j_ex - j_bg), rel=1e-6)


# =============================================================================
# C13 / C14 - spectral_line_analysis
# =============================================================================

@pytest.mark.parametrize("tau_true", [0.1, 0.5, 1.0, 3.0, 10.0])
def test_c13_tau_from_ratio_round_trip(tau_true):
    """tau must be recovered from r = (1-e^-s tau)/(1-e^-tau).

    Pre-fix the function returned `inf` for all five of these cases
    (and for every other physically possible input), because it tested
    1 - r_obs/s <= 0, which holds identically since r_obs >= s.
    """
    s = 0.2
    r_obs = -math.expm1(-s * tau_true) / -math.expm1(-tau_true)
    corr = spectral_line_analysis.OpticalDepthCorrector()
    tau = corr.tau_from_ratio(t_main=1.0, t_satellite=r_obs,
                              strength_ratio=s)
    assert math.isfinite(tau)
    assert tau == pytest.approx(tau_true, rel=1e-8)


def test_c13_tau_from_ratio_limits():
    corr = spectral_line_analysis.OpticalDepthCorrector()
    assert corr.tau_from_ratio(1.0, 0.19, 0.2) == 0.0     # thin limit
    assert corr.tau_from_ratio(1.0, 1.0, 0.2) == 300.0    # saturated


def test_c14_lte_column_absolute_value():
    """Mangum & Shirley (2015) eq. 80, re-derived independently here.

    Pre-fix the routine was low by exactly g_l (h c/k) 1e-5 = 1.4384e-05
    (nu^2/c^2 instead of nu^3/c^3, a spurious h nu/k, a spurious g_l, and
    K km/s fed into a CGS expression): it returned 1.335e+11 cm^-2 where
    the correct answer is 9.281e+15.
    """
    w, nu_ghz, a_ul, t_ex, e_u, g_u, g_l, q = (
        10.0, 115.271202, 7.203e-8, 10.0, 5.5321, 3, 1, 3.968)

    nu = nu_ghz * 1e9
    hnu_k = H_PLANCK * nu / K_B
    j_ex = hnu_k / math.expm1(hnu_k / t_ex)
    j_bg = hnu_k / math.expm1(hnu_k / 2.7255)
    n_true = (8 * math.pi * nu ** 3 / (C_CGS ** 3 * a_ul)
              * (q / g_u) * math.exp(e_u / t_ex)
              / math.expm1(hnu_k / t_ex)
              * (w * 1e5) / (j_ex - j_bg))

    calc = spectral_line_analysis.ColumnDensityCalculator()
    n_code = calc.lte_column(w, nu_ghz, a_ul, t_ex, e_u, g_u, g_l, q)

    assert n_true == pytest.approx(9.281e15, rel=0.01)   # sanity
    # 1e-3 tolerance: the module carries 4-digit CGS constants.
    assert n_code == pytest.approx(n_true, rel=1e-3)


def test_c14_lte_column_ignores_g_lower():
    """g_lower must not enter eq. 80 (it was a spurious factor)."""
    calc = spectral_line_analysis.ColumnDensityCalculator()
    args = (10.0, 115.271202, 7.203e-8, 10.0, 5.5321, 3)
    assert calc.lte_column(*args, 1, 3.968) == \
        calc.lte_column(*args, 7, 3.968)


def test_c14_lte_column_scales_linearly_with_w():
    calc = spectral_line_analysis.ColumnDensityCalculator()
    a = calc.lte_column(1.0, 115.271202, 7.203e-8, 10.0, 5.5321, 3, 1, 3.968)
    b = calc.lte_column(7.0, 115.271202, 7.203e-8, 10.0, 5.5321, 3, 1, 3.968)
    assert b / a == pytest.approx(7.0, rel=1e-12)


# =============================================================================
# C17 / H16 - chemical_networks
# =============================================================================

def _single_reaction_solver():
    net = chemical_networks.ReactionNetwork("audit")
    net.add_species(chemical_networks.Species("A", 1.0))
    net.add_species(chemical_networks.Species("B", 1.0))
    net.add_reaction(chemical_networks.Reaction(
        ["A"], ["B"], chemical_networks.ReactionType.COSMIC_RAY_IONIZATION,
        alpha=1.0))
    return net, chemical_networks.ChemistrySolver(net)


@pytest.mark.parametrize("n_h", [1e2, 1e4])
def test_c17_unary_rate_is_density_independent(n_h):
    """A cosmic-ray ionisation must e-fold in 1/k, independent of n_H.

    zeta = 1e-13 s^-1 -> exact e-folding time 1.000e13 s.
    Pre-fix the solver returned 9.898e10 s at n_H = 1e2 and 9.987e8 s
    at n_H = 1e4 - exactly a factor 1/n_H too fast.
    """
    year_s = 3.156e7
    net, solver = _single_reaction_solver()
    cond = chemical_networks.ChemistryConditions(
        temperature=10.0, density=n_h, zeta_CR=1e-13)
    res = solver.solve({"A": 1.0, "B": 0.0}, cond,
                       t_final=3e13 / year_s, n_output=600)

    y_a = res.abundances[:, net.get_species_index("A")]
    t_s = res.times * year_s
    t_efold = float(np.interp(-math.exp(-1.0), -y_a, t_s))

    assert t_efold == pytest.approx(1.0e13, rel=2e-3)


def test_h16_solve_accepts_large_t_final():
    """t_eval must stay inside t_span despite log10 round-tripping.

    Pre-fix, t_final = 1e12 yr put t_eval[-1] 9.83e4 s above t_span[1]
    and scipy raised "Values in t_eval are not within t_span".
    """
    _, solver = _single_reaction_solver()
    cond = chemical_networks.ChemistryConditions(temperature=10.0,
                                                 density=1e4)
    res = solver.solve({"A": 1.0, "B": 0.0}, cond, t_final=1e12,
                       n_output=50)
    assert res.abundances.shape[0] == 50


# =============================================================================
# C6 / C7 / C8 / H4 / H5 - advanced_lensing
# =============================================================================

def _grad_potential(profile, x, y, h=1e-5):
    px = (profile.potential(np.array(x + h), np.array(y))
          - profile.potential(np.array(x - h), np.array(y))) / (2 * h)
    py = (profile.potential(np.array(x), np.array(y + h))
          - profile.potential(np.array(x), np.array(y - h))) / (2 * h)
    return float(px), float(py)


def _div_alpha(profile, x, y, h=1e-5):
    ax1, _ = profile.alpha(np.array(x + h), np.array(y))
    ax0, _ = profile.alpha(np.array(x - h), np.array(y))
    _, ay1 = profile.alpha(np.array(x), np.array(y + h))
    _, ay0 = profile.alpha(np.array(x), np.array(y - h))
    return float((ax1 - ax0) / (2 * h) + (ay1 - ay0) / (2 * h)) / 2.0


def test_c6_sis_shear_equals_convergence():
    """For a singular isothermal sphere, |gamma| = kappa = theta_E/(2r).

    At theta_E = 1, (x, y) = (1.5, 0.7): kappa = 0.302061.
    Pre-fix `gamma()` differentiated kappa instead of psi and returned
    |gamma| = 0.165366 (-45%), with units of kappa per arcsec^2.
    """
    x, y = 1.5, 0.7
    sis = advanced_lensing.SIEProfile(theta_E=1.0, e=0.0, theta_e=0.0)
    kappa_true = 1.0 / (2.0 * math.hypot(x, y))
    assert kappa_true == pytest.approx(0.3020610, abs=1e-6)

    g1, g2 = sis.gamma(np.array(x), np.array(y))
    # 1e-4 tolerance: gamma() uses a finite-difference step h = 0.01.
    assert math.hypot(float(g1), float(g2)) == pytest.approx(
        kappa_true, rel=2e-4)


def test_c6_external_shear_gamma_is_callable():
    """ExternalShear.__init__ must not shadow the gamma() method.

    Pre-fix: TypeError: 'float' object is not callable.
    """
    shear = advanced_lensing.ExternalShear(0.05, 30.0)
    g1, g2 = shear.gamma(np.array(1.0), np.array(1.0))
    assert float(g1) == pytest.approx(0.05 * math.cos(2 * math.radians(30.0)))
    assert float(g2) == pytest.approx(0.05 * math.sin(2 * math.radians(30.0)))
    assert shear.gamma_ext == 0.05


def test_c8_sie_kappa_consistent_with_alpha():
    """kappa must equal (1/2) div alpha - the same mass distribution.

    Kormann et al. (1994): kappa = sqrt(q) theta_E / (2 sqrt(q^2x^2+y^2))
    = 0.331497 at q = 0.7, (1.5, 0.7).  Pre-fix `kappa()` returned
    0.277350 - a 16.3% inconsistency with its own deflection field.
    """
    x, y, q = 1.5, 0.7, 0.7
    sie = advanced_lensing.SIEProfile(theta_E=1.0, e=1 - q, theta_e=0.0)

    kappa_kormann = math.sqrt(q) / (2 * math.sqrt(q ** 2 * x ** 2 + y ** 2))
    assert kappa_kormann == pytest.approx(0.331497, abs=1e-6)

    assert float(sie.kappa(np.array(x), np.array(y))) == pytest.approx(
        kappa_kormann, rel=1e-9)
    assert _div_alpha(sie, x, y) == pytest.approx(kappa_kormann, rel=1e-6)


def test_c7_sie_potential_gradient_equals_alpha():
    """grad psi == alpha, and psi == x.alpha by Euler's theorem.

    Pre-fix: grad psi = (0.696143, 0.662994) against
    alpha = (0.824581, 0.490949), and psi = 1.508310 against the
    required x.alpha = 1.580536.
    """
    x, y = 1.5, 0.7
    sie = advanced_lensing.SIEProfile(theta_E=1.0, e=0.3, theta_e=0.0)

    ax, ay = (float(v) for v in sie.alpha(np.array(x), np.array(y)))
    px, py = _grad_potential(sie, x, y)
    assert px == pytest.approx(ax, rel=1e-6)
    assert py == pytest.approx(ay, rel=1e-6)

    psi = float(sie.potential(np.array(x), np.array(y)))
    assert psi == pytest.approx(x * ax + y * ay, rel=1e-12)
    assert psi == pytest.approx(1.580536, rel=1e-5)


def test_c7_nfw_potential_gradient_equals_alpha():
    """NFW psi must be the antiderivative of the NFW deflection.

    Pre-fix the "simplified potential" kept only ln^2(u/2); its
    gradient at (2, 1) was (-888.76, -444.38) against
    alpha = (2.0249, 1.0124) - wrong by -439x including the sign - and
    it fed every predicted time delay.
    """
    nfw = advanced_lensing.NFWProfile(M_200=1e14, c=5.0,
                                      z_lens=0.3, z_source=2.0)
    for point in [(2.0, 1.0), (0.4, 0.2), (6.0, -3.0)]:
        ax, ay = nfw.alpha(np.array(point[0]), np.array(point[1]))
        px, py = _grad_potential(nfw, *point, h=1e-4)
        assert px == pytest.approx(float(np.ravel(ax)[0]), rel=1e-4)
        assert py == pytest.approx(float(np.ravel(ay)[0]), rel=1e-4)


def test_h4_nfw_rho_crit_includes_h_squared():
    """r_200 for 1e14 M_sun at z = 0.3, H0 = 70 must be ~864 kpc.

    Ground truth: rho_crit(z) = 2.775e11 h^2 E(z)^2 M_sun/Mpc^3;
    r_200 = (3 M / (4 pi 200 rho_crit))^(1/3).
    Pre-fix the h^2 was dropped, so rho_crit was 2.045x too high and
    r_200 came out 681 kpc.
    """
    cosmo = advanced_lensing.Cosmology()
    nfw = advanced_lensing.NFWProfile(M_200=1e14, c=5.0,
                                      z_lens=0.3, z_source=2.0,
                                      cosmo=cosmo)
    h = cosmo.H0 / 100.0
    rho_crit = 2.775e11 * h ** 2 * cosmo.E(0.3) ** 2
    r200_true = (3 * 1e14 / (4 * math.pi * 200 * rho_crit)) ** (1 / 3) * 1e3
    assert r200_true == pytest.approx(864.0, rel=0.01)
    assert nfw.r_200 == pytest.approx(r200_true, rel=1e-6)


def test_h5_infer_h0_error_has_h0_units():
    """sigma_H0/H0 must equal sigma_dt/dt, not 1/sqrt(sum 1/sigma_dt^2).

    Pre-fix, a single 30 +/- 2 day delay at H0 = 70 returned
    H0_err = 2.0 (a number of days) instead of 4.67 km/s/Mpc.
    """
    lens = advanced_lensing.CompositeLensModel(z_lens=0.5, z_source=2.0)
    lens.add_profile(advanced_lensing.SIEProfile(theta_E=1.5, e=0.2,
                                                 theta_e=30.0))
    tdc = advanced_lensing.TimeDelayCosmography(lens)
    res = tdc.infer_H0({(0, 1): 30.0}, [(1.6, 0.3), (-1.4, -0.2)],
                       {(0, 1): 2.0})
    assert abs(res['H0_err'] / res['H0']) == pytest.approx(2.0 / 30.0,
                                                           rel=1e-9)


# =============================================================================
# C9 / B-MC-3 / B-MC-4 / B-MC-5 - molecular_cloud_physics
# =============================================================================

@pytest.fixture(scope="module")
def cloud_engine():
    return molecular_cloud_physics.MolecularCloudPhysicsEngine()


def test_c9_helpers_exist(cloud_engine):
    """The four helpers lost to file truncation must be back."""
    for name in ("_solve_optical_depth", "_brightness_to_tex",
                 "_column_density_lte", "_planck_function"):
        assert callable(getattr(cloud_engine, name, None)), name


def test_c9_planck_function(cloud_engine):
    """B_nu = 2 h nu^3 / c^2 / (e^{h nu/kT} - 1)."""
    nu = C_CGS / 350e-4          # 350 micron
    b_code = float(cloud_engine._planck_function(np.array(nu), 20.0))
    x = 6.626e-27 * nu / (1.38e-16 * 20.0)
    b_true = 2 * 6.626e-27 * nu ** 3 / (2.998e10) ** 2 / math.expm1(x)
    assert b_code == pytest.approx(b_true, rel=1e-12)
    # Rayleigh-Jeans limit far from the peak.
    nu_rj = C_CGS / 3.0          # 3 cm
    assert float(cloud_engine._planck_function(np.array(nu_rj), 20.0)) == \
        pytest.approx(2 * 1.38e-16 * 20.0 * nu_rj ** 2 / (2.998e10) ** 2,
                      rel=1e-4)


@pytest.mark.parametrize("tau_true", [0.5, 2.0, 10.0])
def test_c9_solve_optical_depth_round_trip(cloud_engine, tau_true):
    r = 77.0
    ratio = -math.expm1(-tau_true) / -math.expm1(-tau_true / r)
    assert cloud_engine._solve_optical_depth(ratio, r) == pytest.approx(
        tau_true, rel=1e-8)


def test_c9_brightness_to_tex(cloud_engine):
    """T_ex = (h nu/k) / ln[1 + (h nu/k)/(T_mb + J(T_bg))].

    Round-trip check: the derived T_ex must reproduce the input T_mb
    through T_mb = J(T_ex) - J(T_bg).
    """
    t_mb, nu_ghz = 10.0, 115.271
    t_ex = cloud_engine._brightness_to_tex(t_mb, nu_ghz)
    j_ex = cloud_engine._j_nu(t_ex, nu_ghz)
    j_bg = cloud_engine._j_nu(2.725, nu_ghz)
    assert j_ex - j_bg == pytest.approx(t_mb, rel=1e-10)
    assert t_ex == pytest.approx(13.4133, rel=1e-4)


def test_c9_column_density_lte_matches_independent_value(cloud_engine):
    """N(13CO) from Mangum & Shirley eq. 80, re-derived here."""
    trans = molecular_cloud_physics.MolecularLineDatabase.get_transition(
        "13CO_1-0")
    n_code = cloud_engine._column_density_lte(10.0, 10.0, trans, 0.0)

    nu = trans.rest_frequency * 1e9
    b_hz = nu / 2.0
    q = float(np.sum((2 * np.arange(200.) + 1)
                     * np.exp(-(6.626e-27 * b_hz / 1.38e-16)
                              * np.arange(200.) * (np.arange(200.) + 1) / 10.0)))
    hnu_k = 6.626e-27 * nu / 1.38e-16
    j_ex = hnu_k / math.expm1(hnu_k / 10.0)
    j_bg = hnu_k / math.expm1(hnu_k / 2.725)
    n_true = (8 * math.pi * nu ** 3 / ((2.998e10) ** 3 * trans.einstein_A)
              * q / 3.0 * math.exp(trans.upper_energy / 10.0)
              / math.expm1(hnu_k / 10.0) * 1e6 / (j_ex - j_bg))
    assert n_code == pytest.approx(n_true, rel=1e-9)
    assert n_code == pytest.approx(9.9e15, rel=0.05)

    # tau -> 0 limit and the opacity correction tau/(1-e^-tau).
    thick = cloud_engine._column_density_lte(10.0, 10.0, trans, 2.0)
    assert thick / n_code == pytest.approx(2.0 / -math.expm1(-2.0), rel=1e-9)


def test_c9_column_density_lte_rejects_non_linear_rotors(cloud_engine):
    trans = molecular_cloud_physics.MolecularLineDatabase.get_transition(
        "NH3_1-1") or molecular_cloud_physics.MolecularLineDatabase.get_transition(
        "CII_158um")
    with pytest.raises(NotImplementedError):
        cloud_engine._column_density_lte(10.0, 10.0, trans, 0.0)


def test_c9_analyze_co_isotopologues_runs(cloud_engine):
    """Pre-fix this raised AttributeError on every call."""
    co12 = molecular_cloud_physics.CloudSpectralLine(
        "12CO_1-0", 10.0, 5.0, 2.0, 25.0, 0.1)
    co13 = molecular_cloud_physics.CloudSpectralLine(
        "13CO_1-0", 2.0, 5.0, 1.8, 4.0, 0.1)
    res = cloud_engine.analyze_co_isotopologues(co12, co13)
    assert res['tau_12CO'] > res['tau_13CO'] > 0
    assert 5.0 < res['T_ex'] < 50.0
    assert 1e20 < res['N_H2'] < 1e24


def test_c9_fit_dust_sed_round_trip():
    """fit_dust_sed must recover an injected (T, beta, M).

    Pre-fix it returned None (the file was truncated inside it).
    """
    eng = molecular_cloud_physics.MolecularCloudPhysicsEngine(
        molecular_cloud_physics.DustModel.OSSENKOPF_THICK)
    dust = eng.dust
    wl = np.array([160., 250., 350., 500., 850.])
    t_true, beta_true, m_true, dist = 18.0, 1.7, 5.0, 250.0

    nu = molecular_cloud_physics.c_light / (wl * 1e-4)
    kappa = dust.kappa_ref * (dust.lambda_ref / wl) ** beta_true
    b_nu = eng._planck_function(nu, t_true)
    flux = ((m_true * molecular_cloud_physics.Msun_to_g) * kappa * b_nu
            / (dist * molecular_cloud_physics.pc_to_cm) ** 2 / 1e-23)

    sed = molecular_cloud_physics.DustSED(
        wavelengths=wl, fluxes=flux, flux_errors=0.05 * flux,
        beam_sizes=np.full(5, 36.0), distance_pc=dist)
    res = eng.fit_dust_sed(sed)

    assert res is not None
    assert res['T_dust'] == pytest.approx(t_true, rel=1e-6)
    assert res['beta'] == pytest.approx(beta_true, rel=1e-6)
    assert res['M_dust'] == pytest.approx(m_true, rel=1e-6)
    assert res['M_gas'] == pytest.approx(m_true * dust.gas_to_dust, rel=1e-6)
    assert res['chi2_reduced'] < 1e-10


def test_c9_restored_public_analytics_exist(cloud_engine):
    """Eight public methods were also lost to the truncation.

    `molecular_cloud_agents.py` calls all of these; pre-fix every one
    raised AttributeError.
    """
    for name in ("column_density_from_submm", "virial_mass",
                 "virial_parameter", "jeans_mass", "bonnor_ebert_mass",
                 "mach_number", "star_formation_threshold",
                 "dense_gas_fraction"):
        assert callable(getattr(cloud_engine, name, None)), name


def test_c9_virial_and_jeans_analytics(cloud_engine):
    """Check each restored formula against closed-form algebra."""
    mu_h2, mu_p = 2.8, 2.33
    k_b, m_h = molecular_cloud_physics.k_B, molecular_cloud_physics.m_H
    g_c, pc = molecular_cloud_physics.G_cgs, molecular_cloud_physics.pc_to_cm
    msun = molecular_cloud_physics.Msun_to_g

    # M_vir = 5 sigma^2 R / G
    m_vir_true = 5 * (1e5) ** 2 * pc / g_c / msun
    assert cloud_engine.virial_mass(1.0, 1.0) == pytest.approx(m_vir_true,
                                                               rel=1e-12)
    # ... and the familiar M_vir = 210 R dV^2 (dV = FWHM) form.
    assert cloud_engine.virial_mass(1.0, 1.0 / 2.3548) == pytest.approx(
        210.0, rel=0.01)

    # alpha_vir = M_vir / M
    assert cloud_engine.virial_parameter(1000.0, 1.0, 1.0) == \
        pytest.approx(m_vir_true / 1000.0, rel=1e-12)

    # M_J = (pi^2.5/6) c_s^3 / (G^1.5 rho^0.5)
    c_s = math.sqrt(k_b * 10.0 / (mu_p * m_h))
    rho = 1e4 * mu_h2 * m_h
    m_j_true = (math.pi ** 2.5 / 6) * c_s ** 3 / (g_c ** 1.5
                                                  * math.sqrt(rho)) / msun
    assert cloud_engine.jeans_mass(10.0, 1e4) == pytest.approx(m_j_true,
                                                               rel=1e-12)
    assert 0.5 < m_j_true < 10.0        # sanity for a 10 K, 1e4 cm^-3 core

    # M_BE = 1.18 c_s^4 / (G^1.5 P^0.5), P = (P/k) k_B
    m_be_true = (1.18 * c_s ** 4
                 / (g_c ** 1.5 * math.sqrt(1e4 * k_b))) / msun
    assert cloud_engine.bonnor_ebert_mass(10.0, 1e4) == pytest.approx(
        m_be_true, rel=1e-12)

    # Mach number
    assert cloud_engine.mach_number(1.0, 10.0) == pytest.approx(1e5 / c_s,
                                                                rel=1e-12)
    assert cloud_engine.sound_speed(10.0) == pytest.approx(c_s, rel=1e-12)


def test_c9_column_density_from_submm_round_trip(cloud_engine):
    """N(H2) must invert the optically thin dust emission exactly."""
    flux, wl, t_d, beam, dist = 1.0, 850.0, 20.0, 14.0, 250.0
    res = cloud_engine.column_density_from_submm(flux, wl, t_d, beam, dist)

    nu = molecular_cloud_physics.c_light / (wl * 1e-4)
    kappa = cloud_engine.dust.kappa_nu(wl)
    b_nu = float(cloud_engine._planck_function(np.array(nu), t_d))
    omega = math.pi * (beam / 206265.0) ** 2 / (4 * math.log(2))
    sigma_gas = (flux * 1e-23) / (omega * kappa * b_nu) \
        * cloud_engine.dust.gas_to_dust
    n_true = sigma_gas / (2.8 * molecular_cloud_physics.m_H)

    assert res['N_H2'] == pytest.approx(n_true, rel=1e-12)
    assert res['A_V'] == pytest.approx(n_true / cloud_engine.N_H2_to_AV,
                                       rel=1e-12)
    assert res['M_beam'] > 0


def test_c9_dense_gas_fraction_refuses_rather_than_inventing():
    """Rule: refuse when the inputs cannot determine the answer."""
    eng = molecular_cloud_physics.MolecularCloudPhysicsEngine()
    with pytest.raises(NotImplementedError, match="emitting area"):
        eng.dense_gas_fraction(1e4, 5.0, 400.0)


def test_bmc3_nh3_thermometer_never_returns_negative_temperature(
        cloud_engine):
    """T_kin must never be negative.

    Pre-fix the Ho & Townes denominator vanished at T_rot = 67.26 K with
    no guard: an observed I(2,2)/I(1,1) = 0.90 returned
    T_kin = -44834 K and a ratio of 1.00 returned -331.65 K.
    """
    def line(intensity):
        return molecular_cloud_physics.CloudSpectralLine(
            "NH3", intensity, 0.0, 1.0, intensity, 0.05)

    for ratio in np.linspace(0.02, 2.0, 60):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = cloud_engine.analyze_nh3_temperature(line(1.0),
                                                       line(ratio))
        t_kin = res['T_kin']
        assert math.isnan(t_kin) or t_kin > 0.0, (ratio, t_kin)

    # A cold-core ratio still gives a sensible answer.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cold = cloud_engine.analyze_nh3_temperature(line(1.0), line(0.3))
    assert 15.0 < cold['T_rot'] < 35.0
    assert 15.0 < cold['T_kin'] < 60.0

    # Beyond the pole: NaN + warning, not a negative temperature.
    with pytest.warns(RuntimeWarning):
        hot = cloud_engine.analyze_nh3_temperature(line(1.0), line(0.90))
    assert math.isnan(hot['T_kin'])          # pre-fix: -44834 K


def test_bmc4_nh3_inverted_ratio_is_flagged_not_clamped(cloud_engine):
    """I(2,2)*3/5 >= I(1,1) has no LTE T_rot.

    Pre-fix `max(T_rot, 8.0)` turned that into a silent 8.0 K, giving a
    x49 discontinuity across ratio = 5/3 (1.50 -> 393.9 K,
    1.67 -> 8.0 K).
    """
    def line(intensity):
        return molecular_cloud_physics.CloudSpectralLine(
            "NH3", intensity, 0.0, 1.0, intensity, 0.05)

    with pytest.warns(RuntimeWarning, match="inverted"):
        res = cloud_engine.analyze_nh3_temperature(line(1.0), line(1.7))
    assert math.isnan(res['T_rot'])          # pre-fix: 8.0 K


def test_bmc5_planck_dust_reference_wavelength():
    """kappa(850 um) must be the defining 0.92 cm^2/g.

    Pre-fix `lambda_ref` held 353 (the frequency in GHz) in a field
    declared in microns, so kappa(850 um) = 0.22158 - 4.152x low - while
    kappa(353 um) spuriously returned exactly 0.92.
    """
    planck = molecular_cloud_physics.DustModelLibrary.get_model(
        molecular_cloud_physics.DustModel.PLANCK)
    assert planck.lambda_ref == pytest.approx(850.0)
    assert planck.kappa_nu(850.0) == pytest.approx(0.92, rel=1e-12)
    # 353 GHz is 849.3 um, so the frequency call must agree with 850 um.
    assert planck.kappa_nu_frequency(353.0) == pytest.approx(0.92, rel=2e-3)


# =============================================================================
# C10 / B-SH-2 - shock_physics
# =============================================================================

def test_c10_triggered_star_formation_runs_and_matches_jeans_mass():
    """Pre-fix: AttributeError - jeans_mass_thermal does not exist.

    The arguments were also swapped and `interface_density` (a MASS
    density) was passed to an API defaulting to number density.
    """
    rho = 4e4 * 2.37 * M_H          # g/cm^3
    temperature = 20.0
    cloud_mass = 1e4 * M_SUN_G

    res = shock_physics.CloudCollisionShock().triggered_star_formation(
        rho, temperature, cloud_mass)

    c_s = math.sqrt(K_B * temperature / (2.37 * M_H))
    m_j_true = (math.pi ** 2.5 / 6.0) * c_s ** 3 / (G_CGS ** 1.5
                                                    * math.sqrt(rho))
    assert res['jeans_mass'] == pytest.approx(m_j_true, rel=2e-3)
    assert res['jeans_mass_msun'] == pytest.approx(3.92, rel=0.01)
    assert res['potential_cores'] == int(cloud_mass / res['jeans_mass'])


def test_bsh2_cshock_width_uses_neutral_ion_rate():
    """L ~ v_s / (<sigma v> x_i n), i.e. proportional to 1/x_i.

    Pre-fix the width used the ION collision frequency and came out
    7.08e9 cm (4.7e-4 AU) at n = 1e4, x_i = 1e-7 - about 1e7 too small -
    and changed by only 0.1% over four decades in x_i.
    """
    cs = shock_physics.CShock()
    n, x_i, v_s, b_field = 1e4, 1e-7, 1e6, 100e-6
    rho = n * shock_physics.MU_MOLECULAR * shock_physics.M_PROTON

    width = cs.shock_width(v_s, b_field, rho, x_i)
    nu_ni_true = 2e-9 * x_i * n
    assert width == pytest.approx(v_s / nu_ni_true, rel=1e-9)
    assert 1e17 < width < 1e18                 # pre-fix: 7.08e9

    # Must now scale as 1/x_i over four decades.
    wide = cs.shock_width(v_s, b_field, rho, 1e-3)
    assert width / wide == pytest.approx(1e-3 / 1e-7, rel=1e-6)


# =============================================================================
# C11 / B-COS-2 - cosmological_context
# =============================================================================

def test_c11_optical_depth_to_reionization():
    """tau must be O(0.05), the Planck 2018 value, not 7.9e38.

    Pre-fix, n_e0 was M_sun/Mpc^3 divided by a mass in grams -
    3.21e33 cm^-3 instead of 1.86e-7 - a factor 1.73e40.
    """
    model = cosmological_context.ReionizationModel()
    tau = model.optical_depth_reionization()
    assert 0.03 < tau < 0.10                    # pre-fix: 7.900e38
    assert tau == pytest.approx(0.054, abs=0.02)

    # And the underlying comoving hydrogen density must be right.
    cosmo = model.cosmo
    rho_b0 = (cosmo.Omega_b * cosmological_context.RHO_CRIT_0
              * cosmo.h ** 2 * M_SUN_G / PC_CM_MPC ** 3)
    n_nucleon = rho_b0 / 1.67e-24
    assert n_nucleon * 0.75 == pytest.approx(1.86e-7, rel=0.02)


PC_CM_MPC = 3.0857e24          # cm per Mpc


def test_bcos2_growth_factor_matches_exact_quadrature():
    """D(z) from the Heath (1977) integral, computed independently.

    Pre-fix, D used Omega_m(z)^0.55 (the growth INDEX) as a
    multiplicative factor: D(0) = 1.01 and D was 25-49% high
    (z = 0.5: 0.96421 vs 0.77318; z = 2: 0.61212 vs 0.42145).
    """
    from scipy.integrate import quad

    hmf = cosmological_context.HaloMassFunction()
    om, ol = hmf.cosmo.Omega_m, hmf.cosmo.Omega_L

    def e_of_a(a):
        return math.sqrt(om / a ** 3 + (1 - om - ol) / a ** 2 + ol)

    def d_unnorm(a):
        return e_of_a(a) * quad(lambda ap: 1.0 / (ap * e_of_a(ap)) ** 3,
                                0.0, a, limit=200)[0]

    assert hmf.growth_factor(0.0) == pytest.approx(1.0, rel=1e-12)
    for z, reference in [(0.5, 0.77318), (1.0, 0.61181),
                         (2.0, 0.42145), (3.0, 0.31884), (10.0, 0.11667)]:
        a = 1.0 / (1.0 + z)
        d_true = d_unnorm(a) / d_unnorm(1.0)
        assert d_true == pytest.approx(reference, rel=1e-4)   # sanity
        assert hmf.growth_factor(z) == pytest.approx(d_true, rel=1e-6)


def test_bcos2_cluster_abundance_no_longer_inverted():
    """Massive clusters must be RARER at z = 2 than at z = 0.

    Audit reference values, recomputed with the exact D(z):

        z     old D       exact D     ratio
        0.0   1.632e-5    1.606e-5    1.02
        0.5   4.453e-5    2.522e-5    1.77
        1.0   6.943e-5    1.936e-5    3.59
        2.0   5.876e-5    2.320e-6    25.3

    Pre-fix n(>1e14) at z = 2 was 3.6x the z = 0 value - cluster
    abundance was inverted in redshift.  It is now 0.14x.

    NOTE (left alone, see report): the residual rise between z = 0 and
    z = 0.5 comes from the (1+z)^3 in `rho_m` (a physical, not comoving,
    background density) and from B-COS-3/B-COS-5, none of which is in
    scope here.
    """
    hmf = cosmological_context.HaloMassFunction()
    counts = {z: hmf.cumulative_number_density(1e14, z, 'tinker')
              for z in (0.0, 0.5, 1.0, 2.0)}
    reference = {0.0: 1.606e-5, 0.5: 2.522e-5,
                 1.0: 1.936e-5, 2.0: 2.320e-6}
    for z, ref in reference.items():
        assert counts[z] == pytest.approx(ref, rel=0.02), z

    assert counts[2.0] < counts[0.0]        # pre-fix: 3.6x LARGER
    assert counts[2.0] / counts[0.0] == pytest.approx(0.14, abs=0.02)


# =============================================================================
# B-SNR-1/2/3/4/6 - supernova_remnant_physics
# =============================================================================

@pytest.fixture(scope="module")
def sedov_profiles():
    return supernova_remnant_physics.sedov_similarity_profiles(5.0 / 3.0)


def test_bsnr1_sedov_density_profile_not_inverted(sedov_profiles):
    """rho must be MAXIMUM at the shock and ~0 at the centre.

    Pre-fix ((1-eta^2)/(1-0.99^2))^0.3 gave 3.238 at eta = 0.011 and
    0.000 at eta = 1.0 - exactly upside down - with a x324 jump at the
    eta = 0.01 guard.
    """
    st = supernova_remnant_physics.SedovTaylorBlastwave()
    sol = st.solve(1e51, 1.0, 1000 * 3.156e7)

    assert sol.density_profile(1.0) == pytest.approx(1.0, rel=1e-9)
    assert sol.density_profile(0.011) < 1e-4          # pre-fix: 3.238
    etas = np.linspace(0.02, 1.0, 60)
    rho = np.array([sol.density_profile(e) for e in etas])
    assert np.all(np.diff(rho) > 0), "density must increase outward"
    # No discontinuity at the old eta = 0.01 guard.
    assert abs(sol.density_profile(0.0099) - sol.density_profile(0.0101)) < 1e-6


def test_bsnr1_sedov_profiles_satisfy_energy_integral(sedov_profiles):
    """The decisive check: Int (GV^2/2 + P/(g-1)) xi^2 dxi = 25/(16 pi xi0^5).

    This is E = const for the similarity solution and simultaneously
    validates xi_0 = 1.15167.
    """
    xi_0 = 1.15167
    required = 25.0 / (16.0 * math.pi * xi_0 ** 5)
    assert required == pytest.approx(0.2454878, abs=1e-6)
    assert sedov_profiles['energy_integral'] == pytest.approx(required,
                                                              rel=1e-3)


def test_bsnr2_sedov_energy_partition(sedov_profiles):
    """E_kin/E = 0.283, E_th/E = 0.717 (standard Sedov values).

    Pre-fix, `E_kin = 0.5 M_swept v_shock^2` gave 0.679 / 0.321 - the
    two essentially swapped (2.4x / 2.2x errors).
    """
    assert sedov_profiles['f_kinetic'] == pytest.approx(0.283, abs=0.002)
    assert sedov_profiles['f_thermal'] == pytest.approx(0.717, abs=0.002)

    ev = supernova_remnant_physics.SNREvolution()
    params = supernova_remnant_physics.SNRParameters(
        explosion_energy=1e51, ejecta_mass=3 * M_SUN_G,
        ambient_density=1.0, age=1000 * 3.156e7)
    state = ev.evolve(params)
    assert state.phase is supernova_remnant_physics.SNRPhase.SEDOV_TAYLOR
    assert state.kinetic_energy / 1e51 == pytest.approx(0.283, abs=0.002)
    assert state.thermal_energy / 1e51 == pytest.approx(0.717, abs=0.002)


def test_bsnr3_radius_continuous_across_phase_boundary():
    """R(t) must not jump at the free-expansion / Sedov boundary.

    Pre-fix it jumped x3.74 (0.787 -> 2.946 pc) because the transition
    time was built from a hardcoded v_ej = 1e9 cm/s while `evolve` used
    0.5 sqrt(2E/M_ej) t.
    """
    ev = supernova_remnant_physics.SNREvolution()
    energy, m_ej, n = 1e51, 3 * M_SUN_G, 1.0
    r_trans, t_trans = ev.transition_free_to_sedov(m_ej, n, energy)

    def radius(t):
        return ev.evolve(supernova_remnant_physics.SNRParameters(
            explosion_energy=energy, ejecta_mass=m_ej,
            ambient_density=n, age=t)).radius

    before, after = radius(t_trans * 0.999), radius(t_trans * 1.001)
    assert before / after == pytest.approx(1.0, rel=5e-3)   # pre-fix: 0.267
    assert before == pytest.approx(r_trans, rel=5e-3)


def test_bsnr4_1ghz_luminosity_unit_scale():
    """Sigma-D: W/m^2/Hz/sr -> CGS is x1e3, and L = 4 pi R^2 * pi Sigma.

    Pre-fix the code used x1e7 and an extra 4 pi, making L 4.00e4 too
    large; `analyze_snr()` reported 1326 Jy for a generic 1 kpc, 5 pc
    remnant - brighter than Cas A.
    """
    ev = supernova_remnant_physics.SNREvolution()
    state = ev.evolve(supernova_remnant_physics.SNRParameters(
        explosion_energy=1e51, ejecta_mass=3 * M_SUN_G,
        ambient_density=1.0, age=1000 * 3.156e7))

    syn = supernova_remnant_physics.SynchrotronEmission()
    lum = syn.luminosity_1ghz(state, 100e-6)

    d_pc = state.radius / supernova_remnant_physics.PC
    sigma_cgs = 1e-21 * d_pc ** (-17.0 / 5.0) * 1e3
    lum_true = 4 * math.pi * state.radius ** 2 * math.pi * sigma_cgs
    assert lum == pytest.approx(lum_true, rel=1e-12)

    # Sanity: a generic remnant is a few tens of mJy at 1 kpc, not 1326 Jy.
    flux_jy = supernova_remnant_physics.analyze_snr()['radio']['flux_1ghz_jy']
    assert 1e-4 < flux_jy < 10.0            # pre-fix: 1326 Jy


@pytest.mark.parametrize("t_boundary", [1e4, 1e5, 1e6, 10 ** 7.5])
@pytest.mark.parametrize("metallicity", [0.1, 1.0])
def test_bsnr6_cooling_function_is_continuous(t_boundary, metallicity):
    """Lambda(T) must be continuous across every branch boundary.

    Pre-fix jumps: x3162 at 1e4 K, x3.16 at 1e5 K, x0.095 at 1e6 K and
    x2.7 at 10^7.5 K (Z = 0.1).
    """
    xray = supernova_remnant_physics.XRayThermalEmission()
    below = xray.cooling_function(t_boundary * (1 - 1e-9), metallicity)
    above = xray.cooling_function(t_boundary * (1 + 1e-9), metallicity)
    assert above / below == pytest.approx(1.0, rel=1e-6)


def test_bsnr6_cooling_function_shape():
    """Peak near 1e5 K, steep Lyman-alpha cut-off below 1e4 K."""
    xray = supernova_remnant_physics.XRayThermalEmission()
    peak = xray.cooling_function(1e5, 1.0)
    assert peak > xray.cooling_function(1e4, 1.0)
    assert peak > xray.cooling_function(1e6, 1.0)
    assert xray.cooling_function(5e3, 1.0) < 1e-5 * peak
