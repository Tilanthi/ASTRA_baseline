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
Numerical regression tests for the August-2026 physics audit fixes (worker B set).

Every test asserts a value derived INDEPENDENTLY of the module under test --
from an analytic result, a published number, or a second implementation written
here -- and records the pre-fix (wrong) value in a comment.

Audit report:  /workspace/astra_baseline_audit_aug2026/02_physics_audit.md
Fix report:    /workspace/astra_baseline_audit_aug2026/05_physics_fixes_B.md

Modules covered: physics, radio_surveys, star_formation, hii_region_physics,
radiative_transfer, turbulence_analysis, sph_gas_dynamics, source_extraction,
time_series_analysis, inference, sed_fitting, radial_velocity,
exoplanet_transit, interferometry.
"""

import math
import warnings

import numpy as np
import pytest

from astra_core.astro_physics import physics as ph
from astra_core.astro_physics import radio_surveys as rs
from astra_core.astro_physics import star_formation as sf
from astra_core.astro_physics import hii_region_physics as hii
from astra_core.astro_physics import radiative_transfer as rt
from astra_core.astro_physics import turbulence_analysis as ta
from astra_core.astro_physics import sph_gas_dynamics as sph
from astra_core.astro_physics import source_extraction as se
from astra_core.astro_physics import time_series_analysis as ts
from astra_core.astro_physics import sed_fitting as sed
from astra_core.astro_physics import radial_velocity as rv
from astra_core.astro_physics import exoplanet_transit as et
from astra_core.astro_physics import interferometry as itf
from astra_core.astro_physics.inference import BayesianSwarmInference
from astra_core.astro_physics.physics import PhysicsEngine, ForwardModel


# =============================================================================
# H1 -- physics.main_sequence_mass_luminosity
# =============================================================================

@pytest.mark.parametrize("m_msun, l_expected", [
    (0.2, 0.23 * 0.2 ** 2.3),        # 0.005677   (pre-fix: 0.0247,  4.35x high)
    (1.0, 1.0),                      # 1.0        (pre-fix: 1.0, the only right one)
    (10.0, 1.4 * 10 ** 3.5),         # 4427.2     (pre-fix: 3162, 0.714x)
    (54.9, 1.4 * 54.9 ** 3.5),       # 1.716e6    (pre-fix: 1.226e6, 0.71x)
    (55.1, 32000.0 * 55.1),          # 1.763e6    (pre-fix: 55.1 (!), 3.1e-5x)
])
def test_h1_mass_luminosity_normalisations(m_msun, l_expected):
    """Standard piecewise M-L relation (Duric 2004): the coefficients
    0.23 / 1 / 1.4 / 32000 had all been dropped."""
    model = ph.StellarStructureModel()
    l_code = model.main_sequence_mass_luminosity(
        m_msun * ph.PhysicalConstants.M_sun) / ph.PhysicalConstants.L_sun
    assert l_code == pytest.approx(l_expected, rel=1e-12)


def test_h1_no_discontinuity_at_55_msun():
    """Pre-fix the relation jumped by a factor 2.2e4 across 55 Msun."""
    model = ph.StellarStructureModel()
    below = model.main_sequence_mass_luminosity(54.999 * ph.PhysicalConstants.M_sun)
    above = model.main_sequence_mass_luminosity(55.001 * ph.PhysicalConstants.M_sun)
    # the published fit itself is discontinuous by 1.8% here; anything beyond
    # a few per cent means a coefficient is missing again
    assert above / below == pytest.approx(1.0, abs=0.05)


# =============================================================================
# M5 -- radio_surveys.estimate_luminosity (was a two-line import stub)
# =============================================================================

def _dl_simpson(z, H0=67.66, Om=0.30966, n=20001):
    """Independent flat-LCDM luminosity distance by Simpson's rule."""
    zs = np.linspace(0.0, z, n)
    e = 1.0 / np.sqrt(Om * (1 + zs) ** 3 + (1 - Om))
    integral = (zs[1] - zs[0]) / 3.0 * (
        e[0] + e[-1] + 4 * e[1:-1:2].sum() + 2 * e[2:-2:2].sum())
    return (1 + z) * (2.99792458e5 / H0) * integral


@pytest.mark.parametrize("z", [0.1, 0.5, 1.0, 2.0])
def test_m5_luminosity_distance_matches_independent_quadrature(z):
    assert rs.luminosity_distance_mpc(z) == pytest.approx(_dl_simpson(z), rel=1e-8)


def test_m5_luminosity_distance_planck18_anchor():
    """astropy's Planck18.luminosity_distance(1) = 6797.4 Mpc."""
    assert rs.luminosity_distance_mpc(1.0) == pytest.approx(6797.4, rel=1e-4)


def test_m5_estimate_luminosity_k_correction():
    """L_nu = 4 pi D_L^2 S_nu / (1+z)^(1+alpha).  Pre-fix: returned None
    (astropy present) or raised ModuleNotFoundError (astropy absent)."""
    z, alpha, s_jy = 0.5, -0.7, 2.5
    d_cm = rs.luminosity_distance_mpc(z) * rs.MPC
    expected = 4 * np.pi * d_cm ** 2 * s_jy * 1e-23 / (1 + z) ** (1 + alpha)
    assert rs.estimate_luminosity(s_jy, z, 1.4e9, alpha) == pytest.approx(expected, rel=1e-12)


# =============================================================================
# B-SF-1/2/3/5 -- star_formation
# =============================================================================

def test_bsf1_hbeta_sfr_calibration():
    """Case B: L(Ha) = 2.86 L(Hb) => C_Hb = 2.86 C_Ha.
    Pre-fix the code DIVIDED by 2.86, so SFR was low by 2.86^2 = 8.18."""
    tracer = sf.StarFormationRateTracer('kroupa')
    l_ha = 1.0 / 5.25e-42                      # L(Ha) for SFR = 1 Msun/yr
    l_hb = l_ha / 2.86
    sfr = tracer.sfr_from_luminosity(l_hb, sf.SFTRindicator.H_BETA)
    assert sfr == pytest.approx(1.0, rel=1e-9)  # pre-fix: 0.12226


def test_bsf2_hcn_calibration():
    """Gao & Solomon (2004): L_IR/L'_HCN = 900 Lsun/(K km/s pc^2), through the
    KE12 TIR calibration => 1.337e-7.  Pre-fix: 1.5e-4 (1122x too high)."""
    tracer = sf.StarFormationRateTracer('kroupa')
    expected = 3.88e-44 * 900.0 * 3.828e33
    assert expected == pytest.approx(1.337e-7, rel=1e-3)
    sfr = tracer.sfr_from_luminosity(1e8, sf.SFTRindicator.HCN)
    assert sfr == pytest.approx(13.37, rel=1e-3)    # pre-fix: 1.50e4
    # round trip
    back = tracer.luminosity_from_sfr(sfr, sf.SFTRindicator.HCN)
    assert back == pytest.approx(1e8, rel=1e-9)


@pytest.mark.parametrize("age", [0.0, 10.0, 50.0])
def test_bsf3_population_age_not_double_counted(age):
    """create_stellar_population(age_myrs=10) used to return stars at 20 Myr."""
    pop = sf.create_stellar_population(50, age_myrs=age, seed=3)
    assert all(star.age_myrs == pytest.approx(age) for star in pop.stars)


@pytest.mark.parametrize("m_init", [8.0, 15.0, 25.0, 40.0, 100.0])
def test_bsf5_yields_conserve_mass(m_init):
    """ejecta + remnant must not exceed the progenitor mass.
    Pre-fix: 100 Msun -> 80 + 70 = 150 Msun."""
    fb = sf.SupernovaFeedback()
    ejecta = fb.yields(m_init)['total']
    remnant = sf.StellarEvolution.remnant_mass(m_init)
    assert ejecta + remnant == pytest.approx(m_init, rel=1e-12)
    assert ejecta >= 0.0


def test_bsf5_metal_yields_are_subsets_of_ejecta():
    fb = sf.SupernovaFeedback()
    for m in (10.0, 20.0, 40.0):
        y = fb.yields(m)
        assert y['O'] + y['C'] + y['Fe'] <= y['total']


# =============================================================================
# B-HII-1/2/3 -- hii_region_physics
# =============================================================================

def test_bhii1_qh_matches_martins05_grid():
    """Q_H(T_eff) is now log-linear interpolation of Martins, Schaerer &
    Hillier (2005) Table 1 (class V).  Pre-fix: 10^(3e-4 T + 39), i.e.
    2.85e52 at 44 850 K (713x the Martins value) and 1e54 at 50 kK."""
    st = hii.StromgrenSphere()
    for t_eff, log_q in hii.StromgrenSphere._MSH05_DWARF_TEFF_LOGQ0:
        assert st.ionizing_photon_rate(temperature=t_eff) == pytest.approx(
            10 ** log_q, rel=1e-9)
    # O3V-like star: Martins+05 gives ~4e49, NOT 2.85e52
    q = st.ionizing_photon_rate(temperature=44850.0)
    assert 3e49 < q < 6e49


def test_bhii1_qh_is_continuous_and_monotonic():
    """Pre-fix there was a hard 0 -> 1e48 jump at exactly 30 000 K."""
    st = hii.StromgrenSphere()
    temps = np.linspace(30500, 44600, 400)
    q = np.array([st.ionizing_photon_rate(temperature=t) for t in temps])
    assert np.all(np.diff(q) > 0)
    assert np.max(np.abs(np.diff(np.log10(q)))) < 0.05     # no jumps
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q_lo = st.ionizing_photon_rate(temperature=30000.0)
    assert q_lo > 0                                        # pre-fix: exactly 0


def test_bhii1_extrapolation_warns():
    st = hii.StromgrenSphere()
    with pytest.warns(RuntimeWarning):
        st.ionizing_photon_rate(temperature=50000.0)
    with pytest.warns(RuntimeWarning):
        st.ionizing_photon_rate(temperature=25000.0)


def test_bhii3_oiii_temperature_osterbrock_anchor():
    """Osterbrock & Ferland (2006) eq. 5.4: j(4959+5007)/j(4363) =
    7.90 exp(3.29e4/T)  =>  R(1e4 K) = 4.72e-3, C = 1/7.90 = 0.1266.
    Pre-fix C = 0.275 (2.17x too large) biased T_e low by 18-45%."""
    nd = hii.NebularDiagnosticsCalculator()
    assert nd.oiii_temperature(4.72e-3) == pytest.approx(1e4, rel=5e-3)
    assert nd.oiii_temperature(0.0100) == pytest.approx(12961, rel=1e-3)  # was 9998
    # round trip against the closed form
    for t_e in (8000.0, 12000.0, 18000.0):
        ratio = (1.0 / 7.90) * math.exp(-32900.0 / t_e)
        assert nd.oiii_temperature(ratio) == pytest.approx(t_e, rel=1e-9)


def test_bhii2_oxygen_abundance_izotov06():
    """Izotov et al. (2006) eqs. 3-4.  Pre-fix the exp(dE/kT) term was absent
    entirely: O/H = 0.048 (231x too high) and RISING with T_e."""
    nd = hii.NebularDiagnosticsCalculator()
    r3, r2, n_e = 5.0, 2.0, 100.0
    for t4 in (0.7, 1.0, 1.5, 2.0):
        t = t4 * 1e4
        x = 1e-4 * n_e * t4 ** -0.5
        o_plus = 10 ** (math.log10(r2) + 5.961 + 1.676 / t4
                        - 0.40 * math.log10(t4) - 0.034 * t4
                        + math.log10(1 + 1.35 * x) - 12.0)
        o_2plus = 10 ** (math.log10(r3) + 6.200 + 1.251 / t4
                         - 0.55 * math.log10(t4) - 0.014 * t4 - 12.0)
        assert nd.oxygen_abundance(t, t, r3, r2, n_e=n_e) == pytest.approx(
            o_plus + o_2plus, rel=1e-9)
    # physically sensible value and the right sign of dO/dT
    oh = nd.oxygen_abundance(1e4, 1e4, r3, r2)
    assert 1e-4 < oh < 5e-4                       # pre-fix: 0.048
    assert 12 + math.log10(oh) == pytest.approx(8.3, abs=0.2)
    hot = nd.oxygen_abundance(2e4, 2e4, r3, r2)
    assert hot < oh / 3.0                          # pre-fix: hot > cool


# =============================================================================
# H6/H7/H8 -- radiative_transfer
# =============================================================================

def _uniform_sphere_beta(tau):
    return 1.5 / tau * (1 - 2 / tau ** 2 + (2 / tau + 2 / tau ** 2) * math.exp(-tau))


def test_h7_uniform_sphere_small_tau_series():
    """Series expansion of the Osterbrock uniform-sphere beta is
    1 - 3 tau/8 + tau^2/10 - tau^3/120, NOT the LVG 1 - tau/2 + tau^2/6.
    Pre-fix, beta(0.0099) = 0.9950663 vs the exact 0.9962973."""
    g = rt.EscapeGeometry.UNIFORM_SPHERE
    assert rt.escape_probability(0.0099, g) == pytest.approx(0.9962973, rel=1e-6)
    # Continuity across the tau = 0.01 branch switch: the step in tau is 2e-6
    # and dbeta/dtau = -3/8, so removing that slope must leave a discontinuity
    # below 1e-7 (it is 1.2e-8, the tau^4 truncation of the series).
    # Pre-fix the discontinuity was 1.16e-3.
    below = rt.escape_probability(0.009999, g)
    above = rt.escape_probability(0.010001, g)
    assert abs((below - above) - 0.375 * 2e-6) < 1e-7


@pytest.mark.parametrize("tau", [0.02, 0.5, 1.0, 10.0, 49.9, 50.1, 100.0, 1000.0])
def test_h6_static_sphere_equals_uniform_sphere(tau):
    """Pre-fix STATIC_SPHERE switched to a Doppler-core asymptote above
    tau = 50: beta jumped x2.74 at tau = 50 and was x3.78 off at tau = 1000."""
    b_static = rt.escape_probability(tau, rt.EscapeGeometry.STATIC_SPHERE)
    b_uniform = rt.escape_probability(tau, rt.EscapeGeometry.UNIFORM_SPHERE)
    assert b_static == pytest.approx(b_uniform, rel=1e-12)
    assert b_uniform == pytest.approx(_uniform_sphere_beta(tau), rel=1e-9)


def test_h6_no_jump_at_tau_50():
    for g in (rt.EscapeGeometry.UNIFORM_SPHERE, rt.EscapeGeometry.STATIC_SPHERE):
        lo = rt.escape_probability(49.999, g)
        hi = rt.escape_probability(50.001, g)
        assert hi == pytest.approx(lo, rel=1e-4)      # pre-fix ratio: 2.740


def test_h6_large_tau_limit():
    """beta * tau -> 1.5 for a uniform sphere."""
    for tau in (1e3, 1e5):
        b = rt.escape_probability(tau, rt.EscapeGeometry.UNIFORM_SPHERE)
        assert b * tau == pytest.approx(1.5, rel=1e-4)


def _three_level_co(einstein_a, freqs, pairs=None):
    levels = [rt.MolecularLevel(J=0, energy=0.0, weight=1),
              rt.MolecularLevel(J=1, energy=5.53, weight=3),
              rt.MolecularLevel(J=2, energy=16.60, weight=5)]
    rates = rt.CollisionRates('H2', np.array([10.0, 20.0, 40.0]),
                              np.full((3, 2), 3.3e-11), [(1, 0), (2, 1)])
    return rt.MolecularData('CO', levels, np.asarray(einstein_a),
                            np.asarray(freqs), [rates],
                            radiative_transitions=pairs)


def test_h8_lamda_style_ladder_mapping():
    """LAMDA files list only allowed transitions. For a 3-level CO with
    einstein_A = [A10, A21] the old packed-index assumption assigned A21 and
    nu21 to the dipole-FORBIDDEN 2->0 transition and silently DROPPED the real
    2->1 line."""
    data = _three_level_co([7.203e-8, 6.910e-7], [115.271e9, 230.538e9])
    solver = rt.StatisticalEquilibriumSolver(data)
    pairs = [(up, low) for up, low, _, _ in solver.radiative]
    assert pairs == [(1, 0), (2, 1)]               # pre-fix: [(1, 0), (2, 0)]
    assert solver.radiative[1][2] == pytest.approx(6.910e-7)
    assert solver.radiative[1][3] == pytest.approx(230.538e9)


def test_h8_packed_ordering_still_supported():
    data = _three_level_co([7.203e-8, 0.0, 6.910e-7],
                           [115.271e9, 1.0, 230.538e9])
    solver = rt.StatisticalEquilibriumSolver(data)
    assert [(u, l) for u, l, _, _ in solver.radiative] == [(1, 0), (2, 1)]


def test_h8_explicit_transitions_are_honoured():
    data = _three_level_co([7.203e-8, 6.910e-7], [115.271e9, 230.538e9],
                           pairs=[(1, 0), (2, 1)])
    solver = rt.StatisticalEquilibriumSolver(data)
    assert [(u, l) for u, l, _, _ in solver.radiative] == [(1, 0), (2, 1)]


def test_h8_ambiguous_layout_raises_instead_of_guessing():
    data = _three_level_co([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="cannot infer"):
        rt.StatisticalEquilibriumSolver(data)


# =============================================================================
# H11/H12 -- turbulence_analysis
# =============================================================================

def test_h11_structure_function_white_noise():
    """For uncorrelated noise of variance sigma^2, S_2(l) = 2 sigma^2 at EVERY
    lag. Pre-fix, dx = int(lag cos theta) made 32 of 36 angles give a zero
    displacement at lag 1, so S_2(1) was ~11x too low."""
    rng = np.random.default_rng(0)
    sigma = 3.0
    field = rng.normal(0.0, sigma, (128, 128))
    res = ta.StructureFunctionAnalysis().compute_2d(field, order=2, max_lag=8)
    assert res.S_p[0] == pytest.approx(2 * sigma ** 2, rel=0.05)   # was ~0.087x
    for value in res.S_p:
        assert value == pytest.approx(2 * sigma ** 2, rel=0.05)


def _brute_force_sf2(field, lag):
    """Independent O(N^4) all-pairs structure function in the same annulus."""
    ny, nx = field.shape
    total, count = 0.0, 0
    for y in range(ny):
        for x in range(nx):
            for yy in range(ny):
                for xx in range(nx):
                    d = math.hypot(xx - x, yy - y)
                    if lag - 0.5 <= d < lag + 0.5:
                        total += (field[y][x] - field[yy][xx]) ** 2
                        count += 1
    return total / count


@pytest.mark.parametrize("lag", [1, 2, 3])
def test_h11_structure_function_matches_brute_force(lag):
    rng = np.random.default_rng(5)
    field = rng.normal(size=(32, 32))
    res = ta.StructureFunctionAnalysis().compute_2d(
        field, order=2, max_lag=4, n_angles=10 ** 6)
    assert res.S_p[lag - 1] == pytest.approx(_brute_force_sf2(field, lag), rel=1e-12)


def test_h12_spectral_pca_centres_each_channel():
    """A cube that is (mean spectrum + 1% noise) has no real structure: PC1
    must NOT carry ~all the variance. Pre-fix the code subtracted a single
    global scalar, so explained_variance_ratio[0] = 0.99915."""
    rng = np.random.default_rng(2)
    spectrum = np.exp(-((np.arange(24) - 12) / 3.0) ** 2)
    cube = np.tile(spectrum, (16, 16, 1)) * (1 + 0.01 * rng.normal(size=(16, 16, 24)))
    out = ta.SpectralPCA(n_components=3).fit(cube)
    assert out['explained_variance_ratio'][0] < 0.5
    # the score maps must be centred (mean ~ 0) once the channels are centred
    assert abs(float(np.mean(out['score_maps'][0]))) < 1e-8


# =============================================================================
# H9/H10 -- SPH kernels
# =============================================================================

@pytest.mark.parametrize("kernel", ['cubic_spline', 'wendland_c4', 'gaussian'])
def test_h10_kernels_normalised_over_their_support(kernel):
    """The Gaussian is truncated at the 2h neighbour-search radius; pre-fix it
    was normalised over infinite support, so int W dV = 0.953988 and the
    lattice density came out 0.952x the truth."""
    from scipy.integrate import quad
    fn = getattr(sph.SPHKernel, kernel)
    integral = quad(lambda r: 4 * np.pi * r ** 2 * fn(np.array([r]), 1.0)[0],
                    0, 3, limit=200)[0]
    assert integral == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("kernel_type", [sph.KernelType.CUBIC_SPLINE,
                                         sph.KernelType.WENDLAND,
                                         sph.KernelType.GAUSSIAN])
@pytest.mark.parametrize("q", [0.2, 0.5, 0.9, 1.3, 1.8, 1.99])
def test_h9_force_kernel_is_derivative_of_density_kernel(kernel_type, q):
    """Pre-fix compute_forces hardcoded cubic_spline_derivative for every
    kernel type, so WENDLAND/GAUSSIAN runs mixed two different kernels."""
    w = sph.SPHKernel.get_kernel(kernel_type)
    dw = sph.SPHKernel.get_kernel_derivative(kernel_type)
    eps = 1e-6
    numeric = (w(np.array([q + eps]), 1.0)[0] - w(np.array([q - eps]), 1.0)[0]) / (2 * eps)
    assert dw(np.array([q]), 1.0)[0] == pytest.approx(numeric, rel=1e-5)


def test_h9_simulation_uses_matching_derivative():
    particles = [sph.SPHParticle(pos=np.zeros(3), vel=np.zeros(3), mass=1.0)]
    sim = sph.SPHSimulation(particles, kernel_type=sph.KernelType.WENDLAND)
    assert sim.kernel_derivative is sph.SPHKernel.wendland_c4_derivative


def test_h9_quartic_spline_raises_instead_of_silently_using_cubic():
    with pytest.raises(NotImplementedError):
        sph.SPHKernel.get_kernel(sph.KernelType.QUARTIC_SPLINE)


def test_h10_gaussian_truncation_constant():
    """C = erf(2) - (4/sqrt(pi)) exp(-4) = 0.9539883845."""
    from scipy.special import erf
    expected = erf(2.0) - 4.0 / np.sqrt(np.pi) * np.exp(-4.0)
    assert expected == pytest.approx(0.9539882943, rel=1e-9)
    assert sph.SPHKernel.GAUSSIAN_TRUNCATION_NORM == pytest.approx(expected, rel=1e-12)


# =============================================================================
# H13 (a) -- sph_gas_dynamics.FilamentFinder
# =============================================================================

def _synthetic_filament(n=200, length=80.0, pa_deg=30.0, sigma=3.0,
                        cx=100.0, cy=100.0, amp=10.0):
    """Gaussian ridge of known length, width and position angle."""
    y, x = np.mgrid[0:n, 0:n]
    th = np.radians(pa_deg)
    dx, dy = x - cx, y - cy
    s = dx * np.cos(th) + dy * np.sin(th)
    t = -dx * np.sin(th) + dy * np.cos(th)
    return amp * np.exp(-0.5 * (t / sigma) ** 2) * (np.abs(s) <= length / 2)


def test_h13a_separate_filaments_are_not_merged():
    """Pre-fix there was no connected-component labelling: four well-separated
    parallel filaments came back as ONE object with length 5406.7."""
    image = np.zeros((200, 200))
    for cy in (40, 80, 120, 160):
        image = np.maximum(image, _synthetic_filament(200, 90.0, 0.0, 2.0, 100.0, cy))
    fils = sph.FilamentFinder(min_length=5.0).find_filaments(image, threshold=1.0)
    assert len(fils) == 4
    for fil in fils:
        assert fil.length == pytest.approx(90.0, rel=0.15)


def test_h13a_recovered_geometry():
    """Length/width/PA of a synthetic ridge, length 80 px, PA 30 deg,
    sigma 3 px -> mask half-width 3 sqrt(2 ln 10) = 6.44 px, so the
    area/length width is 12.88 px.
    Pre-fix (no thinning): length 1504.5 (18.8x) and width identically 1.0."""
    image = _synthetic_filament()
    fils = sph.FilamentFinder(min_length=5.0).find_filaments(image, threshold=1.0)
    assert len(fils) == 1
    fil = fils[0]
    # <=15% from the digital-line (<=8.2%) and end-erosion biases
    assert fil.length == pytest.approx(80.0, rel=0.15)
    assert fil.width == pytest.approx(2 * 3.0 * math.sqrt(2 * math.log(10)), rel=0.20)
    assert fil.width != pytest.approx(1.0, abs=1e-9)
    assert fil.orientation == pytest.approx(30.0, abs=1.0)


def test_h13a_geometry_also_correct_without_skimage(monkeypatch):
    """Same measurement through the no-scikit-image path (Zhang-Suen fallback),
    which is the path the audited environment actually took. Pre-fix that path
    did no thinning at all: length 1504.5 for an 80 px filament, width == 1.0."""
    import sys
    monkeypatch.setitem(sys.modules, 'skimage.morphology', None)
    image = _synthetic_filament()
    fils = sph.FilamentFinder(min_length=5.0).find_filaments(image, threshold=1.0)
    assert len(fils) == 1
    assert fils[0].length == pytest.approx(80.0, rel=0.15)
    assert fils[0].width == pytest.approx(
        2 * 3.0 * math.sqrt(2 * math.log(10)), rel=0.20)


def test_h13a_three_dimensional_input_without_skimage_raises(monkeypatch):
    """The 2-D-only fallback must refuse a cube rather than silently returning
    the un-thinned mask."""
    import sys
    monkeypatch.setitem(sys.modules, 'skimage.morphology', None)
    cube = np.zeros((8, 8, 8))
    cube[3:5, 3:5, :] = 10.0
    with pytest.raises(NotImplementedError):
        sph.FilamentFinder().find_filaments(cube, threshold=1.0)


def test_h13a_zhang_suen_thins_to_one_pixel():
    """The no-skimage fallback used to be `skeleton = mask` (no thinning)."""
    mask = np.zeros((40, 60), dtype=bool)
    mask[15:25, 10:50] = True                      # 10 x 40 bar
    skel = sph.FilamentFinder._zhang_suen(mask)
    assert skel.sum() < mask.sum() / 5             # pre-fix: skel.sum() == 400
    # one pixel per column over the interior
    interior = skel[:, 20:40]
    assert np.all(interior.sum(axis=0) == 1)


def test_h13a_zhang_suen_preserves_a_one_pixel_line():
    mask = np.zeros((40, 60), dtype=bool)
    mask[20, 10:50] = True
    assert np.array_equal(sph.FilamentFinder._zhang_suen(mask), mask)


def test_h13a_geodesic_length_of_a_digital_line():
    """8-connected digital straight line: exact at PA 0/45/90, and never more
    than 8.24% (= sqrt(1+(sqrt2-1)^2)) above the true length in between."""
    for pa, true_len in ((0.0, 100.0), (45.0, 100.0), (90.0, 100.0), (30.0, 100.0)):
        img = np.zeros((260, 260))
        for i in range(101):
            img[int(round(80 + i * np.sin(np.radians(pa)))),
                int(round(80 + i * np.cos(np.radians(pa))))] = 1.0
        measured = sph.FilamentFinder._geodesic_length(np.argwhere(img > 0))
        assert true_len <= measured <= true_len * 1.0824 + 1e-6


# =============================================================================
# H13 (b) -- source_extraction.FilamentFinder
# =============================================================================

def _blob_and_filament(ny=80, nx=120):
    blob = [[10.0 * math.exp(-0.5 * (((x - 60) ** 2 + (y - 40) ** 2) / 8.0 ** 2))
             for x in range(nx)] for y in range(ny)]
    fil = [[10.0 * math.exp(-0.5 * ((y - 40) / 3.0) ** 2)
            for x in range(nx)] for y in range(ny)]
    return blob, fil


def test_h13b_structure_tensor_anisotropy_is_not_identically_one():
    """Pre-fix J was the un-smoothed rank-1 outer product, so lambda_2 == 0 and
    the anisotropy was 1.000000 at every pixel of a CIRCULARLY SYMMETRIC blob."""
    blob, fil = _blob_and_filament()
    for scale, max_blob_anis in ((8.0, 0.4), (12.0, 0.2)):
        finder = se.FilamentFinder(scale=scale)
        results = {}
        for name, image in (('blob', blob), ('fil', fil)):
            gx, gy = finder._compute_gradients(image)
            jxx, jxy, jyy = finder._compute_structure_tensor(gx, gy)
            vals = []
            for y in range(30, 50):
                for x in range(50, 70):
                    l1, l2, _ = finder._eigenvalues(jxx[y][x], jxy[y][x], jyy[y][x])
                    if l1 + l2 > 1e-12:
                        vals.append((l1 - l2) / (l1 + l2))
            results[name] = sum(vals) / len(vals)
        assert results['blob'] < max_blob_anis        # pre-fix: exactly 1.0
        assert results['fil'] > 0.95                  # a real ridge stays ~1


def test_h13b_tracer_follows_the_ridge():
    """Pre-fix the tracer added a spurious +pi/2 and stepped ACROSS the ridge:
    0 filaments on a textbook synthetic filament."""
    _, fil = _blob_and_filament()
    net = se.FilamentFinder(scale=3.0, min_length=10,
                            min_contrast=0.1).find(fil, threshold=1.0)
    assert len(net.filaments) > 10                    # pre-fix: 0
    longest = max(net.filaments, key=lambda f: f.length)
    assert longest.length > 80
    # the ridge is horizontal: the spine must not wander in y
    ys = [p[1] for p in longest.spine_points]
    assert max(ys) - min(ys) <= 1.0


# =============================================================================
# C16 -- source_extraction.build_catalog reads the column-density map
# =============================================================================

def _two_blob_image(n=40):
    return [[10 * math.exp(-0.5 * (((x - 12) ** 2 + (y - 12) ** 2) / 9.0))
             + 8 * math.exp(-0.5 * (((x - 28) ** 2 + (y - 28) ** 2) / 9.0))
             for x in range(n)] for y in range(n)]


def test_c16_catalog_masses_scale_with_the_column_map():
    """Pre-fix `column_density_map` was used only for len(): two column maps
    differing by 100x produced IDENTICAL catalogue masses."""
    image = _two_blob_image()
    dend = se.DendrogramExtractor(min_value=1.0, min_delta=1.0).extract(image)
    builder = se.CoreCatalogBuilder(distance_pc=140.0)
    masses = {}
    for scale in (1e21, 1e23):
        col = [[v / 10.0 * scale for v in row] for row in image]
        cores = builder.build_catalog(dend, col, pixel_scale_arcsec=6.0)
        assert cores
        masses[scale] = [c.mass for c in cores]
    for m_lo, m_hi in zip(masses[1e21], masses[1e23]):
        assert m_hi / m_lo == pytest.approx(100.0, rel=1e-9)


def test_c16_mass_equals_mean_column_times_area():
    """Independent check of the mass: M = <N> mu m_H Omega d^2."""
    image = _two_blob_image()
    dend = se.DendrogramExtractor(min_value=1.0, min_delta=1.0).extract(image)
    col = [[v / 10.0 * 1e22 for v in row] for row in image]
    builder = se.CoreCatalogBuilder(distance_pc=140.0)
    cores = builder.build_catalog(dend, col, pixel_scale_arcsec=6.0)
    leaves = {n.node_id: n for n in dend.nodes if n.is_leaf}
    pixel_sr = (6.0 * math.pi / (180.0 * 3600.0)) ** 2
    for core in cores:
        node = leaves[core.core_id]
        cols = [col[y][x] for (x, y) in node.pixels]
        expected = builder.column_to_mass(sum(cols) / len(cols),
                                          len(cols) * pixel_sr)
        assert core.mass == pytest.approx(expected, rel=1e-12)
        assert core.column_density == pytest.approx(max(cols), rel=1e-12)


# =============================================================================
# C15 -- time_series_analysis significance layer
# =============================================================================

def test_c15_lomb_scargle_white_noise_false_alarm_rate():
    """Pre-fix the Horne-Baliunas normalisation was inflated by N/2:
    300/300 white-noise trials had FAP < 0.05 and max power ~617."""
    rng = np.random.default_rng(1)
    analyzer = ts.PowerSpectrumAnalyzer()
    z_max, faps = [], []
    for _ in range(40):
        t = np.sort(rng.uniform(0, 100, 200))
        series = ts.TimeSeries(time=t, values=rng.normal(size=200))
        res = analyzer.periodogram(series, method='lomb_scargle')
        z_max.append(res.best_power)
        faps.append(res.false_alarm_probability)
    assert np.mean(z_max) == pytest.approx(6.0, abs=2.0)   # ~ln(N_indep)
    assert np.mean(np.array(faps) < 0.05) < 0.25           # pre-fix: 1.000


def test_c15_lomb_scargle_still_finds_a_real_period():
    t = np.arange(400.0)
    values = np.sin(2 * np.pi * t / 25.0) + 0.1 * np.random.default_rng(0).normal(size=400)
    res = ts.PowerSpectrumAnalyzer().periodogram(
        ts.TimeSeries(time=t, values=values), method='lomb_scargle')
    assert res.best_period == pytest.approx(25.0, rel=0.02)
    assert res.false_alarm_probability < 1e-6
    assert res.signal_type == ts.SignalType.PERIODIC


def test_c15_ccf_pvalue_calibration():
    """The permutation null must use the same max-over-lags statistic.
    Pre-fix it used a single (zero) lag: 199/200 pairs of INDEPENDENT
    white-noise series came out with p < 0.05."""
    rng = np.random.default_rng(7)
    cc = ts.CrossCorrelationAnalyzer()
    t = np.arange(256.0)
    p_values = []
    for _ in range(60):
        a = ts.TimeSeries(time=t, values=rng.normal(size=256))
        b = ts.TimeSeries(time=t, values=rng.normal(size=256))
        p_values.append(cc.peak_lag(a, b)['p_value'])
    assert np.mean(np.array(p_values) < 0.05) < 0.20       # pre-fix: 0.995


def test_c15_ccf_still_detects_a_real_lag():
    rng = np.random.default_rng(3)
    t = np.arange(256.0)
    x = rng.normal(size=256)
    y = np.roll(x, 10) + 0.2 * rng.normal(size=256)
    out = ts.CrossCorrelationAnalyzer().peak_lag(
        ts.TimeSeries(time=t, values=x), ts.TimeSeries(time=t, values=y))
    assert out['peak_lag'] == pytest.approx(10.0)
    assert out['p_value'] < 0.05


@pytest.mark.parametrize("method", ['fft', 'lomb_scargle'])
def test_c15_significance_is_unit_independent(method):
    """Pre-fix, the same white-noise series scaled by 1e3 flipped from
    'white_noise' with FAP 1 to 'periodic' with FAP 0."""
    rng = np.random.default_rng(11)
    base = rng.normal(size=512)
    t = np.arange(512.0)
    verdicts, faps = set(), []
    for scale in (1e-3, 1.0, 1e3):
        res = ts.PowerSpectrumAnalyzer().periodogram(
            ts.TimeSeries(time=t, values=base * scale), method=method)
        verdicts.add(res.signal_type)
        faps.append(res.false_alarm_probability)
    assert len(verdicts) == 1
    assert faps[0] == pytest.approx(faps[1]) == pytest.approx(faps[2])


# =============================================================================
# M1 -- inference.BayesianSwarmInference uncertainties
# =============================================================================

class _Line(ForwardModel):
    def parameter_names(self):
        return ['a', 'b']

    def observable_names(self):
        return ['y']

    def predict(self, parameters):
        return {}

    def chi_squared(self, parameters, observations):
        x, y, s = observations['x'], observations['y'], observations['sigma']
        return float(np.sum(((parameters['a'] * x + parameters['b']) - y) ** 2 / s ** 2))


def _line_inference(sigma, n_iterations=50):
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 20)
    y = 2.0 * x + 1.0 + rng.normal(0, sigma, 20)
    obs = {'x': x, 'y': y, 'sigma': sigma}
    design = np.vstack([x, np.ones_like(x)]).T
    analytic = np.sqrt(np.diag(np.linalg.inv(design.T @ design / sigma ** 2)))
    engine = PhysicsEngine()
    engine.register_model('line', _Line())
    swarm = BayesianSwarmInference(engine, 'line')
    swarm.set_parameter_bounds({'a': (-10, 10, ''), 'b': (-10, 10, '')})
    return swarm.infer(obs, n_particles=60, n_iterations=n_iterations), analytic


@pytest.mark.parametrize("sigma", [0.5, 2.0])
def test_m1_uncertainties_match_the_analytic_fisher_matrix(sigma):
    """Pre-fix the "1-sigma" values were percentiles of the collapsed PSO
    swarm: +0.00000/-0.00000 on a noisy 20-point fit at n_iterations = 50."""
    result, analytic = _line_inference(sigma)
    assert result.parameters['a'].uncertainty_upper == pytest.approx(analytic[0], rel=1e-3)
    assert result.parameters['b'].uncertainty_upper == pytest.approx(analytic[1], rel=1e-3)
    assert result.parameters['a'].uncertainty_upper > 0.0


def test_m1_uncertainties_scale_with_the_data_errors():
    """Error bars must scale exactly x4 when the input errors do."""
    small, _ = _line_inference(0.5)
    large, _ = _line_inference(2.0)
    ratio = (large.parameters['a'].uncertainty_upper
             / small.parameters['a'].uncertainty_upper)
    assert ratio == pytest.approx(4.0, rel=1e-3)


def test_m1_log_evidence_is_stable_across_iteration_counts():
    """Pre-fix log Z varied by 9.4 nats on identical data purely with the
    iteration count, and carried two sign errors in the Laplace term."""
    values = [_line_inference(0.5, n_iterations=n)[0].log_evidence
              for n in (10, 50, 300)]
    assert max(values) - min(values) < 1.5             # pre-fix spread: 9.4


def test_m1_flat_likelihood_gives_nan_not_a_fabricated_number():
    """A chi^2 that does not depend on the parameters has no curvature, so the
    result must be NaN with convergence_achieved False -- never a number."""
    class _Flat(_Line):
        def chi_squared(self, parameters, observations):
            return 1.0

    engine = PhysicsEngine()
    engine.register_model('flat', _Flat())
    swarm = BayesianSwarmInference(engine, 'flat')
    swarm.set_parameter_bounds({'a': (-1, 1, ''), 'b': (-1, 1, '')})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = swarm.infer({'x': np.zeros(3)}, n_particles=10, n_iterations=5)
    assert math.isnan(result.parameters['a'].uncertainty_upper)
    assert result.convergence_achieved is False


# =============================================================================
# sed_fitting -- B1.1 (units), B1.2/B1.3 (normalisations)
# =============================================================================

def _mbb_setup():
    mbb = sed.ModifiedBlackbody(distance_Mpc=10.0)
    lam = np.array([7e5, 1.6e6, 2.5e6, 3.5e6, 5e6, 8.7e6])   # 70-870 um in A
    truth = {'T_dust': 25.0, 'beta': 1.8, 'M_dust': 1e7,
             'kappa_0': 0.77, 'lambda_0': 850.0}
    model = mbb.evaluate(lam, truth)
    composite = sed.CompositeSED(distance_Mpc=10.0)
    composite.add_component('dust', mbb)
    return sed.SEDFitter(composite), lam, model, truth


def test_sed_b11_jy_and_cgs_inputs_agree():
    """Pre-fix the docstring said Jy while the model returned CGS: following it
    gave T_dust = 123.4 K (truth 25 K) and chi2_red = 640."""
    fitter, lam, model, truth = _mbb_setup()
    fixed = {'kappa_0': 0.77, 'lambda_0': 850.0}
    cgs = fitter.fit(lam, model, 0.05 * model, fixed=fixed)
    jy = fitter.fit(lam, model / 1e-23, 0.05 * model / 1e-23,
                    fixed=fixed, flux_unit='Jy')
    for res in (cgs, jy):
        assert res.best_params['T_dust'] == pytest.approx(truth['T_dust'], rel=1e-3)
        assert res.best_params['beta'] == pytest.approx(truth['beta'], rel=1e-3)
        assert res.best_params['M_dust'] == pytest.approx(truth['M_dust'], rel=1e-2)
        assert res.reduced_chi_squared < 1e-6


def test_sed_b11_rejects_an_unknown_unit():
    fitter, lam, model, _ = _mbb_setup()
    with pytest.raises(ValueError):
        fitter.fit(lam, model, 0.05 * model, flux_unit='mJy')


def _integrate_over_nu(lam_ang, flux):
    nu = sed.c_light / (lam_ang * 1e-8)
    order = np.argsort(nu)
    return float(np.trapezoid(flux[order], nu[order]))


def test_sed_b12_stellar_population_normalisation():
    """int F_nu dnu must equal L_star/(4 pi D^2).
    Pre-fix it was 9.96e9 times too large (peak F_nu = 6.3e9 Jy)."""
    lam = np.geomspace(100.0, 1e7, 3000)
    pop = sed.StellarPopulation(distance_Mpc=10.0)
    flux = pop.evaluate(lam, {'M_star': 1e10, 'age': 5.0})
    d_cm = 10.0 * 3.086e24
    target = 1e10 * sed.L_sun / (4 * np.pi * d_cm ** 2)
    assert _integrate_over_nu(lam, flux) == pytest.approx(target, rel=0.01)
    assert np.max(flux) / 1e-23 < 10.0        # peak in Jy; pre-fix 6.3e9 Jy


def test_sed_b13_agn_normalisation():
    """int F_nu dnu must equal L_bol/(4 pi D^2).  Pre-fix: 1.4e15x too large."""
    lam = np.geomspace(100.0, 1e7, 3000)
    agn = sed.AGNTemplate(distance_Mpc=10.0)
    flux = agn.evaluate(lam, {'L_bol': 1e45, 'f_torus': 0.3, 'T_torus': 300.0})
    d_cm = 10.0 * 3.086e24
    target = 1e45 / (4 * np.pi * d_cm ** 2)
    assert _integrate_over_nu(lam, flux) == pytest.approx(target, rel=0.02)


# =============================================================================
# B7.1 -- radial_velocity Keplerian model
# =============================================================================

def _standard_rv(t, period, k, ecc, omega, t0, gamma=0.0):
    """Independent implementation of v_r = K[cos(f+w) + e cos w] + gamma."""
    m = 2 * np.pi * (t - t0) / period
    e_anom = m.copy()
    for _ in range(200):
        e_anom = e_anom - (e_anom - ecc * np.sin(e_anom) - m) / (1 - ecc * np.cos(e_anom))
    f = 2 * np.arctan2(np.sqrt(1 + ecc) * np.sin(e_anom / 2),
                       np.sqrt(1 - ecc) * np.cos(e_anom / 2))
    return k * (np.cos(f + omega) + ecc * np.cos(omega)) + gamma


@pytest.mark.parametrize("ecc, omega", [(0.0, 0.7), (0.3, np.pi / 4), (0.6, 1.2)])
def test_b71_rv_model_matches_the_standard_form(ecc, omega):
    """Pre-fix the model was K cos(f - omega): the omega sign was flipped and
    the K e cos(omega) term missing (rms mismatch 45-57 m/s at K = 50)."""
    t = np.linspace(0, 30, 300)
    params = {'period': 10.0, 'K': 50.0, 'ecc': ecc, 'omega': omega,
              'T0': 1.0, 'gamma': 0.0}
    code = rv.KeplerianFitter().rv_model(t, params)
    truth = _standard_rv(t, 10.0, 50.0, ecc, omega, 1.0)
    assert np.sqrt(np.mean((code - truth) ** 2)) < 1e-9    # pre-fix: 49.5 m/s


def test_b71_eccentric_orbit_recovery():
    """Pre-fix a true e = 0.3, omega = 45 deg orbit was recovered as
    e = 0.245, omega = 297.5 deg with chi2/N = 4.85."""
    rng = np.random.default_rng(4)
    t = np.sort(rng.uniform(0, 200, 150))
    velocities = _standard_rv(t, 12.0, 50.0, 0.3, np.pi / 4, 3.0) + rng.normal(0, 2.0, 150)
    params, chi2 = rv.KeplerianFitter().fit(t, velocities, np.full(150, 2.0), 12.0)
    assert params['period'] == pytest.approx(12.0, rel=0.01)
    assert params['K'] == pytest.approx(50.0, rel=0.05)
    assert params['ecc'] == pytest.approx(0.3, abs=0.05)
    assert math.degrees(params['omega']) % 360 == pytest.approx(45.0, abs=10.0)
    assert chi2 / 150 < 2.0


# =============================================================================
# B6.1/B6.2/B6.3 -- exoplanet_transit
# =============================================================================

def _transit_lc(epoch, depth=0.03, period=5.0, duration=0.2, sigma=1e-3,
                n=2000, span=30.0):
    rng = np.random.default_rng(2)
    t = np.linspace(0, span, n)
    flux = np.ones(n) + rng.normal(0, sigma, n)
    phase = ((t - epoch) / period + 0.5) % 1.0 - 0.5
    flux[np.abs(phase * period) < duration / 2] -= depth
    return et.LightCurve(times=t, fluxes=flux)


@pytest.mark.parametrize("epoch", [5.0, 2.5, 3.7])
def test_b61_bls_searches_transit_phase(epoch):
    """Pre-fix the box was pinned at phase 0: epoch 2.5 gave an aliased
    P = 2.502 d and epoch 3.7 gave 0 detections."""
    signals = et.TransitDetector(min_period=1.0,
                                 max_period=10.0).detect_transits(_transit_lc(epoch))
    assert signals
    assert signals[0].period == pytest.approx(5.0, rel=0.01)


def test_b63_shallow_transit_is_not_rejected_by_a_depth_floor():
    """Pre-fix `if depth < 0.001: return None` discarded every Neptune- and
    Earth-sized transit; this 300 ppm, high-S/N signal gave 0 candidates."""
    lc = _transit_lc(3.7, depth=3e-4, sigma=1e-4)
    signals = et.TransitDetector(min_period=1.0, max_period=10.0).detect_transits(lc)
    assert signals
    assert signals[0].depth == pytest.approx(3e-4, rel=0.2)
    assert signals[0].period == pytest.approx(5.0, rel=0.01)


def test_b62_snr_is_a_transit_snr_not_a_per_point_ratio():
    """SNR must be depth/(sigma/sqrt(N_in)), i.e. >> depth/sigma."""
    lc = _transit_lc(3.7, depth=3e-4, sigma=1e-4)
    signal = et.TransitDetector(min_period=1.0,
                                max_period=10.0).detect_transits(lc)[0]
    per_point = signal.depth / 1e-4                    # ~3, the old statistic
    assert signal.snr > 5 * per_point


# =============================================================================
# C12 -- interferometry
# =============================================================================

def _alma_visibilities(n_ha=25):
    array = itf.StandardArrays.alma_compact()
    uv = itf.UVSimulator(array).generate_coverage(-30.0, 230e9,
                                                  np.linspace(-3, 3, n_ha), 10.0)
    n = len(uv.u)
    vis = itf.Visibility(u=uv.u, v=uv.v, w=uv.w, real=np.ones(n),
                         imag=np.zeros(n), weight=np.ones(n), freq=230e9)
    return array, uv, vis


def test_c12_uv_coordinates_are_in_wavelengths_not_centimetres():
    """Pre-fix c (cm/s) was used as m/s: max uv = 2140 lambda for ALMA-compact
    at 230 GHz where B_max/lambda = 215 342."""
    array, uv, _ = _alma_visibilities(n_ha=1)
    lam = 2.998e8 / 230e9
    b_max_lambda = array.max_baseline() / lam
    uv_max = np.max(np.hypot(uv.u, uv.v))
    assert 0.5 * b_max_lambda < uv_max <= b_max_lambda   # projected, so <= B/lam
    assert uv_max > 1e5


@pytest.mark.parametrize("latitude", [0.0, 70.0])
def test_b55_enu_to_equatorial_uses_the_array_latitude(latitude):
    """A pure N-S baseline must give u = 0 at transit and
    v = |b| cos(dec - lat)/lambda.  Pre-fix ENU was used as XYZ and the
    latitude was discarded: identical (u, v) at 0 and 70 deg, u = 3335 at
    transit."""
    array = itf.ArrayConfiguration('test', latitude=latitude)
    array.add_antenna(itf.Antenna('a', 0.0, 0.0, 0.0, 12.0))
    array.add_antenna(itf.Antenna('b', 0.0, 1000.0, 0.0, 12.0))
    cov = itf.UVSimulator(array).generate_coverage(30.0, 1e9, np.array([0.0]), 10.0)
    lam = 2.998e8 / 1e9
    assert cov.u[0] == pytest.approx(0.0, abs=1e-9)
    assert cov.v[0] == pytest.approx(1000.0 / lam * math.cos(math.radians(30 - latitude)),
                                     rel=1e-6)


def test_b53_dirty_image_flux_scale():
    """A 1 Jy point source must peak at 1 Jy.  Pre-fix: 6.1035e-05 = 1/128^2."""
    _, _, vis = _alma_visibilities()
    image = itf.Imager().make_dirty_image(vis, image_size=128, pixel_size=0.1,
                                          weighting=itf.WeightingScheme.NATURAL)
    assert np.max(image.image) == pytest.approx(1.0, rel=1e-6)


def test_b52_weighting_schemes_differ_and_briggs_interpolates():
    """Pre-fix NATURAL and UNIFORM gave bit-identical dirty beams and
    robust = -2/0/+2 all gave bmaj = 8.7009"."""
    _, _, vis = _alma_visibilities()
    imager = itf.Imager()
    natural = imager.make_dirty_image(vis, 128, 0.1, itf.WeightingScheme.NATURAL)
    uniform = imager.make_dirty_image(vis, 128, 0.1, itf.WeightingScheme.UNIFORM)
    assert not np.allclose(natural.beam, uniform.beam)
    assert uniform.beam_params['bmaj'] < natural.beam_params['bmaj']

    bmaj = [imager.make_dirty_image(vis, 128, 0.1, itf.WeightingScheme.BRIGGS,
                                    robust=r).beam_params['bmaj']
            for r in (-2.0, 0.0, 2.0)]
    assert bmaj[0] < bmaj[1] < bmaj[2]                    # monotonic in robust
    assert bmaj[0] == pytest.approx(uniform.beam_params['bmaj'], rel=0.02)
    assert bmaj[2] == pytest.approx(natural.beam_params['bmaj'], rel=0.02)


@pytest.mark.parametrize("pa", [0.0, 30.0, 60.0, 120.0])
def test_b56_restoring_beam_position_angle_round_trip(pa):
    """Pre-fix requesting PA = 30 deg produced a beam that fitted back as
    -30 deg, i.e. restored maps carried a beam rotated by 2 x PA."""
    beam = itf.CLEANDeconvolver._elliptical_gaussian(
        129, 129, {'bmaj': 1.2, 'bmin': 0.6, 'pa': pa}, 0.05)
    fitted = itf.Imager()._fit_beam(beam, 0.05)
    assert fitted['bmaj'] == pytest.approx(1.2, rel=1e-3)
    assert fitted['bmin'] == pytest.approx(0.6, rel=1e-3)
    # position angle is defined modulo 180 degrees
    delta = (fitted['pa'] - pa) % 180.0
    assert min(delta, 180.0 - delta) < 0.5


def test_b54_visibility_modeler_uses_its_documented_convention():
    """V(u,v) = sum I e^{-2 pi i (u l + v m)}.  Pre-fix model_visibilities
    returned the CONJUGATE, so a fit converged on the mirrored source."""
    npix = 64
    dl = 0.1 / 206265.0
    image = np.zeros((npix, npix))
    image[npix // 2 + 3, npix // 2 + 5] = 1.0          # m = +3 px, l = +5 px
    du = 1.0 / (npix * dl)
    u = np.array([2 * du])
    v = np.array([1 * du])
    modelled = itf.VisibilityModeler(pixel_size=0.1).model_visibilities(image, u, v)
    truth = np.exp(-2j * np.pi * (u[0] * 5 * dl + v[0] * 3 * dl))
    assert modelled[0] == pytest.approx(truth, abs=1e-9)


def test_b54_true_image_beats_the_mirrored_one():
    """Consequence test: chi^2 of the true image must be LOWER than that of the
    image mirrored through the phase centre. Pre-fix it was 300x higher."""
    npix = 64
    dl = 0.1 / 206265.0
    image = np.zeros((npix, npix))
    image[npix // 2 + 3, npix // 2 + 5] = 1.0
    mirrored = np.zeros((npix, npix))
    mirrored[npix // 2 - 3, npix // 2 - 5] = 1.0
    du = 1.0 / (npix * dl)
    rng = np.random.default_rng(0)
    u = rng.uniform(-8, 8, 40) * du
    v = rng.uniform(-8, 8, 40) * du
    modeler = itf.VisibilityModeler(pixel_size=0.1)
    data = np.exp(-2j * np.pi * (u * 5 * dl + v * 3 * dl))
    err = np.full(40, 0.01)
    chi2_true = modeler.chi_squared(data, err, image, u, v)
    chi2_mirror = modeler.chi_squared(data, err, mirrored, u, v)
    assert chi2_true < chi2_mirror


# ---------------------------------------------------------------------------
# B7.4 / B7.5 -- radial-velocity periodogram normalisation and FAP
# ---------------------------------------------------------------------------

def test_rv_periodogram_is_not_self_normalised():
    """
    Pre-fix `compute_lombscargle` divided the power by its own maximum, so the
    tallest peak was ALWAYS exactly 1.0 regardless of significance -- pure
    noise looked identical to a strong detection.
    """
    import numpy as np
    from astra_core.astro_physics.radial_velocity import RVPeriodogram

    rng = np.random.default_rng(0)
    pg = RVPeriodogram(1.0, 100.0)

    zmax = []
    for _ in range(50):
        t = np.sort(rng.uniform(0, 300, 100))
        v = rng.normal(0, 10, 100)
        _, power = pg.compute_lombscargle(t, v)
        zmax.append(power.max())
    zmax = np.array(zmax)

    # pre-fix: every single one of these was exactly 1.0
    assert zmax.std() > 0.1, "power still appears to be self-normalised"
    # Horne-Baliunas z is ~ln(N_indep) for white noise, i.e. a few units
    assert 2.0 < zmax.mean() < 30.0

    # an injected signal must stand well clear of the noise floor
    t = np.sort(rng.uniform(0, 300, 100))
    v = 50 * np.sin(2 * np.pi * t / 10.0) + rng.normal(0, 5, 100)
    periods, power = pg.compute_lombscargle(t, v)
    i = int(np.argmax(power))
    assert periods[i] == pytest.approx(10.0, rel=0.02)
    assert power[i] > 3 * zmax.mean()


def test_rv_fap_uses_power_and_corrects_for_trials():
    """
    Pre-fix `_estimate_fap` was `1 - Phi(SNR)`: it ignored the periodogram
    power entirely and applied no trials correction, so any moderately
    well-fitted spurious peak returned FAP ~ 1e-10.
    """
    from astra_core.astro_physics.radial_velocity import RVDetector

    an = RVDetector()

    # identical SNR, very different periodogram power -> FAP must differ
    weak = an._estimate_fap(100, snr=8.0, power=2.0)
    strong = an._estimate_fap(100, snr=8.0, power=50.0)
    assert weak > strong, "FAP still ignores the periodogram power"

    # a marginal peak must not be reported as a 1e-10 certainty
    assert weak > 1e-3

    # trials correction: searching more frequencies must raise the FAP
    an_narrow = RVDetector(min_period=9.0, max_period=11.0)
    an_wide = RVDetector(min_period=1.0, max_period=1000.0)
    z = 8.0
    assert (an_wide._estimate_fap(500, 8.0, z)
            > an_narrow._estimate_fap(500, 8.0, z))
