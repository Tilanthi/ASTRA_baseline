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
Unit tests for threshold-autonomous outlier detection
(calibrated_outliers, benjamini_hochberg).

The detector's flagging threshold carries no contamination parameter: the
flagged fraction is whatever survives BH-FDR control on chi-square
Mahalanobis p-values from a robust MCD bulk fit.

Covers:
1. Benjamini-Hochberg step-up correctness on hand-computed cases.
2. Autonomous fraction tracking on synthetic data with injected anomalies
   at several true fractions (no hand-set contamination anywhere).
3. Purity / recall of the flags against the injected labels.
4. Specificity on clean data (no injected anomalies -> near-zero flags).
5. Determinism under a fixed seed.
"""
import numpy as np

from astra_core.reasoning.v80_discovery_engine.subtle_pattern_detection import (
    benjamini_hochberg,
    calibrated_outliers,
)


def test_bh_known_case():
    # Wikipedia BH example: p = [0.001, 0.008, 0.039, 0.041, 0.042].
    # At q=0.05 the step-up reaches k=5 (0.042 <= 0.05): reject all five.
    # At q=0.02 it stops at k=2 (0.008 <= 0.008): reject the first two.
    # At q=0.01 only k=1 holds (0.001 <= 0.002): reject only the smallest.
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042])
    assert benjamini_hochberg(p, q=0.05).tolist() == [True] * 5
    assert benjamini_hochberg(p, q=0.02).tolist() == [True, True, False, False, False]
    assert benjamini_hochberg(p, q=0.01).tolist() == [True, False, False, False, False]


def test_bh_no_rejections():
    p = np.full(100, 0.5)
    assert not benjamini_hochberg(p, q=0.01).any()


def _make_data(n_bulk=2000, n_anom=100, n_features=5, seed=7, displacement=4.0):
    rng = np.random.default_rng(seed)
    # correlated Gaussian bulk
    cov = np.full((n_features, n_features), 0.5)
    np.fill_diagonal(cov, 1.0)
    bulk = rng.multivariate_normal(np.zeros(n_features), cov, size=n_bulk)
    if n_anom == 0:
        return bulk, np.zeros(n_bulk, dtype=bool)
    # anomalies displaced along multiple features (break correlations)
    anom = rng.multivariate_normal(np.full(n_features, displacement),
                                   np.eye(n_features), size=n_anom)
    X = np.vstack([bulk, anom])
    labels = np.zeros(n_bulk + n_anom, dtype=bool)
    labels[n_bulk:] = True
    return X, labels


def test_autonomous_fraction_tracks_truth():
    # three true fractions spanning an order of magnitude: the flagged
    # fraction must follow the injected fraction without any tuning
    for n_anom, tol in [(60, 0.03), (100, 0.03), (300, 0.03)]:
        X, labels = _make_data(n_anom=n_anom)
        res = calibrated_outliers(X)
        true_fraction = labels.mean()
        assert abs(res.fraction - true_fraction) < tol, (
            n_anom, res.fraction, true_fraction)


def test_flag_purity():
    X, labels = _make_data(n_anom=100)
    res = calibrated_outliers(X)
    purity = res.flags[labels].mean() if res.flags.sum() else 0.0
    recall = res.flags[labels].mean() if res.flags.sum() else 0.0
    assert purity >= 0.95, purity
    assert recall >= 0.95, recall


def test_clean_data_low_flag_rate():
    X, _ = _make_data(n_anom=0)
    res = calibrated_outliers(X)
    assert res.fraction <= 0.01, res.fraction


def test_stricter_q_flags_fewer():
    X, labels = _make_data(n_anom=100)
    strict = calibrated_outliers(X, q=0.001)
    loose = calibrated_outliers(X, q=0.05)
    assert strict.flags.sum() <= loose.flags.sum()


def test_determinism():
    X, _ = _make_data()
    a = calibrated_outliers(X)
    b = calibrated_outliers(X)
    assert np.array_equal(a.flags, b.flags)
    assert np.allclose(a.p_values, b.p_values)


if __name__ == "__main__":
    for name, fn in sorted({k: v for k, v in globals().items()
                            if k.startswith("test_") and callable(v)}.items()):
        fn()
        print(f"PASS {name}")
    print("ALL TESTS PASSED")
