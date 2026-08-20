"""
Temporal Causal Discovery

Causal discovery for time-series data exploiting temporal ordering.
Key principle: Causes must precede effects in time.

Algorithms:
- Granger causality (VAR-based)
- Transfer entropy (information-theoretic)
- VAR-LiNGAM (linear non-Gaussian)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.stats import pearsonr
from itertools import combinations

from .independence import ConditionalIndependenceTest
from ..model.scm import StructuralCausalModel, Variable, VariableType, StructuralEquation


class TemporalCausalDiscovery:
    """
    Temporal causal discovery for time-series data.

    Exploits temporal ordering: causes must precede effects.
    Focuses on lagged causal relationships.

    Example:
        >>> data = pd.DataFrame({'X': x, 'Y': y}, index=date_index)
        >>> tcd = TemporalCausalDiscovery(max_lag=5)
        >>> scm = tcd.discover(data, method='var')
    """

    def __init__(self, max_lag: int = 10):
        """
        Initialize temporal causal discovery.

        Args:
            max_lag: Maximum time lag to consider
        """
        self.max_lag = max_lag

    def discover(self,
                 data: pd.DataFrame,
                 method: str = 'var',
                 alpha: float = 0.05,
                 verbose: bool = False) -> StructuralCausalModel:
        """
        Discover temporal causal structure.

        Args:
            data: Time-series data (index=time, columns=variables)
            method: Method to use ('var', 'transfer_entropy', 'lingam')
            alpha: Significance level
            verbose: Print progress

        Returns:
            StructuralCausalModel with temporal causal edges
        """
        if method == 'var':
            return self._discover_var(data, alpha, verbose)
        elif method == 'transfer_entropy':
            return self._discover_transfer_entropy(data, alpha, verbose)
        elif method == 'lingam':
            return self._discover_var_lingam(data, alpha, verbose)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _discover_var(self,
                      data: pd.DataFrame,
                      alpha: float,
                      verbose: bool) -> StructuralCausalModel:
        """
        Discover causal structure using Vector Autoregression (VAR).

        Granger causality: X Granger-causes Y if lagged values of X
        improve prediction of Y beyond lagged values of Y alone.
        """
        from statsmodels.tsa.api import VAR

        if verbose:
            print(f"VAR-based temporal discovery (max_lag={self.max_lag})")

        # Fit VAR model
        model = VAR(data)
        results = model.fit(maxlags=self.max_lag, ic='aic')

        scm = StructuralCausalModel(name="Temporal_VAR")

        # Add variables
        for var in data.columns:
            v = Variable(name=var, type=VariableType.CONTINUOUS)
            scm.add_variable(v)

        # Extract Granger causalities
        for effect_var in data.columns:
            pass  # Granger causality extraction needed


# =============================================================================
# GRANGER CAUSALITY TEST
# (re-implemented 2026-08; the original public function was lost to file
#  truncation before the August 2026 audit. The F-test formulation matches
#  the internal version in astro_causal_discovery.py.)
# =============================================================================

def granger_causality_test(x: np.ndarray,
                           y: np.ndarray,
                           max_lag: int = 5,
                           significance: float = 0.05) -> Tuple[bool, int, float]:
    """
    Test whether x Granger-causes y, scanning lags 1..max_lag.

    For each lag L two OLS models are fit (both with an intercept):

        restricted:  y_t ~ 1 + sum_{i=1..L} a_i y_{t-i}
        full:        y_t ~ 1 + sum_{i=1..L} a_i y_{t-i} + sum_{i=1..L} b_i x_{t-i}

    and compared with the standard F-test.  The lag with the smallest
    p-value is reported.

    Args:
        x: putative cause series
        y: effect series
        max_lag: maximum lag to consider
        significance: p-value threshold for declaring causation

    Returns:
        (causes, best_lag, p_value) where causes is p < significance
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    best_p, best_lag = 1.0, 1
    first_significant = None
    for lag in range(1, max_lag + 1):
        if n <= 2 * lag + 2:
            break
        Y_target = y[lag:]
        # lag-i columns end at index n-i-1 for target index lag..n-1
        y_lag = np.column_stack([y[lag - i: n - i] for i in range(1, lag + 1)])
        x_lag = np.column_stack([x[lag - i: n - i] for i in range(1, lag + 1)])
        ones = np.ones((len(Y_target), 1))

        restricted = np.hstack([ones, y_lag])
        full = np.hstack([ones, y_lag, x_lag])

        beta_r = np.linalg.lstsq(restricted, Y_target, rcond=None)[0]
        beta_f = np.linalg.lstsq(full, Y_target, rcond=None)[0]
        rss_r = float(np.sum((Y_target - restricted @ beta_r) ** 2))
        rss_f = float(np.sum((Y_target - full @ beta_f) ** 2))

        df1 = lag                      # number of added regressors
        df2 = n - 2 * lag - 1          # residual dof of the full model
        if df2 <= 0 or rss_f <= 0 or rss_r <= 0:
            continue
        F_stat = ((rss_r - rss_f) / df1) / (rss_f / df2)
        p = float(stats.f.sf(F_stat, df1, df2))
        if p < significance and first_significant is None:
            first_significant = (lag, p)   # parsimonious lag order
        if p < best_p:
            best_p, best_lag = p, lag

    if first_significant is not None:
        lag_s, p_s = first_significant
        return True, lag_s, p_s
    return False, best_lag, best_p
