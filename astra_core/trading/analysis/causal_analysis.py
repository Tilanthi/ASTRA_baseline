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
Market Causal Analysis
======================

Causal structure discovery for market price series.

- MarketCausalAnalyzer: finds lead-lag causal relations between asset
  return series with Granger causality tests (re-using the validated
  implementation in causal.discovery.temporal_discovery, with a local
  F-test fallback).
- CausalSignal: one detected lead-lag relation.
- CausalBacktester: evaluates whether signals found in-sample have
  out-of-sample predictive value (directional hit rate vs coin-flip).

(Re-implemented 2026-08: the original bodies were lost, leaving only a
stub comment, before the August 2026 audit.)

NOTE: works on caller-supplied data only - no synthetic data is used.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np

try:
    from scipy import stats as _stats
except ImportError:  # pragma: no cover - scipy is a core dependency
    _stats = None

try:
    from ...causal.discovery.temporal_discovery import granger_causality_test
except Exception:
    granger_causality_test = None


@dataclass
class CausalSignal:
    """A detected lead-lag causal relation between two series."""
    source: str            # leading series name
    target: str            # lagging series name
    lag: int               # lag (in samples) at which source leads target
    p_value: float         # Granger F-test p-value
    strength: float        # lagged cross-correlation, |r| in [0, 1]
    detected_at: str = ""

    def __post_init__(self):
        if not self.detected_at:
            self.detected_at = datetime.now().isoformat()


def _local_granger(x: np.ndarray, y: np.ndarray, max_lag: int):
    """Minimal Granger test (y caused by x) used if the core one is absent."""
    if granger_causality_test is not None:
        return granger_causality_test(x, y, max_lag=max_lag)
    if _stats is None:
        return False, 0, 1.0

    def _ols(design, response):
        coef, *_ = np.linalg.lstsq(design, response, rcond=None)
        resid = response - design @ coef
        return coef, float(resid @ resid)

    n = len(y)
    lags = np.arange(1, max_lag + 1)
    best = (False, 0, 1.0)
    for L in lags:
        if n - max_lag - 1 <= 2 * L:
            break
        y_dep = y[max_lag:]
        X_r = np.column_stack([np.ones(n - max_lag)]
                              + [y[max_lag - k: n - k] for k in lags[:L]])
        X_f = np.column_stack([X_r]
                              + [x[max_lag - k: n - k] for k in lags[:L]])
        _, rss_r = _ols(X_r, y_dep)
        _, rss_f = _ols(X_f, y_dep)
        dof = n - 2 * L - 1
        if rss_f <= 0 or dof <= 0:
            continue
        F = ((rss_r - rss_f) / L) / (rss_f / dof)
        p = float(_stats.f.sf(F, L, dof))
        if p < 0.05:
            return True, int(L), p
        if p < best[2]:
            best = (False, int(L), p)
    return best


class MarketCausalAnalyzer:
    """
    Discovers which series Granger-cause which, across a price universe.

    Inputs are return series (callers should difference/log-return prices
    first; :meth:`to_returns` does this for price levels).
    """

    def __init__(self, max_lag: int = 5, significance: float = 0.05):
        self.max_lag = max_lag
        self.significance = significance

    @staticmethod
    def to_returns(prices: np.ndarray) -> np.ndarray:
        """Convert price levels to log returns."""
        prices = np.asarray(prices, dtype=float)
        return np.diff(np.log(prices))

    def analyze(self, returns: Dict[str, np.ndarray],
                target: Optional[str] = None) -> List[CausalSignal]:
        """
        Test every source -> target pair for Granger causality.

        Returns detected signals sorted by p-value (strongest first).
        """
        names = list(returns)
        targets = [target] if target else names
        signals: List[CausalSignal] = []
        for tgt in targets:
            y = np.asarray(returns[tgt], dtype=float)
            for src in names:
                if src == tgt:
                    continue
                x = np.asarray(returns[src], dtype=float)
                n = min(len(x), len(y))
                if n <= 3 * self.max_lag + 2:
                    continue
                causes, lag, p = _local_granger(x[-n:], y[-n:],
                                                max_lag=self.max_lag)
                if causes and p < self.significance:
                    # Lagged cross-correlation at the detected lag
                    xs, ys = x[-n:], y[-n:]
                    x_lag, y_cur = xs[:n - lag], ys[lag:]
                    vx = x_lag - x_lag.mean()
                    vy = y_cur - y_cur.mean()
                    denom = float(np.sqrt((vx @ vx) * (vy @ vy)))
                    strength = float(abs(vx @ vy) / denom) if denom > 0 else 0.0
                    signals.append(CausalSignal(
                        source=src, target=tgt, lag=lag,
                        p_value=p, strength=strength))
        signals.sort(key=lambda s: (s.p_value, -s.strength))
        return signals


class CausalBacktester:
    """
    Out-of-sample evaluation of causal signals.

    Signals detected on a training window are evaluated on a held-out
    window by directional hit rate: does the lagged source return predict
    the SIGN of the target return better than a coin flip?
    """

    def __init__(self, train_fraction: float = 0.7):
        if not 0.1 < train_fraction < 0.9:
            raise ValueError("train_fraction must be in (0.1, 0.9)")
        self.train_fraction = train_fraction

    def backtest(self, returns: Dict[str, np.ndarray],
                 target: str,
                 analyzer: Optional[MarketCausalAnalyzer] = None
                 ) -> Dict[str, Any]:
        """Backtest causal drivers of ``target`` discovered in-sample."""
        analyzer = analyzer or MarketCausalAnalyzer()
        names = [n for n in returns if n != target]
        lengths = [len(returns[n]) for n in names + [target]]
        if not lengths:
            raise ValueError("no return series provided")
        n_total = min(lengths)
        split = int(n_total * self.train_fraction)

        trimmed = {name: np.asarray(returns[name], dtype=float)[-n_total:]
                   for name in returns}

        train = {name: series[:split] for name, series in trimmed.items()}
        test = {name: series[split:] for name, series in trimmed.items()}

        signals = analyzer.analyze(train, target=target)
        y_test = test[target]

        results: List[Dict[str, Any]] = []
        for sig in signals:
            x_test = test[sig.source]
            n_eval = len(y_test) - sig.lag
            if n_eval <= 0:
                continue
            x_lag_sign = np.sign(x_test[:n_eval])
            y_sign = np.sign(y_test[sig.lag:])
            hits = int(np.sum(x_lag_sign == y_sign))
            results.append({
                'source': sig.source,
                'lag': sig.lag,
                'train_p_value': sig.p_value,
                'n_test_points': int(n_eval),
                'hit_rate': hits / n_eval,
            })

        overall = (sum(r['hit_rate'] for r in results) / len(results)
                   if results else None)
        return {
            'target': target,
            'n_signals_tested': len(results),
            'signals': results,
            'average_hit_rate': overall,
            'train_points': split,
            'test_points': n_total - split,
        }
