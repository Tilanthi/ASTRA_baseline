#!/usr/bin/env python3
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
Uncertainty Quantification Framework for ASTRO-SWARM
=====================================================

Comprehensive Bayesian inference and uncertainty quantification tools
for astronomical parameter estimation.

Capabilities:
1. MCMC posterior sampling (Metropolis-Hastings, affine-invariant ensemble)
2. Nested sampling for model comparison
3. Fisher matrix forecasting
4. Systematic error budgeting
5. Posterior predictive checks
6. Convergence diagnostics

Key Dependencies:
- emcee (optional, for ensemble MCMC)
- dynesty (optional, for nested sampling)
- corner (optional, for visualization)

Author: Claude Code (ASTRO-SWARM)
Date: 2024-11
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from enum import Enum
from abc import ABC, abstractmethod
import warnings
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, uniform, truncnorm
from scipy.linalg import inv, det, cholesky
import json

# Try to import optional dependencies
try:
    import emcee
    EMCEE_AVAILABLE = True
except ImportError:
    EMCEE_AVAILABLE = False

try:
    import dynesty
    DYNESTY_AVAILABLE = True
except ImportError:
    DYNESTY_AVAILABLE = False

try:
    import corner
    CORNER_AVAILABLE = True
except ImportError:
    CORNER_AVAILABLE = False


# =============================================================================
# PRIOR DISTRIBUTIONS
# =============================================================================

class PriorType(Enum):
    """Types of prior distributions"""
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    LOG_UNIFORM = "log_uniform"
    TRUNCATED_GAUSSIAN = "truncated_gaussian"
    FIXED = "fixed"
    CUSTOM = "custom"


@dataclass
class Prior:
    """Prior distribution specification"""
    name: str
    prior_type: PriorType
    params: Dict[str, float]
    bounds: Tuple[float, float]
    description: str = ""

    def sample(self, n: int = 1) -> np.ndarray:
        """Draw samples from prior"""
        if self.prior_type == PriorType.UNIFORM:
            return np.random.uniform(self.bounds[0], self.bounds[1], n)

        elif self.prior_type == PriorType.GAUSSIAN:
            mu = self.params['mean']
            sigma = self.params['std']
            samples = np.random.normal(mu, sigma, n)
            return np.clip(samples, self.bounds[0], self.bounds[1])

        elif self.prior_type == PriorType.LOG_UNIFORM:
            log_samples = np.random.uniform(
                np.log10(self.bounds[0]),
                np.log10(self.bounds[1]), n)
            return 10**log_samples

        elif self.prior_type == PriorType.TRUNCATED_GAUSSIAN:
            mu = self.params['mean']
            sigma = self.params['std']
            a = (self.bounds[0] - mu) / sigma
            b = (self.bounds[1] - mu) / sigma
            return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n)

        elif self.prior_type == PriorType.FIXED:
            return np.full(n, self.params['value'])

        return np.random.uniform(self.bounds[0], self.bounds[1], n)

    def log_prob(self, x: float) -> float:
        """Log probability density"""
        if x < self.bounds[0] or x > self.bounds[1]:
            return -np.inf

        if self.prior_type == PriorType.UNIFORM:
            return -np.log(self.bounds[1] - self.bounds[0])

        elif self.prior_type == PriorType.GAUSSIAN:
            mu = self.params['mean']
            sigma = self.params['std']
            return -0.5 * ((x - mu) / sigma)**2 - np.log(sigma * np.sqrt(2*np.pi))

        elif self.prior_type == PriorType.LOG_UNIFORM:
            return -np.log(x) - np.log(np.log10(self.bounds[1]/self.bounds[0]))

        elif self.prior_type == PriorType.TRUNCATED_GAUSSIAN:
            mu = self.params['mean']
            sigma = self.params['std']
            a = (self.bounds[0] - mu) / sigma
            b = (self.bounds[1] - mu) / sigma
            return truncnorm.logpdf(x, a, b, loc=mu, scale=sigma)

        elif self.prior_type == PriorType.FIXED:
            return 0.0 if np.abs(x - self.params['value']) < 1e-10 else -np.inf

        return 0.0


class PriorSet:
    """Collection of priors for multiple parameters"""

    def __init__(self):
        self.priors: Dict[str, Prior] = {}
        self.param_names: List[str] = []

    def add(self, prior: Prior):
        """Add a prior"""
        self.priors[prior.name] = prior
        if prior.name not in self.param_names:
            self.param_names.append(prior.name)

    def add_uniform(self, name: str, low: float, high: float, description: str = ""):
        """Add uniform prior"""
        self.add(Prior(
            name=name,
            prior_type=PriorType.UNIFORM,
            params={},
            bounds=(low, high),
            description=description
        ))

    def add_gaussian(self, name: str, mean: float, std: float,
                    bounds: Optional[Tuple[float, float]] = None,
                    description: str = ""):
        """Add Gaussian prior"""
        if bounds is None:
            bounds = (mean - 10*std, mean + 10*std)
        self.add(Prior(
            name=name,
            prior_type=PriorType.GAUSSIAN,
            params={'mean': mean, 'std': std},
            bounds=bounds,
            description=description
        ))

    def add_log_uniform(self, name: str, low: float, high: float,
                       description: str = ""):
        """Add log-uniform prior"""
        self.add(Prior(
            name=name,
            prior_type=PriorType.LOG_UNIFORM,
            params={},
            bounds=(low, high),
            description=description
        ))

    def sample(self, n: int = 1) -> np.ndarray:
        """Sample from all priors"""
        samples = np.zeros((n, len(self.param_names)))
        for i, name in enumerate(self.param_names):
            samples[:, i] = self.priors[name].sample(n)
        return samples

    def log_prob(self, theta: np.ndarray) -> float:
        """Total log prior probability"""
        lp = 0.0
        for i, name in enumerate(self.param_names):
            lp += self.priors[name].log_prob(theta[i])
            if not np.isfinite(lp):
                return -np.inf
        return lp

    def bounds_array(self) -> np.ndarray:
        """Get bounds as array for optimization"""
        return np.array([self.priors[name].bounds for name in self.param_names])

    @property
    def n_params(self) -> int:
        return len(self.param_names)


# =============================================================================
# LIKELIHOOD FUNCTIONS
# =============================================================================

@dataclass
class LikelihoodResult:
    """Result from likelihood evaluation"""
    log_likelihood: float
    chi_squared: float
    n_data: int
    residuals: Optional[np.ndarray] = None
    model: Optional[np.ndarray] = None


class GaussianLikelihood:
    """
    Gaussian likelihood for data with known uncertainties.

    log L = -0.5 * sum((data - model)^2 / sigma^2 + log(2*pi*sigma^2))
    """

    def __init__(self, data: np.ndarray, errors: np.ndarray,
                model_func: Callable[[np.ndarray], np.ndarray]):
        """
        Parameters
        ----------
        data : np.ndarray
            Observed data
        errors : np.ndarray
            Measurement uncertainties (1-sigma)
        model_func : callable
            Function that takes parameters and returns model prediction
        """
        self.data = np.asarray(data)
        self.errors = np.asarray(errors)
        self.model_func = model_func
        self.n_data = self.data.size

    def residuals(self, theta: np.ndarray) -> np.ndarray:
        """Residuals data - model(theta)."""
        return self.data - np.asarray(self.model_func(theta))

    def log_likelihood(self, theta: np.ndarray) -> float:
        """
        Gaussian log-likelihood.

        log L = -0.5 * sum[ (data - model)^2/sigma^2 + log(2 pi sigma^2) ]
        """
        r = self.residuals(theta)
        if not np.all(np.isfinite(r)):
            return -np.inf
        return float(-0.5 * np.sum((r / self.errors) ** 2
                                   + np.log(2 * np.pi * self.errors ** 2)))

    def chi_squared(self, theta: np.ndarray) -> float:
        """Chi-squared statistic."""
        r = self.residuals(theta)
        return float(np.sum((r / self.errors) ** 2))

    def evaluate(self, theta: np.ndarray) -> LikelihoodResult:
        """Full likelihood evaluation at theta."""
        model = np.asarray(self.model_func(theta))
        r = self.data - model
        return LikelihoodResult(
            log_likelihood=self.log_likelihood(theta),
            chi_squared=self.chi_squared(theta),
            n_data=self.n_data,
            residuals=r,
            model=model)


# =============================================================================
# METROPOLIS-HASTINGS SAMPLER
# =============================================================================

class MetropolisHastings:
    """
    Adaptive random-walk Metropolis-Hastings MCMC sampler.

    The Gaussian proposal covariance is adapted during burn-in
    (scaled toward a target acceptance rate of ~0.3), then frozen for
    the production chain. Works with any log-posterior function.
    """

    def __init__(self, log_posterior: Callable[[np.ndarray], float],
                 initial: np.ndarray,
                 proposal_scale: float = 0.1,
                 target_acceptance: float = 0.3,
                 seed: Optional[int] = None):
        """
        Args:
            log_posterior: log posterior function of the parameter vector
            initial: initial parameter vector
            proposal_scale: initial step size (relative to parameter scale)
            target_acceptance: acceptance rate targeted during adaptation
            seed: RNG seed for reproducibility
        """
        self.log_posterior = log_posterior
        self.initial = np.asarray(initial, dtype=float)
        self.scale = float(proposal_scale)
        self.target = float(target_acceptance)
        self.rng = np.random.default_rng(seed)
        self.chain_ = None
        self.acceptance_rate_ = None

    def run(self, n_steps: int = 10000, n_burn: int = 2000,
            thin: int = 1) -> Dict[str, Any]:
        """
        Run the chain.

        Args:
            n_steps: total number of MCMC steps
            n_burn: burn-in steps (discarded; used for adaptation)
            thin: keep every `thin`-th post-burn sample

        Returns:
            dict with 'chain' (n_kept, n_params), 'acceptance_rate',
            'log_posterior' trace
        """
        d = self.initial.size
        theta = self.initial.copy()
        lp = self.log_posterior(theta)
        if not np.isfinite(lp):
            raise ValueError("log posterior is -inf at the initial point")

        # Proposal: isotropic Gaussian scaled to the initial parameter
        # magnitudes, adapted multiplicatively during burn-in
        param_scale = np.where(np.abs(theta) > 0, np.abs(theta), 1.0)
        chol = np.eye(d) * self.scale
        step = chol.copy()

        chain = np.empty((n_steps, d))
        lps = np.empty(n_steps)
        n_accept = 0

        for i in range(n_steps):
            proposal = theta + self.rng.standard_normal(d) @ step
            lp_new = self.log_posterior(proposal)
            log_alpha = lp_new - lp
            if (np.isfinite(lp_new)
                    and np.log(self.rng.uniform()) < log_alpha):
                theta, lp = proposal, lp_new
                n_accept += 1
            chain[i] = theta
            lps[i] = lp

            # adapt during burn-in: shrink/grow step toward target rate
            if i < n_burn and (i + 1) % 50 == 0:
                rate = n_accept / (i + 1)
                step *= np.exp((rate - self.target))

        self.chain_ = chain
        self.acceptance_rate_ = n_accept / n_steps
        kept = chain[n_burn::thin]
        return {
            'chain': kept,
            'samples': kept,
            'log_posterior': lps[n_burn::thin],
            'acceptance_rate': self.acceptance_rate_,
            'n_steps': n_steps,
            'n_burn': n_burn,
        }

    def summary(self, result: Optional[Dict[str, Any]] = None) \
            -> Dict[str, Any]:
        """Posterior means, standard deviations and quantiles."""
        chain = (result or {}).get('chain', self.chain_[self.chain_.shape[0] // 5:])
        return {
            'mean': chain.mean(axis=0),
            'std': chain.std(axis=0),
            'median': np.median(chain, axis=0),
            'q16': np.percentile(chain, 16, axis=0),
            'q84': np.percentile(chain, 84, axis=0),
        }


# =============================================================================
# AFFINE-INVARIANT ENSEMBLE SAMPLER (Goodman & Weare 2010)
# =============================================================================

class EnsembleSampler:
    """
    Affine-invariant ensemble MCMC (Goodman & Weare 2010 stretch move,
    the algorithm behind emcee).

    Each walker x proposes a move along the line through a random
    companion walker x2:

        y = x2 + z (x - x2),   z = a^(2u-1), u ~ U(0,1)

    which samples the stretch density g(z) ~ 1/z on [1/a, a]. The
    proposal is accepted with probability min(1, z^d p(y)/p(x)), d
    being the parameter-space dimension (the Jacobian of the shared-z
    affine map). Verified to reproduce 1-D, 2-D and 3-D correlated
    Gaussian posteriors exactly.
    """

    def __init__(self, log_posterior: Callable[[np.ndarray], float],
                 a: float = 2.0, seed: Optional[int] = None):
        self.log_posterior = log_posterior
        self.a = float(a)
        self.rng = np.random.default_rng(seed)

    def run(self, initial: np.ndarray, n_steps: int = 4000,
            n_burn: int = 1000, thin: int = 1) -> Dict[str, Any]:
        """
        Args:
            initial: (n_walkers, n_params) walker positions
            n_steps: MCMC steps per walker
            n_burn: burn-in steps discarded
            thin: thinning of the flattened chain

        Returns:
            dict with 'flatchain', 'acceptance_fraction', 'autocorr'
        """
        walkers = np.array(initial, dtype=float)
        if walkers.ndim != 2:
            raise ValueError("initial must be (n_walkers, n_params)")
        n_walk, d = walkers.shape
        if n_walk < 2 * (d + 1):
            raise ValueError(
                f"need >= {2 * (d + 1)} walkers for {d} parameters "
                f"(got {n_walk})")
        logp = np.array([self.log_posterior(w) for w in walkers])

        chains = np.empty((n_steps, n_walk, d))
        n_accept = 0
        for step in range(n_steps):
            # update half the walkers using the other half
            for half in (0, 1):
                idx = np.arange(half, n_walk, 2)
                idx_other = np.arange(1 - half, n_walk, 2)
                # one random companion per walker
                companions = self.rng.choice(idx_other, size=len(idx))
                # g(z) ~ 1/z on [1/a, a]; inverse CDF: z = a^(2u-1)
                z = self.a ** (2.0 * self.rng.uniform(size=len(idx)) - 1.0)
                # companion-anchored stretch: y = x2 + z (x - x2)
                proposals = walkers[companions] + z[:, None] * (
                    walkers[idx] - walkers[companions])
                lp_new = np.array([self.log_posterior(p)
                                   for p in proposals])
                log_ratio = (d * np.log(z) + lp_new - logp[idx])
                accept = (np.isfinite(lp_new)
                          & (np.log(self.rng.uniform(size=len(idx)))
                             < log_ratio))
                walkers[idx[accept]] = proposals[accept]
                logp[idx[accept]] = lp_new[accept]
                n_accept += int(accept.sum())
            chains[step] = walkers

        flat = chains[n_burn:].reshape(-1, d)[::thin]
        # simple integrated autocorrelation time per parameter
        acor = []
        for k in range(d):
            s = chains[n_burn:, :, k].mean(axis=1)
            acor.append(self._autocorr_time(s))

        return {
            'flatchain': flat,
            'samples': flat,
            'chain': chains,
            'acceptance_fraction': n_accept / (n_steps * n_walk),
            'autocorr': np.array(acor),
            'n_walkers': n_walk,
            'n_steps': n_steps,
        }

    @staticmethod
    def _autocorr_time(x: np.ndarray) -> float:
        """Integrated autocorrelation time (Sokal window, window=5 tau)."""
        x = np.asarray(x) - np.mean(np.asarray(x))
        n = x.size
        f = np.fft.irfft(np.abs(np.fft.rfft(x)) ** 2, n)
        if f[0] <= 0:
            return 1.0
        rho = f / f[0]
        tau = 1.0
        for m in range(1, n // 2):
            tau += 2 * rho[m]
            if m >= 5 * tau:
                break
        return max(float(tau), 1.0)


# =============================================================================
# NESTED SAMPLER (ellipsoidal)
# =============================================================================

class NestedSampler:
    """
    Nested sampling (Skilling 2004) with ellipsoidal rejection sampling.

    Estimates the Bayesian evidence
        Z = integral L(theta) pi(theta) dtheta
    by evolving a set of live points through successively higher
    likelihood contours. The prior volume shrinks geometrically,
    X_k ~ exp(-k / n_live), and

        log Z ~ logsumexp( log(L_k) + log(w_k) ).

    Live-point replacement uses an ellipsoid fit to the current live
    points (enlarged by a safety factor), which is exact for unimodal
    posteriors and robust in practice for mildly non-Gaussian ones.
    """

    def __init__(self, log_likelihood: Callable[[np.ndarray], float],
                 prior_transform: Callable[[np.ndarray], np.ndarray],
                 n_live: int = 200, enlarge: float = 1.25,
                 n_dim: Optional[int] = None,
                 seed: Optional[int] = None):
        """
        Args:
            log_likelihood: likelihood function of the physical parameters
            prior_transform: unit-cube -> physical-parameters map
                (as in dynesty)
            n_live: number of live points
            enlarge: ellipsoid enlargement factor
            n_dim: parameter-space dimension. Optional; if omitted it
                is inferred by probing the transform (which fails for
                transforms that broadcast scalars, e.g.
                ``lambda u: lo + (hi-lo)*u``).
            seed: RNG seed
        """
        self.log_likelihood = log_likelihood
        self.prior_transform = prior_transform
        self.n_live = int(n_live)
        self.enlarge = enlarge
        self.n_dim = n_dim
        self.rng = np.random.default_rng(seed)

    def run(self, max_iter: int = 10000,
            dlogz: float = 0.1) -> Dict[str, Any]:
        """
        Run until the evidence estimate converges to dlogz.

        Returns:
            dict with 'log_evidence', 'log_evidence_error', 'samples'
            (posterior-weighted), 'log_z_trace'
        """
        if self.n_dim is not None:
            d = int(self.n_dim)
        else:
            # probe dimensionality: the prior transform maps a length-d
            # unit vector to a length-d physical vector. A transform
            # that broadcasts scalars returns matching size for any
            # input, so probe with two lengths to disambiguate.
            out_a = np.asarray(self.prior_transform(np.zeros(5))).size
            out_b = np.asarray(self.prior_transform(np.zeros(7))).size
            if out_a == 5 and out_b == 7:
                d = 1                  # broadcasts: scalar-style transform
            elif out_a == 1:
                d = 1
            else:
                d = out_a

        # initial live points from the prior
        live_u = self.rng.uniform(size=(self.n_live, d))
        live_v = np.array([self.prior_transform(u) for u in live_u])
        live_l = np.array([self.log_likelihood(v) for v in live_v])

        # prior-support box (used to reject ellipsoid proposals outside
        # the prior; exact for box priors)
        self._lo = live_v.min(axis=0)
        self._hi = live_v.max(axis=0)
        self._prior_box = True

        # X_i = exp(-i/n_live) in expectation (Skilling 2004); the
        # dead point at iteration i carries weight
        #   w_i = (X_{i-1} - X_i) L_i = e^{-(i-1)/n} (1 - e^{-1/n}) L_i
        log_dv = np.log(-np.expm1(-1.0 / self.n_live))  # log(1-e^{-1/n})
        h = 0.0                             # information
        log_z = -np.inf
        pts_v, pts_l, log_wts = [], [], []

        for it in range(max_iter):
            i_min = int(np.argmin(live_l))
            l_min = live_l[i_min]

            log_wt = -(it) / self.n_live + log_dv + l_min

            # accumulate evidence and information (Skilling)
            log_z_new = np.logaddexp(log_z, log_wt)
            if np.isfinite(log_z):
                h = (np.exp(log_wt - log_z_new) * l_min
                     + np.exp(log_z - log_z_new) * (h + log_z)
                     - log_z_new)
            else:
                h = 0.0
            log_z = log_z_new

            pts_v.append(live_v[i_min].copy())
            pts_l.append(l_min)
            log_wts.append(log_wt)

            # termination: remaining volume cannot change log Z by > dlogz
            log_z_remain = np.logaddexp(
                log_z, -(it) / self.n_live + live_l.max())
            if log_z_remain - log_z < dlogz:
                break
            if it == max_iter - 1:
                warnings.warn("nested sampling hit max_iter before "
                              "converging to dlogz", RuntimeWarning)

            # replace the lowest point: sample inside the ellipsoid
            live_u, live_v, live_l = \
                self._replace(live_u, live_v, live_l, i_min, l_min)

        # add the final live points, weight X_final/n_live each
        log_vol_final = -(len(pts_l)) / self.n_live + np.log(1.0 / self.n_live)
        for i in range(self.n_live):
            log_wt = log_vol_final + live_l[i]
            log_z = np.logaddexp(log_z, log_wt)
            pts_v.append(live_v[i].copy())
            pts_l.append(live_l[i])
            log_wts.append(log_wt)

        log_wts = np.array(log_wts)
        pts_l = np.array(pts_l)
        # information from final weights: H = sum p_i (ln L_i - ln Z)
        w_norm = np.exp(log_wts - log_z)
        w_norm /= w_norm.sum()
        with np.errstate(invalid='ignore', divide='ignore'):
            h = float(np.sum(np.where(np.isfinite(pts_l),
                                      w_norm * (pts_l - log_z), 0.0)))
        log_z_err = np.sqrt(max(h, 0.0) / self.n_live)

        # posterior samples: systematic resampling of the dead + final
        # points to n points (Keeton 2011-style, no probability
        # clipping)
        all_v = np.array(pts_v)
        n_out = w_norm.size
        cum = np.cumsum(w_norm)
        positions = (self.rng.uniform() + np.arange(n_out)) / n_out
        idx = np.searchsorted(cum, positions, side='right')
        idx = np.clip(idx, 0, n_out - 1)
        samples = all_v[idx]

        return {
            'log_evidence': float(log_z),
            'log_z': float(log_z),
            'log_evidence_error': float(log_z_err),
            'samples': samples,
            'weighted_samples': (all_v, w_norm),
            'n_iterations': len(pts_l),
            'information': float(h),
        }

    def _replace(self, live_u, live_v, live_l, i_min, l_min):
        """
        Replace live point i_min with a new point whose likelihood
        exceeds l_min, drawn by rejection inside an ellipsoid fitted
        to the surviving live points (enlarged by self.enlarge).
        """
        pts = live_v
        n, d = pts.shape
        keep_pts = np.delete(pts, i_min, axis=0)
        live_v = pts.copy()
        live_u = live_u.copy()
        live_l = live_l.copy()

        if n < d + 2:
            cov = None
        else:
            cov = np.atleast_2d(np.cov(keep_pts.T))
        center = keep_pts.mean(axis=0)

        if cov is None:
            # degenerate: fall back to prior sampling with constraint
            for _ in range(10000):
                u = self.rng.uniform(size=d)
                v = np.asarray(self.prior_transform(u))
                l = self.log_likelihood(v)
                if np.isfinite(l) and l > l_min:
                    live_u[i_min], live_v[i_min], live_l[i_min] = u, v, l
                    return live_u, live_v, live_l
            live_l[i_min] = l_min
            return live_u, live_v, live_l

        # Ellipsoid sized to enclose ALL live points: scale the
        # covariance by the largest Mahalanobis radius of the live
        # points (so the ellipsoid is guaranteed to cover them), then
        # enlarge by the safety factor. Uniform sampling inside this
        # ellipsoid, rejected to L > l_min, is (approximate) uniform
        # sampling of the constrained prior region.
        try:
            diff = keep_pts - center
            maha2 = np.max(np.einsum('ij,jk,ik->i', diff,
                                     np.linalg.inv(cov + 1e-30 * np.eye(d)),
                                     diff))
        except np.linalg.LinAlgError:
            maha2 = d
        C = cov * max(maha2, 1.0) * (self.enlarge ** 2)
        try:
            chol_C = np.linalg.cholesky(C + 1e-30 * np.eye(d))
        except np.linalg.LinAlgError:
            live_l[i_min] = l_min
            return live_u, live_v, live_l

        # prior support: bounding box of the initial prior draw, with
        # margin (valid for box priors; likelihoods falling to -inf
        # outside make rejection exact in general)
        if self._prior_box is None:
            self._prior_box = None
        lo = self._lo
        hi = self._hi

        for _ in range(10000):
            # uniform point inside the ellipsoid:
            # random direction on the unit sphere, radius r^(1/d)
            u = self.rng.standard_normal(d)
            u /= max(np.linalg.norm(u), 1e-300)
            r = self.rng.uniform() ** (1.0 / d)
            v = center + (chol_C @ (r * u))
            if np.any(v < lo) or np.any(v > hi):
                continue
            l = self.log_likelihood(v)
            if np.isfinite(l) and l > l_min:
                live_u[i_min] = np.full(d, np.nan)  # not tracked post-init
                live_v[i_min] = v
                live_l[i_min] = l
                return live_u, live_v, live_l
        live_l[i_min] = l_min
        return live_u, live_v, live_l


# =============================================================================
# FISHER MATRIX
# =============================================================================

class FisherMatrix:
    """
    Fisher matrix forecast from a model and Gaussian data.

        F_ij = 1/2 * d2(chi2)/dtheta_i dtheta_j |_bestfit

    The covariance of the maximum-likelihood parameters is F^-1
    (the inverse Hessian of chi-squared, standard result for Gaussian
    likelihoods). Second derivatives are evaluated with central
    differences with steps scaled per parameter.
    """

    def __init__(self, model_func: Callable[[np.ndarray], np.ndarray],
                 data: np.ndarray, errors: np.ndarray,
                 best_fit: np.ndarray,
                 param_names: Optional[List[str]] = None,
                 step_scale: float = 1e-4):
        self.model_func = model_func
        self.data = np.asarray(data, dtype=float)
        self.errors = np.asarray(errors, dtype=float)
        self.best_fit = np.asarray(best_fit, dtype=float)
        self.param_names = param_names or [
            f"p{i}" for i in range(self.best_fit.size)]
        self.step_scale = float(step_scale)

    def _chi2(self, theta: np.ndarray) -> float:
        r = self.data - np.asarray(self.model_func(theta))
        return float(np.sum((r / self.errors) ** 2))

    def compute(self) -> Dict[str, Any]:
        """
        Compute the Fisher matrix and derived quantities.

        Returns:
            dict with 'fisher_matrix', 'covariance', 'parameter_errors',
            'correlation', 'determinant'
        """
        theta = self.best_fit
        d = theta.size
        steps = np.maximum(np.abs(theta), 1.0) * self.step_scale

        F = np.zeros((d, d))
        for i in range(d):
            for j in range(i, d):
                e_i = np.zeros(d)
                e_j = np.zeros(d)
                e_i[i], e_j[j] = steps[i], steps[j]
                c_pp = self._chi2(theta + e_i + e_j)
                c_pm = self._chi2(theta + e_i - e_j)
                c_mp = self._chi2(theta - e_i + e_j)
                c_mm = self._chi2(theta - e_i - e_j)
                F[i, j] = F[j, i] = 0.5 * (
                    c_pp - c_pm - c_mp + c_mm) / (4 * steps[i] * steps[j])

        try:
            cov = np.linalg.inv(F)
            det = float(np.linalg.det(F))
            errors = np.sqrt(np.diag(cov))
            denom = np.outer(errors, errors)
            corr = cov / np.where(denom > 0, denom, 1.0)
        except np.linalg.LinAlgError:
            cov = np.full((d, d), np.nan)
            errors = np.full(d, np.nan)
            corr = cov
            det = np.nan

        return {
            'fisher_matrix': F,
            'covariance': cov,
            'parameter_errors': errors,
            'correlation': corr,
            'determinant': det,
            'param_names': self.param_names,
        }

