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
Bayesian Swarm Inference for Astronomy

This module implements the CORE INFERENCE ENGINE that combines:
1. Physics-based forward models (from physics.py)
2. Swarm exploration of parameter space
3. Proper Bayesian posterior estimation
4. MORK persistence for accumulated knowledge

The key innovation: Swarm agents explore the LIKELIHOOD LANDSCAPE,
not a simplified proxy metric. Each agent evaluates the ACTUAL
physics-based chi-squared.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
from pathlib import Path
import json

# Local imports
from .physics import PhysicsEngine, ForwardModel, AstrophysicalConstraints


# =============================================================================
# INFERENCE RESULT STRUCTURES
# =============================================================================

@dataclass
class ParameterEstimate:
    """Estimate of a single parameter with uncertainties"""
    name: str
    value: float
    uncertainty_lower: float
    uncertainty_upper: float
    unit: str
    confidence_level: float = 0.68  # 1-sigma by default

    @property
    def symmetric_uncertainty(self) -> float:
        return (self.uncertainty_lower + self.uncertainty_upper) / 2

    def __str__(self):
        return f"{self.name} = {self.value:.4g} (+{self.uncertainty_upper:.2g}/-{self.uncertainty_lower:.2g}) {self.unit}"


@dataclass
class InferenceResult:
    """Complete result of Bayesian inference"""
    parameters: Dict[str, ParameterEstimate]
    chi_squared: float
    degrees_of_freedom: int
    reduced_chi_squared: float
    log_evidence: float
    # NOTE (audit M1): for method="swarm" this is the set of particle
    # personal-best positions -- an optimiser trace, not a posterior sample.
    # Do not histogram it as a posterior; the parameter uncertainties come from
    # the chi^2 curvature at the optimum instead.
    posterior_samples: Optional[np.ndarray] = None
    convergence_achieved: bool = True
    n_evaluations: int = 0
    wall_time: float = 0.0
    method: str = "swarm"

    def summary(self) -> str:
        lines = ["=" * 60]
        lines.append("INFERENCE RESULT SUMMARY")
        lines.append("=" * 60)
        lines.append(f"Method: {self.method}")
        lines.append(f"Chi-squared: {self.chi_squared:.4f}")
        lines.append(f"Reduced chi-squared: {self.reduced_chi_squared:.4f}")
        lines.append(f"Degrees of freedom: {self.degrees_of_freedom}")
        lines.append(f"Log evidence: {self.log_evidence:.2f}")
        lines.append(f"Converged: {self.convergence_achieved}")
        lines.append(f"Function evaluations: {self.n_evaluations}")
        lines.append(f"Wall time: {self.wall_time:.2f} s")
        lines.append("-" * 60)
        lines.append("PARAMETERS:")
        for name, est in self.parameters.items():
            lines.append(f"  {est}")
        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# SWARM AGENT FOR BAYESIAN INFERENCE
# =============================================================================

# =============================================================================
# BAYESIAN SWARM INFERENCE ENGINE
# (re-implemented 2026-08 from the surviving dataclasses + call sites; the
#  original body was lost to file truncation before the August 2026 audit)
# =============================================================================

class _Particle:
    """One swarm agent exploring the likelihood landscape."""

    __slots__ = ('position', 'velocity', 'best_position', 'best_chi_squared')

    def __init__(self, position: np.ndarray, velocity: np.ndarray):
        self.position = position
        self.velocity = velocity
        self.best_position = position.copy()
        self.best_chi_squared = np.inf


class BayesianSwarmInference:
    """
    Particle-swarm Bayesian inference over a physics forward model.

    Swarm agents explore the ACTUAL physics chi-squared landscape via
    ``PhysicsEngine.compute_chi_squared(model_name, parameters, observations)``.
    Parameter uncertainties are estimated from the final swarm, importance
    weighted by exp(-0.5 * delta-chi-squared); the log evidence uses a
    Laplace approximation around the best fit with uniform priors over the
    declared bounds.
    """

    def __init__(self, physics_engine, model_name: str):
        self.physics = physics_engine
        self.model_name = model_name
        self._bounds: Dict[str, Tuple[float, float, str]] = {}
        # Swarm state (global best may be pre-seeded by callers, e.g. from a
        # differential-evolution solution, before calling infer())
        self.global_best_position: Optional[np.ndarray] = None
        self.global_best_chi_squared: float = np.inf
        self.n_evaluations: int = 0

    # ------------------------------------------------------------------ setup
    def set_parameter_bounds(self, bounds: Dict[str, Tuple[float, float, str]]) -> None:
        """Declare inference parameters as name -> (low, high, unit)."""
        self._bounds = dict(bounds)
        names = list(self._bounds)
        if self.global_best_position is not None and len(names) != len(self.global_best_position):
            # stale seed from a different parametrisation - discard it
            self.global_best_position = None
            self.global_best_chi_squared = np.inf

    @property
    def parameter_names(self) -> List[str]:
        return list(self._bounds)

    # ----------------------------------------------------------------- fitness
    def _chi_squared(self, vector: np.ndarray, observations: Dict) -> float:
        params = {name: float(v) for name, v in zip(self._bounds, vector)}
        self.n_evaluations += 1
        try:
            chi2 = self.physics.compute_chi_squared(self.model_name, params, observations)
        except (ValueError, KeyError, ZeroDivisionError, FloatingPointError):
            return 1e10
        if not np.isfinite(chi2):
            return 1e10
        return float(chi2)

    def _degrees_of_freedom(self, observations: Dict, n_params: int) -> int:
        n_data = 0
        for value in observations.values():
            if isinstance(value, np.ndarray):
                n_data += value.size
            elif isinstance(value, (list, tuple)):
                n_data += len(value)
        dof = max(1, n_data - n_params)
        return dof if n_data else max(1, n_params)  # unconstrained fallback

    # ------------------------------------------------------------------- infer
    def infer(self,
              observations: Dict,
              n_particles: int = 50,
              n_iterations: int = 100,
              convergence_threshold: float = 1e-8,
              verbose: bool = False) -> InferenceResult:
        """
        Run particle-swarm optimisation and summarise the posterior.

        Returns an :class:`InferenceResult` with per-parameter MAP estimates
        and 68% (1-sigma) uncertainties.
        """
        import time as _time
        t0 = _time.time()

        if not self._bounds:
            raise ValueError("No parameter bounds set - call set_parameter_bounds() first")

        names = self.parameter_names
        n_dim = len(names)
        lo = np.array([self._bounds[n][0] for n in names])
        hi = np.array([self._bounds[n][1] for n in names])
        span = hi - lo

        rng = np.random.default_rng(42)

        # --- initialise swarm -------------------------------------------------
        positions = lo + span * rng.random((n_particles, n_dim))
        # First particle inherits any pre-seeded global best (e.g. DE solution)
        if self.global_best_position is not None:
            seed = np.clip(np.asarray(self.global_best_position, dtype=float), lo, hi)
            positions[0] = seed
            # Cluster a quarter of the swarm around the seed (searches along
            # degeneracy directions), remaining particles explore the full prior
            n_cluster = max(2, n_particles // 4)
            positions[1:n_cluster] = seed + 0.05 * span * rng.standard_normal((n_cluster - 1, n_dim))
            positions[1:n_cluster] = np.clip(positions[1:n_cluster], lo, hi)
        elif self.global_best_chi_squared < np.inf:
            self.global_best_chi_squared = np.inf

        velocities = -span + 2 * span * rng.random((n_particles, n_dim))  # [-span, span]
        max_velocity = 0.2 * span

        particles = [_Particle(positions[i], velocities[i]) for i in range(n_particles)]

        # Evaluate initial positions
        for p in particles:
            p.best_chi_squared = self._chi_squared(p.position, observations)
            if p.best_chi_squared < self.global_best_chi_squared:
                self.global_best_chi_squared = p.best_chi_squared
                self.global_best_position = p.position.copy()

        # --- PSO main loop ----------------------------------------------------
        w_start, w_end = 0.9, 0.4          # inertia decay
        c1, c2 = 2.0, 2.0                  # cognitive / social coefficients
        stall_iterations = 0
        swarm_converged = False            # FIX(audit M1): real convergence flag

        for it in range(n_iterations):
            w = w_start - (w_start - w_end) * it / max(1, n_iterations - 1)
            previous_best = self.global_best_chi_squared

            for p in particles:
                r1 = rng.random(n_dim)
                r2 = rng.random(n_dim)
                p.velocity = (w * p.velocity
                              + c1 * r1 * (p.best_position - p.position)
                              + c2 * r2 * (self.global_best_position - p.position))
                p.velocity = np.clip(p.velocity, -max_velocity, max_velocity)
                p.position = np.clip(p.position + p.velocity, lo, hi)

                chi2 = self._chi_squared(p.position, observations)
                if chi2 < p.best_chi_squared:
                    p.best_chi_squared = chi2
                    p.best_position = p.position.copy()
                if chi2 < self.global_best_chi_squared:
                    self.global_best_chi_squared = chi2
                    self.global_best_position = p.position.copy()

            improvement = previous_best - self.global_best_chi_squared
            if verbose and (it % 10 == 0 or it == n_iterations - 1):
                print(f"  [swarm] iter {it:4d}: chi2 = {self.global_best_chi_squared:.6g}")

            if improvement < convergence_threshold:
                stall_iterations += 1
                if stall_iterations >= 10:
                    if verbose:
                        print(f"  [swarm] converged after {it + 1} iterations")
                    # FIX(audit M1): record whether the swarm actually stalled;
                    # `convergence_achieved` used to be hardcoded True.
                    swarm_converged = True
                    break
            else:
                stall_iterations = 0

        best = self.global_best_position.copy()
        best_chi2 = self.global_best_chi_squared

        # --- parameter uncertainties -----------------------------------------
        # FIX(audit M1): the uncertainties used to be percentiles of the swarm's
        # personal-best positions, importance weighted by exp(-dchi2/2). A PSO
        # swarm COLLAPSES onto the optimum, so that spread measures how far the
        # optimiser has converged, not how well the data constrain the model:
        # on a 20-point straight-line fit with sigma = 0.5 it returned
        # +0.00000/-0.00000 for both parameters at n_iterations = 50, and error
        # bars that varied non-monotonically with iteration count.
        # We now use the local curvature of chi^2 at the optimum, exactly as
        # sed_fitting.SEDFitter does (and as verified against the analytic
        # Fisher matrix there): with chi^2 = -2 ln L, the covariance is
        # C = 2 H^-1, H_ij = d^2 chi^2 / dtheta_i dtheta_j.
        # If H is not positive definite (a saddle, a bound-limited or a
        # non-smooth minimum) NO number is invented: the uncertainties are
        # returned as NaN, `convergence_achieved` is False and a warning is
        # issued. A wrong error bar is worse than no error bar.
        cov, cov_ok = self._hessian_covariance(best, observations, lo, hi, span)

        # The swarm's weighted personal bests are still returned as
        # `posterior_samples`, but they are an optimiser trace, NOT a posterior
        # sample -- see the InferenceResult docstring.
        final = np.array([p.best_position for p in particles])
        final_chi2 = np.array([p.best_chi_squared for p in particles])
        delta = final_chi2 - best_chi2
        weights = np.exp(-0.5 * np.clip(delta, 0, 50))
        weights /= weights.sum()
        active = weights > 1e-3
        post = final[active] if active.sum() >= 1 else best.reshape(1, -1)

        parameters = {}
        for i, name in enumerate(names):
            sigma_i = float(np.sqrt(cov[i, i])) if cov_ok else float('nan')
            parameters[name] = ParameterEstimate(
                name=name,
                value=float(best[i]),
                uncertainty_lower=sigma_i,
                uncertainty_upper=sigma_i,
                unit=self._bounds[name][2],
            )

        dof = self._degrees_of_freedom(observations, n_dim)

        # Laplace log-evidence with uniform priors over the declared bounds:
        #   ln Z ~ ln L_max + (d/2) ln(2 pi) + (1/2) ln det C - sum ln(range)
        # FIX(audit M1): the two Gaussian-normalisation terms had the WRONG SIGN
        # (the code used -0.5 d ln 2pi - 0.5 ln det C), and det C was taken from
        # the collapsed swarm, so log Z varied by 9.4 nats on identical data
        # depending only on the iteration count. ln L_max = -chi2_min/2 drops
        # the (constant) 1/sqrt(2 pi sigma^2) factors of the likelihood, so this
        # is an evidence RATIO usable only between models sharing the same data
        # and error bars.
        if cov_ok:
            _, logdet = np.linalg.slogdet(cov)
            log_evidence = (-0.5 * best_chi2
                            + 0.5 * n_dim * np.log(2 * np.pi)
                            + 0.5 * logdet
                            - np.sum(np.log(span)))
        else:
            log_evidence = float('nan')

        return InferenceResult(
            parameters=parameters,
            chi_squared=best_chi2,
            degrees_of_freedom=dof,
            reduced_chi_squared=best_chi2 / dof,
            log_evidence=float(log_evidence),
            posterior_samples=post,
            # FIX(audit M1): was hardcoded True. NOTE this flag means "the swarm
            # stalled and a valid curvature was obtained", NOT "the global
            # minimum was found": PSO can and does stall above the true
            # chi^2 minimum, so a converged run can still be a local optimum.
            convergence_achieved=bool(swarm_converged and cov_ok),
            n_evaluations=self.n_evaluations,
            wall_time=_time.time() - t0,
            method="swarm",
        )

    # ------------------------------------------------------- uncertainties
    def _hessian_covariance(self, theta: np.ndarray, observations: Dict,
                            lo: np.ndarray, hi: np.ndarray,
                            span: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Covariance from the curvature of chi^2 at `theta`: C = 2 H^-1.

        The step for each parameter is chosen adaptively so that the one-sided
        chi^2 increment is of order unity (the scale on which the likelihood
        actually varies); the Hessian is then formed with central differences.

        Returns (covariance, ok). `ok` is False -- and the covariance is filled
        with NaN -- when the Hessian is singular, not positive definite, or the
        optimum sits against a parameter bound, i.e. whenever a Gaussian error
        bar would be meaningless.
        """
        n = theta.size
        chi2_0 = self._chi_squared(theta, observations)

        # --- adaptive step per parameter -------------------------------------
        steps = np.zeros(n)
        for i in range(n):
            h = 1e-3 * span[i]
            for _ in range(40):
                if h <= 0 or not np.isfinite(h):
                    break
                probe = theta.copy()
                probe[i] = np.clip(theta[i] + h, lo[i], hi[i])
                if probe[i] == theta[i]:
                    probe[i] = np.clip(theta[i] - h, lo[i], hi[i])
                d_chi2 = abs(self._chi_squared(probe, observations) - chi2_0)
                if d_chi2 < 1e-2:
                    h *= 2.0
                elif d_chi2 > 4.0:
                    h *= 0.5
                else:
                    break
                if h > 0.5 * span[i]:
                    h = 0.5 * span[i]
                    break
            steps[i] = h

        # --- reject bound-limited optima -------------------------------------
        at_bound = (theta - lo < steps) | (hi - theta < steps)
        if np.any(at_bound):
            import warnings as _warnings
            _warnings.warn(
                "best fit lies within one finite-difference step of a parameter "
                "bound; the chi^2 curvature there is not a valid uncertainty, "
                "so NaN error bars are returned", RuntimeWarning)
            return np.full((n, n), np.nan), False

        # --- central-difference Hessian of chi^2 ------------------------------
        hess = np.zeros((n, n))
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = steps[i]
            f_p = self._chi_squared(theta + ei, observations)
            f_m = self._chi_squared(theta - ei, observations)
            hess[i, i] = (f_p - 2.0 * chi2_0 + f_m) / steps[i] ** 2
            for j in range(i + 1, n):
                ej = np.zeros(n)
                ej[j] = steps[j]
                f_pp = self._chi_squared(theta + ei + ej, observations)
                f_pm = self._chi_squared(theta + ei - ej, observations)
                f_mp = self._chi_squared(theta - ei + ej, observations)
                f_mm = self._chi_squared(theta - ei - ej, observations)
                hess[i, j] = hess[j, i] = (
                    (f_pp - f_pm - f_mp + f_mm) / (4.0 * steps[i] * steps[j]))

        try:
            cov = 2.0 * np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            import warnings as _warnings
            _warnings.warn("chi^2 Hessian is singular at the best fit; "
                           "returning NaN uncertainties", RuntimeWarning)
            return np.full((n, n), np.nan), False

        eigenvalues = np.linalg.eigvalsh(0.5 * (cov + cov.T))
        if not np.all(np.isfinite(cov)) or np.min(eigenvalues) <= 0:
            import warnings as _warnings
            _warnings.warn(
                "chi^2 curvature at the best fit is not positive definite "
                "(saddle point or degenerate parameters); returning NaN "
                "uncertainties rather than an invented number", RuntimeWarning)
            return np.full((n, n), np.nan), False

        return cov, True
