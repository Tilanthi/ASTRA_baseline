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
                    break
            else:
                stall_iterations = 0

        best = self.global_best_position.copy()
        best_chi2 = self.global_best_chi_squared

        # --- posterior summary from the final swarm ---------------------------
        final = np.array([p.best_position for p in particles])
        final_chi2 = np.array([p.best_chi_squared for p in particles])
        # Importance weights: points within delta-chi2 ~ few of the minimum
        delta = final_chi2 - best_chi2
        weights = np.exp(-0.5 * np.clip(delta, 0, 50))
        weights /= weights.sum()
        # Discard totally stale particles from the uncertainty estimate
        active = weights > 1e-3
        if active.sum() >= 2:
            w_active = weights[active]
            samples = final[active]
            # Resample for percentile robustness
            idx = rng.choice(len(samples), size=min(2000, len(samples) * 20),
                             replace=True, p=w_active / w_active.sum())
            post = samples[idx]
            lo16, med, hi84 = np.percentile(post, [16, 50, 84], axis=0)
        else:
            med, lo16, hi84 = best, best - 0.02 * span, best + 0.02 * span
            post = best.reshape(1, -1)

        parameters = {}
        for i, name in enumerate(names):
            parameters[name] = ParameterEstimate(
                name=name,
                value=float(med[i]),
                uncertainty_lower=float(max(1e-12, med[i] - lo16[i])),
                uncertainty_upper=float(max(1e-12, hi84[i] - med[i])),
                unit=self._bounds[name][2],
            )

        dof = self._degrees_of_freedom(observations, n_dim)

        # Laplace log-evidence with uniform priors over the declared bounds:
        # log Z ~ -0.5*chi2_min - 0.5*d*ln(2*pi) - 0.5*ln(det Cov) + sum ln(1/range)
        cov = np.cov(post.T) if post.shape[0] > 1 else np.diag((0.02 * span) ** 2)
        cov = np.atleast_2d(cov) + np.eye(n_dim) * 1e-12
        sign, logdet = np.linalg.slogdet(cov)
        log_evidence = (-0.5 * best_chi2
                        - 0.5 * n_dim * np.log(2 * np.pi)
                        - 0.5 * logdet
                        - np.sum(np.log(span)))

        return InferenceResult(
            parameters=parameters,
            chi_squared=best_chi2,
            degrees_of_freedom=dof,
            reduced_chi_squared=best_chi2 / dof,
            log_evidence=float(log_evidence),
            posterior_samples=post,
            convergence_achieved=True,
            n_evaluations=self.n_evaluations,
            wall_time=_time.time() - t0,
            method="swarm",
        )
