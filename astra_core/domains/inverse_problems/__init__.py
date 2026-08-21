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
Inverse Problems Domain Module for ASTRA

Deconvolution, reconstruction, regularization.

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation. It now wraps
:mod:`astra_core.astro_physics.uncertainty_quantification` and reports a
confidence derived from whether a computation actually ran.

Scope of what is wired
----------------------
The Fisher-matrix layer, i.e. the linear inverse problem: what the data
constrain, given a design. `uncertainty_quantification` is the
best-verified module in `astro_physics` -- during the August 2026 audit the
Fisher matrix reproduced the analytic result to 0.03 %, Metropolis-Hastings
recovered a correlated 2-D Gaussian exactly, and nested sampling recovered
ln Z within 1 sigma in d = 1, 2, 3.

The samplers themselves (`MetropolisHastings`, `EnsembleSampler`,
`NestedSampler`) are correct but stochastic, so they cannot carry a
reproducible `self_check` and are not declared as capabilities here; they
remain available directly from `astro_physics`. Nothing from
`inference.BayesianSwarmInference` is wired: audit M1 (uncertainties are the
collapse width of a PSO swarm, not a posterior) is still open.

Version: 2.0.0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from .. import DomainConfig, register_domain
from .._computational import (
    ComputationalCapability,
    ComputationalDomainModule,
    ImplementationStatus,
)

logger = logging.getLogger(__name__)


class InverseProblemsDomain(ComputationalDomainModule):
    """
    Inverse problems: parameter recovery and its uncertainty.

    Backed by `astro_physics.uncertainty_quantification.FisherMatrix`.
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="inverse_problems",
            version="2.0.0",
            dependencies=[],
            description="Deconvolution, reconstruction, regularization",
            keywords=[
                "inverse problem", "deconvolution", "reconstruction",
                "regularization", "ill-posed", "fisher matrix", "forecast",
                "parameter error", "covariance", "degeneracy",
                "linear inversion",
            ],
            capabilities=[
                "fisher_slope_error", "fisher_intercept_error",
                "fisher_parameter_correlation",
            ],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising inverse_problems domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.uncertainty_quantification import FisherMatrix

        # NOTE ON UNITS. The Fisher matrix of a straight line y = a + b x
        # with N equal-error points is F = [[N, sum x], [sum x, sum x^2]]
        # / sigma^2, so the returned errors carry the units of the data
        # (intercept) and of data/abscissa (slope). The reference cases
        # below are the exact analytic inverses, derived independently of
        # the module:
        #   S_xx = sum (x_i - xbar)^2 = X^2 N(N+1) / (12 (N-1))
        #        = 192.5 for N = 21, X = 10
        #   sigma_b = sigma / sqrt(S_xx)              = 0.5/13.87444 = 0.0360375
        #   sigma_a = sigma sqrt(1/N + xbar^2/S_xx)   = 0.5*0.4212946 = 0.2106473
        #   corr(a,b) = -xbar / sqrt(<x^2>)           = -5/5.845226 = -0.8553989

        def _forecast(n_points: float, x_max: float, sigma: float) -> Dict:
            n = int(round(n_points))
            if n < 3:
                raise ValueError("n_points must be at least 3")
            x = np.linspace(0.0, float(x_max), n)
            fisher = FisherMatrix(
                model_func=lambda theta: theta[0] + theta[1] * x,
                data=np.zeros(n),
                errors=np.full(n, float(sigma)),
                best_fit=np.array([0.0, 0.0]),
                param_names=["intercept", "slope"])
            return fisher.compute()

        params = [
            ("n_points|n", "count", "number of measurements"),
            ("x_max|baseline|span", "abscissa units",
             "abscissa span, sampled uniformly from 0"),
            ("sigma|error|noise", "data units",
             "1-sigma error, identical on every point"),
        ]

        return [
            ComputationalCapability(
                name="fisher_slope_error",
                description=("Fisher forecast of the slope error of a "
                             "straight-line fit"),
                function=lambda n_points, x_max, sigma: float(
                    _forecast(n_points, x_max, sigma)['parameter_errors'][1]),
                parameters=params,
                returns=("sigma_slope", "data units / abscissa units"),
                reference=("sigma_b = sigma / sqrt(sum (x_i - xbar)^2), the "
                           "(1,1) element of F^-1 for y = a + b x "
                           "(Tegmark, Taylor & Heavens 1997)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"n_points": 21.0, "x_max": 10.0, "sigma": 0.5},
                            0.03603750, 1e-4),
            ),
            ComputationalCapability(
                name="fisher_intercept_error",
                description=("Fisher forecast of the intercept error of a "
                             "straight-line fit"),
                function=lambda n_points, x_max, sigma: float(
                    _forecast(n_points, x_max, sigma)['parameter_errors'][0]),
                parameters=params,
                returns=("sigma_intercept", "data units"),
                reference=("sigma_a = sigma sqrt(1/N + xbar^2 / "
                           "sum (x_i - xbar)^2), the (0,0) element of F^-1"),
                test_ref="test_domain_capabilities.py",
                self_check=({"n_points": 21.0, "x_max": 10.0, "sigma": 0.5},
                            0.21064732, 1e-4),
            ),
            ComputationalCapability(
                name="fisher_parameter_correlation",
                description=("Forecast correlation between slope and "
                             "intercept (parameter degeneracy)"),
                function=lambda n_points, x_max, sigma: float(
                    _forecast(n_points, x_max, sigma)['correlation'][0, 1]),
                parameters=params,
                returns=("corr(a,b)", "dimensionless"),
                reference=("corr = -xbar / sqrt(<x^2>), the off-diagonal of "
                           "the normalised F^-1; independent of sigma"),
                test_ref="test_domain_capabilities.py",
                self_check=({"n_points": 21.0, "x_max": 10.0, "sigma": 0.5},
                            -0.85539892, 1e-4),
            ),
        ]


def create_inverse_problems_domain() -> InverseProblemsDomain:
    """Create an InverseProblemsDomain instance."""
    return InverseProblemsDomain()


# Domain registration
try:
    register_domain(InverseProblemsDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
