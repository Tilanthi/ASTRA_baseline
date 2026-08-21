"""
Signal Processing Domain Module for ASTRA

Time series analysis, periodograms, wavelets, filtering.

This module was one of 48 byte-identical copies of a 110-line template whose
`process_query` returned ``f"{description}: Analysis of '{query}'"`` with a
hard-coded ``confidence=0.7`` and performed no computation. It now wraps
:mod:`astra_core.astro_physics.time_series_analysis` and reports a confidence
derived from whether a computation actually ran.

Scope of what is wired
----------------------
The estimators that survived the audit: the FFT/Lomb-Scargle periodogram
(Parseval-correct; the Horne-Baliunas normalisation was corrected in audit
fix C15/B8.1, which had been inflating the power by N/2 and flagging 300/300
pure-noise realisations as significant) and the pairwise structure function
(verified against a direct non-periodic estimator to 0.0 relative error).

`PowerSpectrumAnalyzer.classify` is NOT exposed: audit flag B8.4 (red noise
labelled "quasi_periodic" when its peak is formally significant) is still
open, and it is a taxonomy decision rather than a measurement.

Two of the three capabilities take the series itself, so they cannot be
driven from a free-text query -- `extract_parameters` only recognises
scalars. They are still declared, verified and callable programmatically
(``domain.computations[i].run(time=..., values=...)``); a query that names
one without supplying the data gets an honest "did not supply" answer rather
than an invented number.

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


class SignalProcessingDomain(ComputationalDomainModule):
    """
    Signal processing for astronomical time series.

    Backed by `astro_physics.time_series_analysis`.
    """

    implementation_status = ImplementationStatus.COMPUTATIONAL

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="signal_processing",
            version="2.0.0",
            dependencies=[],
            description=("Time series analysis, periodograms, wavelets, "
                         "filtering"),
            keywords=[
                "signal processing", "time_series", "periodogram", "wavelet",
                "filtering", "lomb-scargle", "power spectrum", "nyquist",
                "structure function", "sampling", "cadence", "period",
            ],
            capabilities=[
                "nyquist_frequency", "dominant_period", "structure_function",
            ],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising signal_processing domain")

    def build_capabilities(self) -> List[ComputationalCapability]:
        from ...astro_physics.time_series_analysis import (
            TimeSeries, compute_structure_function, detect_periodicity,
        )

        # NOTE ON UNITS. `time_series_analysis` is unit-agnostic: every
        # returned quantity carries the units of the input series. The
        # declared units below therefore state the convention used by the
        # reference cases (time in days), and each self_check is an
        # analytically known answer for a signal whose result does not
        # depend on that choice:
        #   * nyquist_frequency: f_Ny = 1/(2 dt) exactly;
        #   * dominant_period: a sinusoid of period 5 d sampled at 0.1 d over
        #     100 d puts f = 0.2 /d exactly on the FFT grid (spacing 0.01 /d);
        #   * structure_function: for a linear ramp y = a t on N uniform
        #     samples, SF_2 = a^2 dt^2 * sum_{d=1}^{N-2}(N-d)d^2 /
        #     (N(N-1)/2 - 1) (the estimator's single geometric bin is
        #     half-open, so the single longest-lag pair is excluded);
        #     N = 50, dt = a = 1 gives 518224/1224 = 423.385621, which an
        #     independent O(N^2) double loop reproduces exactly.

        _t_sin = np.arange(0.0, 100.0, 0.1)
        _y_sin = np.sin(2.0 * np.pi * _t_sin / 5.0)
        _t_ramp = np.arange(50.0)
        _y_ramp = _t_ramp.copy()

        def nyquist_frequency(cadence: float) -> float:
            t = np.arange(4) * cadence
            return float(TimeSeries(t, np.zeros(4)).nyquist_frequency)

        def dominant_period(time, values) -> float:
            return float(detect_periodicity(np.asarray(time, dtype=float),
                                            np.asarray(values, dtype=float)
                                            )['best_period'])

        def structure_function(time, values) -> float:
            sf = compute_structure_function(np.asarray(time, dtype=float),
                                            np.asarray(values, dtype=float),
                                            order=2, n_bins=1)
            return float(sf['sf'][0])

        return [
            ComputationalCapability(
                name="nyquist_frequency",
                description="Nyquist frequency of a sampled time series",
                function=nyquist_frequency,
                parameters=[("cadence|dt|sampling_interval", "d",
                             "sampling interval")],
                returns=("f_Nyquist", "1/d"),
                reference="f_Ny = 1 / (2 dt) (Shannon 1949)",
                test_ref="test_domain_capabilities.py",
                self_check=({"cadence": 0.1}, 5.0, 1e-9),
            ),
            ComputationalCapability(
                name="dominant_period",
                description=("Dominant period of a light curve from its "
                             "periodogram"),
                function=dominant_period,
                parameters=[
                    ("time", "d", "sample times, strictly increasing"),
                    ("values", "flux", "measured values"),
                ],
                returns=("P_best", "d"),
                reference=("P = 1/f at the periodogram maximum; FFT "
                           "periodogram for even sampling, Lomb-Scargle "
                           "(Scargle 1982; Horne & Baliunas 1986) otherwise"),
                test_ref="test_domain_capabilities.py",
                self_check=({"time": _t_sin, "values": _y_sin}, 5.0, 1e-6),
            ),
            ComputationalCapability(
                name="structure_function",
                description=("Second-order structure function of a light "
                             "curve over its full lag range"),
                function=structure_function,
                parameters=[
                    ("time", "d", "sample times, strictly increasing"),
                    ("values", "flux", "measured values"),
                ],
                returns=("SF_2", "flux^2"),
                reference=("SF_2(tau) = < |x(t+tau) - x(t)|^2 >, averaged "
                           "over all sample pairs (Simonetti, Cordes & "
                           "Heeschen 1985)"),
                test_ref="test_domain_capabilities.py",
                self_check=({"time": _t_ramp, "values": _y_ramp},
                            423.385621, 1e-6),
            ),
        ]


def create_signal_processing_domain() -> SignalProcessingDomain:
    """Create a SignalProcessingDomain instance."""
    return SignalProcessingDomain()


# Domain registration
try:
    register_domain(SignalProcessingDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
