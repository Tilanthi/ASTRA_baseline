"""Trading Analysis Package"""

def _degraded_warn(_module: str, _exc: BaseException) -> None:
    """Log why an optional import degraded instead of failing silently."""
    import logging
    logging.getLogger(__name__).warning(
        "%s unavailable (%s: %s) - dependent names set to None",
        _module, type(_exc).__name__, _exc,
    )


try:
    from .causal_analysis import MarketCausalAnalyzer, CausalSignal, CausalBacktester
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".causal_analysis", _exc)
    MarketCausalAnalyzer = CausalSignal = CausalBacktester = None  # degraded: unavailable
