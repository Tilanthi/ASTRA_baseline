"""
Trading Package
"""

try:
    from .analysis.causal_analysis import (
    MarketCausalAnalyzer,
    CausalSignal,
    CausalBacktester
    )
except Exception:
    MarketCausalAnalyzer = CausalSignal = CausalBacktester = None  # degraded: unavailable

__all__ = [
    "MarketCausalAnalyzer",
    "CausalSignal",
    "CausalBacktester",
]
