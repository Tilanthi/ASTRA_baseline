"""Trading Analysis Package"""
try:
    from .causal_analysis import MarketCausalAnalyzer, CausalSignal, CausalBacktester
except Exception:
    MarketCausalAnalyzer = CausalSignal = CausalBacktester = None  # degraded: unavailable
