"""
STAN_IX_ASTRO Mathematical Reasoning Module

This module contains enhanced mathematical reasoning capabilities for STAN,
including the Aletheia-style 3-agent architecture for IMO-ProofBench problems.

Components:
- AletheiaSTANSystem: Enhanced 3-agent architecture (Generator-Verifier-Reviser)
- AletheiaProofSystem: Basic 3-agent architecture
"""

try:
    from .aletheia_stan_architecture import (
    AletheiaSTANSystem,
    ProofStrategy,
    VerdictType,
    ProofAttempt,
    ValidationResult,
    GeneratorOutput
    )
except Exception:
    AletheiaSTANSystem = ProofStrategy = VerdictType = ProofAttempt = ValidationResult = GeneratorOutput = None  # degraded: unavailable

__all__ = [
    'AletheiaSTANSystem',
    'ProofStrategy',
    'VerdictType',
    'ProofAttempt',
    'ValidationResult',
    'GeneratorOutput'
]
