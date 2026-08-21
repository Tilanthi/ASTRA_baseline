"""
High Performance Computing Domain Module for ASTRA

Parallelization, GPU computing, optimization

STATUS: registered but NOT IMPLEMENTED. This module was one of 48 byte-identical
copies of a 110-line template whose `process_query` returned
``f"{description}: Analysis of '{query}'"`` with a hard-coded
``confidence=0.7`` and performed no computation.

There is no computational backend for this domain anywhere in the codebase, and
inventing authored prose to fill the gap would present writing as analysis. It
therefore reports `confidence=0.0` and
`implementation_status=NO_IMPLEMENTATION`, which is a useful signal to an
orchestrator in a way that 0.7 on an echoed query was not.

Version: 2.0.0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .. import DomainConfig, register_domain
from .._computational import ComputationalCapability, ComputationalDomainModule, ImplementationStatus

logger = logging.getLogger(__name__)


class HPCDomain(ComputationalDomainModule):
    """
    High Performance Computing.

    Registered so that queries in this area are recognised and routed, but no
    numerical capability is implemented.
    """

    implementation_status = ImplementationStatus.NO_IMPLEMENTATION
    referral = ("This is an engineering concern, not a physics domain; no capability is provided.")

    def get_default_config(self) -> DomainConfig:
        return self.get_config()

    def get_config(self) -> DomainConfig:
        return DomainConfig(
            domain_name="hpc",
            version="2.0.0",
            dependencies=[],
            description="Parallelization, GPU computing, optimization",
            keywords=['hpc', 'parallel computing', 'gpu', 'cuda', 'optimization', 'mpi', 'openmp'],
            capabilities=[],
        )

    def initialize(self, global_config: Dict[str, Any]) -> None:
        super().initialize(global_config)
        logger.info("Initialising %s domain (no computational backend)", "hpc")

    def build_capabilities(self) -> List[ComputationalCapability]:
        """No computational backend exists for this domain."""
        return []


def create_hpc_domain() -> HPCDomain:
    """Create a High Performance Computing domain instance."""
    return HPCDomain()


try:
    register_domain(HPCDomain)
except ImportError:  # pragma: no cover - registry optional at import time
    pass
