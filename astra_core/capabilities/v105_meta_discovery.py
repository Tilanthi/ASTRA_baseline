"""
V105 Meta-Discovery Transfer -- compatibility module.

The V5.0 discovery capabilities were originally published as flat modules at
``astra_core.capabilities.v105_meta_discovery``. The subpackage reorganisation moved the
implementation to ``astra_core.capabilities.discovery.meta_discovery`` but left nothing behind
at the old path, so every script written against the published API -- including
this repository's own ``tests/test_discovery/test_v5_capabilities.py`` -- broke
with ``ModuleNotFoundError``.

On a developer machine that also held older ASTRA checkouts the import could
still succeed by picking the module up from a *different tree*, which made the
breakage invisible locally while a clean clone from GitHub failed. This module
re-exports the canonical implementation so that a standalone install behaves
the same way.

There is no separate implementation here: these are the same objects.

    >>> from astra_core.capabilities.v105_meta_discovery import MetaDiscoveryTransferEngine
    >>> from astra_core.capabilities.discovery.meta_discovery import MetaDiscoveryTransferEngine as canonical
    >>> MetaDiscoveryTransferEngine is canonical
    True

New code should import from ``astra_core.capabilities.discovery.meta_discovery``.
"""

from __future__ import annotations

from .discovery.meta_discovery import (  # noqa: F401
    MetaDiscoveryTransferEngine,
    DiscoveryPattern,
    create_meta_discovery_transfer_engine,
    meta_discovery_across_domains,
)

__all__ = [
    "MetaDiscoveryTransferEngine",
    "DiscoveryPattern",
    "create_meta_discovery_transfer_engine",
    "meta_discovery_across_domains",
]
