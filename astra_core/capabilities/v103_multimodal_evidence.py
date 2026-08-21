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
V103 Multi-Modal Evidence Fusion -- compatibility module.

The V5.0 discovery capabilities were originally published as flat modules at
``astra_core.capabilities.v103_multimodal_evidence``. The subpackage reorganisation moved the
implementation to ``astra_core.capabilities.multimodal.multimodal_evidence`` but left nothing behind
at the old path, so every script written against the published API -- including
this repository's own ``tests/test_discovery/test_v5_capabilities.py`` -- broke
with ``ModuleNotFoundError``.

On a developer machine that also held older ASTRA checkouts the import could
still succeed by picking the module up from a *different tree*, which made the
breakage invisible locally while a clean clone from GitHub failed. This module
re-exports the canonical implementation so that a standalone install behaves
the same way.

There is no separate implementation here: these are the same objects.

    >>> from astra_core.capabilities.v103_multimodal_evidence import MultiModalEvidenceFusion
    >>> from astra_core.capabilities.multimodal.multimodal_evidence import MultiModalEvidenceFusion as canonical
    >>> MultiModalEvidenceFusion is canonical
    True

New code should import from ``astra_core.capabilities.multimodal.multimodal_evidence``.
"""

from __future__ import annotations

from .multimodal.multimodal_evidence import (  # noqa: F401
    MultiModalEvidenceFusion,
    EvidenceType,
    create_multimodal_evidence_fusion,
)

__all__ = [
    "MultiModalEvidenceFusion",
    "EvidenceType",
    "create_multimodal_evidence_fusion",
]
