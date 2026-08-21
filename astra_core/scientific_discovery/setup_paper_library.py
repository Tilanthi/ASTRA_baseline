#!/usr/bin/env python3
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
Setup Script for Astronomical Paper Library
===========================================

Quick setup to initialize your paper library and add first papers.

Usage:
    python setup_paper_library.py --help

Author: STAN_IX_ASTRO
Date: January 10, 2026
"""


def _degraded_warn(_module: str, _exc: BaseException) -> None:
    """Log why an optional import degraded instead of failing silently."""
    import logging
    logging.getLogger(__name__).warning(
        "%s unavailable (%s: %s) - dependent names set to None",
        _module, type(_exc).__name__, _exc,
    )


try:
    import os
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("<unknown>", _exc)
    os = None  # degraded: unavailable
try:
    import sys
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("<unknown>", _exc)
    sys = None  # degraded: unavailable
try:
    import argparse
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("<unknown>", _exc)
    argparse = None  # degraded: unavailable
try:
    import logging
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("<unknown>", _exc)
    logging = None  # degraded: unavailable
try:
    from pathlib import Path
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("pathlib", _exc)
    Path = None  # degraded: unavailable

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scientific_discovery.paper_library import PaperLibrary
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("scientific_discovery.paper_library", _exc)
    PaperLibrary = None  # degraded: unavailable
try:
    from astra_core.scientific_discovery.paper_rag_query import PaperRAGSystem
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("astra_core.scientific_discovery.paper_rag_query", _exc)
    PaperRAGSystem = None  # degraded: unavailable

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_example_config():
    """Create example configuration file."""
