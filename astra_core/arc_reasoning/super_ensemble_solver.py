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
Super Ensemble ARC Solver - Combines ALL approaches with intelligent voting.
This is the ultimate solver integrating every technique we've built.
"""


def _degraded_warn(_module: str, _exc: BaseException) -> None:
    """Log why an optional import degraded instead of failing silently."""
    import logging
    logging.getLogger(__name__).warning(
        "%s unavailable (%s: %s) - dependent names set to None",
        _module, type(_exc).__name__, _exc,
    )


try:
    import numpy as np
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("<unknown>", _exc)
    np = None  # degraded: unavailable
try:
    from typing import List, Tuple, Dict, Set, Optional, Any
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("typing", _exc)
    List = Tuple = Dict = Set = Optional = Any = None  # degraded: unavailable
try:
    from dataclasses import dataclass
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("dataclasses", _exc)
    dataclass = None  # degraded: unavailable
try:
    from collections import defaultdict, Counter
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn("collections", _exc)
    defaultdict = Counter = None  # degraded: unavailable

# Import geometric operations
try:
    from .improved_solver import (
    rotate_90, rotate_180, rotate_270,
    reflect_h, reflect_v, transpose,
    learn_color_mapping,
    apply_color_map,
    subsample, crop, pad,
    )
except Exception as _exc:  # pragma: no cover - optional surface
    _degraded_warn(".improved_solver", _exc)
    rotate_90 = rotate_180 = rotate_270 = reflect_h = reflect_v = transpose = learn_color_mapping = apply_color_map = subsample = crop = pad = None  # degraded: unavailable


def rotate_90(grid): return [list(row) for row in zip(*grid[::-1])]
def rotate_180(grid): return [row[::-1] for row in grid[::-1]]
def rotate_270(grid): return [list(row)[::-1] for row in zip(*grid)][::-1]
def reflect_h(grid): return grid[::-1]
def reflect_v(grid): return [row[::-1] for row in grid]
def transpose(grid): return [list(row) for row in zip(*grid)]

def learn_color_mapping(train_inputs, train_outputs):
    color_map = {}
    for inp, out in zip(train_inputs, train_outputs):
        h = min(len(inp), len(out))
        w = min(len(inp[0]), len(out[0]))
