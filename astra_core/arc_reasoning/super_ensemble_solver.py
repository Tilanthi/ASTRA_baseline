#!/usr/bin/env python3
"""
Super Ensemble ARC Solver - Combines ALL approaches with intelligent voting.
This is the ultimate solver integrating every technique we've built.
"""

try:
    import numpy as np
except Exception:
    np = None  # degraded: unavailable
try:
    from typing import List, Tuple, Dict, Set, Optional, Any
except Exception:
    List = Tuple = Dict = Set = Optional = Any = None  # degraded: unavailable
try:
    from dataclasses import dataclass
except Exception:
    dataclass = None  # degraded: unavailable
try:
    from collections import defaultdict, Counter
except Exception:
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
except Exception:
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
