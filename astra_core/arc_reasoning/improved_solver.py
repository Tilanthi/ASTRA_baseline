#!/usr/bin/env python3
"""
Improved ARC-AGI-2 Solver
Implements comprehensive transformation detection with composition support.
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import copy
import itertools


# ============================================================================
# Core Transformations
# ============================================================================

def rotate_90(grid):
    """Rotate grid 90 degrees clockwise."""
    return [list(row) for row in zip(*grid[::-1])]

def rotate_180(grid):
    """Rotate grid 180 degrees."""
    return [row[::-1] for row in grid[::-1]]

def rotate_270(grid):
    """Rotate grid 270 degrees clockwise."""
    return [list(row)[::-1] for row in zip(*grid)][::-1]

def reflect_h(grid):
    """Reflect horizontally (flip vertical)."""
    return grid[::-1]

def reflect_v(grid):
    """Reflect vertically (flip horizontal)."""
    return [row[::-1] for row in grid]

def transpose(grid):
    """Transpose grid."""
    return [list(row) for row in zip(*grid)]

def crop(grid, top, bottom, left, right):
    """Crop grid by removing borders."""
    return [row[left:len(row)-right] for row in grid[top:len(grid)-bottom]]

def pad(grid, val, top, bottom, left, right):
    """Pad grid with value."""
    h, w = len(grid), len(grid[0]) if grid else 0
    new_h = h + top + bottom
    new_w = w + left + right
    result = [[val] * new_w for _ in range(new_h)]
    for r in range(h):
        for c in range(w):
            result[r + top][c + left] = grid[r][c]
    return result

def subsample(grid, row_step, col_step):
    """Subsample grid by taking every row_step-th row and col_step-th column."""
    result = []
    for i in range(0, len(grid), row_step):
        row = []
        for j in range(0, len(grid[0]), col_step):
            row.append(grid[i][j])
        result.append(row)
    return result


# ============================================================================
# Color Transformation Detection
# ============================================================================

def learn_color_mapping(train_inputs, train_outputs) -> Optional[Dict[int, int]]:
    """Learn color mapping from training pairs."""
    color_map = {}

    for inp, out in zip(train_inputs, train_outputs):
        h = min(len(inp), len(out))
        w = min(len(inp[0]) if inp else 0, len(out[0]) if out else 0)
        for r in range(h):
            for c in range(w):
                inp_val = inp[r][c]
                out_val = out[r][c]
                if inp_val != out_val:
                    if inp_val in color_map:
                        if color_map[inp_val] != out_val:
                            return None
                    else:
                        color_map[inp_val] = out_val
    return color_map if color_map else None

def apply_color_map(grid, color_map):
    """Apply a color mapping to every cell (unmapped colours pass through)."""
    return [[color_map.get(cell, cell) for cell in row] for row in grid]


# ============================================================================
# Solution Hypotheses and Solver
# (re-implemented 2026-08; bodies lost to file truncation before the audit.
#  Exported via arc_reasoning/__init__.py as ImprovedARC_Solver /
#  SolutionHypothesis; primitive functions are imported by
#  ensemble_arc_solver.py and super_ensemble_solver.py.)
# ============================================================================

@dataclass
class SolutionHypothesis:
    """A candidate transformation with its training-pair verification."""
    description: str
    transform: Callable
    confidence: float = 0.0
    n_correct: int = 0
    n_train: int = 0

    def __repr__(self):
        return f"SolutionHypothesis({self.description!r}, conf={self.confidence:.2f})"


class ImprovedARC_Solver:
    """
    Improved ARC-AGI-2 solver: comprehensive transformation detection
    with composition support.

    Candidates are built from the geometric primitives, the learned
    colour map, and compositions of the two (geometry then colour);
    each is verified against every training pair and the best hypothesis
    is applied to the test input.
    """

    def __init__(self):
        self.geometric_transforms = [
            ("identity", lambda g: g),
            ("rotate_90", rotate_90),
            ("rotate_180", rotate_180),
            ("rotate_270", rotate_270),
            ("reflect_h", reflect_h),
            ("reflect_v", reflect_v),
            ("transpose", transpose),
        ]

    def generate_hypotheses(self, train_inputs, train_outputs) -> List[SolutionHypothesis]:
        """Generate and verify transformation hypotheses on the training pairs."""
        candidates: List[Tuple[str, Callable]] = []

        # Pure geometric transforms
        for name, fn in self.geometric_transforms:
            candidates.append((name, fn))

        # Pure colour mapping
        color_map = learn_color_mapping(train_inputs, train_outputs)
        if color_map:
            candidates.append(("color_map", lambda g, cm=color_map: apply_color_map(g, cm)))

            # Compositions: geometry then colour
            for name, fn in self.geometric_transforms:
                candidates.append((
                    f"{name}+color_map",
                    lambda g, fn=fn, cm=color_map: apply_color_map(fn(g), cm),
                ))

        hypotheses = []
        n_train = len(train_inputs)
        for description, transform in candidates:
            correct = 0
            for inp, out in zip(train_inputs, train_outputs):
                try:
                    if transform(inp) == out:
                        correct += 1
                except Exception:
                    continue
            hypotheses.append(SolutionHypothesis(
                description=description,
                transform=transform,
                confidence=correct / n_train if n_train else 0.0,
                n_correct=correct,
                n_train=n_train,
            ))

        # Best first; simpler descriptions win ties (Occam)
        hypotheses.sort(key=lambda h: (-h.confidence, len(h.description), h.description))
        return hypotheses

    def solve(self, train_inputs, train_outputs, test_input):
        """
        Solve one task: return the transformed test input, or None if no
        hypothesis explains the training pairs.
        """
        hypotheses = self.generate_hypotheses(train_inputs, train_outputs)
        if not hypotheses or hypotheses[0].confidence == 0.0:
            return None
        best = hypotheses[0]
        try:
            return best.transform(test_input)
        except Exception:
            return None
