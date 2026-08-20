#!/usr/bin/env python3
"""
Hypothesis Generation Engine for ARC-AGI

Generates and tests transformation hypotheses from training examples.
Uses a program synthesis approach with compositional primitives.
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import itertools

from .grid_dsl import Grid, GridObject, BoundingBox, empty_grid


class TransformationType(Enum):
    """Categories of transformations"""
    IDENTITY = auto()
    COLOR = auto()
    GEOMETRIC = auto()
    SCALING = auto()
    OBJECT_MANIPULATION = auto()
    PATTERN = auto()
    COMPOSITIONAL = auto()
    CONDITIONAL = auto()


@dataclass
class TransformationHypothesis:
    """A hypothesis about the transformation rule"""
    name: str
    transform_type: TransformationType
    apply_fn: Callable[[Grid], Grid]
    params: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    description: str = ""

    def apply(self, grid: Grid) -> Grid:
        return self.apply_fn(grid)

    def __repr__(self):
        return f"Hypothesis({self.name}, conf={self.confidence:.2f})"


class HypothesisGenerator:
    """
    Generates transformation hypotheses from input-output pairs.
    """

    def __init__(self):
        self.generators = [
            self._gen_identity,
            self._gen_color_mappings,
            self._gen_single_color_replace,
            self._gen_geometric_transforms,
            self._gen_scaling_transforms,
            self._gen_tiling_transforms,
            self._gen_crop_transforms,
            self._gen_object_operations,
            self._gen_fill_operations,
            self._gen_overlay_operations,
            self._gen_pattern_completion,
            self._gen_conditional_transforms,
        ]

    def generate(self, train_pairs: List[Tuple[Grid, Grid]]) -> List[TransformationHypothesis]:
        """Generate all possible hypotheses from training pairs"""
        all_hypotheses = []

        for generator in self.generators:
            try:
                hypotheses = generator(train_pairs)
                all_hypotheses.extend(hypotheses)
            except Exception:
                continue

        # Score and rank hypotheses
        scored = self._score_hypotheses(all_hypotheses, train_pairs)
        return sorted(scored, key=lambda h: -h.confidence)

    def _score_hypotheses(self, hypotheses: List[TransformationHypothesis],
                         train_pairs: List[Tuple[Grid, Grid]]) -> List[TransformationHypothesis]:
        """Score hypotheses by how many training pairs they correctly predict"""
        for h in hypotheses:
            correct = 0
            for inp, out in train_pairs:
                try:
                    predicted = h.apply(inp)
                    if predicted == out:
                        correct += 1
                except Exception:
                    pass
            h.confidence = correct / len(train_pairs) if train_pairs else 0.0
        return hypotheses

    # ========== Identity ==========

    def _gen_identity(self, train_pairs: List[Tuple[Grid, Grid]]) -> List[TransformationHypothesis]:
        """Check if transformation is identity"""
        return [TransformationHypothesis(
            name="identity",
            transform_type=TransformationType.IDENTITY,
            apply_fn=lambda g: g.copy(),
            description="No transformation"
        )]

    # ========== Color Operations ==========

    def _gen_color_mappings(self, train_pairs: List[Tuple[Grid, Grid]]) -> List[TransformationHypothesis]:
        """Generate color mapping hypotheses"""
        hypotheses = []

        # Learn color mapping from all pairs
        color_map = {}
        consistent = True

        for inp, out in train_pairs:
            if inp.shape != out.shape:
                consistent = False
                break

            for r in range(inp.height):
                for c in range(inp.width):
                    in_color = inp[r, c]
                    out_color = out[r, c]
                    if in_color in color_map:
                        if color_map[in_color] != out_color:
                            consistent = False
                            break
                    else:
                        color_map[in_color] = out_color

        if consistent and color_map:
            def apply_map(g: Grid, cm=color_map) -> Grid:
                return g.apply_color_map(cm)

            hypotheses.append(TransformationHypothesis(
                name="color_map",
                transform_type=TransformationType.COLOR,
                apply_fn=apply_map,
                params={'color_map': color_map},
                description=f"Color mapping: {color_map}"
            ))

        return hypotheses

    def _gen_single_color_replace(self, train_pairs: List[Tuple[Grid, Grid]]) -> List[TransformationHypothesis]:
        """Generate single color replacement hypotheses"""
        hypotheses = []

        # Find consistent single color changes
        changes = set()
        for inp, out in train_pairs:
            if inp.shape != out.shape:
                continue

            for r in range(inp.height):
                for c in range(inp.width):
                    if inp[r, c] != out[r, c]:
                        changes.add((inp[r, c], out[r, c]))

        for old_color, new_color in changes:
            def apply_replace(g: Grid, oc=old_color, nc=new_color) -> Grid:
                return g.replace_color(oc, nc)

            hypotheses.append(TransformationHypothesis(
                name=f"replace_{old_color}_with_{new_color}",
                transform_type=TransformationType.COLOR,
                apply_fn=apply_replace,
                params={'old_color': old_color, 'new_color': new_color},
                description=f"Replace color {old_color} with {new_color}"
            ))

        return hypotheses

    # ========== Geometric Transforms ==========

    def _gen_geometric_transforms(self, train_pairs: List[Tuple[Grid, Grid]]) -> List[TransformationHypothesis]:
        """Generate geometric transformation hypotheses"""
        hypotheses = []

        transforms = [
            ("rotate_90", lambda g: g.rotate_90()),
            ("rotate_180", lambda g: g.rotate_180()),
            ("rotate_270", lambda g: g.rotate_270()),
            ("flip_horizontal", lambda g: g.flip_horizontal()),
            ("flip_vertical", lambda g: g.flip_vertical()),
            ("transpose", lambda g: g.transpose()),
        ]

        for name, fn in transforms:
            hypotheses.append(TransformationHypothesis(
                name=name,
                transform_type=TransformationType.GEOMETRIC,
                apply_fn=fn,
                description=f"Geometric: {name}"
            ))

        return hypotheses

    # ========== Scaling ==========

    def _gen_scaling_transforms(self, train_pairs: List[Tuple[Grid, Grid]]) -> List[TransformationHypothesis]:
        """Generate scaling transformation hypotheses"""
        hypotheses = []

        for inp, out in train_pairs:
            if inp.height > 0 and inp.width > 0:
                if out.height % inp.height == 0 and out.width % inp.width == 0:
                    scale_h = out.height // inp.height
                    scale_w = out.width // inp.width

                    def apply_scale(g: Grid, sh=scale_h, sw=scale_w) -> Grid:
                        return g.scale(sh, sw)

                    hypotheses.append(TransformationHypothesis(
                        name=f"scale_{scale_h}x{scale_w}",
                        transform_type=TransformationType.SCALING,
                        apply_fn=apply_scale,
                        params={'scale_h': scale_h, 'scale_w': scale_w},
                        description=f"Scale by {scale_h}x{scale_w}"
                    ))
            break  # Only need one pair to detect scaling

        return hypotheses

    # ========== Tiling ==========

    def _gen_tiling_transforms(self, train_pairs: List[Tuple[Grid, Grid]]) -> List[TransformationHypothesis]:
        """Generate tiling transformation hypotheses"""
        hypotheses = []

        for inp, out in train_pairs:
            if inp.height > 0 and inp.width > 0:
                if out.height % inp.height == 0 and out.width % inp.width == 0:
                    tile_h = out.height // inp.height
                    tile_w = out.width // inp.width

                    # Simple tiling
                    def apply_tile(g: Grid, th=tile_h, tw=tile_w) -> Grid:
                        return g.tile(th, tw)

                    hypotheses.append(TransformationHypothesis(
                        name=f"tile_{tile_h}x{tile_w}",
                        transform_type=TransformationType.SCALING,
                        apply_fn=apply_tile,
                        params={'tile_h': tile_h, 'tile_w': tile_w},
                        description=f"Tile {tile_h}x{tile_w}"
                    ))

                    # Tiling with alternation (for task 00576224)
                    def apply_tile_alternating(g: Grid, th=tile_h, tw=tile_w) -> Grid:
                        result = empty_grid(g.height * th, g.width * tw)
                        for tr in range(th):
                            for tc in range(tw):
                                tile_grid = g if (tr + tc) % 2 == 0 else g.transpose()
                                for r in range(g.height):
                                    for c in range(g.width):
                                        result[tr * g.height + r, tc * g.width + c] = tile_grid[r, c]
                        return result

                    hypotheses.append(TransformationHypothesis(
                        name=f"tile_alternating_{tile_h}x{tile_w}",
                        transform_type=TransformationType.SCALING,
                        apply_fn=apply_tile_alternating,
                        params={'tile_h': tile_h, 'tile_w': tile_w},
                        description=f"Tile {tile_h}x{tile_w} with alternating transpose"
                    ))

                    # Tiling with row alternation (even tile-rows upright,
                    # odd tile-rows vertically flipped)
                    def apply_tile_row_alt(g: Grid, th=tile_h, tw=tile_w) -> Grid:
                        result = empty_grid(g.height * th, g.width * tw)
                        for tr in range(th):
                            for tc in range(tw):
                                tile_grid = g if tr % 2 == 0 else g.flip_vertical()
                                for r in range(g.height):
                                    for c in range(g.width):
                                        result[tr * g.height + r,
                                               tc * g.width + c] = tile_grid[r, c]
                        return result

                    hypotheses.append(TransformationHypothesis(
                        name=f"tile_row_alt_{tile_h}x{tile_w}",
                        transform_type=TransformationType.SCALING,
                        apply_fn=apply_tile_row_alt,
                        params={'tile_h': tile_h, 'tile_w': tile_w},
                        description=f"Tile {tile_h}x{tile_w} alternating rows vertically"
                    ))

                    # Mirror tiling: each column-tile mirrors its neighbour
                    def apply_tile_mirror(g: Grid, th=tile_h, tw=tile_w) -> Grid:
                        result = empty_grid(g.height * th, g.width * tw)
                        for tr in range(th):
                            for tc in range(tw):
                                tile_grid = g if tc % 2 == 0 else g.flip_horizontal()
                                for r in range(g.height):
                                    for c in range(g.width):
                                        result[tr * g.height + r,
                                               tc * g.width + c] = tile_grid[r, c]
                        return result

                    hypotheses.append(TransformationHypothesis(
                        name=f"tile_mirror_{tile_h}x{tile_w}",
                        transform_type=TransformationType.SCALING,
                        apply_fn=apply_tile_mirror,
                        params={'tile_h': tile_h, 'tile_w': tile_w},
                        description=f"Tile {tile_h}x{tile_w} mirroring horizontally"
                    ))
            break  # Only need one pair to detect tiling

        return hypotheses

    # ========== Cropping and Padding ==========
    # (re-implemented 2026-08; bodies lost to file truncation before the audit)

    def _gen_crop_transforms(self, train_pairs: List[Tuple[Grid, Grid]]) -> List[TransformationHypothesis]:
        """Generate crop / pad hypotheses"""
        hypotheses = []

        # Crop to bounding box of non-background content
        if train_pairs and all(
                out.height <= inp.height and out.width <= inp.width
                for inp, out in train_pairs):
            hypotheses.append(TransformationHypothesis(
                name="crop_to_content",
                transform_type=TransformationType.GEOMETRIC,
                apply_fn=lambda g: g.crop_to_content(),
                description="Crop to bounding box of content"
            ))

        # Pad to the first output's size, content anchored at a corner
        if train_pairs:
            inp, out = train_pairs[0]
            if (out.height >= inp.height and out.width >= inp.width
                    and (out.height > inp.height or out.width > inp.width)):
                for corner in ('top_left', 'top_right', 'bottom_left', 'bottom_right'):
                    def apply_pad(g: Grid, oh=out.height, ow=out.width, corner=corner) -> Grid:
                        top = oh - g.height if corner.startswith('bottom') else 0
                        left = ow - g.width if corner.endswith('right') else 0
                        return g.pad(top, oh - g.height - top,
                                     left, ow - g.width - left)

                    hypotheses.append(TransformationHypothesis(
                        name=f"pad_{corner}_{out.height}x{out.width}",
                        transform_type=TransformationType.GEOMETRIC,
                        apply_fn=apply_pad,
                        params={'out_height': out.height, 'out_width': out.width,
                                'corner': corner},
                        description=f"Pad to {out.height}x{out.width} anchored {corner}"
                    ))

        return hypotheses

    # ========== Object Operations ==========

    def _gen_object_operations(self, train_pairs: List[Tuple[Grid, Grid]]) -> List[TransformationHypothesis]:
        """Generate object-translation hypotheses (shift the content block)"""
        hypotheses = []

        # Learn a consistent translation of the non-background content
        deltas = []
        shapes_ok = True
        for inp, out in train_pairs:
            if inp.shape != out.shape:
                shapes_ok = False
                break
            bi, bo = _content_bbox(inp), _content_bbox(out)
            if bi is None or bo is None:
                shapes_ok = False
                break
            deltas.append((bo[0] - bi[0], bo[1] - bi[1]))
        if shapes_ok and deltas and len(set(deltas)) == 1 and deltas[0] != (0, 0):
            dr, dc = deltas[0]

            def apply_translate(g: Grid, dr=dr, dc=dc) -> Grid:
                result = empty_grid(g.height, g.width)
                for r, c in g.where(lambda v: v != 0):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < g.height and 0 <= cc < g.width:
                        result[rr, cc] = g[r, c]
                return result

            hypotheses.append(TransformationHypothesis(
                name=f"translate_{dr}_{dc}",
                transform_type=TransformationType.OBJECT_MANIPULATION,
                apply_fn=apply_translate,
                params={'dr': dr, 'dc': dc},
                description=f"Translate content by (dr={dr}, dc={dc})"
            ))

        return hypotheses

    # ========== Fill Operations ==========

    def _gen_fill_operations(self, train_pairs: List[Tuple[Grid, Grid]]) -> List[TransformationHypothesis]:
        """Generate fill hypotheses (uniform fill, bounding-box fill)"""
        hypotheses = []

        # Uniform fill: every output grid is a single flat colour
        flat_colors = {int(out.data.flat[0]) for _, out in train_pairs
                       if out.data.size and np.all(out.data == out.data.flat[0])}
        if flat_colors and len(flat_colors) == 1 and len(train_pairs) > 0:
            color = flat_colors.pop()

            hypotheses.append(TransformationHypothesis(
                name=f"fill_all_{color}",
                transform_type=TransformationType.COLOR,
                apply_fn=lambda g, c=color: empty_grid(g.height, g.width, c),
                params={'color': color},
                description=f"Fill entire grid with {color}"
            ))

        # Fill the content bounding box with each colour seen beside content
        fill_colors = set()
        for inp, out in train_pairs:
            if inp.shape != out.shape:
                continue
            bi = _content_bbox(inp)
            if bi is None:
                continue
            region = out.data[bi[0]:bi[2] + 1, bi[1]:bi[3] + 1]
            if region.size and np.all(region != inp.data[bi[0]:bi[2] + 1, bi[1]:bi[3] + 1]):
                if np.all(region == region.flat[0]):
                    fill_colors.add(int(region.flat[0]))
        for color in fill_colors:

            def apply_fill_bbox(g: Grid, c=color) -> Grid:
                result = g.copy()
                bbox = _content_bbox(g)
                if bbox is not None:
                    result.data[bbox[0]:bbox[2] + 1, bbox[1]:bbox[3] + 1] = c
                return result

            hypotheses.append(TransformationHypothesis(
                name=f"fill_bbox_{color}",
                transform_type=TransformationType.COLOR,
                apply_fn=apply_fill_bbox,
                params={'color': color},
                description=f"Fill content bounding box with {color}"
            ))

        return hypotheses

    # ========== Overlay Operations ==========

    def _gen_overlay_operations(self, train_pairs: List[Tuple[Grid, Grid]]) -> List[TransformationHypothesis]:
        """Generate symmetry-completion hypotheses (overlay mirrored copy)"""
        hypotheses = []

        overlays = [
            ("symmetrize_horizontal",
             lambda g: g.union(g.flip_horizontal(), priority='other')),
            ("symmetrize_vertical",
             lambda g: g.union(g.flip_vertical(), priority='other')),
            ("symmetrize_both",
             lambda g: g.union(g.flip_horizontal(), priority='other')
                       .union(g.flip_vertical(), priority='other')),
        ]
        for name, fn in overlays:
            hypotheses.append(TransformationHypothesis(
                name=name,
                transform_type=TransformationType.COMPOSITIONAL,
                apply_fn=fn,
                description=f"Complete symmetry: {name}"
            ))

        return hypotheses

    # ========== Pattern Completion ==========

    def _gen_pattern_completion(self, train_pairs: List[Tuple[Grid, Grid]]) -> List[TransformationHypothesis]:
        """Generate pattern-extension hypotheses (mirror doubling, 2x2 mosaic)"""
        hypotheses = []

        # Mirror doubling: append a mirrored copy along one axis
        def apply_mirror_h(g: Grid) -> Grid:
            return Grid(np.concatenate([g.data, np.fliplr(g.data)], axis=1))

        def apply_mirror_v(g: Grid) -> Grid:
            return Grid(np.concatenate([g.data, np.flipud(g.data)], axis=0))

        hypotheses.append(TransformationHypothesis(
            name="mirror_double_horizontal",
            transform_type=TransformationType.PATTERN,
            apply_fn=apply_mirror_h,
            description="Append left-right mirrored copy"
        ))
        hypotheses.append(TransformationHypothesis(
            name="mirror_double_vertical",
            transform_type=TransformationType.PATTERN,
            apply_fn=apply_mirror_v,
            description="Append top-bottom mirrored copy"
        ))

        # 2x2 mirror mosaic: [g, g.fliplr; g.flipud, g.rot180]
        def apply_mosaic_2x2(g: Grid) -> Grid:
            top = np.concatenate([g.data, np.fliplr(g.data)], axis=1)
            bottom = np.concatenate([np.flipud(g.data), np.rot90(g.data, 2)], axis=1)
            return Grid(np.concatenate([top, bottom], axis=0))

        hypotheses.append(TransformationHypothesis(
            name="mirror_mosaic_2x2",
            transform_type=TransformationType.PATTERN,
            apply_fn=apply_mosaic_2x2,
            description="2x2 mirror mosaic"
        ))

        return hypotheses

    # ========== Conditional Transforms ==========

    def _gen_conditional_transforms(self, train_pairs: List[Tuple[Grid, Grid]]) -> List[TransformationHypothesis]:
        """Generate colour rewrites conditional on a trigger colour being present"""
        hypotheses = []

        # Collect (a -> b) changes and the colours present in each input
        change_pairs: Dict[Tuple[int, int], List[Set[int]]] = defaultdict(list)
        for inp, out in train_pairs:
            if inp.shape != out.shape:
                return hypotheses
            colors_in = inp.get_colors()
            for a in colors_in:
                out_colors = set(np.unique(out.data[inp.data == a]))
                for b in out_colors - {int(a)}:
                    change_pairs[(int(a), int(b))].append(colors_in)

        # Trigger colours: present in every input that exhibits the change
        for (a, b), color_sets in change_pairs.items():
            triggers = set.intersection(*color_sets) if color_sets else set()
            for k in sorted(triggers):
                if k == a:
                    continue

                def apply_conditional(g: Grid, a=a, b=b, k=k) -> Grid:
                    result = g.copy()
                    if k in g.get_colors():
                        result = result.replace_color(a, b)
                    return result

                hypotheses.append(TransformationHypothesis(
                    name=f"replace_{a}_with_{b}_if_{k}",
                    transform_type=TransformationType.CONDITIONAL,
                    apply_fn=apply_conditional,
                    params={'old_color': a, 'new_color': b, 'trigger': k},
                    description=f"Replace {a} with {b} when {k} present"
                ))

        return hypotheses


def _content_bbox(grid: Grid, background: int = 0) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box (r_min, c_min, r_max, c_max) of non-background pixels."""
    rows, cols = np.where(grid.data != background)
    if len(rows) == 0:
        return None
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _param_complexity(hypothesis: TransformationHypothesis) -> int:
    """Cheap Occam prior: number of leaf values in the hypothesis params."""
    def _count(v: Any) -> int:
        if isinstance(v, dict):
            return sum(_count(x) for x in v.values())
        if isinstance(v, (list, tuple, set)):
            return sum(_count(x) for x in v)
        return 1
    return sum(_count(v) for v in hypothesis.params.values())


# =============================================================================
# HYPOTHESIS TESTER
# (re-implemented 2026-08; body lost to file truncation before the audit.
#  API reconstructed from surviving call sites: constructed with no arguments
#  by enhanced_solver.ARCEnhancedSolver and reasoning.arc_agi_integration.)
# =============================================================================

class HypothesisTester:
    """
    Tests transformation hypotheses against example input/output pairs.

    Complements :class:`HypothesisGenerator`: the generator proposes and
    roughly ranks candidate transformations; the tester evaluates them in
    detail (exact accuracy, pixel accuracy, per-pair diagnostics) and
    selects the best-scoring hypothesis.
    """

    def __init__(self, generator: Optional[HypothesisGenerator] = None):
        self.generator = generator or HypothesisGenerator()
        self.last_reports: Dict[str, Dict[str, Any]] = {}

    def test_hypothesis(self, hypothesis: TransformationHypothesis,
                        train_pairs: List[Tuple[Grid, Grid]]) -> Dict[str, Any]:
        """
        Evaluate one hypothesis against the training pairs.

        Returns a report with exact accuracy, pixel accuracy and per-pair
        pass/fail diagnostics.
        """
        n_pairs = len(train_pairs)
        exact = 0
        pixel_sum = 0.0
        per_pair: List[bool] = []
        for inp, out in train_pairs:
            try:
                predicted = hypothesis.apply(inp)
            except Exception:
                per_pair.append(False)
                continue
            if predicted == out:
                exact += 1
                per_pair.append(True)
                pixel_sum += 1.0
            elif predicted.shape == out.shape:
                frac = float(np.mean(predicted.data == out.data))
                pixel_sum += frac
                per_pair.append(False)
            else:
                per_pair.append(False)

        report = {
            'name': hypothesis.name,
            'n_pairs': n_pairs,
            'exact_accuracy': exact / n_pairs if n_pairs else 0.0,
            'pixel_accuracy': pixel_sum / n_pairs if n_pairs else 0.0,
            'per_pair': per_pair,
        }
        self.last_reports[hypothesis.name] = report
        return report

    def test_all(self, hypotheses: List[TransformationHypothesis],
                 train_pairs: List[Tuple[Grid, Grid]]) -> List[TransformationHypothesis]:
        """Score, sort and return hypotheses (best first) by test performance.

        Ties on exact and pixel accuracy are broken by an Occam prior:
        the hypothesis with fewer parameters wins.  (An arbitrary colour
        map and a rotation can both fit a single training pair exactly;
        the simpler rule is the one that generalises.)
        """
        scored = []
        for h in hypotheses:
            report = self.test_hypothesis(h, train_pairs)
            key = (report['exact_accuracy'], report['pixel_accuracy'],
                   -_param_complexity(h))
            scored.append((key, h))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        result = [h for _, h in scored]
        # Reflect test performance in the hypothesis confidence
        for h in result:
            h.confidence = self.last_reports[h.name]['exact_accuracy']
        return result

    def find_best(self, train_pairs: List[Tuple[Grid, Grid]],
                  min_confidence: float = 1.0) -> Optional[TransformationHypothesis]:
        """
        Generate hypotheses from the pairs and return the best one that
        reaches ``min_confidence`` exact accuracy (None if none does).
        """
        hypotheses = self.generator.generate(train_pairs)
        ranked = self.test_all(hypotheses, train_pairs)
        for h in ranked:
            if h.confidence >= min_confidence:
                return h
        return None
