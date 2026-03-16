# Copyright (c) 2026
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Spatial hash grid — construction and neighbourhood iteration.

The grid partitions the [0, 1]² domain into GRID_W × GRID_H uniform cells,
each of side length INTERACTION_RADIUS. A particle at position p belongs to
cell (floor(p.x * GRID_W), floor(p.y * GRID_H)).

Construction is a three-pass counting sort executed every frame:

    Pass 1 — count:  count particles per cell            (parallel kernel)
    Pass 2 — prefix: exclusive prefix sum over counts    (CPU, sequential)
    Pass 3 — fill:   place indices into sorted_indices   (parallel kernel)

The prefix sum is intentionally computed on the CPU as a plain Python
function. Taichi parallelises the outermost loop of every @ti.kernel
unconditionally; a prefix sum requires strictly sequential execution because
each entry depends on its predecessor. The cell count is O(GRID_SIZE) ≈ 1600
entries, making the CPU cost negligible relative to the parallel passes.

Reference: Ihmsen et al. (2011), "A Parallel SPH Implementation on
Multi-Core CPUs", Computer Graphics Forum 30(1).
"""

import numpy as np
import taichi as ti

import config
import fields


@ti.func
def position_to_cell(pos: ti.template()) -> ti.Vector:
    """
    Map a normalised position to integer (cx, cy) grid coordinates.

    The y axis is scaled by ASPECT_RATIO so that grid cells remain square
    in screen space despite the domain being mapped to a non-square window.
    """
    cx = int(pos.x * config.GRID_W)
    cy = int(pos.y * config.GRID_H)

    # Clamp to valid range. Particles exactly at 1.0 / ASPECT_RATIO on y
    # would otherwise address one cell past the last row.
    cx = ti.math.clamp(cx, 0, config.GRID_W - 1)
    cy = ti.math.clamp(cy, 0, config.GRID_H - 1)

    return ti.Vector([cx, cy])


@ti.func
def cell_to_index(cx: int, cy: int) -> int:
    """Flatten 2D cell coordinates to a 1D index."""
    return cx + cy * config.GRID_W


@ti.kernel
def _count_particles_per_cell():
    """Pass 1: reset counts and tally particles per cell."""
    for c in fields.cell_count:
        fields.cell_count[c] = 0

    for i in fields.position:
        cell = position_to_cell(fields.position[i])
        c = cell_to_index(cell.x, cell.y)
        ti.atomic_add(fields.cell_count[c], 1)


def _compute_prefix_sum():
    """
    Pass 2: exclusive prefix sum over cell_count → cell_start.

    Plain Python by design — see module docstring for the rationale.
    cell_cursor is initialised to cell_start here for use in the fill pass.
    """
    counts = fields.cell_count.to_numpy()
    starts = np.empty_like(counts)
    starts[0] = 0
    for c in range(1, config.GRID_SIZE):
        starts[c] = starts[c - 1] + counts[c - 1]
    fields.cell_start.from_numpy(starts)
    fields.cell_cursor.from_numpy(starts.copy())


@ti.kernel
def _fill_sorted_indices():
    """
    Pass 3: place particle indices into sorted_indices.

    The atomic increment on cell_cursor assigns each particle a unique slot
    within its cell's contiguous range in sorted_indices.
    """
    for i in fields.position:
        cell = position_to_cell(fields.position[i])
        c = cell_to_index(cell.x, cell.y)

        slot = ti.atomic_add(fields.cell_cursor[c], 1)

        fields.sorted_indices[slot] = i


def rebuild():
    """
    Rebuild the spatial hash from current particle positions.

    Must be called once per frame before any behaviour kernel that performs
    neighbourhood queries.
    """
    _count_particles_per_cell()
    _compute_prefix_sum()
    _fill_sorted_indices()


@ti.func
def iterate_neighbours(i: int, func: ti.template()):
    """
    Invoke func(i, j) for every particle j within INTERACTION_RADIUS of
    particle i, excluding i itself.

    The 3×3 cell neighbourhood is a conservative superset of the true radius
    neighbourhood. A distance check inside func is therefore required to
    exclude false positives from adjacent cells.

    Taichi does not support closures that capture mutable local variables
    from the calling kernel scope. Callers must accumulate results into
    module-level fields rather than local variables. See behaviors.py for
    the canonical accumulation pattern.
    """
    pos_i = fields.position[i]
    cell = position_to_cell(pos_i)

    for dx in ti.static(range(-1, 2)):
        for dy in ti.static(range(-1, 2)):
            nx = cell.x + dx
            ny = cell.y + dy

            if 0 <= nx < config.GRID_W and 0 <= ny < config.GRID_H:
                c = cell_to_index(nx, ny)
                start = fields.cell_start[c]
                count = fields.cell_count[c]

                for k in range(start, start + count):
                    j = fields.sorted_indices[k]

                    if j != i:
                        func(i, j)
