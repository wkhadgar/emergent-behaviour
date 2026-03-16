# Copyright (c) 2026
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Global Taichi field declarations.

All fields are module-level so that kernels across behaviors.py, grid.py,
and renderer.py can import them directly. Fields must be declared after
ti.init() is called — main.py is responsible for calling ti.init() before
importing any project module.

No logic lives here — only allocation and one-time population helpers.
"""

import numpy as np
import taichi as ti

from config import (
    ALL_PARTICLES_COUNT,
    GRID_SIZE,
    INTERACTION_PRESETS,
    N_TYPES,
    TYPE_COLORS,
)

position = ti.Vector.field(2, dtype=ti.f32, shape=ALL_PARTICLES_COUNT)
prev_position = ti.Vector.field(2, dtype=ti.f32, shape=ALL_PARTICLES_COUNT)
particle_type = ti.field(dtype=ti.i32, shape=ALL_PARTICLES_COUNT)

type_color = ti.Vector.field(3, dtype=ti.f32, shape=N_TYPES)
interaction_matrix = ti.field(dtype=ti.f32, shape=N_TYPES * N_TYPES)

cell_count = ti.field(dtype=ti.i32, shape=GRID_SIZE)
cell_start = ti.field(dtype=ti.i32, shape=GRID_SIZE)
cell_cursor = ti.field(dtype=ti.i32, shape=GRID_SIZE)
sorted_indices = ti.field(dtype=ti.i32, shape=ALL_PARTICLES_COUNT)


def populate_type_colors():
    """Upload TYPE_COLORS to the type_color field. Call once at startup."""
    for t, (r, g, b) in enumerate(TYPE_COLORS):
        type_color[t] = ti.Vector([r, g, b])


def populate_interaction_matrix(source):
    """
    Upload an interaction matrix to the GPU field.

    Args:
        source: either a preset name (str) from config.INTERACTION_PRESETS,
                or a flat list/array of N_TYPES² floats in row-major order.
    """
    if isinstance(source, str):
        matrix = INTERACTION_PRESETS[source]
    else:
        matrix = source

    interaction_matrix.from_numpy(np.array(matrix, dtype=np.float32))
