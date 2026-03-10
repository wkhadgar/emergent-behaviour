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
    GRID_SIZE,
    INTERACTION_PRESETS,
    N_TYPES,
    ALL_PARTICLES_COUNT,
    TYPE_COLORS,
)


# Current positions, normalised to [0, 1]².
position = ti.Vector.field(2, dtype=ti.f32, shape=ALL_PARTICLES_COUNT)

# Positions from the previous integration step. Velocity is encoded
# implicitly as (position - prev_position), avoiding a separate field
# and providing the symplectic property of Störmer–Verlet integration.
prev_position = ti.Vector.field(2, dtype=ti.f32, shape=ALL_PARTICLES_COUNT)

# Integer type index in [0, N_TYPES). Determines colour and interaction rules.
particle_type = ti.field(dtype=ti.i32, shape=ALL_PARTICLES_COUNT)

# Per-type RGB colour, populated once at startup from config.TYPE_COLORS.
# Kernels index this by particle_type[i] to retrieve colour without
# branching on a large array.
type_color = ti.Vector.field(3, dtype=ti.f32, shape=N_TYPES)

# N_TYPES × N_TYPES interaction matrix (row-major flat storage).
# interaction_matrix[i * N_TYPES + j] is the force coefficient applied to a
# particle of type i encountering a particle of type j.
interaction_matrix = ti.field(dtype=ti.f32, shape=N_TYPES * N_TYPES)


cell_count = ti.field(dtype=ti.i32, shape=GRID_SIZE)
cell_start = ti.field(dtype=ti.i32, shape=GRID_SIZE)
cell_cursor = ti.field(dtype=ti.i32, shape=GRID_SIZE)
sorted_indices = ti.field(dtype=ti.i32, shape=ALL_PARTICLES_COUNT)


def populate_type_colors():
    """Upload TYPE_COLORS to the type_color field. Call once at startup."""
    for t, (r, g, b) in enumerate(TYPE_COLORS):
        type_color[t] = ti.Vector([r, g, b])


def populate_interaction_matrix(preset_name: str):
    """
    Upload a named interaction preset to the interaction_matrix field.

    Args:
        preset_name: Key into config.INTERACTION_PRESETS.
    """
    matrix = INTERACTION_PRESETS[preset_name]
    data = np.array(matrix, dtype=np.float32)
    interaction_matrix.from_numpy(data)
