# Copyright (c) 2026
#
# SPDX-License-Identifier: BSD-3-Clause

import tomllib as toml
import numpy as np

"""
Simulation-wide constants and interaction presets.

All tunable parameters live here. No magic numbers appear elsewhere in the
codebase. When reporting experimental results, cite these values directly.

Interaction matrices are defined as N_TYPES × N_TYPES flat lists (row-major).
Entry [i * N_TYPES + j] is the force coefficient applied to a particle of
type i when it encounters a particle of type j within INTERACTION_RADIUS.

    Positive coefficient → repulsion  (particles push away)
    Negative coefficient → attraction (particles pull toward)
    Zero                 → no interaction
"""

# Timestep for Störmer–Verlet integration. Reducing this improves stability
# at the cost of requiring more sub-steps to maintain simulation speed.
DT: float = 5e-4

# Sub-steps executed per rendered frame. More sub-steps improve stability at
# high velocities but increase GPU time proportionally.
SUBSTEPS: int = 6

# Viscous drag coefficient applied each integration step.
# 1.0 = energy conserved, values below 1.0 dissipate kinetic energy.
DAMPING: float = 0.985

# Coefficient of restitution on boundary collision.
BOUNCE_RESTITUTION: float = 0.5

# Neighbourhood radius. Grid cell size equals this value, ensuring all
# neighbour candidates reside within the 3×3 cell neighbourhood.
INTERACTION_RADIUS: float = 0.025 * 3

GRID_W: int = int(1.0 / INTERACTION_RADIUS)
GRID_H: int = int(1.0 / INTERACTION_RADIUS)
GRID_SIZE: int = GRID_W * GRID_H

# Hard cap on particles per cell. Particles beyond this are excluded from
# neighbourhood queries for that cell. Raise if dense clusters artifact.
MAX_PARTICLES_PER_CELL: int = 512

with open("behaviors.toml", mode="rb") as toml_file:
    config = toml.load(toml_file)
    type_config = config["types"]
    presets = config["presets"]

PARTICLE_COUNT: list[int] = [type_config[ptype]["population"] for ptype in type_config]
TYPE_COLORS: list[tuple[float, float, float]] = [
    type_config[ptype]["color"] for ptype in type_config
]
N_TYPES: int = len(PARTICLE_COUNT)

PARTICLE_COUNT_CUMSUM = np.cumsum(PARTICLE_COUNT)
ALL_PARTICLES_COUNT = np.sum(PARTICLE_COUNT)

# Each preset is a named N_TYPES×N_TYPES matrix (row-major flat list).
# Rows are the acting type, columns are the target type.
INTERACTION_PRESETS = {
    preset: np.array(presets[preset]["matrix"]).flatten() for preset in presets
}


WINDOW_TITLE: str = "Emergent Behaviours"
WINDOW_WIDTH: int = 980
WINDOW_HEIGHT: int = 980

PARTICLE_RADIUS: float = 0.0009

BACKGROUND_COLOR: tuple[float, float, float] = (0.04, 0.04, 0.08)
