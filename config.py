# Copyright (c) 2026
#
# SPDX-License-Identifier: BSD-3-Clause

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

import tomllib as toml
import numpy as np
from numpy.__config__ import CONFIG


# Timestep for Störmer–Verlet integration. Reducing this improves stability
# at the cost of requiring more sub-steps to maintain simulation speed.
DT: float = 5e-4

# Sub-steps executed per rendered frame. More sub-steps improve stability at
# high velocities but increase GPU time proportionally.
SUBSTEPS: int = 3

with open("behaviors.toml", mode="rb") as _toml_file:
    _loaded = toml.load(_toml_file)
    _type_config = _loaded["types"]
    _presets = _loaded["presets"]
    _config = _loaded["config"]
    _env = _config["environment"]

PARTICLE_COUNT: list[int] = [_type_config[t]["population"] for t in _type_config]
TYPE_COLORS: list[tuple[float, float, float]] = [
    _type_config[t]["color"] for t in _type_config
]

# Per-type density regulation flag. When True, the diagonal interaction
# coefficient for that type is scaled by the normalised Shannon entropy of
# its local neighbourhood, preventing unbounded same-type cluster growth
# unless other types are present. Defaults to False if omitted in the TOML.
DENSITY_REGULATION: list[bool] = [
    _type_config[t].get("density_regulation", False) for t in _type_config
]

N_TYPES: int = len(PARTICLE_COUNT)

PARTICLE_COUNT_CUMSUM: np.ndarray = np.cumsum(PARTICLE_COUNT)
ALL_PARTICLES_COUNT: int = int(np.sum(PARTICLE_COUNT))

# Each preset is a named N_TYPES×N_TYPES matrix stored as a flat row-major
# numpy array. Rows are the acting type, columns are the target type.
INTERACTION_PRESETS: dict[str, np.ndarray] = {
    preset: np.array(_presets[preset]["matrix"], dtype=np.float32).flatten()
    for preset in _presets
}

INTERACTION_RADIUS: float = _config["interaction"] / 100

# Viscous drag coefficient applied each integration step.
# 1.0 = energy conserved, values below 1.0 dissipate kinetic energy.
DAMPING: float = max(100 - _env["damping"], 0) / 100
BACKGROUND_COLOR: tuple[float, float, float] = tuple(_env["color"])

PARTICLE_RADIUS: float = (_config["size"] / 100) / 2

# Stiffness of the overlap repulsion force. Should significantly exceed the
# strongest attraction coefficient to guarantee no singularities form.
OVERLAP_REPULSION: float = _config["separation_intensity"]

# Minimum allowed separation before overlap repulsion activates.
PARTICLE_OVERLAP_DIAMETER: float = PARTICLE_RADIUS * 5

WINDOW_TITLE: str = "Emergent Behaviours"

WINDOW_WIDTH: int = 1890
WINDOW_HEIGHT: int = 1025
WINDOW_DIMS = (WINDOW_WIDTH, WINDOW_HEIGHT)
WINDOW_AT = (15, 45)
ASPECT_RATIO: float = WINDOW_HEIGHT / WINDOW_WIDTH

GRID_W: int = int(1.0 / INTERACTION_RADIUS)
GRID_H: int = int(1.0 / INTERACTION_RADIUS)
GRID_SIZE: int = GRID_W * GRID_H

