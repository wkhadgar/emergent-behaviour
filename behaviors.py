# Copyright (c) 2026
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Physics kernels — initialisation, integration, and behaviour definitions.

Integration scheme:
All behaviours use Störmer–Verlet integration. Velocity is stored implicitly
as the displacement between the current and previous position:

    v ≈ (position - prev_position) / DT

This representation is equivalent to explicit Euler in cost but provides
second-order accuracy and symplectic (energy-conserving) character, preventing
the slow energy drift observed with explicit Euler methods (Hairer et al.,
2006, "Geometric Numerical Integration").

One integration step:

    velocity      = (position - prev_position) * DAMPING
    prev_position = position
    position      = position + velocity + acceleration * DT²

Damping is applied to the reconstructed velocity before the step, modelling
viscous drag. The DT² factor on acceleration is the Verlet characteristic.

Interaction model:
Pairwise forces between particles of types (t_i, t_j) are governed by the
interaction matrix M, where M[t_i * N_TYPES + t_j] is the force coefficient.
The force on particle i due to neighbour j is:

    f = M[t_i, t_j] * (1 - d / R) * (p_i - p_j) / d

where d is the separation distance and R is INTERACTION_RADIUS. The factor
(1 - d/R) is a linear kernel producing maximum force at contact and zero
force at the radius boundary. This is the Particle Life force model
(Ventrella 2017, "Clusters").

Adding a new behaviour:
Write a @ti.kernel following this template and register it in main.py:

    @ti.kernel
    def step_<name>():
        for i in fields.position:
            velocity = (fields.position[i] - fields.prev_position[i]) * config.DAMPING

            acceleration = ti.Vector([0.0, 0.0])
            # ... compute acceleration from interaction rules ...

            fields.prev_position[i] = fields.position[i]
            fields.position[i] += velocity + acceleration * config.DT * config.DT

            _apply_boundary(i)
"""

import taichi as ti

import config
import fields
import grid


@ti.kernel
def init_particles():
    """
    Distribute particles uniformly at random, assigning types.

    Initial velocity is encoded into prev_position so that the first Verlet
    step produces correct motion without a special-case warm-up.
    """

    for i in fields.position:
        pos = ti.Vector([ti.random(ti.f32), ti.random(ti.f32)])
        angle = ti.random(ti.f32) * 2.0 * 3.14159265
        speed = ti.random(ti.f32) * 0.2
        vel = ti.Vector([ti.cos(angle), ti.sin(angle)]) * speed

        fields.position[i] = pos
        fields.prev_position[i] = pos - vel * config.DT

        t = config.N_TYPES - 1
        for k in ti.static(range(config.N_TYPES)):
            if i < config.PARTICLE_COUNT_CUMSUM[k] and t == config.N_TYPES - 1:
                t = k
        fields.particle_type[i] = t


@ti.func
def _apply_boundary(i: int):
    """
    Reflective boundaries on all four walls.

    On penetration, the overshoot is folded back from the wall and
    prev_position is corrected so the implicit velocity reverses along
    the wall normal, attenuated by BOUNCE_RESTITUTION.
    """
    for d in ti.static(range(2)):
        if fields.position[i][d] < 0.0:
            fields.position[i][d] = -fields.position[i][d]
            fields.prev_position[i][d] = fields.position[i][d] + ti.abs(
                fields.position[i][d] - fields.prev_position[i][d]
            )

        if fields.position[i][d] > 1.0:
            fields.position[i][d] = 2.0 - fields.position[i][d]
            fields.prev_position[i][d] = fields.position[i][d] + ti.abs(
                fields.position[i][d] - fields.prev_position[i][d]
            )


# Declared at module level because Taichi kernels cannot allocate arrays of
# dynamic size, and closures over mutable local kernel-scope state are not
# supported by the compiler.
_accumulated_force = ti.Vector.field(2, dtype=ti.f32, shape=config.ALL_PARTICLES_COUNT)


@ti.kernel
def _accumulate_forces():
    """
    Accumulate pairwise interaction forces for all particles.

    For each particle i, the force contribution from neighbour j is:

        coefficient = interaction_matrix[type_i * N_TYPES + type_j]
        falloff     = 1 - distance / INTERACTION_RADIUS   (linear kernel)
        force      += coefficient * falloff * direction(i ← j)

    Newton's third law symmetry is intentionally broken: each particle
    queries its own neighbourhood independently. This avoids atomic writes
    on the force accumulator at the cost of computing each pair twice. On
    GPU, the absence of synchronisation consistently outweighs the doubled
    arithmetic.
    """
    for i in fields.position:
        _accumulated_force[i] = ti.Vector([0.0, 0.0])

    for i in fields.position:
        force = ti.Vector([0.0, 0.0])
        pos_i = fields.position[i]
        type_i = fields.particle_type[i]
        cell = grid.position_to_cell(pos_i)

        for dx in ti.static(range(-1, 2)):
            for dy in ti.static(range(-1, 2)):
                nx = cell.x + dx
                ny = cell.y + dy

                if 0 <= nx < config.GRID_W and 0 <= ny < config.GRID_H:
                    c = grid.cell_to_index(nx, ny)
                    start = fields.cell_start[c]
                    count = fields.cell_count[c]

                    for k in range(
                        start, start + ti.min(count, config.MAX_PARTICLES_PER_CELL)
                    ):
                        j = fields.sorted_indices[k]

                        if j != i:
                            diff = pos_i - fields.position[j]
                            dist = diff.norm() + 1e-6

                            if dist < config.INTERACTION_RADIUS:
                                type_j = fields.particle_type[j]
                                coefficient = fields.interaction_matrix[
                                    type_i * config.N_TYPES + type_j
                                ]
                                falloff = 1.0 - dist / config.INTERACTION_RADIUS

                                # diff / dist gives the unit direction from j to i
                                # (repulsive when coefficient is positive).
                                force += coefficient * falloff * diff / dist

                            # Overlap resistance — applied independently of type, only at very close range.
                            # Uses a stiffer quadratic kernel over PARTICLE_DIAMETER rather than
                            # INTERACTION_RADIUS, producing a hard-ish core without requiring a
                            # constraint solver.
                            if dist < config.PARTICLE_OVERLAP_DIAMETER:
                                overlap_falloff = (
                                    1.0 - dist / config.PARTICLE_OVERLAP_DIAMETER
                                ) ** 2
                                force += (
                                    config.OVERLAP_REPULSION
                                    * overlap_falloff
                                    * diff
                                    / dist
                                )

        _accumulated_force[i] = force


@ti.kernel
def step_interaction():
    """
    Störmer–Verlet integration driven by the active interaction matrix.

    The interaction matrix entirely governs cross-type and within-type
    forces.
    """

    for i in fields.position:
        velocity = (fields.position[i] - fields.prev_position[i]) * config.DAMPING
        acceleration = _accumulated_force[i]

        fields.prev_position[i] = fields.position[i]
        fields.position[i] += velocity + acceleration * config.DT * config.DT

        _apply_boundary(i)


def step():
    """
    One full physics tick.

    Args:
        interaction: When True, accumulate pairwise forces before
                     integrating. When False, only gravity acts.
    """
    _accumulate_forces()
    step_interaction()
