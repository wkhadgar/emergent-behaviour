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

Aspect ratio correction:
Distances are computed in screen space by scaling the x component of the
separation vector by 1/ASPECT_RATIO before taking the norm. This makes the
interaction radius circular on screen rather than elliptical. Force directions
are derived from the original world-space diff vector (not the screen-space
one) so that forces remain symmetric and do not introduce a directional bias.

Density regulation:
For types with density_regulation enabled in behaviors.toml, the diagonal
coefficient M[t_i, t_i] is scaled by the normalised Shannon entropy of the
local neighbourhood type distribution:

    p_t       = count of type-t neighbours / total neighbours
    H         = -Σ_t p_t * log(p_t)
    H_norm    = H / log(N_TYPES)          (normalised to [0, 1])
    effective = M[t_i, t_i] * H_norm

H_norm = 0 when the neighbourhood is monotypic (all same type) — same-type
attraction collapses. H_norm = 1 when all types are equally represented —
full matrix coefficient applies. Cross-type coefficients are unconditional.

This prevents unbounded same-type cluster growth: a large monotypic blob
loses cohesion unless other types are present in its neighbourhood.

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


# Per-type density regulation flag, uploaded from config.DENSITY_REGULATION
# so kernels can read it on the GPU without a Python roundtrip.
_density_regulated = ti.field(dtype=ti.i32, shape=config.N_TYPES)


def upload_density_regulation(regulation: list[bool] | None = None):
    """
    Upload density regulation flags to the GPU field.

    Args:
        regulation: per-type bool list. Defaults to config.DENSITY_REGULATION
                    if not provided. Pass the renderer's live state to reflect
                    GUI changes without restarting.
    """
    source = regulation if regulation is not None else config.DENSITY_REGULATION
    for t in range(config.N_TYPES):
        _density_regulated[t] = int(source[t])


@ti.kernel
def init_particles():
    """
    Distribute particles uniformly at random, assigning types.

    Initial velocity is encoded into prev_position so that the first Verlet
    step produces correct motion without a special-case warm-up.
    """
    for i in fields.position:
        pos = ti.Vector([ti.random(ti.f32), ti.random(ti.f32)])
        angle = ti.random(ti.f32) * 2.0 * ti.math.pi
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
    the wall normal.
    """
    for d in ti.static(range(2)):
        if fields.position[i][d] < config.PARTICLE_RADIUS:
            fields.position[i][d] = 2 * config.PARTICLE_RADIUS - fields.position[i][d]
            fields.prev_position[i][d] = fields.position[i][d] + ti.abs(
                fields.position[i][d] - fields.prev_position[i][d]
            )

        if fields.position[i][d] > 1.0 - config.PARTICLE_RADIUS:
            fields.position[i][d] = (
                2 * (1.0 - config.PARTICLE_RADIUS) - fields.position[i][d]
            )
            fields.prev_position[i][d] = fields.position[i][d] + ti.abs(
                fields.position[i][d] - fields.prev_position[i][d]
            )


# Declared at module level because Taichi kernels cannot allocate arrays of
# dynamic size, and closures over mutable local kernel-scope state are not
# supported by the compiler.
_accumulated_force = ti.Vector.field(2, dtype=ti.f32, shape=config.ALL_PARTICLES_COUNT)


@ti.func
def _neighbourhood_entropy(type_counts: ti.template(), total: int) -> float:
    """
    Compute the normalised Shannon entropy of a neighbourhood type distribution.

        H_norm = -Σ_t (p_t * log(p_t)) / log(N_TYPES)

    Returns a value in [0, 1]. Returns 0 when total == 0 (empty neighbourhood)
    and 1 when all types are equally represented.

    Args:
        type_counts: fixed-size array of per-type neighbour counts.
        total:       sum of all counts (passed to avoid recomputing).
    """
    H = 0.0

    # Guard against empty neighbourhood — entropy is defined as zero.
    # Written without early return since Taichi does not support return
    # inside a non-static conditional.
    if total > 0:
        for t in ti.static(range(config.N_TYPES)):
            p = float(type_counts[t]) / float(total)
            if p > 0.0:
                H -= p * ti.math.log(p)

        # Normalise by maximum possible entropy log(N_TYPES).
        H = H / ti.math.log(float(config.N_TYPES))

    return H


@ti.kernel
def _accumulate_forces():
    """
    Accumulate pairwise interaction forces for all particles.

    For each particle i the neighbourhood is scanned once. Per-type neighbour
    counts are accumulated alongside forces so that entropy can be computed
    without a second pass.

    Distance is measured in screen space (diff_screen) to produce circular
    interaction radii on non-square windows. Force direction is derived from
    the world-space diff vector so that forces are not directionally biased
    by the aspect ratio correction.

    For regulated types, the diagonal coefficient is scaled by H_norm before
    application. Cross-type coefficients are always applied at full value.

    Newton's third law symmetry is intentionally broken — see module docstring.
    """
    for i in fields.position:
        _accumulated_force[i] = ti.Vector([0.0, 0.0])

    FALLOFF_COEF = 2 / (config.INTERACTION_RADIUS - config.PARTICLE_OVERLAP_DIAMETER)
    FALLOFF_OFFSET = (config.INTERACTION_RADIUS + config.PARTICLE_OVERLAP_DIAMETER) / 2

    for i in fields.position:
        force = ti.Vector([0.0, 0.0])
        pos_i = fields.position[i]
        type_i = fields.particle_type[i]
        cell = grid.position_to_cell(pos_i)
        type_counts = ti.Vector([0] * config.N_TYPES, dt=ti.i32)
        total_neighbours = 0
        same_type_force = ti.Vector([0.0, 0.0])

        for dx in ti.static(range(-1, 2)):
            for dy in ti.static(range(-1, 2)):
                nx = cell.x + dx
                ny = cell.y + dy

                if 0 <= nx < config.GRID_W and 0 <= ny < config.GRID_H:
                    c = grid.cell_to_index(nx, ny)
                    start = fields.cell_start[c]
                    count = fields.cell_count[c]

                    for k in range(start, start + count):
                        j = fields.sorted_indices[k]

                        if j != i:
                            diff = pos_i - fields.position[j]

                            # Screen-space distance for circular interaction
                            # radius on non-square windows.
                            diff_screen = ti.Vector(
                                [diff.x / config.ASPECT_RATIO, diff.y]
                            )
                            dist = diff_screen.norm() + 1e-6

                            # World-space unit direction — not scaled by aspect
                            # ratio so forces are not biased toward either axis.
                            direction = diff / dist

                            if dist < config.INTERACTION_RADIUS:
                                type_j = fields.particle_type[j]
                                coefficient = -fields.interaction_matrix[
                                    type_i * config.N_TYPES + type_j
                                ]

                                type_counts[type_j] += 1
                                total_neighbours += 1

                                falloff = max(
                                    1.0 - abs(FALLOFF_COEF * (dist - FALLOFF_OFFSET)), 0
                                )
                                if type_j == type_i:
                                    # Held separately; entropy weight applied
                                    # after the full neighbourhood scan.
                                    same_type_force += coefficient * falloff * direction
                                else:
                                    force += coefficient * falloff * direction

                            # Overlap resistance — type-independent, quadratic kernel.
                            if dist < config.PARTICLE_OVERLAP_DIAMETER:
                                overlap_falloff = (
                                    1.0 - dist / config.PARTICLE_OVERLAP_DIAMETER
                                ) ** 0.5

                                force += (
                                    config.OVERLAP_REPULSION
                                    * overlap_falloff
                                    * direction
                                )

        if _density_regulated[type_i]:
            entropy_weight = _neighbourhood_entropy(type_counts, total_neighbours)
            force += same_type_force * entropy_weight
        else:
            force += same_type_force

        _accumulated_force[i] = force


@ti.kernel
def step_interaction():
    """
    Störmer–Verlet integration driven by accumulated pairwise forces.

    The interaction matrix entirely governs cross-type and within-type
    forces; overlap resistance is applied unconditionally.
    """
    for i in fields.position:
        velocity = (fields.position[i] - fields.prev_position[i]) * config.DAMPING
        acceleration = _accumulated_force[i]

        fields.prev_position[i] = fields.position[i]
        fields.position[i] += velocity + acceleration * config.DT * config.DT

        _apply_boundary(i)


def step():
    """One full physics tick: force accumulation followed by integration."""
    _accumulate_forces()
    step_interaction()
