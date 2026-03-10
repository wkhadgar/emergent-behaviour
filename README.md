# Emergent Behaviours

A real-time large-scale particle simulation platform for studying emergent
behaviour arising from typed local interaction rules. Heterogeneous particle
populations interact through a configurable pairwise force matrix, producing
collective phenomena — clustering, phase separation, pursuit dynamics,
self-organisation — from purely local rules with no global coordination.

The interaction model is based on the Particle Life framework (Ventrella, 2017).

---

## Requirements

- Python 3.11+
- A CUDA, Metal, or Vulkan-capable GPU (for CPU fallback, reduce populations
  in `behaviors.toml` and set `arch=ti.cpu` in `main.py`)

```bash
pip install -r requirements.txt
python main.py
```

---

## Controls

| Key     | Action                                   |
|---------|------------------------------------------|
| `TAB`   | Cycle to next interaction preset         |
| `SPACE` | Reset particle positions and velocities  |
| `S`     | Save screenshot to `screenshots/`        |
| `ESC`   | Quit                                     |

---

## Project layout

```
emergent-behaviour/
├── behaviors.toml  — particle types, populations, and interaction presets
├── config.py       — simulation parameters (timestep, damping, grid, etc.)
├── fields.py       — GPU memory layout and population helpers
├── grid.py         — spatial hash: construction and neighbourhood iteration
├── behaviors.py    — integration scheme and force kernels
├── renderer.py     — window, per-type colouring, screenshot
└── main.py         — simulation loop, input, orchestration
```

Module dependencies are strictly one-directional:

```
behaviors.toml → config → fields → grid → behaviors → renderer → main
```

---

## Configuring types and interactions

All particle definitions and interaction rules live in `behaviors.toml`.
No source file needs to be modified to add, remove, or retune a type or preset.

### Defining types

Each type is a named section under `[types]` with a colour and a population:

```toml
[types.red]
color      = [0.95, 0.30, 0.30]
population = 1000

[types.blue]
color      = [0.30, 0.65, 1.00]
population = 100
```

Type names are arbitrary. Declaration order determines the type index used
in the interaction matrix and GPU fields.

### Interaction presets

Each preset is a named section under `[presets]` with a description and an
`N × N` matrix, where `N` is the number of declared types. Row `i` and
column `j` give the force coefficient applied to a particle of type `i` when
it encounters a particle of type `j` within `INTERACTION_RADIUS`:

```
positive → repulsion
negative → attraction
zero     → no interaction
```

Column and row comments are encouraged for readability:

```toml
[presets.predator_prey]
description = "Asymmetric pursuit chain — type 0 chases 1, 1 chases 2, etc."
matrix = [
  #red,   blue,  green, yellow
  [ 10.0,  -8.0,   4.0,   8.0],  # red
  [  8.0,  10.0,  -8.0,   4.0],  # blue
  [  4.0,   8.0,  10.0,  -8.0],  # green
  [ -8.0,   4.0,   8.0,  10.0],  # yellow
]
```

Presets are cycled at runtime with `TAB` without resetting particle positions,
allowing live comparison between interaction regimes.

### Overlap resistance

Independent of the interaction matrix, all particle pairs closer than
`PARTICLE_OVERLAP_DIAMETER` experience a stiff quadratic repulsion. This
prevents particles from collapsing into singularities under strong attraction.
Its stiffness is configured in `config.py` as it is a numerical stability
concern rather than a behavioural one.

---

## Simulation parameters

Physics and rendering parameters are in `config.py` with inline documentation.

| Constant                   | Effect                                          |
|----------------------------|-------------------------------------------------|
| `DT`                       | Integration timestep                            |
| `SUBSTEPS`                 | Physics iterations per rendered frame           |
| `DAMPING`                  | Viscous drag coefficient                        |
| `INTERACTION_RADIUS`       | Neighbourhood radius and spatial hash cell size |
| `MAX_PARTICLES_PER_CELL`   | Overflow cap for dense local clusters           |
| `PARTICLE_RADIUS`          | Rendered particle size                          |
| `PARTICLE_OVERLAP_DIAMETER`| Minimum separation before overlap resistance    |
| `OVERLAP_REPULSION`        | Stiffness of the overlap resistance force       |

---

## Integration scheme

Störmer–Verlet integration throughout. Velocity is stored implicitly as
`(position - prev_position)`, providing second-order accuracy and symplectic
energy behaviour. See `behaviors.py` for the full derivation and boundary
correction.

---

## Spatial partitioning

Uniform grid hash with cell size equal to `INTERACTION_RADIUS`. Three-pass
counting sort (count → prefix sum → fill), O(N) per frame. The prefix sum
pass runs on CPU — see `grid.py` for implementation details and the
Ihmsen et al. (2011) reference.

---

## References

- Ventrella, J. (2017). *Clusters*. Lulu Press.
- Ihmsen, M. et al. (2011). A parallel SPH implementation on multi-core CPUs.
  *Computer Graphics Forum*, 30(1), 99–112.
- Hairer, E., Lubich, C., & Wanner, G. (2006). *Geometric Numerical
  Integration*. Springer.
