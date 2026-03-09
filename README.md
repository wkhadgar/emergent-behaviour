# Emergent Behaviours

A real-time large-scale particle simulation platform for studying emergent
behaviour arising from typed local interaction rules. The system supports
10 000+ particles at interactive frame rates across multiple particle types,
each governed by a configurable pairwise interaction matrix.

---

## Motivation

Complex collective behaviour — clustering, phase separation, pursuit patterns,
self-organisation — can emerge from populations of particles following simple
local rules with no global coordination. This simulator provides a controlled
environment to observe, tune, and record such phenomena at a scale where
statistical structure becomes visible. The multi-type interaction model is
based on the Particle Life framework (Ventrella, 2017).

---

## Requirements

- Python 3.11+
- A CUDA, Metal, or Vulkan-capable GPU (CPU fallback available — reduce
  `PARTICLE_COUNT` to ≤ 50 000 in `config.py` and set `arch=ti.cpu` in
  `main.py`)

```bash
pip install -r requirements.txt
python main.py
```

---

## Controls

| Key     | Action                                      |
|---------|---------------------------------------------|
| `TAB`   | Cycle to next interaction preset            |
| `SPACE` | Reset particle positions and velocities     |
| `G`     | Toggle interaction forces (gravity only)    |
| `S`     | Save screenshot to `screenshots/`           |
| `ESC`   | Quit                                        |

---

## Project layout

```
particle_simulator/
├── config.py       — parameters and interaction presets
├── fields.py       — GPU memory layout and population helpers
├── grid.py         — spatial hash: construction and neighbourhood iteration
├── behaviors.py    — integration scheme and force kernels
├── renderer.py     — window, per-type colouring, screenshot
└── main.py         — simulation loop, input, orchestration
```

Module dependencies are strictly one-directional:

```
config → fields → grid → behaviors → renderer → main
```

---

## Interaction matrix

Each preset defines an `N_TYPES × N_TYPES` matrix. Entry `[i, j]` is the
force coefficient applied to a particle of type `i` when it encounters a
particle of type `j` within `INTERACTION_RADIUS`:

```
positive → repulsion
negative → attraction
zero     → no interaction
```

Presets are defined in `config.py` under `INTERACTION_PRESETS`. New presets
can be added there without modifying any other file. The active preset is
cycled at runtime with `TAB` and can be changed mid-simulation without
resetting particle positions.

---

## Parameters

All parameters are in `config.py` with inline documentation.

| Constant              | Default   | Effect                                 |
|-----------------------|-----------|----------------------------------------|
| `PARTICLE_COUNT`      | 300 000   | Simulation scale                       |
| `N_TYPES`             | 4         | Number of distinct particle types      |
| `DT`                  | 5e-4      | Integration timestep                   |
| `SUBSTEPS`            | 6         | Physics iterations per rendered frame  |
| `DAMPING`             | 0.985     | Viscous drag coefficient               |
| `INTERACTION_RADIUS`  | 0.025     | Neighbourhood radius and cell size     |

---

## Integration scheme

Störmer–Verlet integration throughout. Velocity is stored implicitly as
`(position - prev_position)`, providing second-order accuracy and symplectic
energy behaviour. See `behaviors.py` for the derivation and boundary
correction.

---

## Spatial partitioning

Uniform grid hash with cell size equal to `INTERACTION_RADIUS`. Three-pass
counting sort (count → prefix sum → fill), O(N) per frame. The prefix sum
pass runs on CPU over ~1600 cells; see `grid.py` for implementation details
and the Ihmsen et al. (2011) reference.

---

## Adding a behaviour

1. Write a physics kernel in `behaviors.py` following the template in that
   file's module docstring.
2. Expose it through the `step()` function or add a dedicated entry point.
3. Register a keypress in `main.py`.

Document the physical model each behaviour approximates and cite relevant
literature where applicable.

---

## References

- Ventrella, J. (2017). *Clusters*. Lulu Press.
- Ihmsen, M. et al. (2011). A parallel SPH implementation on multi-core CPUs.
  *Computer Graphics Forum*, 30(1), 99–112.
- Hairer, E., Lubich, C., & Wanner, G. (2006). *Geometric Numerical
  Integration*. Springer.
