# Copyright (c) 2026
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Entry point — simulation loop, input handling, and preset cycling.

ti.init() is called at module level before any project import. Taichi fields
are declared at module level in fields.py and require an active runtime at
import time; placing ti.init() here guarantees correct initialisation order.

Frame structure per tick:

    1. grid.rebuild()           — reconstruct spatial hash from current positions
    2. substep loop             — run behaviors.step() SUBSTEPS times
    3. renderer.draw()          — composite and present frame

The spatial hash is rebuilt once per frame rather than per substep. This is
valid because the hash is consumed during force accumulation only; positional
changes per substep at DT=5e-4 are well below one cell width (INTERACTION_RADIUS
= 0.025), so the stale hash introduces negligible error.
"""

import taichi as ti

# ti.init() must precede all project imports.
ti.init(arch=ti.cpu)

import behaviors  # noqa: E402
import config  # noqa: E402
import fields  # noqa: E402
import grid  # noqa: E402
import renderer as renderer_module  # noqa: E402


_PRESET_NAMES = list(config.INTERACTION_PRESETS.keys())

_KEY_BINDINGS = """
controls
────────
  SPACE        reset particles
  TAB          cycle interaction preset
  S            save screenshot
  ESC          quit
"""


def _reset(renderer: renderer_module.Renderer, preset_name: str):
    behaviors.init_particles()
    fields.populate_type_colors()
    fields.populate_interaction_matrix(preset_name)
    renderer.on_particles_reset()


def main():
    renderer = renderer_module.Renderer()

    preset_index = 0

    _reset(renderer, _PRESET_NAMES[preset_index])

    print(_KEY_BINDINGS)
    print(f"preset → {_PRESET_NAMES[preset_index]}")

    while renderer.running:
        event = renderer.poll_event()
        while event is not None:
            match event.key:
                case " ":
                    _reset(renderer, _PRESET_NAMES[preset_index])
                    print("particles reset")

                case ti.ui.TAB:
                    preset_index = (preset_index + 1) % len(_PRESET_NAMES)
                    fields.populate_interaction_matrix(_PRESET_NAMES[preset_index])
                    print(f"preset → {_PRESET_NAMES[preset_index]}")

                case "s":
                    renderer.save_screenshot()

                case ti.ui.ESCAPE:
                    renderer.window.running = False

            event = renderer.poll_event()

        grid.rebuild()

        for _ in range(config.SUBSTEPS):
            behaviors.step()

        renderer.draw()


if __name__ == "__main__":
    main()
