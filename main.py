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
    3. renderer.draw_gui()      — draw control panel, returns dirty flag
    4. renderer.draw()          — composite particles and present frame
    5. flush if dirty           — push changed matrix / regulation to GPU fields

GUI state (matrix values, regulation flags) is owned by the Renderer and
flushed to GPU fields only when draw_gui() reports a change. This avoids
unnecessary GPU uploads on frames where nothing changed.
"""

import taichi as ti

# ti.init() must precede all project imports.
ti.init(arch=ti.gpu)

import behaviors  # noqa: E402
import config  # noqa: E402
import fields  # noqa: E402
import grid  # noqa: E402
import renderer as renderer_module  # noqa: E402


_KEY_BINDINGS = """
controls
────────
  SPACE        reset particles
  TAB          cycle interaction preset
  UP / DOWN    nudge selected matrix cell
  S            save screenshot
  ESC          quit
"""


def _reset(renderer: renderer_module.Renderer):
    behaviors.init_particles()
    fields.populate_type_colors()
    fields.populate_interaction_matrix(renderer.matrix)
    behaviors.upload_density_regulation(renderer.density_regulation)
    renderer.on_particles_reset()


def _flush_gui_changes(renderer: renderer_module.Renderer):
    """Push current GUI state to GPU fields."""
    fields.populate_interaction_matrix(renderer.matrix)
    behaviors.upload_density_regulation(renderer.density_regulation)


def main():
    renderer = renderer_module.Renderer()
    preset_names = list(config.INTERACTION_PRESETS.keys())
    preset_index = 0

    _reset(renderer)

    print(_KEY_BINDINGS)
    print(f"preset → {preset_names[preset_index]}")

    while renderer.running:
        event = renderer.poll_event()
        while event is not None:
            match event.key:
                case " ":
                    _reset(renderer)
                    print("particles reset")

                case ti.ui.TAB:
                    preset_index = (preset_index + 1) % len(preset_names)
                    preset = preset_names[preset_index]
                    renderer.matrix = list(
                        map(float, config.INTERACTION_PRESETS[preset])
                    )
                    _flush_gui_changes(renderer)
                    print(f"preset → {preset}")

                case ti.ui.UP | ti.ui.DOWN:
                    if renderer.handle_arrow_nudge(event.key):
                        fields.populate_interaction_matrix(renderer.matrix)

                case "s":
                    renderer.save_screenshot()

                case ti.ui.ESCAPE:
                    renderer.window.running = False

            event = renderer.poll_event()

        grid.rebuild()

        for _ in range(config.SUBSTEPS):
            behaviors.step()

        dirty = renderer.draw_gui()

        if dirty:
            _flush_gui_changes(renderer)

        renderer.draw()


if __name__ == "__main__":
    main()
