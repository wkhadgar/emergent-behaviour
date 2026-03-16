# Copyright (c) 2026
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Renderer — window lifecycle, per-type colouring, GUI, and screenshot capture.

The GUI is drawn via Taichi's IMGUI integration (ti.ui.Gui), overlaid on the
simulation canvas each frame. All matrix and regulation state is owned by the
caller (main.py) as plain Python structures; the GUI reads and writes that
state and returns a dirty flag so the caller can push changes to the GPU.

Sub-window layout (left side, normalised coordinates):
    [0.0, 0.0, 0.22, 1.0] — interaction matrix + density regulation + presets
"""

import os

import taichi as ti

import config
import fields


# Slider range for interaction coefficients.
_COEFF_MIN: float = -20.0
_COEFF_MAX: float = 20.0

# Arrow key increment applied to the selected matrix cell.
_ARROW_STEP: float = 0.5


@ti.data_oriented
class Renderer:
    """Owns the Taichi window and canvas for the lifetime of the simulation."""

    def __init__(self):
        self.window = ti.ui.Window(
            config.WINDOW_TITLE,
            config.WINDOW_DIMS,
            vsync=True,
            pos=config.WINDOW_AT,
        )
        self.canvas = self.window.get_canvas()
        self._gui = self.window.get_gui()

        self._frame_index = 0
        self._screenshot_dir = "screenshots"

        self._color_field = ti.Vector.field(
            3, dtype=ti.f32, shape=config.ALL_PARTICLES_COUNT
        )
        self._rebuild_color_field()

        # Matrix GUI state — flat list matching interaction_matrix field layout.
        # Owned here so the GUI can read/write without touching the GPU field
        # on every slider drag; caller flushes to GPU when dirty.
        self.matrix: list[float] = list(
            config.INTERACTION_PRESETS[list(config.INTERACTION_PRESETS.keys())[0]]
        )

        # Per-type density regulation state.
        self.density_regulation: list[bool] = list(config.DENSITY_REGULATION)

        # Currently selected matrix cell (row, col) for arrow key nudging.
        # None when no cell is selected.
        self.selected_cell: tuple[int, int] | None = (0, 0)

        # Short type name labels derived from TOML key order.
        import tomllib

        with open("behaviors.toml", "rb") as _f:
            self._type_names: list[str] = list(tomllib.load(_f)["types"].keys())

    @ti.kernel
    def _fill_colors(self):
        """Write each particle's type colour into the per-particle colour field."""
        for i in self._color_field:
            self._color_field[i] = fields.type_color[fields.particle_type[i]]

    def _rebuild_color_field(self):
        self._fill_colors()

    def on_particles_reset(self):
        """
        Notify the renderer that particle types have been reassigned.

        Must be called after init_particles() so the colour field reflects
        the new type layout.
        """
        self._rebuild_color_field()

    @property
    def running(self) -> bool:
        return self.window.running

    def poll_event(self):
        """Return the next pending input event, or None if the queue is empty."""
        if self.window.get_event(ti.ui.PRESS):
            return self.window.event
        return None

    def draw_gui(self) -> bool:
        """
        Draw the control panel and return True if any value changed.

        The caller is responsible for flushing changed state to the GPU
        fields when this returns True.

        Sections:
            1. Interaction matrix  — slider_float per cell, arrow nudge on
                                     the selected cell.
            2. Density regulation  — checkbox per type.
            3. Presets             — one button per named preset; clicking
                                     overwrites self.matrix entirely.
            4. Add type (stub)     — placeholder for future dynamic type
                                     addition.

        Returns:
            bool: True if matrix or regulation state changed this frame.
        """
        n = config.N_TYPES
        dirty = False

        with self._gui.sub_window("Controls", 0.0, 0.0, 0.22, 1.0):
            self._gui.text("Interaction Matrix")
            self._gui.text("(row acts on column)")

            # Selected cell display and arrow nudge hint.
            if self.selected_cell is not None:
                r, c = self.selected_cell
                rn = self._type_names[r]
                cn = self._type_names[c]
                val = self.matrix[r * n + c]
                self._gui.text(f"selected: {rn} -> {cn}  ({val:+.2f})")
                self._gui.text("UP / DOWN to nudge")
            else:
                self._gui.text("click a slider to select")

            for row in range(n):
                self._gui.text(self._type_names[row])
                for col in range(n):
                    idx = row * n + col
                    label = f"{self._type_names[col]}##{row}{col}"
                    old_val = self.matrix[idx]
                    new_val = self._gui.slider_float(
                        label, old_val, _COEFF_MIN, _COEFF_MAX
                    )
                    if new_val != old_val:
                        self.matrix[idx] = new_val
                        self.selected_cell = (row, col)
                        dirty = True

            self._gui.text("")
            self._gui.text("Density Regulation")

            for t in range(n):
                old_val = self.density_regulation[t]
                new_val = self._gui.checkbox(self._type_names[t], old_val)
                if new_val != old_val:
                    self.density_regulation[t] = new_val
                    dirty = True

            self._gui.text("")
            self._gui.text("Presets")

            for name, matrix in config.INTERACTION_PRESETS.items():
                if self._gui.button(name):
                    self.matrix = list(map(float, matrix))
                    dirty = True

            self._gui.text("")
            if self._gui.button("+ Add type  [todo]"):
                print("add type: not yet implemented")

        return dirty

    def handle_arrow_nudge(self, key) -> bool:
        """
        Nudge the selected matrix cell by _ARROW_STEP on UP/DOWN keypress.

        Args:
            key: event.key from the main input loop.

        Returns:
            bool: True if the matrix changed.
        """
        if self.selected_cell is None:
            return False

        r, c = self.selected_cell
        idx = r * config.N_TYPES + c

        if key == ti.ui.UP:
            self.matrix[idx] = min(self.matrix[idx] + _ARROW_STEP, _COEFF_MAX)
            return True

        if key == ti.ui.DOWN:
            self.matrix[idx] = max(self.matrix[idx] - _ARROW_STEP, _COEFF_MIN)
            return True

        return False

    def draw(self):
        """Composite particles and present one frame."""
        self.canvas.set_background_color(config.BACKGROUND_COLOR)
        self.canvas.circles(
            fields.position,
            radius=config.PARTICLE_RADIUS,
            per_vertex_color=self._color_field,
        )
        self.window.show()
        self._frame_index += 1

    def save_screenshot(self):
        """Write the current frame to disk as a PNG under screenshots/."""
        os.makedirs(self._screenshot_dir, exist_ok=True)
        path = os.path.join(self._screenshot_dir, f"frame_{self._frame_index:06d}.png")
        self.window.save_image(path)
        print(f"screenshot saved → {path}")
