# Copyright (c) 2026
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Renderer — window lifecycle, per-type colouring, and screenshot capture.

Particle colours are derived from particle_type[i] via a GPU-side lookup
into type_color[t]. This avoids per-frame CPU colour computation and keeps
the render path entirely within GPU memory, eliminating PCIe readback
overhead at 300k+ particles.

Positions are passed directly as a Taichi field in normalised [0, 1]²
coordinates, which matches the simulation domain by design.
"""

import os

import taichi as ti

import config
import fields


@ti.data_oriented
class Renderer:
    """Owns the Taichi window and canvas for the lifetime of the simulation."""

    def __init__(self):
        self.window = ti.ui.Window(
            config.WINDOW_TITLE,
            (config.WINDOW_WIDTH, config.WINDOW_HEIGHT),
            vsync=True,
        )
        self.canvas = self.window.get_canvas()
        self._frame_index = 0
        self._screenshot_dir = "screenshots"

        # Build per-particle colour field on the GPU once; update only when
        # the type assignment changes (i.e. on reset, not every frame).
        self._color_field = ti.Vector.field(
            3, dtype=ti.f32, shape=config.ALL_PARTICLES_COUNT
        )
        self._rebuild_color_field()

    @ti.kernel
    def _fill_colors(self):
        """Write each particle's type colour into the per-particle colour field."""
        for i in self._color_field:
            self._color_field[i] = fields.type_color[fields.particle_type[i]]

    def _rebuild_color_field(self):
        """Recompute the colour field from current particle type assignments."""
        self._fill_colors()

    @property
    def running(self) -> bool:
        return self.window.running

    def poll_event(self):
        """Return the next pending input event, or None if the queue is empty."""
        if self.window.get_event(ti.ui.PRESS):
            return self.window.event
        return None

    def on_particles_reset(self):
        """
        Notify the renderer that particle types have been reassigned.

        Must be called after init_particles() so that the colour field
        reflects the new type layout.
        """
        self._rebuild_color_field()

    def draw(self):
        """Composite and present one frame."""
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
