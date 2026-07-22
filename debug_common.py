# debug_common.py
"""Shared helpers for the interactive debug tuners."""

from __future__ import annotations


class Debouncer:
    """Defer expensive recomputes until the mouse button is released.

    Matplotlib sliders fire ``on_changed`` on every intermediate value while
    dragging. Heavy callbacks should call :meth:`mark_dirty` instead of
    recomputing; the actual ``recompute`` runs once, on the next
    ``button_release_event``.
    """

    def __init__(self, canvas, recompute):
        self._recompute = recompute
        self._dirty = False
        canvas.mpl_connect("button_release_event", self._on_release)

    def mark_dirty(self):
        self._dirty = True

    def _on_release(self, event=None):
        if self._dirty:
            self._dirty = False
            self._recompute()
