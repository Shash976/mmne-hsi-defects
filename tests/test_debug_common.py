import matplotlib
matplotlib.use("Agg")

from debug_common import Debouncer


class FakeCanvas:
    """Captures the button_release_event callback so tests can fire it."""
    def __init__(self):
        self.release_cb = None
    def mpl_connect(self, name, cb):
        if name == "button_release_event":
            self.release_cb = cb
        return 1


def test_recompute_runs_once_on_release_only_when_dirty():
    calls = []
    canvas = FakeCanvas()
    d = Debouncer(canvas, lambda: calls.append(1))

    # release with nothing pending -> no recompute
    canvas.release_cb(None)
    assert calls == []

    # mark dirty (as a heavy slider drag would) -> still no recompute yet
    d.mark_dirty()
    assert calls == []

    # release -> exactly one recompute
    canvas.release_cb(None)
    assert calls == [1]

    # a second release without a new change -> no extra recompute
    canvas.release_cb(None)
    assert calls == [1]
