import matplotlib
matplotlib.use("Agg")

import numpy as np
import debug_film
from debug_film import FilmTuner, synthetic_piece


def _tuner():
    data, mask, wl, bare = synthetic_piece(rows=80, cols=80, bands=24, seed=1)
    return FilmTuner(data, mask, wl, bare, "test")


def test_import_uses_cube_io():
    import inspect
    src = inspect.getsource(debug_film)
    assert "hsi_workflow.io" not in src
    assert "hsi_workflow.cube_io" in src


def test_recovers_some_oxide_within_wafer():
    t = _tuner()
    assert t.sio2.sum() > 0                  # oxide found
    assert not (t.sio2 & ~t.mask).any()      # stays inside the wafer
    assert not (t.sio2 & t.substrate).any()  # disjoint


def test_heavy_param_debounced():
    t = _tuner()
    calls = []
    t._debouncer._recompute = lambda: calls.append(1)
    t._on_param(None)
    assert calls == []
    t._debouncer._on_release(None)
    assert calls == [1]


def test_band_step_is_light():
    t = _tuner()
    t._debouncer._recompute = lambda: (_ for _ in ()).throw(
        AssertionError("band step must not recompute"))
    t._on_band(5)
    assert t.band == 5


def test_reference_switch_recomputes_distance():
    t = _tuner()
    t._distance()
    n0 = len(t._dist_cache)
    t._on_ref("in_piece")
    assert len(t._dist_cache) >= n0 + 1


def test_print_config_runs():
    t = _tuner()

    class E:
        key = "p"
    t._on_key(E())
