import matplotlib
matplotlib.use("Agg")

import numpy as np
from debug_preprocess import PreprocessTuner, synthetic_cube


def _tuner(rows=60, cols=60, bands=40, max_dim=600):
    cube, wl = synthetic_cube(rows=rows, cols=cols, bands=bands, seed=1)
    # no white/dark -> calibrate disabled, raw path
    return PreprocessTuner(cube, wl, 1.0, None, None, 1.0, 1.0, "test", max_dim=max_dim)


def test_heavy_param_is_debounced():
    t = _tuner()
    calls = []
    t._debouncer._recompute = lambda: calls.append(1)
    t._on_param(None)
    assert calls == []                 # nothing recomputed during drag
    t._debouncer._on_release(None)     # mouse up
    assert calls == [1]                # recomputed exactly once


def test_band_step_does_not_recompute():
    t = _tuner()
    t._debouncer._recompute = lambda: (_ for _ in ()).throw(
        AssertionError("band step must not recompute"))
    t._on_band(7)
    assert t.band == 7


def test_contrast_sets_clim_without_recompute():
    t = _tuner()
    t._debouncer._recompute = lambda: (_ for _ in ()).throw(
        AssertionError("contrast must not recompute"))
    t.s_clip.set_val((0.1, 0.9))
    t._on_clip(None)
    lo, hi = t._im.get_clim()
    assert lo < hi


def test_reference_subtract_off_by_default():
    t = _tuner()
    assert t.ref_spectrum is None
    assert t.use_ref_subtract is False


def test_reference_subtract_reduces_reference_pixel():
    t = _tuner()
    r, c = 30, 30
    t._set_reference(r, c)
    t.use_ref_subtract = True
    # isolate the subtraction
    t.use_smooth = t.use_snv = t.use_baseline = False
    before = t._src_pixel(r, c)
    out = t._process(before[None, None, :])[0, 0, :]
    # subtracting the (5x5-averaged) self-reference leaves a smaller residual
    assert np.abs(out).mean() < np.abs(before).mean()


def test_clear_reference_key():
    t = _tuner()
    t._set_reference(10, 10)
    t.use_ref_subtract = True
    assert t.ref_spectrum is not None

    class E:
        key = "c"
    t._on_key(E())
    assert t.ref_spectrum is None
    assert t.use_ref_subtract is False


def test_max_dim_downsamples_working_grid():
    # a cube larger than max_dim must be strided for compute
    t = _tuner(rows=300, cols=300, bands=24, max_dim=100)
    assert t.step >= 3
    assert max(t._refl_ds.shape[:2]) <= 100
    # per-pixel spectra are still read at full spatial resolution + full bands
    assert (t.rows, t.cols) == (300, 300)
    px = t._src_pixel(299, 299)
    assert px.shape == (24,)
