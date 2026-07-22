import matplotlib
matplotlib.use("Agg")

import numpy as np
import debug_masks
from debug_masks import MaskTuner, synthetic_cube


def _tuner():
    cube, wl = synthetic_cube(rows=80, cols=80, bands=24, seed=1)
    return MaskTuner(cube, wl, "test")


def test_import_uses_cube_io():
    # regression: debug_masks must not import the removed hsi_workflow.io
    import inspect
    src = inspect.getsource(debug_masks)
    assert "hsi_workflow.io" not in src
    assert "hsi_workflow.cube_io" in src


def test_heavy_param_is_debounced_not_immediate():
    t = _tuner()
    calls = []
    t._debouncer._recompute = lambda: calls.append(1)
    # simulate a slider drag on a heavy param
    t._on_param(None)
    assert calls == []                 # nothing recomputed during drag
    t._debouncer._on_release(None)     # mouse up
    assert calls == [1]                # recomputed exactly once


def test_band_step_is_light_no_recompute():
    t = _tuner()
    t._debouncer._recompute = lambda: (_ for _ in ()).throw(
        AssertionError("band step must not trigger heavy recompute"))
    t._on_band(5)
    assert t.band == 5


def test_band_step_refreshes_color_scale():
    # regression: _im_band.set_data() alone freezes vmin/vmax at the first
    # band's range; stepping to a band with a different intensity range must
    # refresh the color limits too, or the image looks washed-out/saturated.
    t = _tuner()
    t._on_band(0)
    lo0, hi0 = t._im_band.get_clim()

    # pick the band whose (decimated) data range differs most from band 0
    band = t.cube[:, :, 0]
    step = max(1, int(np.ceil(max(band.shape) / debug_masks.MAX_DISPLAY)))
    sl = (slice(None, None, step), slice(None, None, step))
    ranges = [float(t.cube[:, :, b][sl].max() - t.cube[:, :, b][sl].min())
              for b in range(t.bands)]
    b = int(np.argmax(ranges))
    assert b != 0  # sanity: the synthetic cube must actually vary across bands

    t._on_band(b)
    lo1, hi1 = t._im_band.get_clim()
    expected_lo = float(t.cube[:, :, b][sl].min())
    expected_hi = float(t.cube[:, :, b][sl].max())

    assert (lo1, hi1) != (lo0, hi0)
    assert lo1 == expected_lo
    assert hi1 == expected_hi
