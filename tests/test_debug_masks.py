import matplotlib
matplotlib.use("Agg")

import numpy as np
from dataclasses import replace
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


def test_value_window_limits_mask_to_range():
    t = _tuner()
    dist = t.dist
    lo = float(np.percentile(dist, 80))
    hi = float(dist.max())
    t.s_range.set_val((lo, hi))
    t._on_range(None)
    t._debouncer._on_release(None)
    expected = (dist >= lo) & (dist <= hi)
    # mask keeps only surviving pieces, so it must be a subset of the window
    assert t.mask.sum() <= expected.sum()
    assert not (t.mask & ~expected).any()


def test_method_switch_resnaps_window():
    t = _tuner()
    t._on_method("mahalanobis")
    lo, hi = t.s_range.val
    dist = t.dist
    # window high handle rides the new distance max, low is a sane cutoff inside bounds
    assert hi == t.s_range.valmax
    assert t.s_range.valmin <= lo <= t.s_range.valmax
    # the raw value-window itself isn't collapsed to a near-empty sliver of the
    # new (much larger) distance scale -- this is the regression: a stale
    # absolute (lo, hi) from the old method's scale would leave almost nothing
    # in range once bounds jump from e.g. sam's [0, pi/2] to mahalanobis's
    # unbounded scale. (Whether pieces then *survive* min_area/morphology
    # cleanup is a separate, unrelated concern -- this fixture's default
    # min_area=1000 filters out every synthetic piece regardless of method,
    # so we check the window against the distance map directly.)
    window = (dist >= lo) & (dist <= hi)
    assert window.mean() > 0.05


def test_band_contrast_sets_clim_without_recompute():
    t = _tuner()
    t._debouncer._recompute = lambda: (_ for _ in ()).throw(
        AssertionError("contrast must not recompute"))
    t.s_band_clip.set_val((0.2, 0.8))
    t._on_band_clip(None)
    lo, hi = t._im_band.get_clim()
    assert lo < hi


def test_max_dim_downsamples_working_cube():
    cube, wl = synthetic_cube(rows=300, cols=300, bands=12, seed=3)
    t = MaskTuner(cube, wl, "test", max_dim=100)
    assert t.ds_factor >= 3
    assert max(t.cube.shape[:2]) <= 100


def test_background_bbox_and_flat_field_key_the_distance_cache():
    t = _tuner()
    t._distance()
    n0 = len(t._dist_cache)
    t.piece_cfg = replace(t.piece_cfg, background_bbox=(0, 8, 0, 8))
    d1 = t._distance()
    assert len(t._dist_cache) == n0 + 1        # new key cached, distance recomputed
    assert d1.shape == t.cube.shape[:2]
    t.piece_cfg = replace(t.piece_cfg, flat_field=True)
    t._distance()
    assert len(t._dist_cache) == n0 + 2


def test_calibrate_check_noop_without_references():
    t = _tuner()                                # synthetic cube has no white/dark
    assert t.can_calibrate is False
    t._on_check("calibrate")
    assert t.use_calibrate is False


def test_watershed_check_toggles_config():
    t = _tuner()
    assert t.piece_cfg.watershed_split is False
    t._on_check("watershed")
    assert t.piece_cfg.watershed_split is True


def test_print_config_runs():
    t = _tuner()

    class E:
        key = "p"
    t._on_key(E())                              # must not raise
