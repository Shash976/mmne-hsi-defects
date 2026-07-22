# Debug Tuners Audit + Range/Reference Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `debug_masks.py` and `debug_preprocess.py` fast and non-laggy, fix their bugs, and add a min/max range control (mask value-window + display contrast) and a reference-and-subtract capability to both.

**Architecture:** Both are matplotlib interactive tuners. The core fix is to split every control into a *light* tier (band index, display contrast, overlay toggle → update a persistent artist in place) and a *heavy* tier (segmentation/preprocessing → debounced to mouse-release via a shared `Debouncer`). New features reuse stock matplotlib `RangeSlider` and `RectangleSelector`; the reference-distance math reuses existing `hsi_workflow.pieces` helpers.

**Tech Stack:** Python, NumPy, SciPy (`scipy.ndimage`), scikit-image, matplotlib (widgets: `Slider`, `RangeSlider`, `RadioButtons`, `CheckButtons`, `RectangleSelector`), pytest.

## Global Constraints

- Run everything under the `hsi` conda env: `conda run -n hsi python ...` / `conda run -n hsi pytest ...`.
- Tests must be headless: put `import matplotlib; matplotlib.use("Agg")` at the top of every test module **before** importing `matplotlib.pyplot` or the debug modules.
- No new third-party dependencies. `RangeSlider` and `RectangleSelector` are stock matplotlib.
- Reference-subtraction in preprocess is **debug-only** — do NOT add fields to `PreprocessConfig`. The `'p'` printout mentions it only as a `#` comment.
- Do not refactor `hsi_workflow/` beyond Task 2 (the vectorized component-size count that directly serves the perf goal).
- Keep both tools runnable with `--demo` (no data on disk) at all times.

**Prerequisite check (run once before Task 1):**
```
conda run -n hsi python -c "import pytest, matplotlib; print(pytest.__version__, matplotlib.__version__)"
```
If pytest is missing: `conda run -n hsi pip install pytest`.

---

### Task 1: Shared `Debouncer` (release-triggered recompute)

**Files:**
- Create: `debug_common.py`
- Test: `tests/test_debug_common.py`

**Interfaces:**
- Produces:
  - `class Debouncer` with `__init__(self, canvas, recompute)`, `mark_dirty(self) -> None`, and internal `_on_release(self, event=None) -> None`.
  - Contract: `mark_dirty()` sets a pending flag; the next `button_release_event` (routed to `_on_release`) calls `recompute()` exactly once and clears the flag. Releases with no pending change do nothing.

- [ ] **Step 1: Write the failing test**

`tests/test_debug_common.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hsi pytest tests/test_debug_common.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'debug_common'`).

- [ ] **Step 3: Write minimal implementation**

`debug_common.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hsi pytest tests/test_debug_common.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add debug_common.py tests/test_debug_common.py
git commit -m "feat: add shared Debouncer for release-triggered recompute"
```

---

### Task 2: Vectorize component sizes in `pieces.py`

**Files:**
- Modify: `hsi_workflow/pieces.py` (`label_pieces` at lines 173-196; add `component_sizes`)
- Test: `tests/test_pieces_sizes.py`

**Interfaces:**
- Produces:
  - `component_sizes(labels: np.ndarray) -> np.ndarray` — returns `counts` where `counts[i]` is the pixel count of label `i` (index 0 = background). Length is `labels.max() + 1`.
  - `label_pieces` unchanged signature `(mask, cfg) -> (labels, kept_ids)`, now using `component_sizes` instead of a per-label Python loop.

- [ ] **Step 1: Write the failing test**

`tests/test_pieces_sizes.py`:
```python
import numpy as np
from scipy import ndimage as ndi

from hsi_workflow.pieces import component_sizes, label_pieces
from hsi_workflow.config import PieceConfig


def test_component_sizes_counts_each_label():
    labels = np.array([[0, 1, 1],
                       [2, 2, 1],
                       [2, 0, 0]])
    sizes = component_sizes(labels)
    assert sizes[0] == 4   # background
    assert sizes[1] == 3
    assert sizes[2] == 3


def test_label_pieces_keeps_only_large_components():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:4, 1:4] = True     # 9 px
    mask[8, 8] = True         # 1 px speck
    cfg = PieceConfig(min_area=5, open_iter=0, close_iter=0,
                      fill_holes=False, watershed_split=False)
    labels, kept = label_pieces(mask, cfg)
    # the 9-px block is kept, the speck is dropped
    assert len(kept) == 1
    big = kept[0]
    assert int((labels == big).sum()) == 9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hsi pytest tests/test_pieces_sizes.py -v`
Expected: FAIL (`ImportError: cannot import name 'component_sizes'`).

- [ ] **Step 3: Write minimal implementation**

In `hsi_workflow/pieces.py`, add above `label_pieces`:
```python
def component_sizes(labels: np.ndarray) -> np.ndarray:
    """Pixel count per label id. ``sizes[i]`` = size of label ``i`` (0 = background)."""
    return np.bincount(labels.ravel())
```

Replace the `label_pieces` keep-loop (current lines 192-196):
```python
    sizes = component_sizes(labels)
    kept = [lbl for lbl in range(1, sizes.size) if sizes[lbl] >= cfg.min_area]
    return labels, kept
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hsi pytest tests/test_pieces_sizes.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hsi_workflow/pieces.py tests/test_pieces_sizes.py
git commit -m "perf: vectorize component-size counting in label_pieces"
```

---

### Task 3: masks — fix import + perf refactor (persistent artists, debounce, de-spam)

**Files:**
- Modify: `debug_masks.py`
- Test: `tests/test_debug_masks.py`

**Interfaces:**
- Consumes: `Debouncer` (Task 1), `component_sizes` (Task 2).
- Produces: `MaskTuner` gains `self._debouncer`, a light-tier `_on_band` that updates only the band-image artist, and heavy callbacks that call `self._debouncer.mark_dirty()`. Persistent artists: `self._im_band`, `self._overlay`, `self._im_dist`, `self._im_lab`, `self._roi_coll` (a `matplotlib.collections.PatchCollection`).

This task preserves existing behavior/features; only import correctness and responsiveness change.

- [ ] **Step 1: Write the failing test**

`tests/test_debug_masks.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hsi pytest tests/test_debug_masks.py -v`
Expected: FAIL — first at import (`No module named 'hsi_workflow.io'`).

- [ ] **Step 3: Implement**

In `debug_masks.py`:

(a) Fix the import (line 41):
```python
from hsi_workflow.cube_io import load_cube, iter_cube_paths
```

(b) Add near the other imports:
```python
from matplotlib.collections import PatchCollection
from debug_common import Debouncer
```

(c) In `MaskTuner.__init__`, after `self._build_figure()` and before `self._recompute()`, create the debouncer:
```python
        self._debouncer = Debouncer(self.fig.canvas, self._recompute)
```

(d) Rewrite `_build_figure`'s artist creation so the three images + overlay + ROI collection are created once. At the end of `_build_figure`, add:
```python
        self._im_band = self._im_dist = self._im_lab = None
        self._overlay = None
        self._roi_coll = None
```

(e) Replace `_redraw` with a version that updates persistent artists and never prints:
```python
    def _redraw(self):
        band = self.cube[:, :, self.band]
        step = max(1, int(np.ceil(max(band.shape) / MAX_DISPLAY)))
        sl = (slice(None, None, step), slice(None, None, step))
        extent = (0, band.shape[1], band.shape[0], 0)

        # panel 0: band image + mask overlay
        ax = self.axes[0]
        if self._im_band is None:
            ax.clear(); ax.axis("off")
            self._im_band = ax.imshow(band[sl], cmap="gray", extent=extent)
            self._overlay = ax.imshow(
                np.ma.masked_invalid(np.where(self.mask[sl], 1.0, np.nan)),
                cmap="autumn", alpha=0.35, vmin=0, vmax=1, extent=extent)
        else:
            self._im_band.set_data(band[sl])
            self._overlay.set_data(
                np.ma.masked_invalid(np.where(self.mask[sl], 1.0, np.nan)))
        self._overlay.set_visible(self.show_mask)
        cov = self.mask.mean()
        ax.set_title(f"band {self.band} ({self.wl[self.band]:.0f} nm) + mask "
                     f"({cov:.1%} fg)", fontsize=10)

        # panel 1: distance map
        ax = self.axes[1]
        if self._im_dist is None:
            ax.clear(); ax.axis("off")
            self._im_dist = ax.imshow(self.dist[sl], cmap="magma", extent=extent)
        else:
            self._im_dist.set_data(self.dist[sl])
            self._im_dist.set_clim(float(self.dist.min()), float(self.dist.max()))
        ax.set_title(f"foreground distance ({self.piece_cfg.method}, "
                     f"{self.piece_cfg.threshold})", fontsize=10)

        # panel 2: labeled pieces + ROI grid
        ax = self.axes[2]
        lab = np.where(self.labels[sl] > 0, self.labels[sl], np.nan)
        cm = plt.get_cmap("tab10").copy(); cm.set_bad("0.12")
        if self._im_lab is None:
            ax.clear(); ax.axis("off"); ax.set_facecolor("0.12")
            self._im_lab = ax.imshow(np.ma.masked_invalid(lab % 10), cmap=cm,
                                     vmin=0, vmax=9, interpolation="nearest",
                                     extent=extent)
        else:
            self._im_lab.set_data(np.ma.masked_invalid(lab % 10))
        if self._roi_coll is not None:
            self._roi_coll.remove()
        p = self.roi_cfg.patch
        rects = [plt.Rectangle((c, r), p, p) for (r, c) in self.roi_boxes]
        self._roi_coll = PatchCollection(rects, facecolor="none",
                                         edgecolor="white", linewidth=0.6)
        ax.add_collection(self._roi_coll)
        n_rois = sum(self.roi_counts.values())
        ax.set_title(f"{len(self.kept)} piece(s), {n_rois} ROI(s) "
                     f"[patch {p}, stride {self.roi_cfg.stride}]", fontsize=10)

        self.fig.canvas.draw_idle()
```

(f) In `_recompute`, replace the per-label size loop that fed the removed print. Remove the `print(f"pieces: ...")` block entirely (current lines 218-221). If a size summary is still wanted, compute it with `component_sizes(self.labels)` but do not print on every recompute.

(g) Make `_on_band` light — it must only refresh the band image, never recompute the mask. Replace it with:
```python
    def _on_band(self, b):
        self.band = int(b)
        band = self.cube[:, :, self.band]
        step = max(1, int(np.ceil(max(band.shape) / MAX_DISPLAY)))
        sl = (slice(None, None, step), slice(None, None, step))
        if self._im_band is not None:
            self._im_band.set_data(band[sl])
            self.axes[0].set_title(
                f"band {self.band} ({self.wl[self.band]:.0f} nm) + mask "
                f"({self.mask.mean():.1%} fg)", fontsize=10)
            self.fig.canvas.draw_idle()
```

(h) Make heavy callbacks debounced. Change `_on_param`, and the body of the slider wiring so heavy sliders mark dirty instead of recomputing. Replace `_on_param` with:
```python
    def _on_param(self, _):
        self.piece_cfg = replace(
            self.piece_cfg,
            threshold_percentile=float(self.s_pct.val),
            open_iter=int(self.s_open.val), close_iter=int(self.s_close.val),
            min_area=int(self.s_area.val))
        patch = int(self.s_patch.val)
        self.roi_cfg = replace(self.roi_cfg, patch=patch,
                               stride=max(1, int(self.s_stride.val)),
                               min_coverage=float(self.s_cov.val))
        self._debouncer.mark_dirty()
```
`_on_method` and `_on_thresh` (radio clicks — discrete) keep calling `self._recompute()` directly.

Note: `add_collection` will fail if `self.roi_boxes` is empty because `PatchCollection([])` is valid but empty; guard is unnecessary. Verify in Step 4.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n hsi pytest tests/test_debug_masks.py -v`
Expected: PASS (all three).

Then smoke-test the real entrypoint imports cleanly:
Run: `conda run -n hsi python -c "import debug_masks"`
Expected: no output, exit 0.

- [ ] **Step 5: Commit**

```bash
git add debug_masks.py tests/test_debug_masks.py
git commit -m "fix: repair cube_io import and debounce/persist masks tuner"
```

---

### Task 4: masks — mask value-window RangeSlider

**Files:**
- Modify: `debug_masks.py`
- Test: `tests/test_debug_masks.py` (add tests)

**Interfaces:**
- Consumes: Task 3 artifacts.
- Produces: `MaskTuner` gains `self.s_range` (`RangeSlider`) and computes the raw mask as `(dist >= lo) & (dist <= hi)` from `self.s_range.val`; the otsu/percentile radio snaps the low handle. New helper `self._reset_range_bounds(dist)`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_debug_masks.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hsi pytest tests/test_debug_masks.py::test_value_window_limits_mask_to_range -v`
Expected: FAIL (`AttributeError: 'MaskTuner' object has no attribute 's_range'`).

- [ ] **Step 3: Implement**

(a) Import at top: `from matplotlib.widgets import Slider, RadioButtons, RangeSlider`.

(b) In `_build_figure`, remove the `s_pct` percentile `Slider` and add a `RangeSlider` in its place (reuse the `0.08, 0.22` row):
```python
        ax_rng = self.fig.add_axes([0.08, 0.22, 0.28, 0.025])
        d = self.dist_for_init if hasattr(self, "dist_for_init") else None
        lo0, hi0 = (0.0, 1.0)
        self.s_range = RangeSlider(ax_rng, "mask window", 0.0, 1.0,
                                   valinit=(lo0, hi0))
        self.s_range.on_changed(self._on_range)
```
Remove `self.s_pct` and its `on_changed` wiring. In `_on_param`, drop the `threshold_percentile=float(self.s_pct.val)` line (percentile is now snapped, not slider-driven).

(c) Add the range callback and a bounds reset:
```python
    def _reset_range_bounds(self, dist):
        lo, hi = float(dist.min()), float(dist.max())
        if hi <= lo:
            hi = lo + 1e-9
        self.s_range.valmin = lo
        self.s_range.valmax = hi
        self.s_range.ax.set_xlim(lo, hi)
        cur = self.s_range.val
        self.s_range.set_val((max(lo, min(cur[0], hi)),
                              max(lo, min(cur[1], hi))))

    def _on_range(self, _):
        self._debouncer.mark_dirty()
```

(d) Compute the mask from the window in `_recompute`. Replace the `mask = _threshold_mask(dist, self.piece_cfg)` line with:
```python
        lo, hi = self.s_range.val
        mask = (dist >= lo) & (dist <= hi)
```
(Keep `_threshold_mask` imported for the radio snap below; it's still used to derive the snap cutoff.)

(e) After the distance is (re)computed in `_recompute`, keep the range bounds in sync when they are stale (method/reference change resets them). At the top of `_recompute`, after `dist = self._distance()`:
```python
        if not np.isclose(self.s_range.valmax, float(dist.max())):
            self._reset_range_bounds(dist)
```

(f) Make the threshold radio snap the low handle. Replace `_on_thresh`:
```python
    def _on_thresh(self, label):
        self.piece_cfg = replace(self.piece_cfg, threshold=label)
        dist = self._distance()
        if label == "otsu":
            from skimage.filters import threshold_otsu
            cutoff = float(threshold_otsu(dist))
        else:
            cutoff = float(np.percentile(dist, self.piece_cfg.threshold_percentile))
        _, hi = self.s_range.val
        self.s_range.set_val((min(cutoff, hi), hi))
        self._recompute()
```

(g) On init, after the first `_recompute` sets `self.dist`, call `self._reset_range_bounds(self.dist)` and snap low to the default otsu cutoff so the first frame is sensible. Do this at the end of `__init__`:
```python
        self._reset_range_bounds(self.dist)
        self._on_thresh(self.piece_cfg.threshold)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n hsi pytest tests/test_debug_masks.py -v`
Expected: PASS (all, including the new window test).

- [ ] **Step 5: Commit**

```bash
git add debug_masks.py tests/test_debug_masks.py
git commit -m "feat: mask value-window RangeSlider with otsu/percentile snap"
```

---

### Task 5: masks — display-contrast RangeSliders (band + distance)

**Files:**
- Modify: `debug_masks.py`
- Test: `tests/test_debug_masks.py` (add test)

**Interfaces:**
- Produces: `self.s_band_clip`, `self.s_dist_clip` (`RangeSlider`, fractional 0-1), `self.band_clip`, `self.dist_clip` tuples, and `_apply_band_contrast()` / `_apply_dist_contrast()` light-tier handlers that call `im.set_clim` without recomputing.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_debug_masks.py`:
```python
def test_band_contrast_sets_clim_without_recompute():
    t = _tuner()
    t._debouncer._recompute = lambda: (_ for _ in ()).throw(
        AssertionError("contrast must not recompute"))
    t.s_band_clip.set_val((0.2, 0.8))
    t._on_band_clip(None)
    lo, hi = t._im_band.get_clim()
    assert lo < hi
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hsi pytest tests/test_debug_masks.py::test_band_contrast_sets_clim_without_recompute -v`
Expected: FAIL (`AttributeError: ... 's_band_clip'`).

- [ ] **Step 3: Implement**

(a) In `_build_figure`, add two fractional range sliders (place them on free rows near the bottom-left, e.g. y=0.06 and y=0.02):
```python
        ax_bc = self.fig.add_axes([0.08, 0.06, 0.28, 0.02])
        self.s_band_clip = RangeSlider(ax_bc, "band contrast", 0.0, 1.0, valinit=(0.0, 1.0))
        self.s_band_clip.on_changed(self._on_band_clip)
        ax_dc = self.fig.add_axes([0.08, 0.02, 0.28, 0.02])
        self.s_dist_clip = RangeSlider(ax_dc, "dist contrast", 0.0, 1.0, valinit=(0.0, 1.0))
        self.s_dist_clip.on_changed(self._on_dist_clip)
        self.band_clip = (0.0, 1.0)
        self.dist_clip = (0.0, 1.0)
```

(b) Add handlers (light — no debounce, no recompute):
```python
    def _clim_from_frac(self, data, frac):
        dmin, dmax = float(np.nanmin(data)), float(np.nanmax(data))
        span = dmax - dmin or 1e-9
        return dmin + frac[0] * span, dmin + frac[1] * span

    def _on_band_clip(self, _):
        self.band_clip = tuple(self.s_band_clip.val)
        if self._im_band is not None:
            lo, hi = self._clim_from_frac(self.cube[:, :, self.band], self.band_clip)
            self._im_band.set_clim(lo, max(hi, lo + 1e-9))
            self.fig.canvas.draw_idle()

    def _on_dist_clip(self, _):
        self.dist_clip = tuple(self.s_dist_clip.val)
        if self._im_dist is not None:
            lo, hi = self._clim_from_frac(self.dist, self.dist_clip)
            self._im_dist.set_clim(lo, max(hi, lo + 1e-9))
            self.fig.canvas.draw_idle()
```

(c) Apply the band-contrast fraction inside `_on_band` and `_redraw` after `set_data` on `self._im_band` so stepping bands keeps the chosen contrast:
```python
            lo, hi = self._clim_from_frac(self.cube[:, :, self.band], self.band_clip)
            self._im_band.set_clim(lo, max(hi, lo + 1e-9))
```
Apply the dist-contrast fraction in `_redraw` after `self._im_dist.set_data(...)` (replacing the unconditional `set_clim(min,max)` added in Task 3):
```python
            lo, hi = self._clim_from_frac(self.dist, self.dist_clip)
            self._im_dist.set_clim(lo, max(hi, lo + 1e-9))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n hsi pytest tests/test_debug_masks.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add debug_masks.py tests/test_debug_masks.py
git commit -m "feat: display-contrast range sliders for masks band/distance panels"
```

---

### Task 6: masks — reference region for distance (RectangleSelector)

**Files:**
- Modify: `debug_masks.py`
- Test: `tests/test_debug_masks.py` (add tests)

**Interfaces:**
- Produces:
  - Module function `_distance_from_reference(cube, cfg, ref_mask) -> np.ndarray` (same `(rows, cols)` shape as `foreground_distance`).
  - `MaskTuner` state: `self._ref_mask` (bool array or `None`), `self._ref_version` (int). `_distance()` cache keys on `(method, ref_version-or-None)`. Keys `'R'` arms a `RectangleSelector`; `'c'` clears the reference.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_debug_masks.py`:
```python
from debug_masks import _distance_from_reference
from hsi_workflow.config import PieceConfig


def test_distance_from_reference_shape_and_low_on_reference():
    cube, wl = synthetic_cube(rows=60, cols=60, bands=20, seed=2)
    ref = np.zeros(cube.shape[:2], dtype=bool)
    ref[:8, :8] = True                      # a background corner
    cfg = PieceConfig(method="sam")
    dist = _distance_from_reference(cube, cfg, ref)
    assert dist.shape == cube.shape[:2]
    # pixels inside the reference are, on average, closer to the reference
    assert dist[ref].mean() <= dist[~ref].mean()


def test_setting_reference_invalidates_distance_cache():
    t = _tuner()
    d0 = t._distance().copy()
    t._ref_mask = np.zeros(t.cube.shape[:2], dtype=bool)
    t._ref_mask[:8, :8] = True
    t._ref_version += 1
    d1 = t._distance()
    assert not np.array_equal(d0, d1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hsi pytest tests/test_debug_masks.py::test_distance_from_reference_shape_and_low_on_reference -v`
Expected: FAIL (`ImportError: cannot import name '_distance_from_reference'`).

- [ ] **Step 3: Implement**

(a) Imports: `from matplotlib.widgets import Slider, RadioButtons, RangeSlider, RectangleSelector` and `from hsi_workflow.pieces import (foreground_distance, _threshold_mask, clean_mask, label_pieces, component_sizes, spectral_angle, _mahalanobis_to_background)`.

(b) Module-level function:
```python
def _distance_from_reference(cube, cfg, ref_mask):
    """Foreground distance measured against a user-drawn reference region.

    ``sam`` uses the reference mean spectrum; ``mahalanobis`` uses the reference
    pixels' mean/covariance. ``kmeans`` is unsupervised and ignores the reference
    (falls back to the standard border-background distance).
    """
    rows, cols, bands = cube.shape
    flat = cube.reshape(-1, bands)
    ref_pixels = cube[ref_mask]
    if ref_pixels.shape[0] < 2:
        return foreground_distance(cube, cfg)
    if cfg.method == "mahalanobis":
        dist = _mahalanobis_to_background(flat, ref_pixels)
    elif cfg.method == "kmeans":
        return foreground_distance(cube, cfg)
    else:  # sam and default
        dist = spectral_angle(flat, ref_pixels.mean(axis=0))
    return dist.reshape(rows, cols)
```

(c) `__init__`: initialize `self._ref_mask = None`, `self._ref_version = 0`, `self._selector = None` before `_build_figure`.

(d) Replace `_distance` to key on the reference:
```python
    def _distance(self):
        m = self.piece_cfg.method
        ref_key = self._ref_version if self._ref_mask is not None else None
        key = (m, ref_key)
        if key not in self._dist_cache:
            print(f"Computing foreground distance ({m}, ref={ref_key}) ... ",
                  end="", flush=True)
            if self._ref_mask is not None:
                self._dist_cache[key] = _distance_from_reference(
                    self.cube, self.piece_cfg, self._ref_mask)
            else:
                self._dist_cache[key] = foreground_distance(self.cube, self.piece_cfg)
            print("done")
        return self._dist_cache[key]
```

(e) In `_on_key`, add reference controls:
```python
        elif event.key == "R":
            self._arm_reference_selector()
        elif event.key == "c":
            self._ref_mask = None
            self._recompute()
```

(f) Add the selector setup:
```python
    def _arm_reference_selector(self):
        def on_select(eclick, erelease):
            r0, r1 = sorted((int(eclick.ydata), int(erelease.ydata)))
            c0, c1 = sorted((int(eclick.xdata), int(erelease.xdata)))
            m = np.zeros(self.cube.shape[:2], dtype=bool)
            m[r0:r1 + 1, c0:c1 + 1] = True
            if m.sum() >= 2:
                self._ref_mask = m
                self._ref_version += 1
                self._recompute()
        self._selector = RectangleSelector(
            self.axes[0], on_select, useblit=True, button=[1],
            minspanx=3, minspany=3, spancoords="data", interactive=False)
        print("reference selector armed: drag a box on the band image "
              "('c' to clear)")
```

(g) Update the on-screen help text (the `self.fig.text(...)` in `_build_figure`) to mention `'R' = ref box  'c' = clear ref`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n hsi pytest tests/test_debug_masks.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add debug_masks.py tests/test_debug_masks.py
git commit -m "feat: reference-region distance for masks tuner"
```

---

### Task 7: preprocess — perf refactor (persistent artists, debounce, remove dead arg)

**Files:**
- Modify: `debug_preprocess.py`
- Test: `tests/test_debug_preprocess.py`

**Interfaces:**
- Consumes: `Debouncer` (Task 1).
- Produces: `PreprocessTuner` gains `self._debouncer`; heavy callbacks (`_on_param`, `_on_check`) mark dirty; light `_on_band` updates only the band image + spectrum. Persistent artists: `self._im` (band image), `self._line_before`, `self._line_after`, `self._txt` (metrics). `_recompute` loses the unused `full=` argument.

- [ ] **Step 1: Write the failing test**

`tests/test_debug_preprocess.py`:
```python
import matplotlib
matplotlib.use("Agg")

import numpy as np
from debug_preprocess import PreprocessTuner, synthetic_cube


def _tuner():
    cube, wl = synthetic_cube(rows=60, cols=60, bands=40, seed=1)
    # no white/dark -> calibrate disabled, raw path
    return PreprocessTuner(cube, wl, 1.0, None, None, 1.0, 1.0, "test")


def test_heavy_param_is_debounced():
    t = _tuner()
    calls = []
    t._debouncer._recompute = lambda: calls.append(1)
    t._on_param(None)
    assert calls == []
    t._debouncer._on_release(None)
    assert calls == [1]


def test_band_step_does_not_recompute():
    t = _tuner()
    t._debouncer._recompute = lambda: (_ for _ in ()).throw(
        AssertionError("band step must not recompute"))
    t._on_band(7)
    assert t.band == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hsi pytest tests/test_debug_preprocess.py -v`
Expected: FAIL (`AttributeError: ... '_debouncer'`).

- [ ] **Step 3: Implement**

(a) Import: `from debug_common import Debouncer`.

(b) `_recompute` signature: change `def _recompute(self, full=False):` to `def _recompute(self):` and drop the `full=True` argument at its call site in `__init__` (`self._recompute()`).

(c) In `__init__`, after `_build_figure()` and before the first `_recompute()`:
```python
        self._debouncer = Debouncer(self.fig.canvas, self._recompute)
```

(d) Persist the band image + colorbar. In `_update_band_image`, create `self._im` once:
```python
    def _update_band_image(self):
        band = self.display_cube[:, :, self.band]
        if getattr(self, "_im", None) is None:
            self._im = self.ax_img.imshow(
                band, cmap="magma",
                extent=(0, self.raw.shape[1], self.raw.shape[0], 0))
            self._cbar = self.fig.colorbar(self._im, ax=self.ax_img, fraction=0.046)
            self._pixmark, = self.ax_img.plot([], [], "c+", ms=14, mew=2)
        else:
            self._im.set_data(band)
            self._im.set_clim(float(np.nanmin(band)), float(np.nanmax(band)))
        r, c = self.pixel
        self._pixmark.set_data([c], [r])
        self.ax_img.set_title(f"band {self.band}  ({self.wl[self.band]:.0f} nm)  "
                              f"[decimation x{self.step}]", fontsize=11)
```
Remove the old `self.ax_img.clear()` at the top of that method.

(e) Make `_on_param` and `_on_check` mark dirty instead of recompute:
```python
    def _on_param(self, _):
        self.sg_window = int(self.s_window.val) | 1
        self.sg_polyorder = min(int(self.s_poly.val), self.sg_window - 1)
        self.baseline_order = int(self.s_base.val)
        self._debouncer.mark_dirty()
```
In `_on_check`, keep the branch logic but replace the trailing `self._recompute()` with `self._debouncer.mark_dirty()`. (Check clicks are single events, but marking dirty + relying on the click's own release is fine; if the checkbox does not emit a release, call `self._recompute()` directly instead — verify in Step 4 and use whichever fires. Default: `self._recompute()` directly for checks since a click is discrete.)

Decision: checks are discrete → keep `self._recompute()` in `_on_check`. Only slider drags use the debouncer.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n hsi pytest tests/test_debug_preprocess.py -v`
Expected: PASS.

Smoke: `conda run -n hsi python -c "import debug_preprocess"` → exit 0.

- [ ] **Step 5: Commit**

```bash
git add debug_preprocess.py tests/test_debug_preprocess.py
git commit -m "fix: debounce and persist artists in preprocess tuner"
```

---

### Task 8: preprocess — display-contrast RangeSlider

**Files:**
- Modify: `debug_preprocess.py`
- Test: `tests/test_debug_preprocess.py` (add test)

**Interfaces:**
- Produces: `self.s_clip` (`RangeSlider`, fractional 0-1), `self.band_clip`, `_on_clip` light handler using `self._im.set_clim`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_debug_preprocess.py`:
```python
def test_contrast_sets_clim_without_recompute():
    t = _tuner()
    t._debouncer._recompute = lambda: (_ for _ in ()).throw(
        AssertionError("contrast must not recompute"))
    t.s_clip.set_val((0.1, 0.9))
    t._on_clip(None)
    lo, hi = t._im.get_clim()
    assert lo < hi
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hsi pytest tests/test_debug_preprocess.py::test_contrast_sets_clim_without_recompute -v`
Expected: FAIL (`AttributeError: ... 's_clip'`).

- [ ] **Step 3: Implement**

(a) Import `RangeSlider`: `from matplotlib.widgets import Slider, CheckButtons, RangeSlider`.

(b) In `_build_figure`, add a contrast slider (use a free row, e.g. below the checks column or at `[0.55, 0.22, 0.14, 0.03]`):
```python
        ax_clip = self.fig.add_axes([0.55, 0.24, 0.14, 0.03])
        self.s_clip = RangeSlider(ax_clip, "contrast", 0.0, 1.0, valinit=(0.0, 1.0))
        self.s_clip.on_changed(self._on_clip)
        self.band_clip = (0.0, 1.0)
```

(c) Add handler + apply in `_update_band_image`:
```python
    def _on_clip(self, _):
        self.band_clip = tuple(self.s_clip.val)
        if getattr(self, "_im", None) is not None:
            band = self.display_cube[:, :, self.band]
            lo, hi = self._clim_from_frac(band, self.band_clip)
            self._im.set_clim(lo, max(hi, lo + 1e-9))
            self.fig.canvas.draw_idle()

    def _clim_from_frac(self, data, frac):
        dmin, dmax = float(np.nanmin(data)), float(np.nanmax(data))
        span = dmax - dmin or 1e-9
        return dmin + frac[0] * span, dmin + frac[1] * span
```
In `_update_band_image`, replace the unconditional `set_clim(min,max)` from Task 7 with the fractional version:
```python
            lo, hi = self._clim_from_frac(band, self.band_clip)
            self._im.set_clim(lo, max(hi, lo + 1e-9))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n hsi pytest tests/test_debug_preprocess.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add debug_preprocess.py tests/test_debug_preprocess.py
git commit -m "feat: display-contrast range slider for preprocess band image"
```

---

### Task 9: preprocess — reference spectrum + subtract

**Files:**
- Modify: `debug_preprocess.py`
- Test: `tests/test_debug_preprocess.py` (add tests)

**Interfaces:**
- Produces:
  - `self.ref_spectrum` (`np.ndarray` of length `bands`, or `None`), `self.use_ref_subtract` (bool), a 5th checkbutton `"subtract ref"`.
  - `_process` subtracts `self.ref_spectrum` (broadcast over the band axis) right after `_src`, before smoothing, when `use_ref_subtract` and a reference is set.
  - `shift+click` sets the reference from a 5x5 window average; `'c'` clears it.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_debug_preprocess.py`:
```python
def test_reference_subtract_zeroes_reference_pixel():
    t = _tuner()
    r, c = 30, 30
    # set reference from the pixel itself
    t._set_reference(r, c)
    t.use_ref_subtract = True
    sel = (slice(r, r + 1), slice(c, c + 1), slice(None))
    # disable smoothing/snv/baseline to isolate the subtraction
    t.use_smooth = t.use_snv = t.use_baseline = False
    out = t._process(sel)[0, 0, :]
    # subtracting the (5x5-averaged) reference leaves a near-zero residual
    assert np.abs(out).mean() < np.abs(t._src(sel)[0, 0, :]).mean()


def test_reference_subtract_off_by_default():
    t = _tuner()
    assert t.ref_spectrum is None
    assert t.use_ref_subtract is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hsi pytest tests/test_debug_preprocess.py::test_reference_subtract_off_by_default -v`
Expected: FAIL (`AttributeError: ... 'ref_spectrum'`).

- [ ] **Step 3: Implement**

(a) `__init__`: add state near the other flags:
```python
        self.ref_spectrum = None
        self.use_ref_subtract = False
```

(b) Add the reference setter (5x5 average around the pixel, in the current "before" domain):
```python
    def _set_reference(self, r, c):
        r0, r1 = max(0, r - 2), min(self.raw.shape[0], r + 3)
        c0, c1 = max(0, c - 2), min(self.raw.shape[1], c + 3)
        sel = (slice(r0, r1), slice(c0, c1), slice(None))
        self.ref_spectrum = self._src(sel).reshape(-1, self.bands).mean(axis=0)
```

(c) Apply subtraction in `_process` (first thing after `_src`):
```python
    def _process(self, sel):
        data = self._src(sel)
        if self.use_ref_subtract and self.ref_spectrum is not None:
            data = data - self.ref_spectrum
        if self.use_smooth:
            data = savgol_smooth(data, self.sg_window, self.sg_polyorder)
        if self.use_baseline:
            data = baseline_correct(data, "poly", self.baseline_order)
        if self.use_snv:
            data = normalize_intensity(data, "snv")
        return data
```

(d) Add the 5th checkbutton. In `_build_figure`:
```python
        labels = ["calibrate", "SG smooth", "SNV", "baseline", "subtract ref"]
        state = [self.use_calibrate, self.use_smooth, self.use_snv,
                 self.use_baseline, self.use_ref_subtract]
```
In `_on_check`, add:
```python
        elif label == "subtract ref":
            if self.ref_spectrum is None:
                print("No reference set. shift+click a pixel first.")
                return
            self.use_ref_subtract = not self.use_ref_subtract
```

(e) shift+click sets the reference. In `_on_click`, branch on the modifier:
```python
    def _on_click(self, event):
        if event.inaxes is not self.ax_img or event.xdata is None:
            return
        c, r = int(event.xdata), int(event.ydata)
        if not (0 <= r < self.raw.shape[0] and 0 <= c < self.raw.shape[1]):
            return
        if event.key == "shift":
            self._set_reference(r, c)
            print(f"reference set from 5x5 around ({r}, {c})")
            self._update_spectrum()
            self.fig.canvas.draw_idle()
            return
        self.pixel = (r, c)
        self._update_band_image()
        self._update_spectrum()
        self.fig.canvas.draw_idle()
```

(f) Draw the reference on the spectrum panel. In `_update_spectrum`, after plotting `before`, add:
```python
        if self.ref_spectrum is not None:
            self.ax_spec.plot(self.wl, self.ref_spectrum, color="tab:green",
                              lw=0.9, ls=":", label="reference")
```

(g) `'c'` clears the reference. In `_on_key`, add:
```python
        elif event.key == "c":
            self.ref_spectrum = None
            self.use_ref_subtract = False
            self._recompute()
```

(h) `'p'` printout: add a comment note when reference-subtract is on. In `_on_key`'s `'p'` branch, before the final blank print:
```python
            if self.use_ref_subtract and self.ref_spectrum is not None:
                print("#   (debug-only: a reference spectrum was subtracted; "
                      "not part of PreprocessConfig)")
```

(i) Update the help text (`self.fig.text(...)`) to mention `shift+click = set ref` and `'c' = clear ref`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n hsi pytest tests/test_debug_preprocess.py -v`
Expected: PASS (all).

Full suite: `conda run -n hsi pytest tests/ -v` → all green.

- [ ] **Step 5: Commit**

```bash
git add debug_preprocess.py tests/test_debug_preprocess.py
git commit -m "feat: reference spectrum subtract in preprocess tuner"
```

---

## Self-Review

**Spec coverage:**
- Perf: debounce (Tasks 3, 7), persistent artists (3, 7), vectorized sizes (2), de-spam (3), dead `full=` arg (7). ✓
- Critical import bug (`hsi_workflow.io`): Task 3. ✓
- Range — mask value-window: Task 4. Range — display contrast both tools: Tasks 5, 8. ✓
- Reference & subtract — masks region distance: Task 6. Preprocess subtract: Task 9. ✓
- Debug-only (no `PreprocessConfig` change): Task 9 note + `'p'` comment. ✓
- `--demo` still works: import smoke tests in Tasks 3, 7; synthetic cubes used throughout. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; commands have expected output. ✓

**Type consistency:** `component_sizes` (Task 2) consumed in Tasks 3/6; `Debouncer._recompute`/`mark_dirty`/`_on_release` (Task 1) used consistently in Tasks 3, 5, 7, 8; `_distance_from_reference(cube, cfg, ref_mask)` signature matches its Task-6 test; `_clim_from_frac(data, frac)` defined and used in Tasks 5 and 8; `_set_reference(r, c)`/`ref_spectrum`/`use_ref_subtract` consistent in Task 9. ✓
