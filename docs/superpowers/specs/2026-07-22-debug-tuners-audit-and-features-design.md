# Debug Tuners: Performance Audit + Range/Reference Features

**Date:** 2026-07-22
**Files:** `debug_masks.py`, `debug_preprocess.py`, new `debug_common.py`
**Status:** Approved design

## Problem

Both interactive matplotlib tuners (`debug_masks.py`, `debug_preprocess.py`) are
slow, laggy, and buggy. The user also wants two new capabilities in both tools:
a min/max **range** control (especially for masks) and a **reference + subtract**
capability.

## Audit findings (root causes)

1. **Sliders recompute the full heavy pipeline on every drag tick.**
   Matplotlib `Slider.on_changed` fires continuously while dragging. Each fire runs:
   - *masks*: `_threshold_mask` + `clean_mask` (`binary_opening`/`closing`, up to
     8-15 iterations on the full-res mask) + `label_pieces` + the Python ROI
     double-loop + a full 3-panel `ax.clear()`+`imshow` redraw + a `print`.
   - *preprocess*: reprocesses the whole decimated cube, then `_update_metrics`
     runs `noise_metrics` **twice** (each does its own savgol on 3000 px) plus a
     third savgol.
   This is the dominant lag source.

2. **Per-label Python loops for component sizes.** `label_pieces` and the masks
   `_redraw` compute `(labels == lbl).sum()` in a loop per label — O(labels x pixels).
   The redraw recomputes these on *every* band step.

3. **`ax.clear()` + fresh `imshow` on every redraw** instead of updating a
   persistent artist. Band-stepping in masks redraws all 3 panels and re-adds
   every ROI `Rectangle`.

4. **`print(...)` on every redraw** — terminal spam that itself adds lag.

5. **Dead code:** preprocess `_recompute(full=...)` — `full` is unused.

## Design

### Component 1 — `debug_common.py` (new, shared)

A single small helper, `Debouncer`, shared by both tools:
- Heavy slider callbacks call `debouncer.mark_dirty()` (stash values, set a flag),
  they do **not** recompute.
- `Debouncer` connects to the figure canvas `button_release_event`; on release, if
  dirty, it calls the tool's `recompute()` **once** and clears the flag.
- Discrete controls (radios, checkbuttons, `RectangleSelector` release) call
  `recompute()` directly — they don't drag.

Keep this module minimal (~20 lines). All domain logic stays in each tool's file.
`RangeSlider` and `RectangleSelector` are used directly from stock matplotlib —
no wrappers, no new dependencies.

### Component 2 — Performance/correctness refactor (both tools)

- **Two-tier updates.**
  - *Light (instant, during drag):* band index, display-contrast vmin/vmax,
    mask-overlay toggle -> update the existing artist (`set_data` / `set_clim` /
    `set_ydata`), no pipeline rerun.
  - *Heavy (on release):* threshold range, morphology iters, min-area, ROI params,
    SG window/poly, baseline, SNV, reference -> debounced via `Debouncer`.
- **Persistent artists.** Each `imshow` / line / text artist is created once and
  updated in place. Band-stepping in masks no longer touches the dist/label panels
  or re-adds ROI rectangles (ROI rectangles held in one persistent collection,
  rebuilt only on heavy recompute).
- **Vectorize sizes.** Replace per-label `.sum()` loops with a single
  `np.bincount(labels.ravel())` in `label_pieces` and the masks redraw path.
- **De-spam.** Remove per-redraw `print`s; keep the on-demand `'p'` config printout.
- **Remove dead `full=` arg** in preprocess `_recompute`.

### Component 3 — Range / min-max (both meanings)

- **Mask value window (masks).** Replace the single `percentile` slider with a
  `RangeSlider` over the distance-map value range: mask = `lo <= dist <= hi`.
  The otsu/percentile radio becomes a convenience that snaps the `lo` handle to
  that cutoff (single-threshold behavior = `hi` at max). RangeSlider bounds reset
  to the new distance min/max whenever the method changes.
- **Display contrast (both).** `RangeSlider`(s) driving `im.set_clim()` on the band
  image (and the distance map in masks). Purely visual, light-tier, never affects
  the mask/segmentation.

### Component 4 — Reference & subtract (both)

- **Preprocess.** `shift+click` sets a reference spectrum (5x5-window average around
  the clicked pixel for stability). A new **"subtract ref"** checkbutton subtracts
  it in the reflectance domain, *before* smooth/baseline/SNV. The reference is drawn
  as a dotted line on the spectrum panel. `'c'` clears the reference.
- **Masks.** Key **`'R'`** arms a `RectangleSelector` on the band image; the drawn
  box's pixels become the background reference, and the foreground distance
  recomputes against that region's mean/covariance instead of the auto border-frame.
  `'c'` reverts to border-background. The distance cache key includes the reference
  identity so it invalidates correctly. A debug-local
  `_distance_from_reference(cube, cfg, ref_mask)` mirrors `foreground_distance` but
  uses the supplied reference pixels (reusing `spectral_angle` /
  `_mahalanobis_to_background`).

## Scope decisions (YAGNI)

- Reference-subtraction is **debug-only**. It is **not** plumbed into
  `PreprocessConfig`; the `'p'` printout mentions it only as a comment note. The
  production pipeline/config is untouched.
- No new dependencies: `RangeSlider` and `RectangleSelector` are stock matplotlib.
- No unrelated refactoring of `hsi_workflow/` beyond the vectorized `label_pieces`
  size count (which directly serves the perf goal).

## Success criteria

- Dragging any slider no longer stutters: heavy work runs once on release; band,
  contrast, and overlay changes are instant.
- No per-redraw terminal spam; `'p'` still prints paste-ready configs.
- Masks: RangeSlider value-window and per-image contrast work; a drawn reference
  box re-derives the distance map.
- Preprocess: contrast range works; shift+click reference + "subtract ref" toggle
  shows correct before/after spectra and metrics.
- Both tools still run under `--demo` with no data on disk.
