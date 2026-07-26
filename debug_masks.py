# debug_masks.py
"""Interactive piece-extraction & ROI tuner for hyperspectral cubes.

Tune the spectral foreground mask (which pixels are sample vs dish) and the ROI
tiling, live, until the pieces and patch grid look right — then press 'p' to
print paste-ready ``PieceConfig(...)`` / ``RoiConfig(...)`` snippets.

Panels
    1. band image (band slider) with the current mask as a red overlay
    2. the foreground *distance* map the mask is thresholded from
    3. labeled pieces (each color = one piece) with the kept ROI grid drawn on top

Controls
    radios    method: sam | mahalanobis | kmeans      threshold: otsu | percentile
    checks    flat field | calibrate | watershed
    sliders   band | mask window | band contrast | dist contrast | open iter |
              close iter | min area | ROI patch | ROI stride | min coverage | flat sigma
    keys      ←/→ = step band, 'm' = toggle mask overlay, 'p' = print configs,
              'R' = draw a clean-background reference box, 'c' = clear the box

Fixing messy extraction (white-dish scan)
    The auto background is the outer border frame; on the two-dish scan that
    frame is contaminated (rim, paper, 2nd dish). Press 'R' and drag a box over a
    clean empty-dish patch — the foreground distance is then measured against
    *those* pixels (this becomes ``PieceConfig.background_bbox``). Add "flat field"
    to divide out the lighting gradient and "calibrate" to work on reflectance.

Performance
    The whole scan is decimated to ``--max-dim`` px on its long axis so it stays
    interactive; the spectral distance map is computed once per (method, background,
    flat-field, calibrate) combination and cached. Thresholding, morphology,
    labeling and ROI tiling rerun instantly on every slider move (debounced to
    mouse-release). Spatial config values printed by 'p' are rescaled back to full
    resolution.

Usage
    python debug_masks.py --dataset sio2_dish_white_20
    python debug_masks.py --dataset sio2_dish_white_20 --crop 0 700 0 700
    python debug_masks.py --hdr "path\\to\\scan.bip.hdr"
    python debug_masks.py --demo
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import (Slider, RadioButtons, RangeSlider, CheckButtons,
                                RectangleSelector)
from matplotlib.collections import PatchCollection

from hsi_workflow.config import DATASETS, PieceConfig, RoiConfig
from hsi_workflow.cube_io import (open_cube_reader, array_cube_reader, CubeReader,
                                  iter_cube_paths, load_reference_spectrum)
from hsi_workflow.preprocessing import calibrate_reflectance
from hsi_workflow.pieces import foreground_distance, clean_mask, label_pieces
from debug_common import Debouncer

MAX_DISPLAY = 500          # display grid cap (px on the long axis)
DEFAULT_MAX_WORK = 700     # working/compute grid cap (px on the long axis)


def synthetic_cube(rows=300, cols=300, bands=100, seed=0):
    """Pieces-on-a-dish phantom: a few rectangles with a distinct spectrum."""
    rng = np.random.default_rng(seed)
    wl = np.linspace(368, 1008, bands)
    dish = 0.15 + 0.02 * np.sin(wl / 90)
    piece = 0.5 + 0.2 * np.sin(wl / 150 + 1.0)
    cube = np.tile(dish, (rows, cols, 1))
    margin = max(2, min(rows, cols) // 10)
    h_lo, h_hi = max(4, rows // 8), max(5, rows // 4)
    w_lo, w_hi = max(4, cols // 8), max(5, cols // 4)
    for _ in range(5):
        h, w = rng.integers(h_lo, h_hi), rng.integers(w_lo, w_hi)
        r = rng.integers(margin, max(margin + 1, rows - h - margin))
        c = rng.integers(margin, max(margin + 1, cols - w - margin))
        cube[r:r + h, c:c + w, :] = piece + rng.normal(0, 0.01, bands)
    cube += rng.normal(0, 0.01, cube.shape)
    return cube, wl


def load_inputs(args):
    if args.demo:
        cube, wl = synthetic_cube(seed=args.seed)
        return cube, wl, "synthetic demo", None, None, 1.0, 1.0, 1.0
    if args.hdr:
        hdr, label = args.hdr, args.hdr
        white_hdr = dark_hdr = None
    else:
        ds = DATASETS[args.dataset.lower()]
        pairs = iter_cube_paths(ds)
        if not pairs:
            raise SystemExit(f"No cubes found for dataset {args.dataset!r} under {ds.data_dir}")
        label, hdr = pairs[args.index]
        white_hdr, dark_hdr = ds.white_ref, ds.dark_ref
        print(f"Loading cube {label!r} ({args.index + 1}/{len(pairs)}) ...")
    crop = tuple(args.crop) if args.crop else None
    reader = open_cube_reader(hdr, crop=crop)
    wl, shutter = reader.wavelengths, reader.shutter
    white = dark = None
    sw = sd = 1.0
    if white_hdr and dark_hdr:
        print("Loading white/dark references (streamed, cached after first use) ...")
        white, sw = load_reference_spectrum(white_hdr)
        dark, sd = load_reference_spectrum(dark_hdr)
    if wl is None:
        wl = np.arange(reader.shape[-1], dtype=float)
    return reader, np.asarray(wl, float), label, white, dark, sw, sd, shutter


class MaskTuner:
    def __init__(self, cube, wl, label, white=None, dark=None, sw=1.0, sd=1.0,
                 shutter=1.0, max_dim=DEFAULT_MAX_WORK):
        self.wl = wl
        self.label = label

        # Backend-agnostic source: an on-disk CubeReader (big scans) or an
        # ndarray wrapped in the same interface (demo/tests).
        source = cube if isinstance(cube, CubeReader) else array_cube_reader(
            cube, wavelengths=wl, shutter=shutter, label=str(label))

        # Decimate the whole scan for interactive compute; 'p' rescales spatial
        # config values by this factor back to full resolution. The decimated
        # grid is streamed off disk -- the full cube is never materialized.
        step = max(1, int(np.ceil(max(source.shape[:2]) / max_dim)))
        self.ds_factor = step
        self._raw_ds = source.decimated(step)
        if white is not None and dark is not None:
            self._refl_ds = calibrate_reflectance(self._raw_ds, shutter, white, sw, dark, sd)
            self.can_calibrate = True
        else:
            self._refl_ds = None
            self.can_calibrate = False
        self.use_calibrate = False
        self.cube = self._raw_ds            # active working cube (raw or reflectance)

        self.bands = self.cube.shape[-1]
        self.band = self.bands // 2
        self.show_mask = True
        self.band_clip = (0.0, 1.0)
        self.dist_clip = (0.0, 1.0)

        self.piece_cfg = PieceConfig()
        self.roi_cfg = RoiConfig()
        self._dist_cache = {}
        self._selector = None

        self._build_figure()
        self._debouncer = Debouncer(self.fig.canvas, self._recompute)
        self._recompute()
        self._reset_range_bounds(self.dist)
        self._on_thresh(self.piece_cfg.threshold)

    # --- pipeline ---------------------------------------------------------

    def _dist_key(self):
        c = self.piece_cfg
        return (c.method, c.background_bbox, c.flat_field, round(float(c.flat_field_sigma), 3),
                self.use_calibrate)

    def _distance(self):
        key = self._dist_key()
        if key not in self._dist_cache:
            print(f"Computing foreground distance {key} ... ", end="", flush=True)
            self._dist_cache[key] = foreground_distance(self.cube, self.piece_cfg)
            print("done")
        return self._dist_cache[key]

    def _recompute(self):
        dist = self._distance()
        if not np.isclose(self.s_range.valmax, float(dist.max())):
            self._reset_range_bounds(dist)
        lo, hi = self.s_range.val
        mask = (dist >= lo) & (dist <= hi)
        mask = clean_mask(mask, self.piece_cfg)
        labels, kept = label_pieces(mask, self.piece_cfg)
        self.dist, self.kept = dist, kept
        self.mask = np.isin(labels, kept)
        self.labels = np.where(self.mask, labels, 0)

        # ROI tiling per piece (grid rectangles + counts)
        self.roi_boxes, self.roi_counts = [], {}
        p, s, mc = self.roi_cfg.patch, self.roi_cfg.stride, self.roi_cfg.min_coverage
        for lbl in kept:
            comp = labels == lbl
            rows = np.any(comp, axis=1); cols = np.any(comp, axis=0)
            r0, r1 = np.where(rows)[0][[0, -1]]; c0, c1 = np.where(cols)[0][[0, -1]]
            sub = comp[r0:r1 + 1, c0:c1 + 1]
            n = 0
            for rr in range(0, sub.shape[0] - p + 1, s):
                for cc in range(0, sub.shape[1] - p + 1, s):
                    if sub[rr:rr + p, cc:cc + p].mean() >= mc:
                        self.roi_boxes.append((r0 + rr, c0 + cc))
                        n += 1
            self.roi_counts[lbl] = n
        self._redraw()

    # --- figure -----------------------------------------------------------

    def _build_figure(self):
        self.fig, self.axes = plt.subplots(1, 3, figsize=(17, 6.8))
        try:
            self.fig.canvas.manager.set_window_title(f"mask tuner — {self.label}")
        except Exception:
            pass
        self.fig.subplots_adjust(left=0.04, right=0.99, top=0.92, bottom=0.34, wspace=0.15)

        def slider(x, y, w, name, lo, hi, init, step=None):
            ax = self.fig.add_axes([x, y, w, 0.022])
            return Slider(ax, name, lo, hi, valinit=init, valstep=step)

        c = self.piece_cfg
        LX, LW = 0.07, 0.25
        self.s_band = slider(LX, 0.27, LW, "band", 0, self.bands - 1, self.band, 1)
        ax_rng = self.fig.add_axes([LX, 0.235, LW, 0.022])
        self.s_range = RangeSlider(ax_rng, "mask window", 0.0, 1.0, valinit=(0.0, 1.0))
        self.s_range.on_changed(self._on_range)
        ax_bc = self.fig.add_axes([LX, 0.20, LW, 0.022])
        self.s_band_clip = RangeSlider(ax_bc, "band contrast", 0.0, 1.0, valinit=(0.0, 1.0))
        self.s_band_clip.on_changed(self._on_band_clip)
        ax_dc = self.fig.add_axes([LX, 0.165, LW, 0.022])
        self.s_dist_clip = RangeSlider(ax_dc, "dist contrast", 0.0, 1.0, valinit=(0.0, 1.0))
        self.s_dist_clip.on_changed(self._on_dist_clip)
        self.s_open = slider(LX, 0.13, LW, "open iter", 0, 8, c.open_iter, 1)
        self.s_close = slider(LX, 0.095, LW, "close iter", 0, 15, c.close_iter, 1)
        self.s_area = slider(LX, 0.06, LW, "min area", 0, 20000, c.min_area, 100)

        RX, RW = 0.52, 0.25
        # Mins sit below the defaults (patch 8 / stride 4) so the small pieces here
        # can still be tuned downward for more ROIs.
        self.s_patch = slider(RX, 0.235, RW, "ROI patch", 4, 128, self.roi_cfg.patch, 4)
        self.s_stride = slider(RX, 0.20, RW, "ROI stride", 2, 128, self.roi_cfg.stride, 2)
        self.s_cov = slider(RX, 0.165, RW, "min coverage", 0.3, 1.0, self.roi_cfg.min_coverage, 0.05)
        self.s_flat = slider(RX, 0.13, RW, "flat sigma", 5, 100, c.flat_field_sigma, 5)

        self.s_band.on_changed(lambda v: self._on_band(int(v)))
        for s in (self.s_open, self.s_close, self.s_area,
                  self.s_patch, self.s_stride, self.s_cov, self.s_flat):
            s.on_changed(self._on_param)

        ax_m = self.fig.add_axes([RX, 0.02, 0.10, 0.09])
        self.r_method = RadioButtons(ax_m, ("sam", "euclidean", "mahalanobis", "kmeans"))
        ax_m.set_title("method", fontsize=9)
        self.r_method.on_clicked(self._on_method)

        ax_t = self.fig.add_axes([0.635, 0.02, 0.10, 0.09])
        self.r_thresh = RadioButtons(ax_t, ("otsu", "percentile"))
        ax_t.set_title("threshold", fontsize=9)
        self.r_thresh.on_clicked(self._on_thresh)

        ax_ck = self.fig.add_axes([0.75, 0.02, 0.13, 0.09])
        self.checks = CheckButtons(ax_ck, ("flat field", "calibrate", "watershed"),
                                   (c.flat_field, self.use_calibrate, c.watershed_split))
        ax_ck.set_title("options", fontsize=9)
        self.checks.on_clicked(self._on_check)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.text(0.89, 0.06,
                      "'p' = print configs\n'm' = toggle overlay\n"
                      "'R' = ref box  'c' = clear\n←/→ = step band",
                      fontsize=9, family="monospace")

        self._im_band = self._im_dist = self._im_lab = None
        self._overlay = None
        self._roi_coll = None

    def _clim_from_frac(self, data, frac):
        dmin, dmax = float(np.nanmin(data)), float(np.nanmax(data))
        span = (dmax - dmin) or 1e-9
        lo = dmin if frac[0] <= 0.0 else dmin + frac[0] * span
        hi = dmax if frac[1] >= 1.0 else dmin + frac[1] * span
        return lo, max(hi, lo + 1e-9)

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
        self._im_band.set_clim(*self._clim_from_frac(band[sl], self.band_clip))
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
        self._im_dist.set_clim(*self._clim_from_frac(self.dist[sl], self.dist_clip))
        bg = "border" if self.piece_cfg.background_bbox is None else "ref-box"
        ax.set_title(f"distance ({self.piece_cfg.method}, {self.piece_cfg.threshold}, "
                     f"bg={bg})", fontsize=10)

        # panel 2: labeled pieces + ROI grid
        ax = self.axes[2]
        lab = np.where(self.labels[sl] > 0, self.labels[sl], np.nan)
        cm = plt.get_cmap("tab10").with_extremes(bad="0.12")
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

    # --- events -------------------------------------------------------------

    def _on_band(self, b):
        self.band = int(b)
        band = self.cube[:, :, self.band]
        step = max(1, int(np.ceil(max(band.shape) / MAX_DISPLAY)))
        sl = (slice(None, None, step), slice(None, None, step))
        if self._im_band is not None:
            self._im_band.set_data(band[sl])
            self._im_band.set_clim(*self._clim_from_frac(band[sl], self.band_clip))
            self.axes[0].set_title(
                f"band {self.band} ({self.wl[self.band]:.0f} nm) + mask "
                f"({self.mask.mean():.1%} fg)", fontsize=10)
            self.fig.canvas.draw_idle()

    def _reset_range_bounds(self, dist):
        lo, hi = float(dist.min()), float(dist.max())
        if hi <= lo:
            hi = lo + 1e-9
        self.s_range.valmin = lo
        self.s_range.valmax = hi
        self.s_range.ax.set_xlim(lo, hi)
        self.s_range.set_val((lo, hi))

    def _on_range(self, _):
        self._debouncer.mark_dirty()

    def _on_band_clip(self, _):
        self.band_clip = tuple(self.s_band_clip.val)
        if self._im_band is not None:
            band = self.cube[:, :, self.band]
            step = max(1, int(np.ceil(max(band.shape) / MAX_DISPLAY)))
            sl = (slice(None, None, step), slice(None, None, step))
            self._im_band.set_clim(*self._clim_from_frac(band[sl], self.band_clip))
            self.fig.canvas.draw_idle()

    def _on_dist_clip(self, _):
        self.dist_clip = tuple(self.s_dist_clip.val)
        if self._im_dist is not None and hasattr(self, "dist"):
            step = max(1, int(np.ceil(max(self.dist.shape) / MAX_DISPLAY)))
            sl = (slice(None, None, step), slice(None, None, step))
            self._im_dist.set_clim(*self._clim_from_frac(self.dist[sl], self.dist_clip))
            self.fig.canvas.draw_idle()

    def _on_param(self, _):
        self.piece_cfg = replace(
            self.piece_cfg,
            open_iter=int(self.s_open.val), close_iter=int(self.s_close.val),
            min_area=int(self.s_area.val), flat_field_sigma=float(self.s_flat.val))
        patch = int(self.s_patch.val)
        self.roi_cfg = replace(self.roi_cfg, patch=patch,
                               stride=max(1, int(self.s_stride.val)),
                               min_coverage=float(self.s_cov.val))
        self._debouncer.mark_dirty()

    def _on_method(self, label):
        self.piece_cfg = replace(self.piece_cfg, method=label)
        dist = self._distance()
        self._reset_range_bounds(dist)
        self._on_thresh(self.piece_cfg.threshold)

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

    def _on_check(self, label):
        if label == "flat field":
            self.piece_cfg = replace(self.piece_cfg, flat_field=not self.piece_cfg.flat_field)
        elif label == "calibrate":
            if not self.can_calibrate:
                print("No white/dark references loaded; calibrate unavailable.")
                return
            self.use_calibrate = not self.use_calibrate
            self.cube = self._refl_ds if self.use_calibrate else self._raw_ds
        elif label == "watershed":
            self.piece_cfg = replace(self.piece_cfg,
                                     watershed_split=not self.piece_cfg.watershed_split)
        dist = self._distance()
        self._reset_range_bounds(dist)
        self._on_thresh(self.piece_cfg.threshold)

    def _arm_reference_selector(self):
        def on_select(eclick, erelease):
            r0, r1 = sorted((int(eclick.ydata), int(erelease.ydata)))
            c0, c1 = sorted((int(eclick.xdata), int(erelease.xdata)))
            if (r1 - r0) >= 2 and (c1 - c0) >= 2:
                self.piece_cfg = replace(self.piece_cfg,
                                         background_bbox=(r0, r1 + 1, c0, c1 + 1))
                dist = self._distance()
                self._reset_range_bounds(dist)
                self._on_thresh(self.piece_cfg.threshold)
        self._selector = RectangleSelector(
            self.axes[0], on_select, useblit=True, button=[1],
            minspanx=3, minspany=3, spancoords="data", interactive=False)
        print("reference selector armed: drag a clean-background box on the band "
              "image ('c' to revert to the border frame)")

    def _on_key(self, event):
        if event.key == "right":
            self.s_band.set_val(min(self.band + 1, self.bands - 1))
        elif event.key == "left":
            self.s_band.set_val(max(self.band - 1, 0))
        elif event.key == "m":
            self.show_mask = not self.show_mask
            self._redraw()
        elif event.key == "R":
            self._arm_reference_selector()
        elif event.key == "c":
            if self.piece_cfg.background_bbox is not None:
                self.piece_cfg = replace(self.piece_cfg, background_bbox=None)
                dist = self._distance()
                self._reset_range_bounds(dist)
                self._on_thresh(self.piece_cfg.threshold)
        elif event.key == "p":
            self._print_config()

    def _print_config(self):
        k = self.ds_factor
        c, r = self.piece_cfg, self.roi_cfg
        bbox = c.background_bbox
        if bbox is not None:
            bbox = tuple(int(v * k) for v in bbox)   # -> full-res coords
        print("\n# paste into WorkflowConfig().piece / .roi "
              f"(spatial values scaled x{k} to full resolution):")
        print(f"PieceConfig(method={c.method!r}, threshold={c.threshold!r}, "
              f"threshold_percentile={c.threshold_percentile}, "
              f"open_iter={c.open_iter}, close_iter={c.close_iter}, "
              f"min_area={int(c.min_area * k * k)}, "
              # erode_iter isn't tuned here (it applies to the per-piece analysis
              # mask after cropping), so it passes through unscaled.
              f"erode_iter={c.erode_iter}, "
              f"watershed_split={c.watershed_split}, "
              f"flat_field={c.flat_field}, flat_field_sigma={c.flat_field_sigma}, "
              f"on_reflectance={self.use_calibrate}, background_bbox={bbox})")
        print(f"RoiConfig(patch={int(r.patch * k)}, stride={int(r.stride * k)}, "
              f"min_coverage={r.min_coverage})\n")


def main():
    p = argparse.ArgumentParser(description="Interactive piece-extraction & ROI tuner.")
    p.add_argument("--dataset", default="sio2_dish_white_20", type=str.lower, choices=sorted(DATASETS))
    p.add_argument("--index", type=int, default=0, help="Which cube of the dataset (0-based).")
    p.add_argument("--hdr", default=None, help="Direct ENVI header path (overrides --dataset).")
    p.add_argument("--crop", type=int, nargs=4, metavar=("R0", "R1", "C0", "C1"),
                   default=None, help="Work on a spatial window of the scan.")
    p.add_argument("--max-dim", type=int, default=DEFAULT_MAX_WORK,
                   help="Decimate the working grid to at most this many px on the long axis.")
    p.add_argument("--demo", action="store_true", help="Synthetic cube (no data needed).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    data, wl, label, white, dark, sw, sd, shutter = load_inputs(args)
    MaskTuner(data, wl, label, white=white, dark=dark, sw=sw, sd=sd,
              shutter=shutter, max_dim=args.max_dim)
    plt.show()


if __name__ == "__main__":
    main()
