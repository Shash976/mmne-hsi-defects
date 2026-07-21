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
    sliders   band | percentile | open iter | close iter | min area |
              ROI patch | ROI stride | min coverage
    keys      left/right = step band, 'p' = print configs, 'm' = toggle mask overlay

Usage
    python debug_masks.py --dataset sio2_dish_black
    python debug_masks.py --dataset sio2_dish_white_20 --crop 0 700 0 700
    python debug_masks.py --hdr "path\\to\\scan.bip.hdr"
    python debug_masks.py --demo

The spectral distance map is computed once per method (the slow part on big
scans — use --crop); thresholding, morphology, labeling, and ROI tiling rerun
instantly on every slider move.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from scipy import ndimage as ndi

from hsi_workflow.config import DATASETS, PieceConfig, RoiConfig
from hsi_workflow.io import load_cube, iter_cube_paths
from hsi_workflow.pieces import foreground_distance, _threshold_mask, clean_mask, label_pieces

MAX_DISPLAY = 500


def synthetic_cube(rows=300, cols=300, bands=100, seed=0):
    """Pieces-on-a-dish phantom: a few rectangles with a distinct spectrum."""
    rng = np.random.default_rng(seed)
    wl = np.linspace(368, 1008, bands)
    dish = 0.15 + 0.02 * np.sin(wl / 90)
    piece = 0.5 + 0.2 * np.sin(wl / 150 + 1.0)
    cube = np.tile(dish, (rows, cols, 1))
    for _ in range(5):
        r, c = rng.integers(30, rows - 90), rng.integers(30, cols - 90)
        h, w = rng.integers(40, 80), rng.integers(40, 80)
        cube[r:r + h, c:c + w, :] = piece + rng.normal(0, 0.01, bands)
    cube += rng.normal(0, 0.01, cube.shape)
    return cube, wl


def load_inputs(args):
    if args.demo:
        cube, wl = synthetic_cube(seed=args.seed)
        return cube, wl, "synthetic demo"
    if args.hdr:
        hdr, label = args.hdr, args.hdr
    else:
        ds = DATASETS[args.dataset.lower()]
        pairs = iter_cube_paths(ds)
        if not pairs:
            raise SystemExit(f"No cubes found for dataset {args.dataset!r} under {ds.data_dir}")
        label, hdr = pairs[args.index]
        print(f"Loading cube {label!r} ({args.index + 1}/{len(pairs)}) ...")
    cube = load_cube(hdr)
    data, wl = cube.data, cube.wavelengths
    if args.crop:
        r0, r1, c0, c1 = args.crop
        data = data[r0:r1, c0:c1, :]
    if wl is None:
        wl = np.arange(data.shape[-1], dtype=float)
    return data, np.asarray(wl, float), label


class MaskTuner:
    def __init__(self, cube, wl, label):
        self.cube = cube.astype(np.float64)
        self.wl = wl
        self.label = label
        self.bands = cube.shape[-1]
        self.band = self.bands // 2
        self.show_mask = True

        self.piece_cfg = PieceConfig()
        self.roi_cfg = RoiConfig()
        self._dist_cache = {}

        self._build_figure()
        self._recompute()

    # --- pipeline ---------------------------------------------------------

    def _distance(self):
        m = self.piece_cfg.method
        if m not in self._dist_cache:
            print(f"Computing foreground distance ({m}) ... ", end="", flush=True)
            self._dist_cache[m] = foreground_distance(self.cube, self.piece_cfg)
            print("done")
        return self._dist_cache[m]

    def _recompute(self):
        dist = self._distance()
        mask = _threshold_mask(dist, self.piece_cfg)
        mask = clean_mask(mask, self.piece_cfg)
        labels, kept = label_pieces(mask, self.piece_cfg)
        self.dist, self.kept = dist, kept
        # keep only surviving pieces in the displayed mask
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
        self.fig, self.axes = plt.subplots(1, 3, figsize=(17, 6.5))
        try:
            self.fig.canvas.manager.set_window_title(f"mask tuner — {self.label}")
        except Exception:
            pass
        self.fig.subplots_adjust(left=0.04, right=0.99, top=0.90, bottom=0.34, wspace=0.15)

        def slider(x, y, w, name, lo, hi, init, step=None):
            ax = self.fig.add_axes([x, y, w, 0.025])
            return Slider(ax, name, lo, hi, valinit=init, valstep=step)

        c = self.piece_cfg
        self.s_band = slider(0.08, 0.26, 0.28, "band", 0, self.bands - 1, self.band, 1)
        self.s_pct = slider(0.08, 0.22, 0.28, "percentile", 50, 99, c.threshold_percentile, 0.5)
        self.s_open = slider(0.08, 0.18, 0.28, "open iter", 0, 8, c.open_iter, 1)
        self.s_close = slider(0.08, 0.14, 0.28, "close iter", 0, 15, c.close_iter, 1)
        self.s_area = slider(0.08, 0.10, 0.28, "min area", 0, 20000, c.min_area, 100)
        self.s_patch = slider(0.55, 0.22, 0.28, "ROI patch", 8, 128, self.roi_cfg.patch, 4)
        self.s_stride = slider(0.55, 0.18, 0.28, "ROI stride", 4, 128, self.roi_cfg.stride, 4)
        self.s_cov = slider(0.55, 0.14, 0.28, "min coverage", 0.3, 1.0, self.roi_cfg.min_coverage, 0.05)

        self.s_band.on_changed(lambda v: self._on_band(int(v)))
        for s in (self.s_pct, self.s_open, self.s_close, self.s_area,
                  self.s_patch, self.s_stride, self.s_cov):
            s.on_changed(self._on_param)

        ax_m = self.fig.add_axes([0.55, 0.02, 0.12, 0.10])
        self.r_method = RadioButtons(ax_m, ("sam", "mahalanobis", "kmeans"))
        ax_m.set_title("method", fontsize=9)
        self.r_method.on_clicked(self._on_method)

        ax_t = self.fig.add_axes([0.70, 0.02, 0.12, 0.10])
        self.r_thresh = RadioButtons(ax_t, ("otsu", "percentile"))
        ax_t.set_title("threshold", fontsize=9)
        self.r_thresh.on_clicked(self._on_thresh)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.text(0.86, 0.07, "'p' = print configs\n'm' = toggle overlay\n"
                                  "←/→ = step band", fontsize=9, family="monospace")

    def _redraw(self):
        band = self.cube[:, :, self.band]
        step = max(1, int(np.ceil(max(band.shape) / MAX_DISPLAY)))
        sl = (slice(None, None, step), slice(None, None, step))
        extent = (0, band.shape[1], band.shape[0], 0)

        ax = self.axes[0]; ax.clear()
        ax.imshow(band[sl], cmap="gray", extent=extent)
        if self.show_mask:
            overlay = np.where(self.mask[sl], 1.0, np.nan)
            ax.imshow(np.ma.masked_invalid(overlay), cmap="autumn", alpha=0.35,
                      vmin=0, vmax=1, extent=extent)
        cov = self.mask.mean()
        ax.set_title(f"band {self.band} ({self.wl[self.band]:.0f} nm) + mask "
                     f"({cov:.1%} fg)", fontsize=10)
        ax.axis("off")

        ax = self.axes[1]; ax.clear()
        ax.imshow(self.dist[sl], cmap="magma", extent=extent)
        ax.set_title(f"foreground distance ({self.piece_cfg.method}, "
                     f"{self.piece_cfg.threshold})", fontsize=10)
        ax.axis("off")

        ax = self.axes[2]; ax.clear()
        lab = np.where(self.labels[sl] > 0, self.labels[sl], np.nan)
        cm = plt.get_cmap("tab10").copy(); cm.set_bad("0.12")
        ax.set_facecolor("0.12")
        ax.imshow(np.ma.masked_invalid(lab % 10), cmap=cm, vmin=0, vmax=9,
                  interpolation="nearest", extent=extent)
        p = self.roi_cfg.patch
        for (r, c) in self.roi_boxes:
            ax.add_patch(plt.Rectangle((c, r), p, p, fill=False, ec="white", lw=0.6))
        n_rois = sum(self.roi_counts.values())
        ax.set_title(f"{len(self.kept)} piece(s), {n_rois} ROI(s) "
                     f"[patch {p}, stride {self.roi_cfg.stride}]", fontsize=10)
        ax.axis("off")

        sizes = sorted((int((self.labels == l).sum()) for l in self.kept), reverse=True)
        print(f"pieces: {len(self.kept)}  sizes(px): {sizes[:8]}"
              f"{'...' if len(sizes) > 8 else ''}  ROIs/piece: "
              f"{[self.roi_counts[l] for l in self.kept][:8]}")
        self.fig.canvas.draw_idle()

    # --- events -------------------------------------------------------------

    def _on_band(self, b):
        self.band = b
        self._redraw()

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
        self._recompute()

    def _on_method(self, label):
        self.piece_cfg = replace(self.piece_cfg, method=label)
        self._recompute()

    def _on_thresh(self, label):
        self.piece_cfg = replace(self.piece_cfg, threshold=label)
        self._recompute()

    def _on_key(self, event):
        if event.key == "right":
            self.s_band.set_val(min(self.band + 1, self.bands - 1))
        elif event.key == "left":
            self.s_band.set_val(max(self.band - 1, 0))
        elif event.key == "m":
            self.show_mask = not self.show_mask
            self._redraw()
        elif event.key == "p":
            c, r = self.piece_cfg, self.roi_cfg
            print("\n# paste into WorkflowConfig().piece / .roi:")
            print(f"PieceConfig(method={c.method!r}, threshold={c.threshold!r}, "
                  f"threshold_percentile={c.threshold_percentile}, "
                  f"open_iter={c.open_iter}, close_iter={c.close_iter}, "
                  f"min_area={c.min_area})")
            print(f"RoiConfig(patch={r.patch}, stride={r.stride}, "
                  f"min_coverage={r.min_coverage})\n")


def main():
    p = argparse.ArgumentParser(description="Interactive piece-extraction & ROI tuner.")
    p.add_argument("--dataset", default="sio2_bare_si", type=str.lower, choices=sorted(DATASETS))
    p.add_argument("--index", type=int, default=0, help="Which cube of the dataset (0-based).")
    p.add_argument("--hdr", default=None, help="Direct ENVI header path (overrides --dataset).")
    p.add_argument("--crop", type=int, nargs=4, metavar=("R0", "R1", "C0", "C1"),
                   default=None, help="Work on a spatial window of the scan.")
    p.add_argument("--demo", action="store_true", help="Synthetic cube (no data needed).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cube, wl, label = load_inputs(args)
    MaskTuner(cube, wl, label)
    plt.show()


if __name__ == "__main__":
    main()
