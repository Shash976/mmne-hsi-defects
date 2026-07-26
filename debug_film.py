# debug_film.py
"""Interactive SiO2-film extractor for a single silicon piece.

Piece extraction (debug_masks.py) crops each wafer out of the dish. This tool
tunes the *next* step: isolating the SiO2 sub-region **within** one wafer, by
spectral distance from bare silicon. Tune the value-window / method / reference
until the oxide patch is cleanly separated, then press 'p' for a paste-ready
``FilmConfig(...)``.

Panels
    1. band image with the SiO2 mask as a red overlay
    2. the film *distance* map (distance from bare silicon; oxide = bright)
    3. SiO2 (red) vs bare silicon (blue) within the wafer

Controls
    radios    method: sam | mahalanobis | kmeans   reference: control | in_piece
              threshold: otsu | percentile
    sliders   band | mask window | band contrast | open iter | close iter | min area
    keys      ←/→ = step band, 'm' = toggle overlay, 'p' = print FilmConfig

Both the target piece and the bare-silicon control are calibrated to reflectance
so cross-scan illumination cancels. Reference = "control" uses the mean spectrum
of the ``sio2_bare_si`` dataset; "in_piece" uses the wafer's own 2-cluster split.

Usage
    python debug_film.py --demo
    python debug_film.py --dataset sio2_dish_white_20 --piece 0
    python debug_film.py --dataset sio2_dish_white_20 --crop 0 700 0 700
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, RangeSlider
from scipy import ndimage as ndi

from hsi_workflow.config import DATASETS, PieceConfig, FilmConfig
from hsi_workflow.cube_io import Cube, load_cube, iter_cube_paths, load_reference_spectrum
from hsi_workflow.preprocessing import calibrate_reflectance
from hsi_workflow.pieces import extract_pieces, clean_mask, component_sizes
from hsi_workflow.film import film_distance
from debug_common import Debouncer

MAX_DISPLAY = 500
DEFAULT_MAX_WORK = 700


def synthetic_piece(rows=120, cols=120, bands=60, seed=0):
    """A wafer with a bare-Si region and an SiO2 (fringed) patch, plus a control."""
    rng = np.random.default_rng(seed)
    wl = np.linspace(368, 1008, bands)
    bare = 0.5 + 0.10 * np.sin(wl / 200)
    oxide = bare + 0.15 * np.sin(wl / 55)
    data = np.tile(bare, (rows, cols, 1)).astype(np.float64)
    mask = np.zeros((rows, cols), bool)
    mask[15:105, 15:105] = True
    oxide_region = np.zeros((rows, cols), bool)
    oxide_region[30:70, 35:85] = True
    data[oxide_region] = oxide
    data += rng.normal(0, 0.004, data.shape)
    data[~mask] = 0.02 + rng.normal(0, 0.002, (int((~mask).sum()), bands))
    return data, mask, wl, bare.copy()


def _decimate(data, max_dim):
    step = max(1, int(np.ceil(max(data.shape[:2]) / max_dim)))
    return data[::step, ::step, :], step


def _prepare_dataset_reflectance(ds, crop, max_dim):
    """Load a dataset's first cube, crop/decimate, calibrate to reflectance."""
    pairs = iter_cube_paths(ds)
    if not pairs:
        raise SystemExit(f"No cubes for dataset {ds.name!r} under {ds.data_dir}")
    label, hdr = pairs[0]
    cube = load_cube(hdr)
    data = cube.data
    if crop:
        r0, r1, c0, c1 = crop
        data = data[r0:r1, c0:c1, :]
    data, _ = _decimate(data, max_dim)
    white, sw = load_reference_spectrum(ds.white_ref)
    dark, sd = load_reference_spectrum(ds.dark_ref)
    refl = calibrate_reflectance(data, cube.shutter, white, sw, dark, sd)
    wl = cube.wavelengths if cube.wavelengths is not None else np.arange(refl.shape[-1], float)
    return Cube(data=refl, wavelengths=np.asarray(wl, float), shutter=1.0,
                ceiling=cube.ceiling, path=hdr, label=label, material=ds.material)


def load_inputs(args):
    if args.demo:
        data, mask, wl, bare = synthetic_piece(seed=args.seed)
        return data, mask, wl, bare, "synthetic demo"

    target_ds = DATASETS[args.dataset.lower()]
    print(f"Loading target {target_ds.name!r} (calibrating to reflectance) ...")
    tcube = _prepare_dataset_reflectance(target_ds, args.crop, args.max_dim)
    print("Extracting pieces ...")
    pieces = extract_pieces(tcube, PieceConfig())
    if not pieces:
        raise SystemExit("No pieces found in target; try --crop or debug_masks.py first.")
    idx = min(args.piece, len(pieces) - 1)
    piece = pieces[idx]
    print(f"Using piece {idx + 1}/{len(pieces)}: {piece.piece_id} "
          f"({piece.mask.sum()} px)")

    print("Loading bare-silicon control for the reference spectrum ...")
    ctrl_ds = DATASETS["sio2_bare_si"]
    ccube = _prepare_dataset_reflectance(ctrl_ds, None, args.max_dim)
    cpieces = extract_pieces(ccube, PieceConfig())
    if cpieces:
        bare = np.vstack([cp.data[cp.mask] for cp in cpieces]).mean(axis=0)
    else:
        bare = ccube.data.reshape(-1, ccube.data.shape[-1]).mean(axis=0)
    wl = piece.wavelengths if piece.wavelengths is not None else np.arange(piece.data.shape[-1], float)
    return piece.data, piece.mask, np.asarray(wl, float), bare, piece.piece_id


class FilmTuner:
    def __init__(self, data, mask, wl, bare_ref, label):
        self.cube = data.astype(np.float64)
        self.mask = mask.astype(bool)
        self.wl = wl
        self.bare_ref = None if bare_ref is None else np.asarray(bare_ref, float)
        self.label = label
        self.bands = self.cube.shape[-1]
        self.band = self.bands // 2
        self.show_overlay = True
        self.band_clip = (0.0, 1.0)

        self.film_cfg = FilmConfig(open_iter=1, close_iter=1, min_area=20)
        self._dist_cache = {}

        self._build_figure()
        self._debouncer = Debouncer(self.fig.canvas, self._recompute)
        self._recompute()
        self._reset_range_bounds(self.dist)
        self._on_thresh(self.film_cfg.threshold)

    # --- pipeline ---------------------------------------------------------

    def _distance(self):
        key = (self.film_cfg.method, self.film_cfg.reference)
        if key not in self._dist_cache:
            print(f"Computing film distance {key} ... ", end="", flush=True)
            self._dist_cache[key] = film_distance(self.cube, self.mask,
                                                  self.bare_ref, self.film_cfg)
            print("done")
        return self._dist_cache[key]

    def _recompute(self):
        dist = self._distance()
        in_mask = dist[self.mask]
        dmax = float(in_mask.max()) if in_mask.size else 1.0
        if not np.isclose(self.s_range.valmax, dmax):
            self._reset_range_bounds(dist)
        lo, hi = self.s_range.val
        sio2 = self.mask & (dist >= lo) & (dist <= hi)
        sio2 = clean_mask(sio2, self.film_cfg) & self.mask
        labels, _ = ndi.label(sio2)
        sizes = component_sizes(labels)
        keep = sizes >= self.film_cfg.min_area
        keep[0] = False
        self.sio2 = keep[labels] if labels.max() else np.zeros_like(self.mask)
        self.substrate = self.mask & ~self.sio2
        self.dist = dist
        self._redraw()

    # --- figure -----------------------------------------------------------

    def _build_figure(self):
        self.fig, self.axes = plt.subplots(1, 3, figsize=(16, 6.4))
        try:
            self.fig.canvas.manager.set_window_title(f"film tuner — {self.label}")
        except Exception:
            pass
        self.fig.subplots_adjust(left=0.04, right=0.99, top=0.92, bottom=0.32, wspace=0.15)

        def slider(x, y, w, name, lo, hi, init, step=None):
            ax = self.fig.add_axes([x, y, w, 0.022])
            return Slider(ax, name, lo, hi, valinit=init, valstep=step)

        c = self.film_cfg
        LX, LW = 0.07, 0.26
        self.s_band = slider(LX, 0.25, LW, "band", 0, self.bands - 1, self.band, 1)
        ax_rng = self.fig.add_axes([LX, 0.215, LW, 0.022])
        self.s_range = RangeSlider(ax_rng, "mask window", 0.0, 1.0, valinit=(0.0, 1.0))
        self.s_range.on_changed(self._on_range)
        ax_bc = self.fig.add_axes([LX, 0.18, LW, 0.022])
        self.s_band_clip = RangeSlider(ax_bc, "band contrast", 0.0, 1.0, valinit=(0.0, 1.0))
        self.s_band_clip.on_changed(self._on_band_clip)
        self.s_open = slider(LX, 0.11, LW, "open iter", 0, 6, c.open_iter, 1)
        self.s_close = slider(LX, 0.075, LW, "close iter", 0, 10, c.close_iter, 1)
        self.s_area = slider(LX, 0.04, LW, "min area", 0, 3000, c.min_area, 20)
        self.s_band.on_changed(lambda v: self._on_band(int(v)))
        for s in (self.s_open, self.s_close, self.s_area):
            s.on_changed(self._on_param)

        ax_m = self.fig.add_axes([0.52, 0.03, 0.11, 0.10])
        self.r_method = RadioButtons(ax_m, ("sam", "mahalanobis", "kmeans"))
        ax_m.set_title("method", fontsize=9)
        self.r_method.on_clicked(self._on_method)

        ax_r = self.fig.add_axes([0.65, 0.03, 0.11, 0.10])
        self.r_ref = RadioButtons(ax_r, ("control", "in_piece"))
        ax_r.set_title("reference", fontsize=9)
        self.r_ref.on_clicked(self._on_ref)

        ax_t = self.fig.add_axes([0.78, 0.03, 0.10, 0.10])
        self.r_thresh = RadioButtons(ax_t, ("otsu", "percentile"))
        ax_t.set_title("threshold", fontsize=9)
        self.r_thresh.on_clicked(self._on_thresh)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.text(0.90, 0.06, "'p' = print FilmConfig\n'm' = toggle overlay\n"
                                  "←/→ = step band", fontsize=9, family="monospace")

        self._im_band = self._im_dist = self._im_seg = None
        self._overlay = None

    def _clim_from_frac(self, data, frac):
        dmin, dmax = float(np.nanmin(data)), float(np.nanmax(data))
        span = (dmax - dmin) or 1e-9
        lo = dmin if frac[0] <= 0.0 else dmin + frac[0] * span
        hi = dmax if frac[1] >= 1.0 else dmin + frac[1] * span
        return lo, max(hi, lo + 1e-9)

    def _sl(self, shape):
        step = max(1, int(np.ceil(max(shape) / MAX_DISPLAY)))
        return (slice(None, None, step), slice(None, None, step))

    def _redraw(self):
        band = self.cube[:, :, self.band]
        sl = self._sl(band.shape)
        extent = (0, band.shape[1], band.shape[0], 0)

        ax = self.axes[0]
        if self._im_band is None:
            ax.clear(); ax.axis("off")
            self._im_band = ax.imshow(band[sl], cmap="gray", extent=extent)
            self._overlay = ax.imshow(
                np.ma.masked_invalid(np.where(self.sio2[sl], 1.0, np.nan)),
                cmap="autumn", alpha=0.4, vmin=0, vmax=1, extent=extent)
        else:
            self._im_band.set_data(band[sl])
            self._overlay.set_data(np.ma.masked_invalid(np.where(self.sio2[sl], 1.0, np.nan)))
        self._im_band.set_clim(*self._clim_from_frac(band[sl], self.band_clip))
        self._overlay.set_visible(self.show_overlay)
        frac = self.sio2.sum() / max(1, self.mask.sum())
        ax.set_title(f"band {self.band} ({self.wl[self.band]:.0f} nm) + SiO2 "
                     f"({frac:.1%} of wafer)", fontsize=10)

        ax = self.axes[1]
        dshow = np.where(self.mask, self.dist, np.nan)[sl]
        if self._im_dist is None:
            ax.clear(); ax.axis("off")
            self._im_dist = ax.imshow(dshow, cmap="magma", extent=extent)
        else:
            self._im_dist.set_data(dshow)
        finite = self.dist[self.mask]
        if finite.size:
            self._im_dist.set_clim(float(finite.min()), float(finite.max()))
        ax.set_title(f"film distance ({self.film_cfg.method}, {self.film_cfg.reference})",
                     fontsize=10)

        ax = self.axes[2]
        seg = np.full(self.mask.shape, np.nan)
        seg[self.substrate] = 0.0
        seg[self.sio2] = 1.0
        cm = plt.get_cmap("bwr").with_extremes(bad="0.12")
        if self._im_seg is None:
            ax.clear(); ax.axis("off"); ax.set_facecolor("0.12")
            self._im_seg = ax.imshow(np.ma.masked_invalid(seg[sl]), cmap=cm,
                                     vmin=0, vmax=1, interpolation="nearest", extent=extent)
        else:
            self._im_seg.set_data(np.ma.masked_invalid(seg[sl]))
        ax.set_title("SiO2 (red) vs bare Si (blue)", fontsize=10)
        self.fig.canvas.draw_idle()

    # --- events -----------------------------------------------------------

    def _on_band(self, b):
        self.band = int(b)
        band = self.cube[:, :, self.band]
        sl = self._sl(band.shape)
        if self._im_band is not None:
            self._im_band.set_data(band[sl])
            self._im_band.set_clim(*self._clim_from_frac(band[sl], self.band_clip))
            self.fig.canvas.draw_idle()

    def _reset_range_bounds(self, dist):
        in_mask = dist[self.mask]
        lo = float(in_mask.min()) if in_mask.size else 0.0
        hi = float(in_mask.max()) if in_mask.size else 1.0
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
            sl = self._sl(band.shape)
            self._im_band.set_clim(*self._clim_from_frac(band[sl], self.band_clip))
            self.fig.canvas.draw_idle()

    def _on_param(self, _):
        self.film_cfg = replace(self.film_cfg, open_iter=int(self.s_open.val),
                                close_iter=int(self.s_close.val),
                                min_area=int(self.s_area.val))
        self._debouncer.mark_dirty()

    def _on_method(self, label):
        self.film_cfg = replace(self.film_cfg, method=label)
        self._reset_range_bounds(self._distance())
        self._on_thresh(self.film_cfg.threshold)

    def _on_ref(self, label):
        self.film_cfg = replace(self.film_cfg, reference=label)
        self._reset_range_bounds(self._distance())
        self._on_thresh(self.film_cfg.threshold)

    def _on_thresh(self, label):
        self.film_cfg = replace(self.film_cfg, threshold=label)
        dist = self._distance()
        vals = dist[self.mask]
        if label == "otsu" and vals.size:
            from skimage.filters import threshold_otsu
            try:
                cutoff = float(threshold_otsu(vals))
            except ValueError:
                cutoff = float(np.median(vals))
        else:
            cutoff = float(np.percentile(vals, self.film_cfg.threshold_percentile)) if vals.size else 0.0
        _, hi = self.s_range.val
        self.s_range.set_val((min(cutoff, hi), hi))
        self._recompute()

    def _on_key(self, event):
        if event.key == "right":
            self.s_band.set_val(min(self.band + 1, self.bands - 1))
        elif event.key == "left":
            self.s_band.set_val(max(self.band - 1, 0))
        elif event.key == "m":
            self.show_overlay = not self.show_overlay
            self._redraw()
        elif event.key == "p":
            c = self.film_cfg
            print("\n# paste into WorkflowConfig().film (enable with --extract-film):")
            print(f"FilmConfig(enabled=True, reference={c.reference!r}, method={c.method!r}, "
                  f"threshold={c.threshold!r}, threshold_percentile={c.threshold_percentile}, "
                  f"open_iter={c.open_iter}, close_iter={c.close_iter}, "
                  f"min_area={c.min_area}, invert={c.invert})\n")


def main():
    p = argparse.ArgumentParser(description="Interactive SiO2-film extractor.")
    p.add_argument("--dataset", default="sio2_dish_white_20", type=str.lower, choices=sorted(DATASETS))
    p.add_argument("--piece", type=int, default=0, help="Which extracted piece (0 = largest).")
    p.add_argument("--crop", type=int, nargs=4, metavar=("R0", "R1", "C0", "C1"), default=None)
    p.add_argument("--max-dim", type=int, default=DEFAULT_MAX_WORK)
    p.add_argument("--demo", action="store_true", help="Synthetic piece (no data needed).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    data, mask, wl, bare, label = load_inputs(args)
    FilmTuner(data, mask, wl, bare, label)
    plt.show()


if __name__ == "__main__":
    main()
