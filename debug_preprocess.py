# debug_preprocess.py
"""Interactive preprocessing tuner for hyperspectral cubes.

Play with the Stage 2-3 knobs (Savitzky-Golay window/polyorder, SNV, polynomial
baseline, calibration on/off) and *see* what they do, live:

    - left: a band image of the currently-processed cube (band slider);
      click any pixel to inspect it
    - top right: the clicked pixel's spectrum BEFORE (calibrated reflectance)
      vs AFTER (current settings)
    - bottom right: live noise metrics (RMS high-frequency noise + spectral SNR
      before vs after smoothing) and the reflectance-range check

Controls
    sliders   band | SG window | SG polyorder | baseline order
    checks    calibrate | SG smooth | SNV | baseline
    keys      left/right = step band, 'p' = print a paste-ready
              PreprocessConfig(...) for the current settings, 'r' = reset view

Usage
    python debug_preprocess.py --dataset sio2_bare_si
    python debug_preprocess.py --dataset sio2_dish_black --crop 100 500 100 500
    python debug_preprocess.py --hdr "path\\to\\scan.bip.hdr"
    python debug_preprocess.py --demo            # synthetic cube, no data needed

Big scans: pass --crop r0 r1 c0 c1 (raw-scan pixel coords) to work on a window;
the display grid is auto-decimated to keep sliders responsive, but clicked
spectra and metrics always use full spectral resolution.
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons

from hsi_workflow.config import DATASETS
from hsi_workflow.cube_io import load_cube, iter_cube_paths, load_reference_spectrum
from hsi_workflow.preprocessing import (calibrate_reflectance, savgol_smooth,
                                        baseline_correct, normalize_intensity,
                                        noise_metrics)

MAX_DISPLAY = 400          # display grid decimated to at most this many rows/cols


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def synthetic_cube(rows=240, cols=240, bands=300, seed=0):
    """A film-with-blemishes phantom so the tool runs without data on disk."""
    rng = np.random.default_rng(seed)
    wl = np.linspace(368, 1008, bands)
    base = 0.4 + 0.25 * np.sin((wl - 368) / 640 * 3 * np.pi)
    cube = np.tile(base, (rows, cols, 1))
    yy, xx = np.mgrid[0:rows, 0:cols]
    for _ in range(6):                       # blemishes with a shifted spectrum
        r, c = rng.integers(20, rows - 20), rng.integers(20, cols - 20)
        blob = np.exp(-(((yy - r) ** 2 + (xx - c) ** 2) / (2 * rng.uniform(3, 10) ** 2)))
        shift = 0.15 * np.sin((wl - 368) / 640 * 5 * np.pi + rng.uniform(0, np.pi))
        cube += blob[:, :, None] * shift[None, None, :]
    cube += rng.normal(0, 0.02, cube.shape)  # sensor noise for the SG demo
    return cube, wl


def load_inputs(args):
    """Returns (raw cube, wavelengths, shutter, white/dark spectra or None, label)."""
    if args.demo:
        cube, wl = synthetic_cube(seed=args.seed)
        return cube, wl, 1.0, None, None, 1.0, 1.0, "synthetic demo"
    if args.hdr:
        hdr, white_hdr, dark_hdr, label = args.hdr, args.white, args.dark, args.hdr
    else:
        ds = DATASETS[args.dataset]
        pairs = iter_cube_paths(ds)
        if not pairs:
            raise SystemExit(f"No cubes found for dataset {args.dataset!r} under {ds.data_dir}")
        label, hdr = pairs[args.index]
        white_hdr, dark_hdr = ds.white_ref, ds.dark_ref
        print(f"Loading cube {label!r} ({args.index + 1}/{len(pairs)}) ...")
    cube = load_cube(hdr)
    data, wl, shutter = cube.data, cube.wavelengths, cube.shutter
    if args.crop:
        r0, r1, c0, c1 = args.crop
        data = data[r0:r1, c0:c1, :]
    white = dark = None
    sw = sd = 1.0
    if white_hdr and dark_hdr:
        print("Loading white/dark references (cached after first use) ...")
        white, sw = load_reference_spectrum(white_hdr)
        dark, sd = load_reference_spectrum(dark_hdr)
    if wl is None:
        wl = np.arange(data.shape[-1], dtype=float)
    return data, np.asarray(wl, float), shutter, white, dark, sw, sd, label


# --------------------------------------------------------------------------
# The app
# --------------------------------------------------------------------------

class PreprocessTuner:
    def __init__(self, raw, wl, shutter, white, dark, sw, sd, label):
        self.wl = wl
        self.label = label
        self.bands = raw.shape[-1]

        # Reflectance is the fixed "before" reference: calibrate once.
        if white is not None and dark is not None:
            self.reflectance = calibrate_reflectance(raw, shutter, white, sw, dark, sd)
            self.can_calibrate = True
        else:
            self.reflectance = raw.astype(np.float64)
            self.can_calibrate = False
            print("No white/dark references: 'calibrate' toggle disabled, raw DN used.")
        self.raw = raw.astype(np.float64)

        # Decimated grid for the live band image (full-res kept for spectra).
        step = max(1, int(np.ceil(max(raw.shape[:2]) / MAX_DISPLAY)))
        self.step = step
        self.sel = (slice(None, None, step), slice(None, None, step), slice(None))

        # State
        self.band = self.bands // 2
        self.sg_window, self.sg_polyorder = 11, 2
        self.baseline_order = 2
        self.use_calibrate, self.use_smooth = self.can_calibrate, True
        self.use_snv, self.use_baseline = True, False
        self.pixel = (raw.shape[0] // 2, raw.shape[1] // 2)

        self._build_figure()
        self._recompute(full=True)

    # --- processing ---------------------------------------------------

    def _src(self, sel):
        """The 'before' array for the current calibrate toggle, at ``sel``."""
        base = self.reflectance if (self.use_calibrate or not self.can_calibrate) else self.raw
        return base[sel]

    def _process(self, sel):
        """Apply the current settings to the cube region selected by ``sel``."""
        data = self._src(sel)
        if self.use_smooth:
            data = savgol_smooth(data, self.sg_window, self.sg_polyorder)
        if self.use_baseline:
            data = baseline_correct(data, "poly", self.baseline_order)
        if self.use_snv:
            data = normalize_intensity(data, "snv")
        return data

    def _recompute(self, full=False):
        self.display_cube = self._process(self.sel)
        self._update_band_image()
        self._update_spectrum()
        self._update_metrics()
        self.fig.canvas.draw_idle()

    # --- figure ---------------------------------------------------------

    def _build_figure(self):
        self.fig = plt.figure(figsize=(15, 8.5))
        self._ax2 = None
        try:
            self.fig.canvas.manager.set_window_title(f"preprocess tuner — {self.label}")
        except Exception:
            pass
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[2.2, 1],
                                   left=0.05, right=0.98, top=0.92, bottom=0.24,
                                   hspace=0.3, wspace=0.18)
        self.ax_img = self.fig.add_subplot(gs[:, 0])
        self.ax_spec = self.fig.add_subplot(gs[0, 1])
        self.ax_text = self.fig.add_subplot(gs[1, 1]); self.ax_text.axis("off")

        # Sliders
        def slider(y, name, lo, hi, init, step=None, fmt=None):
            ax = self.fig.add_axes([0.08, y, 0.36, 0.03])
            s = Slider(ax, name, lo, hi, valinit=init, valstep=step, valfmt=fmt)
            return s
        self.s_band = slider(0.15, "band", 0, self.bands - 1, self.band, step=1, fmt="%0.0f")
        self.s_window = slider(0.11, "SG window", 3, min(51, self.bands - 1), self.sg_window,
                               step=2, fmt="%0.0f")
        self.s_poly = slider(0.07, "SG polyorder", 1, 5, self.sg_polyorder, step=1, fmt="%0.0f")
        self.s_base = slider(0.03, "baseline order", 1, 4, self.baseline_order, step=1, fmt="%0.0f")
        self.s_band.on_changed(self._on_band)
        for s in (self.s_window, self.s_poly, self.s_base):
            s.on_changed(self._on_param)

        labels = ["calibrate", "SG smooth", "SNV", "baseline"]
        state = [self.use_calibrate, self.use_smooth, self.use_snv, self.use_baseline]
        ax_checks = self.fig.add_axes([0.55, 0.03, 0.14, 0.15])
        self.checks = CheckButtons(ax_checks, labels, state)
        self.checks.on_clicked(self._on_check)

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.text(0.72, 0.10, "click image = pick pixel\n←/→ = step band\n"
                                  "'p' = print PreprocessConfig", fontsize=9,
                      va="center", family="monospace")

    # --- panel updates ---------------------------------------------------

    def _update_band_image(self):
        self.ax_img.clear()
        band = self.display_cube[:, :, self.band]
        im = self.ax_img.imshow(band, cmap="magma",
                                extent=(0, self.raw.shape[1], self.raw.shape[0], 0))
        if not hasattr(self, "_cbar"):
            self._cbar = self.fig.colorbar(im, ax=self.ax_img, fraction=0.046)
        else:
            self._cbar.update_normal(im)
        r, c = self.pixel
        self.ax_img.plot(c, r, "c+", ms=14, mew=2)
        self.ax_img.set_title(f"band {self.band}  ({self.wl[self.band]:.0f} nm)  "
                              f"[decimation x{self.step}]", fontsize=11)

    def _update_spectrum(self):
        if getattr(self, "_ax2", None) is not None:
            self._ax2.remove()
            self._ax2 = None
        self.ax_spec.clear()
        r, c = self.pixel
        pixel_sel = (slice(r, r + 1), slice(c, c + 1), slice(None))
        before = self._src(pixel_sel)[0, 0, :]
        after = self._process(pixel_sel)[0, 0, :]
        before_label = "before (reflectance)" if self.can_calibrate else "before (raw)"
        self.ax_spec.plot(self.wl, before, color="0.6", lw=1, label=before_label)
        if self.use_snv:                       # different scale -> twin axis
            self._ax2 = self.ax_spec.twinx()
            ax2 = self._ax2
        else:
            ax2 = self.ax_spec
        ax2.plot(self.wl, after, color="tab:red", lw=1.2, label="after (current settings)")
        self.ax_spec.axvline(self.wl[self.band], color="tab:cyan", lw=0.8, ls=":")
        self.ax_spec.set_xlabel("wavelength (nm)")
        self.ax_spec.set_ylabel("reflectance")
        if self.use_snv:
            ax2.set_ylabel("SNV value", color="tab:red")
        self.ax_spec.set_title(f"pixel ({r}, {c}) spectrum: before vs after", fontsize=11)
        lines, labels = self.ax_spec.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels() if ax2 is not self.ax_spec else ([], [])
        self.ax_spec.legend(lines + l2, labels + lb2, fontsize=8)

    def _update_metrics(self):
        src = self._src(self.sel)
        nb = noise_metrics(src, self.sg_window, self.sg_polyorder, sample=3000)
        after_smooth = (savgol_smooth(src, self.sg_window, self.sg_polyorder)
                        if self.use_smooth else src)
        na = noise_metrics(after_smooth, self.sg_window, self.sg_polyorder, sample=3000)
        vals = src[::4, ::4, :].ravel()
        vals = vals[np.isfinite(vals)]
        oor = ((vals < 0) | (vals > 1)).mean() if self.can_calibrate and vals.size else float("nan")
        red = (1 - na["rms_noise"] / nb["rms_noise"]) if nb["rms_noise"] else float("nan")
        txt = (f"NOISE (subsampled)\n"
               f"  RMS HF noise   before {nb['rms_noise']:.4g}   after {na['rms_noise']:.4g}"
               f"   ({red:.0%} reduction)\n"
               f"  spectral SNR   before {nb['snr']:.1f}   after {na['snr']:.1f}\n\n"
               f"REFLECTANCE RANGE\n"
               f"  outside [0, 1]: {oor:.2%}\n\n"
               f"SETTINGS  window={self.sg_window} poly={self.sg_polyorder} "
               f"snv={self.use_snv} baseline={self.use_baseline}")
        self.ax_text.clear(); self.ax_text.axis("off")
        self.ax_text.text(0.0, 0.95, txt, va="top", family="monospace", fontsize=10)

    # --- events -----------------------------------------------------------

    def _on_band(self, val):
        self.band = int(val)
        self._update_band_image()
        self._update_spectrum()
        self.fig.canvas.draw_idle()

    def _on_param(self, _):
        self.sg_window = int(self.s_window.val) | 1        # keep odd
        self.sg_polyorder = min(int(self.s_poly.val), self.sg_window - 1)
        self.baseline_order = int(self.s_base.val)
        self._recompute()

    def _on_check(self, label):
        if label == "calibrate":
            if not self.can_calibrate:
                print("No white/dark references loaded; calibrate unavailable.")
                return
            self.use_calibrate = not self.use_calibrate
        elif label == "SG smooth":
            self.use_smooth = not self.use_smooth
        elif label == "SNV":
            self.use_snv = not self.use_snv
        elif label == "baseline":
            self.use_baseline = not self.use_baseline
        self._recompute()

    def _on_click(self, event):
        if event.inaxes is not self.ax_img or event.xdata is None:
            return
        c, r = int(event.xdata), int(event.ydata)
        if 0 <= r < self.raw.shape[0] and 0 <= c < self.raw.shape[1]:
            self.pixel = (r, c)
            self._update_band_image()
            self._update_spectrum()
            self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == "right":
            self.s_band.set_val(min(self.band + 1, self.bands - 1))
        elif event.key == "left":
            self.s_band.set_val(max(self.band - 1, 0))
        elif event.key == "p":
            smooth = "savgol" if self.use_smooth else "none"
            baseline = "poly" if self.use_baseline else "none"
            normalize = "snv" if self.use_snv else "none"
            print("\n# paste into WorkflowConfig().preprocess or PreprocessConfig(...):")
            print(f"PreprocessConfig(calibrate={self.use_calibrate}, "
                  f"smooth={smooth!r}, sg_window={self.sg_window}, "
                  f"sg_polyorder={self.sg_polyorder}, baseline={baseline!r}, "
                  f"baseline_order={self.baseline_order}, normalize={normalize!r})\n")


def main():
    p = argparse.ArgumentParser(description="Interactive preprocessing (filter/window) tuner.")
    p.add_argument("--dataset", default="sio2_bare_si", choices=sorted(DATASETS))
    p.add_argument("--index", type=int, default=0, help="Which cube of the dataset (0-based).")
    p.add_argument("--hdr", default=None, help="Direct ENVI header path (overrides --dataset).")
    p.add_argument("--white", default=None, help="White reference .hdr (with --hdr).")
    p.add_argument("--dark", default=None, help="Dark reference .hdr (with --hdr).")
    p.add_argument("--crop", type=int, nargs=4, metavar=("R0", "R1", "C0", "C1"),
                   default=None, help="Work on a spatial window of the scan.")
    p.add_argument("--demo", action="store_true", help="Synthetic cube (no data needed).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    inputs = load_inputs(args)
    PreprocessTuner(*inputs)
    plt.show()


if __name__ == "__main__":
    main()
