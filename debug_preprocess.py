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

Performance
    Heavy work (reprocessing the cube) is *debounced*: it runs once when you
    release a slider, not on every intermediate tick. Light changes (band index,
    display contrast, picking a pixel) update the existing artists instantly.
    The working cube is spatially decimated to ``--max-dim`` px on its long axis
    so the whole scan stays interactive; clicked-pixel spectra always use full
    spectral resolution.

Controls
    sliders   band | SG window | SG polyorder | baseline order | contrast
    checks    calibrate | SG smooth | SNV | baseline | subtract ref
    click     pick a pixel (inspect its spectrum)
    shift+click  set the reference spectrum (5x5 average around the pixel)
    keys      left/right = step band, 'p' = print a paste-ready
              PreprocessConfig(...), 'c' = clear the reference

Usage
    python debug_preprocess.py --dataset sio2_dish_white_20
    python debug_preprocess.py --dataset sio2_bare_si --crop 100 500 100 500
    python debug_preprocess.py --hdr "path\\to\\scan.bip.hdr"
    python debug_preprocess.py --demo            # synthetic cube, no data needed

Big scans: pass --crop r0 r1 c0 c1 (raw-scan pixel coords) to work on a window,
and/or --max-dim to trade resolution for responsiveness.
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RangeSlider

from hsi_workflow.config import DATASETS
from hsi_workflow.cube_io import (open_cube_reader, array_cube_reader, CubeReader,
                                  iter_cube_paths, load_reference_spectrum)
from hsi_workflow.preprocessing import (calibrate_reflectance, savgol_smooth,
                                        baseline_correct, normalize_intensity,
                                        noise_metrics)
from debug_common import Debouncer

MAX_DISPLAY = 600          # default working/display grid cap (long axis, px)


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
    """Returns (cube source, wavelengths, shutter, white/dark spectra or None, label).

    The cube is opened *lazily* (:class:`CubeReader`): only a decimated working
    grid and individual pixel spectra are ever read, so multi-GB scans never have
    to fit in memory. ``--crop`` is pushed into the reader so even the crop is
    read straight off disk. The demo path returns a plain array, which the tuner
    wraps in the same reader interface.
    """
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
    return reader, np.asarray(wl, float), shutter, white, dark, sw, sd, label


# --------------------------------------------------------------------------
# The app
# --------------------------------------------------------------------------

class PreprocessTuner:
    def __init__(self, raw, wl, shutter, white, dark, sw, sd, label, max_dim=MAX_DISPLAY):
        self.wl = wl
        self.label = label

        # Backend-agnostic source: an on-disk CubeReader (big scans) or an
        # ndarray wrapped in the same interface (demo/tests). Nothing here holds
        # the full-res cube -- per-pixel "before" spectra are read on demand.
        source = raw if isinstance(raw, CubeReader) else array_cube_reader(
            raw, wavelengths=wl, shutter=shutter, label=str(label))
        self.source = source
        self.shutter, self.white, self.dark, self.sw, self.sd = shutter, white, dark, sw, sd
        self.rows, self.cols, self.bands = source.shape
        self.can_calibrate = white is not None and dark is not None
        if not self.can_calibrate:
            print("No white/dark references: 'calibrate' toggle disabled, raw DN used.")

        # Decimated *working* grid: all cube-level compute (band image + metrics)
        # runs on this, so the whole scan stays interactive. It is streamed off
        # disk (never the whole cube); clicked-pixel spectra read at full res.
        step = max(1, int(np.ceil(max(self.rows, self.cols) / max_dim)))
        self.step = step
        self._raw_ds = source.decimated(step)
        self._refl_ds = (calibrate_reflectance(self._raw_ds, shutter, white, sw, dark, sd)
                         if self.can_calibrate else self._raw_ds)

        # State
        self.band = self.bands // 2
        self.sg_window, self.sg_polyorder = 11, 2
        self.baseline_order = 2
        self.use_calibrate, self.use_smooth = self.can_calibrate, True
        self.use_snv, self.use_baseline = True, False
        self.pixel = (self.rows // 2, self.cols // 2)
        self.band_clip = (0.0, 1.0)
        self.ref_spectrum = None
        self.use_ref_subtract = False

        self._im = None
        self._build_figure()
        self._debouncer = Debouncer(self.fig.canvas, self._recompute)
        self._recompute()

    # --- processing ---------------------------------------------------

    def _src_ds(self):
        """Decimated 'before' cube for the current calibrate toggle."""
        return self._refl_ds if (self.use_calibrate or not self.can_calibrate) else self._raw_ds

    def _src_pixel(self, r, c):
        """Full-res 'before' spectrum at a pixel (read + calibrated on demand)."""
        raw_px = self.source.pixel(r, c)
        if self.can_calibrate and self.use_calibrate:
            return calibrate_reflectance(raw_px, self.shutter, self.white, self.sw,
                                         self.dark, self.sd)
        return raw_px

    def _process(self, arr):
        """Apply the current settings to any (..., bands) array."""
        data = arr
        if self.use_ref_subtract and self.ref_spectrum is not None:
            data = data - self.ref_spectrum
        if self.use_smooth:
            data = savgol_smooth(data, self.sg_window, self.sg_polyorder)
        if self.use_baseline:
            data = baseline_correct(data, "poly", self.baseline_order)
        if self.use_snv:
            data = normalize_intensity(data, "snv")
        return data

    def _recompute(self):
        self.display_cube = self._process(self._src_ds())
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

        # Display-contrast range slider (light-tier: never recomputes)
        ax_clip = self.fig.add_axes([0.55, 0.20, 0.14, 0.03])
        self.s_clip = RangeSlider(ax_clip, "contrast", 0.0, 1.0, valinit=(0.0, 1.0))
        self.s_clip.on_changed(self._on_clip)

        labels = ["calibrate", "SG smooth", "SNV", "baseline", "subtract ref"]
        state = [self.use_calibrate, self.use_smooth, self.use_snv,
                 self.use_baseline, self.use_ref_subtract]
        ax_checks = self.fig.add_axes([0.55, 0.03, 0.14, 0.15])
        self.checks = CheckButtons(ax_checks, labels, state)
        self.checks.on_clicked(self._on_check)

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.text(0.72, 0.10, "click = pick pixel\nshift+click = set ref\n"
                                  "←/→ = step band\n'p' = print config  'c' = clear ref",
                      fontsize=9, va="center", family="monospace")

    # --- panel updates ---------------------------------------------------

    def _clim_from_frac(self, data, frac):
        dmin, dmax = float(np.nanmin(data)), float(np.nanmax(data))
        span = (dmax - dmin) or 1e-9
        return dmin + frac[0] * span, dmin + frac[1] * span

    def _update_band_image(self):
        band = self.display_cube[:, :, self.band]
        if self._im is None:
            self._im = self.ax_img.imshow(
                band, cmap="magma",
                extent=(0, self.cols, self.rows, 0))
            self._cbar = self.fig.colorbar(self._im, ax=self.ax_img, fraction=0.046)
            self._pixmark, = self.ax_img.plot([], [], "c+", ms=14, mew=2)
        else:
            self._im.set_data(band)
        lo, hi = self._clim_from_frac(band, self.band_clip)
        self._im.set_clim(lo, max(hi, lo + 1e-9))
        r, c = self.pixel
        self._pixmark.set_data([c], [r])
        self.ax_img.set_title(f"band {self.band}  ({self.wl[self.band]:.0f} nm)  "
                              f"[decimation x{self.step}]", fontsize=11)

    def _update_spectrum(self):
        if getattr(self, "_ax2", None) is not None:
            self._ax2.remove()
            self._ax2 = None
        self.ax_spec.clear()
        r, c = self.pixel
        before = self._src_pixel(r, c)
        after = self._process(before[None, :])[0]
        before_label = "before (reflectance)" if self.can_calibrate else "before (raw)"
        self.ax_spec.plot(self.wl, before, color="0.6", lw=1, label=before_label)
        if self.ref_spectrum is not None:
            self.ax_spec.plot(self.wl, self.ref_spectrum, color="tab:green",
                              lw=0.9, ls=":", label="reference")
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
        src = self._src_ds()
        if self.use_ref_subtract and self.ref_spectrum is not None:
            src = src - self.ref_spectrum
        nb = noise_metrics(src, self.sg_window, self.sg_polyorder, sample=3000)
        smoothed = (savgol_smooth(src, self.sg_window, self.sg_polyorder)
                    if self.use_smooth else src)
        na = noise_metrics(smoothed, self.sg_window, self.sg_polyorder, sample=3000)
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
               f"snv={self.use_snv} baseline={self.use_baseline} "
               f"ref_sub={self.use_ref_subtract}")
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
        self._debouncer.mark_dirty()

    def _on_clip(self, _):
        self.band_clip = tuple(self.s_clip.val)
        if self._im is not None:
            band = self.display_cube[:, :, self.band]
            lo, hi = self._clim_from_frac(band, self.band_clip)
            self._im.set_clim(lo, max(hi, lo + 1e-9))
            self.fig.canvas.draw_idle()

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
        elif label == "subtract ref":
            if self.ref_spectrum is None:
                print("No reference set. shift+click a pixel first.")
                return
            self.use_ref_subtract = not self.use_ref_subtract
        self._recompute()          # checks are discrete clicks -> recompute now

    def _set_reference(self, r, c):
        r0, r1 = max(0, r - 2), min(self.rows, r + 3)
        c0, c1 = max(0, c - 2), min(self.cols, c + 3)
        patch = self.source.patch(r0, r1, c0, c1)
        if self.can_calibrate and self.use_calibrate:
            patch = calibrate_reflectance(patch, self.shutter, self.white, self.sw,
                                          self.dark, self.sd)
        self.ref_spectrum = patch.reshape(-1, self.bands).mean(axis=0)

    def _on_click(self, event):
        if event.inaxes is not self.ax_img or event.xdata is None:
            return
        c, r = int(event.xdata), int(event.ydata)
        if not (0 <= r < self.rows and 0 <= c < self.cols):
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

    def _on_key(self, event):
        if event.key == "right":
            self.s_band.set_val(min(self.band + 1, self.bands - 1))
        elif event.key == "left":
            self.s_band.set_val(max(self.band - 1, 0))
        elif event.key == "c":
            self.ref_spectrum = None
            self.use_ref_subtract = False
            self._recompute()
        elif event.key == "p":
            smooth = "savgol" if self.use_smooth else "none"
            baseline = "poly" if self.use_baseline else "none"
            normalize = "snv" if self.use_snv else "none"
            print("\n# paste into WorkflowConfig().preprocess or PreprocessConfig(...):")
            print(f"PreprocessConfig(calibrate={self.use_calibrate}, "
                  f"smooth={smooth!r}, sg_window={self.sg_window}, "
                  f"sg_polyorder={self.sg_polyorder}, baseline={baseline!r}, "
                  f"baseline_order={self.baseline_order}, normalize={normalize!r})")
            if self.use_ref_subtract and self.ref_spectrum is not None:
                print("#   (debug-only: a reference spectrum was subtracted; "
                      "not part of PreprocessConfig)")
            print()


def main():
    p = argparse.ArgumentParser(description="Interactive preprocessing (filter/window) tuner.")
    p.add_argument("--dataset", default="sio2_dish_white_20", choices=sorted(DATASETS))
    p.add_argument("--index", type=int, default=0, help="Which cube of the dataset (0-based).")
    p.add_argument("--hdr", default=None, help="Direct ENVI header path (overrides --dataset).")
    p.add_argument("--white", default=None, help="White reference .hdr (with --hdr).")
    p.add_argument("--dark", default=None, help="Dark reference .hdr (with --hdr).")
    p.add_argument("--crop", type=int, nargs=4, metavar=("R0", "R1", "C0", "C1"),
                   default=None, help="Work on a spatial window of the scan.")
    p.add_argument("--max-dim", type=int, default=MAX_DISPLAY,
                   help="Decimate the working grid to at most this many px on the long axis.")
    p.add_argument("--demo", action="store_true", help="Synthetic cube (no data needed).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    inputs = load_inputs(args)
    PreprocessTuner(*inputs, max_dim=args.max_dim)
    plt.show()


if __name__ == "__main__":
    main()
