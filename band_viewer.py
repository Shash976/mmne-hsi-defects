# band_viewer.py
"""
Interactive band-by-band viewer for a hyperspectral (HSI) cube.

Loads an ENVI-format cube (.hdr/.img or .hdr/.raw pair) via the `spectral`
package and shows a single band at a time, with a slider + arrow keys to
scrub through bands and see how the scene looks at different
wavelengths. Meant as a quick "does this look right" testing tool, not
part of the analysis pipeline.

Usage:
    python band_viewer.py --hdr data/lig/sample1.hdr
    python band_viewer.py --demo          # no data on disk? use a synthetic cube
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox

try:
    import spectral
except ImportError:
    spectral = None


def load_cube(hdr_path):
    """Load an ENVI header file into a (rows, cols, bands) float array + wavelengths (nm)."""
    if spectral is None:
        raise ImportError("The 'spectral' package is required to load ENVI files (pip install spectral).")
    img = spectral.open_image(hdr_path)
    cube = np.asarray(img.load(), dtype=np.float64)
    wavelengths = np.asarray(img.bands.centers, dtype=np.float64) if img.bands is not None and img.bands.centers else None
    return cube, wavelengths


def synthetic_cube(rows=150, cols=200, bands=200, seed=0):
    """Generate a fake cube (a few material patches with distinct spectra) for testing without real data."""
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(400, 1000, bands)

    def gaussian_peak(center, width, height):
        return height * np.exp(-0.5 * ((wavelengths - center) / width) ** 2)

    spec_a = 0.2 + gaussian_peak(550, 40, 0.5) + gaussian_peak(850, 60, 0.3)
    spec_b = 0.15 + gaussian_peak(650, 50, 0.6)
    spec_c = 0.1 + gaussian_peak(950, 30, 0.7)

    yy, xx = np.mgrid[0:rows, 0:cols]
    mix_ab = (xx / cols)
    disc = ((yy - rows * 0.65) ** 2 + (xx - cols * 0.3) ** 2) < (min(rows, cols) * 0.18) ** 2

    cube = np.outer(mix_ab.ravel(), spec_a).reshape(rows, cols, bands) + \
        np.outer(1 - mix_ab.ravel(), spec_b).reshape(rows, cols, bands)
    cube[disc] = spec_c
    cube += rng.normal(0, 0.02, cube.shape)
    return np.clip(cube, 0, None), wavelengths


class BandViewer:
    """Matplotlib-based interactive band viewer with a slider, arrow-key stepping,
    and a jump-to-wavelength text box."""

    def __init__(self, cube, wavelengths=None, cmap="gray", pct_clip=(1, 99)):
        self.cube = cube
        self.wavelengths = wavelengths
        self.n_bands = cube.shape[-1]
        self.band = 0
        self.pct_clip = pct_clip

        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        plt.subplots_adjust(bottom=0.22)

        self.im = self.ax.imshow(self._band_image(self.band), cmap=cmap)
        self.cbar = self.fig.colorbar(self.im, ax=self.ax, fraction=0.046, pad=0.04)
        self._update_title()

        ax_slider = plt.axes([0.15, 0.1, 0.6, 0.03])
        self.slider = Slider(ax_slider, "Band", 0, self.n_bands - 1, valinit=0, valstep=1)
        self.slider.on_changed(self._on_slider)

        ax_box = plt.axes([0.8, 0.1, 0.12, 0.045])
        label = "nm" if wavelengths is not None else "band"
        self.box = TextBox(ax_box, "", initial="")
        self.box.label.set_text(f"Go to {label}:")
        self.box.on_submit(self._on_box_submit)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _band_image(self, band):
        img = self.cube[:, :, band]
        lo, hi = np.percentile(img, self.pct_clip)
        if hi <= lo:
            hi = lo + 1e-6
        return np.clip((img - lo) / (hi - lo), 0, 1)

    def _update_title(self):
        if self.wavelengths is not None:
            wl = self.wavelengths[self.band]
            self.ax.set_title(f"Band {self.band + 1}/{self.n_bands}  ({wl:.1f} nm)")
        else:
            self.ax.set_title(f"Band {self.band + 1}/{self.n_bands}")

    def set_band(self, band):
        self.band = int(np.clip(band, 0, self.n_bands - 1))
        self.im.set_data(self._band_image(self.band))
        self._update_title()
        if self.slider.val != self.band:
            self.slider.eventson = False
            self.slider.set_val(self.band)
            self.slider.eventson = True
        self.fig.canvas.draw_idle()

    def _on_slider(self, val):
        self.set_band(int(val))

    def _on_box_submit(self, text):
        text = text.strip()
        if not text:
            return
        try:
            value = float(text)
        except ValueError:
            return
        if self.wavelengths is not None:
            band = int(np.argmin(np.abs(self.wavelengths - value)))
        else:
            band = int(value)
        self.set_band(band)

    def _on_key(self, event):
        if event.key == "right":
            self.set_band(self.band + 1)
        elif event.key == "left":
            self.set_band(self.band - 1)

    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Interactive band-by-band HSI cube viewer.")
    parser.add_argument("--hdr", help="Path to an ENVI .hdr file.")
    parser.add_argument("--demo", action="store_true", help="Use a synthetic cube instead of loading data.")
    parser.add_argument("--cmap", default="gray", help="Matplotlib colormap name (default: gray).")
    args = parser.parse_args()

    if args.demo or not args.hdr:
        cube, wavelengths = synthetic_cube()
    else:
        cube, wavelengths = load_cube(args.hdr)

    print(f"Cube shape: {cube.shape} (rows, cols, bands)")
    if wavelengths is not None:
        print(f"Wavelength range: {wavelengths[0]:.1f}-{wavelengths[-1]:.1f} nm")

    viewer = BandViewer(cube, wavelengths, cmap=args.cmap)
    viewer.show()


if __name__ == "__main__":
    main()
