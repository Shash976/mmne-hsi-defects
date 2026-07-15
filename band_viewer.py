# band_viewer.py
"""
Interactive multi-cube hyperspectral (HSI) band viewer.

Loads one or more ENVI-format cubes (.hdr/.img or .hdr/.raw pairs) via the
`spectral` package and shows them side by side, one band at a time, driven
by a single shared slider. Meant as a quick "does this look right" /
calibration-sanity testing tool, not part of the analysis pipeline.

Cubes are compared by wavelength (nm), not raw band index, so cubes with
different band counts or spectral sampling still line up correctly. If a
cube has no wavelength metadata it falls back to a fractional position
through its own band range.

Features:
    - N cubes displayed side by side, one shared slider (spectral position)
    - difference panels for calibration (e.g. raw - dark), via --diff
    - per-cube mask overlay: either a supplied boolean .npy mask, or a live
      adjustable percentile threshold on the current band
    - keyboard: left/right step bands, 'm' toggle mask, '[' ']' adjust mask
      threshold, 'c' cycle colormap, 'g' then digits + Enter to jump to a
      wavelength (or band index if no wavelengths), Esc cancels

Usage:
    python band_viewer.py --hdr data/lig/raw.hdr data/lig/dark.hdr \
        --labels raw dark --diff raw,dark

    python band_viewer.py --demo 2       # no data on disk? synthesize N cubes
    python band_viewer.py --hdr scan.hdr --mask scan=data/lig/roi_mask.npy
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from matplotlib.colors import TwoSlopeNorm

try:
    import spectral
except ImportError:
    spectral = None


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

class NamedCube:
    """A cube plus the display metadata the viewer needs."""

    def __init__(self, label, data, wavelengths, cmap="gray", is_diff=False, mask=None):
        self.label = label
        self.data = data
        self.wavelengths = wavelengths  # 1D array (nm) or None
        self.cmap = cmap
        self.is_diff = is_diff
        self.external_mask = mask  # 2D bool array or None, spatial only

    @property
    def n_bands(self):
        return self.data.shape[-1]

    def band_for_t(self, t, global_wl_range):
        """Map a spectral position t in [0, 1] to this cube's own band index."""
        if self.wavelengths is not None and global_wl_range is not None:
            lo, hi = global_wl_range
            target_nm = lo + t * (hi - lo)
            return int(np.argmin(np.abs(self.wavelengths - target_nm)))
        return int(round(t * (self.n_bands - 1)))


def load_cube(hdr_path):
    """Load an ENVI header file into a (rows, cols, bands) float array + wavelengths (nm)."""
    if spectral is None:
        raise ImportError("The 'spectral' package is required to load ENVI files (pip install spectral).")
    img = spectral.open_image(hdr_path)
    data = np.asarray(img.load(), dtype=np.float64)
    wavelengths = np.asarray(img.bands.centers, dtype=np.float64) if img.bands is not None and img.bands.centers else None
    return data, wavelengths


def synthetic_cube(rows=150, cols=200, bands=200, seed=0, scale=1.0, noise=0.02):
    """Generate a fake cube (a few material patches with distinct spectra) for testing without real data."""
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(400, 1000, bands)

    def gaussian_peak(center, width, height):
        return height * np.exp(-0.5 * ((wavelengths - center) / width) ** 2)

    spec_a = 0.2 + gaussian_peak(550, 40, 0.5) + gaussian_peak(850, 60, 0.3)
    spec_b = 0.15 + gaussian_peak(650, 50, 0.6)
    spec_c = 0.1 + gaussian_peak(950, 30, 0.7)

    yy, xx = np.mgrid[0:rows, 0:cols]
    mix_ab = xx / cols
    disc = ((yy - rows * 0.65) ** 2 + (xx - cols * 0.3) ** 2) < (min(rows, cols) * 0.18) ** 2

    cube = np.outer(mix_ab.ravel(), spec_a).reshape(rows, cols, bands) + \
        np.outer(1 - mix_ab.ravel(), spec_b).reshape(rows, cols, bands)
    cube[disc] = spec_c
    cube = cube * scale + rng.normal(0, noise, cube.shape)
    return np.clip(cube, 0, None), wavelengths


# --------------------------------------------------------------------------
# Viewer
# --------------------------------------------------------------------------

class BandExplorer:
    """Side-by-side multi-cube band viewer with a single shared spectral slider."""

    COLORMAPS = ["gray", "viridis", "inferno"]

    def __init__(self, cubes):
        if not cubes:
            raise ValueError("BandExplorer needs at least one cube.")
        self.cubes = cubes
        self.cmap_idx = 0
        self.show_mask = False
        self.mask_pct = 50.0
        self.t = 0.0
        self.typing = False
        self.type_buffer = ""

        wl_mins = [c.wavelengths.min() for c in cubes if c.wavelengths is not None]
        wl_maxs = [c.wavelengths.max() for c in cubes if c.wavelengths is not None]
        self.global_wl_range = (min(wl_mins), max(wl_maxs)) if wl_mins else None

        n = len(cubes)
        self.fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 5.2), squeeze=False)
        self.axes = axes[0]
        plt.subplots_adjust(bottom=0.24, top=0.88, wspace=0.35)

        self.images = []
        self.mask_overlays = []
        for ax, cube in zip(self.axes, cubes):
            cmap = "RdBu_r" if cube.is_diff else self.COLORMAPS[self.cmap_idx]
            im = ax.imshow(np.zeros(cube.data.shape[:2]), cmap=cmap)
            self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            overlay = ax.imshow(np.zeros((*cube.data.shape[:2], 4)), zorder=5)
            ax.set_xticks([])
            ax.set_yticks([])
            self.images.append(im)
            self.mask_overlays.append(overlay)

        ax_slider = self.fig.add_axes([0.15, 0.1, 0.55, 0.03])
        self.slider = Slider(ax_slider, "Spectral pos.", 0.0, 1.0, valinit=0.0)
        self.slider.on_changed(self._on_slider)

        ax_check = self.fig.add_axes([0.78, 0.02, 0.15, 0.08])
        self.check = CheckButtons(ax_check, ["show mask"], [False])
        self.check.on_clicked(self._on_check)

        self.status_text = self.fig.text(0.15, 0.05, "", fontsize=9, family="monospace")
        self.help_text = self.fig.text(
            0.01, 0.97,
            "←/→ step band  |  m mask  |  [ ] mask thresh  |  c colormap  |  g<value>Enter jump  |  Esc cancel",
            fontsize=8, color="dimgray",
        )

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._render()

    # -- data --------------------------------------------------------------

    def _band_image(self, cube, band):
        img = cube.data[:, :, band]
        if cube.is_diff:
            vmax = np.percentile(np.abs(img), 99) or 1e-6
            return img, TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
        lo, hi = np.percentile(img, (1, 99))
        if hi <= lo:
            hi = lo + 1e-6
        return np.clip((img - lo) / (hi - lo), 0, 1), None

    def _mask_for(self, cube, img):
        if cube.external_mask is not None:
            return cube.external_mask
        cutoff = np.percentile(img, self.mask_pct)
        return img >= cutoff

    # -- render --------------------------------------------------------------

    def _render(self):
        for ax, cube, im, overlay in zip(self.axes, self.cubes, self.images, self.mask_overlays):
            band = cube.band_for_t(self.t, self.global_wl_range)
            display, norm = self._band_image(cube, band)
            im.set_data(display)
            if norm is not None:
                im.set_norm(norm)

            if self.show_mask:
                mask = self._mask_for(cube, cube.data[:, :, band])
                rgba = np.zeros((*mask.shape, 4))
                rgba[mask] = (1.0, 0.15, 0.15, 0.45)
                overlay.set_data(rgba)
            else:
                overlay.set_data(np.zeros((*cube.data.shape[:2], 4)))

            if cube.wavelengths is not None:
                ax.set_title(f"{cube.label}\nband {band + 1}/{cube.n_bands}  ({cube.wavelengths[band]:.1f} nm)")
            else:
                ax.set_title(f"{cube.label}\nband {band + 1}/{cube.n_bands}")

        status = f"mask: {'on' if self.show_mask else 'off'} (pct={self.mask_pct:.0f})"
        if self.typing:
            status += f"   go to: {self.type_buffer}_"
        self.status_text.set_text(status)
        self.fig.canvas.draw_idle()

    def set_t(self, t):
        self.t = float(np.clip(t, 0.0, 1.0))
        if abs(self.slider.val - self.t) > 1e-9:
            self.slider.eventson = False
            self.slider.set_val(self.t)
            self.slider.eventson = True
        self._render()

    def step(self, direction):
        max_bands = max(c.n_bands for c in self.cubes)
        self.set_t(self.t + direction / (max_bands - 1))

    # -- widget callbacks ----------------------------------------------------

    def _on_slider(self, val):
        self.set_t(val)

    def _on_check(self, label):
        self.show_mask = not self.show_mask
        self._render()

    def _cycle_colormap(self):
        self.cmap_idx = (self.cmap_idx + 1) % len(self.COLORMAPS)
        for cube, im in zip(self.cubes, self.images):
            if not cube.is_diff:
                im.set_cmap(self.COLORMAPS[self.cmap_idx])
        self._render()

    def _jump_to(self, value):
        if self.global_wl_range is not None:
            lo, hi = self.global_wl_range
            t = (value - lo) / (hi - lo) if hi > lo else 0.0
        else:
            max_bands = max(c.n_bands for c in self.cubes)
            t = value / (max_bands - 1)
        self.set_t(t)

    def _on_key(self, event):
        if self.typing:
            if event.key == "enter":
                self.typing = False
                try:
                    self._jump_to(float(self.type_buffer))
                except ValueError:
                    pass
                self.type_buffer = ""
                self._render()
            elif event.key == "escape":
                self.typing = False
                self.type_buffer = ""
                self._render()
            elif event.key == "backspace":
                self.type_buffer = self.type_buffer[:-1]
                self._render()
            elif event.key is not None and len(event.key) == 1 and (event.key.isdigit() or event.key in ".-"):
                self.type_buffer += event.key
                self._render()
            return

        if event.key == "right":
            self.step(1)
        elif event.key == "left":
            self.step(-1)
        elif event.key == "m":
            self.check.set_active(0)  # triggers _on_check, which flips show_mask and renders
        elif event.key == "[":
            self.mask_pct = max(1.0, self.mask_pct - 5.0)
            self._render()
        elif event.key == "]":
            self.mask_pct = min(99.0, self.mask_pct + 5.0)
            self._render()
        elif event.key == "c":
            self._cycle_colormap()
        elif event.key == "g":
            self.typing = True
            self.type_buffer = ""
            self._render()

    def show(self):
        plt.show()


# --------------------------------------------------------------------------
# CLI plumbing
# --------------------------------------------------------------------------

def _label_for(path):
    return os.path.splitext(os.path.basename(path))[0]


def build_cubes(args):
    cubes = {}
    order = []

    if args.demo is not None:
        for i in range(args.demo):
            label = f"demo{i}"
            scale = 1.0 if i == 0 else 0.9
            seed = i
            data, wl = synthetic_cube(seed=seed, scale=scale)
            cubes[label] = NamedCube(label, data, wl)
            order.append(label)
    else:
        if not args.hdr:
            raise SystemExit("Provide --hdr (one or more) or --demo N.")
        labels = args.labels if args.labels else [_label_for(p) for p in args.hdr]
        if len(labels) != len(args.hdr):
            raise SystemExit("--labels must have the same count as --hdr.")
        for label, path in zip(labels, args.hdr):
            data, wl = load_cube(path)
            cubes[label] = NamedCube(label, data, wl)
            order.append(label)

    for spec in args.mask or []:
        label, _, mask_path = spec.partition("=")
        if label not in cubes:
            raise SystemExit(f"--mask references unknown label '{label}'. Known labels: {order}")
        mask = np.load(mask_path).astype(bool)
        if mask.shape != cubes[label].data.shape[:2]:
            raise SystemExit(f"Mask '{mask_path}' shape {mask.shape} does not match cube '{label}' spatial shape {cubes[label].data.shape[:2]}.")
        cubes[label].external_mask = mask

    diff_cubes = []
    for spec in args.diff or []:
        a_label, _, b_label = spec.partition(",")
        for lbl in (a_label, b_label):
            if lbl not in cubes:
                raise SystemExit(f"--diff references unknown label '{lbl}'. Known labels: {order}")
        a, b = cubes[a_label], cubes[b_label]
        if a.data.shape != b.data.shape:
            raise SystemExit(
                f"--diff {spec}: cubes must have identical shape to subtract "
                f"({a_label}={a.data.shape} vs {b_label}={b.data.shape})."
            )
        diff_data = a.data - b.data
        diff_wl = a.wavelengths if a.wavelengths is not None else b.wavelengths
        diff_cubes.append(NamedCube(f"{a_label} - {b_label}", diff_data, diff_wl, is_diff=True))

    return [cubes[label] for label in order] + diff_cubes


def main():
    parser = argparse.ArgumentParser(description="Interactive multi-cube HSI band viewer.")
    parser.add_argument("--hdr", nargs="+", help="Path(s) to ENVI .hdr file(s).")
    parser.add_argument("--labels", nargs="+", help="Labels for each --hdr cube, same order/count.")
    parser.add_argument("--demo", type=int, nargs="?", const=2, default=None,
                         help="Use N synthetic cubes instead of loading data (default 2 if flag given with no value).")
    parser.add_argument("--mask", action="append",
                         help="LABEL=path/to/mask.npy — boolean spatial mask overlay for a cube. Repeatable.")
    parser.add_argument("--diff", action="append",
                         help="LABEL_A,LABEL_B — add a live A-B difference panel (cubes must match shape). Repeatable.")
    args = parser.parse_args()

    cubes = build_cubes(args)

    for cube in cubes:
        print(f"{cube.label}: shape={cube.data.shape}" +
              (f", wl={cube.wavelengths[0]:.1f}-{cube.wavelengths[-1]:.1f}nm" if cube.wavelengths is not None else ""))

    explorer = BandExplorer(cubes)
    explorer.show()


if __name__ == "__main__":
    main()
