"""Slide-10 (Stage 4 exploratory) figures — sample imagery, not just plots.

fig10  specimen gallery: pseudo-RGB of all 24 extracted pieces
fig11  raw dish scan with the extracted pieces outlined (what the camera sees
       vs. what the front-end cuts out)
fig12  the Stage-4 six-panel for one representative SiO2 piece
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spectral
from matplotlib.patches import Rectangle

REPO = r"C:\Users\shash\Desktop\Code\MMNE\HSI"
OUT = os.path.join(REPO, "out", "figures", "symposium")
EXT = os.path.join(REPO, "out", "workflow", "extract")
os.makedirs(OUT, exist_ok=True)

SI = "#4A6E8A"
SIO2 = "#1F7A6B"
ACCENT = "#D9822B"
INK = "#243440"
MUTED = "#8A97A0"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.edgecolor": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 200, "savefig.bbox": "tight",
})

samples = pd.read_csv(os.path.join(REPO, "data", "samples.csv"))


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)


def pseudo_rgb(cube, wl, targets=(650.0, 550.0, 450.0), mask=None):
    """Three-band stretch as in hsi_workflow/viz.py.

    ``mask`` restricts the percentile stretch to in-piece pixels. Without it the
    bright dish sets the range and the wafers crush to black — fine for the
    whole-scan view, wrong for a per-piece gallery.
    """
    idx = [int(np.argmin(np.abs(np.asarray(wl) - t))) for t in targets]
    rgb = cube[:, :, idx].astype(np.float64)
    ref = rgb[mask] if mask is not None else rgb
    lo, hi = np.nanpercentile(ref, 2), np.nanpercentile(ref, 98)
    return np.clip((rgb - lo) / (hi - lo + 1e-12), 0, 1)


def load_piece(dataset, pid):
    d = os.path.join(EXT, dataset, pid)
    img = spectral.open_image(os.path.join(d, pid + ".hdr"))
    wl = np.asarray(img.bands.centers, dtype=float)
    arr = np.asarray(img.load(), dtype=np.float64)
    mask = np.load(os.path.join(d, pid + "_mask.npy")).astype(bool)
    meta = json.load(open(os.path.join(d, "meta.json")))
    return arr, wl, mask, meta


# ================================================================ FIG 10
def fig_gallery():
    rows = [r for _, r in samples.iterrows()
            if os.path.isdir(os.path.join(EXT, r.dataset, r.sample_id))]
    si = [r for r in rows if r.material == "silicon"]
    ox = [r for r in rows if r.material != "silicon"]

    ncol = 9
    fig = plt.figure(figsize=(16.5, 7.6))
    gs = fig.add_gridspec(3, ncol, hspace=0.40, wspace=0.14,
                          left=0.055, right=0.995, top=0.845, bottom=0.075)

    def draw(r, gpos, edge):
        arr, wl, mask, _ = load_piece(r.dataset, r.sample_id)
        rgb = pseudo_rgb(arr, wl, mask=mask)          # stretch on the piece only
        rgba = np.dstack([rgb, mask.astype(float)])   # background transparent
        ax = fig.add_subplot(gs[gpos])
        ax.imshow(rgba)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(edge); s.set_linewidth(1.8)
        short = (r.sample_id.replace("bare silicon all_", "Si ")
                 .replace("sio2 all 20 dish white_", "SiO\u2082 "))
        ax.set_title(f"{short}\n{r.area_px:,} px \u00b7 {r.n_rois} ROIs",
                     fontsize=8.4, fontweight="normal", pad=3.0, color=INK)

    for i, r in enumerate(si):
        draw(r, (0, i), SI)
    for i, r in enumerate(ox):
        draw(r, (1, i) if i < ncol else (2, i - ncol), SIO2)

    fig.text(0.005, 0.845, f"Bare silicon\ncontrol\n(n={len(si)})", fontsize=11.5,
             fontweight="bold", color=SI, va="top", linespacing=1.5)
    fig.text(0.005, 0.545, f"Processed\nSiO\u2082\n(n={len(ox)})", fontsize=11.5,
             fontweight="bold", color=SIO2, va="top", linespacing=1.5)
    fig.suptitle("Every specimen the pipeline analysed",
                 fontsize=18, fontweight="bold", color=INK, x=0.005, ha="left",
                 y=0.975)
    fig.text(0.005, 0.040,
             "Pseudo-RGB (650 / 550 / 450 nm) of each extracted piece cube, "
             "background masked out and the contrast stretch computed on the "
             "piece only.",
             fontsize=10.5, color=MUTED, style="italic")
    fig.text(0.005, 0.008,
             "The interference colour and device patterning on the SiO\u2082 pieces "
             "is the thin-film signal; the bare-Si controls are featureless by "
             "comparison. Panels are not to a common scale.",
             fontsize=10.5, color=MUTED, style="italic")
    save(fig, "fig10_specimen_gallery.png")


# ================================================================ FIG 11
def fig_scan_overlay():
    """The raw dish scan, decimated, with each extracted piece outlined."""
    import sys
    sys.path.insert(0, REPO)
    from hsi_workflow.cube_io import open_cube_reader

    hdr = (r"C:\Users\shash\OneDrive - purdue.edu\Summer\HSI\sio2"
           r"\sio2 all 20 dish white.bil.hdr")
    step = 3
    rd = open_cube_reader(hdr, material="sio2")
    print("scan shape", rd.shape, "-> decimating by", step)
    cube = rd.decimated(step)
    rgb = pseudo_rgb(cube, rd.wavelengths)
    print("preview", rgb.shape)

    ox = [r for _, r in samples.iterrows() if r.material != "silicon"]
    boxes = []
    for r in ox:
        mp = os.path.join(EXT, r.dataset, r.sample_id, "meta.json")
        if os.path.exists(mp):
            m = json.load(open(mp))
            boxes.append((r.sample_id[-3:],
                          [v / step for v in m["bbox_in_scan"]]))

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 8.2))
    for ax, titled in zip(axes, (False, True)):
        ax.imshow(rgb)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_title("what the camera records\none scan, many pieces",
                      loc="left", fontsize=12.5)
    axes[1].set_title(f"what the front-end returns\n{len(boxes)} specimens, "
                      "cut spectrally", loc="left", fontsize=12.5)

    for lbl, (r0, r1, c0, c1) in boxes:
        axes[1].add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False,
                                    edgecolor=ACCENT, lw=1.6))
        axes[1].text((c0 + c1) / 2, (r0 + r1) / 2, lbl, fontsize=7.5,
                     color="white", fontweight="bold", ha="center",
                     va="center",
                     bbox=dict(boxstyle="round,pad=0.16", fc=ACCENT,
                               ec="none", alpha=0.92))

    fig.suptitle("Stage 3.1 piece extraction, on the raw dish scan",
                 fontsize=17, fontweight="bold", color=INK, x=0.005, ha="left",
                 y=0.985)
    fig.text(0.005, 0.052,
             f"Pseudo-RGB of the full {rd.shape[0]}\u00d7{rd.shape[1]}\u00d7"
             f"{rd.shape[2]} cube, shown decimated {step}\u00d7. Boxes are what the "
             "spectral front-end returned \u2014 no RGB thresholding, no manual "
             "selection. Each box becomes its own sub-cube and its own "
             "hold-out unit.",
             fontsize=10, color=MUTED, style="italic")
    fig.text(0.005, 0.005,
             "Note the unboxed pieces: the shipped SAM backend is "
             "brightness-invariant and misses low-contrast wafers \u2014 the "
             "extraction gap on the limitations slide, visible here.",
             fontsize=10, color=ACCENT, style="italic")
    fig.tight_layout(rect=[0, 0.085, 1, 0.945])
    save(fig, "fig11_scan_extraction.png")


# ================================================================ FIG 12
def fig_stage4(pid="sio2 all 20 dish white_p03", dataset="sio2_dish_white_20"):
    arr, wl, mask, meta = load_piece(dataset, pid)
    rgb = pseudo_rgb(arr, wl, mask=mask)
    rgba = np.dstack([rgb, mask.astype(float)])
    var = np.where(mask, arr.var(axis=-1), np.nan)
    spec = arr[mask]
    mu, sd = spec.mean(axis=0), spec.std(axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))

    ax = axes[0, 0]
    ax.imshow(rgba)
    ax.set_title("pseudo-RGB (650/550/450 nm)", fontsize=12.5)

    for ax, target in zip((axes[0, 1], axes[0, 2], axes[1, 0]),
                          (450.0, 650.0, 850.0)):
        b = int(np.argmin(np.abs(wl - target)))
        band = np.where(mask, arr[:, :, b], np.nan)
        im = ax.imshow(np.ma.masked_invalid(band), cmap="gray")
        ax.set_title(f"band @ {wl[b]:.0f} nm", fontsize=12.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=9)

    ax = axes[1, 1]
    im = ax.imshow(np.ma.masked_invalid(var), cmap="magma")
    ax.set_title("spectral variance map", fontsize=12.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=9)

    for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]):
        ax.axis("off")

    ax = axes[1, 2]
    ax.plot(wl, mu, color=SIO2, lw=2.6)
    ax.fill_between(wl, mu - sd, mu + sd, color=SIO2, alpha=0.20, lw=0)
    for t, col in ((450.0, MUTED), (650.0, MUTED), (850.0, MUTED)):
        ax.axvline(t, color=col, ls=":", lw=1.2)
        ax.text(t + 6, ax.get_ylim()[1], f"{t:.0f}", fontsize=9, color=col,
                va="top")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("reflectance")
    ax.set_title(f"mean spectrum \u00b1 1\u03c3  (n={len(spec):,} px)",
                 fontsize=12.5)
    ax.set_xlim(wl.min(), wl.max())
    ax.yaxis.grid(True, lw=0.8); ax.set_axisbelow(True)

    short = pid.replace("sio2 all 20 dish white_", "SiO\u2082 ")
    fig.suptitle(f"Stage 4 exploratory visualisation \u2014 {short}",
                 fontsize=18, fontweight="bold", color=INK, x=0.005, ha="left")
    fig.text(0.005, 0.005,
             "Every panel comes from the same 300-band cube. This is what we "
             "look at before any ML runs: is the piece cleanly cut out, is the "
             "reflectance in range, and does spatial structure show up in the "
             "spectra rather than only in brightness?",
             fontsize=10.5, color=MUTED, style="italic")
    fig.tight_layout(rect=[0, 0.035, 1, 0.945])
    save(fig, "fig12_stage4_panel.png")


if __name__ == "__main__":
    fig_gallery()
    fig_stage4()
    fig_scan_overlay()
    print("\ndone ->", OUT)
