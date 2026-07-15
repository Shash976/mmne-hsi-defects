"""Preview panels for the workflow stages (preprocessing + analysis maps)."""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .preprocessing import Preprocessed


def pseudo_rgb(cube: np.ndarray, wavelengths: Optional[np.ndarray],
               targets=(650.0, 550.0, 450.0)) -> np.ndarray:
    """Stretch three bands near ``targets`` (nm) into a display RGB.

    Falls back to evenly spaced band indices when wavelengths are unavailable.
    """
    bands = cube.shape[-1]
    if wavelengths is not None:
        idx = [int(np.argmin(np.abs(np.asarray(wavelengths) - t))) for t in targets]
    else:
        idx = [int(bands * f) for f in (0.7, 0.5, 0.3)]
    rgb = cube[:, :, idx].astype(np.float64)
    lo, hi = np.nanpercentile(rgb, 2), np.nanpercentile(rgb, 98)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-12), 0, 1)
    return rgb


def _band_index(wavelengths, target, n_bands):
    if wavelengths is not None:
        return int(np.argmin(np.abs(np.asarray(wavelengths) - target)))
    return n_bands // 2


def save_preprocess_preview(pre: Preprocessed, od_cube: np.ndarray,
                            out_dir: str, od_band_nm: float = 550.0) -> str:
    """Five-panel preview: pseudo-RGB, film/substrate segmentation, an OD band
    map, and the mean OD spectra for film vs. substrate. Saved as a PNG.
    """
    os.makedirs(out_dir, exist_ok=True)
    rgb = pseudo_rgb(pre.data, pre.wavelengths)
    seg = pre.segmentation
    b = _band_index(pre.wavelengths, od_band_nm, od_cube.shape[-1])

    od_band = od_cube[:, :, b]
    od_display = np.ma.masked_invalid(od_band)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4))

    axes[0].imshow(rgb)
    axes[0].set_title(f"{pre.label}\npseudo-RGB")

    seg_img = np.full(seg.foreground.shape, np.nan)
    seg_img[seg.substrate] = 0
    seg_img[seg.foreground] = 1
    axes[1].imshow(rgb)
    axes[1].imshow(np.ma.masked_invalid(seg_img), cmap="cool", alpha=0.45, vmin=0, vmax=1)
    axes[1].set_title("Segmentation\n(film = magenta)")

    im = axes[2].imshow(od_display, cmap="inferno")
    axes[2].set_title(f"Optical density\n@ {od_band_nm:.0f} nm")
    fig.colorbar(im, ax=axes[2], fraction=0.046)

    wl = (pre.wavelengths if pre.wavelengths is not None
          else np.arange(od_cube.shape[-1], dtype=float))
    if seg.foreground.sum():
        axes[3].plot(wl, od_cube[seg.foreground].mean(axis=0), label="film", color="tab:red")
    if seg.substrate.sum():
        axes[3].plot(wl, od_cube[seg.substrate].mean(axis=0), label="substrate", color="tab:blue")
    axes[3].axhline(0, color="k", lw=0.6, ls="--")
    axes[3].set_title("Mean OD spectrum")
    axes[3].set_xlabel("wavelength (nm)")
    axes[3].set_ylabel("OD")
    axes[3].legend(fontsize=8)

    for ax in axes[:3]:
        ax.axis("off")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{pre.label}_preprocess.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Analysis-stage panels (Stages 5-10): PCA summary + per-piece maps
# --------------------------------------------------------------------------

def save_pca_summary(explained_variance_ratio: np.ndarray, loadings: np.ndarray,
                     wavelengths: Optional[np.ndarray], out_dir: str,
                     name: str = "pca") -> str:
    """Explained-variance bar chart + PC loading curves (Stage 5 deliverable)."""
    os.makedirs(out_dir, exist_ok=True)
    k = loadings.shape[0]
    wl = wavelengths if wavelengths is not None else np.arange(loadings.shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(np.arange(1, k + 1), explained_variance_ratio[:k])
    axes[0].set_xlabel("principal component")
    axes[0].set_ylabel("explained variance ratio")
    axes[0].set_title("Explained variance")
    for i in range(k):
        axes[1].plot(wl, loadings[i], label=f"PC{i + 1}")
    axes[1].set_xlabel("wavelength (nm)")
    axes[1].set_ylabel("loading")
    axes[1].set_title("PC loadings")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{name}_summary.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_analysis_maps(label: str, rgb: np.ndarray, pc_scores: np.ndarray,
                       cluster_map: np.ndarray, anomaly_map: np.ndarray,
                       flagged: np.ndarray, out_dir: str) -> str:
    """Per-piece spatial deliverables: RGB, PC1-3, cluster map, anomaly heatmap,
    flagged-region overlay. All maps are masked where off-piece (NaN / -1)."""
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"{label}\npseudo-RGB")

    pc = pc_scores.copy()
    lo, hi = np.nanpercentile(pc, 2), np.nanpercentile(pc, 98)
    pc_disp = np.clip((pc[:, :, :3] - lo) / (hi - lo + 1e-12), 0, 1)
    axes[0, 1].imshow(pc_disp)
    axes[0, 1].set_title("PC1-3 (RGB)")

    cmap_masked = np.ma.masked_where(cluster_map < 0, cluster_map)
    im = axes[0, 2].imshow(cmap_masked, cmap="tab10")
    axes[0, 2].set_title("cluster map")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046)

    im = axes[1, 0].imshow(np.ma.masked_invalid(anomaly_map), cmap="inferno")
    axes[1, 0].set_title("anomaly heatmap")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].imshow(rgb)
    axes[1, 1].imshow(np.ma.masked_where(~flagged, flagged), cmap="autumn", alpha=0.6)
    axes[1, 1].set_title("flagged anomalous regions")

    single = pc_scores[:, :, 0]
    im = axes[1, 2].imshow(np.ma.masked_invalid(single), cmap="viridis")
    axes[1, 2].set_title("PC1 score map")
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046)

    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{label}_analysis.png")
    plt.savefig(path, dpi=140)
    plt.close(fig)
    return path
