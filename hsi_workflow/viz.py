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


_OFF = "0.12"          # dark grey for off-piece background in map panels


def _show_map(ax, arr, cmap, title, fig, mask=None, discrete=False):
    """imshow a masked map with a dark off-piece background + tidy colorbar."""
    m = np.ma.masked_invalid(arr) if mask is None else np.ma.masked_where(~mask, arr)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(_OFF)
    ax.set_facecolor(_OFF)
    im = ax.imshow(m, cmap=cm, interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    if not discrete:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    return im


def save_analysis_figure(analysis, primary: str, threshold: float, out_dir: str) -> str:
    """Interpretable 6-panel analysis figure for one piece.

    ``analysis`` is a ``pipeline.PieceAnalysis``. Panels: spectral structure
    (PC1-3), clustered populations (with legend), anomaly score map, outlined +
    numbered flagged regions, mean spectrum of normal vs anomalous pixels, and the
    anomaly-score histogram with the flag threshold. The spectral panel is the
    scientific payoff -- it shows *how* the flagged regions differ.
    """
    from matplotlib.patches import Patch
    from skimage.measure import find_contours

    os.makedirs(out_dir, exist_ok=True)
    piece = analysis.piece
    mask = piece.mask
    data = piece.data
    flagged = analysis.flagged
    wl = (piece.wavelengths if piece.wavelengths is not None
          else np.arange(data.shape[-1], dtype=float))
    amap = analysis.anomaly_maps[primary]
    pc = analysis.pc_score_image

    n_fg = max(1, int(mask.sum()))
    frac = float(flagged.sum()) / n_fg
    sil = analysis.cluster_metrics.get("silhouette", float("nan"))
    n_reg = len(analysis.regions)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    fig.suptitle(f"{piece.piece_id}   [{piece.material}]    "
                 f"anomalous {frac:.1%}  ·  {n_reg} region(s)  ·  "
                 f"silhouette {sil:.2f}", fontsize=14, y=0.99)

    # (A) spectral structure: PC1-3 as false colour, stretched, off-piece dark.
    pc3 = pc[:, :, :3].astype(np.float64)
    lo, hi = np.nanpercentile(pc3, 2), np.nanpercentile(pc3, 98)
    disp = np.clip((pc3 - lo) / (hi - lo + 1e-12), 0, 1)
    disp = np.where(mask[:, :, None], disp, np.nan)
    axes[0, 0].set_facecolor(_OFF)
    axes[0, 0].imshow(np.ma.masked_invalid(disp))
    axes[0, 0].set_title("Spectral structure (PC1-3 false colour)", fontsize=11)
    axes[0, 0].axis("off")

    # (B) clusters as discrete populations + legend.
    cmap_arr = analysis.cluster_map
    present = sorted(int(v) for v in np.unique(cmap_arr) if v >= 0)
    _show_map(axes[0, 1], cmap_arr.astype(float), "tab10",
              f"Clusters ({len(present)} spectral populations)", fig,
              mask=(cmap_arr >= 0), discrete=True)
    tab = plt.get_cmap("tab10")
    axes[0, 1].legend(handles=[Patch(color=tab(c % 10), label=f"cluster {c}") for c in present],
                      fontsize=8, loc="lower right", framealpha=0.8)

    # (C) anomaly score heatmap.
    _show_map(axes[0, 2], amap, "inferno", f"Anomaly score ({primary})", fig, mask=mask)

    # (D) flagged regions outlined + numbered over a grey PC1 base.
    base = np.where(mask, pc[:, :, 0], np.nan)
    axes[1, 0].set_facecolor(_OFF)
    gm = plt.get_cmap("gray").copy(); gm.set_bad(_OFF)
    axes[1, 0].imshow(np.ma.masked_invalid(base), cmap=gm)
    if flagged.any():
        for contour in find_contours(flagged.astype(float), 0.5):
            axes[1, 0].plot(contour[:, 1], contour[:, 0], color="#ff3b3b", lw=1.5)
        for r in analysis.regions:
            axes[1, 0].text(r.centroid[1], r.centroid[0], str(r.region_id),
                            color="#ffe000", fontsize=9, ha="center", va="center",
                            fontweight="bold")
        d_title = f"Flagged anomalies ({n_reg} region(s), outlined)"
    else:
        axes[1, 0].text(0.5, 0.5, "no anomalies flagged", transform=axes[1, 0].transAxes,
                        ha="center", va="center", color="white", fontsize=12)
        d_title = "Flagged anomalies (none)"
    axes[1, 0].set_title(d_title, fontsize=11)
    axes[1, 0].axis("off")

    # (E) mean spectrum: normal vs anomalous -- the "why".
    ax = axes[1, 1]
    normal = data[mask & ~flagged]
    if normal.size:
        mu = normal.mean(axis=0); sd = normal.std(axis=0)
        ax.plot(wl, mu, color="tab:blue", label=f"normal (n={normal.shape[0]})")
        ax.fill_between(wl, mu - sd, mu + sd, color="tab:blue", alpha=0.15)
    if flagged.any():
        anom = data[flagged]
        mu = anom.mean(axis=0); sd = anom.std(axis=0)
        ax.plot(wl, mu, color="tab:red", label=f"anomalous (n={anom.shape[0]})")
        ax.fill_between(wl, mu - sd, mu + sd, color="tab:red", alpha=0.15)
    ax.set_title("Mean spectrum: normal vs anomalous", fontsize=11)
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("SNV-normalized reflectance")
    ax.legend(fontsize=8)

    # (F) anomaly score histogram with threshold.
    ax = axes[1, 2]
    scores = amap[mask]
    scores = scores[np.isfinite(scores)]
    ax.hist(scores, bins=60, color="0.6")
    ax.axvline(threshold, color="#ff3b3b", ls="--", label=f"threshold ({threshold:.2f})")
    ax.set_title("Anomaly score distribution", fontsize=11)
    ax.set_xlabel(f"{primary} score"); ax.set_ylabel("pixels")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, f"{piece.piece_id}_analysis.png")
    plt.savefig(path, dpi=140)
    plt.close(fig)
    return path
