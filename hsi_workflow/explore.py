"""Stage 4 -- Exploratory spectral visualization.

This stage matters more than usual because it replaces the missing reference
library: before any ML, look at the data. For each piece it produces a mean
spectrum, band images at a few wavelengths, an RGB composite, and a spectral
variance map. The key sanity check the document calls for: silicon should look
spectrally *homogeneous* (low variance) and processed SiO2 more *heterogeneous*.

All maps are computed from full spectra; the RGB panel is display-only.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pieces import Piece
from viz import pseudo_rgb


def spectral_variance_map(cube: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-pixel spectral variance (variance across bands) -- heterogeneity proxy.

    High where a pixel's spectrum has strong structure/contrast across
    wavelengths. Off-mask pixels are set to NaN so they don't skew the display.
    """
    var = cube.var(axis=-1)
    if mask is not None:
        var = np.where(mask, var, np.nan)
    return var


def mean_spectrum(piece: Piece) -> np.ndarray:
    """Mean reflectance spectrum over the piece's in-mask pixels."""
    return piece.foreground_spectra().mean(axis=0)


def save_piece_exploration(piece: Piece, out_dir: str,
                           band_targets: Sequence[float] = (450.0, 650.0, 850.0)) -> str:
    """Six-panel Stage-4 figure for one piece; returns the PNG path."""
    os.makedirs(out_dir, exist_ok=True)
    wl = (piece.wavelengths if piece.wavelengths is not None
          else np.arange(piece.n_bands, dtype=float))
    rgb = pseudo_rgb(piece.data, piece.wavelengths)
    var_map = spectral_variance_map(piece.data, piece.mask)
    mean_spec = mean_spectrum(piece)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"{piece.piece_id} ({piece.material})\npseudo-RGB")

    # Band images at requested wavelengths.
    for ax, target in zip(axes[0, 1:], band_targets[:2]):
        b = int(np.argmin(np.abs(wl - target)))
        band = np.where(piece.mask, piece.data[:, :, b], np.nan)
        im = ax.imshow(np.ma.masked_invalid(band), cmap="gray")
        ax.set_title(f"band @ {wl[b]:.0f} nm")
        fig.colorbar(im, ax=ax, fraction=0.046)

    im = axes[1, 0].imshow(np.ma.masked_invalid(var_map), cmap="magma")
    axes[1, 0].set_title("spectral variance map")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)

    b = int(np.argmin(np.abs(wl - band_targets[-1])))
    band = np.where(piece.mask, piece.data[:, :, b], np.nan)
    im = axes[1, 1].imshow(np.ma.masked_invalid(band), cmap="gray")
    axes[1, 1].set_title(f"band @ {wl[b]:.0f} nm")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)

    axes[1, 2].plot(wl, mean_spec, color="tab:blue")
    axes[1, 2].set_title("mean spectrum (in-mask)")
    axes[1, 2].set_xlabel("wavelength (nm)")
    axes[1, 2].set_ylabel("reflectance")

    for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]):
        ax.axis("off")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{piece.piece_id}_explore.png")
    plt.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_material_mean_spectra(pieces: List[Piece], out_dir: str) -> str:
    """Overlay mean spectra grouped by material (Si baseline vs SiO2)."""
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    colors = {"silicon": "tab:blue", "sio2": "tab:red", "lig": "tab:green"}
    seen = set()
    for p in pieces:
        wl = (p.wavelengths if p.wavelengths is not None
              else np.arange(p.n_bands, dtype=float))
        label = p.material if p.material not in seen else None
        seen.add(p.material)
        plt.plot(wl, mean_spectrum(p), color=colors.get(p.material, "gray"),
                 alpha=0.5, lw=1, label=label)
    plt.xlabel("wavelength (nm)")
    plt.ylabel("mean reflectance")
    plt.title("Mean spectra by material (Si baseline vs SiO2)")
    plt.legend()
    path = os.path.join(out_dir, "material_mean_spectra.png")
    plt.savefig(path, dpi=140)
    plt.close()
    return path
