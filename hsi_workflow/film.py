"""Stage 3.1b -- SiO2 film extraction within a silicon piece.

Piece extraction (``pieces.py``) crops each wafer out of the dish. But a wafer is
part **bare silicon** and part **SiO2-covered**, and the anomaly stage should
reason about the oxide, not the whole wafer. This module isolates the SiO2
sub-region inside a piece's mask.

The discriminator mirrors piece extraction, but references **bare silicon**
instead of the dish: SiO2 on Si shows thin-film interference, so its reflectance
differs in *shape* from bare silicon (which the ``sio2_bare_si`` control captures).

Two reference modes (``FilmConfig.reference``):

- ``"control"`` -- distance from the mean spectrum of the bare-silicon control
  dataset (one global reference, computed in the same analysis space as the
  piece data).
- ``"in_piece"`` -- a 2-cluster split of the wafer's own pixels; the bare cluster
  is the one spectrally closest to the control reference (falling back to the
  larger cluster when no reference is given). Robust to per-piece lighting.

Everything is on full spectra. Reuses ``pieces.spectral_angle`` /
``pieces._mahalanobis_to_background`` / ``pieces.clean_mask`` /
``pieces.component_sizes`` and ``segmentation.segment``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy import ndimage as ndi

from .config import FilmConfig, WorkflowConfig, DatasetConfig
from .pieces import (Piece, spectral_angle, _mahalanobis_to_background,
                     clean_mask, component_sizes)
from .segmentation import segment


@dataclass
class FilmMask:
    """SiO2 vs bare-silicon split of one piece's foreground (piece-bbox grid)."""

    sio2_mask: np.ndarray        # (rows, cols) bool -- oxide pixels within the piece
    substrate_mask: np.ndarray   # (rows, cols) bool -- bare-silicon pixels
    dist: np.ndarray             # (rows, cols) distance-from-bare-silicon (0 off-mask)


# --------------------------------------------------------------------------
# Bare-silicon reference
# --------------------------------------------------------------------------

def bare_si_reference_from_pieces(pieces: List[Piece], wf: WorkflowConfig) -> np.ndarray:
    """Mean bare-silicon spectrum (analysis space) from already-prepared pieces."""
    from .pipeline import pooled_foreground
    if not pieces:
        raise ValueError("no baseline pieces to build a bare-silicon reference from")
    pooled = pooled_foreground(pieces, wf.pca.max_fit_pixels, wf.pca.seed)
    return pooled.mean(axis=0)


def bare_si_reference_spectrum(baseline_cfg: DatasetConfig, wf: WorkflowConfig,
                               verbose: bool = False) -> np.ndarray:
    """Load the bare-silicon control dataset and return its mean foreground spectrum.

    Convenience wrapper (used by the film tuner). ``run_workflow`` instead derives
    the reference from the baseline pieces it already prepared, to avoid a second
    pass over the control cube.
    """
    from .pipeline import prepare_pieces
    pieces = prepare_pieces(baseline_cfg, wf, verbose=verbose)
    return bare_si_reference_from_pieces(pieces, wf)


# --------------------------------------------------------------------------
# In-piece bare estimate + film distance
# --------------------------------------------------------------------------

def _in_piece_bare(piece_data: np.ndarray, mask: np.ndarray,
                   ref_spectrum: Optional[np.ndarray], cfg: FilmConfig):
    """(bare_pixels, bare_mean) from the wafer's own 2-cluster split.

    The bare-silicon cluster is the one whose mean spectrum is closest (SAM) to
    ``ref_spectrum``; without a reference, the larger cluster is taken as bare
    (bare silicon usually fills more of a wafer than the oxide patch).
    """
    seg = segment(piece_data, valid_mask=mask, seed=cfg.seed)
    a_mask = seg.foreground & mask
    b_mask = seg.substrate & mask
    a_px, b_px = piece_data[a_mask], piece_data[b_mask]
    if a_px.shape[0] == 0 or b_px.shape[0] == 0:
        px = piece_data[mask]
        return px, px.mean(axis=0)
    a_mean, b_mean = a_px.mean(axis=0), b_px.mean(axis=0)
    if ref_spectrum is not None:
        a_ang = float(spectral_angle(a_mean[None, :], ref_spectrum)[0])
        b_ang = float(spectral_angle(b_mean[None, :], ref_spectrum)[0])
        bare_is_a = a_ang < b_ang
    else:
        bare_is_a = a_px.shape[0] >= b_px.shape[0]
    if cfg.invert:
        bare_is_a = not bare_is_a
    bare_px = a_px if bare_is_a else b_px
    return bare_px, bare_px.mean(axis=0)


def film_distance(piece_data: np.ndarray, mask: np.ndarray,
                  ref_spectrum: Optional[np.ndarray], cfg: FilmConfig) -> np.ndarray:
    """(rows, cols) distance-from-bare-silicon for in-mask pixels; 0 elsewhere."""
    rows, cols, bands = piece_data.shape
    flat = piece_data.reshape(-1, bands)
    mflat = mask.reshape(-1)
    dist = np.zeros(rows * cols, dtype=np.float64)
    fg = flat[mflat]
    if fg.shape[0] == 0:
        return dist.reshape(rows, cols)

    if cfg.reference == "in_piece" or cfg.method == "kmeans":
        bare_px, bare_mean = _in_piece_bare(piece_data, mask, ref_spectrum, cfg)
    else:
        if ref_spectrum is None:
            raise ValueError("film reference='control' needs a bare-silicon reference spectrum")
        bare_px, bare_mean = None, ref_spectrum

    if cfg.method == "mahalanobis" and bare_px is not None and bare_px.shape[0] > bands:
        d = _mahalanobis_to_background(fg, bare_px)
    else:
        d = spectral_angle(fg, bare_mean)      # sam, kmeans-derived, or maha fallback
    dist[mflat] = d
    return dist.reshape(rows, cols)


def _threshold_film_mask(dist: np.ndarray, mask: np.ndarray, cfg: FilmConfig) -> np.ndarray:
    """Otsu/percentile on the *in-mask* distances; oxide = far-from-bare pixels."""
    vals = dist[mask]
    if vals.size == 0:
        return np.zeros_like(mask)
    if cfg.threshold == "otsu":
        from skimage.filters import threshold_otsu
        try:
            t = float(threshold_otsu(vals))
        except ValueError:                     # single-valued -> nothing to split
            t = float(np.median(vals))
    else:
        t = float(np.percentile(vals, cfg.threshold_percentile))
    return mask & (dist > t)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def extract_film(piece: Piece, cfg: FilmConfig,
                 ref_spectrum: Optional[np.ndarray] = None) -> FilmMask:
    """Split one preprocessed :class:`Piece` into SiO2 vs bare-silicon sub-masks."""
    cfg.validate()
    dist = film_distance(piece.data, piece.mask, ref_spectrum, cfg)
    raw = _threshold_film_mask(dist, piece.mask, cfg)
    sio2 = clean_mask(raw, cfg) & piece.mask   # clean_mask duck-types on open/close/fill
    labels, _ = ndi.label(sio2)
    sizes = component_sizes(labels)
    keep = sizes >= cfg.min_area
    keep[0] = False                            # background label
    sio2 = keep[labels]
    substrate = piece.mask & ~sio2
    return FilmMask(sio2_mask=sio2, substrate_mask=substrate, dist=dist)
