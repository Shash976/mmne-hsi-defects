"""Silicon baseline: pool bare-Si piece spectra into a reusable "normal"
population, cached to disk so it isn't recomputed on every run_analyze call.

Three layers, in this file:

- **data model + pure computation** (this section) -- ``SiliconBaseline``,
  ``PieceBaselineStats``, ``baseline_from_pieces``. Operates on an
  already-prepared ``List[Piece]``; no file I/O, fully unit-testable with
  synthetic pieces.
- **disk cache** -- ``save_silicon_baseline``/``load_silicon_baseline``/
  ``baseline_cache_valid``.
- **orchestration** -- ``compute_silicon_baseline`` (loads the real bare-Si
  scan via ``pipeline.prepare_pieces``), ``load_or_compute_baseline`` (the
  entry point ``pipeline.run_workflow`` uses), ``save_piece_diagnostics``.

Baseline stats are computed in *analysis space* (post calibrate -> SG smooth
-> SNV), matching how ``film.py`` already treats "bare silicon".
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import DatasetConfig, WorkflowConfig
from .pieces import Piece, spectral_angle

BASELINE_POOL_CAP = 200_000   # capped sample size stored in the cache for refitting detectors


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class PieceBaselineStats:
    """Per-piece QA stats -- the debug view analogous to run_analyze's per-piece table."""

    piece_id: str
    n_px: int
    mean_reflectance: float      # scalar, physical (pre-SNV) reflectance
    std_spectrum_mean: float     # mean of per-band std across this piece's foreground
    snr: float
    sam_from_global: float       # spectral angle (radians) of this piece's mean vs the pooled mean
    flag_outlier: bool           # sam_from_global > mean + 2*std across all pieces in this baseline


@dataclass
class SiliconBaseline:
    """The cached silicon baseline: summary stats + a capped raw sample."""

    dataset: str
    wavelengths: np.ndarray
    mean_spectrum: np.ndarray        # (bands,) analysis-space
    std_spectrum: np.ndarray         # (bands,)
    cov: np.ndarray                  # (bands, bands)
    pooled_spectra: np.ndarray       # (<=pool_cap, bands) capped subsample, all pieces
    piece_stats: List[PieceBaselineStats]
    config_snapshot: dict            # {"piece": ..., "preprocess": ...}, JSON-normalized
    computed_at: str


def subsample_spectra(arr: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """Cap a flat ``(n, bands)`` array to ``cap`` rows via reproducible random choice."""
    if arr.shape[0] <= cap:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.shape[0], cap, replace=False)
    return arr[idx]


def _config_snapshot(wf: WorkflowConfig) -> dict:
    """JSON-normalized snapshot of the configs that affect extraction/preprocessing.

    Round-tripping through JSON here (not just ``asdict``) means a freshly built
    snapshot compares equal to one loaded back from ``meta.json`` -- otherwise a
    tuple field like ``PieceConfig.background_bbox`` would never equal the list
    it becomes after being written to and read from JSON.
    """
    snap = {"piece": asdict(wf.piece), "preprocess": asdict(wf.preprocess)}
    return json.loads(json.dumps(snap))


def baseline_from_pieces(dataset_name: str, pieces: List[Piece], wf: WorkflowConfig,
                         pool_cap: int = BASELINE_POOL_CAP, seed: int = 0) -> SiliconBaseline:
    """Pool already-prepared pieces into a :class:`SiliconBaseline`.

    Mean/std/cov are computed over *every* in-mask pixel (not the capped
    subsample) so they aren't subsample-noisy; ``pooled_spectra`` is the capped
    sample later stages refit detectors on without reloading the raw scan.
    """
    if not pieces:
        raise ValueError(f"no pieces to build a silicon baseline from (dataset={dataset_name!r})")

    wavelengths = pieces[0].wavelengths
    full_pooled = np.vstack([p.foreground_spectra() for p in pieces])
    mean_spectrum = full_pooled.mean(axis=0)
    std_spectrum = full_pooled.std(axis=0)
    cov = np.cov(full_pooled, rowvar=False)
    pooled_spectra = subsample_spectra(full_pooled, pool_cap, seed)

    piece_means = np.vstack([p.foreground_spectra().mean(axis=0) for p in pieces])
    sam = spectral_angle(piece_means, mean_spectrum)
    sam_thresh = float(sam.mean() + 2 * sam.std())

    piece_stats: List[PieceBaselineStats] = []
    for p, sam_val in zip(pieces, sam):
        fg = p.foreground_spectra()
        if p.reflectance_mean is not None:
            mean_refl = float(np.nanmean(p.reflectance_mean[p.mask]))
        else:
            mean_refl = float(fg.mean())
        snr = float(p.noise["after"]["snr"]) if p.noise and "after" in p.noise else float("nan")
        piece_stats.append(PieceBaselineStats(
            piece_id=p.piece_id, n_px=int(p.mask.sum()), mean_reflectance=mean_refl,
            std_spectrum_mean=float(fg.std(axis=0).mean()), snr=snr,
            sam_from_global=float(sam_val), flag_outlier=bool(sam_val > sam_thresh),
        ))

    return SiliconBaseline(
        dataset=dataset_name, wavelengths=wavelengths, mean_spectrum=mean_spectrum,
        std_spectrum=std_spectrum, cov=cov, pooled_spectra=pooled_spectra,
        piece_stats=piece_stats, config_snapshot=_config_snapshot(wf),
        computed_at=datetime.now(timezone.utc).isoformat(),
    )
