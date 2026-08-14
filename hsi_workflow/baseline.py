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


# --------------------------------------------------------------------------
# Disk cache
# --------------------------------------------------------------------------

def save_silicon_baseline(baseline: SiliconBaseline, out_dir: str) -> None:
    """Write the baseline to ``out_dir`` as ``baseline.npz`` + ``meta.json`` +
    ``piece_stats.csv``. Writes go through a temp path + ``os.replace`` so a
    killed process never leaves a partially-written cache that looks valid."""
    os.makedirs(out_dir, exist_ok=True)

    npz_path = os.path.join(out_dir, "baseline.npz")
    tmp_npz = npz_path + ".tmp"
    # Pass an open file handle (not a bare path) so np.savez doesn't append
    # its own ".npz" suffix to our ".tmp" temp name.
    with open(tmp_npz, "wb") as f:
        np.savez(f, wavelengths=baseline.wavelengths, mean_spectrum=baseline.mean_spectrum,
                 std_spectrum=baseline.std_spectrum, cov=baseline.cov,
                 pooled_spectra=baseline.pooled_spectra)
    os.replace(tmp_npz, npz_path)

    csv_path = os.path.join(out_dir, "piece_stats.csv")
    tmp_csv = csv_path + ".tmp"
    pd.DataFrame([asdict(ps) for ps in baseline.piece_stats]).to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, csv_path)

    meta = {
        "dataset": baseline.dataset,
        "computed_at": baseline.computed_at,
        "config_snapshot": baseline.config_snapshot,
        "n_pieces": len(baseline.piece_stats),
        "n_pixels_total": int(sum(ps.n_px for ps in baseline.piece_stats)),
        "n_pixels_pooled_cache": int(baseline.pooled_spectra.shape[0]),
    }
    meta_path = os.path.join(out_dir, "meta.json")
    tmp_meta = meta_path + ".tmp"
    with open(tmp_meta, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp_meta, meta_path)


def load_silicon_baseline(out_dir: str) -> Optional[SiliconBaseline]:
    """Load a cached baseline, or ``None`` if missing/unreadable (never raises)."""
    npz_path = os.path.join(out_dir, "baseline.npz")
    meta_path = os.path.join(out_dir, "meta.json")
    csv_path = os.path.join(out_dir, "piece_stats.csv")
    if not (os.path.exists(npz_path) and os.path.exists(meta_path) and os.path.exists(csv_path)):
        return None
    try:
        with np.load(npz_path) as z:
            wavelengths = z["wavelengths"]
            mean_spectrum = z["mean_spectrum"]
            std_spectrum = z["std_spectrum"]
            cov = z["cov"]
            pooled_spectra = z["pooled_spectra"]
        with open(meta_path) as f:
            meta = json.load(f)
        df = pd.read_csv(csv_path)
        piece_stats = [PieceBaselineStats(**row) for row in df.to_dict(orient="records")]
        return SiliconBaseline(
            dataset=meta["dataset"], wavelengths=wavelengths, mean_spectrum=mean_spectrum,
            std_spectrum=std_spectrum, cov=cov, pooled_spectra=pooled_spectra,
            piece_stats=piece_stats, config_snapshot=meta["config_snapshot"],
            computed_at=meta["computed_at"],
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def baseline_cache_valid(baseline: SiliconBaseline, ds_cfg: DatasetConfig,
                         wf: WorkflowConfig) -> bool:
    """A cached baseline is valid for ``ds_cfg``/``wf`` iff the dataset name and
    the extraction/preprocessing config snapshot both match exactly."""
    if baseline.dataset != ds_cfg.name:
        return False
    return baseline.config_snapshot == _config_snapshot(wf)


# --------------------------------------------------------------------------
# Orchestration (touches the real bare-Si scan)
# --------------------------------------------------------------------------

def compute_silicon_baseline(ds_cfg: DatasetConfig, wf: WorkflowConfig,
                             verbose: bool = True) -> Tuple[SiliconBaseline, List[Piece]]:
    """Extract + preprocess ``ds_cfg`` fresh and pool it into a baseline.

    Imports ``prepare_pieces`` lazily to avoid a circular import: ``pipeline.py``
    imports this module at the top level (for ``load_or_compute_baseline``), so
    this module can't import ``pipeline`` at the top level too. Same trick
    already used by ``film.bare_si_reference_from_pieces``.
    """
    from .pipeline import prepare_pieces
    pieces = prepare_pieces(ds_cfg, wf, verbose=verbose)
    return baseline_from_pieces(ds_cfg.name, pieces, wf), pieces


def load_or_compute_baseline(ds_cfg: DatasetConfig, wf: WorkflowConfig, cache_root: str,
                             force: bool = False, verbose: bool = True) -> SiliconBaseline:
    """The entry point ``pipeline.run_workflow`` uses.

    Loads a valid cache when present; otherwise recomputes from the raw scan,
    saves the cache + per-piece diagnostics, and returns the fresh baseline.
    """
    out_dir = os.path.join(cache_root, ds_cfg.name)
    if not force:
        cached = load_silicon_baseline(out_dir)
        if cached is not None and baseline_cache_valid(cached, ds_cfg, wf):
            if verbose:
                print(f"Silicon baseline ({ds_cfg.name}): loaded from cache at {out_dir}")
            return cached

    if verbose:
        reason = "forced" if force else "no valid cache"
        print(f"Silicon baseline ({ds_cfg.name}): computing fresh ({reason}) ...")
    baseline, pieces = compute_silicon_baseline(ds_cfg, wf, verbose=verbose)
    save_silicon_baseline(baseline, out_dir)
    save_piece_diagnostics(baseline, pieces, out_dir)
    if verbose:
        print(f"Silicon baseline ({ds_cfg.name}): cached to {out_dir}")
    return baseline


def save_piece_diagnostics(baseline: SiliconBaseline, pieces: List[Piece], out_dir: str) -> None:
    """Per piece: mean spectrum +/-1 std vs the pooled baseline mean -- the
    visual debug check for whether a piece's extraction looks right."""
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    wl = baseline.wavelengths
    for p in pieces:
        fg = p.foreground_spectra()
        piece_mean = fg.mean(axis=0)
        piece_std = fg.std(axis=0)
        plt.figure(figsize=(7, 4))
        plt.plot(wl, baseline.mean_spectrum, color="tab:gray", lw=1.5,
                 label="pooled baseline mean")
        plt.plot(wl, piece_mean, color="tab:blue", lw=1.2, label=f"{p.piece_id} mean")
        plt.fill_between(wl, piece_mean - piece_std, piece_mean + piece_std,
                         color="tab:blue", alpha=0.2, label="+/-1 std")
        plt.xlabel("wavelength (nm)")
        plt.ylabel("analysis-space reflectance")
        plt.title(f"{p.piece_id} vs pooled silicon baseline")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{p.piece_id}_baseline.png"), dpi=140)
        plt.close()
