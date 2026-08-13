# Cached Silicon Baseline

**Date:** 2026-08-13
**Files:** new `hsi_workflow/baseline.py`, new `hsi_workflow/run_baseline.py`,
`hsi_workflow/pipeline.py`, `hsi_workflow/config.py`, `hsi_workflow/run_analyze.py`
**Status:** Approved design

## Problem

`pipeline.run_workflow` treats the bare-silicon baseline as disposable: every
`run_analyze` call reloads the raw `sio2_bare_si` scan (multi-GB), re-extracts its
pieces, and re-preprocesses them from scratch, just to pool their spectra for (a)
the always-on silicon-contrast Mahalanobis detector, (b) the PCA fit pool, and (c)
the `fit_on="baseline"` anomaly-detector fit. There's no way to inspect the
baseline on its own (per-piece stats/figures) to sanity-check that the bare-Si
extraction looks right, and no persisted artifact a future "subtract silicon from
an SiO2 wafer piece" step could load without redoing all of the above.

## Design

### Data model (`hsi_workflow/baseline.py`)

```python
@dataclass
class PieceBaselineStats:
    piece_id: str
    n_px: int
    mean_reflectance: float      # scalar, physical
    std_spectrum_mean: float     # scalar summary of std across bands
    snr: float
    sam_from_global: float       # spectral angle of this piece's mean vs the pooled global mean
    flag_outlier: bool           # sam_from_global > mean + 2*std across pieces

@dataclass
class SiliconBaseline:
    dataset: str
    wavelengths: np.ndarray
    mean_spectrum: np.ndarray        # analysis-space (post calibrate/smooth/SNV)
    std_spectrum: np.ndarray
    cov: np.ndarray                  # bands x bands
    pooled_spectra: np.ndarray       # capped subsample (200k) of in-mask spectra, all pieces
    piece_stats: List[PieceBaselineStats]
    config_snapshot: dict            # asdict(wf.piece) + asdict(wf.preprocess)
    computed_at: str
```

Computed in **analysis space** (calibrate -> SG smooth -> SNV), matching how
`film.py` and the existing anomaly detectors already treat "bare silicon" --
consistent with the codebase's existing convention, not raw reflectance.

`mean_spectrum`/`std_spectrum`/`cov` are computed over the full pooled foreground
(not just the capped subsample) so they're not subsample-noisy; `pooled_spectra`
is the capped sample used to refit detectors cheaply without reloading the scan.

### Functions

- `compute_silicon_baseline(ds_cfg, wf, verbose=True) -> Tuple[SiliconBaseline, List[Piece]]`
  -- calls `prepare_pieces`, builds the stats above. Returns the `Piece` list too
  so the CLI can render diagnostics without a second extraction pass.
- `save_silicon_baseline(baseline, out_dir)` -- writes `baseline.npz`, `meta.json`,
  `piece_stats.csv`.
- `load_silicon_baseline(out_dir) -> Optional[SiliconBaseline]` -- `None` if
  missing or unreadable (corrupt/partial write).
- `baseline_cache_valid(baseline, ds_cfg, wf) -> bool` -- dataset name matches
  **and** `config_snapshot == {**asdict(wf.piece), **asdict(wf.preprocess)}`
  (only the configs that affect extraction/preprocessing invalidate the cache;
  PCA/cluster/anomaly settings don't).
- `load_or_compute_baseline(ds_cfg, wf, cache_root, force=False, verbose=True) -> SiliconBaseline`
  -- the entry point `pipeline.py` uses. Loads if valid, else recomputes (same
  cost as today, once) and saves.
- `save_piece_diagnostics(baseline, pieces, out_dir)` -- per piece: mean spectrum
  +/- std band plotted against the pooled global mean; writes
  `figures/<piece_id>_baseline.png`.
- `subsample_spectra(arr, cap, seed)` -- small shared helper for capping a flat
  `(n, bands)` array (used both when building `pooled_spectra` and when
  `pipeline.py` draws a further-capped sample from it).

### Caching

- Cache root: `out/workflow/baseline/<dataset>/` (new `BASELINE_CACHE_ROOT =
  os.path.join("out", "workflow", "baseline")` constant in `config.py`, alongside
  `ORGANIZED_DATA_ROOT`).
- Files: `baseline.npz` (arrays), `meta.json` (config snapshot + summary counts +
  timestamp), `piece_stats.csv`, `figures/<piece_id>_baseline.png`.
- Invalidation: config-snapshot mismatch (see above) or missing/corrupt cache
  triggers a full recompute + overwrite. `force=True` (CLI `--force`) always
  recomputes.

### Diagnostics output (`run_baseline.py`)

Console table mirroring `run_analyze`'s per-piece table style:

```
piece_id                n_px  mean_refl   snr    sam_from_global  outlier
bare silicon all_p01   23358      0.412  34.2            0.014
...
```

Plus the per-piece PNGs and the CSV/JSON described above.

### CLI (`hsi_workflow/run_baseline.py`)

```
python -m hsi_workflow.run_baseline --dataset sio2_bare_si
python -m hsi_workflow.run_baseline --dataset sio2_bare_si --force
```

Mirrors `run_extract.py`'s structure (`argparse`, `DATASETS` choices, `--out`
default). `--dataset` defaults to `sio2_bare_si` but accepts any dataset (so a
different bare-Si scan could be baselined later without code changes).

### Wiring into `pipeline.py`

`run_workflow` currently does:

```python
baseline_pieces = prepare_pieces(baseline_cfg, wf, verbose=verbose)
...
pooled = pooled_foreground(baseline_pieces + target_pieces, wf.pca.max_fit_pixels, wf.pca.seed)
...
baseline_fg = pooled_foreground(baseline_pieces, wf.anomaly.max_fit_pixels, wf.anomaly.seed)
...
baseline_spectral = MahalanobisDetector().fit(baseline_fg)
```

and (in `prepare_pieces`, for film extraction) `bare_si_reference_from_pieces(baseline_pieces, wf)`.

New flow:

```python
sb = load_or_compute_baseline(baseline_cfg, wf, BASELINE_CACHE_ROOT,
                              force=force_baseline, verbose=verbose)
```

1. **PCA fit pool** -- mix `subsample_spectra(sb.pooled_spectra, wf.pca.max_fit_pixels // 2, wf.pca.seed)`
   with `pooled_foreground(target_pieces, wf.pca.max_fit_pixels // 2, wf.pca.seed)`
   (even split between baseline and target, approximating today's even per-piece
   split across the combined list).
2. **`fit_on="baseline"` detector fit + threshold** -- subsample `sb.pooled_spectra`
   directly with `wf.anomaly.max_fit_pixels`.
3. **Silicon-contrast Mahalanobis detector** (always computed) -- fit on the same
   subsampled `sb.pooled_spectra` (`wf.anomaly.max_fit_pixels`).
4. **Film reference** (`wf.film.enabled`) -- `sb.mean_spectrum` directly, replacing
   `bare_si_reference_from_pieces(baseline_pieces, wf)`. `prepare_pieces` gains an
   optional `film_reference` param already (see current code) -- just pass
   `sb.mean_spectrum` from `run_workflow` instead of computing it from a freshly
   prepared baseline piece list.

When the cache is valid, `run_workflow` never calls `prepare_pieces(baseline_cfg,
...)` at all -- the raw bare-Si cube is not loaded, extracted, or preprocessed.
`film.bare_si_reference_from_pieces`/`bare_si_reference_spectrum` stay as-is for
the film tuner script, which doesn't go through `run_workflow`.

### `run_analyze.py` changes

- New `--force-baseline` flag, passed through to `run_workflow(..., force_baseline=...)`.
- Print whether the baseline was loaded from cache or recomputed (one line), so
  cache behavior is visible without reading code.

### Error handling

- `load_or_compute_baseline` on a missing/corrupt cache file falls back to
  recompute (never raises for a bad cache -- only for the same failures
  `prepare_pieces` already raises, e.g. no cubes found).
- `save_silicon_baseline` writes to a temp path and renames into place (avoid a
  half-written cache from a killed process being read as valid).

### Testing plan

Run via the `hsi` conda env, batched/backgrounded (conda-run overhead is
1-3 min/call):

1. `python -m hsi_workflow.run_baseline --dataset sio2_bare_si` -- confirm
   `baseline.npz`/`meta.json`/`piece_stats.csv`/`figures/*.png` + console table.
2. `python -m hsi_workflow.run_analyze --target sio2_dish_white_20` with the
   baseline cache absent (builds it inline) then present (loads it) -- confirm
   the "loaded cached baseline" vs "computed baseline" log line differs between
   the two runs, and that figures/thresholds/region counts are consistent with
   pre-change output (spot check, not bit-for-bit -- the PCA-pool mixing ratio
   changes slightly per point 1 above).

## Out of scope

Actually *subtracting* the silicon baseline from SiO2 wafer pieces (a new
preprocessing/film step) is future work -- this design only makes the artifact
(`mean_spectrum`, `cov`, `pooled_spectra`) available and cached for that later
step to consume.
