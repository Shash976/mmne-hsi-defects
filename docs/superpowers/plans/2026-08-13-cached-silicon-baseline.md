# Cached Silicon Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract bare-silicon pieces once, pool their spectra into a "silicon baseline" (mean/std/cov + a capped sample), cache it to disk with per-piece diagnostics, and make `run_analyze`/`pipeline.run_workflow` load that cache instead of re-extracting the raw bare-Si scan on every run.

**Architecture:** New `hsi_workflow/baseline.py` holds the data model (`SiliconBaseline`, `PieceBaselineStats`) and pure computation (`baseline_from_pieces`), separated from the disk-cache layer (`save_silicon_baseline`/`load_silicon_baseline`/`baseline_cache_valid`) and the orchestration layer (`compute_silicon_baseline`/`load_or_compute_baseline`/`save_piece_diagnostics`) that touches real scans. A new CLI `hsi_workflow/run_baseline.py` drives it standalone; `pipeline.run_workflow` is rewired to call `load_or_compute_baseline` instead of `prepare_pieces(baseline_cfg, ...)` directly.

**Tech Stack:** Python, numpy, pandas, matplotlib (Agg backend), pytest. Follows this repo's existing dataclass-config + `run_*.py` CLI conventions (see `hsi_workflow/run_extract.py`, `hsi_workflow/dataset.py`, `hsi_workflow/film.py`).

## Global Constraints

- Baseline stats are computed in **analysis space** (post calibrate -> SG smooth -> SNV), matching `film.py`'s existing convention for "what bare silicon looks like" — not raw physical reflectance.
- Cache validity is keyed on `wf.piece` + `wf.preprocess` config only (the configs that affect extraction/preprocessing). PCA/cluster/anomaly config changes must NOT invalidate the cache.
- No test may depend on the real scan files under `C:\Users\shash\OneDrive - purdue.edu\Summer\hsi` — every automated test uses small synthetic `Piece` objects built in-memory, following the pattern already established in `tests/test_film.py`.
- Run all commands (pytest, CLIs) via the `hsi` conda env: `conda run -n hsi <cmd>`. `conda run` overhead is ~1-3 min per call — batch test runs together rather than invoking one test at a time.
- Cache writes must be atomic (temp file + `os.replace`) so a killed process never leaves a half-written cache that `load_silicon_baseline` would treat as valid.

---

## File Structure

- **Create** `hsi_workflow/baseline.py` — data model, pure stats computation, disk cache, orchestration, diagnostics plotting. All new logic for this feature lives here.
- **Create** `hsi_workflow/run_baseline.py` — CLI, mirrors `run_extract.py`'s shape.
- **Create** `tests/test_baseline.py` — unit tests for everything in `baseline.py` except the real-data-dependent `compute_silicon_baseline` (matches the existing repo convention that `prepare_pieces`/`run_workflow` aren't unit tested either — see `tests/test_film.py`, which tests `bare_si_reference_from_pieces` directly on synthetic pieces rather than through `prepare_pieces`).
- **Modify** `hsi_workflow/config.py` — add `BASELINE_CACHE_ROOT` constant (needs `import os`, currently absent).
- **Modify** `hsi_workflow/pipeline.py` — `run_workflow` calls `load_or_compute_baseline` instead of `prepare_pieces(baseline_cfg, ...)`; gains a `force_baseline` parameter.
- **Modify** `hsi_workflow/run_analyze.py` — add `--force-baseline` CLI flag, threaded through to `run_workflow`.

---

### Task 1: `baseline.py` — data model + pure stats computation

**Files:**
- Create: `hsi_workflow/baseline.py`
- Test: `tests/test_baseline.py`

**Interfaces:**
- Produces: `PieceBaselineStats` (dataclass: `piece_id: str`, `n_px: int`, `mean_reflectance: float`, `std_spectrum_mean: float`, `snr: float`, `sam_from_global: float`, `flag_outlier: bool`)
- Produces: `SiliconBaseline` (dataclass: `dataset: str`, `wavelengths: np.ndarray`, `mean_spectrum: np.ndarray`, `std_spectrum: np.ndarray`, `cov: np.ndarray`, `pooled_spectra: np.ndarray`, `piece_stats: List[PieceBaselineStats]`, `config_snapshot: dict`, `computed_at: str`)
- Produces: `BASELINE_POOL_CAP: int = 200_000` (module constant)
- Produces: `subsample_spectra(arr: np.ndarray, cap: int, seed: int) -> np.ndarray`
- Produces: `_config_snapshot(wf: WorkflowConfig) -> dict` (JSON-normalized `{"piece": asdict(wf.piece), "preprocess": asdict(wf.preprocess)}` — normalizing through `json.loads(json.dumps(...))` so tuple fields like `PieceConfig.background_bbox` compare equal to the list they become after a disk round-trip)
- Produces: `baseline_from_pieces(dataset_name: str, pieces: List[Piece], wf: WorkflowConfig, pool_cap: int = BASELINE_POOL_CAP, seed: int = 0) -> SiliconBaseline`
- Consumes: `Piece` and `spectral_angle` from `hsi_workflow.pieces` (`Piece.foreground_spectra() -> np.ndarray`, `spectral_angle(flat: np.ndarray, ref: np.ndarray) -> np.ndarray`)
- Consumes: `WorkflowConfig` from `hsi_workflow.config`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_baseline.py`:

```python
import numpy as np
import pytest

from hsi_workflow.config import WorkflowConfig
from hsi_workflow.pieces import Piece
from hsi_workflow.baseline import subsample_spectra, baseline_from_pieces


def _make_silicon_piece(piece_id, seed, rows=20, cols=20, bands=15, amp=0.05, freq=150.0):
    """A homogeneous bare-Si-like piece. ``amp``/``freq`` control spectral *shape*
    (not just brightness) so outlier tests actually move the spectral angle."""
    rng = np.random.default_rng(seed)
    wl = np.linspace(400, 1000, bands)
    base = 0.4 + amp * np.sin(wl / freq)
    data = np.tile(base, (rows, cols, 1)).astype(np.float64)
    data += rng.normal(0, 0.002, data.shape)
    mask = np.ones((rows, cols), dtype=bool)
    reflectance_mean = data.mean(axis=-1).astype(np.float32)
    noise = {"after": {"rms_noise": 0.01, "snr": 12.5, "n_pixels": rows * cols}}
    return Piece(data=data, mask=mask, material="silicon", piece_id=piece_id,
                source_label="bare silicon all", bbox=(0, rows, 0, cols),
                wavelengths=wl, reflectance_mean=reflectance_mean, noise=noise)


def test_subsample_spectra_caps_and_is_deterministic():
    arr = np.arange(1000 * 5, dtype=np.float64).reshape(1000, 5)
    out1 = subsample_spectra(arr, 100, seed=0)
    out2 = subsample_spectra(arr, 100, seed=0)
    assert out1.shape == (100, 5)
    np.testing.assert_array_equal(out1, out2)


def test_subsample_spectra_returns_all_when_under_cap():
    arr = np.arange(10 * 5, dtype=np.float64).reshape(10, 5)
    out = subsample_spectra(arr, 100, seed=0)
    np.testing.assert_array_equal(out, arr)


def test_baseline_from_pieces_shapes_and_mean():
    pieces = [_make_silicon_piece("p01", seed=0), _make_silicon_piece("p02", seed=1)]
    wf = WorkflowConfig()
    sb = baseline_from_pieces("sio2_bare_si", pieces, wf, pool_cap=500, seed=0)
    n_bands = pieces[0].n_bands
    assert sb.mean_spectrum.shape == (n_bands,)
    assert sb.std_spectrum.shape == (n_bands,)
    assert sb.cov.shape == (n_bands, n_bands)
    assert sb.pooled_spectra.shape[1] == n_bands
    assert sb.pooled_spectra.shape[0] <= 500
    assert len(sb.piece_stats) == 2
    all_fg = np.vstack([p.foreground_spectra() for p in pieces])
    np.testing.assert_allclose(sb.mean_spectrum, all_fg.mean(axis=0))


def test_baseline_from_pieces_flags_outlier_piece():
    normal = [_make_silicon_piece(f"p{i:02d}", seed=i) for i in range(5)]
    outlier = _make_silicon_piece("p_odd", seed=99, amp=0.3, freq=20.0)  # different shape
    wf = WorkflowConfig()
    sb = baseline_from_pieces("sio2_bare_si", normal + [outlier], wf)
    stats = {ps.piece_id: ps for ps in sb.piece_stats}
    assert stats["p_odd"].flag_outlier is True
    assert all(not stats[p.piece_id].flag_outlier for p in normal)


def test_baseline_from_pieces_empty_raises():
    with pytest.raises(ValueError):
        baseline_from_pieces("sio2_bare_si", [], WorkflowConfig())


def test_piece_stats_fields():
    piece = _make_silicon_piece("p01", seed=0)
    sb = baseline_from_pieces("sio2_bare_si", [piece], WorkflowConfig())
    ps = sb.piece_stats[0]
    assert ps.n_px == int(piece.mask.sum())
    assert ps.snr == pytest.approx(12.5)
    assert ps.sam_from_global == pytest.approx(0.0, abs=1e-6)  # only piece == global mean
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n hsi python -m pytest tests/test_baseline.py -v`
Expected: FAIL/ERROR — `hsi_workflow.baseline` does not exist yet.

- [ ] **Step 3: Implement `hsi_workflow/baseline.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n hsi python -m pytest tests/test_baseline.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add hsi_workflow/baseline.py tests/test_baseline.py
git commit -m "feat: add silicon baseline data model and pure stats computation"
```

---

### Task 2: `baseline.py` — disk cache (save/load/validity)

**Files:**
- Modify: `hsi_workflow/baseline.py`
- Test: `tests/test_baseline.py`

**Interfaces:**
- Consumes: `SiliconBaseline`, `PieceBaselineStats`, `_config_snapshot` from Task 1
- Produces: `save_silicon_baseline(baseline: SiliconBaseline, out_dir: str) -> None`
- Produces: `load_silicon_baseline(out_dir: str) -> Optional[SiliconBaseline]`
- Produces: `baseline_cache_valid(baseline: SiliconBaseline, ds_cfg: DatasetConfig, wf: WorkflowConfig) -> bool`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_baseline.py`:

```python
import os

from hsi_workflow.config import DATASETS
from hsi_workflow.baseline import (
    save_silicon_baseline, load_silicon_baseline, baseline_cache_valid,
)


def test_save_and_load_round_trip(tmp_path):
    pieces = [_make_silicon_piece("p01", seed=0), _make_silicon_piece("p02", seed=1)]
    wf = WorkflowConfig()
    sb = baseline_from_pieces("sio2_bare_si", pieces, wf)
    out_dir = str(tmp_path / "sio2_bare_si")
    save_silicon_baseline(sb, out_dir)
    loaded = load_silicon_baseline(out_dir)
    assert loaded is not None
    assert loaded.dataset == sb.dataset
    np.testing.assert_allclose(loaded.mean_spectrum, sb.mean_spectrum)
    np.testing.assert_allclose(loaded.cov, sb.cov)
    np.testing.assert_allclose(loaded.pooled_spectra, sb.pooled_spectra)
    assert len(loaded.piece_stats) == len(sb.piece_stats)
    assert loaded.piece_stats[0].piece_id == sb.piece_stats[0].piece_id
    assert loaded.piece_stats[0].flag_outlier == sb.piece_stats[0].flag_outlier
    assert baseline_cache_valid(loaded, DATASETS["sio2_bare_si"], wf)


def test_load_silicon_baseline_missing_returns_none(tmp_path):
    assert load_silicon_baseline(str(tmp_path / "nope")) is None


def test_baseline_cache_valid_detects_config_and_dataset_change(tmp_path):
    pieces = [_make_silicon_piece("p01", seed=0)]
    wf = WorkflowConfig()
    sb = baseline_from_pieces("sio2_bare_si", pieces, wf)
    out_dir = str(tmp_path / "sio2_bare_si")
    save_silicon_baseline(sb, out_dir)
    loaded = load_silicon_baseline(out_dir)
    ds_cfg = DATASETS["sio2_bare_si"]
    assert baseline_cache_valid(loaded, ds_cfg, wf) is True

    wf2 = WorkflowConfig()
    wf2.piece.min_area = wf2.piece.min_area + 500
    assert baseline_cache_valid(loaded, ds_cfg, wf2) is False

    assert baseline_cache_valid(loaded, DATASETS["sio2_dish_white_20"], wf) is False


def test_baseline_cache_valid_survives_tuple_config_field(tmp_path):
    """PieceConfig.background_bbox is a tuple; JSON round-trips it to a list --
    cache validity must not be fooled by that type change."""
    pieces = [_make_silicon_piece("p01", seed=0)]
    wf = WorkflowConfig()
    wf.piece.background_bbox = (1, 2, 3, 4)
    sb = baseline_from_pieces("sio2_bare_si", pieces, wf)
    out_dir = str(tmp_path / "sio2_bare_si")
    save_silicon_baseline(sb, out_dir)
    loaded = load_silicon_baseline(out_dir)
    assert baseline_cache_valid(loaded, DATASETS["sio2_bare_si"], wf) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n hsi python -m pytest tests/test_baseline.py -v -k "save_and_load or cache_valid or missing_returns_none"`
Expected: FAIL/ERROR — `save_silicon_baseline`/`load_silicon_baseline`/`baseline_cache_valid` not defined.

- [ ] **Step 3: Implement the cache functions**

Append to `hsi_workflow/baseline.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n hsi python -m pytest tests/test_baseline.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add hsi_workflow/baseline.py tests/test_baseline.py
git commit -m "feat: add disk cache for the silicon baseline"
```

---

### Task 3: `baseline.py` — orchestration (compute/load-or-compute/diagnostics)

**Files:**
- Modify: `hsi_workflow/baseline.py`
- Test: `tests/test_baseline.py`

**Interfaces:**
- Consumes: everything from Task 1 + Task 2
- Consumes: `prepare_pieces(ds_cfg: DatasetConfig, wf: WorkflowConfig, verbose: bool = True, film_reference=None) -> List[Piece]` from `hsi_workflow.pipeline` (imported lazily, inside the function, to avoid a circular import — `pipeline.py` will import from `baseline.py` at module level starting in Task 6)
- Produces: `compute_silicon_baseline(ds_cfg: DatasetConfig, wf: WorkflowConfig, verbose: bool = True) -> Tuple[SiliconBaseline, List[Piece]]`
- Produces: `load_or_compute_baseline(ds_cfg: DatasetConfig, wf: WorkflowConfig, cache_root: str, force: bool = False, verbose: bool = True) -> SiliconBaseline`
- Produces: `save_piece_diagnostics(baseline: SiliconBaseline, pieces: List[Piece], out_dir: str) -> None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_baseline.py`:

```python
from hsi_workflow.baseline import load_or_compute_baseline, save_piece_diagnostics


def test_save_piece_diagnostics_creates_figures(tmp_path):
    pieces = [_make_silicon_piece("p01", seed=0), _make_silicon_piece("p02", seed=1)]
    wf = WorkflowConfig()
    sb = baseline_from_pieces("sio2_bare_si", pieces, wf)
    out_dir = str(tmp_path / "sio2_bare_si")
    save_piece_diagnostics(sb, pieces, out_dir)
    for p in pieces:
        assert os.path.exists(os.path.join(out_dir, "figures", f"{p.piece_id}_baseline.png"))


def test_load_or_compute_baseline_uses_cache_on_second_call(tmp_path, monkeypatch):
    calls = {"n": 0}
    pieces = [_make_silicon_piece("p01", seed=0)]
    wf = WorkflowConfig()
    ds_cfg = DATASETS["sio2_bare_si"]

    def fake_compute(ds_cfg_, wf_, verbose=True):
        calls["n"] += 1
        return baseline_from_pieces(ds_cfg_.name, pieces, wf_), pieces

    monkeypatch.setattr("hsi_workflow.baseline.compute_silicon_baseline", fake_compute)
    cache_root = str(tmp_path)

    sb1 = load_or_compute_baseline(ds_cfg, wf, cache_root, verbose=False)
    assert calls["n"] == 1
    sb2 = load_or_compute_baseline(ds_cfg, wf, cache_root, verbose=False)
    assert calls["n"] == 1   # cache hit, no recompute
    np.testing.assert_allclose(sb1.mean_spectrum, sb2.mean_spectrum)


def test_load_or_compute_baseline_force_recomputes(tmp_path, monkeypatch):
    calls = {"n": 0}
    pieces = [_make_silicon_piece("p01", seed=0)]
    wf = WorkflowConfig()
    ds_cfg = DATASETS["sio2_bare_si"]

    def fake_compute(ds_cfg_, wf_, verbose=True):
        calls["n"] += 1
        return baseline_from_pieces(ds_cfg_.name, pieces, wf_), pieces

    monkeypatch.setattr("hsi_workflow.baseline.compute_silicon_baseline", fake_compute)
    cache_root = str(tmp_path)

    load_or_compute_baseline(ds_cfg, wf, cache_root, verbose=False)
    load_or_compute_baseline(ds_cfg, wf, cache_root, force=True, verbose=False)
    assert calls["n"] == 2


def test_load_or_compute_baseline_recomputes_on_config_change(tmp_path, monkeypatch):
    calls = {"n": 0}
    pieces = [_make_silicon_piece("p01", seed=0)]
    ds_cfg = DATASETS["sio2_bare_si"]

    def fake_compute(ds_cfg_, wf_, verbose=True):
        calls["n"] += 1
        return baseline_from_pieces(ds_cfg_.name, pieces, wf_), pieces

    monkeypatch.setattr("hsi_workflow.baseline.compute_silicon_baseline", fake_compute)
    cache_root = str(tmp_path)

    load_or_compute_baseline(ds_cfg, WorkflowConfig(), cache_root, verbose=False)
    wf2 = WorkflowConfig()
    wf2.piece.min_area = wf2.piece.min_area + 500
    load_or_compute_baseline(ds_cfg, wf2, cache_root, verbose=False)
    assert calls["n"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n hsi python -m pytest tests/test_baseline.py -v -k "diagnostics or load_or_compute"`
Expected: FAIL/ERROR — `load_or_compute_baseline`/`save_piece_diagnostics` not defined.

- [ ] **Step 3: Implement the orchestration functions**

Append to `hsi_workflow/baseline.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n hsi python -m pytest tests/test_baseline.py -v`
Expected: PASS (15 tests)

- [ ] **Step 5: Commit**

```bash
git add hsi_workflow/baseline.py tests/test_baseline.py
git commit -m "feat: add silicon baseline orchestration and per-piece diagnostics"
```

---

### Task 4: `config.py` — cache root constant

**Files:**
- Modify: `hsi_workflow/config.py`

**Interfaces:**
- Produces: `BASELINE_CACHE_ROOT: str` (module constant, `"out/workflow/baseline"` via `os.path.join`)
- Consumes: nothing new

- [ ] **Step 1: Add the `import os` and constant**

In `hsi_workflow/config.py`, add `import os` to the imports at the top (currently only `dataclasses`/`typing`):

```python
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
```

Then, directly below `ORGANIZED_DATA_ROOT = "data"` (around line 154), add:

```python
# Where run_baseline / pipeline.run_workflow cache the pooled silicon baseline
# (mean/cov/pooled spectra + per-piece diagnostics) so the raw bare-Si scan
# isn't re-extracted on every run_analyze call. See hsi_workflow/baseline.py.
BASELINE_CACHE_ROOT = os.path.join("out", "workflow", "baseline")
```

- [ ] **Step 2: Verify the module still imports cleanly**

Run: `conda run -n hsi python -c "from hsi_workflow.config import BASELINE_CACHE_ROOT; print(BASELINE_CACHE_ROOT)"`
Expected: prints `out\workflow\baseline` (or `out/workflow/baseline` on non-Windows) with no errors.

- [ ] **Step 3: Commit**

```bash
git add hsi_workflow/config.py
git commit -m "feat: add BASELINE_CACHE_ROOT config constant"
```

---

### Task 5: `run_baseline.py` CLI

**Files:**
- Create: `hsi_workflow/run_baseline.py`

**Interfaces:**
- Consumes: `DATASETS`, `WorkflowConfig`, `BASELINE_CACHE_ROOT` from `hsi_workflow.config`
- Consumes: `load_or_compute_baseline` from `hsi_workflow.baseline`

- [ ] **Step 1: Implement the CLI**

Create `hsi_workflow/run_baseline.py`, mirroring `hsi_workflow/run_extract.py`'s structure:

```python
"""CLI: build (or reuse the cache for) the silicon baseline.

    python -m hsi_workflow.run_baseline --dataset sio2_bare_si
    python -m hsi_workflow.run_baseline --dataset sio2_bare_si --force

Extracts every piece of the bare-Si dataset, pools their spectra into a
:class:`~hsi_workflow.baseline.SiliconBaseline` (mean/std/cov + a capped
sample), and caches it under ``out/workflow/baseline/<dataset>/`` --
``baseline.npz`` (arrays), ``meta.json`` (config snapshot + summary),
``piece_stats.csv`` (per-piece QA table), and ``figures/<piece_id>_baseline.png``
(per-piece mean spectrum vs the pooled baseline mean, for visual debugging).

``pipeline.run_workflow`` loads this same cache on every ``run_analyze`` call
instead of re-extracting the raw bare-Si scan; running this CLI is the way to
inspect/refresh that cache on its own.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    __package__ = "hsi_workflow"

import argparse
import os

from .config import DATASETS, WorkflowConfig, BASELINE_CACHE_ROOT
from .baseline import load_or_compute_baseline


def main():
    p = argparse.ArgumentParser(description="Build/refresh the cached silicon baseline.")
    p.add_argument("--dataset", default="sio2_bare_si", type=str.lower, choices=sorted(DATASETS))
    p.add_argument("--out", default=BASELINE_CACHE_ROOT)
    p.add_argument("--force", action="store_true",
                   help="Recompute even if a valid cache already exists.")
    args = p.parse_args()

    ds_cfg = DATASETS[args.dataset]
    wf = WorkflowConfig()

    sb = load_or_compute_baseline(ds_cfg, wf, args.out, force=args.force, verbose=True)

    print("\n{:<22} {:>8} {:>10} {:>8} {:>16} {:>8}".format(
        "piece_id", "n_px", "mean_refl", "snr", "sam_from_global", "outlier"))
    for ps in sb.piece_stats:
        print("{:<22} {:>8} {:>10.3f} {:>8.1f} {:>16.4f} {:>8}".format(
            ps.piece_id, ps.n_px, ps.mean_reflectance, ps.snr,
            ps.sam_from_global, "YES" if ps.flag_outlier else ""))

    n_outliers = sum(ps.flag_outlier for ps in sb.piece_stats)
    print(f"\n{len(sb.piece_stats)} pieces, {n_outliers} flagged as outliers "
          f"(sam_from_global > mean + 2*std across pieces).")
    print(f"Cache + figures + piece_stats.csv under {os.path.join(args.out, ds_cfg.name)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Manual smoke test against the real bare-Si scan**

This CLI touches the real multi-GB scan, so it isn't covered by pytest -- verify it by hand, same as `run_extract.py`/`run_analyze.py` (neither has a pytest suite either). Run in the background (this loads/extracts/preprocesses the whole `sio2_bare_si` cube, so it will take a few minutes):

Run: `conda run -n hsi python -m hsi_workflow.run_baseline --dataset sio2_bare_si`

Expected:
- Console prints `Silicon baseline (sio2_bare_si): computing fresh (no valid cache) ...` then the per-piece table (9 rows, matching the 9 pieces already visible under `out/workflow/extract/sio2_bare_si/`) then `Silicon baseline (sio2_bare_si): cached to ...`.
- `out/workflow/baseline/sio2_bare_si/baseline.npz`, `meta.json`, `piece_stats.csv` exist.
- `out/workflow/baseline/sio2_bare_si/figures/` has one `<piece_id>_baseline.png` per piece (9 files).

Then run it again to confirm the cache hits:

Run: `conda run -n hsi python -m hsi_workflow.run_baseline --dataset sio2_bare_si`
Expected: prints `Silicon baseline (sio2_bare_si): loaded from cache at ...` and returns in a few seconds (no re-extraction), same per-piece table.

- [ ] **Step 3: Commit**

```bash
git add hsi_workflow/run_baseline.py
git commit -m "feat: add run_baseline CLI"
```

---

### Task 6: Wire the cache into `pipeline.run_workflow` and `run_analyze`

**Files:**
- Modify: `hsi_workflow/pipeline.py:33-44` (imports), `hsi_workflow/pipeline.py:248-315` (`run_workflow`)
- Modify: `hsi_workflow/run_analyze.py`

**Interfaces:**
- Consumes: `load_or_compute_baseline`, `subsample_spectra` from `hsi_workflow.baseline` (Tasks 1-3)
- Consumes: `BASELINE_CACHE_ROOT` from `hsi_workflow.config` (Task 4)
- Produces: `run_workflow(target: str, wf: Optional[WorkflowConfig] = None, baseline: str = DEFAULT_BASELINE, verbose: bool = True, force_baseline: bool = False) -> WorkflowResult` (adds the `force_baseline` param; return type/other params unchanged)

- [ ] **Step 1: Update `pipeline.py` imports**

In `hsi_workflow/pipeline.py`, change the import block (currently lines 33-44):

```python
from .config import (DatasetConfig, WorkflowConfig, DATASETS, DEFAULT_BASELINE,
                     BASELINE_CACHE_ROOT)
from .cube_io import Cube, iter_cube_paths, load_dataset_cube, load_reference_spectrum
from .pieces import Piece, extract_pieces
from .preprocessing import preprocess, saturation_mask, calibrate_reflectance
from .decomposition import fit_pca, PcaModel
from .clustering import cluster, cluster_map, cluster_metrics, ClusterResult
from .anomaly import (fit_detectors, MahalanobisDetector, anomaly_map,
                      flag_threshold, to_probability)
from .postprocess import clean_binary_map, label_regions
from .regions import (characterize_regions, regions_to_table, RegionStats,
                      spectral_distance_map)
from .rois import tile_rois, roi_feature_matrix, build_roi_table, Roi
from .baseline import load_or_compute_baseline, subsample_spectra
```

(only the `.config` import line and the new `.baseline` import line actually change; the rest are unchanged and shown for context.)

- [ ] **Step 2: Rewrite `run_workflow`'s baseline handling**

Replace the body of `run_workflow` (currently `hsi_workflow/pipeline.py:248-315`) with:

```python
def run_workflow(target: str, wf: Optional[WorkflowConfig] = None,
                 baseline: str = DEFAULT_BASELINE, verbose: bool = True,
                 force_baseline: bool = False) -> WorkflowResult:
    """Run the full pipeline: fit on the silicon baseline, analyze the target.

    ``target``/``baseline`` are dataset preset names. The silicon baseline is
    loaded from ``BASELINE_CACHE_ROOT`` (see ``hsi_workflow.baseline``) when a
    cache valid for ``baseline``/``wf.piece``/``wf.preprocess`` exists;
    otherwise it's computed fresh from the raw scan and cached for next time.
    ``force_baseline=True`` always recomputes. Returns a :class:`WorkflowResult`
    holding the shared PCA/detectors and one :class:`PieceAnalysis` per target
    piece, plus the aggregated ROI table.
    """
    wf = wf or WorkflowConfig()
    wf.validate()
    target_cfg = DATASETS[target]
    baseline_cfg = DATASETS[baseline]

    if verbose:
        print(f"Baseline (normal) dataset: {baseline!r} [{baseline_cfg.material}]")
    sb = load_or_compute_baseline(baseline_cfg, wf, BASELINE_CACHE_ROOT,
                                  force=force_baseline, verbose=verbose)

    # Stage 3.1b: the cached baseline's pooled mean is the bare-silicon
    # reference for narrowing SiO2 pieces to their oxide sub-region.
    film_reference = None
    if wf.film.enabled:
        film_reference = sb.mean_spectrum
        if verbose:
            print("Film extraction ON: narrowing SiO2 pieces to their oxide sub-region.")
    if verbose:
        print(f"Target dataset: {target!r} [{target_cfg.material}]")
    target_pieces = prepare_pieces(target_cfg, wf, verbose=verbose,
                                   film_reference=film_reference)

    # --- Stage 5: PCA on pooled foreground (baseline cache + target) ---
    # Split the fit cap evenly between the two populations (the cached baseline
    # sample is already a subsample of every bare-Si piece; the target side is
    # pooled fresh per piece as before).
    baseline_pool = subsample_spectra(sb.pooled_spectra, wf.pca.max_fit_pixels // 2, wf.pca.seed)
    target_pool = pooled_foreground(target_pieces, wf.pca.max_fit_pixels // 2, wf.pca.seed)
    pooled = np.vstack([baseline_pool, target_pool])
    pca = fit_pca(pooled, wf.pca)
    if verbose:
        evr = pca.explained_variance_ratio
        print(f"PCA explained variance: " + ", ".join(f"PC{i+1}={v:.1%}" for i, v in enumerate(evr)))

    # --- Stage 8: fit anomaly detectors on the "normal" population ---
    # fit_on="self" -> the target's own majority (finds localized anomalies within
    # the film); fit_on="baseline" -> the cached silicon baseline (material
    # contrast). Thresholds come from the same population the detectors were fit on.
    baseline_fg = subsample_spectra(sb.pooled_spectra, wf.anomaly.max_fit_pixels, wf.anomaly.seed)
    if wf.anomaly.fit_on == "baseline":
        normal_fg = baseline_fg
    else:
        normal_fg = pooled_foreground(target_pieces, wf.anomaly.max_fit_pixels, wf.anomaly.seed)
    normal_feat = pca.transform(normal_fg)
    detectors = fit_detectors(normal_feat, wf.anomaly)
    thresholds = {name: flag_threshold(det.score(normal_feat), wf.anomaly.anomaly_percentile)
                  for name, det in detectors.items()}
    if verbose:
        print(f"Anomaly detectors fit on {wf.anomaly.fit_on!r} population "
              f"({normal_fg.shape[0]} spectra); methods={wf.anomaly.methods}")
    # Spectral-space Mahalanobis on the cached baseline sample, always, for the
    # region "distance from silicon baseline" feature.
    baseline_spectral = MahalanobisDetector().fit(baseline_fg)

    # --- Per-target-piece analysis ---
    analyses = [analyze_piece(p, pca, detectors, baseline_spectral, thresholds, wf)
                for p in target_pieces]

    # --- Aggregate ROI table across all target pieces ---
    all_rois: List[Roi] = [r for a in analyses for r in a.rois]
    wl = target_pieces[0].wavelengths if target_pieces else None
    roi_table = build_roi_table(all_rois, wl) if all_rois else None

    return WorkflowResult(pca=pca, detectors=detectors, baseline_thresholds=thresholds,
                          analyses=analyses, roi_table=roi_table)
```

- [ ] **Step 3: Add `--force-baseline` to `run_analyze.py`**

In `hsi_workflow/run_analyze.py`, add the flag next to `--baseline` in `main()`'s `argparse` block:

```python
    p.add_argument("--baseline", default=DEFAULT_BASELINE, choices=sorted(DATASETS))
    p.add_argument("--force-baseline", action="store_true",
                   help="Recompute the silicon baseline from the raw bare-Si scan "
                        "even if a valid cache exists under out/workflow/baseline/<baseline>/.")
```

Then update the `run_workflow` call in `main()`:

```python
    res = run_workflow(args.target, wf, baseline=args.baseline, force_baseline=args.force_baseline)
```

- [ ] **Step 4: Run the existing test suite to check nothing broke**

Run: `conda run -n hsi python -m pytest tests/ -v`
Expected: PASS — all pre-existing tests (`test_film.py`, `test_pieces_extract.py`, etc.) plus the new `test_baseline.py` still pass. None of them exercise `pipeline.run_workflow` directly (confirmed: no existing test imports `run_workflow`), so this is a regression check on the modules Task 6 touches transitively, not a direct test of the rewritten function.

- [ ] **Step 5: Manual integration smoke test**

`run_workflow` itself has no pytest coverage (matches the existing convention — it always touches real scans), so verify the wiring by hand. If the cache from Task 5's smoke test is still at `out/workflow/baseline/sio2_bare_si/`, remove it first so this exercises the cache-miss path inside `run_analyze` itself:

Run: `conda run -n hsi python -m hsi_workflow.run_analyze --target sio2_dish_white_20`

Expected: prints `Silicon baseline ('sio2_bare_si'): computing fresh (no valid cache) ...` early in the log, produces the usual figures/region tables/report under `out/workflow/analyze/sio2_dish_white_20/`, and also leaves a fresh cache under `out/workflow/baseline/sio2_bare_si/`.

Run it again immediately:

Run: `conda run -n hsi python -m hsi_workflow.run_analyze --target sio2_dish_white_20`

Expected: prints `Silicon baseline ('sio2_bare_si'): loaded from cache at ...`, finishes noticeably faster (skips extracting/preprocessing the bare-Si scan), and produces figures/thresholds/region counts in the same ballpark as the first run (not necessarily bit-identical — the PCA-fit pooling ratio between baseline and target changed from an even per-piece split to an even per-population split, per the design doc).

- [ ] **Step 6: Commit**

```bash
git add hsi_workflow/pipeline.py hsi_workflow/run_analyze.py
git commit -m "feat: wire pipeline.run_workflow to the cached silicon baseline"
```

---

## Self-Review Notes

- **Spec coverage:** data model (Task 1), disk cache (Task 2), orchestration + diagnostics (Task 3), CLI (Task 5), pipeline wiring incl. `--force-baseline` (Task 6), config constant (Task 4) — every section of the design doc has a task. The "Out of scope" section (actual silicon subtraction) correctly has no task.
- **Placeholder scan:** no TBDs; every step has real code or an exact command + expected output.
- **Type consistency:** `SiliconBaseline`/`PieceBaselineStats` field names are identical across Tasks 1-3 and the save/load round trip (`asdict(ps)` on write, `PieceBaselineStats(**row)` on read — same field set). `load_or_compute_baseline`'s signature in Task 3 matches exactly what Task 6 calls in `pipeline.py`. `BASELINE_CACHE_ROOT` is defined once (Task 4) and only ever imported, never redefined.
