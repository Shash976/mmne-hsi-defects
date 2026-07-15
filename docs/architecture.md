# Architecture — How the code is organized (and how to change it)

The guiding principle: **one stage = one module + one config dataclass**. A single
`WorkflowConfig` composes all the stage configs, and the orchestrator runs the
stages in order. You can retune or replace any stage in isolation.

## Module map

| Module | Responsibility |
|---|---|
| [`config.py`](../hsi_workflow/config.py) | Dataset presets + every per-stage config dataclass + `WorkflowConfig` |
| [`io.py`](../hsi_workflow/io.py) | `Cube` value object, ENVI loading, discovery, cached reference spectra |
| [`pieces.py`](../hsi_workflow/pieces.py) | Stage 3.1 — split a multi-piece scan into `Piece` crops |
| [`preprocessing.py`](../hsi_workflow/preprocessing.py) | Stages 2–3 — calibrate, Savitzky-Golay, baseline, SNV |
| [`segmentation.py`](../hsi_workflow/segmentation.py) | 2-cluster KMeans film/substrate split (helper) |
| [`rois.py`](../hsi_workflow/rois.py) | Fixed-patch ROI tiling → ML table + specimen split |
| [`explore.py`](../hsi_workflow/explore.py) | Stage 4 — exploratory figures |
| [`decomposition.py`](../hsi_workflow/decomposition.py) | Stage 5 — PCA |
| [`clustering.py`](../hsi_workflow/clustering.py) | Stages 6–7 — clustering registry + maps + metrics |
| [`anomaly.py`](../hsi_workflow/anomaly.py) | Stage 8 — anomaly detector registry |
| [`postprocess.py`](../hsi_workflow/postprocess.py) | Stage 9 — spatial cleanup |
| [`regions.py`](../hsi_workflow/regions.py) | Stages 10–11 — region characterization |
| [`viz.py`](../hsi_workflow/viz.py) | Preview panels (preprocess, PCA, analysis maps) |
| [`optical_density.py`](../hsi_workflow/optical_density.py) | Old Step 10 — retained, off the default path |
| [`pipeline.py`](../hsi_workflow/pipeline.py) | Orchestrator tying stages together |
| `run_extract.py` / `run_explore.py` / `run_analyze.py` | Command-line entry points |

## The config system

Every stage has a dataclass with a `validate()` method and plain fields carrying a
short comment. `WorkflowConfig` bundles them:

```python
from hsi_workflow.config import WorkflowConfig

cfg = WorkflowConfig()          # sensible defaults everywhere
cfg.pca.n_components = 5
cfg.cluster.method = "gmm"
cfg.anomaly.methods = ["iforest", "lof", "mahalanobis"]
cfg.anomaly.fit_on = "baseline"
cfg.validate()                  # validates all stages at once
```

The stage configs:

| Config | Key fields |
|---|---|
| `PreprocessConfig` | `calibrate`, `smooth`+`sg_window`/`sg_polyorder`, `baseline`, `normalize`, `od_method` |
| `PieceConfig` | `method` (sam/mahalanobis/kmeans), `border_width`, `threshold`, `open_iter`/`close_iter`, `min_area`, `watershed_split` |
| `RoiConfig` | `patch`, `stride`, `min_coverage`, `save_patches` |
| `PcaConfig` | `n_components`, `whiten`, `max_fit_pixels` |
| `ClusterConfig` | `method`, `n_clusters`, `dbscan_eps`/`dbscan_min_samples` |
| `AnomalyConfig` | `methods`, `fit_on`, `contamination`, `anomaly_percentile` |
| `PostprocConfig` | `median_size`, `opening_radius`, `min_component` |

## Datasets are presets

A `DatasetConfig` says where a scan lives, how it's named/calibrated, and its
`material` (`"silicon"` or `"sio2"`). Adding a new scan = adding a preset — no code
changes elsewhere. Current presets in `config.DATASETS`:

| Name | File | Material |
|---|---|---|
| `lig` | LIG ROI scans (paired) | sio2 (test bed) |
| `sio2_bare_si` | `bare silicon all.bip` | **silicon** (baseline) |
| `sio2_dish_white_20` | `sio2 all 20 dish white.bil` | sio2 |
| `sio2_dish_black` | `Dish on Black - 1.bip` | sio2 |
| `sio2_dish_white_1` | `Dish on White 1.bip` | sio2 |

`material` rides along from the dataset → `Cube` → `Piece` → `Roi`, so the anomaly
stage can tell baseline from experimental. `DEFAULT_BASELINE = "sio2_bare_si"`.

## The registry pattern (how to add an algorithm)

Clustering and anomaly detection use a `{name: builder}` dictionary so adding a
method is one function + one entry — the orchestrator and CLIs don't change.

**Add a clustering algorithm** (`clustering.py`):

```python
def _spectral(features, cfg):
    from sklearn.cluster import SpectralClustering
    return SpectralClustering(n_clusters=cfg.n_clusters).fit_predict(features)

_CLUSTERERS["spectral"] = _spectral   # done — selectable via cfg.cluster.method
```

**Add an anomaly detector** (`anomaly.py`): implement a class with `.fit(normal_X)`
and `.score(X)` (higher = more anomalous), then register it:

```python
class MyDetector:
    def fit(self, normal_X): ...; return self
    def score(self, X): ...        # returns (n,) higher-is-weirder

_DETECTORS["mydet"] = lambda cfg: MyDetector()
```

## Data objects (what flows between stages)

- `io.Cube` — `data (rows, cols, bands)`, `wavelengths`, `shutter`, `ceiling`,
  `label`, `material`.
- `pieces.Piece` — `data`, `mask` (which bbox pixels are the fragment), `material`,
  `piece_id`, `source_label`, `bbox`. `foreground_spectra()` returns in-mask rows.
- `preprocessing.Preprocessed` — the processed cube + saturation mask (+ optional
  film/substrate segmentation for the OD path).
- `rois.Roi` — one patch's features (`mean_spectrum`, `std`, `spectral_variance`,
  `pca`, `anomaly`, hierarchical ids).
- `pipeline.PieceAnalysis` / `WorkflowResult` — per-piece maps/tables and the shared
  PCA/detectors.

## Orchestration

`pipeline.run_workflow(target, wf, baseline)`:

1. `prepare_pieces(baseline)` and `prepare_pieces(target)` — extract + preprocess.
2. `fit_pca` on pooled foreground (baseline + target).
3. `fit_detectors` on the "normal" population (`fit_on`), compute flag thresholds.
4. `analyze_piece` for each target piece — cluster, score, clean, characterize, tile ROIs.
5. Aggregate the cross-specimen ROI table.

Returns a `WorkflowResult`. The CLIs are thin wrappers that build a `WorkflowConfig`
from arguments, call this, and write figures/tables.

## Design rules that keep it modular

- **Spectral, not RGB** — every mask/decision uses full 300-band spectra; pseudo-RGB
  is display-only.
- **Front-ends are independent** — piece and ROI extraction produce plain data that
  downstream stages consume; they don't care how it was produced.
- **Docstring-heavy** — each function explains *what and why*, matching the existing
  style, so changes are safe to make.
