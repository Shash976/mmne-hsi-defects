# Usage — CLIs, arguments, and reading the outputs

Run everything in the `hsi` conda environment:

```bash
conda run -n hsi python -m hsi_workflow.<command> [args]
```

The entry points map to phases of the pipeline.

---

## `run_organize` — build the `data/` sample inventory + organized tree

Implements the document's **Sample Inventory** stage: extracts pieces from every
semiconductor scan and writes the hierarchical Specimen → Piece → ROI tree plus
the sample database into the repo's `data/` folder.

```bash
python -m hsi_workflow.run_organize                       # all four Si/SiO₂ presets
python -m hsi_workflow.run_organize --datasets sio2_bare_si sio2_dish_black
python -m hsi_workflow.run_organize --no-roi-cubes        # skip per-ROI cubes
```

| Argument | Default | Meaning |
|---|---|---|
| `--datasets` | the 4 Si/SiO₂ presets | Which presets to organize |
| `--data-root` | `data` | Repo folder to organize into |
| `--radiometry` | `reflectance` | Cropped cubes as calibrated reflectance or `raw` DN |
| `--patch` / `--stride` / `--min-coverage` | `32`/`32`/`0.85` | ROI tiling |
| `--no-roi-cubes` | off | Keep folders + `roi_index.csv`, skip ROI cubes |

**Outputs:** `data/samples.csv` (the sample database — fill in `notes` by hand),
`data/inventory_summary.json` (counts, sizes, imaging area),
`data/manifest.json` (raw-scan + calibration provenance), and
`data/organized/<dataset>/<piece_id>/...` trees. See [data/README.md](../data/README.md).

---

## `run_extract` — organize into a piece/ROI dataset (generic out-root)

Splits each scan into pieces, tiles ROIs, and writes a **hierarchical on-disk
dataset**: one folder per piece, each with the cropped piece cube and a `rois/`
subfolder of individual cropped ROI cubes, plus metadata.

```bash
python -m hsi_workflow.run_extract --dataset sio2_bare_si
python -m hsi_workflow.run_extract --dataset sio2_dish_white_20 --radiometry raw --no-roi-cubes
```

| Argument | Default | Meaning |
|---|---|---|
| `--dataset` | `sio2_bare_si` | Which preset to process |
| `--radiometry` | `reflectance` | Save cropped cubes as calibrated reflectance or `raw` DN |
| `--piece-method` | `sam` | Foreground backend (`sam`/`mahalanobis`/`kmeans`) |
| `--min-area` | `1000` | Minimum piece size (px) |
| `--patch` / `--stride` | `32` / `32` | ROI patch size and step |
| `--min-coverage` | `0.85` | Fraction of a patch that must be in-mask |
| `--no-roi-cubes` | off | Skip per-ROI cubes (keep folders + `roi_index.csv`) |
| `--out` | `out/workflow/extract` | Output root |

**Output tree** (`out/workflow/extract/<dataset>/`):

```
manifest.json                 # dataset index (pieces, counts, material)
roi_table.csv                 # aggregated ML table (mean spectra + scalar features)
<piece_id>/
    <piece_id>.hdr / .img     # cropped piece cube (ENVI, reflectance by default)
    <piece_id>_mask.npy       # fragment footprint
    meta.json                 # material, bbox-in-scan, shape, counts
    roi_index.csv             # one row per ROI in this piece
    rois/
        <roi_id>.hdr / .img   # cropped ROI sub-cube
```

Every cube is a standard ENVI pair (wavelengths preserved) — reload with
`hsi_workflow.io.load_cube`. See [extraction.md](extraction.md#on-disk-dataset-layout).

---

## `run_explore` — Stage 4 exploratory figures

Per-piece mean spectrum, band images, RGB, and spectral variance map, plus a
by-material mean-spectra overlay. **Uses reflectance (SNV off)** so variance is
meaningful. Pass **several presets** to get the control-vs-experimental
comparison (silicon + SiO₂) in one figure:

```bash
python -m hsi_workflow.run_explore --dataset sio2_bare_si sio2_dish_black
```

| Argument | Default | Meaning |
|---|---|---|
| `--dataset` | `sio2_bare_si` | One or more presets (space separated) |
| `--out` | `out/workflow/explore` | Output root (subfolder = joined preset names) |

**Outputs:** `<piece>_explore.png` per piece, `material_mean_spectra.png`
(genuinely overlays materials when you pass both), `material_variance.csv`
(the Si-low / SiO₂-high check, persisted), `noise_metrics.csv` (RMS noise + SNR
before/after SG smoothing), and `reflectance_histogram.png` (the Stage-2 "values
mostly 0–1, no clipping" check).

---

## `run_analyze` — Stages 5–11 full analysis

Fits PCA + anomaly detectors, analyzes every target piece, writes maps + region
tables + the ROI table.

```bash
python -m hsi_workflow.run_analyze \
    --target sio2_dish_white_20 --baseline sio2_bare_si \
    --cluster kmeans --n-clusters 4 \
    --anomaly iforest mahalanobis --fit-on self
```

| Argument | Default | Meaning |
|---|---|---|
| `--target` | `sio2_dish_white_20` | Dataset to screen for anomalies |
| `--baseline` | `sio2_bare_si` | Silicon control dataset |
| `--pca-components` | `3` | Number of PCs |
| `--cluster` | `kmeans` | `kmeans`/`dbscan`/`gmm` |
| `--n-clusters` | `4` | k for kmeans/gmm |
| `--anomaly` | `iforest mahalanobis` | One or more detectors |
| `--fit-on` | `self` | Which population drives the *flags*: `self` (within film) or `baseline` (vs silicon). The silicon-contrast map is produced either way |
| `--anomaly-percentile` | `97.5` | Flag threshold percentile of normal scores |
| `--compare-clusters` | off | Also run kmeans/dbscan/gmm on the same features → stability CSV |
| `--out` | `out/workflow/analyze` | Output root |

**Outputs** (`out/workflow/analyze/<target>/`):

- `pca_summary.png` — explained variance + PC loadings
- `pca_scatter.png` — PC1 vs PC2 colored by piece (Stage 5 deliverable)
- `spectral_histogram.png` — distribution of the analysis values
- `<piece>_analysis.png` — 9-panel: PC1–3, cluster map, within-film anomaly
  heatmap, **silicon-baseline contrast**, **spectral distance**, **0–1 anomaly
  probability**, flagged regions overlay, normal-vs-anomalous spectra, score
  histogram
- `<piece>_regions.csv` — the region table (**always written**; empty = nothing
  flagged, so stale files can't survive a rerun)
- `roi_table.csv` — the cross-specimen ML table
- `roi_evaluation.csv` — specimen-level hold-out scores (`split_by_specimen`;
  written when ≥ 2 specimens have ROIs)
- `cluster_comparison.csv` — per-method metrics + pairwise adjusted Rand index
  (with `--compare-clusters`)
- `report.md` — the Stage-11 final report (per-piece stats, edge-share
  diagnostics, the document's questions answered)

Prints a per-piece summary: silhouette, #clusters, anomalous fraction, #regions.

---

## Reading the outputs

### The 9-panel analysis figure

The header shows the piece id, material, anomalous fraction, region count, and
silhouette. Panels (dark = off-piece):

| Panel | What to look for |
|---|---|
| Spectral structure (PC1–3) | broad spectral variation as false colour; smooth = homogeneous film |
| Clusters | discrete spectral populations, with a legend; coherent bands are good |
| Anomaly score (within-film) | bright = unusual vs the film's own majority; should be sparse and localized |
| Distance from Si baseline | the hypothesis deliverable — uniformly high is just the material difference; *variation* is the signal |
| Spectral distance from piece mean | a model-free cross-check of the anomaly map |
| Anomaly probability (0–1) | the score map rescaled for comparison across pieces |
| Flagged anomalies | regions **outlined in red and numbered** (numbers match the region CSV) |
| **Mean spectrum: normal vs anomalous** | *the payoff* — how the flagged spectra differ in shape from the film |
| Anomaly score distribution | histogram with the flag threshold; the tail past the line is what gets flagged |

The mean-spectrum panel is the one to read first when asking "is this a real
anomaly?" — a genuine anomaly shows a distinct spectral shape (a shifted peak, a
dip), not just noise.

### The region table (`<piece>_regions.csv`)

One row per flagged region: `area, perimeter, compactness, centroid, mean_anomaly,
spectral_variance, baseline_distance`. Rank by `area` or `mean_anomaly` to prioritize
follow-up. Ask the objective's questions: are anomalies localized? repeated? near
edges? (Edge-heavy anomalies are often a masking artifact — see [tuning.md](tuning.md).)

### The ROI table (`roi_table.csv`)

One row per patch. Metadata (`roi_id, specimen, image, material`), scalar features,
`pca_1..k`, `anomaly_<method>`, and 300 wavelength-named mean-spectrum columns. Use
`rois.split_by_specimen` for leakage-free train/test splits.

---

## Programmatic use

```python
from hsi_workflow.config import WorkflowConfig
from hsi_workflow.pipeline import run_workflow

cfg = WorkflowConfig()
cfg.anomaly.methods = ["iforest", "lof", "mahalanobis"]
res = run_workflow("sio2_dish_black", cfg, baseline="sio2_bare_si")

for a in res.analyses:
    print(a.piece.piece_id, len(a.regions), a.cluster_metrics["silhouette"])
res.roi_table.to_parquet("rois.parquet")
```
