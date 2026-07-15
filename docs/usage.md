# Usage — CLIs, arguments, and reading the outputs

Run everything in the `hsi` conda environment:

```bash
conda run -n hsi python -m hsi_workflow.<command> [args]
```

The three entry points map to phases of the pipeline.

---

## `run_extract` — organize into a piece/ROI dataset

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
meaningful.

```bash
python -m hsi_workflow.run_explore --dataset sio2_dish_white_20
```

| Argument | Default | Meaning |
|---|---|---|
| `--dataset` | `sio2_bare_si` | Which preset to explore |
| `--out` | `out/workflow/explore` | Output root |

**Outputs:** `<piece>_explore.png` per piece, `material_mean_spectra.png`. Prints the
mean spectral variance per piece and per material — silicon should be **low**.

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
| `--fit-on` | `self` | `self` (anomalies within film) or `baseline` (vs silicon) |
| `--anomaly-percentile` | `97.5` | Flag threshold percentile of normal scores |
| `--out` | `out/workflow/analyze` | Output root |

**Outputs** (`out/workflow/analyze/<target>/`):

- `pca_summary.png` — explained variance + PC loadings
- `<piece>_analysis.png` — 6-panel: RGB, PC1–3, cluster map, anomaly heatmap,
  flagged regions overlay, PC1 map
- `<piece>_regions.csv` — the region table for that piece
- `roi_table.csv` — the cross-specimen ML table

Prints a per-piece summary: silhouette, #clusters, anomalous fraction, #regions.

---

## Reading the outputs

### The 6-panel analysis figure

The header shows the piece id, material, anomalous fraction, region count, and
silhouette. Panels (dark = off-piece):

| Panel | What to look for |
|---|---|
| Spectral structure (PC1–3) | broad spectral variation as false colour; smooth = homogeneous film |
| Clusters | discrete spectral populations, with a legend; coherent bands are good |
| Anomaly score | bright = unusual; should be sparse and localized |
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
