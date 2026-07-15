# Analysis — PCA, clustering, anomaly, cleanup, regions

The analysis stages turn preprocessed piece spectra into cluster maps, anomaly
heatmaps, and region tables. All run on the **SG + SNV reflectance** produced by
preprocessing. Orchestrated by `pipeline.analyze_piece` / `pipeline.run_workflow`.

## PCA (`decomposition.py`)

`fit_pca(pooled_spectra, cfg)` fits `sklearn.decomposition.PCA` on a **pooled,
subsampled** set of spectra (baseline + target) so one basis is shared across all
pieces. The returned `PcaModel` gives:

- `transform(flat)` → PC scores for clustering/anomaly,
- `score_image(cube)` → per-pixel PC score maps,
- `explained_variance_ratio`, `loadings`.

Subsampling is bounded by `PcaConfig.max_fit_pixels`. We observed **PC1 ≈ 84%,
PC2 ≈ 9%, PC3 ≈ 3%** on the dish-black SiO₂ set — a healthy, low-noise result.

## Clustering (`clustering.py`)

`cluster(features, cfg)` runs a method from the registry on the PCA scores:

| Method | Notes |
|---|---|
| `kmeans` (default) | k = `n_clusters` (default 4), spherical clusters |
| `dbscan` | density-based; `eps`/`min_samples`; label `-1` = noise |
| `gmm` | Gaussian mixture; soft clusters, hardened to labels |

`cluster_map(result, shape, mask)` reshapes labels back to the image (off-mask =
`-1`). `cluster_metrics` reports **silhouette**, **Davies-Bouldin**, and
**Calinski-Harabasz** on a subsample. Clusters are *spectral populations only* — we
never label them as physical defects.

## Anomaly scoring (`anomaly.py`) — the important one

Every detector implements the same tiny protocol: `fit(normal_X)` and
`score(X)` (higher = more anomalous). Registry:

| Method | Idea |
|---|---|
| `iforest` | Isolation Forest — how easily a point is isolated by random splits |
| `lof` | Local Outlier Factor — density vs local neighborhood (novelty mode) |
| `mahalanobis` | Distance from the normal mean under a shrinkage covariance (the RX detector, generalized from `legacy/`) |
| `ocsvm` | One-Class SVM — signed distance to a learned boundary |

`fit_detectors(normal_X, cfg)` fits every method in `cfg.methods` on the **normal**
population; `flag_threshold` sets the flag cutoff at a high percentile of the normal
scores; `anomaly_map` paints per-pixel scores back to the image. Defaults:
Isolation Forest + Mahalanobis.

### What is "normal"? The `fit_on` switch (read this)

This single setting changes the *meaning* of the results:

| `AnomalyConfig.fit_on` | "Normal" = | Result |
|---|---|---|
| `"self"` **(default)** | the target's own majority population | **Localized anomalies within the film** — the small unusual regions the objective wants |
| `"baseline"` | the bare-silicon dataset | **Material contrast** — every SiO₂ pixel scored by how unlike silicon it is |

**Why the default is `"self"`:** silicon and SiO₂ are *different materials*, so if
you fit "normal" on silicon and score SiO₂, essentially **100% of the film is
flagged** (it's all unlike silicon). That's the degenerate result we saw first. The
document's real intent — "which ROIs differ from the **majority**", "small localized
regions" — is intra-sample outlier detection, i.e. `fit_on="self"`. With it, the
anomalous fraction dropped to a sensible **0–4%**, localized.

The silicon baseline is **still used** regardless of `fit_on`, for the per-region
"distance from silicon baseline" feature (a spectral-space Mahalanobis fit on
silicon).

## Spatial postprocessing (`postprocess.py`)

`clean_binary_map(flag, cfg)`:

1. **median filter** (`median_size`) — smooth away single-pixel speckle,
2. **morphological opening** (`opening_radius`) — remove thin protrusions,
3. **min component** (`min_component`) — drop connected blobs smaller than N pixels.

Flags are then clamped to the piece mask (`flagged &= mask`) so filters can't bleed
onto the dish. `label_regions` gives the connected components for characterization.

## Region characterization (`regions.py`)

`characterize_regions(labels, n, cube, anomaly_map, baseline_detector)` measures each
region into a `RegionStats`:

| Field | Meaning |
|---|---|
| `area`, `perimeter`, `compactness` | size and shape (`compactness = 4π·area / perimeter²`, 1 = disk) |
| `centroid` | location |
| `mean_reflectance` | mean of the analysis spectrum over the region* |
| `spectral_variance` | heterogeneity within the region |
| `baseline_distance` | Mahalanobis distance of the region mean to the **silicon** baseline |
| `mean_anomaly` | average anomaly score over the region |

`regions_to_table` tidies these into a DataFrame (the document's region table).
`spectral_distance_map` produces the standalone "distance from a reference spectrum"
map (Euclidean, or Mahalanobis with a precision matrix).

> *`mean_reflectance` is computed on the **SNV** analysis cube, which is
> zero-centered per pixel — so it reads near 0, not true 0–1 reflectance. Treat it as
> "mean analysis value." See [tuning.md](tuning.md) if you need physical reflectance
> here.

## How it all composes per piece

`analyze_piece` (in `pipeline.py`) does, for one target piece:

```
fg   = piece.foreground_spectra()          # in-mask spectra
feat = pca.transform(fg)                    # PCA scores
cluster(feat) → cluster_map                 # Stages 6–7
score(feat) for each detector → anomaly_map # Stage 8
flag = score > threshold → clean → &mask    # Stage 9
label_regions → characterize_regions        # Stages 10–11
tile_rois(piece) + attach pca/anomaly        # ROI track
```

producing a `PieceAnalysis` (maps + region table + ROIs) that the CLI writes out.
