# Piece & ROI Extraction — the spectral front-end

This is the novel part of the pipeline: turning one raw scan of **many pieces on a
dish** into individual pieces, then into ROI samples. Every decision here uses the
**full 300-band spectrum**, never RGB brightness.

```
raw scan (e.g. 1417×900×300, ~20 SiO₂ pieces on a dish)
      │  pieces.extract_pieces  — spectral foreground + connected components
      ▼
individual PIECE sub-cubes (one per fragment: bbox crop + mask, all 300 bands)
      │  rois.tile_rois  — fixed 32×32 patch grid inside each piece mask
      ▼
ROIs (one mean-spectrum sample per patch) ──► pixel maps + cross-specimen ROI table
```

---

## Piece extraction (`pieces.py`)

`extract_pieces(cube, cfg) -> list[Piece]`, in four steps:

### 1. Estimate the dish/background spectrum

`border_background_spectrum(cube, width)` takes the **outermost frame** of pixels
(almost always empty dish/holder) and returns their **median spectrum**. This is a
label-free estimate of "what background looks like" *for this specific scan* — so it
adapts to a black dish or a white dish automatically.

### 2. Flag foreground by spectral distance (not brightness)

`foreground_distance(cube, cfg)` computes, per pixel, how *unlike the background*
its spectrum is. Backends (`PieceConfig.method`):

- **`sam`** (default) — **Spectral Angle Mapper**: the angle between a pixel's
  spectrum and the background spectrum. Scale-invariant, so it keys on spectral
  *shape*, not intensity. Cheap and robust — this is what separated 10 clean silicon
  pieces and the SiO₂ pieces in testing.
- **`mahalanobis`** — distance accounting for the background's covariance
  (Ledoit-Wolf shrinkage), reusing the RX idea from `legacy/`.
- **`kmeans`** — a 2-cluster split over all bands; the cluster whose mean matches the
  background spectrum is called background.

The distance map is binarized (`_threshold_mask`) with Otsu (default) or a
percentile.

### 3. Clean the mask

`clean_mask(mask, cfg)` applies, in order:

- **opening** (`open_iter`) — erodes then dilates → removes thin dish-rim arcs and
  dust specks.
- **closing** (`close_iter`) — dilates then erodes → merges within-piece gaps so a
  **patterned device doesn't fragment** into many pieces.
- **fill holes** — solidifies each piece.

> Tuning `close_iter`/`min_area` is the main lever when patterned SiO₂ chips break
> into pieces or dish rim leaks in. See [tuning.md](tuning.md).

### 4. Label + crop

`label_pieces` runs connected-component labeling (`scipy.ndimage.label`; optional
watershed split for touching pieces via `watershed_split`) and keeps components
≥ `min_area`. Each surviving component's **bounding box** is cropped out of the full
cube. Pieces come back largest-first with ids `"<scan>_p01"`, `"<scan>_p02"`, …

Each `Piece` carries:

| Field | Meaning |
|---|---|
| `data` | the (rows, cols, 300) sub-cube for the bounding box |
| `mask` | which bbox pixels are actually the fragment (the rest is dish) |
| `material` | inherited from the source scan (`silicon` / `sio2`) |
| `piece_id`, `source_label`, `bbox` | provenance |

**Single-piece scans (LIG):** the foreground is one blob, so the whole frame comes
back as one `Piece` and downstream code is identical.

**Persisting crops:** `save_piece_crops(pieces, out_dir)` writes each piece as its own
ENVI `.hdr`/data pair plus a `*_mask.npy`, so a pipeline can restart from crops.

---

## ROI tiling

The `rois.py` module. `tile_rois(piece, cfg) -> list[Roi]` runs on **each piece**
(not the raw scan):

1. Lay a fixed **patch grid** (`RoiConfig.patch`, stepped by `stride`) over the piece.
2. Keep a patch only if at least `min_coverage` (default 0.85) of it is inside the
   piece mask — so ROIs never straddle the dish or the piece edge.
3. Compute each ROI's features from the in-mask spectra: `mean_spectrum` (300),
   `std`, `spectral_variance` (variance across the patch's pixels, a heterogeneity
   proxy). PCA scores and anomaly scores are attached later by the analysis stages.

### Why ROIs at all? (avoiding data leakage)

Neighboring pixels are almost identical (same physical spot), so training on every
pixel massively overstates performance — this is **spatial autocorrelation
leakage**. Making the ROI the unit of analysis, and organizing data hierarchically
(**specimen → image → ROI**), fixes it. Each ROI's `specimen` field is its piece id.

### The ML table

`build_roi_table(rois, wavelengths)` produces a tidy **pandas DataFrame**:

- ids/metadata: `roi_id, specimen, image, material`
- bbox + `coverage`
- scalar features: `std`, `spectral_variance`
- `pca_1..k` and `anomaly_<method>` (once the analysis stages fill them in)
- the mean spectrum expanded to per-band columns named by wavelength (`m450nm`, …)

### Leakage-free splits

`split_by_specimen(df, test_fraction, seed)` holds out **whole specimens** — every
ROI of a test piece goes entirely to the test set, none leaks into training. This is
the realistic evaluation the objective argues for: "how well does anomaly detection
generalize to *new* samples?" Verified to produce disjoint specimen sets.

> **Sizing note:** the document targets ~100–300 ROIs per piece. That assumes large
> images; small test pieces yield only a handful at 32×32. Shrink `patch` or
> `stride` for more ROIs — see [tuning.md](tuning.md).

---

## On-disk dataset layout

`dataset.export_dataset` (driven by `run_extract`) writes the hierarchy the
document recommends — **specimen → piece → ROI** — so the data is organized and
easy to modify:

```
out/workflow/extract/<dataset>/
    manifest.json                 # dataset index: pieces, counts, material, radiometry
    roi_table.csv                 # aggregated ML table (mean spectra + scalar features)
    <piece_id>/
        <piece_id>.hdr / .img     # cropped piece cube (ENVI; reflectance by default)
        <piece_id>_mask.npy       # fragment footprint within the crop
        meta.json                 # material, bbox-in-scan, shape, n_px, n_rois
        roi_index.csv             # one row per ROI (id, bbox, coverage, variance)
        rois/
            <roi_id>.hdr / .img   # cropped ROI sub-cube
            ...
```

- **Cubes are calibrated reflectance** by default (`--radiometry raw` keeps DN).
  ROI *features* in the ML table are computed on SNV — the on-disk cubes stay
  physical/reflectance so you can reprocess them however you like.
- Everything is a standard ENVI pair with wavelengths preserved; reload with
  `hsi_workflow.io.load_cube("<piece_id>.hdr")`.
- `--no-roi-cubes` keeps the folders + `roi_index.csv` but skips writing the many
  small ROI cubes (useful for the large 20-piece scan).
