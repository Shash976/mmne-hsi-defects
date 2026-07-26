# Piece & ROI Extraction — the spectral front-end

This is the novel part of the pipeline: turning one raw scan of **many pieces on a
dish** into individual pieces, then into ROI samples. Every decision here uses the
**full 300-band spectrum**, never RGB brightness.

```
raw scan (e.g. 1417×900×300, ~20 SiO₂ pieces on a dish)
      │  pieces.extract_pieces  — spectral foreground + connected components
      ▼
individual PIECE sub-cubes (one per fragment: bbox crop + mask, all 300 bands)
      │  rois.tile_rois  — fixed patch grid (8×8, stride 4) inside each piece mask
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

### 5. Erode the analysis mask

`erode_iter` (default **1 px**) shrinks each piece's mask inward *after* the bbox
is fixed. Boundary pixels are **mixed** (part film, part dish), so without this the
anomaly detectors flag the rim rather than the film. The crop and bbox are
unaffected — only which pixels count as film. If erosion would erase a piece
entirely, the un-eroded mask is kept so no specimen drops out of the study.

Each `Piece` carries:

| Field | Meaning |
|---|---|
| `data` | the (rows, cols, 300) sub-cube for the bounding box |
| `mask` | which bbox pixels are actually the fragment (the rest is dish) |
| `material` | inherited from the source scan (`silicon` / `sio2`) |
| `piece_id`, `source_label`, `bbox` | provenance |

**Single-piece scans (LIG):** the foreground is one blob, so the whole frame comes
back as one `Piece` and downstream code is identical.

**Persisting crops:** `dataset.export_dataset` (via `run_organize` /
`run_extract`) writes each piece as its own calibrated ENVI `.hdr`/data pair plus
a `*_mask.npy`, so a pipeline can restart from crops.

### Extraction-quality knobs (esp. the white-dish 20-piece scan)

The default border-frame background fails when the outer frame is contaminated
(two dishes, rim, paper, shadow). Three `PieceConfig` knobs fix this — dial them in
live with `debug_masks.py` (see [debug_tools.md](debug_tools.md)):

- **`background_bbox=(r0,r1,c0,c1)`** — take the background spectrum from a clean
  empty-dish region you specify, instead of the border. The single biggest lever.
  In the tuner, press `R` and drag the box.
- **`flat_field=True`** (+ `flat_field_sigma`) — divide out a smooth spatial
  illumination gradient (`flat_field_correct`) before the distance is computed.
- **`on_reflectance=True`** — find the piece masks on white/dark-calibrated
  reflectance (the piece *data* is still cropped from the raw cube, so the
  per-piece `preprocess()` still calibrates exactly once).

---

## SiO₂ film extraction (`film.py`) — Stage 3.1b

A wafer piece is part bare silicon, part SiO₂. `extract_film(piece, cfg, ref)`
isolates the oxide *within* a piece's mask, mirroring piece extraction but
referencing **bare silicon** instead of the dish (SiO₂ on Si shows thin-film
interference, so its reflectance differs in shape from bare silicon).

`FilmConfig.reference` picks what "bare silicon" is:

- **`"control"`** — the mean spectrum of the `sio2_bare_si` control dataset (one
  global reference in the analysis space); SiO₂ = pixels unlike it.
- **`"in_piece"`** — a 2-cluster split of the wafer; the bare cluster is the one
  spectrally closest to the control reference. Robust to per-wafer lighting.

Off the main path by default (`enabled=False`). Enable it with
`run_analyze --extract-film [--film-reference control|in_piece]`: SiO₂ pieces then
carry only their oxide mask, so PCA/clustering/anomaly run on the film alone
(bare-silicon baseline pieces are never film-masked). Tune it live with
`debug_film.py`.

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
generalize to *new* samples?" `run_analyze` runs this automatically (when ≥ 2
specimens have ROIs) and writes the held-out scores to `roi_evaluation.csv`.

> **Sizing note:** the document targets ~100–300 ROIs per piece, assuming large
> images. These pieces are small (median ~4,175 foreground px), so the original
> 32×32 non-overlapping grid yielded only ~1–4 ROIs per piece (73 across all 41).
> The defaults are therefore `patch=8, stride=4`, measured at a median of ~195
> ROIs per piece. Because `stride < patch` the patches **overlap**, so ROIs within
> a piece are spatially correlated — safe only because `split_by_specimen` holds
> out whole specimens (above). Never switch to a random per-ROI split.
> See [tuning.md](tuning.md).

---

## On-disk dataset layout

`dataset.export_dataset` (driven by `run_organize`, which targets the repo's
`data/` folder and also writes the sample database, or by `run_extract` for an
arbitrary out-root) writes the hierarchy the document recommends —
**specimen → piece → ROI** — so the data is organized and easy to modify:

```
data/organized/<dataset>/        (run_organize; run_extract uses out/workflow/extract/)
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
