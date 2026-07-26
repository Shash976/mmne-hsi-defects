# Tuning, gotchas, and known limitations

Practical guidance for getting good results, plus the sharp edges to know about.

## Knobs that matter most

| Symptom | Knob | Where |
|---|---|---|
| Dish/background not separating (messy mask everywhere) | `background_bbox` — point at a clean empty-dish region (press `R` in `debug_masks.py`) | `PieceConfig` |
| Uneven lighting biases the mask | `flat_field=True` (+ `flat_field_sigma`), or `on_reflectance=True` | `PieceConfig` |
| SiO₂ film not isolated from bare silicon within a wafer | `FilmConfig(enabled=True, ...)` / `run_analyze --extract-film` | `FilmConfig` |
| Patterned SiO₂ chip breaks into many pieces | ↑ `close_iter`, ↑ `min_area` | `PieceConfig` / `--min-area` |
| Mask covers only the coloured oxide, not the dark wafer | `method="euclidean"` (gotcha #3b) | `PieceConfig` / `--piece-method` |
| Separate wafers merged into one giant "piece" | ↓ `close_iter` (6 → 2) | `PieceConfig` |
| `mahalanobis` finds no pieces at all | `threshold="percentile"` (gotcha #3c) | `PieceConfig` |
| Dish rim/dust leaks in as "pieces" | ↑ `open_iter`, ↑ `min_area` | `PieceConfig` |
| Touching pieces merged into one | `watershed_split=True` | `PieceConfig` |
| Too few ROIs per piece | ↓ `patch`, ↓ `stride` (overlap) | `RoiConfig` / `--patch` `--stride` |
| Everything flagged anomalous | `fit_on="self"` | `AnomalyConfig` / `--fit-on` |
| Anomaly map too noisy | ↑ `median_size`, ↑ `min_component` | `PostprocConfig` |
| Anomaly fraction too high/low | `anomaly_percentile`, `contamination` | `AnomalyConfig` |
| Over-smoothed spectra | ↓ `sg_window` | `PreprocessConfig` |

## Gotchas (things that will bite you if you forget)

### 1. `fit_on` decides what "normal" means
Fitting anomaly detectors on the **silicon baseline** flags ~100% of SiO₂ (it's a
different material). Default `fit_on="self"` finds anomalies *within* the film. Use
`"baseline"` only when you specifically want a material-contrast map. Detailed in
[analysis.md](analysis.md).

### 2. SNV flattens per-pixel variance
SNV normalizes every pixel to zero-mean/unit-variance. So:
- The **variance map** must be computed on **reflectance** — `run_explore` sets
  `normalize="none"` for this reason.
- Region tables report both `mean_reflectance` (physical, from the pre-SNV
  band-mean that every `Piece` now carries) and `mean_snv` (the analysis-space
  mean, ≈ 0 by construction). ROI `std`/`spectral_variance` in the organized
  dataset export are computed on the reflectance cube for the same reason.

### 3. Anomalies love edges
Piece boundaries have **mixed pixels** (part fragment, part dish) whose spectra are
genuinely unusual, so they light up the anomaly map. This is real but usually not
what you care about. Mitigation: `PieceConfig.erode_iter` (default **1 px**,
`run_analyze --erode N`) shrinks each piece's *analysis* mask inward before ROI
tiling and anomaly fitting; the crop/bbox is unchanged. A piece that erosion would
erase entirely keeps its un-eroded mask rather than dropping out of the study.

Erosion helps but does not fully solve it: on `sio2_dish_white_20`, 1 px cut the
mean edge share from 39% → 34%, yet individual pieces stayed edge-dominated
(p13 99%, p12 61%). The `edge share` column in `report.md` is the number to watch —
raise `--erode` if it stays high, at the cost of usable film area on small pieces.

### 3b. `method="sam"` silently drops dark substrate
SAM is **scale-invariant** — it compares spectral *shape* and normalizes magnitude
away. Bare silicon is dark but spectrally smooth, so its angle to a bright dish is
tiny and it fails the foreground test, while interference-coloured SiO₂ passes.
On a partly-coated wafer the mask therefore keeps only the oxide and clips the
substrate off. Measured on `sio2_dish_white_20`: **85k** foreground px with `sam`
vs **179k** with `euclidean` — roughly half of every wafer.

Use `method="euclidean"` whenever a piece must include dark substrate. It is
magnitude sensitive but still uses all 300 bands (it is *not* a brightness
threshold). `sam` remains the default for datasets already tuned around it.

### 3c. `method="mahalanobis"` + Otsu returns nothing
The Mahalanobis map is heavy-tailed: background pixels sit at d² ≈ n_bands (χ²
expectation — measured median 278 for 300 bands), but specular/rim outliers reach
d² ≈ 5.6e5. Otsu histograms that 2000× range into 256 bins, so nearly every pixel
lands in bin 0 and the threshold splits off only the extreme tail — **0.24% of
pixels, zero surviving pieces**. The distance is correct; the *thresholding* is
what breaks. Otsu assumes a **bimodal** histogram, and this map is a tight
background mode plus a long continuous tail.

Use `threshold="chi2"` (`chi2_quantile`, default 0.999), which is grounded in the
model: if the background is multivariate normal, its squared distance is χ² with
`df = n_bands`, so anything past a high quantile is not background.

**Measured on `sio2_dish_white_20`** (`close_iter=2, min_area=500`):

| method + threshold | pieces | foreground px |
|---|---|---|
| `mahalanobis` + `otsu` | **0** | — |
| `mahalanobis` + `chi2` | 18 | 24,415 |
| `mahalanobis` + `percentile` (p86) | 30 | 79,269 |
| `euclidean` + `otsu` | 18 | **172,220** |

So `chi2` fixes the silent-zero failure, but be honest about what it buys: it
selects only the *cores* of wafers (median piece 850 px), and percentile
over-fragments into 30 pieces. **For whole-wafer extraction on this scan,
`euclidean` is the right backend** — mahalanobis is now usable rather than good.
When extraction does yield nothing, `prepare_pieces` raises a clear error naming
the dataset instead of failing later inside PCA.

### 4. Large reference cubes
The white/dark references are ~750 MB each. `io.load_reference_spectrum` is
`lru_cache`d so they load **once per process** — but each CLI invocation is a fresh
process, so expect a one-time load cost at startup.

### 5. `conda run -c` and newlines
`conda run -n hsi python -c "<multi-line>"` fails. Put multi-line code in a `.py`
file and run that; if it's outside the repo, add
`sys.path.insert(0, r"...\HSI")` so `hsi_workflow` imports.

### 6. LIG calibration paths
The `LIG` preset's white/dark now point at the shared
`...\hsi\calibration_whitedark\` folder (the old `lig_dataset\calibration_whitedark`
path didn't exist). Verify before trusting LIG calibration.

## Known limitations (honest list)

- **Mask erosion only partly fixes edge-dominated anomalies** (gotcha #3). Some
  pieces remain edge-heavy at the 1 px default; the report's `edge share` column
  quantifies this per piece. Boundary-region *filtering* is still unimplemented.
- **ROI defaults are tuned for these small pieces** (`patch=8, stride=4`, median
  ~195 ROIs/piece). Because `stride < patch` the ROIs overlap, so ROIs within one
  piece are correlated — this is only safe because `split_by_specimen` holds out
  whole specimens. Never switch to a random per-ROI split.
- **Per-ROI anomaly scoring loops** one ROI at a time — fine for hundreds of ROIs,
  could be batched if you scale to many thousands.
- **Whole pieces held in memory** during a run (bbox-cropped, so far manageable).
- **Anomaly "probability" is a percentile rescale** of the raw scores (ranking
  preserved), not a calibrated statistical probability.

## Ideas / future work

- Boundary-region filtering (drop regions *touching* the mask edge) to finish the
  job mask erosion starts.
- ROI patch/stride sized automatically per piece area, instead of one global default.
- Batch the per-ROI scoring.
- Stage 12: wire representative regions to a follow-up SEM/AFM/Raman worklist.

## Interactive tuning tools

Before committing knob values to `config.py`, find them visually. See
[debug_tools.md](debug_tools.md) for the full step-by-step guide.

- `python debug_preprocess.py --dataset <name>` — SG window/polyorder sliders,
  calibrate/SNV/baseline toggles, click-a-pixel before/after spectra, live noise
  metrics, display contrast, and `shift+click` reference-subtract. `p` prints a
  paste-ready `PreprocessConfig(...)`.
- `python debug_masks.py --dataset <name>` — extraction method/threshold/
  morphology/min-area sliders with a live mask overlay, labeled-piece view, and
  the ROI grid. Press `R` to drag a clean-background reference box, toggle
  flat-field / calibrate / watershed. `p` prints `PieceConfig`/`RoiConfig`
  (spatial values rescaled to full resolution).
- `python debug_film.py --dataset <name>` — SiO₂-within-wafer extractor: pick a
  piece, choose the bare-silicon reference (control / in-piece), tune the window
  and morphology. `p` prints `FilmConfig`.
- `notebooks/playground.ipynb` / `notebooks/sio2_extraction.ipynb` — ad-hoc
  scratchpads using the same package API.

All three scripts take `--crop R0 R1 C0 C1` and `--max-dim N` for big scans and
`--demo` for a synthetic cube. They are debounced and downsample the whole scan
for responsiveness.

## Sanity checklist for a new dataset

1. `run_organize` (or `run_extract`) → do the pieces look right? dish excluded?
   (`debug_masks.py` to fix extraction if not)
2. `run_explore --dataset sio2_bare_si <new_dataset>` → is silicon low-variance,
   SiO₂ higher? mean spectra plausible? reflectance mostly in [0, 1]?
3. `run_analyze` → PC1 a large fraction? silhouette positive? anomaly fraction small
   and localized (not ~100%)?
4. Open a `<piece>_analysis.png` → are flagged regions where you'd expect, or all on
   edges (masking artifact)? Check `report.md`'s edge-share column.
