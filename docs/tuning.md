# Tuning, gotchas, and known limitations

Practical guidance for getting good results, plus the sharp edges to know about.

## Knobs that matter most

| Symptom | Knob | Where |
|---|---|---|
| Patterned SiO₂ chip breaks into many pieces | ↑ `close_iter`, ↑ `min_area` | `PieceConfig` / `--min-area` |
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
- `region.mean_reflectance` on SNV data reads ≈ 0, not physical reflectance. It's
  "mean analysis value." If you need true reflectance per region, carry a
  pre-SNV cube through and recompute (not done by default).

### 3. Anomalies love edges
Piece boundaries have **mixed pixels** (part fragment, part dish) whose spectra are
genuinely unusual, so they light up the anomaly map. This is real but usually not
what you care about. Mitigations (not yet implemented — see below): erode the piece
mask a few pixels before analysis, or filter regions touching the mask boundary.

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

- **No mask erosion** → edge-dominated anomalies (gotcha #3).
- **`mean_reflectance` is SNV-space**, not physical reflectance (gotcha #2).
- **ROI yield is small on tiny test pieces** — the ~100–300/piece target assumes
  larger images; shrink `patch`/`stride` for the big `sio2_dish_white_20` scan.
- **Per-ROI anomaly scoring loops** one ROI at a time — fine for hundreds of ROIs,
  could be batched if you scale to many thousands.
- **Whole pieces held in memory** during a run (bbox-cropped, so far manageable).
- **Piece crops aren't calibrated on disk** — `save_piece_crops` stores raw analysis
  data; treat crops as convenience, not archival calibrated cubes.

## Ideas / future work

- Mask erosion + boundary-region filtering to suppress edge artifacts.
- Carry a reflectance cube alongside SNV so region tables report physical reflectance.
- Overlapping-ROI defaults sized per image to hit the 100–300 target automatically.
- ROI-level anomaly evaluation with `split_by_specimen` (train on some specimens,
  score held-out ones) as a generalization check.
- Batch the per-ROI scoring.
- Stage 12: wire representative regions to a follow-up SEM/AFM/Raman worklist.

## Sanity checklist for a new dataset

1. `run_extract --figures` → do the pieces look right? dish excluded?
2. `run_explore` → is silicon low-variance, SiO₂ higher? mean spectra plausible?
3. `run_analyze` → PC1 a large fraction? silhouette positive? anomaly fraction small
   and localized (not ~100%)?
4. Open a `<piece>_analysis.png` → are flagged regions where you'd expect, or all on
   edges (masking artifact)?
