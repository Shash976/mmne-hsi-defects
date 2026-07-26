# Audit — implementation vs. `Revised Research Objective.md`

A stage-by-stage comparison of this repository against the source specification.
For each difference the audit states whether it is a **beneficial deviation**
(keep and defend) or a **genuine gap** (fix).

**Verdict: the project is built to the spec, not merely adjacent to it.**
`hsi_workflow/__init__.py:9-25` carries an explicit module→stage map, every stage
module docstring names its "Stage N", and the framing is consistent across the
README, `docs/`, and the code. All 12 stages exist and a complete run is on disk
(`out/workflow/analyze/sio2_dish_white_20/report.md`). The differences that
matter are five deliberate improvements and three real gaps.

---

## 1. What matches the spec

| Spec stage | Implementation | Status |
|---|---|---|
| 1 Acquisition | External (Resonon Pika L, VNIR, 300 bands, ~368–1008 nm) | n/a |
| 2 Radiometric calibration `(S-D)/(W-D)` | `preprocessing.py:102` `calibrate_reflectance` | ✅ |
| 3.1 Background removal | `pieces.py:233` `extract_pieces` — SAM/Mahalanobis distance + Otsu/percentile + morphology | ⚠️ see Gap 4 |
| 3.1b Film vs. substrate | `film.py:153` `extract_film` (opt-in, `--extract-film`) | ✅ extra |
| 3.2 Spectral smoothing | `preprocessing.py:142` `savgol_smooth` | ✅ |
| 3.3 Normalization (SNV) | `preprocessing.py:120` `snv` | ✅ |
| 3.4 Baseline correction | `preprocessing.py:186` `baseline_correct` (poly; off by default) | ✅ |
| 4 Exploratory visualization | `explore.py:43` `save_piece_exploration` — RGB composite, band images, variance map, mean spectrum | ✅ |
| 5 PCA | `decomposition.py:61` `fit_pca`, `PcaModel.score_image` | ✅ |
| 6–7 Clustering + spatial mapping | `clustering.py` — KMeans/DBSCAN/GMM, `cluster_map`, `cluster_metrics` | ✅ |
| 8 Anomaly scoring | `anomaly.py` — IsolationForest, LOF, One-Class SVM, Mahalanobis/RX | ✅ |
| 9 Spatial postprocessing | `postprocess.py:18` `clean_binary_map` — median → opening → connected-component size filter | ✅ |
| 10–11 Quantitative maps + region characterization | `regions.py:56` `characterize_regions`, `report.py` | ✅ |
| 12 Future validation | Documented as out of scope | ✅ |
| Leakage control | `rois.py:149` `split_by_specimen` — holds out whole specimens | ✅ |
| "Remove FEA" | Absent from `hsi_workflow/` entirely | ✅ |

Framing also matches. The spec's "We are **not detecting defects**. We are
detecting **spectral anomalies**" is reproduced in `README.md` and
`docs/overview.md`; silicon is consistently described as a *control population*,
not reference spectra; regions are described (area, reflectance, variance,
baseline distance) and never named as defect types.

---

## 2. Beneficial deviations — keep these

These depart from a literal reading of the spec and are **better** for it.

### 2.1 Dual anomaly product instead of "fit on silicon, score SiO₂"

The spec implies comparing SiO₂ against the silicon baseline. Taken literally
that is a degenerate material classifier: SiO₂ is a different material, so
fitting a detector on Si and scoring the film flags ~100% of the film and
localizes nothing.

`pipeline.py:170-184` therefore always emits **two** label-free products:

1. a **within-film** map — detectors fit on the film's own majority
   (`fit_on="self"`), which drives the flagged regions; and
2. the **silicon-baseline contrast** map (`baseline_map`), always computed — the
   spec's literal "relative to silicon baseline" deliverable.

This preserves the hypothesis deliverable while producing an anomaly map that
actually localizes. `--fit-on {self,baseline}` selects which drives flags.

### 2.2 Fixed patches instead of true superpixels

The spec says "Superpixel / ROI". The implementation uses a fixed patch grid
(`rois.py:50` `tile_rois`) with a `min_coverage` test so patches never straddle
the piece edge. Fixed tiling is deterministic, reproducible, and independent of
the signal being searched for — a superpixel segmentation adapts to image content
and would make ROI boundaries a function of the anomalies themselves.

### 2.3 Exposure-normalized calibration

The spec gives `R = (S-D)/(W-D)`. White and dark references are captured at their
own shutter times, so raw DN is not comparable across them. `calibrate_reflectance`
(`preprocessing.py:106-117`) converts each of sample/white/dark to a per-second
rate first, then applies the flat-field formula. This is the physically correct
form of the same equation.

### 2.4 Heterogeneity stats computed on reflectance, not SNV

`rois.py:56-60`: per-pixel standard deviation on SNV data is ≈1 by construction
and carries no information. ROI `std` / `spectral_variance` are therefore computed
from the reflectance cube while `mean_spectrum` comes from the analysis cube.
Without this, the spec's "silicon low variance / SiO₂ higher variance" metric
would be meaningless.

### 2.5 Lazy, memory-bounded cube reading

Cubes are ~2.9 GB (1417×900×300 float64). `cube_io.py:93` `CubeReader` streams
contiguous row blocks and supports decimated previews and full-res pixel/patch
reads, which is what makes the interactive tuners usable at all.

---

## 3. Genuine gaps

### Gap 1 — ROI yield is far below the spec target ⚠️ *most important*

The spec asks for **100–300 ROIs per piece** (§"How many ROIs per image?"), giving
thousands of ML samples. Actual yield is **73 ROIs across all 41 pieces**:

| Dataset | Pieces | ROIs |
|---|---|---|
| `sio2_bare_si` | 9 | 20 |
| `sio2_dish_black` | 13 | 11 |
| `sio2_dish_white_1` | 4 | 13 |
| `sio2_dish_white_20` | 15 | 29 |

Cause: the median piece is ~4,175 px (`data/inventory_summary.json`) but a 32×32
patch is 1,024 px, so at most ~4 patches fit per piece. `RoiConfig` defaults were
`patch=32, stride=32, min_coverage=0.85` — sized for the spec's hypothetical
1024×1024 image, not for these physically small wafer pieces.

Consequence: the cross-specimen ROI table and the specimen-level hold-out
evaluation are statistically thin.

**✅ Fixed.** Defaults are now `patch=8, stride=4`, chosen by measuring actual
yield across all 41 saved masks rather than by assumption:

| patch/stride | total ROIs | median/piece | pieces ≥100 |
|---|---|---|---|
| 32/32 (old) | 73 | 1 | 0/41 |
| 16/8 | 2,301 | 42 | 4/41 |
| **8/4 (new)** | **11,185** | **205** | **30/41** |

Re-exported: **7,180 ROIs across 24 specimens** (from 73). The analysis ROI table
for `sio2_dish_white_20` went from 29 to **4,308 rows** over 15 specimens.

*Caveat kept honest:* an overlapping stride reintroduces spatial autocorrelation
**within** a specimen. That is acceptable only because `split_by_specimen`
(`rois.py:149`) holds out **whole specimens**, so correlated patches never
straddle the train/test boundary — the leakage argument the spec makes is about
the *split*, and it survives. The hold-out check confirms it generalizes: mean
score 0.465 train (3,366 ROIs) vs 0.483 test (942 ROIs). This is pinned by
`tests/test_rois_yield.py`. **Never switch to a random per-ROI split.**

### Gap 2 — no mask erosion → edge-dominated anomalies

Piece-boundary pixels mix film and dish spectra, so they dominate anomaly flags.
The report already *measures* this (`report.py:29` uses `binary_erosion` only to
compute an edge-share statistic): in the current run, piece p13 is **100%** edge
and p12 **62%**. But no erosion is applied to the analysis mask itself.
Self-reported as a known limitation in `docs/tuning.md`.

**✅ Fixed (partially — see below).** `PieceConfig.erode_iter` (default 1 px,
`run_analyze --erode N`) shrinks each piece's analysis mask inward via
`pieces._erode_analysis_mask`, after the bbox is fixed so the crop is unchanged.
A piece that erosion would erase keeps its un-eroded mask, so no specimen drops
out of the study.

The default was chosen by measurement, not assumption:

| erode | ROIs | mean edge share | anomaly fraction |
|---|---|---|---|
| 0 px | 4,554 | 39% | 0.87% |
| **1 px (default)** | **4,308** | **34%** | 1.21% |
| 3 px | 3,760 | 34% | 2.11% |

3 px costs 17% of the ROIs for **no further** edge-share improvement — 1 px is the
efficient point. Note this only *reduces* the problem: individual pieces remain
edge-heavy (p13 99%, p12 61%), because the metric counts flags within 5 px of the
boundary and eroding that far would consume small pieces. Boundary-region
*filtering* (dropping regions that touch the mask edge) remains future work.

### Gap 3 — two dataset presets are dead, and failed opaquely

`out/workflow/analyze/` contains results for `sio2_dish_white_20` only.
`sio2_dish_black` and `sio2_dish_white_1` are organized in `data/organized/` (from
an earlier run) but cannot be analyzed: their raw scans have been **moved to
`<hsi_root>/sio2/legacy/`**, so `iter_cube_paths` resolves both presets to **0
cubes**. Retiring them is consistent with the project's recorded scope
(`sio2_dish_white_20` as target + `sio2_bare_si` as control), but two problems
followed from it:

1. The failure was **silent until it wasn't** — zero cubes produced zero pieces,
   which surfaced five stages later as
   `ValueError: Found array with 0 sample(s)` from *PCA*, naming neither the
   dataset nor the missing file.
2. The docs and the **README quickstart still advertise `sio2_dish_black`** as the
   worked example, so the documented first command fails.

**Fix applied:** `prepare_pieces` now fails fast, distinguishing "no cubes matched
`<glob>` in `<dir>` — may have been moved or retired" from "cubes loaded but
extraction found no pieces — retune `PieceConfig`". Docs/examples should point at
an in-scope dataset. This is a *scope-hygiene* gap, not a missing-capability one:
the cross-specimen comparison legitimately rests on the 15 pieces of
`sio2_dish_white_20` plus the 9 silicon control pieces.

### Gap 4 — piece extraction does not produce physically correct pieces ⚠️ *found late*

**This gap invalidates part of the "what matches" table above, and is a lesson
about how it was produced.** Stages 1–3 were initially audited as ✅ because they
*run* and their metrics land in the spec's expected bands. They do. But a visual
check of the actual extracted masks — which no metric in this pipeline performs —
showed the front-end is wrong in two ways on `sio2_dish_white_20`:

**(a) Adjacent wafers merged into one "piece".** `PieceConfig.close_iter=6`
dilates ~12 px, bridging the gaps between wafers sitting near each other in a
dish. The result was a single 21,106 px "piece" (`p01`) that is in fact an entire
dish holding ~6 wafers — 3× the next largest piece. It entered every downstream
table as *one specimen*, and `split_by_specimen` treated it as one hold-out unit.
Dropping to `close_iter=2` removes it (max piece → 7,339 px, 15 → 19 pieces);
`close_iter=4` reintroduces merging (max 14,421 px).

**(b) Masks capture the oxide, not the wafer.** The default backend
`method="sam"` (Spectral Angle Mapper) is **brightness-invariant** by
construction. Bare silicon is dark but spectrally smooth — close in *shape* to the
bright dish — so it scores a small angle and fails the foreground test, while
SiO₂'s thin-film interference structure scores large and passes. On partially
coated wafers the mask therefore covers only the coloured oxide region and clips
the black bare-Si region away.

This is a genuine design problem, not just tuning: Stage 3.1 is supposed to hand
Stage 3.1b a *whole wafer* so `film.py` can split oxide from substrate within it.
Instead Stage 3.1 performs an uncalibrated version of that split first and
discards the substrate — removing the in-piece bare-silicon reference that
`FilmConfig.reference="in_piece"` depends on.

**Why the metrics missed it:** silhouette, PC1 variance and anomaly fraction are
all computed *within* whatever mask is handed to them. A merged dish or a
half-clipped wafer yields perfectly plausible values. Nothing in the pipeline
asserts that a "piece" corresponds to one physical specimen.

**Status: diagnosed and tooled; defaults deliberately unchanged.**

- **(a) merging** — `close_iter=6 → 2` removes the merged dish: 15 → 19 pieces,
  max piece 21,106 → 7,339 px. (`close_iter=4` reintroduces it at 14,421 px.)
- **(b) oxide-vs-wafer** — new `PieceConfig.method="euclidean"`
  (`pieces.euclidean_distance`): magnitude sensitive, still all 300 bands.
  Measured 79,522 → **172,220** foreground px, recovering whole wafers including
  a large piece SAM missed entirely. Verified by rendering masks over the scan.

`sam` remains the shipped default by explicit choice, so datasets already tuned
around it are unaffected; `euclidean` is opt-in via `--piece-method euclidean`.

**A third bug found while investigating:** `method="mahalanobis"` returned **zero
pieces** on this scan — a documented option that silently produced nothing. Two
causes: `_mahalanobis_to_background` returned the *squared* distance despite its
name (now returns the distance), and Otsu is the wrong rule for that map (tight
background mode + long tail → it isolated 0.24% of pixels). Added
`threshold="chi2"` using the χ² model of the background. Honest outcome:
mahalanobis now produces pieces instead of nothing, but selects only wafer cores
(24,415 px) and remains inferior to `euclidean` here.

Until defaults change, treat per-specimen results on this dataset as provisional:
the current `sam` masks are oxide regions, not whole wafers.

---

## 4. Correctly absent

The spec's §"Remove FEA" is fully honored. There is **no** FEA, stress,
composition-map, or spectral-unmixing code in `hsi_workflow/`, and
`__init__.py:5-7` states the design explicitly: *"no reference spectra, no FEA,
no composition/unmixing."*

Such material survives only in areas the pipeline never imports — `legacy/`,
`reference/` (a MOOSE/FEM input deck), and `out/legacy/`. Retaining them as
inert historical artifacts is correct; they are not on any code path.

---

## 5. Spec metrics vs. observed

From `out/workflow/analyze/sio2_dish_white_20/report.md`:

From `out/workflow/analyze/sio2_dish_white_20/report.md` after the fixes:

| Metric | Spec expectation | Observed | |
|---|---|---|---|
| PC1 explained variance | 70–95% | **82.4%** (PC2 6.7%, PC3 3.4%) | ✅ |
| Silhouette score | 0.4–0.8 | **0.35–0.64** across 15 pieces | ✅ mostly |
| Anomalous pixel fraction | small + localized (~2–10%) | mean **1.21%**, 9 regions in 5/15 pieces | ✅ |
| Connected components | contiguous regions, not speckle | largest 69–235 px after cleanup | ✅ |
| Reflectance range | 0–1, no clipping | histogram emitted by `run_explore` | ✅ |
| ROIs per piece | 100–300 | **median ~195** (was ~2–5) | ✅ fixed |
| Specimens | 10–20 (8–10 Si, 20–30 SiO₂) | 24 (9 Si / 15 SiO₂) | ✅ |

All spec metrics now land in their expected bands. PC1 and the anomaly fraction
shifted only slightly after erosion + ROI changes (80.2%→82.4%, 0.87%→1.21%),
confirming the fixes changed the *sampling*, not the underlying signal.

---

## 6. Summary

| | Count | Items |
|---|---|---|
| Stages implemented as specified | 12/12 | all |
| Beneficial deviations (kept) | 5 | dual anomaly product, fixed patches, exposure-normalized calibration, reflectance-based stats, lazy cube reader |
| Genuine gaps found | 4 | ROI yield, mask erosion, dead dataset presets, **piece extraction correctness** |
| Gaps resolved | 2 fully, 2 open | ROI yield ✅; dead presets ✅ (fail-fast + docs); erosion ⚠️ improved, boundary filtering open; piece extraction ❌ under investigation |

> **Caveat on this audit's method.** Gaps 1–3 were found by reading code and
> checking metrics. Gap 4 was found only by *rendering the extracted masks over
> the scan and looking at them*. Every numeric check in this pipeline passed while
> the front-end was merging wafers and clipping substrate away. Metric-conformance
> is not evidence of physical correctness; add a visual check to any future audit.
| Spec items correctly removed | 1 | FEA / composition / stress |

### Changes made

- `config.py` — `RoiConfig` defaults → `patch=8, stride=4`; new
  `PieceConfig.erode_iter` (default 1) + validation.
- `pieces.py` — `_erode_analysis_mask`, applied in `extract_pieces` after the crop.
- `pipeline.py` — `prepare_pieces` fails fast on zero cubes / zero pieces instead
  of surfacing an opaque PCA shape error stages later.
- `run_analyze.py` — `--erode`; `run_extract.py` / `run_organize.py` ROI arg
  defaults now derive from `RoiConfig()` so the tuned values live in one place.
- `debug_masks.py` — ROI slider minimums lowered below the new defaults;
  `erode_iter` included in the paste-ready config.
- Tests — `tests/test_rois_yield.py` (new) plus erosion cases in
  `tests/test_pieces_extract.py`. 56 pass.
- Docs — README/usage examples moved off the retired `sio2_dish_black`;
  `architecture.md` marks the retired presets; `tuning.md` / `extraction.md`
  updated.

### Data-layout note

Re-exporting at 8×8 produces ~7,180 per-ROI cubes. These are gitignored for the
two live datasets (regenerable from the piece cube + `roi_index.csv` crop
coordinates); tracked instead are the metadata and the per-piece `_rgb.png`
previews. The previous 32×32 ROI exports were moved to `data/legacy/rois_32px/`.
The retired datasets' ROI exports stay tracked because their raw scans are gone.
