# LIG Hyperspectral Analysis Pipeline

What each script does, why it's built that way, and what's still open. Written
for future-you (or anyone else picking this up), not as a code walkthrough --
the "why" is the point, since the code itself explains the "what".

## Data

Resonon Pika L VNIR pushbroom scans (367.8-1007.9 nm, 300 bands), `.bip`/`.bip.hdr`
ENVI pairs, in `C:\Users\...\lig_dataset\roi_scans`. 8 samples (`C_E01`-`C_E08`,
same fabrication config), each with two scans:

- **ROI-1**: wide field of view, whole device pattern (~54x58 px).
- **ROI-2**: tighter zoom on a specific region of interest (~21x23 px).

Radiometric calibration references live in `lig_dataset\calibration_whitedark`
(`white_ref.bil.hdr`, `dark_correction.bil.hdr`). These were captured at
*different* shutter/exposure times than the sample scans (and from each
other), so calibration exposure-normalizes (divides by shutter time) before
applying the standard `(raw-dark)/(white-dark)` reflectance formula -- see
`calibrate_to_reflectance` in `unsupervised_defect.py`.

`lig_dataset\roi_mean_spectra` contains one `.spec` file per existing ROI scan
-- but each is just that scan's own whole-image mean spectrum, not an
independent reference for a *different* material (pristine LIG, substrate,
defect, etc). It's redundant with the cube itself, so it does **not** count as
a usable endmember library for spectral unmixing (see below).

## Two complementary analyses

Both scripts share the same calibration/segmentation groundwork
(`unsupervised_defect.py` defines it; `composition_pipeline.py` imports it)
but ask a different question:

| | `unsupervised_defect.py` | `composition_pipeline.py` |
|---|---|---|
| Question | "Is this pixel anomalous?" | "What material class is this pixel?" |
| Method | RX (Mahalanobis) anomaly score | PCA + K-means classification |
| Output | Continuous anomaly score + threshold | Discrete class labels + per-class deviation |
| Best for | Flagging rare, spatially localized outliers | Understanding composition, spotting large/systematic regions |

Run them both and cross-check: a real defect should show up as an outlier in
the RX map *and* as a minority-class blob in the composition map. If it only
shows up in one, be suspicious of it.

## Shared groundwork (`unsupervised_defect.py`)

1. **Calibration** (`calibrate_to_reflectance`): exposure-normalized
   reflectance from the white/dark references, as above.
2. **Saturation masking** (`saturated_pixel_mask`): a few scans (C_E07-ROI-1,
   C_E08-ROI-1) have pixels at the sensor ceiling (8190 DN). Dividing by a
   near-zero calibration denominator at a saturated pixel produces
   reflectance values of 20-30+ (even negative), which then dominates
   Euclidean-distance-based clustering. These pixels are masked out before
   anything else touches them.
3. **Substrate/LIG segmentation** (`fit_segmenter`, `foreground_mask`): 2-cluster
   KMeans over pooled spectra from all scans, splitting the LIG pattern
   (traces/pads) from the surrounding substrate. Which cluster is "LIG" is
   decided by a smaller-area heuristic that works for ROI-1 but is *backwards*
   for ROI-2 (those crops are zoomed almost entirely inside the LIG pattern,
   so "smaller cluster" picks the wrong one there) -- hence
   `--invert-foreground` is required for this dataset. Always sanity-check
   against the `*_segmentation.png` previews before trusting a run.
4. **Detrending** (`detrend_plane_masked`, `local_stats_masked`): removes two
   confounds that would otherwise get mistaken for defects/composition classes:
   - A per-band linear (row, col) **plane fit**, removing slow illumination
     gradients across a scan.
   - **Local mean + std normalization** over a small window, removing
     row/time-dependent noise drift (a pushbroom scan-direction artifact) and
     any residual substrate/LIG edge mixing the plane fit missed. Normalizing
     by local *std*, not just mean, matters -- a noisier-but-not-offset region
     survives mean-only detrending untouched.
   
   Both matter in practice: this dataset's imaging setup has a **known,
   confirmed lighting asymmetry** (uneven illumination across the stage) that
   showed up as a large, suspiciously identical-looking "second class" across
   independent samples until both detrending steps were applied together (a
   plane fit alone left a residual split in some samples, because the real
   asymmetry isn't perfectly linear).

## `unsupervised_defect.py`: RX anomaly detection

Pools LIG-pixel residuals (after all of the above) across the 8 ROI-1 scans
to define a single shared "normal" distribution (Ledoit-Wolf regularized
covariance, since 300 bands vs. a few thousand pooled pixels needs
shrinkage). Every pixel -- in both ROI-1 and ROI-2 -- gets a Mahalanobis
distance from that distribution; a percentile threshold (default 97.5th)
flags outliers.

**Validation signal**: ROI-2 (zoomed toward whatever the person doing the scan
thought was worth a closer look) scores ~2.7x higher on average than ROI-1,
and the score histograms show ROI-2 has a long tail past the threshold that
ROI-1 doesn't -- consistent with ROI-2 actually containing more anomalies,
not just an artifact inflating everything.

Outputs: `out/unsupervised/<sample>/*_anomaly.png` (RGB / score map / flagged
overlay), `*_segmentation.png`, `score_histogram.png`.

## `composition_pipeline.py`: composition / defect classification

```
reference spectra available? --yes--> Linear Spectral Unmixing (implemented, unused -- see below)
                             --no---> PCA -> K-means (k chosen by silhouette score, k=2..6)
  -> classify every LIG pixel (nearest class + deviation-from-centroid score)
  -> composition map (discrete class labels) + defect map (minority classes)
  -> spatial filter (3x3 mode filter, removes salt-and-pepper misclassification)
  -> upsample (nearest for labels, cubic for the continuous deviation map)
  -> quantitative maps + per-sample summary stats
```

**Why PCA+K-means and not Linear Spectral Unmixing**: LSU needs true endmember
spectra for physically distinct materials. `find_usable_endmembers` checks
`roi_mean_spectra` for exactly that, and correctly finds none (every file
there just duplicates a scan's own mean) -- so this always falls back to
PCA+K-means for now. `linear_spectral_unmix` (fully-constrained NNLS,
sum-to-one + non-negativity) is implemented and wired in so that dropping a
real endmember library into that folder (with labels distinct from the
`<sample>-roi-<n>` naming) switches the pipeline over automatically, without
code changes.

**Mask erosion matters here more than for RX**: classification is much more
sensitive to substrate/LIG boundary mixed pixels than the anomaly detector
was, because a consistent "boundary spectrum" is compact enough for K-means
to carve out as its own class (a ring tracing every pattern's outline) rather
than just inflating an anomaly score. The LIG mask is eroded by 2px
(`--erode`) before classification to exclude this; even so, a faint ring can
still survive (partial-volume mixing may extend past 2px) -- treat any
ring-shaped "defect" class in the output with suspicion, it's probably this,
not a real defect.

**A recurring central blob**: in early runs, a small cluster near the center
of the circular pad shows up as a minority class in multiple independent
samples. This needs a domain check, not a code fix -- it could be a real
device feature (a contact via, fiducial mark) rather than a defect. Compare
against the actual device layout before treating it as an anomaly.

Outputs: `out/composition/<sample>/*_composition.png` (RGB / raw & filtered
class maps / deviation map / defect map), `*_upsampled.png`.

## What about `pipeline.py`?

`pipeline.py` sketches a **supervised** classification pipeline: PCA -> PLS-DA
/ Random Forest / SVM, with leave-one-film-out cross-validation, assuming
labeled ROIs (`pristine` / `grain_boundary` / `strained`) and a `film_id` for
grouping. It doesn't run as-is (`read_roi` is a stub, and we don't have those
labels), but it's not nonsense either -- it's the natural **next stage** once
this unsupervised work produces a validated set of labels:

1. Visually review the composition/defect maps from `composition_pipeline.py`
   and the anomaly maps from `unsupervised_defect.py`.
2. Where the two agree (flagged in both) and it isn't one of the known
   confounds above (boundary ring, central blob, illumination), that's a
   plausible real defect region -- hand-label a handful of ROIs this way.
3. `pipeline.py`'s `build_pipeline` / `evaluate_leave_one_film_out` (grouped
   CV so validation is per-*sample*, not per-pixel -- avoids the classic
   mistake of leaking spatially adjacent pixels from the same sample into
   both train and test) is directly reusable at that point. Its
   `make_synthetic_rois` (linear mixing + noise) is also a reasonable
   augmentation step if labeled ROIs turn out to be scarce.

Don't reuse `pipeline.py` before that -- it needs labels this project doesn't
have yet, and the unsupervised stages above are how those labels get produced.

## What's next, and what's blocking it

Requested downstream stages -- **Finite Element Mesh -> Stress/Strain
Simulation -> Correlation with Device Performance** -- are not started, because
they need inputs this hyperspectral data can't provide on its own:

- **Material mechanical properties** (Young's modulus, Poisson ratio,
  thickness) for the LIG and substrate, to define the FEA material model.
- **A geometry/mesh definition** -- the quantitative/composition maps here
  are a 2D raster of a thin film; going to a mesh needs a decision on how
  film thickness and any out-of-plane structure are represented.
- **Device performance measurements** (e.g. sheet resistance, strain-gauge
  response) per sample, to correlate against the composition/defect maps.

Once those exist, the natural bridge is the per-sample summary statistics
already being computed in `composition_pipeline.py` (defect area fraction,
mean deviation, per-class area fractions) and the upsampled quantitative maps
-- those are the kind of per-sample scalar/spatial inputs a correlation study
or a simplified FEA material-property map would consume.
