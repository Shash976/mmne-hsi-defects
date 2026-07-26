# Interactive debug tuners — how to dial in extraction & preprocessing

Three standalone matplotlib apps let you tune the pipeline's knobs *visually* on a
real scan, then print a paste-ready config. They are the fastest way to fix messy
extraction on a new dataset.

| Tool | Tunes | Prints |
|---|---|---|
| `debug_masks.py` | piece extraction (which pixels are a wafer vs the dish) + ROI grid | `PieceConfig` / `RoiConfig` |
| `debug_film.py` | SiO2-vs-bare-silicon split **within** one wafer | `FilmConfig` |
| `debug_preprocess.py` | calibration + Savitzky-Golay + SNV + baseline | `PreprocessConfig` |

Run everything in the `hsi` conda env:

```
conda run -n hsi python debug_masks.py --dataset sio2_dish_white_20
```

All three accept `--demo` (a synthetic cube, no data needed — good for a first
look), `--crop R0 R1 C0 C1` (work on a full-resolution window), and `--max-dim N`
(decimate the whole scan to N px on its long axis for responsiveness; default 700).

## Why it now feels fast

- **Debounced.** Dragging a slider recomputes **once, on release** — not on every
  intermediate tick. Band-stepping and contrast are instant (they only touch the
  image, never rerun the pipeline).
- **Downsampled compute.** The heavy spectral distance runs on the `--max-dim`
  grid over the *whole* scan, so the full dish stays interactive. Zoom into
  full resolution on one piece with `--crop` when you want to check fine detail.
- **Cached.** The distance map is computed once per (method, background, flat-field,
  calibrate) combination; thresholding/morphology/labeling rerun instantly.

Because you tune on the decimated grid, `debug_masks.py`'s `'p'` printout **rescales
the spatial values** (`min_area`, ROI `patch`/`stride`, `background_bbox`) back to
full resolution, so the printed config is correct for a full-res `run_extract`.

---

## `debug_masks.py` — piece extraction

Three panels: **band image + red mask overlay** · **foreground distance map** ·
**labeled pieces + white ROI grid**.

### The extraction workflow (recommended order)

1. **Look at the distance map (panel 2).** Foreground pixels (wafers) should be
   bright, background (dish) dark. If the whole frame looks similar, the background
   reference is bad — go to step 2.
2. **Fix the background reference — the biggest lever on the white-dish scan.**
   The auto reference is the outer **border frame**; on the two-dish scan that
   frame is contaminated (rim, paper, the second dish). Press **`R`**, then drag a
   box over a **clean empty-dish patch**. The distance is now measured against
   *those* pixels (this becomes `PieceConfig.background_bbox`). Press **`c`** to
   revert to the border frame.
3. **Even out the lighting if needed.** Tick **flat field** (divides out the smooth
   brightness gradient; tune `flat sigma`) and/or **calibrate** (work on
   white/dark reflectance instead of raw counts). SAM is scale-invariant so a pure
   brightness gradient barely moves it — these mainly help the Otsu cutoff, the
   Mahalanobis method, and spectrally-*colored* gradients.
4. **Set the mask window.** Use the **otsu**/**percentile** radio to snap a sensible
   cutoff, then fine-tune the **mask window** RangeSlider (`lo ≤ distance ≤ hi`). Aim
   for the red overlay to cover the wafers and nothing else.
5. **Clean the mask.** `open iter` removes thin rim arcs/dust; `close iter` merges
   within-wafer gaps so a patterned chip doesn't fragment; `min area` drops
   specks. Tick **watershed** to split wafers that touch.
6. **Check the pieces (panel 3).** Each wafer should be one solid color. You want
   ~15–20 pieces on the 20-dish scan, not 200 (over-segmented) or 1 (merged).
7. **Set the ROI grid.** `ROI patch` / `ROI stride` / `min coverage` — the white
   squares are the kept ROIs. Aim for ~100–300 per piece.
8. **Press `p`.** Paste the printed `PieceConfig(...)` / `RoiConfig(...)` into
   `hsi_workflow/config.py` (or a `WorkflowConfig` override), then run the real
   extraction full-res:
   ```
   conda run -n hsi python -m hsi_workflow.run_extract --dataset sio2_dish_white_20
   ```
9. **Spot-check full-res.** `--crop` into one wafer to confirm the morphology looks
   right at native resolution.

### Controls
- **radios:** method `sam | mahalanobis | kmeans`; threshold `otsu | percentile`
- **checks:** flat field · calibrate · watershed
- **sliders:** band · mask window · band contrast · dist contrast · open iter ·
  close iter · min area · ROI patch · ROI stride · min coverage · flat sigma
- **keys:** `←/→` step band · `m` toggle overlay · `R` draw reference box ·
  `c` clear reference box · `p` print configs

### Contrast sliders
`band contrast` / `dist contrast` only change the display color limits (light,
never recompute) — use them to *see* faint structure; they never affect the mask.

---

## `debug_film.py` — SiO2 within a wafer

After piece extraction, a wafer is still part bare silicon and part SiO2. This tool
isolates the oxide. Panels: **band image + red SiO2 overlay** · **film distance map
(distance from bare silicon)** · **SiO2 (red) vs bare Si (blue)**.

1. **Pick the wafer.** `--piece 0` is the largest; step through with `--piece N`.
2. **Choose the reference (radio).**
   - `control` — distance from the mean bare-silicon spectrum of the `sio2_bare_si`
     dataset (loaded and calibrated automatically). Good global reference.
   - `in_piece` — split this wafer into two clusters; the bare cluster is the one
     closest to the control. Robust to per-wafer lighting.
3. **Set the window / threshold** exactly like `debug_masks` (otsu snap, then the
   mask window). Oxide = the far-from-bare pixels.
4. **Clean** with `open`/`close`/`min area`.
5. **`p`** prints `FilmConfig(enabled=True, ...)`. Enable it in the pipeline with
   `run_analyze --extract-film` (see below) so anomaly detection runs on oxide only.

Both the wafer and the control are calibrated to reflectance first, so lighting
differences between the two scans cancel.

---

## `debug_preprocess.py` — calibration & filtering

Panels: **band image** · **clicked-pixel spectrum, before vs after** · **live noise
metrics**.

1. **Click a pixel** to inspect its spectrum. The gray line is the "before"
   (reflectance), the red line the "after" (current settings).
2. **Toggle** calibrate / SG smooth / SNV / baseline (checkboxes) and watch the
   spectrum + the RMS-noise / SNR readout update.
3. **Tune** `SG window` / `SG polyorder` (smoothing) and `baseline order`. Watch the
   "% outside [0,1]" reflectance sanity check.
4. **Reference subtract (debug-only).** `shift+click` a pixel to set a reference
   spectrum (5×5 average), then tick **subtract ref** to see it removed before
   smoothing; `c` clears it. This is an exploration aid — it is *not* part of
   `PreprocessConfig`.
5. **`p`** prints a paste-ready `PreprocessConfig(...)`.

`band contrast` (RangeSlider) is display-only.

---

## From tuner to pipeline

```
# 1. tune piece extraction, paste PieceConfig/RoiConfig into config.py, then:
conda run -n hsi python -m hsi_workflow.run_organize --datasets sio2_bare_si sio2_dish_white_20

# 2. (optional) tune film, paste FilmConfig, then analyze oxide-only:
conda run -n hsi python -m hsi_workflow.run_analyze --target sio2_dish_white_20 \
    --baseline sio2_bare_si --extract-film --film-reference control
```

See [tuning.md](tuning.md) for the knob-by-symptom table and [extraction.md](extraction.md)
for how the stages fit together.
