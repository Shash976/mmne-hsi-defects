# unsupervised_defect.py
"""
Unsupervised defect detection for LIG hyperspectral scans.

Assumes all samples share the same fabrication configuration, so pixel
spectra pooled across the wide-FOV (ROI-1) scans define a single "normal"
background distribution. Each pixel -- in both the ROI-1 and the tighter
ROI-2 scans -- is then scored against that pooled distribution with a
regularized RX (Mahalanobis) detector. No labels required.

Data: Resonon Pika L .bip/.bip.hdr pairs, one ROI-1 (wide) + one ROI-2
(zoomed) file per sample, named like "C_E01-ROI-1.bip[.hdr]".

Usage:
    python unsupervised_defect.py
    python unsupervised_defect.py --data-dir "path\to\roi_scans" --out out/unsupervised
"""

import argparse
import os
import re
import glob

import numpy as np
import matplotlib.pyplot as plt
import spectral
from sklearn.covariance import LedoitWolf
from sklearn.cluster import KMeans
from scipy.ndimage import uniform_filter

DEFAULT_DATA_DIR = r"C:\Users\shash\OneDrive - purdue.edu\Summer\hsi\lig_dataset\roi_scans"
DEFAULT_OUT_DIR = "out/unsupervised"
DEFAULT_WHITE_REF = (r"C:\Users\shash\OneDrive - purdue.edu\Summer\hsi\lig_dataset"
                      r"\calibration_whitedark\white_ref.bil.hdr")
DEFAULT_DARK_REF = (r"C:\Users\shash\OneDrive - purdue.edu\Summer\hsi\lig_dataset"
                     r"\calibration_whitedark\dark_correction.bil.hdr")

PAIR_RE = re.compile(r"^(?P<sample>.+)-[Rr][Oo][Ii]-(?P<roi>\d+)\.bip\.hdr$")


# --------------------------------------------------------------------------
# Data discovery / loading
# --------------------------------------------------------------------------

def find_pairs(data_dir):
    """Return {sample_id: {roi_num: hdr_path}}, e.g. {'C_E01': {1: '...ROI-1.bip.hdr', 2: '...ROI-2.bip.hdr'}}."""
    pairs = {}
    for hdr in glob.glob(os.path.join(data_dir, "*.bip.hdr")):
        m = PAIR_RE.match(os.path.basename(hdr))
        if not m:
            continue
        sample_id = m.group("sample")
        roi_num = int(m.group("roi"))
        pairs.setdefault(sample_id, {})[roi_num] = hdr
    return dict(sorted(pairs.items()))


# --------------------------------------------------------------------------
# Radiometric calibration (white/dark reference, exposure-normalized)
# --------------------------------------------------------------------------

def load_cube_with_shutter(hdr_path):
    """Like hsi_explore.load_cube, but also returns the shutter (exposure) time
    and sensor ceiling from the ENVI header -- needed because the white/dark
    references here were captured at different exposures than the sample
    scans, and a few scans have saturated pixels that must be masked out
    before calibration (dividing by a near-zero white-dark denominator at a
    saturated pixel produces huge/negative reflectance values that dominate
    the pooled KMeans segmentation).
    """
    img = spectral.open_image(hdr_path)
    cube = np.asarray(img.load(), dtype=np.float64)
    wavelengths = img.bands.centers if img.bands is not None else None
    shutter = float(img.metadata.get("shutter", 1.0))
    ceiling = float(img.metadata.get("ceiling", np.inf))
    return cube, wavelengths, shutter, ceiling


def saturated_pixel_mask(cube, ceiling):
    """True where a pixel hits the sensor ceiling in any band (unreliable spectrum)."""
    return (cube.max(axis=-1) >= ceiling - 1)


def load_reference_spectrum(hdr_path):
    """Whole-frame mean spectrum + shutter time for a white/dark reference cube."""
    cube, _, shutter, _ = load_cube_with_shutter(hdr_path)
    mean_spectrum = cube.reshape(-1, cube.shape[-1]).mean(axis=0)
    return mean_spectrum, shutter


def calibrate_to_reflectance(cube, shutter_sample, white_mean, shutter_white, dark_mean, shutter_dark):
    """Exposure-normalized (raw-dark)/(white-dark) reflectance.

    White/dark refs were captured at different shutter times than the sample
    scans, so raw DN isn't directly comparable across them -- divide by
    shutter time first (assumes linear sensor response) to get a rate, then
    apply the standard flat-field formula on those rates.
    """
    sample_rate = cube / shutter_sample
    white_rate = white_mean / shutter_white
    dark_rate = dark_mean / shutter_dark
    denom = white_rate - dark_rate
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    return (sample_rate - dark_rate) / denom


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

def snv(flat_spectra):
    """Standard Normal Variate: per-pixel mean/std normalization.

    Removes multiplicative/additive illumination differences between scans
    that were never calibrated to a white/dark reference.
    """
    mean = flat_spectra.mean(axis=1, keepdims=True)
    std = flat_spectra.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (flat_spectra - mean) / std


def preprocess(cube, method="snv"):
    flat = cube.reshape(-1, cube.shape[-1]).astype(np.float64)
    if method == "snv":
        flat = snv(flat)
    elif method == "none":
        pass
    else:
        raise ValueError(f"Unknown preprocess method: {method}")
    return flat.reshape(cube.shape)


# --------------------------------------------------------------------------
# Substrate / LIG segmentation
# --------------------------------------------------------------------------

def fit_segmenter(pooled_spectra, seed=0):
    """2-cluster KMeans over pooled spectra (all ROIs, all samples) to split
    substrate vs. LIG material. The smaller-area cluster is assumed to be the
    LIG pattern (traces/pads occupy less area than the surrounding substrate
    in these scans) -- verify against segmentation_preview.png and pass
    --invert-foreground if the assumption is backwards for your layout.
    """
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(pooled_spectra)
    counts = np.bincount(km.labels_, minlength=2)
    foreground_label = int(np.argmin(counts))
    return km, foreground_label


def foreground_mask(cube_flat, shape, km, foreground_label):
    labels = km.predict(cube_flat)
    return (labels == foreground_label).reshape(shape)


def save_segmentation_preview(sample_id, roi_num, cube, wavelengths, mask, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rgb = pseudo_rgb(cube, wavelengths)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(rgb)
    axes[0].set_title(f"{sample_id} ROI-{roi_num}: pseudo-RGB")
    axes[0].axis("off")
    axes[1].imshow(rgb)
    axes[1].imshow(np.ma.masked_where(~mask, mask), cmap="cool", alpha=0.5)
    axes[1].set_title("LIG foreground (segmented)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{sample_id}_roi{roi_num}_segmentation.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Detrending (removes illumination gradients / edge effects)
# --------------------------------------------------------------------------

def detrend_plane_masked(cube, mask):
    """Fit and remove a per-band linear (row, col) plane using only foreground
    pixels. Cancels slow illumination gradients / vignetting regardless of
    image size -- a local box-filter mean can't fully remove a gradient when
    the window is a large fraction of a small (e.g. zoomed ROI-2) image.
    """
    rows, cols, bands = cube.shape
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    ys, xs = yy[mask], xx[mask]
    design_fg = np.column_stack([np.ones_like(xs), xs, ys])
    design_full = np.column_stack([np.ones(rows * cols), xx.ravel(), yy.ravel()])

    flat = cube.reshape(-1, bands)
    coeffs, _, _, _ = np.linalg.lstsq(design_fg, flat.reshape(rows, cols, bands)[mask], rcond=None)
    trend = (design_full @ coeffs).reshape(rows, cols, bands)
    return cube - trend


def local_stats_masked(cube, mask, window):
    """Per-band local mean AND std using only foreground (mask=True) neighbors.

    Normalizing by local std (not just subtracting local mean) is what
    actunally cancels row/time-dependent noise-level drift (e.g. a pushbroom
    scan-direction artifact where some rows are just noisier, not offset) --
    a mean-only detrend leaves that noise untouched since it never shifts
    the mean.
    """
    mask_f = mask.astype(np.float64)
    weights = uniform_filter(mask_f, size=window, mode="nearest")
    bands = cube.shape[-1]
    mean_num = np.empty_like(cube)
    sq_num = np.empty_like(cube)
    for b in range(bands):
        band_vals = cube[:, :, b] * mask_f
        mean_num[:, :, b] = uniform_filter(band_vals, size=window, mode="nearest")
        sq_num[:, :, b] = uniform_filter(band_vals * cube[:, :, b], size=window, mode="nearest")
    denom = np.where(weights < 1e-6, 1.0, weights)[:, :, None]
    local_mean = mean_num / denom
    local_var = np.clip(sq_num / denom - local_mean ** 2, 1e-8, None)
    local_std = np.sqrt(local_var)
    valid = weights >= (5.0 / (window * window))  # need at least ~5 foreground neighbors
    return local_mean, local_std, valid & mask


# --------------------------------------------------------------------------
# RX (Mahalanobis) anomaly detector
# --------------------------------------------------------------------------

def fit_rx_detector(pooled_residuals):
    """Fit Ledoit-Wolf shrinkage covariance on pooled, already-detrended background residuals."""
    cov = LedoitWolf().fit(pooled_residuals)
    return cov.precision_


def rx_scores(residuals, precision):
    """Mahalanobis distance of every (already-centered) residual spectrum."""
    return np.einsum("ij,jk,ik->i", residuals, precision, residuals)


def pool_background(sample_arrays, max_per_image=4000, seed=0):
    rng = np.random.default_rng(seed)
    pooled = []
    for arr in sample_arrays:
        if arr.shape[0] > max_per_image:
            idx = rng.choice(arr.shape[0], max_per_image, replace=False)
            arr = arr[idx]
        pooled.append(arr)
    return np.vstack(pooled)


# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------

def pseudo_rgb(cube, wavelengths, targets=(650, 550, 450)):
    idx = [int(np.argmin(np.abs(np.asarray(wavelengths) - t))) for t in targets]
    rgb = cube[:, :, idx].astype(np.float64)
    rgb = (rgb - rgb.min()) / (np.ptp(rgb) + 1e-12)
    return rgb


def save_sample_plots(sample_id, roi_num, cube, wavelengths, score_map, valid_mask, threshold, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rgb = pseudo_rgb(cube, wavelengths)
    flagged = valid_mask & (score_map > threshold)

    score_display = np.ma.masked_where(~valid_mask, score_map)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(rgb)
    axes[0].set_title(f"{sample_id} ROI-{roi_num}: pseudo-RGB")
    axes[0].axis("off")

    im = axes[1].imshow(score_display, cmap="inferno")
    axes[1].set_title("Local RX score (LIG pixels only)")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    axes[2].imshow(rgb)
    axes[2].imshow(np.ma.masked_where(~flagged, flagged), cmap="autumn", alpha=0.6)
    axes[2].set_title(f"Flagged pixels (> {threshold:.1f})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{sample_id}_roi{roi_num}_anomaly.png"), dpi=150)
    plt.close(fig)


def save_score_histogram(all_scores_roi1, all_scores_roi2, threshold, out_dir):
    plt.figure()
    plt.hist(np.concatenate(all_scores_roi1), bins=100, alpha=0.6, density=True, label="ROI-1 (wide)")
    plt.hist(np.concatenate(all_scores_roi2), bins=100, alpha=0.6, density=True, label="ROI-2 (zoomed)")
    plt.axvline(threshold, color="k", linestyle="--", label=f"threshold ({threshold:.1f})")
    plt.xlabel("RX anomaly score (Mahalanobis distance)")
    plt.ylabel("Density")
    plt.title("Anomaly score distribution: wide vs. zoomed scans")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "score_histogram.png"), dpi=150)
    plt.close()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unsupervised local-RX defect detection for LIG HSI scans.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--preprocess", type=str, default="none", choices=["snv", "none"])
    parser.add_argument("--white-ref", type=str, default=DEFAULT_WHITE_REF)
    parser.add_argument("--dark-ref", type=str, default=DEFAULT_DARK_REF)
    parser.add_argument("--no-calibrate", action="store_true",
                         help="Skip white/dark reflectance calibration and use raw DN counts.")
    parser.add_argument("--percentile", type=float, default=97.5,
                         help="Percentile of pooled ROI-1 LIG-residual scores used as the flagging threshold.")
    parser.add_argument("--max-per-image", type=int, default=4000,
                         help="Max pixels sampled per ROI-1 image when pooling the background model.")
    parser.add_argument("--window", type=int, default=7,
                         help="Local neighborhood window size (pixels) for local-mean detrending.")
    parser.add_argument("--invert-foreground", action="store_true",
                         help="Flip which KMeans cluster is treated as the LIG foreground, if the "
                              "default (smaller-area cluster = LIG) is wrong for your layout.")
    args = parser.parse_args()

    pairs = find_pairs(args.data_dir)
    missing = {s: rois for s, rois in pairs.items() if 1 not in rois or 2 not in rois}
    if missing:
        print(f"Warning: incomplete ROI-1/ROI-2 pair for samples: {list(missing)}")
    samples = [s for s in pairs if 1 in pairs[s] and 2 in pairs[s]]
    print(f"Found {len(samples)} complete sample pairs: {samples}")

    white_mean = dark_mean = shutter_white = shutter_dark = None
    if not args.no_calibrate:
        white_mean, shutter_white = load_reference_spectrum(args.white_ref)
        dark_mean, shutter_dark = load_reference_spectrum(args.dark_ref)
        print(f"Loaded white ref (shutter={shutter_white:.2f}) and dark ref (shutter={shutter_dark:.2f})")

    cubes = {}      # sample_id -> {1: (cube, wl), 2: (cube, wl)}
    sat_masks = {}  # sample_id -> {1: saturated_mask, 2: saturated_mask}
    n_saturated_total = 0
    for sample_id in samples:
        cubes[sample_id] = {}
        sat_masks[sample_id] = {}
        for roi_num in (1, 2):
            raw_cube, wl, shutter_sample, ceiling = load_cube_with_shutter(pairs[sample_id][roi_num])
            sat_mask = saturated_pixel_mask(raw_cube, ceiling)
            n_saturated_total += int(sat_mask.sum())
            sat_masks[sample_id][roi_num] = sat_mask

            cube = raw_cube
            if not args.no_calibrate:
                cube = calibrate_to_reflectance(cube, shutter_sample, white_mean, shutter_white,
                                                 dark_mean, shutter_dark)
            cube = preprocess(cube, args.preprocess)
            cubes[sample_id][roi_num] = (cube, wl)
    print(f"Masked {n_saturated_total} saturated pixels (sensor ceiling) across all scans")

    # --- Segment substrate vs. LIG, using one KMeans fit shared across all scans (saturated pixels excluded). ---
    all_flats = [cubes[s][r][0].reshape(-1, cubes[s][r][0].shape[-1])[~sat_masks[s][r].ravel()]
                 for s in samples for r in (1, 2)]
    seg_pool = pool_background(all_flats, max_per_image=args.max_per_image)
    km, fg_label = fit_segmenter(seg_pool)
    if args.invert_foreground:
        fg_label = 1 - fg_label
    print(f"Segmentation: cluster {fg_label} treated as LIG foreground "
          f"(cluster sizes in pooled fit: {np.bincount(km.labels_, minlength=2).tolist()})")

    masks = {}  # sample_id -> {1: mask, 2: mask}
    for sample_id in samples:
        masks[sample_id] = {}
        for roi_num in (1, 2):
            cube, wl = cubes[sample_id][roi_num]
            flat = cube.reshape(-1, cube.shape[-1])
            mask = foreground_mask(flat, cube.shape[:2], km, fg_label) & ~sat_masks[sample_id][roi_num]
            masks[sample_id][roi_num] = mask
            save_segmentation_preview(sample_id, roi_num, cube, wl, mask,
                                       os.path.join(args.out, sample_id))

    # --- Detrend every scan within its LIG mask: plane fit (gradients) then local mean (edge mixing). ---
    residual_cubes = {}  # sample_id -> {roi_num: (residuals_flat, valid_mask)}
    for sample_id in samples:
        residual_cubes[sample_id] = {}
        for roi_num in (1, 2):
            cube, wl = cubes[sample_id][roi_num]
            mask = masks[sample_id][roi_num]
            plane_detrended = detrend_plane_masked(cube, mask)
            local_mean, local_std, valid = local_stats_masked(plane_detrended, mask, args.window)
            residual = (plane_detrended - local_mean) / local_std
            residual_cubes[sample_id][roi_num] = (residual, valid)

    # --- Fit the shared RX covariance on pooled, detrended ROI-1 residuals (valid LIG pixels only). ---
    roi1_valid_residuals = []
    for sample_id in samples:
        residual, valid = residual_cubes[sample_id][1]
        roi1_valid_residuals.append(residual[valid])
    pooled = pool_background(roi1_valid_residuals, max_per_image=args.max_per_image)
    print(f"Pooled LIG background residuals: {pooled.shape[0]} pixels x {pooled.shape[1]} bands")
    precision = fit_rx_detector(pooled)

    pooled_scores = rx_scores(pooled, precision)
    threshold = np.percentile(pooled_scores, args.percentile)
    print(f"RX threshold ({args.percentile}th percentile of pooled ROI-1 LIG residual scores): {threshold:.2f}")

    all_scores_roi1, all_scores_roi2 = [], []
    summary_rows = []
    for sample_id in samples:
        for roi_num in (1, 2):
            cube, wl = cubes[sample_id][roi_num]
            residual, valid = residual_cubes[sample_id][roi_num]
            flat_residual = residual.reshape(-1, residual.shape[-1])
            scores = rx_scores(flat_residual, precision)
            score_map = scores.reshape(cube.shape[:2])
            valid_map = valid  # already (rows, cols)

            out_dir = os.path.join(args.out, sample_id)
            save_sample_plots(sample_id, roi_num, cube, wl, score_map, valid_map, threshold, out_dir)

            valid_scores = score_map[valid_map]
            flagged_frac = float((valid_scores > threshold).mean()) if valid_scores.size else float("nan")
            mean_score = float(valid_scores.mean()) if valid_scores.size else float("nan")
            max_score = float(valid_scores.max()) if valid_scores.size else float("nan")
            summary_rows.append((sample_id, roi_num, flagged_frac, mean_score, max_score, int(valid_map.sum())))
            (all_scores_roi1 if roi_num == 1 else all_scores_roi2).append(valid_scores)

    save_score_histogram(all_scores_roi1, all_scores_roi2, threshold, args.out)

    print("\n=== Summary (sample, ROI, flagged_frac, mean_score, max_score, n_LIG_pixels) ===")
    for row in summary_rows:
        print(f"{row[0]:>8}  ROI-{row[1]}  flagged={row[2]:.2%}  mean={row[3]:.1f}  max={row[4]:.1f}  n={row[5]}")

    roi1_means = np.array([r[3] for r in summary_rows if r[1] == 1])
    roi2_means = np.array([r[3] for r in summary_rows if r[1] == 2])
    print(f"\nMean RX score -- ROI-1 (wide): {roi1_means.mean():.1f}   ROI-2 (zoomed): {roi2_means.mean():.1f}")
    if roi2_means.mean() > roi1_means.mean():
        print("Zoomed scans score higher on average -- consistent with them targeting anomalous regions.")
    else:
        print("Zoomed scans do NOT score higher on average -- worth checking whether ROI-2 targets defects "
              "or just a region of general interest, and whether the threshold/preprocessing needs tuning.")


if __name__ == "__main__":
    main()
