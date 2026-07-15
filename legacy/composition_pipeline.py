# composition_pipeline.py
"""
Composition / defect classification pipeline for LIG hyperspectral scans.

Picks up after unsupervised_defect.py's calibration + LIG segmentation, but
attacks the problem from the materials-composition angle instead of pure
anomaly detection:

    reference spectra available? --unmix--> Linear Spectral Unmixing
                                 --else----> PCA -> K-means clustering
    -> Spectral Classification (assign every LIG pixel to a class)
    -> Composition / Defect maps (per-sample class + binary defect map)
    -> Spatial filtering (mode filter, removes salt-and-pepper misclassification)
    -> Interpolation (upsample to a finer grid for visualization / future meshing)
    -> Quantitative maps (per-pixel deviation-from-class score + per-sample stats)

No true multi-material endmember library was found in roi_mean_spectra (see
PIPELINE.md) -- every .spec file there is just a scan's own whole-image mean,
redundant with the cube -- so this run falls back to PCA + K-means. The LSU
path is implemented and will be used automatically if you add a real
endmember library later (see find_usable_endmembers).

Usage:
    python composition_pipeline.py
    python composition_pipeline.py --invert-foreground
"""

import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.optimize import nnls
from scipy.ndimage import zoom, generic_filter, binary_erosion
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from unsupervised_defect import (
    DEFAULT_DATA_DIR, DEFAULT_WHITE_REF, DEFAULT_DARK_REF,
    find_pairs, load_cube_with_shutter, saturated_pixel_mask, load_reference_spectrum,
    calibrate_to_reflectance, fit_segmenter, foreground_mask, pool_background, pseudo_rgb,
    detrend_plane_masked, local_stats_masked,
)

DEFAULT_OUT_DIR = "out/composition"
DEFAULT_REFERENCE_DIR = r"C:\Users\shash\OneDrive - purdue.edu\Summer\hsi\lig_dataset\roi_mean_spectra"


# --------------------------------------------------------------------------
# Reference spectra / endmember detection
# --------------------------------------------------------------------------

def find_usable_endmembers(reference_dir, pairs):
    """Look for a true multi-material endmember library, as opposed to
    per-scan mean spectra that just duplicate each cube's own average.

    A .spec file counts as a *reference* only if it does NOT correspond 1:1
    to one of our known ROI scans (those are each scan's own whole-image
    mean -- not an independent spectrum for a different material). Returns
    None if no such extra file exists; otherwise {label: mean_spectrum}.
    """
    known_stems = {f"{s}-roi-{r}".lower() for s, rois in pairs.items() for r in rois}

    extra = []
    for hdr in glob.glob(os.path.join(reference_dir, "*.spec.hdr")):
        stem = os.path.basename(hdr).split("-mean-")[0].lower()
        if stem not in known_stems:
            extra.append(hdr)

    if not extra:
        return None

    endmembers = {}
    for hdr in extra:
        mean_spectrum, _ = load_reference_spectrum(hdr)
        label = os.path.basename(hdr).replace(".spec.hdr", "")
        endmembers[label] = mean_spectrum
    return endmembers


def linear_spectral_unmix(pixels_flat, endmembers):
    """Fully constrained (non-negative, sum-to-one) least squares unmixing.

    endmembers: {label: spectrum (n_bands,)}. Returns (abundances, labels)
    where abundances has shape (n_pixels, n_endmembers).
    """
    labels = list(endmembers)
    matrix = np.stack([endmembers[l] for l in labels])  # (n_end, n_bands)
    n_end = matrix.shape[0]
    sum_to_one_weight = np.linalg.norm(matrix, axis=1).mean()  # scale constraint row like the data
    design = np.vstack([matrix.T, sum_to_one_weight * np.ones((1, n_end))])

    abundances = np.empty((pixels_flat.shape[0], n_end))
    for i, pixel in enumerate(pixels_flat):
        target = np.concatenate([pixel, [sum_to_one_weight]])
        abundances[i], _ = nnls(design, target)
    return abundances, labels


# --------------------------------------------------------------------------
# PCA + K-means spectral classification (fallback when no endmembers exist)
# --------------------------------------------------------------------------

def fit_pca_kmeans(pooled_spectra, k_range=range(2, 7), variance_target=0.95,
                    seed=0, max_silhouette_sample=3000):
    """PCA down to `variance_target` explained variance, then K-means with k
    chosen by silhouette score over k_range (no assumption about how many
    distinct material/defect classes exist).
    """
    full_pca = PCA(n_components=min(pooled_spectra.shape)).fit(pooled_spectra)
    n_components = int(np.searchsorted(np.cumsum(full_pca.explained_variance_ratio_), variance_target)) + 1
    pca = PCA(n_components=n_components).fit(pooled_spectra)
    reduced = pca.transform(pooled_spectra)

    rng = np.random.default_rng(seed)
    sil_idx = (rng.choice(reduced.shape[0], max_silhouette_sample, replace=False)
               if reduced.shape[0] > max_silhouette_sample else np.arange(reduced.shape[0]))

    scores_by_k, best_k, best_score, best_km = {}, None, -1.0, None
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(reduced)
        score = silhouette_score(reduced[sil_idx], km.labels_[sil_idx])
        scores_by_k[k] = score
        if score > best_score:
            best_k, best_score, best_km = k, score, km

    return pca, best_km, best_k, scores_by_k


def classify_pixels(cube_flat, pca, km):
    """Assign each pixel to its nearest K-means class and score how far it
    sits from that class's centroid (a continuous, per-pixel deviation score
    -- large values mean 'spectrally atypical for its assigned class').
    """
    reduced = pca.transform(cube_flat)
    labels = km.predict(reduced)
    centroids = km.cluster_centers_[labels]
    deviation = np.linalg.norm(reduced - centroids, axis=1)
    return labels, deviation


# --------------------------------------------------------------------------
# Spatial filtering (mode filter over class labels)
# --------------------------------------------------------------------------

def _mode_ignoring_invalid(values):
    valid = values[values >= 0]
    if valid.size == 0:
        return -1
    return np.argmax(np.bincount(valid.astype(int)))


def mode_filter_labels(label_map, valid_mask, size=3):
    """Majority-vote spatial filter: replaces salt-and-pepper misclassified
    single pixels with the most common class in their neighborhood. Only
    considers/updates valid (LIG) pixels; substrate stays excluded (-1).
    """
    filled = np.where(valid_mask, label_map, -1)
    filtered = generic_filter(filled, _mode_ignoring_invalid, size=size, mode="nearest")
    return np.where(valid_mask, filtered, -1).astype(int)


# --------------------------------------------------------------------------
# Interpolation (upsample to a finer grid)
# --------------------------------------------------------------------------

def upsample_maps(label_map, deviation_map, valid_mask, factor=4):
    """Nearest-neighbor upsample for discrete labels/mask (never invent a
    class that isn't there), cubic upsample for the continuous deviation
    map (smooth), re-masked so the finer grid still respects the original
    LIG/substrate boundary.
    """
    label_up = zoom(label_map, factor, order=0)
    mask_up = zoom(valid_mask.astype(np.float64), factor, order=0) > 0.5
    deviation_up = zoom(np.where(valid_mask, deviation_map, 0.0), factor, order=3)
    label_up = np.where(mask_up, label_up, -1)
    deviation_up = np.where(mask_up, deviation_up, np.nan)
    return label_up, deviation_up, mask_up


# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------

CLASS_CMAP = ListedColormap(plt.get_cmap("tab10").colors)


def save_composition_plots(sample_id, roi_num, cube, wavelengths, raw_labels, filtered_labels,
                            deviation_map, valid_mask, majority_label, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rgb = pseudo_rgb(cube, wavelengths)
    defect_map = valid_mask & (filtered_labels != majority_label) & (filtered_labels >= 0)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(rgb)
    axes[0].set_title(f"{sample_id} ROI-{roi_num}: pseudo-RGB")

    axes[1].imshow(np.ma.masked_where(~valid_mask, raw_labels), cmap=CLASS_CMAP, vmin=0, vmax=9)
    axes[1].set_title("Composition map (raw)")

    axes[2].imshow(np.ma.masked_where(~valid_mask, filtered_labels), cmap=CLASS_CMAP, vmin=0, vmax=9)
    axes[2].set_title("Composition map (mode-filtered)")

    im = axes[3].imshow(np.ma.masked_where(~valid_mask, deviation_map), cmap="viridis")
    axes[3].set_title("Quantitative deviation map")
    fig.colorbar(im, ax=axes[3], fraction=0.046)

    axes[4].imshow(rgb)
    axes[4].imshow(np.ma.masked_where(~defect_map, defect_map), cmap="autumn", alpha=0.6)
    axes[4].set_title("Defect map (minority classes)")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{sample_id}_roi{roi_num}_composition.png"), dpi=150)
    plt.close(fig)


def save_upsampled_plot(sample_id, roi_num, label_up, deviation_up, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(np.ma.masked_where(label_up < 0, label_up), cmap=CLASS_CMAP, vmin=0, vmax=9)
    axes[0].set_title("Upsampled composition map")
    im = axes[1].imshow(deviation_up, cmap="viridis")
    axes[1].set_title("Upsampled deviation map")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{sample_id}_roi{roi_num}_upsampled.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Composition/defect classification for LIG HSI scans.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--reference-dir", type=str, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--white-ref", type=str, default=DEFAULT_WHITE_REF)
    parser.add_argument("--dark-ref", type=str, default=DEFAULT_DARK_REF)
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--invert-foreground", action="store_true",
                         help="Flip which KMeans cluster is treated as the LIG foreground "
                              "(see unsupervised_defect.py -- needed for this dataset's ROI-2 crops).")
    parser.add_argument("--max-per-image", type=int, default=4000)
    parser.add_argument("--upsample-factor", type=int, default=4)
    parser.add_argument("--filter-size", type=int, default=3, help="Mode filter neighborhood size (pixels).")
    parser.add_argument("--erode", type=int, default=2,
                         help="Pixels to erode off the LIG mask before classification, to exclude "
                              "substrate/LIG boundary pixels whose mixed spectra otherwise form a "
                              "spurious 'ring' class tracing the pattern outline (not a real defect).")
    parser.add_argument("--window", type=int, default=7,
                         help="Local neighborhood window size (pixels) for local mean/std detrending, "
                              "applied after the plane fit -- needed because the known lighting "
                              "asymmetry isn't perfectly linear (a plane fit alone leaves a residual "
                              "top/bottom split in some samples).")
    args = parser.parse_args()

    pairs = find_pairs(args.data_dir)
    samples = [s for s in pairs if 1 in pairs[s] and 2 in pairs[s]]
    print(f"Found {len(samples)} complete sample pairs: {samples}")

    endmembers = find_usable_endmembers(args.reference_dir, pairs)
    if endmembers:
        print(f"Found {len(endmembers)} usable endmember spectra -- using Linear Spectral Unmixing.")
    else:
        print("No usable endmember library found in roi_mean_spectra (every .spec file there is just "
              "a scan's own whole-image mean, not an independent reference spectrum) -- "
              "falling back to PCA + K-means classification.")

    white_mean = dark_mean = shutter_white = shutter_dark = None
    if not args.no_calibrate:
        white_mean, shutter_white = load_reference_spectrum(args.white_ref)
        dark_mean, shutter_dark = load_reference_spectrum(args.dark_ref)

    cubes, sat_masks = {}, {}
    for sample_id in samples:
        cubes[sample_id], sat_masks[sample_id] = {}, {}
        for roi_num in (1, 2):
            raw_cube, wl, shutter_sample, ceiling = load_cube_with_shutter(pairs[sample_id][roi_num])
            sat_masks[sample_id][roi_num] = saturated_pixel_mask(raw_cube, ceiling)
            cube = raw_cube
            if not args.no_calibrate:
                cube = calibrate_to_reflectance(cube, shutter_sample, white_mean, shutter_white,
                                                 dark_mean, shutter_dark)
            cubes[sample_id][roi_num] = (cube, wl)

    # --- Segment substrate vs. LIG (same approach as unsupervised_defect.py). ---
    all_flats = [cubes[s][r][0].reshape(-1, cubes[s][r][0].shape[-1])[~sat_masks[s][r].ravel()]
                 for s in samples for r in (1, 2)]
    seg_pool = pool_background(all_flats, max_per_image=args.max_per_image)
    seg_km, fg_label = fit_segmenter(seg_pool)
    if args.invert_foreground:
        fg_label = 1 - fg_label

    masks = {}
    for sample_id in samples:
        masks[sample_id] = {}
        for roi_num in (1, 2):
            cube, wl = cubes[sample_id][roi_num]
            flat = cube.reshape(-1, cube.shape[-1])
            raw_mask = foreground_mask(flat, cube.shape[:2], seg_km, fg_label) & ~sat_masks[sample_id][roi_num]
            eroded_mask = binary_erosion(raw_mask, iterations=args.erode) if args.erode > 0 else raw_mask
            masks[sample_id][roi_num] = eroded_mask

    # --- Plane-detrend each scan within its LIG mask: removes the known lighting asymmetry
    #     (confirmed fixed across the imaging stage, not a material property) before it can
    #     get picked up as its own PCA/K-means class -- see unsupervised_defect.py, where the
    #     same detrend already fixed an analogous confound for the RX anomaly detector. ---
    detrended = {}
    for sample_id in samples:
        detrended[sample_id] = {}
        for roi_num in (1, 2):
            cube, wl = cubes[sample_id][roi_num]
            mask = masks[sample_id][roi_num]
            plane_detrended = detrend_plane_masked(cube, mask)
            local_mean, local_std, _ = local_stats_masked(plane_detrended, mask, args.window)
            detrended[sample_id][roi_num] = (plane_detrended - local_mean) / local_std

    # --- LIG pixel pool (ROI-1 scans define the reference population, as in unsupervised_defect.py). ---
    roi1_lig_pixels = [detrended[s][1].reshape(-1, detrended[s][1].shape[-1])[masks[s][1].ravel()]
                       for s in samples]
    pooled_lig = pool_background(roi1_lig_pixels, max_per_image=args.max_per_image)
    print(f"Pooled LIG pixels for classification: {pooled_lig.shape[0]} x {pooled_lig.shape[1]} bands")

    if endmembers:
        pooled_abundances, endmember_labels = linear_spectral_unmix(pooled_lig, endmembers)
        # Hard-assign each pixel to its dominant endmember so the rest of the
        # pipeline (composition maps, spatial filter, etc.) works the same
        # way regardless of which branch produced the classification.
        pooled_labels = pooled_abundances.argmax(axis=1)
        majority_label = int(np.bincount(pooled_labels).argmax())
        pca = km = None
    else:
        pca, km, best_k, scores_by_k = fit_pca_kmeans(pooled_lig)
        print(f"Selected k={best_k} by silhouette score (candidates tried: {scores_by_k})")
        pooled_labels = km.predict(pca.transform(pooled_lig))
        majority_label = int(np.bincount(pooled_labels).argmax())
    print(f"Majority (bulk LIG) class: {majority_label} "
          f"({np.bincount(pooled_labels)[majority_label] / len(pooled_labels):.1%} of pooled LIG pixels)")

    summary_rows = []
    for sample_id in samples:
        for roi_num in (1, 2):
            cube, wl = cubes[sample_id][roi_num]
            mask = masks[sample_id][roi_num]

            if endmembers:
                # LSU compares against true reflectance endmembers -- detrending would
                # distort that physical meaning, so unmix the raw calibrated cube.
                flat = cube.reshape(-1, cube.shape[-1])
                abundances, _ = linear_spectral_unmix(flat, endmembers)
                raw_labels_flat = abundances.argmax(axis=1)
                deviation_flat = 1.0 - abundances.max(axis=1)  # low max abundance = poor fit to any endmember
            else:
                flat = detrended[sample_id][roi_num].reshape(-1, cube.shape[-1])
                raw_labels_flat, deviation_flat = classify_pixels(flat, pca, km)

            raw_labels = raw_labels_flat.reshape(cube.shape[:2])
            deviation_map = deviation_flat.reshape(cube.shape[:2])

            filtered_labels = mode_filter_labels(raw_labels, mask, size=args.filter_size)

            out_dir = os.path.join(args.out, sample_id)
            save_composition_plots(sample_id, roi_num, cube, wl, raw_labels, filtered_labels,
                                    deviation_map, mask, majority_label, out_dir)

            label_up, deviation_up, mask_up = upsample_maps(filtered_labels, deviation_map, mask,
                                                              factor=args.upsample_factor)
            save_upsampled_plot(sample_id, roi_num, label_up, deviation_up, out_dir)

            valid_labels = filtered_labels[mask]
            n_valid = valid_labels.size
            class_fracs = {int(c): float((valid_labels == c).sum()) / n_valid
                           for c in np.unique(valid_labels) if c >= 0} if n_valid else {}
            defect_frac = 1.0 - class_fracs.get(majority_label, 0.0)
            mean_dev = float(deviation_map[mask].mean()) if n_valid else float("nan")
            summary_rows.append((sample_id, roi_num, n_valid, defect_frac, mean_dev, class_fracs))

    print("\n=== Summary (sample, ROI, n_LIG_pixels, defect_area_frac, mean_deviation, class_fractions) ===")
    for row in summary_rows:
        print(f"{row[0]:>8}  ROI-{row[1]}  n={row[2]:5d}  defect_frac={row[3]:.2%}  "
              f"mean_dev={row[4]:.3f}  classes={row[5]}")


if __name__ == "__main__":
    main()
