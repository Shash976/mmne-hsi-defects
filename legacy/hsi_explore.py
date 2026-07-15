# hsi_explore.py
"""
Quick-look hyperspectral image (HSI) exploration and QC metrics.

Loads an ENVI-format cube (.hdr/.img or .hdr/.raw pair) via the `spectral`
package, computes summary statistics useful for judging whether a scan is
usable, and saves a few overview plots. Meant as a sanity-check / testing
step before building out the ML pipeline in pipeline.py -- if no real data
is available yet, it falls back to a synthetic cube so the whole script can
still be exercised end to end.

Usage:
    python hsi_explore.py --hdr data/lig/sample1.hdr --out out/sample1
    python hsi_explore.py --demo --out out/demo
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

try:
    import spectral
except ImportError:
    spectral = None


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_cube(hdr_path):
    """Load an ENVI header file into a (rows, cols, bands) float array."""
    if spectral is None:
        raise ImportError("The 'spectral' package is required to load ENVI files (pip install spectral).")
    img = spectral.open_image(hdr_path)
    cube = np.asarray(img.load(), dtype=np.float64)
    wavelengths = img.bands.centers if img.bands is not None else None
    return cube, wavelengths


def synthetic_cube(rows=120, cols=160, bands=224, seed=0):
    """Generate a fake cube (two materials + noise) for testing the script without real data."""
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(400, 1000, bands)

    def gaussian_peak(center, width, height):
        return height * np.exp(-0.5 * ((wavelengths - center) / width) ** 2)

    spec_a = 0.2 + gaussian_peak(550, 40, 0.5) + gaussian_peak(850, 60, 0.3)
    spec_b = 0.15 + gaussian_peak(650, 50, 0.6)

    mix = rng.uniform(0, 1, size=(rows, cols))
    cube = np.outer(mix, spec_a).reshape(rows, cols, bands) + \
        np.outer(1 - mix, spec_b).reshape(rows, cols, bands)
    cube += rng.normal(0, 0.02, cube.shape)
    cube = np.clip(cube, 0, None)

    # a few dead/saturated pixels to make the QC checks non-trivial
    cube[0, 0, :] = 0.0
    cube[-1, -1, :] = cube.max() * 2

    return cube, wavelengths


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def band_stats(cube):
    """Per-band mean/std/min/max over all pixels. Shapes: (bands,)."""
    flat = cube.reshape(-1, cube.shape[-1])
    return {
        "mean": flat.mean(axis=0),
        "std": flat.std(axis=0),
        "min": flat.min(axis=0),
        "max": flat.max(axis=0),
    }


def snr_estimate(cube):
    """Rough per-band SNR = mean / std across pixels (higher is better)."""
    stats = band_stats(cube)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(stats["std"] > 0, stats["mean"] / stats["std"], 0.0)
    return snr


def saturation_fraction(cube, saturation_value=None):
    """Fraction of pixels at/near the max observed value, per band."""
    if saturation_value is None:
        saturation_value = cube.max()
    return (cube >= 0.99 * saturation_value).reshape(-1, cube.shape[-1]).mean(axis=0)


def dead_pixel_fraction(cube, threshold=1e-6):
    """Fraction of pixels near zero across all bands (likely dead/masked)."""
    flat = cube.reshape(-1, cube.shape[-1])
    return (flat.max(axis=1) <= threshold).mean()


def pca_variance_summary(cube, n_components=10):
    """Explained variance ratio of the first n_components principal components."""
    from sklearn.decomposition import PCA
    flat = cube.reshape(-1, cube.shape[-1])
    n_components = min(n_components, flat.shape[1], flat.shape[0])
    pca = PCA(n_components=n_components)
    pca.fit(flat)
    return pca.explained_variance_ratio_


def summarize(cube, wavelengths=None):
    """Build a dict of headline metrics for a cube."""
    rows, cols, bands = cube.shape
    stats = band_stats(cube)
    snr = snr_estimate(cube)
    explained_var = pca_variance_summary(cube)

    summary = {
        "shape": (rows, cols, bands),
        "n_pixels": rows * cols,
        "wavelength_range": (float(wavelengths[0]), float(wavelengths[-1])) if wavelengths is not None else None,
        "global_min": float(cube.min()),
        "global_max": float(cube.max()),
        "mean_intensity": float(stats["mean"].mean()),
        "mean_snr": float(np.nanmean(snr)),
        "worst_band_snr": float(np.nanmin(snr)),
        "dead_pixel_fraction": float(dead_pixel_fraction(cube)),
        "saturation_fraction_max": float(saturation_fraction(cube).max()),
        "pca_variance_top5": explained_var[:5].tolist(),
        "pca_components_for_95pct": int(np.searchsorted(np.cumsum(explained_var), 0.95) + 1)
        if explained_var.sum() >= 0.95 else None,
    }
    return summary


def print_summary(summary):
    print("=== HSI Quick-Look Summary ===")
    print(f"Cube shape (rows, cols, bands): {summary['shape']}")
    print(f"Pixels: {summary['n_pixels']}")
    if summary["wavelength_range"]:
        lo, hi = summary["wavelength_range"]
        print(f"Wavelength range: {lo:.1f} - {hi:.1f} nm")
    print(f"Intensity range: [{summary['global_min']:.4f}, {summary['global_max']:.4f}]")
    print(f"Mean intensity: {summary['mean_intensity']:.4f}")
    print(f"Mean per-band SNR: {summary['mean_snr']:.2f} (worst band: {summary['worst_band_snr']:.2f})")
    print(f"Dead-pixel fraction: {summary['dead_pixel_fraction']:.4%}")
    print(f"Max per-band saturation fraction: {summary['saturation_fraction_max']:.4%}")
    print(f"Top-5 PCA explained variance: {['%.3f' % v for v in summary['pca_variance_top5']]}")
    print(f"Components needed for 95% variance: {summary['pca_components_for_95pct']}")


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def plot_overview(cube, wavelengths, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows, cols, bands = cube.shape
    x_axis = wavelengths if wavelengths is not None else np.arange(bands)

    stats = band_stats(cube)

    # mean +/- std spectrum
    plt.figure()
    plt.plot(x_axis, stats["mean"], label="mean")
    plt.fill_between(x_axis, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                      alpha=0.3, label="+/- 1 std")
    plt.xlabel("Wavelength (nm)" if wavelengths is not None else "Band index")
    plt.ylabel("Intensity")
    plt.title("Mean spectrum over full scene")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "mean_spectrum.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # per-band SNR
    plt.figure()
    plt.plot(x_axis, snr_estimate(cube))
    plt.xlabel("Wavelength (nm)" if wavelengths is not None else "Band index")
    plt.ylabel("SNR (mean/std)")
    plt.title("Per-band SNR")
    plt.savefig(os.path.join(out_dir, "snr.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # pseudo-RGB / three-band composite (evenly spaced bands)
    b_idx = sorted(set(np.linspace(0, bands - 1, 3).astype(int)))
    while len(b_idx) < 3:
        b_idx.append(b_idx[-1])
    rgb = cube[:, :, b_idx]
    rgb = (rgb - rgb.min()) / (np.ptp(rgb) + 1e-12)
    plt.figure()
    plt.imshow(rgb)
    plt.title(f"Pseudo-RGB (bands {b_idx})")
    plt.axis("off")
    plt.savefig(os.path.join(out_dir, "pseudo_rgb.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # PCA scree plot
    explained_var = pca_variance_summary(cube)
    plt.figure()
    plt.plot(np.arange(1, len(explained_var) + 1), np.cumsum(explained_var), marker="o")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA scree plot")
    plt.axhline(0.95, color="gray", linestyle="--", linewidth=1)
    plt.savefig(os.path.join(out_dir, "pca_scree.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plots to {out_dir}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Quick-look HSI stats and plots.")
    parser.add_argument("--hdr", type=str, default=None, help="Path to ENVI .hdr file")
    parser.add_argument("--demo", action="store_true", help="Use a synthetic cube instead of real data")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for --demo (vary this to get a different synthetic cube)")
    parser.add_argument("--out", type=str, default="out/hsi_explore", help="Directory for output plots")
    args = parser.parse_args()

    if args.hdr:
        if not os.path.exists(args.hdr):
            raise FileNotFoundError(
                f"'{args.hdr}' does not exist. Pass a valid ENVI .hdr path, or use --demo explicitly."
            )
        cube, wavelengths = load_cube(args.hdr)
    elif args.demo:
        cube, wavelengths = synthetic_cube(seed=args.seed)
    else:
        raise ValueError("Pass --hdr <path> for real data or --demo for a synthetic test cube.")

    source = args.hdr if args.hdr else f"synthetic (seed={args.seed})"
    print(f"Source: {source}")
    summary = summarize(cube, wavelengths)
    print_summary(summary)
    plot_overview(cube, wavelengths, args.out)


if __name__ == "__main__":
    main()
