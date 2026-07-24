"""Revised Stages 2-3 -- Calibration + Preprocessing.

Revised order (see ``Revised Research Objective.md``):

    Raw cube
      -> (optional) register a separate reference scan
      -> radiometric calibration (dark/white -> reflectance)   [Stage 2]
      -> Savitzky-Golay spectral smoothing                     [Stage 3.2]
      -> (optional) baseline correction                        [Stage 3.4]
      -> SNV normalization                                     [Stage 3.3]

The analysis cube handed to PCA / clustering / anomaly is this SG+SNV
reflectance. Optical density (the old Step 10) is retained in
``optical_density.py`` but off the default path (``od_method="none"``); the
film/substrate segmentation is therefore only computed when OD needs it.

Each stage is a small standalone function; ``preprocess`` wires them together
according to a ``PreprocessConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter

from .config import PreprocessConfig
from .cube_io import Cube, load_reference_spectrum
from .segmentation import segment, Segmentation


@dataclass
class Preprocessed:
    """Output of :func:`preprocess`.

    ``data`` is the processed analysis cube (SG+SNV reflectance by default).
    ``saturated`` flags clipped pixels. ``segmentation`` is the film/substrate
    split -- only populated when the optical-density substrate path needs it,
    otherwise ``None`` (piece extraction supplies the foreground mask instead).
    ``reflectance_mean`` is the (rows, cols) band-mean of the *pre-SNV*
    reflectance so later stages can report physical reflectance. ``noise`` holds
    the before/after smoothing noise metrics (the document's Stage 3/4 metric).
    """

    data: np.ndarray
    wavelengths: Optional[np.ndarray]
    saturated: np.ndarray            # (rows, cols) bool
    segmentation: Optional[Segmentation]
    label: str
    reflectance_mean: Optional[np.ndarray] = None
    noise: Optional[dict] = None


# --------------------------------------------------------------------------
# Individual Step-9 stages
# --------------------------------------------------------------------------

def saturation_mask(cube: np.ndarray, ceiling: float) -> np.ndarray:
    """True where a pixel hits the sensor ceiling in any band.

    A saturated pixel's spectrum is clipped and unreliable; dividing by a
    near-zero calibration denominator there also blows up. Flag them so they can
    be excluded from segmentation, I0 estimation and statistics.
    """
    if not np.isfinite(ceiling):
        return np.zeros(cube.shape[:2], dtype=bool)
    return cube.max(axis=-1) >= ceiling - 1


def register(moving: np.ndarray, reference: Optional[np.ndarray] = None,
             enable: bool = False) -> tuple[np.ndarray, tuple[float, float]]:
    """Translation-align ``moving`` onto ``reference`` (band-mean images).

    Used when the background / I0 comes from a *separate* scan that isn't pixel
    aligned to the sample (the future transmission geometry). For a single,
    already-co-registered cube this is a no-op. Returns the aligned cube and the
    (row, col) shift that was applied.

    Kept dependency-light: uses ``skimage.registration.phase_cross_correlation``
    on the mean-over-bands image and applies an integer-pixel roll.
    """
    if not enable or reference is None:
        return moving, (0.0, 0.0)

    from skimage.registration import phase_cross_correlation

    mov_img = moving.mean(axis=-1)
    ref_img = reference.mean(axis=-1)
    shift, _, _ = phase_cross_correlation(ref_img, mov_img, upsample_factor=1)
    dr, dc = int(round(shift[0])), int(round(shift[1]))
    aligned = np.roll(moving, shift=(dr, dc), axis=(0, 1))
    return aligned, (float(dr), float(dc))


def subtract_background(cube: np.ndarray, dark_spectrum: np.ndarray) -> np.ndarray:
    """Subtract a per-band baseline (dark reference) spectrum from every pixel."""
    return cube - dark_spectrum[None, None, :]


def calibrate_reflectance(cube: np.ndarray, shutter_sample: float,
                          white_mean: np.ndarray, shutter_white: float,
                          dark_mean: np.ndarray, shutter_dark: float,
                          eps: float = 1e-6) -> np.ndarray:
    """Exposure-normalized (raw-dark)/(white-dark) reflectance.

    White/dark references are captured at their own shutter times, so raw DN
    isn't directly comparable across them -- convert everything to a per-second
    rate first (linear-sensor assumption), then apply the flat-field formula.
    """
    sample_rate = cube / shutter_sample
    white_rate = white_mean / shutter_white
    dark_rate = dark_mean / shutter_dark
    denom = white_rate - dark_rate
    denom = np.where(np.abs(denom) < eps, eps, denom)
    return (sample_rate - dark_rate) / denom


def snv(flat_spectra: np.ndarray) -> np.ndarray:
    """Standard Normal Variate: per-pixel (per-row) mean/std normalization.

    Removes per-pixel multiplicative/additive scale so downstream steps key on
    spectral shape rather than overall brightness.
    """
    mean = flat_spectra.mean(axis=1, keepdims=True)
    std = flat_spectra.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (flat_spectra - mean) / std


def normalize_intensity(cube: np.ndarray, method: str) -> np.ndarray:
    """Optional per-pixel normalization applied after reflectance calibration."""
    if method == "none":
        return cube
    if method == "snv":
        flat = snv(cube.reshape(-1, cube.shape[-1]))
        return flat.reshape(cube.shape)
    raise ValueError(f"unknown normalize method: {method!r}")


def savgol_smooth(cube: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay smoothing along the spectral axis (Stage 3.2).

    Fits a low-order polynomial in a sliding spectral window, which suppresses
    per-band sensor noise while preserving peak positions and widths far better
    than a boxcar/moving average. ``window`` must be odd and > ``polyorder``.
    """
    window = min(window, cube.shape[-1] - (1 - cube.shape[-1] % 2))  # keep <= n_bands, odd
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        return cube
    return savgol_filter(cube, window_length=window, polyorder=polyorder, axis=-1)


def noise_metrics(cube: np.ndarray, window: int, polyorder: int,
                  sample: int = 5000, seed: int = 0) -> dict:
    """High-frequency RMS noise + spectral SNR on a random pixel subsample.

    The document's Stage 3/4 metric: "reduced high-frequency noise while
    retaining spectral shape". Noise is estimated as the RMS residual between
    each spectrum and its Savitzky-Golay fit (the smooth component); SNR is the
    mean absolute signal over that noise. Compute this on the cube *before* and
    *after* smoothing to quantify the reduction.
    """
    rows, cols, bands = cube.shape
    flat = cube.reshape(-1, bands)
    rng = np.random.default_rng(seed)
    if flat.shape[0] > sample:
        flat = flat[rng.choice(flat.shape[0], sample, replace=False)]
    w = min(window, bands - (1 - bands % 2))
    if w % 2 == 0:
        w -= 1
    if w <= polyorder:
        return {"rms_noise": float("nan"), "snr": float("nan"), "n_pixels": flat.shape[0]}
    smooth = savgol_filter(flat, window_length=w, polyorder=polyorder, axis=-1)
    resid = flat - smooth
    rms = float(np.sqrt(np.mean(resid ** 2)))
    signal = float(np.mean(np.abs(smooth)))
    return {"rms_noise": rms,
            "snr": signal / rms if rms > 0 else float("inf"),
            "n_pixels": int(flat.shape[0])}


def baseline_correct(cube: np.ndarray, method: str, order: int = 2) -> np.ndarray:
    """Optional per-pixel baseline removal (Stage 3.4).

    ``method="poly"`` fits a low-order polynomial to each pixel's spectrum across
    the band index and subtracts it, removing slow scattering/offset trends while
    leaving sharper spectral features. ``method="none"`` is a pass-through.
    """
    if method == "none":
        return cube
    if method != "poly":
        raise ValueError(f"unknown baseline method: {method!r}")
    rows, cols, bands = cube.shape
    x = np.linspace(-1.0, 1.0, bands)
    # Vandermonde least-squares fit shared across all pixels (design matrix fixed).
    V = np.vander(x, order + 1, increasing=True)          # (bands, order+1)
    flat = cube.reshape(-1, bands).T                      # (bands, n_pixels)
    coeffs, *_ = np.linalg.lstsq(V, flat, rcond=None)     # (order+1, n_pixels)
    trend = (V @ coeffs).T.reshape(rows, cols, bands)
    return cube - trend


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

def preprocess(cube: Cube, cfg: PreprocessConfig,
               white_ref_hdr: Optional[str] = None,
               dark_ref_hdr: Optional[str] = None,
               reference_cube: Optional[Cube] = None) -> Preprocessed:
    """Run Step 9 on one cube.

    Order follows the document: (optional) register a separate reference to the
    sample -> subtract background -> normalize intensity (reflectance
    calibration, then optional SNV). Saturation masking and film/substrate
    segmentation are computed on the result for the OD step and later stages.
    """
    cfg.validate()
    data = cube.data
    saturated = saturation_mask(data, cube.ceiling)

    # --- Align a separate reference/background scan onto the sample, if asked. ---
    if cfg.register and reference_cube is not None:
        _, shift = register(reference_cube.data, data, enable=True)
        # We align the *reference* frame; store nothing further here -- the OD
        # step consumes reference_cube, so shift it to match the sample grid.
        reference_cube = Cube(
            data=np.roll(reference_cube.data, shift=(int(shift[0]), int(shift[1])), axis=(0, 1)),
            wavelengths=reference_cube.wavelengths, shutter=reference_cube.shutter,
            ceiling=reference_cube.ceiling, path=reference_cube.path, label=reference_cube.label,
            material=reference_cube.material,
        )

    # --- Stage 2: reflectance calibration ---
    # (raw-dark)/(white-dark) folds dark background subtraction and white
    # normalization into one formula when white/dark refs exist. Without refs,
    # fall back to explicit dark subtraction when a dark spectrum is available.
    white_mean = dark_mean = None
    shutter_white = shutter_dark = 1.0
    if white_ref_hdr and dark_ref_hdr:
        white_mean, shutter_white = load_reference_spectrum(white_ref_hdr)
        dark_mean, shutter_dark = load_reference_spectrum(dark_ref_hdr)

    if cfg.calibrate and white_mean is not None and dark_mean is not None:
        data = calibrate_reflectance(data, cube.shutter, white_mean, shutter_white,
                                     dark_mean, shutter_dark, eps=cfg.eps)
    elif cfg.background == "dark" and dark_mean is not None:
        data = subtract_background(data, dark_mean)

    # --- Stage 3.2 -> 3.4 -> 3.3: smooth, baseline, normalize ---
    # Noise metrics (Stage 3/4 deliverable): RMS high-frequency noise + spectral
    # SNR before vs after smoothing, on a pixel subsample.
    noise = {"before": noise_metrics(data, cfg.sg_window, cfg.sg_polyorder, seed=cfg.seed)}
    if cfg.smooth == "savgol":
        data = savgol_smooth(data, cfg.sg_window, cfg.sg_polyorder)
    noise["after"] = noise_metrics(data, cfg.sg_window, cfg.sg_polyorder, seed=cfg.seed)
    data = baseline_correct(data, cfg.baseline, cfg.baseline_order)
    # Physical reflectance band-mean, captured before SNV flattens intensity.
    reflectance_mean = data.mean(axis=-1).astype(np.float32)
    data = normalize_intensity(data, cfg.normalize)

    # Film/substrate segmentation is only needed for the substrate-referenced OD
    # path; skip the (expensive) KMeans otherwise -- piece extraction supplies
    # the foreground mask for the anomaly pipeline.
    seg = None
    if cfg.od_method == "substrate":
        seg = segment(data, valid_mask=~saturated, invert=cfg.invert_foreground, seed=cfg.seed)
    return Preprocessed(data=data, wavelengths=cube.wavelengths, saturated=saturated,
                        segmentation=seg, label=cube.label,
                        reflectance_mean=reflectance_mean, noise=noise)
