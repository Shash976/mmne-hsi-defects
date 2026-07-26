"""Stage 3.1 -- Piece extraction (raw multi-piece scan -> individual pieces).

The SiO2/Si scans image *several physical fragments at once* on a dish/holder
(e.g. ~20 SiO2 pieces, or ~10 bare-Si fragments). Every downstream stage wants
to reason about one physical piece at a time, so this module splits a raw cube
into a list of :class:`Piece` sub-cubes.

The whole separation is done on **full spectra**, never on pseudo-RGB brightness:

1. Estimate the dish/background spectrum from a border frame of the cube.
2. Flag foreground pixels by spectral *distance* from that background (Spectral
   Angle Mapper, or Mahalanobis, or a 2-cluster KMeans over all bands).
3. Clean the mask morphologically (opening removes thin rim arcs/dust; closing
   merges within-piece gaps so patterned devices stay whole) and label
   connected components; drop anything smaller than ``min_area``.
4. Crop each component's bounding box out of the full cube.

For single-piece scans (the LIG ROIs) the foreground is one blob and the whole
frame comes back as a single :class:`Piece`, so downstream code is identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

from .config import PieceConfig
from .cube_io import Cube


@dataclass
class Piece:
    """One physical fragment cropped out of a multi-piece scan.

    ``data`` is the (rows, cols, bands) sub-cube for the piece's bounding box;
    ``mask`` is the (rows, cols) boolean of which of those pixels actually belong
    to the fragment (the rest of the bbox is dish/other-piece and must be ignored
    by later stages). ``material`` is inherited from the source cube; ``bbox`` is
    ``(row0, row1, col0, col1)`` in the original scan's coordinates.
    """

    data: np.ndarray
    mask: np.ndarray
    material: str
    piece_id: str
    source_label: str
    bbox: Tuple[int, int, int, int]
    wavelengths: Optional[np.ndarray] = None
    # (rows, cols) band-mean of the *pre-SNV reflectance*, so later stages can
    # report physical "mean reflectance" even though ``data`` is SNV-normalized.
    reflectance_mean: Optional[np.ndarray] = None
    # Per-piece noise metrics from preprocessing (before/after smoothing).
    noise: Optional[dict] = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def n_bands(self) -> int:
        return self.data.shape[-1]

    def foreground_spectra(self) -> np.ndarray:
        """(n_fg_pixels, bands) matrix of the in-mask spectra."""
        return self.data[self.mask]


# --------------------------------------------------------------------------
# Background model + foreground distance (full-spectrum)
# --------------------------------------------------------------------------

def border_background_spectrum(cube: np.ndarray, width: int = 8) -> np.ndarray:
    """Robust mean spectrum of a border frame (assumed dish/holder).

    The outermost ``width`` rows/cols of a scan are almost always empty
    dish/holder, so their median spectrum is a label-free estimate of "what
    background looks like" for this specific scan -- valid for both dark- and
    light-background dishes.
    """
    rows, cols, _ = cube.shape
    frame = np.zeros((rows, cols), dtype=bool)
    frame[:width, :] = frame[-width:, :] = True
    frame[:, :width] = frame[:, -width:] = True
    return np.median(cube[frame], axis=0)


def flat_field_correct(cube: np.ndarray, sigma: float) -> np.ndarray:
    """Divide out a smooth spatial illumination gradient, per band.

    The white-dish scan has an uneven brightness field across the frame. A
    large-sigma Gaussian blur of each band estimates that low-frequency field;
    dividing it out flattens the lighting so thresholding/Mahalanobis are not
    biased by *where* a piece sits. Done in one call with a per-axis sigma
    (spatial only, none across bands) so spectral shape is preserved up to the
    per-pixel/per-band scalar.
    """
    from scipy.ndimage import gaussian_filter
    field = gaussian_filter(cube, sigma=(sigma, sigma, 0.0))
    field = np.where(np.abs(field) < 1e-9, 1e-9, field)
    return cube / field


def _background_pixels(cube: np.ndarray, cfg: PieceConfig) -> np.ndarray:
    """(n_bg, bands) background spectra: an explicit bbox if given, else the border."""
    rows, cols, bands = cube.shape
    if cfg.background_bbox is not None:
        r0, r1, c0, c1 = cfg.background_bbox
        return cube[r0:r1, c0:c1, :].reshape(-1, bands)
    w = cfg.border_width
    frame = np.zeros((rows, cols), dtype=bool)
    frame[:w, :] = frame[-w:, :] = True
    frame[:, :w] = frame[:, -w:] = True
    return cube[frame]


def spectral_angle(flat: np.ndarray, ref: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Spectral Angle Mapper: angle (radians) between each spectrum and ``ref``.

    Scale-invariant (keys on spectral *shape*, not brightness), so it separates
    a shiny/dark fragment from the dish by their differing spectral signatures
    rather than by intensity.
    """
    num = flat @ ref
    # np.linalg.norm(flat, axis=1) materializes a full (n_pixels, bands) squared
    # temporary before reducing it -- doubles peak memory and OOMs on large
    # multi-piece scans. einsum fuses the square+reduce into a per-row scalar.
    row_norms = np.sqrt(np.einsum("ij,ij->i", flat, flat))
    den = row_norms * (np.linalg.norm(ref) + eps) + eps
    return np.arccos(np.clip(num / den, -1.0, 1.0))


def euclidean_distance(flat: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Euclidean distance (all bands) between each spectrum and ``ref``.

    Unlike :func:`spectral_angle`, this is **magnitude sensitive**. That matters
    for wafer-vs-dish separation: bare silicon is dark but spectrally smooth, so
    its *angle* to a bright dish is small and SAM drops it, keeping only the
    interference-coloured SiO2 areas. Euclidean keeps the whole wafer because the
    dark substrate is far from the bright dish in magnitude, while still using all
    300 bands rather than collapsing to a brightness scalar.

    Uses the same fused einsum trick as ``spectral_angle`` to avoid materializing
    an (n_pixels, bands) temporary on multi-GB scans.
    """
    centered = flat - ref[None, :]
    return np.sqrt(np.einsum("ij,ij->i", centered, centered))


def _mahalanobis_to_background(flat: np.ndarray, bg_pixels: np.ndarray) -> np.ndarray:
    """Mahalanobis distance of every spectrum to the background distribution.

    Returns the **distance**, not the squared distance -- the einsum computes d^2
    and this takes the root. That matters for thresholding: d^2 over 300 bands
    spans ~2000x (background sits at d^2 ~= n_bands by the chi-square
    expectation, while specular/rim outliers reach ~5.6e5), and Otsu histograms
    that range into 256 bins, so nearly every pixel lands in bin 0. Taking the
    root compresses the tail. It is monotonic, so percentile thresholds are
    unchanged. See ``_threshold_mask`` for the chi-square rule that suits this
    map better than Otsu.
    """
    from sklearn.covariance import LedoitWolf
    mean = bg_pixels.mean(axis=0)
    precision = LedoitWolf().fit(bg_pixels).precision_
    centered = flat - mean
    d2 = np.einsum("ij,jk,ik->i", centered, precision, centered)
    return np.sqrt(np.maximum(d2, 0.0))


def foreground_distance(cube: np.ndarray, cfg: PieceConfig) -> np.ndarray:
    """(rows, cols) map of how spectrally unlike the background each pixel is.

    Backend chosen by ``cfg.method``: ``"sam"`` (default, cheap, robust, but
    scale-invariant so it keeps only spectrally *distinctive* regions),
    ``"euclidean"`` (magnitude sensitive -- use when the target is dark against a
    bright background, e.g. whole wafers incl. bare silicon vs a white dish),
    ``"mahalanobis"`` (accounts for background covariance), or ``"kmeans"``
    (2-cluster split, distance = |cluster assignment - background cluster|).
    """
    if cfg.flat_field:
        cube = flat_field_correct(cube, cfg.flat_field_sigma)
    rows, cols, bands = cube.shape
    flat = cube.reshape(-1, bands)

    if cfg.method == "sam":
        bg = np.median(_background_pixels(cube, cfg), axis=0)
        dist = spectral_angle(flat, bg)
    elif cfg.method == "euclidean":
        bg = np.median(_background_pixels(cube, cfg), axis=0)
        dist = euclidean_distance(flat, bg)
    elif cfg.method == "mahalanobis":
        dist = _mahalanobis_to_background(flat, _background_pixels(cube, cfg))
    elif cfg.method == "kmeans":
        from .segmentation import segment
        seg = segment(cube, invert=False, seed=cfg.seed)
        # segment() calls the larger cluster "substrate"; here substrate==background,
        # so foreground distance is simply the foreground membership as {0,1}.
        dist = seg.foreground.reshape(-1).astype(np.float64)
    else:
        raise ValueError(f"unknown piece method: {cfg.method!r}")
    return dist.reshape(rows, cols)


def _threshold_mask(dist: np.ndarray, cfg: PieceConfig,
                    n_bands: Optional[int] = None) -> np.ndarray:
    """Binarize the distance map into a raw foreground mask.

    ``threshold="chi2"`` is the statistically-grounded rule for the Mahalanobis
    backend: if the background is multivariate normal, its squared distance is
    chi-square with ``n_bands`` degrees of freedom, so anything past a high
    quantile is not background. Otsu is a poor fit for that map -- it assumes a
    bimodal histogram, whereas Mahalanobis gives a tight background mode plus a
    long continuous tail, and Otsu ends up isolating only the extreme tail
    (measured: 0.24% of pixels, zero surviving pieces).
    """
    if cfg.method == "kmeans":
        return dist > 0.5   # already {0,1}
    if cfg.threshold == "chi2":
        if n_bands is None:
            raise ValueError("threshold='chi2' needs n_bands")
        from scipy.stats import chi2
        # dist is the (unsquared) Mahalanobis distance -> compare against sqrt.
        t = float(np.sqrt(chi2.ppf(cfg.chi2_quantile, df=n_bands)))
    elif cfg.threshold == "otsu":
        from skimage.filters import threshold_otsu
        t = threshold_otsu(dist)
    else:
        t = np.percentile(dist, cfg.threshold_percentile)
    return dist > t


# --------------------------------------------------------------------------
# Mask cleanup + connected components
# --------------------------------------------------------------------------

def clean_mask(mask: np.ndarray, cfg: PieceConfig) -> np.ndarray:
    """Opening (drop thin arcs/dust) -> closing (merge device gaps) -> fill holes."""
    out = mask
    if cfg.open_iter > 0:
        out = ndi.binary_opening(out, iterations=cfg.open_iter)
    if cfg.close_iter > 0:
        out = ndi.binary_closing(out, iterations=cfg.close_iter)
    if cfg.fill_holes:
        out = ndi.binary_fill_holes(out)
    return out


def _erode_analysis_mask(mask: np.ndarray, cfg: PieceConfig) -> np.ndarray:
    """Shrink a piece's analysis mask inward by ``cfg.erode_iter`` pixels.

    Piece-boundary pixels mix film and dish spectra, so they dominate the anomaly
    flags unless dropped. Applied after the bbox is fixed, so the crop is
    unchanged -- only which pixels count as film. If erosion would erase the piece
    entirely (small fragments), the un-eroded mask is kept rather than losing a
    specimen from the study.
    """
    if cfg.erode_iter <= 0:
        return mask
    eroded = ndi.binary_erosion(mask, iterations=cfg.erode_iter)
    return eroded if eroded.any() else mask


def component_sizes(labels: np.ndarray) -> np.ndarray:
    """Pixel count per label id. ``sizes[i]`` = size of label ``i`` (0 = background)."""
    return np.bincount(labels.ravel())


def label_pieces(mask: np.ndarray, cfg: PieceConfig) -> Tuple[np.ndarray, List[int]]:
    """Label connected components, keeping only those >= ``min_area``.

    With ``watershed_split`` the mask is first split on its distance transform so
    fragments that touch are separated; otherwise plain connectivity labeling is
    used. Returns ``(labels, kept_ids)``.
    """
    if cfg.watershed_split:
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max
        dt = ndi.distance_transform_edt(mask)
        coords = peak_local_max(dt, labels=mask, min_distance=max(5, int(np.sqrt(cfg.min_area) / 2)))
        markers = np.zeros(mask.shape, dtype=int)
        for i, (r, c) in enumerate(coords, start=1):
            markers[r, c] = i
        labels = watershed(-dt, markers, mask=mask)
    else:
        labels, _ = ndi.label(mask)

    sizes = component_sizes(labels)
    kept = [lbl for lbl in range(1, sizes.size) if sizes[lbl] >= cfg.min_area]
    return labels, kept


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def extract_pieces(cube: Cube, cfg: PieceConfig,
                   valid_mask: Optional[np.ndarray] = None) -> List[Piece]:
    """Split a multi-piece :class:`Cube` into individual :class:`Piece` crops.

    ``valid_mask`` optionally excludes pixels (e.g. saturated) from the
    foreground mask. Pieces are returned largest-area first and get ids
    ``"<source_label>_p01"``, ``"<source_label>_p02"``, ...
    """
    cfg.validate()
    data = cube.data

    dist = foreground_distance(data, cfg)
    mask = _threshold_mask(dist, cfg, n_bands=data.shape[-1])
    if valid_mask is not None:
        mask &= valid_mask
    mask = clean_mask(mask, cfg)

    labels, kept = label_pieces(mask, cfg)
    # Largest piece first for stable, human-friendly ids.
    kept = sorted(kept, key=lambda l: int((labels == l).sum()), reverse=True)

    pieces: List[Piece] = []
    for i, lbl in enumerate(kept, start=1):
        comp = labels == lbl
        rows = np.any(comp, axis=1)
        cols = np.any(comp, axis=0)
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        r1 += 1
        c1 += 1
        pieces.append(Piece(
            data=data[r0:r1, c0:c1, :].copy(),
            mask=_erode_analysis_mask(comp[r0:r1, c0:c1].copy(), cfg),
            material=cube.material,
            piece_id=f"{cube.label}_p{i:02d}",
            source_label=cube.label,
            bbox=(int(r0), int(r1), int(c0), int(c1)),
            wavelengths=cube.wavelengths,
        ))
    return pieces
