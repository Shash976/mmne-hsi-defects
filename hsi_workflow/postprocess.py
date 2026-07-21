"""Stage 9 -- Spatial postprocessing of pixel-level maps.

Pixel-wise clustering/anomaly maps contain isolated noisy pixels. A median
filter + morphological opening + connected-component size filter removes those
speckles so what remains is contiguous regions worth characterizing (Stage 10).
Real features survive as large connected components; noise (single pixels) does
not.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from config import PostprocConfig


def clean_binary_map(flag: np.ndarray, cfg: PostprocConfig) -> np.ndarray:
    """Denoise a boolean flag map: median smooth -> opening -> drop tiny blobs.

    ``flag`` is the (rows, cols) boolean "anomalous / this-cluster" mask. Returns
    a cleaned boolean map of the same shape.
    """
    cfg.validate()
    out = flag.copy()

    if cfg.median_size and cfg.median_size > 1:
        out = ndi.median_filter(out.astype(np.uint8), size=cfg.median_size).astype(bool)

    if cfg.opening_radius and cfg.opening_radius > 0:
        from skimage.morphology import opening, disk
        out = opening(out, disk(cfg.opening_radius))

    if cfg.min_component and cfg.min_component > 0:
        labels, n = ndi.label(out)
        if n:
            sizes = ndi.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1))
            too_small = np.where(sizes < cfg.min_component)[0] + 1
            out[np.isin(labels, too_small)] = False
    return out


def label_regions(clean_flag: np.ndarray):
    """Connected-component labeling of a cleaned flag map -> ``(labels, n)``."""
    return ndi.label(clean_flag)
