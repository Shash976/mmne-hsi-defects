"""Step 10 -- Optical Density.

    OD = -ln(I / I0)

turns the preprocessed intensity/reflectance cube into quantitative
absorbance, where I0 is the incident/reference signal. The document defines I0
as the bare support (Si3N4 window) in a transmission geometry; here I0 is
**configurable**, defaulting to the faithful substrate-referenced version.

I0 options (``PreprocessConfig.od_method``):

- ``"substrate"`` (default) -- mean spectrum over the segmented bare-substrate
  pixels of this same cube. Substrate OD ~ 0 by construction; the film shows up
  as positive absorbance. Works on reflectance data with no separate reference.
- ``"white"``      -- the white-reference mean spectrum as I0.
- ``"reference_scan"`` -- a separate bare-support cube's mean spectrum, i.e. the
  true transmission I0 for the forthcoming Si3N4-window TMD scans.
- ``"none"``       -- skip OD, pass the reflectance cube through unchanged
  (regression/comparison against the pre-OD image).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import PreprocessConfig
from .io import Cube, load_reference_spectrum
from .preprocessing import Preprocessed


def optical_density(cube: np.ndarray, i0: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """``-ln(I / I0)`` with the ratio floored at ``eps`` to keep the log finite.

    ``i0`` is a per-band spectrum (n_bands,) broadcast over the image, or a full
    (rows, cols, bands) reference for a per-pixel I0.
    """
    i0 = np.asarray(i0, dtype=np.float64)
    if i0.ndim == 1:
        i0 = i0[None, None, :]
    ratio = cube / np.where(np.abs(i0) < eps, eps, i0)
    ratio = np.clip(ratio, eps, None)
    return -np.log(ratio)


# --------------------------------------------------------------------------
# I0 builders
# --------------------------------------------------------------------------

def i0_from_substrate(cube: np.ndarray, substrate_mask: np.ndarray) -> np.ndarray:
    """Mean spectrum over bare-substrate pixels (the default reference)."""
    if substrate_mask.sum() == 0:
        raise ValueError("substrate mask is empty; cannot build I0 from substrate")
    return cube[substrate_mask].mean(axis=0)


def i0_from_white(white_mean: np.ndarray) -> np.ndarray:
    """White-reference mean spectrum as I0."""
    return np.asarray(white_mean, dtype=np.float64)


def i0_from_reference_scan(reference_cube: Cube,
                           mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Mean spectrum of a separate bare-support scan (true-transmission I0).

    ``mask`` optionally restricts the average to genuinely bare pixels of the
    reference scan.
    """
    flat = reference_cube.data.reshape(-1, reference_cube.n_bands)
    if mask is not None:
        flat = reference_cube.data[mask]
    return flat.mean(axis=0)


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

def to_optical_density(pre: Preprocessed, cfg: PreprocessConfig,
                       white_ref_hdr: Optional[str] = None,
                       reference_cube: Optional[Cube] = None):
    """Apply Step 10 to a :class:`Preprocessed` cube per ``cfg.od_method``.

    Returns ``(od_cube, i0)``. For ``od_method="none"`` the reflectance cube is
    returned unchanged and ``i0`` is ``None``.
    """
    method = cfg.od_method
    if method == "none":
        return pre.data, None

    if method == "substrate":
        i0 = i0_from_substrate(pre.data, pre.segmentation.substrate)
    elif method == "white":
        if not white_ref_hdr:
            raise ValueError("od_method='white' requires a white reference header")
        white_mean, _ = load_reference_spectrum(white_ref_hdr)
        i0 = i0_from_white(white_mean)
    elif method == "reference_scan":
        if reference_cube is None:
            raise ValueError("od_method='reference_scan' requires a reference cube")
        i0 = i0_from_reference_scan(reference_cube)
    else:
        raise ValueError(f"unknown od_method: {method!r}")

    od = optical_density(pre.data, i0, eps=cfg.eps)
    return od, i0
