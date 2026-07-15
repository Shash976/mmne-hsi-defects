"""Substrate vs. film segmentation.

Not an explicit box in the document, but the faithful optical-density step
(Step 10, ``od_method="substrate"``) needs to know which pixels are bare
substrate so their mean can serve as I0. A 2-cluster K-means on the pixel
spectra separates the deposited film/pattern from the surrounding substrate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class Segmentation:
    """Boolean masks over the (rows, cols) grid."""

    foreground: np.ndarray   # film / deposited pattern
    substrate: np.ndarray    # bare substrate (used as the OD reference region)


def segment(cube: np.ndarray, valid_mask: np.ndarray | None = None,
            invert: bool = False, seed: int = 0) -> Segmentation:
    """Split a cube's pixels into film (foreground) vs. substrate.

    Two-cluster K-means over the spectra. The **larger-area** cluster is taken
    to be the substrate (bare support usually fills more of the frame than the
    deposited pattern). Set ``invert=True`` when a crop sits almost entirely
    inside the film so that heuristic is backwards (the analogous LIG ROI-2
    caveat). ``valid_mask`` excludes pixels (e.g. saturated) from both the fit
    and the output masks.
    """
    rows, cols, bands = cube.shape
    flat = cube.reshape(-1, bands)
    if valid_mask is None:
        valid_flat = np.ones(rows * cols, dtype=bool)
    else:
        valid_flat = valid_mask.ravel()

    labels_full = np.full(rows * cols, -1, dtype=int)
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(flat[valid_flat])
    labels_full[valid_flat] = km.labels_

    counts = np.bincount(km.labels_, minlength=2)
    substrate_label = int(np.argmax(counts))   # larger cluster = substrate
    if invert:
        substrate_label = 1 - substrate_label
    foreground_label = 1 - substrate_label

    foreground = (labels_full == foreground_label).reshape(rows, cols)
    substrate = (labels_full == substrate_label).reshape(rows, cols)
    return Segmentation(foreground=foreground, substrate=substrate)
