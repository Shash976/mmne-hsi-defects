"""Stage 5 -- Dimensionality reduction (PCA).

PCA answers "is there anything interesting before ML?" and provides the compact
feature space (PC1..PCk) that clustering (Stage 6) and, optionally, anomaly
scoring (Stage 8) run on. The model is fit once on a pooled, subsampled set of
spectra so a single consistent basis is shared across pieces/images, then every
pixel or ROI is projected into it.

Everything here operates on the full-band spectra handed in by preprocessing --
no RGB reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

from .config import PcaConfig


@dataclass
class PcaModel:
    """A fitted PCA plus the metadata the deliverables need.

    ``transform`` projects (n, bands) spectra to (n, n_components) scores;
    ``score_image`` reshapes a cube's per-pixel scores back to an image stack.
    """

    pca: PCA
    n_components: int

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_

    @property
    def loadings(self) -> np.ndarray:
        """(n_components, bands) -- how each PC weights the original wavelengths."""
        return self.pca.components_

    def transform(self, flat_spectra: np.ndarray) -> np.ndarray:
        return self.pca.transform(flat_spectra)

    def score_image(self, cube: np.ndarray) -> np.ndarray:
        """Project every pixel -> (rows, cols, n_components) PC score maps."""
        rows, cols, bands = cube.shape
        scores = self.pca.transform(cube.reshape(-1, bands))
        return scores.reshape(rows, cols, self.n_components)


def _subsample(flat: np.ndarray, cap: int, seed: int) -> np.ndarray:
    if flat.shape[0] <= cap:
        return flat
    rng = np.random.default_rng(seed)
    return flat[rng.choice(flat.shape[0], cap, replace=False)]


def fit_pca(flat_spectra: np.ndarray, cfg: PcaConfig) -> PcaModel:
    """Fit PCA on pooled spectra (subsampled to ``cfg.max_fit_pixels``).

    Pass the pooled foreground spectra of all pieces/images so the PCs describe
    variation across the whole dataset, not one image.
    """
    cfg.validate()
    fit_data = _subsample(np.asarray(flat_spectra, dtype=np.float64),
                          cfg.max_fit_pixels, cfg.seed)
    pca = PCA(n_components=cfg.n_components, whiten=cfg.whiten,
              random_state=cfg.seed).fit(fit_data)
    return PcaModel(pca=pca, n_components=cfg.n_components)
