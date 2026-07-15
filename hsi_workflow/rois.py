"""ROI tiling -- turn each piece into fixed patches (the cross-specimen ML table).

The document warns that training on every pixel massively overstates performance
because neighbouring pixels are near-identical (spatial autocorrelation = data
leakage). The fix is to make the *ROI* the unit of analysis: tile each piece into
fixed patches, average each patch to one spectrum, and treat that as one ML
sample. Patches are organized hierarchically (specimen -> image -> ROI) and the
train/test split holds out whole specimens so no ROI leaks across the split.

This module runs on the :class:`~hsi_workflow.pieces.Piece` sub-cubes produced by
Stage 3.1 -- i.e. ROIs are extracted *from each individual piece*, not from the
raw multi-piece scan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .config import RoiConfig
from .pieces import Piece


@dataclass
class Roi:
    """One fixed patch and its per-ROI features (all from full spectra).

    ``specimen`` is the piece id (the hold-out unit for splitting); ``image`` is
    the source scan label. ``bbox`` is ``(r0, r1, c0, c1)`` within the piece.
    Feature fields beyond the mean spectrum (``pca``, ``anomaly``) are filled in
    later by the PCA / anomaly stages.
    """

    roi_id: str
    specimen: str
    image: str
    material: str
    bbox: tuple
    coverage: float
    mean_spectrum: np.ndarray
    std: float
    spectral_variance: float
    pca: Optional[np.ndarray] = None
    anomaly: dict = field(default_factory=dict)


def tile_rois(piece: Piece, cfg: RoiConfig) -> List[Roi]:
    """Tile one piece into ROIs, keeping only patches well inside the mask.

    A patch is accepted when at least ``cfg.min_coverage`` of its pixels are in
    the piece mask, so ROIs never straddle the dish or the piece edge. Each ROI's
    features come from the mean over its in-mask pixels.
    """
    cfg.validate()
    data = piece.data
    mask = piece.mask
    rows, cols, bands = data.shape
    p, s = cfg.patch, cfg.stride

    rois: List[Roi] = []
    idx = 0
    for r0 in range(0, rows - p + 1, s):
        for c0 in range(0, cols - p + 1, s):
            r1, c1 = r0 + p, c0 + p
            sub_mask = mask[r0:r1, c0:c1]
            coverage = float(sub_mask.mean())
            if coverage < cfg.min_coverage:
                continue
            sub = data[r0:r1, c0:c1, :]
            spectra = sub[sub_mask]                       # (n_in_mask, bands)
            mean_spec = spectra.mean(axis=0)
            idx += 1
            rois.append(Roi(
                roi_id=f"{piece.piece_id}_r{idx:04d}",
                specimen=piece.piece_id,
                image=piece.source_label,
                material=piece.material,
                bbox=(int(r0), int(r1), int(c0), int(c1)),
                coverage=coverage,
                mean_spectrum=mean_spec,
                std=float(spectra.std()),
                # spectral variance = mean over bands of the per-band variance
                # across the patch's pixels; a scalar "how heterogeneous is this ROI".
                spectral_variance=float(spectra.var(axis=0).mean()),
            ))
    return rois


def roi_feature_matrix(rois: Sequence[Roi]) -> np.ndarray:
    """(n_rois, bands) stack of ROI mean spectra -- the input to PCA/anomaly."""
    return np.vstack([r.mean_spectrum for r in rois])


def build_roi_table(rois: Sequence[Roi],
                    wavelengths: Optional[np.ndarray] = None):
    """Assemble the ROIs into a tidy pandas DataFrame (the document's ML table).

    Columns: ids/metadata (``roi_id, specimen, image, material``), the bbox,
    ``coverage``, scalar features (``std``, ``spectral_variance``), any populated
    ``pca_1..k`` and ``anomaly_<method>`` columns, and the mean spectrum expanded
    into per-band columns (``m0000`` ... named by wavelength when available) so
    the table is self-contained and parquet-friendly.
    """
    import pandas as pd

    base = []
    for r in rois:
        row = {
            "roi_id": r.roi_id, "specimen": r.specimen, "image": r.image,
            "material": r.material, "r0": r.bbox[0], "r1": r.bbox[1],
            "c0": r.bbox[2], "c1": r.bbox[3], "coverage": r.coverage,
            "std": r.std, "spectral_variance": r.spectral_variance,
        }
        if r.pca is not None:
            for i, v in enumerate(r.pca, start=1):
                row[f"pca_{i}"] = float(v)
        for k, v in r.anomaly.items():
            row[f"anomaly_{k}"] = float(v)
        base.append(row)
    df = pd.DataFrame(base)

    spectra = roi_feature_matrix(rois)
    if wavelengths is not None and len(wavelengths) == spectra.shape[1]:
        band_cols = [f"m{int(round(w))}nm" for w in wavelengths]
    else:
        band_cols = [f"m{i:04d}" for i in range(spectra.shape[1])]
    df = pd.concat([df, pd.DataFrame(spectra, columns=band_cols, index=df.index)], axis=1)
    return df


def split_by_specimen(df, test_fraction: float = 0.3, seed: int = 0):
    """Hold out whole specimens (pieces) for the test set -- no ROI leakage.

    Every ROI of a chosen specimen goes entirely to train or entirely to test,
    which is the leakage-free evaluation the document argues for. Returns
    ``(train_df, test_df)``.
    """
    rng = np.random.default_rng(seed)
    specimens = np.array(sorted(df["specimen"].unique()))
    rng.shuffle(specimens)
    n_test = max(1, int(round(len(specimens) * test_fraction)))
    test_specimens = set(specimens[:n_test].tolist())
    is_test = df["specimen"].isin(test_specimens)
    return df[~is_test].copy(), df[is_test].copy()
