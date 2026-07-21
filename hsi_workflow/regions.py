"""Stages 10-11 -- Quantitative maps + region characterization.

Turns the cleaned anomaly map into a table of measured regions. Per the revised
objective we never label a region ("this is a vacancy"); we *describe* it --
area, shape, spectrum, variance, distance from the silicon baseline, anomaly
score. Downstream that table is what tells you whether anomalies are localized,
repeated, near edges, etc., and which regions merit follow-up SEM/AFM/Raman.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from anomaly import MahalanobisDetector


@dataclass
class RegionStats:
    """One characterized region (a connected anomalous component)."""

    region_id: int
    area: int
    perimeter: float
    compactness: float               # 4*pi*area / perimeter^2 (1.0 = perfect disk)
    centroid: tuple
    mean_reflectance: float          # physical reflectance (from the pre-SNV band mean)
    mean_snv: float                  # mean of the SNV analysis values (~0 by construction)
    spectral_variance: float
    baseline_distance: float         # Mahalanobis distance of region mean to Si baseline
    mean_anomaly: float
    mean_spectrum: np.ndarray
    pca: Optional[np.ndarray] = None  # PCA coordinates of the region mean spectrum


def spectral_distance_map(cube: np.ndarray, reference_spectrum: np.ndarray,
                          precision: Optional[np.ndarray] = None) -> np.ndarray:
    """(rows, cols) distance of each pixel spectrum from a reference.

    With ``precision`` (from a fitted baseline covariance) this is Mahalanobis
    distance; without it, plain Euclidean. Used for the "spectral distance map"
    deliverable and for per-region baseline distances.
    """
    rows, cols, bands = cube.shape
    flat = cube.reshape(-1, bands)
    c = flat - reference_spectrum
    if precision is None:
        d = np.sqrt((c ** 2).sum(axis=1))
    else:
        d = np.einsum("ij,jk,ik->i", c, precision, c)
    return d.reshape(rows, cols)


def characterize_regions(labels: np.ndarray, n: int, cube: np.ndarray,
                         anomaly_map: np.ndarray,
                         baseline_detector: Optional[MahalanobisDetector] = None,
                         reflectance_mean: Optional[np.ndarray] = None,
                         pca=None) -> List[RegionStats]:
    """Measure every labeled region against the analysis cube + anomaly map.

    ``cube`` is the preprocessed analysis cube (rows, cols, bands); ``labels``/``n``
    come from :func:`~hsi_workflow.postprocess.label_regions`. ``baseline_detector``
    (fit on silicon) supplies the Mahalanobis distance-from-baseline;
    ``reflectance_mean`` (the piece's pre-SNV band-mean image) supplies physical
    mean reflectance; ``pca`` (a fitted PcaModel) supplies each region's PCA
    coordinates. Absent inputs yield NaN fields.
    """
    from skimage.measure import regionprops

    props = {p.label: p for p in regionprops(labels)}
    out: List[RegionStats] = []
    for rid in range(1, n + 1):
        comp = labels == rid
        area = int(comp.sum())
        if area == 0:
            continue
        spectra = cube[comp]                       # (area, bands)
        mean_spec = spectra.mean(axis=0)
        p = props.get(rid)
        perim = float(p.perimeter) if p is not None and p.perimeter > 0 else float("nan")
        compact = (4 * np.pi * area / (perim ** 2)) if perim and np.isfinite(perim) else float("nan")
        centroid = tuple(map(float, p.centroid)) if p is not None else (float("nan"),) * 2

        baseline_dist = float("nan")
        if baseline_detector is not None:
            baseline_dist = float(baseline_detector.score(mean_spec[None, :])[0])

        mean_refl = float("nan")
        if reflectance_mean is not None:
            mean_refl = float(np.nanmean(reflectance_mean[comp]))

        region_pca = None
        if pca is not None:
            region_pca = np.asarray(pca.transform(mean_spec[None, :])[0], dtype=np.float64)

        out.append(RegionStats(
            region_id=rid,
            area=area,
            perimeter=perim,
            compactness=float(compact),
            centroid=centroid,
            mean_reflectance=mean_refl,
            mean_snv=float(mean_spec.mean()),
            spectral_variance=float(spectra.var(axis=0).mean()),
            baseline_distance=baseline_dist,
            mean_anomaly=float(np.nanmean(anomaly_map[comp])),
            mean_spectrum=mean_spec,
            pca=region_pca,
        ))
    return out


def regions_to_table(regions: List[RegionStats]):
    """Tidy the region stats into a pandas DataFrame (the document's region table)."""
    import pandas as pd
    rows = []
    for r in regions:
        row = {
            "region_id": r.region_id, "area": r.area, "perimeter": r.perimeter,
            "compactness": r.compactness, "centroid_row": r.centroid[0],
            "centroid_col": r.centroid[1], "mean_reflectance": r.mean_reflectance,
            "mean_snv": r.mean_snv, "spectral_variance": r.spectral_variance,
            "baseline_distance": r.baseline_distance, "mean_anomaly": r.mean_anomaly,
        }
        if r.pca is not None:
            for i, v in enumerate(r.pca, start=1):
                row[f"pca_{i}"] = float(v)
        rows.append(row)
    # Fixed column set even when empty, so an artifact with zero regions is
    # still a valid, recognizable table (and stale-file detection is easy).
    columns = ["region_id", "area", "perimeter", "compactness", "centroid_row",
               "centroid_col", "mean_reflectance", "mean_snv", "spectral_variance",
               "baseline_distance", "mean_anomaly"]
    return pd.DataFrame(rows, columns=columns if not rows else None)
