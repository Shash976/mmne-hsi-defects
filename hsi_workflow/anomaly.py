"""Stage 8 -- Anomaly scoring.

The scientific core of the revised objective: compare every spectrum (pixel or
ROI) against what "normal" looks like and emit a continuous anomaly score. The
pipeline produces **two** notions of "normal", both label-free:

- **within-film** (``AnomalyConfig.fit_on="self"``, the default): detectors are
  fit on the target's own majority population, so high scores mark small,
  localized regions that differ from the bulk film. This drives the flagged
  regions and region tables (the document's "small localized anomalies").
- **silicon-baseline contrast** (always computed): a Mahalanobis detector fit on
  the bare-silicon control population scores every pixel by how unlike silicon
  it is -- the document's literal "relative to a spectrally homogeneous silicon
  baseline" deliverable. Because silicon and SiO2 are different materials this
  map is a material-contrast map, not a defect screen; it is reported alongside,
  never used to flag regions unless ``fit_on="baseline"`` is chosen explicitly.

Each detector implements the same tiny protocol -- ``fit(normal) -> self`` and
``score(X) -> higher-is-more-anomalous`` -- and is registered in ``_DETECTORS``.
Adding a method is one function + one registry entry.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from .config import AnomalyConfig


def _subsample(X: np.ndarray, cap: int, seed: int) -> np.ndarray:
    if X.shape[0] <= cap:
        return X
    rng = np.random.default_rng(seed)
    return X[rng.choice(X.shape[0], cap, replace=False)]


# --------------------------------------------------------------------------
# Detectors: each has .fit(normal_X) and .score(X) (higher = more anomalous)
# --------------------------------------------------------------------------

class MahalanobisDetector:
    """Distance from the baseline mean under a Ledoit-Wolf shrinkage covariance.

    This is the RX detector from ``legacy/unsupervised_defect.py``, generalized:
    fit on the normal (silicon) spectra, score anything. Shrinkage keeps the
    covariance invertible when bands outnumber samples / are collinear.
    """

    def fit(self, normal_X: np.ndarray) -> "MahalanobisDetector":
        from sklearn.covariance import LedoitWolf
        self.mean_ = normal_X.mean(axis=0)
        self.precision_ = LedoitWolf().fit(normal_X).precision_
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        c = X - self.mean_
        return np.einsum("ij,jk,ik->i", c, self.precision_, c)


class IForestDetector:
    """sklearn IsolationForest; score = negative of ``score_samples`` (higher = odd)."""

    def __init__(self, contamination: float, seed: int):
        from sklearn.ensemble import IsolationForest
        self.model = IsolationForest(contamination=contamination, random_state=seed)

    def fit(self, normal_X: np.ndarray) -> "IForestDetector":
        self.model.fit(normal_X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(X)


class LOFDetector:
    """Local Outlier Factor in novelty mode (fit on normal, score new points)."""

    def __init__(self, contamination: float):
        from sklearn.neighbors import LocalOutlierFactor
        self.model = LocalOutlierFactor(novelty=True, contamination=contamination)

    def fit(self, normal_X: np.ndarray) -> "LOFDetector":
        self.model.fit(normal_X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(X)


class OCSVMDetector:
    """One-Class SVM; score = negative signed distance to the boundary."""

    def __init__(self, contamination: float):
        from sklearn.svm import OneClassSVM
        self.model = OneClassSVM(nu=min(0.5, max(1e-3, contamination)), gamma="scale")

    def fit(self, normal_X: np.ndarray) -> "OCSVMDetector":
        self.model.fit(normal_X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_function(X).ravel() if hasattr(self.model, "score_function") \
            else -self.model.decision_function(X).ravel()


# Registry of known detectors. Add a method = add a class + one entry here.
_DETECTORS: Dict[str, Callable] = {
    "mahalanobis": lambda cfg: MahalanobisDetector(),
    "iforest": lambda cfg: IForestDetector(cfg.contamination, cfg.seed),
    "lof": lambda cfg: LOFDetector(cfg.contamination),
    "ocsvm": lambda cfg: OCSVMDetector(cfg.contamination),
}


def _make_detector(name: str, cfg: AnomalyConfig):
    if name not in _DETECTORS:
        raise ValueError(f"unknown anomaly method: {name!r}")
    return _DETECTORS[name](cfg)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def fit_detectors(normal_X: np.ndarray, cfg: AnomalyConfig) -> Dict[str, object]:
    """Fit every detector in ``cfg.methods`` on the normal (baseline) spectra."""
    cfg.validate()
    fit_X = _subsample(np.asarray(normal_X, dtype=np.float64), cfg.max_fit_pixels, cfg.seed)
    return {name: _make_detector(name, cfg).fit(fit_X) for name in cfg.methods}


def anomaly_map(scores: np.ndarray, shape, mask: Optional[np.ndarray] = None,
                fill: float = np.nan) -> np.ndarray:
    """Reshape per-pixel scores to a (rows, cols) heatmap; off-mask = ``fill``."""
    rows, cols = shape
    out = np.full(rows * cols, fill, dtype=np.float64)
    if mask is None:
        out[:] = scores
    else:
        out[mask.reshape(-1)] = scores
    return out.reshape(rows, cols)


def flag_threshold(normal_scores: np.ndarray, percentile: float) -> float:
    """Flagging threshold = a high percentile of the normal-population scores.

    Anything above this (learned purely from the population the detector was
    fit on) is flagged anomalous, keeping the flag rate low and interpretable.
    """
    return float(np.percentile(normal_scores, percentile))


def to_probability(scores: np.ndarray, lo_pct: float = 1.0,
                   hi_pct: float = 99.0) -> np.ndarray:
    """Rescale raw detector scores to a 0-1 anomaly probability map (Stage 10).

    Percentile-clipped min-max over the *finite* scores, so a handful of extreme
    outliers can't compress the rest of the map to zero. NaNs (off-mask pixels)
    pass through unchanged. This is a display/reporting normalization -- ranking
    is preserved; it is not a calibrated statistical probability.
    """
    out = np.asarray(scores, dtype=np.float64).copy()
    finite = np.isfinite(out)
    if not finite.any():
        return out
    lo = np.percentile(out[finite], lo_pct)
    hi = np.percentile(out[finite], hi_pct)
    if hi <= lo:
        hi = lo + 1e-12
    out[finite] = np.clip((out[finite] - lo) / (hi - lo), 0.0, 1.0)
    return out
