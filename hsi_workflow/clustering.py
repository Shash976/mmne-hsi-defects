"""Stages 6-7 -- Unsupervised clustering + spatial mapping.

Replaces the old reference-spectra + linear-unmixing step. We cluster the PCA
scores to find naturally occurring spectral populations, then project the labels
back onto the image to get a cluster map. Clusters are *spectral populations
only* -- we deliberately do not name them "vacancy", "crack", etc.

Adding a new algorithm = adding one entry to ``_CLUSTERERS``; the orchestrator
and CLIs are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from .config import ClusterConfig


@dataclass
class ClusterResult:
    """Labels + fit metadata. ``-1`` denotes noise/unassigned (DBSCAN)."""

    labels: np.ndarray               # (n_samples,) int cluster id per sample
    method: str
    n_clusters: int


# --------------------------------------------------------------------------
# Registry: name -> function(features, cfg) -> labels
# --------------------------------------------------------------------------

def _kmeans(features: np.ndarray, cfg: ClusterConfig) -> np.ndarray:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=cfg.n_clusters, n_init=10, random_state=cfg.seed)
    return km.fit_predict(features)


def _dbscan(features: np.ndarray, cfg: ClusterConfig) -> np.ndarray:
    from sklearn.cluster import DBSCAN
    db = DBSCAN(eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples)
    return db.fit_predict(features)


def _gmm(features: np.ndarray, cfg: ClusterConfig) -> np.ndarray:
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=cfg.n_clusters, random_state=cfg.seed)
    return gm.fit_predict(features)


_CLUSTERERS: Dict[str, Callable[[np.ndarray, ClusterConfig], np.ndarray]] = {
    "kmeans": _kmeans,
    "dbscan": _dbscan,
    "gmm": _gmm,
}


def cluster(features: np.ndarray, cfg: ClusterConfig) -> ClusterResult:
    """Cluster (n_samples, n_features) rows -- usually PCA scores -- per ``cfg``."""
    cfg.validate()
    labels = _CLUSTERERS[cfg.method](features, cfg)
    n = len(set(labels.tolist()) - {-1})
    return ClusterResult(labels=labels, method=cfg.method, n_clusters=n)


def cluster_map(result: ClusterResult, shape, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Reshape flat labels back to a (rows, cols) map.

    ``mask`` (foreground) is where the labels apply; off-mask pixels get ``-1``.
    When given, ``len(result.labels)`` must equal ``mask.sum()``.
    """
    rows, cols = shape
    out = np.full(rows * cols, -1, dtype=int)
    if mask is None:
        out[:] = result.labels
    else:
        out[mask.reshape(-1)] = result.labels
    return out.reshape(rows, cols)


# --------------------------------------------------------------------------
# Cluster-quality metrics (Stage 7 deliverable)
# --------------------------------------------------------------------------

def cluster_metrics(features: np.ndarray, labels: np.ndarray,
                    max_samples: int = 20_000, seed: int = 0) -> dict:
    """Silhouette / Davies-Bouldin / Calinski-Harabasz on a subsample.

    Metrics need >=2 clusters and are O(n^2) (silhouette), hence the subsample.
    Noise points (label -1) are dropped before scoring.
    """
    from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                                 calinski_harabasz_score)
    keep = labels != -1
    X, y = features[keep], labels[keep]
    if len(set(y.tolist())) < 2:
        return {"silhouette": float("nan"), "davies_bouldin": float("nan"),
                "calinski_harabasz": float("nan"), "n_clusters": len(set(y.tolist()))}
    if X.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], max_samples, replace=False)
        X, y = X[idx], y[idx]
    return {
        "silhouette": float(silhouette_score(X, y)),
        "davies_bouldin": float(davies_bouldin_score(X, y)),
        "calinski_harabasz": float(calinski_harabasz_score(X, y)),
        "n_clusters": len(set(y.tolist())),
    }


def compare_methods(features: np.ndarray, cfg: ClusterConfig,
                    methods: tuple = ("kmeans", "dbscan", "gmm"),
                    max_samples: int = 20_000, seed: int = 0) -> dict:
    """Stage 7's "compare cluster stability": run several algorithms, score each.

    Runs every method in ``methods`` on the same (subsampled) features, computes
    each one's quality metrics, and the pairwise Adjusted Rand Index between the
    label assignments -- high ARI means two methods find the *same* spectral
    populations, i.e. the structure is stable rather than an algorithmic artifact.

    Returns ``{"per_method": {name: metrics}, "pairwise_ari": {"a|b": ari}}``.
    """
    from dataclasses import replace
    from sklearn.metrics import adjusted_rand_score

    X = features
    if X.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        X = X[rng.choice(X.shape[0], max_samples, replace=False)]

    labels: Dict[str, np.ndarray] = {}
    per_method = {}
    for m in methods:
        res = cluster(X, replace(cfg, method=m))
        labels[m] = res.labels
        per_method[m] = cluster_metrics(X, res.labels, max_samples=max_samples, seed=seed)

    pairwise = {}
    names = list(methods)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            # ARI over points labeled by both (DBSCAN noise -1 dropped).
            keep = (labels[a] != -1) & (labels[b] != -1)
            pairwise[f"{a}|{b}"] = (float(adjusted_rand_score(labels[a][keep], labels[b][keep]))
                                    if keep.sum() > 1 else float("nan"))
    return {"per_method": per_method, "pairwise_ari": pairwise}
