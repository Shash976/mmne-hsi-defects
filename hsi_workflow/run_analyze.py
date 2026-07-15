"""CLI: Stages 5-11 -- full anomaly-detection analysis.

    python -m hsi_workflow.run_analyze --target sio2_dish_white_20
    python -m hsi_workflow.run_analyze --target lig --baseline lig    # test bed
    python -m hsi_workflow.run_analyze --target sio2_dish_black --cluster gmm --anomaly iforest lof mahalanobis

Fits PCA + anomaly detectors on the silicon baseline, analyzes every target
piece, and writes per-piece analysis maps + region tables, a PCA summary figure,
and the aggregated ROI table. Prints cluster metrics and anomaly fractions.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .config import DATASETS, WorkflowConfig, DEFAULT_BASELINE
from .pipeline import run_workflow
from .viz import save_pca_summary, save_analysis_figure

DEFAULT_OUT = os.path.join("out", "workflow", "analyze")


def build_cfg(args) -> WorkflowConfig:
    wf = WorkflowConfig()
    wf.pca.n_components = args.pca_components
    wf.cluster.method = args.cluster
    wf.cluster.n_clusters = args.n_clusters
    wf.anomaly.methods = args.anomaly
    wf.anomaly.fit_on = args.fit_on
    wf.anomaly.anomaly_percentile = args.anomaly_percentile
    return wf


def main():
    p = argparse.ArgumentParser(description="Stages 5-11 anomaly analysis.")
    p.add_argument("--target", default="sio2_dish_white_20", choices=sorted(DATASETS))
    p.add_argument("--baseline", default=DEFAULT_BASELINE, choices=sorted(DATASETS))
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--pca-components", type=int, default=3)
    p.add_argument("--cluster", default="kmeans", choices=["kmeans", "dbscan", "gmm"])
    p.add_argument("--n-clusters", type=int, default=4)
    p.add_argument("--anomaly", nargs="+", default=["iforest", "mahalanobis"],
                   choices=["iforest", "lof", "mahalanobis", "ocsvm"])
    p.add_argument("--fit-on", default="self", choices=["self", "baseline"],
                   help="'self': anomalies within the target film (default); "
                        "'baseline': contrast vs bare silicon.")
    p.add_argument("--anomaly-percentile", type=float, default=97.5)
    args = p.parse_args()

    wf = build_cfg(args)
    out_dir = os.path.join(args.out, args.target)
    os.makedirs(out_dir, exist_ok=True)

    res = run_workflow(args.target, wf, baseline=args.baseline)
    if not res.analyses:
        print("No target pieces analyzed.")
        return

    save_pca_summary(res.pca.explained_variance_ratio, res.pca.loadings,
                     res.analyses[0].piece.wavelengths, out_dir)

    primary = wf.anomaly.methods[0]
    threshold = res.baseline_thresholds[primary]
    print("\n{:<22} {:>10} {:>10} {:>12} {:>10}".format(
        "piece_id", "silhouette", "n_clust", "anom_frac", "n_regions"))
    for a in res.analyses:
        piece = a.piece
        save_analysis_figure(a, primary, threshold, out_dir)
        if len(a.region_table):
            a.region_table.to_csv(os.path.join(out_dir, f"{piece.piece_id}_regions.csv"), index=False)
        anom_frac = float(a.flagged.sum()) / max(1, int(piece.mask.sum()))
        print("{:<22} {:>10.3f} {:>10} {:>11.2%} {:>10}".format(
            piece.piece_id, a.cluster_metrics.get("silhouette", float("nan")),
            a.cluster_metrics.get("n_clusters", 0), anom_frac, len(a.regions)))

    if res.roi_table is not None:
        roi_csv = os.path.join(out_dir, "roi_table.csv")
        res.roi_table.to_csv(roi_csv, index=False)
        print(f"\nROI table ({len(res.roi_table)} rows) -> {roi_csv}")
    print(f"Figures + region tables under {out_dir}")
    print(f"Baseline anomaly thresholds ({primary}): {res.baseline_thresholds[primary]:.3f}")


if __name__ == "__main__":
    main()
