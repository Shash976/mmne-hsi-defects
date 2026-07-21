"""CLI: Stages 5-11 -- full anomaly-detection analysis.

    python -m hsi_workflow.run_analyze --target sio2_dish_white_20
    python -m hsi_workflow.run_analyze --target lig --baseline lig    # test bed
    python -m hsi_workflow.run_analyze --target sio2_dish_black --cluster gmm --anomaly iforest lof mahalanobis
    python -m hsi_workflow.run_analyze --target sio2_dish_black --compare-clusters

Fits PCA + anomaly detectors, analyzes every target piece, and writes:
per-piece 9-panel analysis maps (within-film anomaly + silicon-baseline
contrast + spectral distance + 0-1 probability), region tables (always, even if
empty), PCA summary + scatter, spectral histogram, the aggregated ROI table, a
specimen-level hold-out evaluation, an optional cluster-stability comparison,
and the Stage-11 ``report.md``.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from config import DATASETS, WorkflowConfig, DEFAULT_BASELINE, ORGANIZED_DATA_ROOT
from pipeline import run_workflow, pooled_foreground
from clustering import compare_methods
from anomaly import fit_detectors
from rois import split_by_specimen
from report import write_report
from viz import save_pca_summary, save_analysis_figure, save_pca_scatter, save_spectral_histogram

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


def _save_cluster_comparison(res, wf, out_dir: str) -> None:
    """Stage 7 stability check: same features, several algorithms, one CSV."""
    import pandas as pd
    pooled = pooled_foreground([a.piece for a in res.analyses],
                               wf.cluster.max_fit_pixels, wf.cluster.seed)
    feats = res.pca.transform(pooled)
    cmp = compare_methods(feats, wf.cluster)
    rows = [{"method": m, **metrics} for m, metrics in cmp["per_method"].items()]
    rows += [{"method": pair, "adjusted_rand_index": ari}
             for pair, ari in cmp["pairwise_ari"].items()]
    path = os.path.join(out_dir, "cluster_comparison.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Cluster stability comparison -> {path}")
    for pair, ari in cmp["pairwise_ari"].items():
        print(f"  ARI {pair}: {ari:.3f}")


def _save_roi_evaluation(roi_table, wf, out_dir: str) -> None:
    """Leakage-free check: fit on train specimens' ROIs, score held-out ones.

    Uses the ROI PCA features and the primary detector. Whole specimens go
    entirely to train or test (the document's split), so held-out scores
    estimate generalization to *new* pieces.
    """
    import pandas as pd
    pca_cols = [c for c in roi_table.columns if c.startswith("pca_")]
    if not pca_cols or roi_table["specimen"].nunique() < 2:
        print("ROI evaluation skipped (need >=2 specimens with PCA features).")
        return
    train, test = split_by_specimen(roi_table, test_fraction=0.3, seed=wf.anomaly.seed)
    if train.empty or test.empty:
        print("ROI evaluation skipped (empty split).")
        return
    primary = wf.anomaly.methods[0]
    from dataclasses import replace
    det = fit_detectors(train[pca_cols].to_numpy(),
                        replace(wf.anomaly, methods=[primary]))[primary]
    rows = []
    for split_name, df in (("train", train), ("test", test)):
        scores = det.score(df[pca_cols].to_numpy())
        for (_, r), s in zip(df.iterrows(), scores):
            rows.append({"roi_id": r["roi_id"], "specimen": r["specimen"],
                         "material": r["material"], "split": split_name,
                         f"score_{primary}": float(s)})
    out = pd.DataFrame(rows)
    path = os.path.join(out_dir, "roi_evaluation.csv")
    out.to_csv(path, index=False)
    tr = out[out.split == "train"][f"score_{primary}"]
    te = out[out.split == "test"][f"score_{primary}"]
    print(f"ROI hold-out evaluation ({len(tr)} train / {len(te)} test ROIs, "
          f"{roi_table['specimen'].nunique()} specimens) -> {path}")
    print(f"  score mean train={tr.mean():.3f}  test={te.mean():.3f} "
          "(similar values = generalizes; much higher test = new-piece drift)")


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
                   help="Which population drives the flagged regions: 'self' "
                        "(anomalies within the target film, default) or 'baseline' "
                        "(vs bare silicon). The silicon-contrast map is produced "
                        "either way.")
    p.add_argument("--anomaly-percentile", type=float, default=97.5)
    p.add_argument("--compare-clusters", action="store_true",
                   help="Also run kmeans/dbscan/gmm on the same features and "
                        "write a stability comparison CSV.")
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

    # Stage 5 deliverables: PC1-vs-PC2 scatter by piece + spectral histogram.
    scatter_pts, scatter_lbl, all_vals = [], [], []
    rng = np.random.default_rng(0)
    for a in res.analyses:
        fg = a.piece.foreground_spectra()
        if fg.shape[0] > 4000:
            fg = fg[rng.choice(fg.shape[0], 4000, replace=False)]
        scatter_pts.append(res.pca.transform(fg))
        scatter_lbl += [a.piece.piece_id] * scatter_pts[-1].shape[0]
        all_vals.append(fg.ravel())
    save_pca_scatter(np.vstack(scatter_pts), scatter_lbl, out_dir)
    save_spectral_histogram(np.concatenate(all_vals), out_dir,
                            xlabel="SNV-normalized reflectance",
                            title="Spectral histogram (analysis values, in-mask)")

    primary = wf.anomaly.methods[0]
    threshold = res.baseline_thresholds[primary]
    print("\n{:<22} {:>10} {:>10} {:>12} {:>10}".format(
        "piece_id", "silhouette", "n_clust", "anom_frac", "n_regions"))
    for a in res.analyses:
        piece = a.piece
        save_analysis_figure(a, primary, threshold, out_dir)
        # Always write the region CSV -- an empty table is a result too, and it
        # prevents stale CSVs from an earlier run surviving next to new figures.
        a.region_table.to_csv(os.path.join(out_dir, f"{piece.piece_id}_regions.csv"),
                              index=False)
        anom_frac = float(a.flagged.sum()) / max(1, int(piece.mask.sum()))
        print("{:<22} {:>10.3f} {:>10} {:>11.2%} {:>10}".format(
            piece.piece_id, a.cluster_metrics.get("silhouette", float("nan")),
            a.cluster_metrics.get("n_clusters", 0), anom_frac, len(a.regions)))

    if res.roi_table is not None:
        roi_csv = os.path.join(out_dir, "roi_table.csv")
        res.roi_table.to_csv(roi_csv, index=False)
        print(f"\nROI table ({len(res.roi_table)} rows) -> {roi_csv}")
        _save_roi_evaluation(res.roi_table, wf, out_dir)

    if args.compare_clusters:
        _save_cluster_comparison(res, wf, out_dir)

    report_path = write_report(res, args.target, args.baseline, wf, out_dir,
                               samples_csv=os.path.join(ORGANIZED_DATA_ROOT, "samples.csv"))
    print(f"\nReport -> {report_path}")
    print(f"Figures + region tables under {out_dir}")
    print(f"Flag threshold ({primary}, fit_on={wf.anomaly.fit_on!r}): {threshold:.3f}")


if __name__ == "__main__":
    main()
