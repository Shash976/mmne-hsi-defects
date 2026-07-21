"""Stage 11 -- Final report generation.

Aggregates a :class:`~hsi_workflow.pipeline.WorkflowResult` into the document's
capstone deliverable: a single ``report.md`` containing the run configuration,
PCA/cluster/anomaly statistics per piece, the silicon-baseline comparison, and
data-driven answers to the document's questions (are anomalies localized?
repeated? near edges? random?) -- all *descriptive*, never naming a region as a
specific defect type.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from scipy import ndimage as ndi


def _edge_share(flagged: np.ndarray, mask: np.ndarray, margin: int = 5) -> float:
    """Fraction of flagged pixels lying within ``margin`` px of the mask boundary.

    High values mean the "anomalies" hug the piece edge -- usually mixed
    fragment/dish pixels rather than film features (a known masking artifact).
    """
    n = int(flagged.sum())
    if n == 0:
        return float("nan")
    interior = ndi.binary_erosion(mask, iterations=margin)
    edge_zone = mask & ~interior
    return float((flagged & edge_zone).sum()) / n


def piece_summary_rows(result) -> list:
    """One stats dict per analyzed piece (feeds the report's summary table)."""
    rows = []
    for a in result.analyses:
        piece = a.piece
        n_fg = max(1, int(piece.mask.sum()))
        flag_frac = float(a.flagged.sum()) / n_fg
        areas = [r.area for r in a.regions]
        base = a.baseline_map[piece.mask]
        base = base[np.isfinite(base)]
        rows.append({
            "piece_id": piece.piece_id,
            "material": piece.material,
            "n_px": n_fg,
            "silhouette": a.cluster_metrics.get("silhouette", float("nan")),
            "n_clusters": a.cluster_metrics.get("n_clusters", 0),
            "anom_frac": flag_frac,
            "n_regions": len(a.regions),
            "largest_region_px": max(areas) if areas else 0,
            "mean_region_px": float(np.mean(areas)) if areas else float("nan"),
            "mean_anomaly": float(np.mean([r.mean_anomaly for r in a.regions]))
                            if a.regions else float("nan"),
            "edge_share": _edge_share(a.flagged, piece.mask),
            "median_baseline_dist": float(np.median(base)) if base.size else float("nan"),
        })
    return rows


def _fmt(v, spec=".3f") -> str:
    if isinstance(v, float):
        return "nan" if not np.isfinite(v) else format(v, spec)
    return str(v)


def write_report(result, target: str, baseline: str, wf, out_dir: str,
                 samples_csv: Optional[str] = None) -> str:
    """Write ``report.md`` for one analyzed target dataset; returns its path."""
    os.makedirs(out_dir, exist_ok=True)
    rows = piece_summary_rows(result)
    evr = result.pca.explained_variance_ratio
    primary = wf.anomaly.methods[0]

    lines = []
    add = lines.append
    add(f"# Anomaly-detection report — `{target}`\n")
    add("Unsupervised spectral-anomaly screening of SiO₂ thin-film pieces "
        "(no defect labels, no reference spectra). Two anomaly products per piece: "
        "the **within-film** map (detectors fit on the film's own majority; drives "
        "the flagged regions below) and the **silicon-baseline contrast** map "
        f"(distance from the `{baseline}` control population — the hypothesis "
        "deliverable). Regions are *described*, never named as defect types.\n")

    add("## Run configuration\n")
    add(f"- Target: `{target}` · Baseline: `{baseline}`")
    add(f"- PCA components: {wf.pca.n_components} — explained variance: "
        + ", ".join(f"PC{i+1}={v:.1%}" for i, v in enumerate(evr)))
    add(f"- Clustering: `{wf.cluster.method}` (k={wf.cluster.n_clusters})")
    add(f"- Anomaly detectors: {wf.anomaly.methods} · fit_on=`{wf.anomaly.fit_on}` · "
        f"flag percentile {wf.anomaly.anomaly_percentile}")
    add(f"- Postprocessing: median={wf.postproc.median_size}, "
        f"opening={wf.postproc.opening_radius}, min_component={wf.postproc.min_component}\n")
    if samples_csv and os.path.exists(samples_csv):
        add(f"Sample inventory: [`{samples_csv}`]({samples_csv.replace(os.sep, '/')})\n")

    add("## Per-piece summary\n")
    add("| piece | silhouette | clusters | anomalous | regions | largest (px) | "
        "edge share | median Si-dist |")
    add("|---|---|---|---|---|---|---|---|")
    for r in rows:
        add(f"| {r['piece_id']} | {_fmt(r['silhouette'], '.2f')} | {r['n_clusters']} | "
            f"{_fmt(r['anom_frac'], '.2%')} | {r['n_regions']} | {r['largest_region_px']} | "
            f"{_fmt(r['edge_share'], '.0%')} | {_fmt(r['median_baseline_dist'], '.1f')} |")
    add("")

    # --- The document's questions, answered from the numbers. ---
    add("## The document's questions\n")
    total_regions = sum(r["n_regions"] for r in rows)
    fracs = [r["anom_frac"] for r in rows]
    edge_shares = [r["edge_share"] for r in rows if np.isfinite(r["edge_share"])]
    mean_frac = float(np.mean(fracs)) if fracs else float("nan")

    add(f"- **Localized?** Mean anomalous fraction across pieces is "
        f"{_fmt(mean_frac, '.2%')} ({total_regions} region(s) total). "
        + ("Small and localized — consistent with the expected 2–10% band."
           if np.isfinite(mean_frac) and mean_frac < 0.10 else
           "A large fraction is flagged — inspect whether the film is genuinely "
           "heterogeneous or the threshold/postprocessing needs retuning."))
    pieces_with = sum(1 for r in rows if r["n_regions"] > 0)
    add(f"- **Repeated across pieces?** {pieces_with}/{len(rows)} pieces have at "
        "least one flagged region. Recurring regions in similar positions across "
        "pieces suggest a process signature; isolated ones suggest local events.")
    if edge_shares:
        mean_edge = float(np.mean(edge_shares))
        add(f"- **Near edges?** On average {_fmt(mean_edge, '.0%')} of flagged pixels "
            "lie within 5 px of the piece boundary. "
            + ("Edge-dominated — treat with suspicion (mixed fragment/dish pixels)."
               if mean_edge > 0.5 else "Not edge-dominated."))
    else:
        add("- **Near edges?** No flagged pixels to assess.")
    add("- **Random?** Compare the flagged-region overlay against the cluster map "
        "in each `<piece>_analysis.png`: regions that respect cluster boundaries "
        "are spectrally coherent populations; scattered speckle that survives "
        "postprocessing suggests noise.\n")

    add("## Silicon baseline vs processed film\n")
    add("`median Si-dist` above is each piece's median Mahalanobis distance from "
        "the bare-silicon control population (spectral space). Uniformly large "
        "values simply reflect the material difference (SiO₂ ≠ Si); *variation* "
        "between pieces or within a piece (see the baseline-contrast panel) is "
        "the interesting signal.\n")

    add("## Artifacts\n")
    add("- `pca_summary.png` — explained variance + PC loadings")
    add("- `pca_scatter.png` — PC1 vs PC2 by piece")
    add("- `spectral_histogram.png` — distribution of analysis values")
    add("- `<piece>_analysis.png` — 9-panel maps (PCs, clusters, anomaly, baseline "
        "contrast, spectral distance, probability, regions, spectra, histogram)")
    add("- `<piece>_regions.csv` — region tables (always written; empty = none flagged)")
    add("- `roi_table.csv` — cross-specimen ROI ML table")
    add("- `cluster_comparison.csv` — method-stability comparison (when requested)")
    add("- `roi_evaluation.csv` — specimen-level hold-out scores (when ≥2 specimens)\n")

    add("## Not established here (future work)\n")
    add("Physical origin of any region requires SEM/AFM/Raman/XPS/TEM follow-up "
        "(document Stage 12). This report only ranks *where* to look.\n")

    path = os.path.join(out_dir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
