"""CLI: build (or reuse the cache for) the silicon baseline.

    python -m hsi_workflow.run_baseline --dataset sio2_bare_si
    python -m hsi_workflow.run_baseline --dataset sio2_bare_si --force

Extracts every piece of the bare-Si dataset, pools their spectra into a
:class:`~hsi_workflow.baseline.SiliconBaseline` (mean/std/cov + a capped
sample), and caches it under ``out/workflow/baseline/<dataset>/`` --
``baseline.npz`` (arrays), ``meta.json`` (config snapshot + summary),
``piece_stats.csv`` (per-piece QA table), and ``figures/<piece_id>_baseline.png``
(per-piece mean spectrum vs the pooled baseline mean, for visual debugging).

``pipeline.run_workflow`` loads this same cache on every ``run_analyze`` call
instead of re-extracting the raw bare-Si scan; running this CLI is the way to
inspect/refresh that cache on its own.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    __package__ = "hsi_workflow"

import argparse
import os

from .config import DATASETS, WorkflowConfig, BASELINE_CACHE_ROOT
from .baseline import load_or_compute_baseline


def main():
    p = argparse.ArgumentParser(description="Build/refresh the cached silicon baseline.")
    p.add_argument("--dataset", default="sio2_bare_si", type=str.lower, choices=sorted(DATASETS))
    p.add_argument("--out", default=BASELINE_CACHE_ROOT)
    p.add_argument("--force", action="store_true",
                   help="Recompute even if a valid cache already exists.")
    args = p.parse_args()

    ds_cfg = DATASETS[args.dataset]
    wf = WorkflowConfig()

    sb = load_or_compute_baseline(ds_cfg, wf, args.out, force=args.force, verbose=True)

    print("\n{:<22} {:>8} {:>10} {:>8} {:>16} {:>8}".format(
        "piece_id", "n_px", "mean_refl", "snr", "sam_from_global", "outlier"))
    for ps in sb.piece_stats:
        print("{:<22} {:>8} {:>10.3f} {:>8.1f} {:>16.4f} {:>8}".format(
            ps.piece_id, ps.n_px, ps.mean_reflectance, ps.snr,
            ps.sam_from_global, "YES" if ps.flag_outlier else ""))

    n_outliers = sum(ps.flag_outlier for ps in sb.piece_stats)
    print(f"\n{len(sb.piece_stats)} pieces, {n_outliers} flagged as outliers "
          f"(sam_from_global > mean + 2*std across pieces).")
    print(f"Cache + figures + piece_stats.csv under {os.path.join(args.out, ds_cfg.name)}")


if __name__ == "__main__":
    main()
