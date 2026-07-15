"""CLI: organize a scan into a hierarchical piece/ROI dataset.

    python -m hsi_workflow.run_extract --dataset sio2_bare_si
    python -m hsi_workflow.run_extract --dataset sio2_dish_white_20 --radiometry raw
    python -m hsi_workflow.run_extract --dataset sio2_dish_black --no-roi-cubes

Writes, under out/workflow/extract/<dataset>/, one folder per physical piece
containing the cropped piece cube, a mask, per-piece metadata, and a ``rois/``
subfolder of individual cropped ROI cubes -- plus a dataset-level manifest and
the aggregated ROI feature table. See docs/extraction.md.
"""

from __future__ import annotations

import argparse
import os

from .config import DATASETS, WorkflowConfig
from .dataset import export_dataset

DEFAULT_OUT = os.path.join("out", "workflow", "extract")


def build_cfg(args) -> WorkflowConfig:
    wf = WorkflowConfig()
    wf.piece.method = args.piece_method
    wf.piece.min_area = args.min_area
    wf.roi.patch = args.patch
    wf.roi.stride = args.stride
    wf.roi.min_coverage = args.min_coverage
    return wf


def main():
    p = argparse.ArgumentParser(description="Hierarchical piece/ROI dataset export.")
    p.add_argument("--dataset", default="sio2_bare_si", choices=sorted(DATASETS))
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--radiometry", default="reflectance", choices=["reflectance", "raw"],
                   help="Save cropped cubes as calibrated reflectance (default) or raw DN.")
    p.add_argument("--piece-method", default="sam", choices=["sam", "mahalanobis", "kmeans"])
    p.add_argument("--min-area", type=int, default=1000)
    p.add_argument("--patch", type=int, default=32)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--min-coverage", type=float, default=0.85)
    p.add_argument("--no-roi-cubes", action="store_true",
                   help="Skip writing per-ROI cubes (keep folders + roi_index.csv only).")
    args = p.parse_args()

    ds = DATASETS[args.dataset]
    wf = build_cfg(args)

    print(f"Organizing dataset {ds.name!r} ({ds.material}) into {args.out}/{ds.name}/ ...")
    rois, manifest = export_dataset(ds, wf, args.out, radiometry=args.radiometry,
                                    save_roi_cubes=not args.no_roi_cubes)
    print(f"\n{manifest['n_pieces']} pieces, {manifest['n_rois']} ROIs "
          f"({args.radiometry} cubes).")
    print(f"Tree + manifest.json + roi_table.csv under {os.path.join(args.out, ds.name)}")


if __name__ == "__main__":
    main()
