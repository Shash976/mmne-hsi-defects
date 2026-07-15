"""CLI: piece + ROI extraction.

    python -m hsi_workflow.run_extract --dataset sio2_bare_si
    python -m hsi_workflow.run_extract --dataset sio2_dish_white_20 --save-crops

Splits each raw scan into individual pieces, tiles ROIs inside every piece, and
writes the cross-specimen ROI table (parquet + csv) plus optional piece crops and
per-piece exploration figures. Prints how many pieces/ROIs came out of each scan.
"""

from __future__ import annotations

import argparse
import os

from .config import DATASETS, WorkflowConfig
from .pipeline import prepare_pieces
from .pieces import save_piece_crops
from .rois import tile_rois, build_roi_table
from .explore import save_piece_exploration

DEFAULT_OUT = os.path.join("out", "workflow", "extract")


def build_cfg(args) -> WorkflowConfig:
    wf = WorkflowConfig()
    wf.piece.min_area = args.min_area
    wf.piece.method = args.piece_method
    wf.roi.patch = args.patch
    wf.roi.stride = args.stride
    wf.roi.min_coverage = args.min_coverage
    return wf


def main():
    p = argparse.ArgumentParser(description="Piece + ROI extraction.")
    p.add_argument("--dataset", default="sio2_bare_si", choices=sorted(DATASETS))
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--piece-method", default="sam", choices=["sam", "mahalanobis", "kmeans"])
    p.add_argument("--min-area", type=int, default=1000)
    p.add_argument("--patch", type=int, default=32)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--min-coverage", type=float, default=0.85)
    p.add_argument("--save-crops", action="store_true", help="Persist each piece as an ENVI cube.")
    p.add_argument("--figures", action="store_true", help="Save a Stage-4 figure per piece.")
    args = p.parse_args()

    ds = DATASETS[args.dataset]
    wf = build_cfg(args)
    out_dir = os.path.join(args.out, ds.name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Extracting pieces from dataset {ds.name!r} ({ds.material}) ...")
    pieces = prepare_pieces(ds, wf)
    if not pieces:
        print("No pieces found.")
        return

    if args.save_crops:
        save_piece_crops(pieces, os.path.join(out_dir, "pieces"))

    all_rois = []
    print("\n{:<22} {:>8} {:>8}".format("piece_id", "px", "rois"))
    for piece in pieces:
        rois = tile_rois(piece, wf.roi)
        all_rois.extend(rois)
        if args.figures:
            save_piece_exploration(piece, os.path.join(out_dir, "figures"))
        print("{:<22} {:>8} {:>8}".format(piece.piece_id, int(piece.mask.sum()), len(rois)))

    wl = pieces[0].wavelengths
    table = build_roi_table(all_rois, wl)
    csv_path = os.path.join(out_dir, "roi_table.csv")
    table.to_csv(csv_path, index=False)
    try:
        table.to_parquet(os.path.join(out_dir, "roi_table.parquet"), index=False)
    except Exception as e:                       # parquet engine optional
        print(f"(parquet skipped: {e})")

    print(f"\n{len(pieces)} pieces, {len(all_rois)} ROIs -> {csv_path}")
    print("ROIs per material:")
    print(table.groupby("material")["roi_id"].count().to_string())


if __name__ == "__main__":
    main()
