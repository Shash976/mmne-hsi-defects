"""CLI: Stage 4 exploratory spectral visualization.

    python -m hsi_workflow.run_explore --dataset sio2_bare_si
    python -m hsi_workflow.run_explore --dataset sio2_dish_white_20

Writes a per-piece figure (mean spectrum, band images, RGB, spectral variance
map) and an overlay of mean spectra grouped by material, then prints the mean
spectral variance per material -- the document's key sanity check (silicon low,
processed SiO2 higher).
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .config import DATASETS, WorkflowConfig
from .pipeline import prepare_pieces
from .explore import save_piece_exploration, save_material_mean_spectra, spectral_variance_map

DEFAULT_OUT = os.path.join("out", "workflow", "explore")


def main():
    p = argparse.ArgumentParser(description="Stage 4 exploratory visualization.")
    p.add_argument("--dataset", default="sio2_bare_si", choices=sorted(DATASETS))
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()

    ds = DATASETS[args.dataset]
    wf = WorkflowConfig()
    # Exploratory viz is done on calibrated *reflectance*: SNV would flatten every
    # pixel to unit variance and destroy the Si-low / SiO2-high variance signal.
    wf.preprocess.normalize = "none"
    out_dir = os.path.join(args.out, ds.name)

    pieces = prepare_pieces(ds, wf)
    if not pieces:
        print("No pieces found.")
        return

    for piece in pieces:
        save_piece_exploration(piece, out_dir)
    save_material_mean_spectra(pieces, out_dir)

    print("\nMean spectral variance per piece (heterogeneity proxy):")
    by_mat = {}
    for piece in pieces:
        v = float(np.nanmean(spectral_variance_map(piece.data, piece.mask)))
        by_mat.setdefault(piece.material, []).append(v)
        print(f"  {piece.piece_id:<22} {piece.material:<8} {v:.4g}")
    print("\nMean per material:")
    for mat, vals in by_mat.items():
        print(f"  {mat:<8} {np.mean(vals):.4g}")
    print(f"\nFigures written under {out_dir}")


if __name__ == "__main__":
    main()
