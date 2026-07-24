"""CLI: Stage 4 exploratory spectral visualization.

    python -m hsi_workflow.run_explore --dataset sio2_bare_si
    python -m hsi_workflow.run_explore --dataset sio2_bare_si sio2_dish_black

Accepts one or *several* dataset presets so the key control-vs-experimental
comparison (bare silicon LOW spectral variance vs processed SiO2 HIGHER) can be
made in a single figure. Writes, per run:

- ``<piece>_explore.png``          per-piece figure (mean spectrum, band images,
                                   RGB, spectral variance map)
- ``material_mean_spectra.png``    mean spectra overlaid by material
- ``material_variance.csv``        per-piece + per-material spectral variance
- ``noise_metrics.csv``            RMS noise + SNR before/after SG smoothing
- ``reflectance_histogram.png``    pooled in-mask reflectance (Stage 2 check:
                                   values mostly 0-1, no clipping)
"""

from __future__ import annotations

# Allow running this file directly (python run_xxx.py) as well as
# as a module (python -m hsi_workflow.run_xxx). When run as a script the
# package context is missing, so add the repo root and set __package__ so
# the relative imports below resolve (PEP 366).
if __package__ in (None, ""):
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    __package__ = "hsi_workflow"

import argparse
import os

import numpy as np

from .config import DATASETS, WorkflowConfig
from .pipeline import prepare_pieces
from .explore import save_piece_exploration, save_material_mean_spectra, spectral_variance_map
from .viz import save_spectral_histogram

DEFAULT_OUT = os.path.join("out", "workflow", "explore")


def main():
    p = argparse.ArgumentParser(description="Stage 4 exploratory visualization.")
    p.add_argument("--dataset", nargs="+", default=["sio2_bare_si"],
                   choices=sorted(DATASETS),
                   help="One or more dataset presets (pass silicon + sio2 "
                        "together for the material comparison).")
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()

    import pandas as pd

    wf = WorkflowConfig()
    # Exploratory viz is done on calibrated *reflectance*: SNV would flatten every
    # pixel to unit variance and destroy the Si-low / SiO2-high variance signal.
    wf.preprocess.normalize = "none"
    run_name = "+".join(args.dataset)
    out_dir = os.path.join(args.out, run_name)

    pieces = []
    for name in args.dataset:
        pieces.extend(prepare_pieces(DATASETS[name], wf))
    if not pieces:
        print("No pieces found.")
        return

    for piece in pieces:
        save_piece_exploration(piece, out_dir)
    save_material_mean_spectra(pieces, out_dir)

    # --- Stage 4/5 metric: spectral variance, silicon low vs SiO2 high ---
    print("\nMean spectral variance per piece (heterogeneity proxy):")
    var_rows, by_mat = [], {}
    for piece in pieces:
        v = float(np.nanmean(spectral_variance_map(piece.data, piece.mask)))
        by_mat.setdefault(piece.material, []).append(v)
        var_rows.append({"piece_id": piece.piece_id, "material": piece.material,
                         "mean_spectral_variance": v})
        print(f"  {piece.piece_id:<22} {piece.material:<8} {v:.4g}")
    print("\nMean per material:")
    for mat, vals in by_mat.items():
        var_rows.append({"piece_id": f"<all {mat}>", "material": mat,
                         "mean_spectral_variance": float(np.mean(vals))})
        print(f"  {mat:<8} {np.mean(vals):.4g}")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(var_rows).to_csv(os.path.join(out_dir, "material_variance.csv"), index=False)

    # --- Stage 3/4 metric: noise reduction from SG smoothing ---
    noise_rows = []
    for piece in pieces:
        if not piece.noise:
            continue
        b, a = piece.noise.get("before", {}), piece.noise.get("after", {})
        noise_rows.append({
            "piece_id": piece.piece_id, "material": piece.material,
            "rms_noise_before": b.get("rms_noise"), "rms_noise_after": a.get("rms_noise"),
            "snr_before": b.get("snr"), "snr_after": a.get("snr"),
        })
    if noise_rows:
        pd.DataFrame(noise_rows).to_csv(os.path.join(out_dir, "noise_metrics.csv"), index=False)
        nb = np.nanmean([r["rms_noise_before"] for r in noise_rows])
        na = np.nanmean([r["rms_noise_after"] for r in noise_rows])
        print(f"\nSG smoothing noise reduction: RMS {nb:.4g} -> {na:.4g} "
              f"({(1 - na / nb):.0%} lower)" if nb else "")

    # --- Stage 2 metric: reflectance range (0-1, no clipping) ---
    rng = np.random.default_rng(0)
    pooled = []
    for piece in pieces:
        fg = piece.foreground_spectra()
        if fg.shape[0] > 2000:
            fg = fg[rng.choice(fg.shape[0], 2000, replace=False)]
        pooled.append(fg.ravel())
    vals = np.concatenate(pooled)
    vals = vals[np.isfinite(vals)]
    out_of_range = float(((vals < 0) | (vals > 1)).mean())
    save_spectral_histogram(vals, out_dir, name="reflectance_histogram",
                            xlabel="reflectance",
                            title=f"In-mask reflectance ({out_of_range:.2%} outside [0, 1])")
    print(f"\nReflectance range check: {out_of_range:.2%} of in-mask values outside [0, 1] "
          "(expected: small; large values suggest calibration/saturation issues)")
    print(f"\nFigures + metrics written under {out_dir}")


if __name__ == "__main__":
    main()
