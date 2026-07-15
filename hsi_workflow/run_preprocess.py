"""CLI for Steps 9-10: load -> preprocess -> optical density -> previews + stats.

    python -m hsi_workflow.run_preprocess --dataset lig
    python -m hsi_workflow.run_preprocess --dataset lig --od-method none
    python -m hsi_workflow.run_preprocess --dataset sio2 --no-calibrate

Writes preview PNGs to out/workflow/preprocessing/<label>/ and prints per-cube
sanity statistics (shape, % saturated, finite check, film/substrate mean OD).
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .config import DATASETS, PreprocessConfig
from .io import load_cube, iter_cube_paths
from .preprocessing import preprocess
from .optical_density import to_optical_density
from .viz import save_preprocess_preview

DEFAULT_OUT = os.path.join("out", "workflow", "preprocessing")


def build_config(args) -> PreprocessConfig:
    return PreprocessConfig(
        register=args.register,
        background="none" if args.no_background else "dark",
        calibrate=not args.no_calibrate,
        normalize=args.normalize,
        od_method=args.od_method,
        invert_foreground=args.invert_foreground,
        seed=args.seed,
    )


def main():
    p = argparse.ArgumentParser(description="HSI workflow Step 9-10: preprocessing + optical density.")
    p.add_argument("--dataset", default="lig", choices=sorted(DATASETS), help="Dataset preset.")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--od-method", default="substrate",
                   choices=["substrate", "white", "reference_scan", "none"])
    p.add_argument("--normalize", default="none", choices=["none", "snv"])
    p.add_argument("--no-calibrate", action="store_true",
                   help="Skip white/dark reflectance calibration (use raw/background-subtracted DN).")
    p.add_argument("--no-background", action="store_true",
                   help="Skip explicit dark subtraction in the no-calibrate path.")
    p.add_argument("--register", action="store_true",
                   help="Align a separate reference/background scan onto the sample (needs --reference).")
    p.add_argument("--reference", default=None,
                   help="Header path of a bare-support scan for od-method=reference_scan / --register.")
    p.add_argument("--invert-foreground", action="store_true",
                   help="Flip which KMeans cluster is the film (for crops sitting inside the film).")
    p.add_argument("--od-band-nm", type=float, default=400.0, help="Band (nm) shown in the OD preview.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg_ds = DATASETS[args.dataset]
    cfg = build_config(args)
    reference_cube = load_cube(args.reference) if args.reference else None

    cubes = iter_cube_paths(cfg_ds)
    if not cubes:
        print(f"No cubes found for dataset {cfg_ds.name!r} under {cfg_ds.data_dir}")
        return
    print(f"Dataset {cfg_ds.name!r}: {len(cubes)} cube(s). "
          f"calibrate={cfg.calibrate} od_method={cfg.od_method} normalize={cfg.normalize}")

    print("\n{:<16} {:>12} {:>7} {:>7} {:>10} {:>10} {:>7}".format(
        "label", "shape", "%sat", "finite", "OD_film", "OD_subs", "band"))
    for label, hdr in cubes:
        cube = load_cube(hdr)
        pre = preprocess(cube, cfg,
                         white_ref_hdr=cfg_ds.white_ref, dark_ref_hdr=cfg_ds.dark_ref,
                         reference_cube=reference_cube)
        od, _ = to_optical_density(pre, cfg,
                                   white_ref_hdr=cfg_ds.white_ref, reference_cube=reference_cube)

        pct_sat = 100.0 * pre.saturated.mean()
        finite = np.isfinite(od).all()
        seg = pre.segmentation
        od_film = float(od[seg.foreground].mean()) if seg.foreground.sum() else float("nan")
        od_subs = float(od[seg.substrate].mean()) if seg.substrate.sum() else float("nan")

        out_dir = os.path.join(args.out, label)
        save_preprocess_preview(pre, od, out_dir, od_band_nm=args.od_band_nm)

        shape_str = "x".join(map(str, cube.shape))
        print("{:<16} {:>12} {:>6.2f}% {:>7} {:>10.4f} {:>10.4f} {:>7.0f}".format(
            label, shape_str, pct_sat, str(bool(finite)), od_film, od_subs, args.od_band_nm))

    print(f"\nPreviews written under {args.out}")
    if args.od_method == "substrate":
        print("Expected: OD_subs ~ 0 (substrate is the I0 reference); OD_film shifted away from 0.")


if __name__ == "__main__":
    main()
