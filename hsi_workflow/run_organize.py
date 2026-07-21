"""CLI: organize the raw scans into the document's ``data/`` structure.

    python -m hsi_workflow.run_organize
    python -m hsi_workflow.run_organize --datasets sio2_bare_si sio2_dish_black
    python -m hsi_workflow.run_organize --no-roi-cubes

Implements the document's *Sample Inventory* stage plus its hierarchical
Specimen -> Image -> ROI organization:

    data/
        samples.csv               # the sample database (ID | Material | Dimensions | Notes)
        inventory_summary.json    # metrics: number of samples, sizes, imaging area
        manifest.json             # raw-scan + calibration paths, dataset -> tree map
        organized/<dataset>/<piece_id>/
            <piece_id>.hdr/.img   # cropped piece cube (calibrated reflectance)
            <piece_id>_mask.npy   # fragment footprint
            meta.json             # material, bbox-in-scan, shape, counts
            roi_index.csv         # one row per ROI
            rois/<roi_id>.hdr     # cropped ROI sub-cubes

The raw scans stay where they were acquired (the OneDrive archive referenced by
the dataset presets); this tree is the analysis-ready, self-describing copy.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

from config import DATASETS, WorkflowConfig, ORGANIZED_DATA_ROOT
from dataset import export_dataset

# The semiconductor datasets the document inventories. LIG is a test bed and is
# organized only when asked for explicitly.
DEFAULT_DATASETS = ["sio2_bare_si", "sio2_dish_black", "sio2_dish_white_1",
                    "sio2_dish_white_20"]


def build_samples_rows(manifest: dict) -> list:
    """Sample-database rows (one physical piece = one sample) from a manifest."""
    rows = []
    for m in manifest["pieces"]:
        rr, cc = m["shape"][0], m["shape"][1]
        rows.append({
            "sample_id": m["piece_id"],
            "material": m["material"],
            "dataset": m["dataset"],
            "source_scan": m["source_label"],
            "height_px": rr,
            "width_px": cc,
            "area_px": m["n_foreground_px"],
            "n_rois": m["n_rois"],
            "notes": "",
        })
    return rows


def main():
    p = argparse.ArgumentParser(description="Organize raw scans into data/ (sample "
                                            "database + specimen/piece/ROI tree).")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                   choices=sorted(DATASETS))
    p.add_argument("--data-root", default=ORGANIZED_DATA_ROOT,
                   help="Repo data folder to organize into (default: data/).")
    p.add_argument("--radiometry", default="reflectance", choices=["reflectance", "raw"])
    p.add_argument("--no-roi-cubes", action="store_true",
                   help="Skip per-ROI cubes (keep folders + roi_index.csv).")
    p.add_argument("--patch", type=int, default=32)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--min-coverage", type=float, default=0.85)
    args = p.parse_args()

    import pandas as pd

    wf = WorkflowConfig()
    wf.roi.patch, wf.roi.stride, wf.roi.min_coverage = args.patch, args.stride, args.min_coverage
    organized_root = os.path.join(args.data_root, "organized")
    os.makedirs(organized_root, exist_ok=True)

    all_rows, dataset_manifests = [], {}
    for name in args.datasets:
        ds = DATASETS[name]
        print(f"\n=== Organizing {name!r} ({ds.material}) ===")
        _, manifest = export_dataset(ds, wf, organized_root,
                                     radiometry=args.radiometry,
                                     save_roi_cubes=not args.no_roi_cubes)
        dataset_manifests[name] = manifest
        all_rows.extend(build_samples_rows(manifest))

    # --- Sample database (the document's Sample Inventory deliverable) ---
    samples = pd.DataFrame(all_rows, columns=["sample_id", "material", "dataset",
                                              "source_scan", "height_px", "width_px",
                                              "area_px", "n_rois", "notes"])
    samples_path = os.path.join(args.data_root, "samples.csv")
    samples.to_csv(samples_path, index=False)

    # --- Inventory metrics: number of samples, sample size, imaging area ---
    by_mat = samples.groupby("material")
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_samples": int(len(samples)),
        "n_samples_by_material": {m: int(n) for m, n in samples["material"].value_counts().items()},
        "total_imaging_area_px": int(samples["area_px"].sum()),
        "median_sample_area_px": float(samples["area_px"].median()) if len(samples) else None,
        "area_px_by_material": {m: int(g["area_px"].sum()) for m, g in by_mat},
        "n_rois_total": int(samples["n_rois"].sum()),
        "document_target": "10-20 independent specimens (8-10 bare Si, 20-30 SiO2)",
    }
    with open(os.path.join(args.data_root, "inventory_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- Top-level manifest: where everything came from ---
    manifest = {
        "generated_utc": summary["generated_utc"],
        "radiometry": args.radiometry,
        "roi": {"patch": wf.roi.patch, "stride": wf.roi.stride,
                "min_coverage": wf.roi.min_coverage},
        "datasets": {
            name: {
                "material": DATASETS[name].material,
                "raw_data_dir": DATASETS[name].data_dir,
                "raw_hdr_glob": DATASETS[name].hdr_glob,
                "white_ref": DATASETS[name].white_ref,
                "dark_ref": DATASETS[name].dark_ref,
                "organized_tree": os.path.join("organized", name),
                "n_pieces": dataset_manifests[name]["n_pieces"],
                "n_rois": dataset_manifests[name]["n_rois"],
            } for name in args.datasets
        },
    }
    with open(os.path.join(args.data_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSample database ({len(samples)} samples) -> {samples_path}")
    print("Samples by material: "
          + ", ".join(f"{m}={n}" for m, n in summary["n_samples_by_material"].items()))
    print(f"Total imaging area: {summary['total_imaging_area_px']:,} px; "
          f"total ROIs: {summary['n_rois_total']}")
    print(f"Tree + manifests under {args.data_root}{os.sep}")


if __name__ == "__main__":
    main()
