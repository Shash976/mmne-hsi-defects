"""Hierarchical on-disk dataset export (specimen -> piece -> ROI).

The document recommends organizing the data hierarchically so it is easy to
inspect and modify. This module writes exactly that tree: one folder per physical
**piece**, containing the cropped piece cube and a ``rois/`` subfolder of the
individual cropped ROI cubes, plus JSON metadata at every level.

    <out_root>/<dataset>/
        manifest.json                     # dataset-level index
        <piece_id>/
            <piece_id>.hdr / .img         # cropped piece cube (ENVI)
            <piece_id>_mask.npy           # which pixels are the fragment
            meta.json                     # material, bbox, shape, counts
            roi_index.csv                 # one row per ROI in this piece
            rois/
                <roi_id>.hdr / .img       # cropped ROI sub-cube (ENVI)
                ...

Cubes are saved as **calibrated reflectance** by default (physically meaningful
and analysis-ready) or raw DN if requested. The heavy per-ROI spectral *features*
(PCA, anomaly) are an analysis product and live in ``run_analyze``'s ROI table,
not here -- this module is about organizing the raw sample data.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import DatasetConfig, WorkflowConfig
from .cube_io import Cube, iter_cube_paths, load_dataset_cube, save_envi_cube
from .preprocessing import preprocess, saturation_mask, normalize_intensity
from .pieces import Piece, extract_pieces
from .rois import Roi, tile_rois, build_roi_table
from .viz import pseudo_rgb


def _reflectance_cube(raw_piece: Piece, src: Cube, ds_cfg: DatasetConfig,
                      wf: WorkflowConfig, radiometry: str) -> np.ndarray:
    """Calibrated reflectance (or raw DN) for one piece's bounding box."""
    if radiometry == "raw":
        return raw_piece.data
    # Reflectance = calibrate + smooth, but NOT SNV (keep physical reflectance).
    pcfg = replace(wf.preprocess, normalize="none", od_method="none")
    piece_cube = Cube(data=raw_piece.data, wavelengths=src.wavelengths,
                      shutter=src.shutter, ceiling=src.ceiling, path=src.path,
                      label=raw_piece.piece_id, material=src.material)
    return preprocess(piece_cube, pcfg, white_ref_hdr=ds_cfg.white_ref,
                      dark_ref_hdr=ds_cfg.dark_ref).data


def export_dataset(ds_cfg: DatasetConfig, wf: WorkflowConfig, out_root: str,
                   radiometry: str = "reflectance", save_roi_cubes: bool = True,
                   verbose: bool = True) -> Tuple[List[Roi], dict]:
    """Export one dataset preset into the hierarchical piece/ROI tree.

    Returns ``(all_rois, manifest)``. ``all_rois`` carries SNV mean-spectrum
    features (for the aggregated ML table); the cubes on disk are reflectance/raw.
    """
    wf.validate()
    if radiometry not in ("reflectance", "raw"):
        raise ValueError("radiometry must be 'reflectance' or 'raw'")

    ds_dir = os.path.join(out_root, ds_cfg.name)
    os.makedirs(ds_dir, exist_ok=True)

    all_rois: List[Roi] = []
    piece_records: List[dict] = []
    wavelengths = None

    for label, hdr in iter_cube_paths(ds_cfg):
        cube = load_dataset_cube(hdr, ds_cfg)
        wavelengths = cube.wavelengths
        sat = saturation_mask(cube.data, cube.ceiling)
        raw_pieces = extract_pieces(cube, wf.piece, valid_mask=~sat)
        if verbose:
            print(f"  {label}: {len(raw_pieces)} piece(s) [{cube.material}]")

        for rp in raw_pieces:
            piece_dir = os.path.join(ds_dir, rp.piece_id)
            roi_dir = os.path.join(piece_dir, "rois")
            os.makedirs(roi_dir if save_roi_cubes else piece_dir, exist_ok=True)

            save_cube = _reflectance_cube(rp, cube, ds_cfg, wf, radiometry)

            # --- piece cube + mask ---
            save_envi_cube(os.path.join(piece_dir, f"{rp.piece_id}.hdr"),
                           save_cube, wavelengths=cube.wavelengths, material=rp.material)
            np.save(os.path.join(piece_dir, f"{rp.piece_id}_mask.npy"), rp.mask)

            # --- piece pseudo-RGB image ---
            piece_rgb = pseudo_rgb(save_cube, cube.wavelengths)
            plt.imsave(os.path.join(piece_dir, f"{rp.piece_id}_rgb.png"), piece_rgb)

            # --- ROIs: mean-spectrum features from SNV; scalar stats (std,
            # spectral_variance, mean_reflectance) from the physical
            # reflectance cube, where they are actually informative (on SNV
            # data per-pixel std is ~1 by construction).
            snv_cube = normalize_intensity(save_cube, "snv")
            feat_piece = Piece(data=snv_cube, mask=rp.mask, material=rp.material,
                               piece_id=rp.piece_id, source_label=rp.source_label,
                               bbox=rp.bbox, wavelengths=cube.wavelengths)
            rois = tile_rois(feat_piece, wf.roi, stats_data=save_cube)

            roi_rows = []
            for roi in rois:
                r0, r1, c0, c1 = roi.bbox
                if save_roi_cubes:
                    save_envi_cube(os.path.join(roi_dir, f"{roi.roi_id}.hdr"),
                                   save_cube[r0:r1, c0:c1, :],
                                   wavelengths=cube.wavelengths, material=roi.material)
                    roi_rgb = pseudo_rgb(save_cube[r0:r1, c0:c1, :], cube.wavelengths)
                    plt.imsave(os.path.join(roi_dir, f"{roi.roi_id}_rgb.png"), roi_rgb)
                roi_rows.append({"roi_id": roi.roi_id, "r0": r0, "r1": r1,
                                 "c0": c0, "c1": c1, "coverage": roi.coverage,
                                 "spectral_variance": roi.spectral_variance,
                                 "mean_reflectance": roi.mean_reflectance})
            all_rois.extend(rois)

            # --- per-piece index + metadata ---
            import pandas as pd
            pd.DataFrame(roi_rows).to_csv(os.path.join(piece_dir, "roi_index.csv"), index=False)
            meta = {
                "piece_id": rp.piece_id, "source_label": rp.source_label,
                "material": rp.material, "dataset": ds_cfg.name,
                "bbox_in_scan": list(rp.bbox), "shape": list(save_cube.shape),
                "n_foreground_px": int(rp.mask.sum()), "n_rois": len(rois),
                "radiometry": radiometry,
            }
            with open(os.path.join(piece_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            piece_records.append(meta)
            if verbose:
                print(f"    {rp.piece_id}: {int(rp.mask.sum())} px, {len(rois)} ROIs")

    manifest = {
        "dataset": ds_cfg.name, "material": ds_cfg.material,
        "radiometry": radiometry, "n_pieces": len(piece_records),
        "n_rois": len(all_rois), "pieces": piece_records,
    }
    with open(os.path.join(ds_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Aggregated flat ML table alongside the tree (mean spectra + scalar features).
    if all_rois:
        build_roi_table(all_rois, wavelengths).to_csv(
            os.path.join(ds_dir, "roi_table.csv"), index=False)

    return all_rois, manifest
