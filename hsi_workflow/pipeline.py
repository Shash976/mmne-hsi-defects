"""End-to-end orchestrator for the revised anomaly-detection workflow.

Wires the stages together while keeping each one swappable:

    raw scan(s)
      -> extract_pieces            (Stage 3.1, pieces.py)
      -> preprocess per piece      (Stages 2-3, preprocessing.py)      => "analysis piece"
      -> fit_pca on pooled fg      (Stage 5, decomposition.py)
      -> fit detectors on "normal" (Stage 8, anomaly.py; fit_on config)
      -> per target piece:
           cluster PCA scores      (Stages 6-7, clustering.py)
           score anomalies         (Stage 8: self-fit maps + silicon-contrast map)
           clean + label regions   (Stage 9, postprocess.py)
           characterize regions    (Stages 10-11, regions.py)
           tile ROIs               (rois.py)
      -> aggregate ROI table across specimens (rois.py)

The baseline (silicon) and target (sio2) datasets are separate presets. Every
run produces BOTH anomaly products: the within-film maps (detectors fit per
``AnomalyConfig.fit_on``, default the film's own majority -- these drive the
flagged regions) and the silicon-baseline contrast map (always computed; the
document's literal hypothesis deliverable). Everything downstream of extraction
works on full-band spectra; pseudo-RGB is only for the figures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from config import (DatasetConfig, WorkflowConfig, DATASETS, DEFAULT_BASELINE)
from cube_io import Cube, iter_cube_paths, load_dataset_cube
from pieces import Piece, extract_pieces
from preprocessing import preprocess, saturation_mask
from decomposition import fit_pca, PcaModel
from clustering import cluster, cluster_map, cluster_metrics, ClusterResult
from anomaly import (fit_detectors, MahalanobisDetector, anomaly_map,
                      flag_threshold, to_probability)
from postprocess import clean_binary_map, label_regions
from regions import (characterize_regions, regions_to_table, RegionStats,
                      spectral_distance_map)
from rois import tile_rois, roi_feature_matrix, build_roi_table, Roi


# --------------------------------------------------------------------------
# Piece preparation (extract raw pieces -> preprocessed "analysis pieces")
# --------------------------------------------------------------------------

def prepare_pieces(ds_cfg: DatasetConfig, wf: WorkflowConfig,
                   verbose: bool = True) -> List[Piece]:
    """Load every cube in a dataset, split into pieces, and preprocess each.

    Piece extraction runs on the raw cube (spectral-angle vs the dish background);
    each resulting piece is then calibrated/smoothed/SNV'd on its own so the
    returned :class:`Piece` objects carry *analysis-ready* spectra in ``.data``.
    """
    wf.validate()
    pieces: List[Piece] = []
    for label, hdr in iter_cube_paths(ds_cfg):
        cube = load_dataset_cube(hdr, ds_cfg)
        sat = saturation_mask(cube.data, cube.ceiling)
        raw_pieces = extract_pieces(cube, wf.piece, valid_mask=~sat)
        if verbose:
            print(f"  {label}: {len(raw_pieces)} piece(s) "
                  f"[{cube.material}] from {cube.shape[0]}x{cube.shape[1]}")
        for rp in raw_pieces:
            piece_cube = Cube(
                data=rp.data, wavelengths=cube.wavelengths, shutter=cube.shutter,
                ceiling=cube.ceiling, path=hdr, label=rp.piece_id, material=cube.material,
            )
            pre = preprocess(piece_cube, wf.preprocess,
                             white_ref_hdr=ds_cfg.white_ref, dark_ref_hdr=ds_cfg.dark_ref)
            pieces.append(Piece(
                data=pre.data, mask=rp.mask, material=rp.material, piece_id=rp.piece_id,
                source_label=rp.source_label, bbox=rp.bbox, wavelengths=cube.wavelengths,
                reflectance_mean=pre.reflectance_mean, noise=pre.noise,
            ))
    return pieces


def pooled_foreground(pieces: List[Piece], cap: int, seed: int = 0) -> np.ndarray:
    """Stack in-mask spectra from all pieces, subsampled per piece to bound memory."""
    rng = np.random.default_rng(seed)
    per = max(1, cap // max(1, len(pieces)))
    chunks = []
    for p in pieces:
        fg = p.foreground_spectra()
        if fg.shape[0] > per:
            fg = fg[rng.choice(fg.shape[0], per, replace=False)]
        chunks.append(fg)
    return np.vstack(chunks) if chunks else np.empty((0, 0))


# --------------------------------------------------------------------------
# Per-piece analysis result
# --------------------------------------------------------------------------

@dataclass
class PieceAnalysis:
    piece: Piece
    pc_score_image: np.ndarray                 # (rows, cols, k), off-mask = NaN
    cluster_map: np.ndarray                    # (rows, cols) int, off-mask = -1
    cluster_metrics: dict
    anomaly_maps: Dict[str, np.ndarray]        # method -> (rows, cols) score map
    probability_map: np.ndarray                # (rows, cols) 0-1 anomaly probability
    baseline_map: np.ndarray                   # (rows, cols) distance-from-silicon contrast
    spectral_distance: np.ndarray              # (rows, cols) distance from piece mean spectrum
    flagged: np.ndarray                        # (rows, cols) bool, cleaned
    regions: List[RegionStats]
    region_table: object                       # pandas DataFrame
    rois: List[Roi]


@dataclass
class WorkflowResult:
    pca: PcaModel
    detectors: Dict[str, object]
    baseline_thresholds: Dict[str, float]
    analyses: List[PieceAnalysis]
    roi_table: object                          # pandas DataFrame across all pieces


# --------------------------------------------------------------------------
# Analyze one target piece
# --------------------------------------------------------------------------

def analyze_piece(piece: Piece, pca: PcaModel, detectors: Dict[str, object],
                  baseline_spectral: MahalanobisDetector,
                  thresholds: Dict[str, float], wf: WorkflowConfig) -> PieceAnalysis:
    """Run Stages 5-11 + ROI tiling on a single preprocessed piece."""
    mask = piece.mask
    fg = piece.foreground_spectra()                       # (n_fg, bands)
    feat = pca.transform(fg)                              # (n_fg, k)

    # --- Stage 6-7: cluster PCA scores, paint back to the image ---
    cres = cluster(feat, wf.cluster)
    cmap = cluster_map(cres, mask.shape, mask)
    cmetrics = cluster_metrics(feat, cres.labels)

    # --- Stage 8: anomaly scores. Two products, both label-free:
    # (1) detectors fit on the configured "normal" population (default: the
    #     film's own majority) -- these drive flags/regions;
    # (2) the silicon-baseline contrast map (always computed) -- the document's
    #     literal "relative to silicon baseline" deliverable.
    amaps: Dict[str, np.ndarray] = {}
    fg_scores: Dict[str, np.ndarray] = {}
    for name, det in detectors.items():
        s = det.score(feat)
        fg_scores[name] = s
        amaps[name] = anomaly_map(s, mask.shape, mask)
    primary = wf.anomaly.methods[0]
    probability_map = to_probability(amaps[primary])

    baseline_map = anomaly_map(baseline_spectral.score(fg), mask.shape, mask)

    # Stage 10 deliverable: distance of every pixel from the piece's own mean
    # spectrum (Euclidean in the analysis space).
    global_mean = fg.mean(axis=0)
    sdist = spectral_distance_map(piece.data, global_mean)
    sdist = np.where(mask, sdist, np.nan)

    # --- Stage 9: flag + spatially clean the primary anomaly map ---
    flag_flat = fg_scores[primary] > thresholds[primary]
    flag_img = np.zeros(mask.shape, dtype=bool)
    flag_img[mask] = flag_flat
    flagged = clean_binary_map(flag_img, wf.postproc)
    flagged &= mask                                       # spatial filters can bleed off-piece

    # --- Stage 10-11: characterize the surviving regions ---
    labels, n = label_regions(flagged)
    regions = characterize_regions(labels, n, piece.data, amaps[primary],
                                   baseline_detector=baseline_spectral,
                                   reflectance_mean=piece.reflectance_mean,
                                   pca=pca)
    region_table = regions_to_table(regions)

    # --- PC score image (full bbox; off-mask -> NaN for display) ---
    pc_img = pca.score_image(piece.data)
    pc_img = np.where(mask[:, :, None], pc_img, np.nan)

    # --- ROIs: attach PCA + anomaly to each tile ---
    rois = tile_rois(piece, wf.roi)
    if rois:
        roi_feats = pca.transform(roi_feature_matrix(rois))
        for r, f in zip(rois, roi_feats):
            r.pca = f
            r.anomaly = {name: float(det.score(f[None, :])[0])
                         for name, det in detectors.items()}

    return PieceAnalysis(
        piece=piece, pc_score_image=pc_img, cluster_map=cmap, cluster_metrics=cmetrics,
        anomaly_maps=amaps, probability_map=probability_map, baseline_map=baseline_map,
        spectral_distance=sdist, flagged=flagged, regions=regions,
        region_table=region_table, rois=rois,
    )


# --------------------------------------------------------------------------
# Top-level driver
# --------------------------------------------------------------------------

def run_workflow(target: str, wf: Optional[WorkflowConfig] = None,
                 baseline: str = DEFAULT_BASELINE, verbose: bool = True) -> WorkflowResult:
    """Run the full pipeline: fit on the silicon baseline, analyze the target.

    ``target``/``baseline`` are dataset preset names. Returns a
    :class:`WorkflowResult` holding the shared PCA/detectors and one
    :class:`PieceAnalysis` per target piece, plus the aggregated ROI table.
    """
    wf = wf or WorkflowConfig()
    wf.validate()
    target_cfg = DATASETS[target]
    baseline_cfg = DATASETS[baseline]

    if verbose:
        print(f"Baseline (normal) dataset: {baseline!r} [{baseline_cfg.material}]")
    baseline_pieces = prepare_pieces(baseline_cfg, wf, verbose=verbose)
    if verbose:
        print(f"Target dataset: {target!r} [{target_cfg.material}]")
    target_pieces = prepare_pieces(target_cfg, wf, verbose=verbose)

    # --- Stage 5: PCA on pooled foreground (baseline + target) ---
    pooled = pooled_foreground(baseline_pieces + target_pieces, wf.pca.max_fit_pixels, wf.pca.seed)
    pca = fit_pca(pooled, wf.pca)
    if verbose:
        evr = pca.explained_variance_ratio
        print(f"PCA explained variance: " + ", ".join(f"PC{i+1}={v:.1%}" for i, v in enumerate(evr)))

    # --- Stage 8: fit anomaly detectors on the "normal" population ---
    # fit_on="self" -> the target's own majority (finds localized anomalies within
    # the film); fit_on="baseline" -> the external silicon baseline (material
    # contrast). Thresholds come from the same population the detectors were fit on.
    baseline_fg = pooled_foreground(baseline_pieces, wf.anomaly.max_fit_pixels, wf.anomaly.seed)
    if wf.anomaly.fit_on == "baseline":
        normal_fg = baseline_fg
    else:
        normal_fg = pooled_foreground(target_pieces, wf.anomaly.max_fit_pixels, wf.anomaly.seed)
    normal_feat = pca.transform(normal_fg)
    detectors = fit_detectors(normal_feat, wf.anomaly)
    thresholds = {name: flag_threshold(det.score(normal_feat), wf.anomaly.anomaly_percentile)
                  for name, det in detectors.items()}
    if verbose:
        print(f"Anomaly detectors fit on {wf.anomaly.fit_on!r} population "
              f"({normal_fg.shape[0]} spectra); methods={wf.anomaly.methods}")
    # Spectral-space Mahalanobis on silicon, always, for the region
    # "distance from silicon baseline" feature.
    baseline_spectral = MahalanobisDetector().fit(baseline_fg)

    # --- Per-target-piece analysis ---
    analyses = [analyze_piece(p, pca, detectors, baseline_spectral, thresholds, wf)
                for p in target_pieces]

    # --- Aggregate ROI table across all target pieces ---
    all_rois: List[Roi] = [r for a in analyses for r in a.rois]
    wl = target_pieces[0].wavelengths if target_pieces else None
    roi_table = build_roi_table(all_rois, wl) if all_rois else None

    return WorkflowResult(pca=pca, detectors=detectors, baseline_thresholds=thresholds,
                          analyses=analyses, roi_table=roi_table)
