"""hsi_workflow -- unsupervised spectral-anomaly-detection for semiconductor films.

Implements the revised pipeline from ``Revised Research Objective.md``: detect
and spatially localize *spectrally anomalous* regions in SiO2 thin films, using
bare silicon as a label-free baseline population (no reference spectra, no FEA,
no composition/unmixing). The earlier LIG-specific analysis lives in ``../legacy``
and is referenced (Mahalanobis/RX) but not imported here.

Stage map (module : revised stage):

    io / config          -- loading + dataset presets + per-stage config
    pieces               -- Stage 3.1  raw multi-piece scan -> individual pieces
    preprocessing        -- Stages 2-3 calibration, SG smoothing, baseline, SNV
    rois                 -- fixed-patch ROI table (cross-specimen, leakage-free)
    explore / viz        -- Stage 4    exploratory + per-stage figures
    decomposition        -- Stage 5    PCA
    clustering           -- Stages 6-7 KMeans/DBSCAN/GMM + cluster maps + metrics
    anomaly              -- Stage 8    IsolationForest/LOF/Mahalanobis/OneClassSVM
    postprocess          -- Stage 9    median/opening/connected-components cleanup
    regions              -- Stages 10-11 quantitative maps + region characterization
    pipeline             -- orchestrator tying the stages together
    run_extract / run_explore / run_analyze -- CLIs

Optical density (``optical_density.py``) is retained but off the default path.
"""

__all__ = [
    "config", "io", "pieces", "preprocessing", "segmentation", "rois",
    "dataset", "explore", "decomposition", "clustering", "anomaly",
    "postprocess", "regions", "viz", "optical_density", "pipeline",
]
