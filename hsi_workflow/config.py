"""Dataset presets and per-stage configuration for the revised HSI workflow.

The revised objective (see ``Revised Research Objective.md``) is an
**unsupervised spectral-anomaly-detection** pipeline for SiO2 thin films, with
bare silicon as a *baseline/control* population. This module holds two kinds of
configuration:

- ``DatasetConfig`` -- where a scan lives, how it is named/calibrated, and what
  *material* it is (silicon vs sio2). One preset per physical scan. Adding a new
  scan = adding a preset; no code changes elsewhere.
- Per-stage config dataclasses (``PieceConfig``, ``RoiConfig``, ``PcaConfig``,
  ``ClusterConfig``, ``AnomalyConfig``, ``PostprocConfig``) plus the existing
  ``PreprocessConfig``. ``WorkflowConfig`` bundles them so the orchestrator can
  be driven by a single object, and any one stage can be re-tuned in isolation.

Design note: every knob that changes scientific behaviour lives here as a plain
dataclass field with a short comment, so the pipeline can be reconfigured
without touching the stage implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# --------------------------------------------------------------------------
# Dataset presets
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetConfig:
    """Location + naming + calibration + material for one scan (or scan family).

    ``hdr_glob`` finds the cube header(s) under ``data_dir``. It can be a wide
    glob (``*.bip.hdr``, LIG's many ROI scans) or a single filename (one
    multi-piece SiO2 scan). ``pair_regex`` is optional: when set (LIG), cubes are
    grouped into sample/ROI pairs; when ``None``, every matching header is an
    independent cube named by its file stem.

    ``material`` tags the physical sample type -- ``"silicon"`` (the anomaly
    baseline population) or ``"sio2"`` (the experimental samples). It rides along
    to every ``Piece`` and ROI so the anomaly stage can fit "normal" on silicon
    and score sio2. ``background`` is a hint for piece extraction (the dish/holder
    colour); the default full-spectrum extractor keys on a border-estimated
    background spectrum, so this is informational rather than load-bearing.
    """

    name: str
    data_dir: str
    hdr_glob: str = "*.bip.hdr"
    pair_regex: Optional[str] = None
    white_ref: Optional[str] = None
    dark_ref: Optional[str] = None
    material: str = "sio2"            # {"silicon", "sio2"} -- physical sample type
    background: str = "auto"          # {"auto", "dark", "white"} -- dish/holder hint
    geometry: str = "reflectance"


_HSI_ROOT = r"C:\Users\shash\OneDrive - purdue.edu\Summer\hsi"

# Shared white/dark references live at the hsi root (not under lig_dataset).
_WHITE_REF = rf"{_HSI_ROOT}\calibration_whitedark\white_ref.bil.hdr"
_DARK_REF = rf"{_HSI_ROOT}\calibration_whitedark\dark_correction.bil.hdr"

# --- LIG: secondary test dataset (single-piece ROI scans, sample/ROI-paired). ---
LIG = DatasetConfig(
    name="lig",
    data_dir=rf"{_HSI_ROOT}\lig_dataset\roi_scans",
    hdr_glob="*.bip.hdr",
    pair_regex=r"^(?P<sample>.+)-[Rr][Oo][Ii]-(?P<roi>\d+)\.bip\.hdr$",
    white_ref=_WHITE_REF,
    dark_ref=_DARK_REF,
    material="sio2",          # LIG is not a semiconductor; treated as experimental here
    background="auto",
    geometry="reflectance",
)

# --- SiO2 semiconductor scans (the focus). Each is one multi-piece scan. ---
_SIO2_DIR = rf"{_HSI_ROOT}\sio2"

SIO2_BARE_SI = DatasetConfig(
    name="sio2_bare_si",
    data_dir=_SIO2_DIR,
    hdr_glob="bare silicon all.bip.hdr",
    pair_regex=None,
    white_ref=_WHITE_REF,
    dark_ref=_DARK_REF,
    material="silicon",       # the baseline / control population
    background="dark",
)

SIO2_DISH_WHITE_20 = DatasetConfig(
    name="sio2_dish_white_20",
    data_dir=_SIO2_DIR,
    hdr_glob="sio2 all 20 dish white.bil.hdr",
    pair_regex=None,
    white_ref=_WHITE_REF,
    dark_ref=_DARK_REF,
    material="sio2",
    background="white",
)

SIO2_DISH_BLACK = DatasetConfig(
    name="sio2_dish_black",
    data_dir=_SIO2_DIR,
    hdr_glob="Dish on Black - 1.bip.hdr",
    pair_regex=None,
    white_ref=_WHITE_REF,
    dark_ref=_DARK_REF,
    material="sio2",
    background="dark",
)

SIO2_DISH_WHITE_1 = DatasetConfig(
    name="sio2_dish_white_1",
    data_dir=_SIO2_DIR,
    hdr_glob="Dish on White 1.bip.hdr",
    pair_regex=None,
    white_ref=_WHITE_REF,
    dark_ref=_DARK_REF,
    material="sio2",
    background="white",
)

DATASETS = {cfg.name: cfg for cfg in (
    LIG, SIO2_BARE_SI, SIO2_DISH_WHITE_20, SIO2_DISH_BLACK, SIO2_DISH_WHITE_1,
)}

# The default silicon baseline dataset used by the anomaly stage when a caller
# does not pass its own baseline. Kept here so the choice is configurable in one
# place rather than hard-coded in the orchestrator.
DEFAULT_BASELINE = "sio2_bare_si"


# --------------------------------------------------------------------------
# Stage 2-3: preprocessing / optical-density options
# --------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    """Knobs for calibration + preprocessing (revised Stages 2-3).

    Defaults reproduce the revised path: exposure-normalized reflectance ->
    Savitzky-Golay smoothing -> (optional baseline) -> SNV. Optical density is
    retained in the codebase but *off* the default analysis path
    (``od_method="none"``).
    """

    # --- Stage 2: registration + calibration ---
    register: bool = False            # align a separate background/reference scan to the sample
    background: str = "dark"          # {"dark", "none"} -- subtract the dark reference spectrum
    calibrate: bool = True            # exposure-normalized (raw-dark)/(white-dark) reflectance
    # --- Stage 3.2: spectral smoothing ---
    smooth: str = "savgol"            # {"savgol", "none"}
    sg_window: int = 11               # Savitzky-Golay window (odd, in bands)
    sg_polyorder: int = 2             # Savitzky-Golay polynomial order
    # --- Stage 3.4: optional baseline correction ---
    baseline: str = "none"            # {"none", "poly"}
    baseline_order: int = 2           # polynomial order for baseline="poly"
    # --- Stage 3.3: spectral normalization ---
    normalize: str = "snv"            # {"none", "snv"} -- per-pixel SNV on the smoothed reflectance
    # --- Stage 10 (legacy path): optical density; off by default ---
    od_method: str = "none"           # {"none", "substrate", "white", "reference_scan"}
    eps: float = 1e-6                 # floor for ratios inside log / division
    # --- segmentation (film/substrate; only needed for od_method="substrate") ---
    invert_foreground: bool = False
    seed: int = 0

    def validate(self) -> None:
        if self.background not in ("dark", "none"):
            raise ValueError(f"background must be 'dark' or 'none', got {self.background!r}")
        if self.smooth not in ("savgol", "none"):
            raise ValueError(f"smooth must be 'savgol' or 'none', got {self.smooth!r}")
        if self.smooth == "savgol":
            if self.sg_window % 2 == 0 or self.sg_window < 3:
                raise ValueError(f"sg_window must be odd and >=3, got {self.sg_window}")
            if self.sg_polyorder >= self.sg_window:
                raise ValueError("sg_polyorder must be < sg_window")
        if self.baseline not in ("none", "poly"):
            raise ValueError(f"baseline must be 'none' or 'poly', got {self.baseline!r}")
        if self.normalize not in ("none", "snv"):
            raise ValueError(f"normalize must be 'none' or 'snv', got {self.normalize!r}")
        if self.od_method not in ("none", "substrate", "white", "reference_scan"):
            raise ValueError(f"od_method invalid: {self.od_method!r}")


# --------------------------------------------------------------------------
# Stage 3.1: piece extraction
# --------------------------------------------------------------------------

@dataclass
class PieceConfig:
    """Knobs for splitting a multi-piece scan into individual piece sub-cubes.

    The extractor estimates the dish/holder spectrum from a border frame and
    flags foreground pixels by spectral distance (angle / Mahalanobis) over the
    full band range -- never by RGB brightness. Morphology then cleans the mask:
    ``opening`` removes thin rim arcs and dust; ``closing`` merges within-piece
    gaps (patterned SiO2 devices otherwise fragment); ``min_area`` drops anything
    too small to be a real piece.
    """

    method: str = "sam"               # {"sam", "mahalanobis", "kmeans"} foreground backend
    border_width: int = 8             # px frame sampled to estimate the background spectrum
    threshold: str = "otsu"           # {"otsu", "percentile"} on the distance map
    threshold_percentile: float = 80.0  # used when threshold="percentile"
    open_iter: int = 2                # binary opening iterations (remove arcs/dust)
    close_iter: int = 6               # binary closing iterations (merge device structure)
    fill_holes: bool = True
    min_area: int = 1000              # px; components smaller than this are discarded
    watershed_split: bool = False     # split touching pieces via distance-transform watershed
    seed: int = 0

    def validate(self) -> None:
        if self.method not in ("sam", "mahalanobis", "kmeans"):
            raise ValueError(f"piece method invalid: {self.method!r}")
        if self.threshold not in ("otsu", "percentile"):
            raise ValueError(f"threshold must be 'otsu' or 'percentile', got {self.threshold!r}")


# --------------------------------------------------------------------------
# ROI tiling
# --------------------------------------------------------------------------

@dataclass
class RoiConfig:
    """Fixed-patch ROI tiling within each piece (the cross-specimen ML samples).

    Each piece is tiled into ``patch`` x ``patch`` windows stepped by ``stride``;
    a patch is kept only if at least ``min_coverage`` of it lies inside the piece
    mask (so ROIs never straddle dish/edges). Tune ``patch``/``stride`` so each
    piece yields ~100-300 ROIs (the document's target).
    """

    patch: int = 32
    stride: int = 32                  # == patch => non-overlapping tiles
    min_coverage: float = 0.85        # fraction of patch pixels that must be in-mask
    save_patches: bool = False        # also persist each ROI as its own sub-cube

    def validate(self) -> None:
        if self.patch < 2:
            raise ValueError("patch must be >= 2")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        if not 0 < self.min_coverage <= 1:
            raise ValueError("min_coverage must be in (0, 1]")


# --------------------------------------------------------------------------
# Stage 5: PCA
# --------------------------------------------------------------------------

@dataclass
class PcaConfig:
    """Dimensionality reduction (Stage 5)."""

    n_components: int = 3
    whiten: bool = False
    max_fit_pixels: int = 200_000     # subsample cap when fitting on pooled spectra
    seed: int = 0

    def validate(self) -> None:
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1")


# --------------------------------------------------------------------------
# Stage 6-7: clustering
# --------------------------------------------------------------------------

@dataclass
class ClusterConfig:
    """Unsupervised clustering on PCA scores (Stages 6-7)."""

    method: str = "kmeans"            # {"kmeans", "dbscan", "gmm"}
    n_clusters: int = 4               # kmeans / gmm
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 20
    max_fit_pixels: int = 100_000     # subsample cap for the fit
    seed: int = 0

    def validate(self) -> None:
        if self.method not in ("kmeans", "dbscan", "gmm"):
            raise ValueError(f"cluster method invalid: {self.method!r}")


# --------------------------------------------------------------------------
# Stage 8: anomaly scoring
# --------------------------------------------------------------------------

@dataclass
class AnomalyConfig:
    """Anomaly scoring (Stage 8).

    ``methods`` lists the detectors to run; each is fit on the *normal*
    population and used to score every pixel/ROI. ``contamination`` feeds
    sklearn's Isolation Forest / LOF; ``anomaly_percentile`` sets the flag
    threshold on the *normal* score distribution.

    ``fit_on`` chooses what "normal" means:

    - ``"self"`` (default) -- fit on the target's own majority population, so the
      detector finds regions that differ from the bulk SiO2 film (small, localized
      anomalies -- the document's intent). Fitting "normal" on a *different*
      material (silicon) instead flags the entire film as anomalous.
    - ``"baseline"`` -- fit on the external silicon baseline dataset; every pixel
      is scored by how unlike bare silicon it is (a material-contrast map).

    The silicon baseline is *always* used for the per-region "distance from
    silicon baseline" feature regardless of ``fit_on``.
    """

    methods: List[str] = field(default_factory=lambda: ["iforest", "mahalanobis"])
    fit_on: str = "self"              # {"self", "baseline"}
    contamination: float = 0.05
    anomaly_percentile: float = 97.5
    max_fit_pixels: int = 100_000
    seed: int = 0

    def validate(self) -> None:
        known = {"iforest", "lof", "mahalanobis", "ocsvm"}
        bad = set(self.methods) - known
        if bad:
            raise ValueError(f"unknown anomaly methods: {sorted(bad)} (known: {sorted(known)})")
        if self.fit_on not in ("self", "baseline"):
            raise ValueError(f"fit_on must be 'self' or 'baseline', got {self.fit_on!r}")


# --------------------------------------------------------------------------
# Stage 9: spatial postprocessing
# --------------------------------------------------------------------------

@dataclass
class PostprocConfig:
    """Spatial cleanup of the pixel-level anomaly/cluster map (Stage 9)."""

    median_size: int = 3              # median filter window (px); 0/1 disables
    opening_radius: int = 1           # morphological opening disk radius; 0 disables
    min_component: int = 25           # drop flagged components smaller than this (px)

    def validate(self) -> None:
        if self.median_size < 0 or self.opening_radius < 0:
            raise ValueError("median_size and opening_radius must be >= 0")


# --------------------------------------------------------------------------
# Composite workflow config
# --------------------------------------------------------------------------

@dataclass
class WorkflowConfig:
    """Bundles every per-stage config so the orchestrator takes one object.

    Construct with defaults and override individual stages, e.g.::

        cfg = WorkflowConfig()
        cfg.cluster.method = "gmm"
        cfg.anomaly.methods = ["iforest", "lof", "mahalanobis"]
    """

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    piece: PieceConfig = field(default_factory=PieceConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    pca: PcaConfig = field(default_factory=PcaConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    postproc: PostprocConfig = field(default_factory=PostprocConfig)

    def validate(self) -> None:
        for stage in (self.preprocess, self.piece, self.roi, self.pca,
                      self.cluster, self.anomaly, self.postproc):
            stage.validate()
