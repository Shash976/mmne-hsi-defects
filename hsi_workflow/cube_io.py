"""Cube loading and discovery (ENVI .bip/.bil pairs via the `spectral` package).

Kept deliberately small: a ``Cube`` value object plus discovery helpers that
work for both the paired LIG scans and the flat list of forthcoming sio2 crops.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import spectral

from config import DatasetConfig


@dataclass
class Cube:
    """A loaded hyperspectral cube plus the header metadata we act on.

    ``data`` is (rows, cols, bands) float64. ``shutter`` and ``ceiling`` come
    from the ENVI header (exposure time and sensor-saturation DN); ``wavelengths``
    is the per-band center list (nm) or ``None`` if absent. ``material``
    (``"silicon"`` / ``"sio2"``) is carried from the dataset preset so the
    anomaly stage can distinguish the baseline population from the experimental
    samples; it rides along to every ``Piece`` and ROI derived from this cube.
    """

    data: np.ndarray
    wavelengths: Optional[np.ndarray]
    shutter: float
    ceiling: float
    path: str
    label: str
    material: str = "sio2"

    @property
    def shape(self):
        return self.data.shape

    @property
    def n_bands(self) -> int:
        return self.data.shape[-1]


def load_cube(hdr_path: str, material: str = "sio2") -> Cube:
    """Load one ENVI cube header/data pair into a ``Cube``.

    ``material`` tags the sample type (defaults to ``"sio2"``); callers that have
    a :class:`~hsi_workflow.config.DatasetConfig` should pass ``cfg.material`` so
    the tag propagates downstream (see :func:`load_dataset_cube`).
    """
    img = spectral.open_image(hdr_path)
    data = np.asarray(img.load(), dtype=np.float64)
    wavelengths = (np.asarray(img.bands.centers, dtype=np.float64)
                   if img.bands is not None and img.bands.centers is not None else None)
    shutter = float(img.metadata.get("shutter", 1.0))
    ceiling = float(img.metadata.get("ceiling", np.inf))
    label = str(img.metadata.get("label") or _stem(hdr_path))
    return Cube(data=data, wavelengths=wavelengths, shutter=shutter,
                ceiling=ceiling, path=hdr_path, label=label, material=material)


def load_dataset_cube(hdr_path: str, cfg: "DatasetConfig") -> Cube:
    """Load a cube, tagging it with the dataset preset's ``material``."""
    return load_cube(hdr_path, material=cfg.material)


def save_envi_cube(hdr_path: str, data: np.ndarray,
                   wavelengths: Optional[np.ndarray] = None,
                   material: Optional[str] = None, dtype=np.float32) -> str:
    """Write an ndarray as an ENVI ``.hdr``/data pair (wavelengths preserved).

    Used to persist cropped piece/ROI sub-cubes so the organized dataset is made
    of standard, reloadable ENVI cubes. Returns the header path.
    """
    meta = {}
    if wavelengths is not None:
        meta["wavelength"] = [float(w) for w in wavelengths]
        meta["wavelength units"] = "nm"
    if material is not None:
        meta["material"] = material
    spectral.envi.save_image(hdr_path, np.asarray(data), metadata=meta, dtype=dtype, force=True)
    print("Saved ENVI cube:", hdr_path)
    print("Saved Image:", hdr_path[:-4] + ".img")
    return hdr_path


@lru_cache(maxsize=8)
def load_reference_spectrum(hdr_path: str):
    """Whole-frame mean spectrum + shutter time for a white/dark reference cube.

    Cached: the white/dark reference cubes are large (~750 MB) and reused for
    every piece/scan, so we read and reduce each one only once per process.
    """
    cube = load_cube(hdr_path)
    mean_spectrum = cube.data.reshape(-1, cube.n_bands).mean(axis=0)
    return mean_spectrum, cube.shutter


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def _stem(hdr_path: str) -> str:
    base = os.path.basename(hdr_path)
    for suffix in (".bip.hdr", ".bil.hdr", ".hdr"):
        if base.lower().endswith(suffix):
            return base[: -len(suffix)]
    return base


def find_lig_pairs(cfg: DatasetConfig) -> Dict[str, Dict[int, str]]:
    """Group cubes into {sample_id: {roi_num: hdr_path}} using ``cfg.pair_regex``."""
    if cfg.pair_regex is None:
        raise ValueError(f"dataset {cfg.name!r} has no pair_regex; use discover_cubes instead")
    pat = re.compile(cfg.pair_regex)
    pairs: Dict[str, Dict[int, str]] = {}
    for hdr in glob.glob(os.path.join(cfg.data_dir, cfg.hdr_glob)):
        m = pat.match(os.path.basename(hdr))
        if not m:
            continue
        pairs.setdefault(m.group("sample"), {})[int(m.group("roi"))] = hdr
    return dict(sorted(pairs.items()))


def discover_cubes(cfg: DatasetConfig) -> Dict[str, str]:
    """Flat {name: hdr_path} for datasets without ROI pairing (e.g. sio2 crops)."""
    found = {}
    for hdr in sorted(glob.glob(os.path.join(cfg.data_dir, cfg.hdr_glob))):
        found[_stem(hdr)] = hdr
    return found


def iter_cube_paths(cfg: DatasetConfig) -> List[tuple]:
    """Unified iteration order for a dataset, as (label, hdr_path) pairs.

    Paired datasets yield ``("<sample>-roi<n>", path)``; flat datasets yield
    ``("<stem>", path)``. Lets the CLI treat both the same way.
    """
    if cfg.pair_regex is not None:
        out = []
        for sample, rois in find_lig_pairs(cfg).items():
            for roi_num, hdr in sorted(rois.items()):
                out.append((f"{sample}-roi{roi_num}", hdr))
        return out
    return list(discover_cubes(cfg).items())
