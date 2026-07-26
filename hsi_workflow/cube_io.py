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

from .config import DatasetConfig


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


def _read_meta(img, hdr_path: str):
    """Pull the header fields we act on from an opened SpyFile.

    Returns ``(wavelengths, shutter, ceiling, label)`` -- shared by the eager
    :func:`load_cube` and the lazy :class:`CubeReader` so both agree on how a
    header maps onto our metadata.
    """
    wavelengths = (np.asarray(img.bands.centers, dtype=np.float64)
                   if img.bands is not None and img.bands.centers is not None else None)
    shutter = float(img.metadata.get("shutter", 1.0))
    ceiling = float(img.metadata.get("ceiling", np.inf))
    label = str(img.metadata.get("label") or _stem(hdr_path))
    return wavelengths, shutter, ceiling, label


def load_cube(hdr_path: str, material: str = "sio2") -> Cube:
    """Load one ENVI cube header/data pair into a ``Cube``.

    ``material`` tags the sample type (defaults to ``"sio2"``); callers that have
    a :class:`~hsi_workflow.config.DatasetConfig` should pass ``cfg.material`` so
    the tag propagates downstream (see :func:`load_dataset_cube`).

    This materializes the *whole* cube as float64. For big scans that only need a
    decimated view (the interactive tuners), prefer :func:`open_cube_reader`,
    which never holds the full cube in memory.
    """
    img = spectral.open_image(hdr_path)
    data = np.asarray(img.load(), dtype=np.float64)
    wavelengths, shutter, ceiling, label = _read_meta(img, hdr_path)
    return Cube(data=data, wavelengths=wavelengths, shutter=shutter,
                ceiling=ceiling, path=hdr_path, label=label, material=material)


def load_dataset_cube(hdr_path: str, cfg: "DatasetConfig") -> Cube:
    """Load a cube, tagging it with the dataset preset's ``material``."""
    return load_cube(hdr_path, material=cfg.material)


# --------------------------------------------------------------------------
# Lazy reading (memory-bounded working views for the interactive tuners)
# --------------------------------------------------------------------------

class CubeReader:
    """Memory-bounded reader over one ENVI cube (or an in-memory array).

    The scans are multi-GB as float64 (e.g. 1417x900x300 -> ~2.9 GB), which does
    not fit alongside everything else on a workstation. The tuners never need the
    whole cube at full resolution: they compute on a *decimated* working grid and
    only ever inspect a handful of *individual* pixel spectra at full resolution.
    This reader serves exactly those two needs without materializing the cube:

    - :meth:`decimated` streams contiguous row *blocks* off disk (BIL/BIP-friendly
      -- strided ``img[::step]`` reads are ~30x slower here) and keeps every
      ``step``-th row/col, so peak memory is one block, not the whole cube.
    - :meth:`pixel` / :meth:`patch` read single pixels/small windows at full
      resolution via the SpyFile's own indexing (interleave-correct, ~ms).

    An optional ``crop`` ``(r0, r1, c0, c1)`` restricts every read to a spatial
    window; region coordinates passed to the methods are relative to that crop,
    matching the debug tools' existing ``--crop`` semantics. Backing the reader
    with an ndarray instead (see :func:`array_cube_reader`) gives the same
    interface for the synthetic/demo path, so callers stay backend-agnostic.
    """

    def __init__(self, *, img=None, arr=None, wavelengths=None, shutter=1.0,
                 ceiling=np.inf, label="cube", material: str = "sio2",
                 crop=None, block_rows: int = 128):
        if (img is None) == (arr is None):
            raise ValueError("CubeReader needs exactly one of img/arr")
        self._img = img
        self._arr = None if arr is None else np.asarray(arr)
        self.wavelengths = wavelengths
        self.shutter = shutter
        self.ceiling = ceiling
        self.label = label
        self.material = material
        self._block_rows = max(1, int(block_rows))

        full_rows, full_cols = (self._arr.shape[:2] if img is None else img.shape[:2])
        bands = (self._arr.shape[-1] if img is None else img.shape[-1])
        r0, r1, c0, c1 = crop if crop is not None else (0, full_rows, 0, full_cols)
        self._r0, self._r1 = int(r0), int(r1)
        self._c0, self._c1 = int(c0), int(c1)
        self.shape = (self._r1 - self._r0, self._c1 - self._c0, int(bands))

    @property
    def n_bands(self) -> int:
        return self.shape[-1]

    def decimated(self, step: int) -> np.ndarray:
        """Dense float64 grid equal to ``region[::step, ::step, :]``.

        For the file backend this streams row blocks so only one block is ever
        resident; the result matches a plain strided slice exactly.
        """
        step = max(1, int(step))
        r0, r1, c0, c1 = self._r0, self._r1, self._c0, self._c1
        if self._arr is not None:
            return np.asarray(self._arr[r0:r1:step, c0:c1:step, :], dtype=np.float64)
        parts = []
        for br0 in range(r0, r1, self._block_rows):
            br1 = min(br0 + self._block_rows, r1)
            block = np.asarray(self._img[br0:br1, c0:c1, :])
            first = (-(br0 - r0)) % step        # first kept row within this block
            parts.append(block[first::step, ::step, :])
        return np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float64)

    def pixel(self, r: int, c: int) -> np.ndarray:
        """Full-resolution spectrum ``(bands,)`` float64 at region coords ``(r, c)``."""
        src = self._arr if self._arr is not None else self._img
        return np.asarray(src[self._r0 + int(r), self._c0 + int(c), :],
                          dtype=np.float64).ravel()

    def patch(self, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
        """Full-resolution ``(r1-r0, c1-c0, bands)`` float64 patch at region coords."""
        src = self._arr if self._arr is not None else self._img
        return np.asarray(src[self._r0 + int(r0):self._r0 + int(r1),
                              self._c0 + int(c0):self._c0 + int(c1), :],
                          dtype=np.float64)


def open_cube_reader(hdr_path: str, material: str = "sio2", crop=None,
                     block_rows: int = 128) -> CubeReader:
    """Open an ENVI cube as a memory-bounded :class:`CubeReader` (no pixel load)."""
    img = spectral.open_image(hdr_path)
    wavelengths, shutter, ceiling, label = _read_meta(img, hdr_path)
    return CubeReader(img=img, wavelengths=wavelengths, shutter=shutter,
                      ceiling=ceiling, label=label, material=material,
                      crop=crop, block_rows=block_rows)


def array_cube_reader(arr: np.ndarray, wavelengths=None, shutter: float = 1.0,
                      ceiling: float = np.inf, label: str = "array",
                      material: str = "sio2", crop=None) -> CubeReader:
    """Wrap an in-memory cube in the :class:`CubeReader` interface (demo/tests)."""
    return CubeReader(arr=arr, wavelengths=wavelengths, shutter=shutter,
                      ceiling=ceiling, label=label, material=material, crop=crop)


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
def load_reference_spectrum(hdr_path: str, block_rows: int = 128):
    """Whole-frame mean spectrum + shutter time for a white/dark reference cube.

    The reference cubes are full-frame (several GB as float64), so this streams
    contiguous row blocks and accumulates the sum rather than loading the whole
    cube -- peak memory is one block. The result is exact (a mean over all
    pixels), not a decimated estimate. Cached: each reference is read and reduced
    only once per process.
    """
    img = spectral.open_image(hdr_path)
    _, shutter, _, _ = _read_meta(img, hdr_path)
    rows, cols, bands = img.shape
    acc = np.zeros(bands, dtype=np.float64)
    for r0 in range(0, rows, max(1, int(block_rows))):
        r1 = min(r0 + block_rows, rows)
        block = np.asarray(img[r0:r1, :, :], dtype=np.float64)
        acc += block.reshape(-1, bands).sum(axis=0)
    mean_spectrum = acc / (rows * cols)
    return mean_spectrum, shutter


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
