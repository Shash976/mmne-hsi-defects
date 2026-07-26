"""Lazy, memory-bounded cube reading (CubeReader + streamed reference mean).

These exercise the path that lets the interactive tuners open multi-GB scans
without materializing the whole cube: a decimated working grid streamed off disk
plus full-resolution per-pixel reads. Correctness is checked against the plain
in-memory equivalents.
"""

import numpy as np
import spectral

from hsi_workflow.cube_io import (open_cube_reader, array_cube_reader,
                                  load_reference_spectrum, CubeReader)


def _write_envi(tmp_path, arr, name="cube", shutter=None, interleave="bil"):
    hdr = str(tmp_path / f"{name}.hdr")
    meta = {} if shutter is None else {"shutter": shutter}
    spectral.envi.save_image(hdr, arr, metadata=meta, dtype=np.float32,
                             interleave=interleave, force=True)
    return hdr


# --- array backend (demo/tests) -------------------------------------------

def test_array_reader_decimated_matches_strided():
    arr = np.random.default_rng(0).random((40, 30, 8))
    rdr = array_cube_reader(arr)
    assert rdr.shape == (40, 30, 8)
    for step in (1, 2, 3, 5):
        assert np.array_equal(rdr.decimated(step), arr[::step, ::step, :])


def test_array_reader_pixel_and_patch():
    arr = np.random.default_rng(1).random((20, 25, 6))
    rdr = array_cube_reader(arr)
    assert np.array_equal(rdr.pixel(7, 9), arr[7, 9, :])
    assert rdr.pixel(7, 9).shape == (6,)
    assert np.array_equal(rdr.patch(3, 8, 4, 10), arr[3:8, 4:10, :])


def test_array_reader_crop_offsets_every_read():
    arr = np.random.default_rng(2).random((30, 30, 5))
    rdr = array_cube_reader(arr, crop=(5, 25, 4, 20))
    assert rdr.shape == (20, 16, 5)
    assert np.array_equal(rdr.pixel(0, 0), arr[5, 4, :])          # region (0,0) == full (5,4)
    assert np.array_equal(rdr.decimated(2), arr[5:25:2, 4:20:2, :])
    assert np.array_equal(rdr.patch(0, 3, 0, 3), arr[5:8, 4:7, :])


# --- file backend (streamed off disk) -------------------------------------

def test_file_reader_matches_full_cube(tmp_path):
    arr = np.random.default_rng(3).random((50, 37, 10)).astype(np.float32)
    hdr = _write_envi(tmp_path, arr)
    rdr = open_cube_reader(hdr, block_rows=8)       # tiny blocks -> exercise streaming
    assert isinstance(rdr, CubeReader)
    assert rdr.shape == (50, 37, 10)
    for step in (1, 3, 4, 7):
        assert np.allclose(rdr.decimated(step), arr[::step, ::step, :], atol=1e-5)
    assert np.allclose(rdr.pixel(41, 20), arr[41, 20, :], atol=1e-5)
    assert np.allclose(rdr.patch(10, 14, 5, 9), arr[10:14, 5:9, :], atol=1e-5)


def test_file_reader_block_size_does_not_change_result(tmp_path):
    # streaming in different block sizes must yield the identical decimated grid
    arr = np.random.default_rng(6).random((33, 20, 4)).astype(np.float32)
    hdr = _write_envi(tmp_path, arr)
    a = open_cube_reader(hdr, block_rows=1).decimated(3)
    b = open_cube_reader(hdr, block_rows=7).decimated(3)
    c = open_cube_reader(hdr, block_rows=1000).decimated(3)
    assert np.allclose(a, b) and np.allclose(b, c)
    assert np.allclose(a, arr[::3, ::3, :], atol=1e-5)


def test_file_reader_crop(tmp_path):
    arr = np.random.default_rng(4).random((40, 40, 6)).astype(np.float32)
    hdr = _write_envi(tmp_path, arr)
    rdr = open_cube_reader(hdr, crop=(6, 34, 3, 25), block_rows=5)
    assert rdr.shape == (28, 22, 6)
    assert np.allclose(rdr.decimated(3), arr[6:34:3, 3:25:3, :], atol=1e-5)
    assert np.allclose(rdr.pixel(0, 0), arr[6, 3, :], atol=1e-5)


def test_reference_mean_is_exact_and_streamed(tmp_path):
    arr = np.random.default_rng(5).random((45, 30, 7)).astype(np.float32)
    hdr = _write_envi(tmp_path, arr, name="white_ref", shutter=2.5)
    mean, shutter = load_reference_spectrum(hdr, block_rows=8)
    assert np.allclose(mean, arr.reshape(-1, arr.shape[-1]).mean(axis=0), atol=1e-5)
    assert shutter == 2.5
