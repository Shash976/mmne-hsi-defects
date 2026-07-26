import numpy as np
import pytest
from dataclasses import replace

from hsi_workflow.config import PieceConfig
from hsi_workflow.cube_io import Cube
from hsi_workflow.pieces import (foreground_distance, flat_field_correct,
                                 extract_pieces, _background_pixels,
                                 _erode_analysis_mask, euclidean_distance,
                                 spectral_angle)


def _cube_with_pieces(rows=60, cols=60, bands=20, seed=0):
    """Dish background with two bright rectangular pieces (distinct spectrum)."""
    rng = np.random.default_rng(seed)
    wl = np.linspace(368, 1008, bands)
    dish = 0.15 + 0.02 * np.sin(wl / 90)
    piece = 0.6 + 0.2 * np.sin(wl / 150 + 1.0)
    cube = np.tile(dish, (rows, cols, 1)).astype(np.float64)
    cube[10:25, 10:25, :] = piece
    cube[35:50, 30:45, :] = piece
    cube += rng.normal(0, 0.004, cube.shape)
    return cube, wl


def test_foreground_distance_border_vs_bbox_both_separate_pieces():
    cube, _ = _cube_with_pieces()
    cfg = PieceConfig(method="sam")
    d_border = foreground_distance(cube, cfg)
    cfg_bbox = replace(cfg, background_bbox=(0, 8, 0, 8))   # a clean dish corner
    d_bbox = foreground_distance(cube, cfg_bbox)
    # piece pixel (15,15) is far from background; dish pixel (2,2) is near it
    assert d_border[15, 15] > d_border[2, 2]
    assert d_bbox[15, 15] > d_bbox[2, 2]


def test_background_pixels_uses_bbox_when_set():
    cube, _ = _cube_with_pieces()
    cfg = PieceConfig(background_bbox=(0, 5, 0, 5))
    bg = _background_pixels(cube, cfg)
    assert bg.shape == (25, cube.shape[-1])


def test_flat_field_flattens_spatial_gradient():
    rows = cols = 60
    bands = 16
    wl = np.linspace(368, 1008, bands)
    base = 0.4 + 0.1 * np.sin(wl / 120)
    grad = (np.linspace(0.5, 1.5, rows)[:, None] * np.linspace(0.6, 1.4, cols)[None, :])
    cube = grad[:, :, None] * base[None, None, :]
    out = flat_field_correct(cube, sigma=10.0)
    b = 5
    raw_cv = cube[:, :, b].std() / abs(cube[:, :, b].mean())
    out_cv = out[:, :, b].std() / abs(out[:, :, b].mean())
    assert out_cv < raw_cv * 0.2      # gradient largely removed


def test_extract_pieces_with_bbox_background():
    cube, wl = _cube_with_pieces()
    c = Cube(data=cube, wavelengths=wl, shutter=1.0, ceiling=np.inf,
             path="x", label="t", material="sio2")
    cfg = PieceConfig(method="sam", background_bbox=(0, 8, 0, 8),
                      min_area=20, open_iter=0, close_iter=1)
    pieces = extract_pieces(c, cfg)
    assert len(pieces) >= 2            # the two rectangles


def test_piececonfig_validate_rejects_bad_bbox_and_sigma():
    with pytest.raises(ValueError):
        PieceConfig(background_bbox=(1, 2, 3)).validate()
    with pytest.raises(ValueError):
        PieceConfig(flat_field_sigma=0).validate()
    with pytest.raises(ValueError):
        PieceConfig(erode_iter=-1).validate()


# --- analysis-mask erosion (drops mixed film/dish boundary pixels) ------------

def test_erode_analysis_mask_removes_boundary_ring():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True                      # 10x10 block = 100 px
    out = _erode_analysis_mask(mask, PieceConfig(erode_iter=1))
    assert out.sum() == 64                       # 8x8 interior
    assert not out[5, 5]                         # corner (boundary) dropped
    assert out[9, 9]                             # interior kept


def test_erode_analysis_mask_off_when_zero():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    out = _erode_analysis_mask(mask, PieceConfig(erode_iter=0))
    assert out.sum() == mask.sum()


def test_erode_analysis_mask_keeps_piece_it_would_erase():
    """A fragment thinner than the kernel must not vanish from the study."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[4:6, 4:6] = True                        # 2x2 -> erosion would empty it
    out = _erode_analysis_mask(mask, PieceConfig(erode_iter=2))
    assert out.sum() == mask.sum()               # fell back to un-eroded


# --- euclidean backend: whole wafers incl. dark bare silicon ----------------

def _cube_dark_piece(rows=60, cols=60, bands=20, seed=1):
    """Bright dish + a piece that is a *dimmed copy* of the dish spectrum.

    Same spectral shape, much lower magnitude -- exactly the bare-silicon case
    that SAM is blind to (it normalizes magnitude away).
    """
    rng = np.random.default_rng(seed)
    wl = np.linspace(368, 1008, bands)
    dish = 0.80 + 0.05 * np.sin(wl / 100)
    cube = np.tile(dish, (rows, cols, 1)).astype(np.float64)
    cube[15:40, 15:40, :] = dish * 0.12          # dark, identical shape
    cube += rng.normal(0, 0.002, cube.shape)
    return cube, wl


def test_euclidean_is_magnitude_sensitive_where_sam_is_blind():
    bands = 20
    ref = np.linspace(0.5, 0.9, bands)
    dimmed = (ref * 0.1)[None, :]                # same shape, 10x darker
    reshaped = (ref[::-1])[None, :]              # genuinely different shape

    # SAM is scale-invariant: a dimmed copy is ~0 rad away (only arccos rounding),
    # orders of magnitude less than a real shape change.
    assert float(spectral_angle(dimmed, ref)[0]) < 1e-4
    assert float(spectral_angle(dimmed, ref)[0]) < \
        float(spectral_angle(reshaped, ref)[0]) / 1000
    # Euclidean, by contrast, registers the magnitude drop clearly.
    assert float(euclidean_distance(dimmed, ref)[0]) > 1.0


def test_euclidean_finds_dark_piece_that_sam_misses():
    cube, _ = _cube_dark_piece()
    cfg_sam = PieceConfig(method="sam", threshold="otsu")
    cfg_euc = PieceConfig(method="euclidean", threshold="otsu")
    inside, outside = (25, 25), (2, 2)

    d_sam = foreground_distance(cube, cfg_sam)
    d_euc = foreground_distance(cube, cfg_euc)
    # SAM barely separates the dimmed piece from the dish; euclidean separates strongly.
    sam_sep = d_sam[inside] - d_sam[outside]
    euc_sep = d_euc[inside] - d_euc[outside]
    assert euc_sep > 0
    assert euc_sep > sam_sep


def test_extract_pieces_euclidean_recovers_dark_piece():
    cube, wl = _cube_dark_piece()
    c = Cube(data=cube, wavelengths=wl, shutter=1.0, ceiling=np.inf,
             path="x", label="t", material="sio2")
    cfg = PieceConfig(method="euclidean", min_area=50, open_iter=0,
                      close_iter=1, erode_iter=0)
    pieces = extract_pieces(c, cfg)
    assert len(pieces) == 1
    # the 25x25 dark square should be substantially recovered
    assert pieces[0].mask.sum() > 400


def test_piececonfig_accepts_euclidean_and_rejects_unknown():
    PieceConfig(method="euclidean").validate()
    with pytest.raises(ValueError):
        PieceConfig(method="cosine").validate()


# --- mahalanobis: true distance + chi2 thresholding -------------------------

def test_mahalanobis_returns_distance_not_squared():
    """Regression: the helper is named '..._distance' and must return d, not d^2."""
    from hsi_workflow.pieces import _mahalanobis_to_background
    rng = np.random.default_rng(0)
    bands = 8
    bg = rng.normal(0.0, 1.0, (500, bands))
    probe = np.zeros((1, bands))
    probe[0, 0] = 6.0                       # ~6 sigma along one axis
    d = float(_mahalanobis_to_background(probe, bg)[0])
    # d should be on the order of 6, not 36 (the squared value).
    assert 3.0 < d < 12.0


def test_threshold_chi2_selects_far_more_than_otsu_on_heavy_tail():
    """Otsu collapses on a tight-mode + long-tail map; chi2 does not."""
    from hsi_workflow.pieces import _threshold_mask
    rng = np.random.default_rng(0)
    n_bands = 300
    # background: sqrt(chi2(df)) ~ the bulk; plus a heavy outlier tail
    bulk = np.sqrt(rng.chisquare(n_bands, size=(300, 300)))
    dist = bulk.copy()
    dist[:30, :] = np.sqrt(rng.chisquare(n_bands, size=(30, 300)) * 40)   # foreground
    dist[0, 0] = 1e3                                                      # extreme outlier

    otsu_cfg = PieceConfig(method="mahalanobis", threshold="otsu")
    chi2_cfg = PieceConfig(method="mahalanobis", threshold="chi2", chi2_quantile=0.999)
    m_otsu = _threshold_mask(dist, otsu_cfg, n_bands=n_bands)
    m_chi2 = _threshold_mask(dist, chi2_cfg, n_bands=n_bands)

    assert m_chi2.mean() > m_otsu.mean()
    assert m_chi2[:30, :].mean() > 0.9      # recovers the planted foreground
    assert m_chi2[30:, :].mean() < 0.05     # without flooding the background


def test_chi2_threshold_requires_n_bands_and_validates():
    from hsi_workflow.pieces import _threshold_mask
    dist = np.linspace(0, 10, 100).reshape(10, 10)
    with pytest.raises(ValueError):
        _threshold_mask(dist, PieceConfig(threshold="chi2"), n_bands=None)
    with pytest.raises(ValueError):
        PieceConfig(chi2_quantile=1.0).validate()
    with pytest.raises(ValueError):
        PieceConfig(threshold="nonsense").validate()
    PieceConfig(threshold="chi2").validate()


def test_extract_pieces_erosion_shrinks_mask_but_not_crop():
    cube, wl = _cube_with_pieces()
    c = Cube(data=cube, wavelengths=wl, shutter=1.0, ceiling=np.inf,
             path="x", label="t", material="sio2")
    base = PieceConfig(method="sam", background_bbox=(0, 8, 0, 8),
                       min_area=20, open_iter=0, close_iter=1, erode_iter=0)
    plain = extract_pieces(c, base)
    eroded = extract_pieces(c, replace(base, erode_iter=1))
    assert len(plain) == len(eroded)
    for a, b in zip(plain, eroded):
        assert a.bbox == b.bbox                  # crop geometry untouched
        assert b.mask.sum() < a.mask.sum()       # but fewer analysis pixels
