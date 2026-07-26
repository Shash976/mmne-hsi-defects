import numpy as np
import pytest

from hsi_workflow.config import FilmConfig, WorkflowConfig
from hsi_workflow.pieces import Piece
from hsi_workflow.film import (extract_film, film_distance,
                               bare_si_reference_from_pieces)


def _make_piece(rows=40, cols=40, bands=30, seed=0):
    """A wafer piece: mostly bare silicon with a rectangular SiO2 (fringed) patch."""
    rng = np.random.default_rng(seed)
    wl = np.linspace(368, 1008, bands)
    bare = 0.5 + 0.10 * np.sin(wl / 200)          # smooth bare-Si-like
    oxide = bare + 0.15 * np.sin(wl / 55)         # thin-film interference (different shape)
    data = np.tile(bare, (rows, cols, 1)).astype(np.float64)
    mask = np.zeros((rows, cols), bool)
    mask[5:35, 5:35] = True                       # the wafer footprint
    oxide_region = np.zeros((rows, cols), bool)
    oxide_region[10:20, 10:25] = True             # oxide patch inside the wafer
    data[oxide_region] = oxide
    data += rng.normal(0, 0.004, data.shape)
    piece = Piece(data=data, mask=mask, material="sio2", piece_id="t_p01",
                  source_label="t", bbox=(0, rows, 0, cols), wavelengths=wl)
    return piece, oxide_region, bare


def _iou(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


def test_extract_film_control_recovers_oxide_patch():
    piece, oxide, bare = _make_piece()
    cfg = FilmConfig(reference="control", method="sam", min_area=10,
                     open_iter=0, close_iter=1)
    fm = extract_film(piece, cfg, ref_spectrum=bare)
    assert _iou(fm.sio2_mask, oxide) > 0.5
    assert not (fm.sio2_mask & ~piece.mask).any()          # stays inside the wafer
    assert not (fm.sio2_mask & fm.substrate_mask).any()    # disjoint
    assert (fm.sio2_mask | fm.substrate_mask == piece.mask).all()   # partition the mask


def test_extract_film_in_piece_recovers_oxide_patch():
    piece, oxide, bare = _make_piece(seed=2)
    cfg = FilmConfig(reference="in_piece", method="sam", min_area=10,
                     open_iter=0, close_iter=1)
    fm = extract_film(piece, cfg, ref_spectrum=bare)
    assert _iou(fm.sio2_mask, oxide) > 0.4


def test_extract_film_in_piece_without_reference_flags_minority():
    piece, oxide, bare = _make_piece(seed=3)
    cfg = FilmConfig(reference="in_piece", method="sam", min_area=10,
                     open_iter=0, close_iter=1)
    fm = extract_film(piece, cfg, ref_spectrum=None)   # larger cluster assumed bare
    assert fm.sio2_mask.sum() > 0
    assert _iou(fm.sio2_mask, oxide) > 0.3


def test_control_reference_required():
    piece, oxide, bare = _make_piece()
    with pytest.raises(ValueError):
        film_distance(piece.data, piece.mask, None, FilmConfig(reference="control"))


def test_bare_si_reference_from_pieces_shape():
    piece, oxide, bare = _make_piece()
    ref = bare_si_reference_from_pieces([piece], WorkflowConfig())
    assert ref.shape == (piece.data.shape[-1],)


def test_film_disabled_by_default():
    assert FilmConfig().enabled is False


def test_filmconfig_validate():
    with pytest.raises(ValueError):
        FilmConfig(reference="bogus").validate()
    with pytest.raises(ValueError):
        FilmConfig(method="bogus").validate()
    with pytest.raises(ValueError):
        FilmConfig(threshold="bogus").validate()
    FilmConfig().validate()   # defaults are valid
