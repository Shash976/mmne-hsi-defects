import numpy as np
import pytest

from hsi_workflow.config import WorkflowConfig
from hsi_workflow.pieces import Piece
from hsi_workflow.baseline import subsample_spectra, baseline_from_pieces


def _make_silicon_piece(piece_id, seed, rows=20, cols=20, bands=15, amp=0.05, freq=150.0):
    """A homogeneous bare-Si-like piece. ``amp``/``freq`` control spectral *shape*
    (not just brightness) so outlier tests actually move the spectral angle."""
    rng = np.random.default_rng(seed)
    wl = np.linspace(400, 1000, bands)
    base = 0.4 + amp * np.sin(wl / freq)
    data = np.tile(base, (rows, cols, 1)).astype(np.float64)
    data += rng.normal(0, 0.002, data.shape)
    mask = np.ones((rows, cols), dtype=bool)
    reflectance_mean = data.mean(axis=-1).astype(np.float32)
    noise = {"after": {"rms_noise": 0.01, "snr": 12.5, "n_pixels": rows * cols}}
    return Piece(data=data, mask=mask, material="silicon", piece_id=piece_id,
                source_label="bare silicon all", bbox=(0, rows, 0, cols),
                wavelengths=wl, reflectance_mean=reflectance_mean, noise=noise)


def test_subsample_spectra_caps_and_is_deterministic():
    arr = np.arange(1000 * 5, dtype=np.float64).reshape(1000, 5)
    out1 = subsample_spectra(arr, 100, seed=0)
    out2 = subsample_spectra(arr, 100, seed=0)
    assert out1.shape == (100, 5)
    np.testing.assert_array_equal(out1, out2)


def test_subsample_spectra_returns_all_when_under_cap():
    arr = np.arange(10 * 5, dtype=np.float64).reshape(10, 5)
    out = subsample_spectra(arr, 100, seed=0)
    np.testing.assert_array_equal(out, arr)


def test_baseline_from_pieces_shapes_and_mean():
    pieces = [_make_silicon_piece("p01", seed=0), _make_silicon_piece("p02", seed=1)]
    wf = WorkflowConfig()
    sb = baseline_from_pieces("sio2_bare_si", pieces, wf, pool_cap=500, seed=0)
    n_bands = pieces[0].n_bands
    assert sb.mean_spectrum.shape == (n_bands,)
    assert sb.std_spectrum.shape == (n_bands,)
    assert sb.cov.shape == (n_bands, n_bands)
    assert sb.pooled_spectra.shape[1] == n_bands
    assert sb.pooled_spectra.shape[0] <= 500
    assert len(sb.piece_stats) == 2
    all_fg = np.vstack([p.foreground_spectra() for p in pieces])
    np.testing.assert_allclose(sb.mean_spectrum, all_fg.mean(axis=0))


def test_baseline_from_pieces_flags_outlier_piece():
    normal = [_make_silicon_piece(f"p{i:02d}", seed=i) for i in range(5)]
    outlier = _make_silicon_piece("p_odd", seed=99, amp=0.3, freq=20.0)  # different shape
    wf = WorkflowConfig()
    sb = baseline_from_pieces("sio2_bare_si", normal + [outlier], wf)
    stats = {ps.piece_id: ps for ps in sb.piece_stats}
    assert stats["p_odd"].flag_outlier is True
    assert all(not stats[p.piece_id].flag_outlier for p in normal)


def test_baseline_from_pieces_empty_raises():
    with pytest.raises(ValueError):
        baseline_from_pieces("sio2_bare_si", [], WorkflowConfig())


def test_piece_stats_fields():
    piece = _make_silicon_piece("p01", seed=0)
    sb = baseline_from_pieces("sio2_bare_si", [piece], WorkflowConfig())
    ps = sb.piece_stats[0]
    assert ps.n_px == int(piece.mask.sum())
    assert ps.snr == pytest.approx(12.5)
    assert ps.sam_from_global == pytest.approx(0.0, abs=2e-6)  # only piece == global mean
