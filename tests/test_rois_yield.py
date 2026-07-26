"""ROI tiling yield -- the defaults must hit the document's 100-300 per piece.

The document (`Revised Research Objective.md`, "How many ROIs per image?") asks
for 100-300 ROIs per specimen. The extracted pieces here are small (median ~4,175
foreground px), so 32x32 non-overlapping tiles yielded only ~1-4 per piece. These
tests pin the corrected defaults and the leakage guarantee that makes the
overlapping stride acceptable.
"""
import numpy as np

from hsi_workflow.config import RoiConfig
from hsi_workflow.pieces import Piece
from hsi_workflow.rois import tile_rois, build_roi_table, split_by_specimen


def _piece(size=65, bands=12, piece_id="s_p01", material="sio2"):
    """A square piece of ~size^2 foreground px (median real piece ~4,175)."""
    rng = np.random.default_rng(0)
    data = rng.normal(0.5, 0.01, (size, size, bands))
    mask = np.ones((size, size), dtype=bool)
    return Piece(data=data, mask=mask, material=material, piece_id=piece_id,
                 source_label="s", bbox=(0, size, 0, size),
                 wavelengths=np.linspace(368, 1008, bands))


def test_default_roi_config_is_overlapping_small_patch():
    cfg = RoiConfig()
    assert cfg.patch == 8
    assert cfg.stride == 4
    assert cfg.stride < cfg.patch          # overlapping by design


def test_default_yield_reaches_document_target_on_typical_piece():
    """~4,175 px piece should give ROIs in the document's 100-300 band."""
    rois = tile_rois(_piece(), RoiConfig())
    assert 100 <= len(rois) <= 300, f"got {len(rois)}"


def test_old_defaults_would_have_missed_the_target():
    """Regression guard documenting why the defaults changed."""
    rois = tile_rois(_piece(), RoiConfig(patch=32, stride=32))
    assert len(rois) < 10                  # the gap this fixed


def test_rois_stay_inside_mask():
    size = 65
    piece = _piece(size=size)
    piece.mask[:, : size // 2] = False     # blank out the left half
    rois = tile_rois(piece, RoiConfig())
    assert rois
    for r in rois:
        assert r.coverage >= RoiConfig().min_coverage
        assert r.bbox[2] >= size // 2 - RoiConfig().patch


def test_overlapping_rois_never_straddle_the_specimen_split():
    """Overlap is only safe because whole specimens are held out."""
    rois = []
    for pid in ("s_p01", "s_p02", "s_p03", "s_p04"):
        rois.extend(tile_rois(_piece(piece_id=pid), RoiConfig()))
    df = build_roi_table(rois)
    train, test = split_by_specimen(df, test_fraction=0.5, seed=0)
    assert not set(train["specimen"]) & set(test["specimen"])
    assert len(train) and len(test)
