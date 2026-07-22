import numpy as np
from scipy import ndimage as ndi

from hsi_workflow.pieces import component_sizes, label_pieces
from hsi_workflow.config import PieceConfig


def test_component_sizes_counts_each_label():
    labels = np.array([[0, 1, 1],
                       [2, 2, 1],
                       [2, 0, 0]])
    sizes = component_sizes(labels)
    assert sizes[0] == 3   # background
    assert sizes[1] == 3
    assert sizes[2] == 3


def test_label_pieces_keeps_only_large_components():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:4, 1:4] = True     # 9 px
    mask[8, 8] = True         # 1 px speck
    cfg = PieceConfig(min_area=5, open_iter=0, close_iter=0,
                      fill_holes=False, watershed_split=False)
    labels, kept = label_pieces(mask, cfg)
    # the 9-px block is kept, the speck is dropped
    assert len(kept) == 1
    big = kept[0]
    assert int((labels == big).sum()) == 9
