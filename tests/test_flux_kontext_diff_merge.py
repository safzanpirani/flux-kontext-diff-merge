from pathlib import Path
import sys
import types
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FakeTensor:
    def __init__(self, array):
        self._array = np.asarray(array, dtype=np.float32)

    @property
    def shape(self):
        return self._array.shape

    def cpu(self):
        return self

    def detach(self):
        return self

    def numpy(self):
        return self._array

    def __getitem__(self, item):
        return FakeTensor(self._array[item])


def fake_from_numpy(array):
    return FakeTensor(np.array(array, copy=True))


def fake_cat(tensors, dim=0):
    arrays = [tensor.numpy() for tensor in tensors]
    return FakeTensor(np.concatenate(arrays, axis=dim))


sys.modules.setdefault(
    "torch",
    types.SimpleNamespace(
        from_numpy=fake_from_numpy,
        cat=fake_cat,
    ),
)

from flux_kontext_diff_merge import FluxKontextDiffMerge


def as_tensor(array):
    return FakeTensor(array)


def test_detect_ssim_changes_returns_empty_mask_for_identical_images():
    node = FluxKontextDiffMerge()
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    mask = node.detect_ssim_changes(image, image.copy(), threshold=0.02)

    assert np.count_nonzero(mask) == 0


def test_poisson_blend_uses_soft_mask_for_output_edges():
    node = FluxKontextDiffMerge()
    source = np.full((7, 7, 3), 255, dtype=np.uint8)
    target = np.zeros((7, 7, 3), dtype=np.uint8)
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[1:6, 1:6] = 64
    mask[2:5, 2:5] = 255

    with patch("flux_kontext_diff_merge.cv2.seamlessClone", return_value=source):
        result = node.poisson_blend(source, target, mask)

    assert result[3, 3, 0] == 255
    assert 0 < result[1, 1, 0] < 255


def test_merge_diff_broadcasts_original_and_manual_mask():
    node = FluxKontextDiffMerge()
    original = as_tensor(np.zeros((1, 4, 4, 3), dtype=np.float32))
    edited = as_tensor(
        np.stack(
            [
                np.ones((4, 4, 3), dtype=np.float32),
                np.full((4, 4, 3), 0.5, dtype=np.float32),
            ]
        )
    )
    manual_mask = as_tensor(np.ones((4, 4), dtype=np.float32))

    merged, mask, preview = node.merge_diff(
        original_image=original,
        edited_image=edited,
        threshold=0.02,
        detection_method="adaptive",
        blend_method="alpha",
        mask_blur=0,
        mask_expand=0,
        edge_feather=0,
        min_change_area=0,
        global_threshold=0.15,
        manual_mask=manual_mask,
    )

    assert tuple(merged.shape) == (2, 4, 4, 3)
    assert tuple(mask.shape) == (2, 4, 4)
    assert tuple(preview.shape) == (2, 4, 4, 3)
    assert np.allclose(merged[0].numpy(), edited[0].numpy(), atol=1 / 255)
    assert np.allclose(merged[1].numpy(), edited[1].numpy(), atol=1 / 255)
