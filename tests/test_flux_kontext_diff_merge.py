from pathlib import Path
import sys
import types
from unittest.mock import patch

import numpy as np
import cv2
import pytest


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


def textured_image():
    return np.random.default_rng(10).integers(
        0, 256, (192, 192, 3), dtype=np.uint8)


def translated(image, dx, dy):
    return cv2.warpAffine(
        image, np.float32([[1, 0, dx], [0, 1, dy]]),
        (image.shape[1], image.shape[0]))


def merge_for_alignment(original, edited, **kwargs):
    return FluxKontextDiffMerge().merge_diff(
        as_tensor(original[None].astype(np.float32) / 255),
        as_tensor(edited.astype(np.float32) / 255),
        threshold=0.02, detection_method="color_diff", blend_method="alpha",
        mask_blur=0, mask_expand=0, edge_feather=0, min_change_area=0,
        global_threshold=0.15, **kwargs)


def test_translation_alignment_preserves_local_edits_and_batched_coordinates():
    original = textured_image()
    edit = original.copy()
    edit[82:110, 82:110] = (255, 0, 0)
    shifted = np.stack([translated(edit, 3, -2), translated(edit, -4, 5)])
    merged, masks, _ = merge_for_alignment(
        original, shifted, alignment="translation")

    # Registration must give the same result as the existing unshifted pipeline,
    # including its morphological opening of rectangle corners.
    expected, expected_mask, _ = merge_for_alignment(original, edit[None])
    assert np.allclose(merged.numpy(), expected.numpy(), atol=1 / 255)
    assert np.array_equal(masks.numpy(), np.repeat(expected_mask.numpy(), 2, axis=0))
    for shifted_edit in shifted:
        aligned, valid = FluxKontextDiffMerge.align_translation(original, shifted_edit)
        assert np.array_equal(aligned[valid], edit[valid])
    assert np.all(masks.numpy()[:, 86:106, 86:106] == 1)
    outside_edit = np.ones(original.shape[:2], dtype=bool)
    outside_edit[80:112, 80:112] = False
    assert not np.any(masks.numpy()[:, outside_edit])


def test_pure_translation_is_not_an_edit_but_default_keeps_pixel_space_policy():
    original = textured_image()
    shifted = translated(original, 3, -2)[None]
    merged, masks, _ = merge_for_alignment(
        original, shifted, alignment="translation")
    assert not np.any(masks.numpy())
    assert np.allclose(merged.numpy()[0], original / 255, atol=1 / 255)
    _, default_masks, _ = merge_for_alignment(original, shifted)
    assert np.count_nonzero(default_masks.numpy()) > original.shape[0] ** 2 // 2


@pytest.mark.parametrize("case", ["flat", "unrelated", "local", "too_far", "small"])
def test_ambiguous_or_unsupported_alignment_leaves_edit_untouched(case):
    original = textured_image()
    if case == "flat":
        original[:] = 100
        edited = np.full_like(original, 150)
    elif case == "unrelated":
        edited = np.random.default_rng(20).integers(
            0, 256, original.shape, dtype=np.uint8)
    elif case == "local":
        edited = original.copy()
        edited[64:128, 64:128] = translated(original, 3, -2)[64:128, 64:128]
    elif case == "too_far":
        edited = translated(original, 7, 0)
    else:
        original = original[:64, :64]
        edited = translated(original, 3, -2)

    aligned, valid = FluxKontextDiffMerge.align_translation(
        original, edited, max_shift=4)
    assert np.array_equal(aligned, edited)
    assert np.all(valid)


def test_manual_mask_stays_in_original_coordinates_and_excludes_missing_borders():
    original = textured_image()
    edit = original.copy()
    edit[82:110, 82:110] = (255, 0, 0)
    manual = np.zeros(original.shape[:2], dtype=np.float32)
    manual[82:110, 82:110] = 1
    manual[:, -2:] = 1  # No translated source pixels exist at this border.
    merged, masks, _ = merge_for_alignment(
        original, translated(edit, 3, 0)[None], alignment="translation",
        manual_mask=as_tensor(manual))
    expected, _, _ = merge_for_alignment(
        original, edit[None], manual_mask=as_tensor(manual))
    assert np.allclose(merged.numpy(), expected.numpy(), atol=1 / 255)
    assert not np.any(masks.numpy()[0, :, -3:])


def test_fractional_translation_reduces_background_error():
    original = cv2.GaussianBlur(textured_image(), (5, 5), 1)
    edited = translated(original, 2.5, -1.5)
    aligned, valid = FluxKontextDiffMerge.align_translation(original, edited)
    interior = np.zeros(original.shape[:2], dtype=bool)
    interior[16:-16, 16:-16] = True
    assert not np.all(valid)
    before = np.mean(np.abs(original[interior].astype(float) - edited[interior]))
    after = np.mean(np.abs(original[interior].astype(float) - aligned[interior]))
    assert after < before * 0.6


@pytest.mark.parametrize("blend", ["alpha", "poisson", "multiband", "gaussian"])
def test_all_blenders_preserve_uncovered_borders_after_feathering(blend):
    original = textured_image()
    edit = original.copy()
    edit[82:110, 82:110] = (255, 0, 0)
    merged, masks, _ = FluxKontextDiffMerge().merge_diff(
        as_tensor(original[None].astype(np.float32) / 255),
        as_tensor(translated(edit, 3, -2)[None].astype(np.float32) / 255),
        threshold=0.02, detection_method="color_diff", blend_method=blend,
        mask_blur=5, mask_expand=5, edge_feather=5, min_change_area=0,
        global_threshold=0.15, alignment="translation",
        manual_mask=as_tensor(np.ones(original.shape[:2], dtype=np.float32)))
    border = np.zeros(original.shape[:2], dtype=bool)
    border[:2] = True
    border[:, -3:] = True
    assert not np.any(masks.numpy()[0, border])
    assert np.allclose(merged.numpy()[0, border], original[border] / 255,
                       atol=1 / 255)
