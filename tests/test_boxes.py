import numpy as np

from dronesynth.datagen.boxes import binarize_mask, extract_boxes


def blank_mask(h=64, w=64):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_blank_mask_has_no_boxes():
    binary = binarize_mask(blank_mask(), threshold=12)
    assert extract_boxes(binary, min_box_area=16) == []


def test_pixels_at_threshold_are_background():
    mask = blank_mask()
    mask[10:20, 10:20] = 12  # exactly the threshold — must not count
    binary = binarize_mask(mask, threshold=12)
    assert not binary.any()


def test_single_blob_box_coordinates():
    mask = blank_mask()
    mask[10:30, 5:15] = 200  # rows 10..29, cols 5..14
    binary = binarize_mask(mask, threshold=12)
    boxes = extract_boxes(binary, min_box_area=16)
    assert len(boxes) == 1
    box = boxes[0]
    assert (box.x, box.y, box.w, box.h) == (5, 10, 10, 20)
    assert box.mask_area == 200  # solid rectangle: every pixel set
    assert box.fill_ratio == 1.0


def test_bright_single_channel_counts_as_drone():
    mask = blank_mask()
    mask[5:10, 5:10, 2] = 255  # only one channel bright
    binary = binarize_mask(mask, threshold=12)
    assert extract_boxes(binary, min_box_area=1)


def test_two_blobs_two_boxes_largest_first():
    mask = blank_mask()
    mask[5:10, 5:10] = 255    # 25 px
    mask[40:60, 40:60] = 255  # 400 px
    binary = binarize_mask(mask, threshold=12)
    boxes = extract_boxes(binary, min_box_area=1)
    assert len(boxes) == 2
    assert boxes[0].mask_area == 400
    assert boxes[1].mask_area == 25


def test_speck_below_min_area_dropped():
    mask = blank_mask()
    mask[5:8, 5:8] = 255      # 3x3 box = 9 px area
    mask[40:60, 40:60] = 255  # real blob
    binary = binarize_mask(mask, threshold=12)
    boxes = extract_boxes(binary, min_box_area=16)
    assert len(boxes) == 1
    assert boxes[0].box_area == 400


def test_opaque_alpha_channel_is_ignored():
    mask = np.zeros((32, 32, 4), dtype=np.uint8)
    mask[:, :, 3] = 255  # fully opaque, all-black image — no drone anywhere
    mask[4:12, 4:12, :3] = 255  # except one real blob
    binary = binarize_mask(mask, threshold=12)
    boxes = extract_boxes(binary, min_box_area=1)
    assert len(boxes) == 1
    assert boxes[0].box_area == 64


def test_grayscale_mask_supported():
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    binary = binarize_mask(mask, threshold=12)
    assert len(extract_boxes(binary, min_box_area=1)) == 1
