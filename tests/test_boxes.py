import numpy as np

from dronesynth.datagen.boxes import binarize_mask, extract_boxes, extract_instances


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


# --- instances ---------------------------------------------------------------
# Every foreground pixel is drone, so islands are pieces of one object rather
# than objects. These cover the grouping that turns pieces back into a label.


def test_clean_silhouette_is_one_undivided_instance():
    mask = blank_mask()
    mask[10:30, 5:15] = 255
    binary = binarize_mask(mask, threshold=12)
    (box,) = extract_instances(binary, min_box_area=16)
    assert (box.x, box.y, box.w, box.h) == (5, 10, 10, 20)
    assert box.components == 1


def test_detached_piece_joins_its_object_instead_of_becoming_a_label():
    # The propeller-blade case: a blade whose supporting arm is thinner than a
    # pixel renders as its own island inside the airframe's extent. It is not a
    # second drone, and labelling it as one taught the detector that a handful
    # of pixels is a whole aircraft.
    mask = blank_mask()
    mask[20:40, 20:50] = 255  # airframe
    mask[22:26, 24:37] = 0    # gap...
    mask[23:25, 26:35] = 255  # ...with a blade stranded inside it
    binary = binarize_mask(mask, threshold=12)
    boxes = extract_instances(binary, min_box_area=16)
    assert len(boxes) == 1
    assert (boxes[0].x, boxes[0].y, boxes[0].w, boxes[0].h) == (20, 20, 30, 20)
    assert boxes[0].components == 2


def test_a_piece_outside_the_body_still_widens_the_box():
    # Why grouping beats dropping contained boxes or filtering by size: when the
    # stranded piece is the outermost one, the airframe's own component stops
    # short of the true extent. Discarding the piece would ship a box that
    # clips the drone, and it is the same defect wearing different clothes.
    mask = blank_mask()
    mask[20:40, 20:40] = 255  # body
    mask[28:32, 46:52] = 255  # blade tip, detached and beyond the body
    binary = binarize_mask(mask, threshold=12)
    (box,) = extract_instances(binary, min_box_area=16)
    assert (box.x, box.w) == (20, 32)  # spans body through blade tip
    assert box.components == 2


def test_occluded_drone_stays_a_single_box():
    # The case varied backgrounds will introduce: a branch across the middle
    # splits the silhouette by far more than any closing kernel would bridge.
    # One drone behind an occluder is still one drone.
    mask = blank_mask()
    mask[10:20, 10:40] = 255
    mask[30:40, 10:40] = 255
    binary = binarize_mask(mask, threshold=12)
    (box,) = extract_instances(binary, min_box_area=16)
    assert (box.y, box.h) == (10, 30)
    assert box.components == 2


def test_fill_ratio_counts_pixels_not_the_span():
    # mask_area sums the pieces rather than filling the union, so a gappy
    # silhouette reads as gappy and QC's low-fill flag still fires.
    mask = blank_mask()
    mask[10:20, 10:40] = 255
    mask[30:40, 10:40] = 255
    binary = binarize_mask(mask, threshold=12)
    (box,) = extract_instances(binary, min_box_area=16)
    assert box.mask_area == 600
    assert box.fill_ratio == 600 / (30 * 30)


def test_min_box_area_judges_the_object_not_its_pieces():
    # Deliberately unlike extract_boxes, which drops small components as noise.
    # Under grouping the small component is part of the drone, so the size test
    # belongs on the assembled object.
    mask = blank_mask()
    mask[5:8, 5:8] = 255  # 3x3 -- would be dropped as a speck on its own
    mask[40:60, 40:60] = 255
    binary = binarize_mask(mask, threshold=12)
    (box,) = extract_instances(binary, min_box_area=16)
    assert box.components == 2
    assert box.mask_area == 9 + 400


def test_object_below_min_box_area_is_dropped():
    mask = blank_mask()
    mask[5:8, 5:8] = 255  # 3x3 box = 9 px, the whole frame's foreground
    binary = binarize_mask(mask, threshold=12)
    assert extract_instances(binary, min_box_area=16) == []


def test_blank_mask_has_no_instances():
    binary = binarize_mask(blank_mask(), threshold=12)
    assert extract_instances(binary, min_box_area=16) == []
