import cv2
import numpy as np
import pytest

from dronesynth.datagen.annotations import (
    annotate_frame,
    read_annotations,
    write_annotations,
)
from dronesynth.datagen.pairing import FramePair


def write_png(path, h, w, blob=None):
    image = np.zeros((h, w, 3), dtype=np.uint8)
    if blob is not None:
        y0, y1, x0, x1 = blob
        image[y0:y1, x0:x1] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def make_pair(tmp_path, index=0, mask_blob=None, mask_size=(48, 64)):
    normal = tmp_path / "normal" / f"seq.{index:04d}.png"
    mask = tmp_path / "mask" / f"seq.{index:04d}.png"
    write_png(normal, 48, 64)
    write_png(mask, *mask_size, blob=mask_blob)
    return FramePair(index=index, normal=normal, mask=mask)


def test_annotates_frame_with_drone(tmp_path):
    pair = make_pair(tmp_path, index=7, mask_blob=(10, 30, 5, 15))
    annotation = annotate_frame(pair, threshold=12, min_box_area=16, class_id=0)
    assert annotation.frame_index == 7
    assert annotation.normal == "seq.0007.png"
    assert (annotation.width, annotation.height) == (64, 48)
    assert len(annotation.boxes) == 1
    box = annotation.boxes[0]
    assert (box.x, box.y, box.w, box.h) == (5, 10, 10, 20)
    assert box.class_id == 0
    assert box.fill_ratio == 1.0


def test_annotates_empty_frame(tmp_path):
    pair = make_pair(tmp_path, mask_blob=None)
    annotation = annotate_frame(pair, threshold=12, min_box_area=16, class_id=0)
    assert annotation.boxes == ()


def test_rejects_mismatched_dimensions(tmp_path):
    pair = make_pair(tmp_path, index=3, mask_size=(48, 32))
    with pytest.raises(ValueError, match="frame 3.*not from the same camera"):
        annotate_frame(pair, threshold=12, min_box_area=16, class_id=0)


def test_annotations_round_trip_through_json(tmp_path):
    pair = make_pair(tmp_path, index=1, mask_blob=(10, 30, 5, 15))
    original = [annotate_frame(pair, threshold=12, min_box_area=16, class_id=0)]
    path = tmp_path / "out" / "annotations.json"
    write_annotations(original, path)
    assert read_annotations(path) == original


def test_contrast_comes_from_the_normal_render(tmp_path):
    """The one field the mask cannot supply, so the one that says it is visible."""
    blob = (10, 30, 5, 15)
    normal = tmp_path / "normal" / "seq.0000.png"
    mask = tmp_path / "mask" / "seq.0000.png"
    write_png(mask, 48, 64, blob=blob)

    scene = np.full((48, 64, 3), 160, dtype=np.uint8)
    scene[10:30, 5:15] = 60  # a dark drone on a bright sky
    normal.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(normal), scene)

    annotation = annotate_frame(
        FramePair(index=0, normal=normal, mask=mask),
        threshold=12,
        min_box_area=16,
        class_id=0,
    )

    assert annotation.boxes[0].contrast == -100.0


def test_a_drone_the_render_lost_reads_as_no_contrast(tmp_path):
    """A crisp mask over a drone the sky has swallowed -- the wrong label.

    Nothing derived from the mask can tell this from a good frame: the box, the
    area and the fill ratio are identical either way.
    """
    blob = (10, 30, 5, 15)
    normal = tmp_path / "normal" / "seq.0000.png"
    mask = tmp_path / "mask" / "seq.0000.png"
    write_png(mask, 48, 64, blob=blob)

    scene = np.full((48, 64, 3), 160, dtype=np.uint8)  # drone indistinguishable
    normal.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(normal), scene)

    box = annotate_frame(
        FramePair(index=0, normal=normal, mask=mask),
        threshold=12,
        min_box_area=16,
        class_id=0,
    ).boxes[0]

    assert box.contrast == 0.0
    assert box.fill_ratio == 1.0  # the mask is perfectly happy


def test_a_box_written_before_contrast_existed_loads_as_unmeasured(tmp_path):
    """None rather than 0.0: not knowing is not the same as no contrast."""
    path = tmp_path / "old.json"
    path.write_text(
        '[{"frame_index": 0, "normal": "a.png", "width": 64, "height": 48, '
        '"boxes": [{"class_id": 0, "x": 1, "y": 2, "w": 3, "h": 4, '
        '"mask_area": 12, "fill_ratio": 1.0}]}]'
    )

    assert read_annotations(path)[0].boxes[0].contrast is None
