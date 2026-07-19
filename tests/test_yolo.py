import cv2
import numpy as np
import pytest
import yaml

from dronesynth.datagen.annotations import AnnotatedBox, FrameAnnotation
from dronesynth.datagen.yolo import ExportItem, export_yolo, yolo_label_lines


def annotation(index, boxes=(), width=64, height=48):
    return FrameAnnotation(
        frame_index=index, normal=f"seq.{index:04d}.png",
        width=width, height=height, boxes=tuple(boxes),
    )


def box(x, y, w, h, class_id=0):
    return AnnotatedBox(
        class_id=class_id, x=x, y=y, w=w, h=h, mask_area=w * h, fill_ratio=1.0
    )


def item(tmp_path, run_id, index, boxes=()):
    image = tmp_path / "src" / run_id / f"seq.{index:04d}.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image), np.zeros((48, 64, 3), dtype=np.uint8))
    return ExportItem(run_id=run_id, annotation=annotation(index, boxes), image_path=image)


def test_label_line_math():
    lines = yolo_label_lines(annotation(0, [box(5, 10, 10, 20)]))
    # cx = (5 + 5)/64, cy = (10 + 10)/48, w = 10/64, h = 20/48
    assert lines == ["0 0.156250 0.416667 0.156250 0.416667"]


def test_empty_frame_has_no_lines():
    assert yolo_label_lines(annotation(0)) == []


def test_export_layout_and_contents(tmp_path):
    items = [
        item(tmp_path, "run_0001", 0, [box(5, 10, 10, 20)]),
        item(tmp_path, "run_0001", 1),  # drone-less frame
        item(tmp_path, "run_0002", 0, [box(0, 0, 8, 8)]),
    ]
    dest = tmp_path / "yolo"
    assignments = {"run_0001": "train", "run_0002": "val"}
    dataset_yaml = export_yolo(items, dest, {0: "drone"}, assignments)

    train_images = sorted(p.name for p in (dest / "images" / "train").iterdir())
    assert train_images == ["run_0001_000000.png", "run_0001_000001.png"]
    assert [p.name for p in (dest / "images" / "val").iterdir()] == ["run_0002_000000.png"]

    label = (dest / "labels" / "train" / "run_0001_000000.txt").read_text()
    assert label == "0 0.156250 0.416667 0.156250 0.416667\n"
    # empty label file must exist for the drone-less frame, not be omitted
    assert (dest / "labels" / "train" / "run_0001_000001.txt").read_text() == ""

    config = yaml.safe_load(dataset_yaml.read_text())
    assert config["train"] == "images/train"
    assert config["val"] == "images/val"
    assert config["names"] == {0: "drone"}


def test_export_rejects_unassigned_run(tmp_path):
    items = [item(tmp_path, "run_0007", 0)]
    with pytest.raises(ValueError, match="run_0007"):
        export_yolo(items, tmp_path / "yolo", {0: "drone"}, {"run_0001": "train"})
