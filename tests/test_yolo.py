import yaml

from dronesynth.datagen.annotations import AnnotatedBox, FrameAnnotation
from dronesynth.datagen.yolo import dataset_yaml_text, yolo_label_lines


def annotation(index, boxes=(), width=64, height=48):
    return FrameAnnotation(
        frame_index=index, normal=f"seq.{index:04d}.png",
        width=width, height=height, boxes=tuple(boxes),
    )


def box(x, y, w, h, class_id=0):
    return AnnotatedBox(
        class_id=class_id, x=x, y=y, w=w, h=h, mask_area=w * h, fill_ratio=1.0
    )


def test_label_line_math():
    lines = yolo_label_lines(annotation(0, [box(5, 10, 10, 20)]))
    # cx = (5 + 5)/64, cy = (10 + 10)/48, w = 10/64, h = 20/48
    assert lines == ["0 0.156250 0.416667 0.156250 0.416667"]


def test_empty_frame_has_no_lines():
    """A drone-less frame still gets a label file, it is just empty."""
    assert yolo_label_lines(annotation(0)) == []


def test_dataset_descriptor():
    config = yaml.safe_load(dataset_yaml_text({0: "drone"}))
    assert config["path"] == "."
    assert config["train"] == "images/train"
    assert config["val"] == "images/val"
    assert config["names"] == {0: "drone"}


def test_dataset_descriptor_orders_classes_by_id():
    config = yaml.safe_load(dataset_yaml_text({1: "bird", 0: "drone"}))
    assert list(config["names"]) == [0, 1]
