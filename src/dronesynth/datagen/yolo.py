"""Render canonical annotations in the ultralytics YOLO format.

A YOLO dataset is a *view* of the canonical annotations, never a second source
of truth: ``class cx cy w h`` per box, normalized, plus a ``dataset.yaml``
naming the classes. Frames with no drone get an empty label file rather than no
file — they teach the model what background looks like and must not be dropped.

Both functions are pure text, which is what lets `dronesynth build` assemble a
dataset straight through the storage layer, on S3 or locally, without staging a
tree on the machine running it.
"""

from __future__ import annotations

import yaml

from dronesynth.datagen.annotations import FrameAnnotation


def yolo_label_lines(annotation: FrameAnnotation) -> list[str]:
    """YOLO box format: ``class cx cy w h``, center-based, normalized to [0, 1]."""
    lines = []
    for box in annotation.boxes:
        cx = (box.x + box.w / 2) / annotation.width
        cy = (box.y + box.h / 2) / annotation.height
        w = box.w / annotation.width
        h = box.h / annotation.height
        lines.append(f"{box.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def dataset_yaml_text(class_map: dict[int, str]) -> str:
    """The ultralytics dataset descriptor, as text."""
    return yaml.safe_dump(
        {
            "path": ".",
            "train": "images/train",
            "val": "images/val",
            "names": {int(key): name for key, name in sorted(class_map.items())},
        },
        sort_keys=False,
    )
