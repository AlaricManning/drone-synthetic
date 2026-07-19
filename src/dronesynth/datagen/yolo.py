"""Export canonical annotations to the ultralytics YOLO layout.

The export is a *view* of the canonical annotations: images/{train,val} and
labels/{train,val} with one ``<run_id>_<frame>.txt`` per image, plus a
``dataset.yaml``. Frames with no drone get an empty label file — they teach
the model what background looks like and must not be dropped. The export is
deterministic: same annotations in, same layout out.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from dronesynth.datagen.annotations import FrameAnnotation


@dataclass(frozen=True)
class ExportItem:
    """One frame to export: its annotation plus where its image lives."""

    run_id: str
    annotation: FrameAnnotation
    image_path: Path


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


def export_yolo(
    items: list[ExportItem],
    dest: Path,
    class_map: dict[int, str],
    assignments: dict[str, str],
) -> Path:
    """Write the YOLO dataset layout under dest; returns the dataset.yaml path."""
    missing = sorted({i.run_id for i in items} - set(assignments))
    if missing:
        raise ValueError(f"no train/val assignment for run(s): {missing}")

    for subset in ("train", "val"):
        (dest / "images" / subset).mkdir(parents=True, exist_ok=True)
        (dest / "labels" / subset).mkdir(parents=True, exist_ok=True)

    for item in sorted(items, key=lambda i: (i.run_id, i.annotation.frame_index)):
        subset = assignments[item.run_id]
        stem = f"{item.run_id}_{item.annotation.frame_index:06d}"
        shutil.copy2(item.image_path, dest / "images" / subset / f"{stem}{item.image_path.suffix}")
        lines = yolo_label_lines(item.annotation)
        label_path = dest / "labels" / subset / f"{stem}.txt"
        label_path.write_text("\n".join(lines) + "\n" if lines else "")

    dataset_yaml = dest / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "images/train",
                "val": "images/val",
                "names": {int(key): name for key, name in sorted(class_map.items())},
            },
            sort_keys=False,
        )
    )
    return dataset_yaml
