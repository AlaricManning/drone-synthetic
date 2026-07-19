"""Canonical per-frame annotations.

One JSON record per frame is the dataset's source of truth; format-specific
layouts (YOLO today, COCO or segmentation later) are exports generated from
these records, never the other way around.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from dronesynth.datagen.boxes import (
    binarize_mask,
    extract_boxes,
    image_size,
    load_mask,
)
from dronesynth.datagen.pairing import FramePair


@dataclass(frozen=True)
class AnnotatedBox:
    class_id: int
    x: int
    y: int
    w: int
    h: int
    mask_area: int
    fill_ratio: float


@dataclass(frozen=True)
class FrameAnnotation:
    frame_index: int
    normal: str  # filename of the normal render this annotation labels
    width: int
    height: int
    boxes: tuple[AnnotatedBox, ...]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> FrameAnnotation:
        boxes = tuple(AnnotatedBox(**b) for b in data.pop("boxes"))
        return cls(boxes=boxes, **data)


def annotate_frame(
    pair: FramePair, *, threshold: int, min_box_area: int, class_id: int
) -> FrameAnnotation:
    """Threshold one mask render into the canonical annotation for its frame."""
    width, height = image_size(pair.normal)
    mask_width, mask_height = image_size(pair.mask)
    if (mask_width, mask_height) != (width, height):
        raise ValueError(
            f"frame {pair.index}: normal is {width}x{height} but mask is "
            f"{mask_width}x{mask_height} — renders are not from the same camera setup"
        )

    binary = binarize_mask(load_mask(pair.mask), threshold)
    boxes = tuple(
        AnnotatedBox(
            class_id=class_id,
            x=b.x,
            y=b.y,
            w=b.w,
            h=b.h,
            mask_area=b.mask_area,
            fill_ratio=round(b.fill_ratio, 4),
        )
        for b in extract_boxes(binary, min_box_area)
    )
    return FrameAnnotation(
        frame_index=pair.index,
        normal=pair.normal.name,
        width=width,
        height=height,
        boxes=boxes,
    )


def write_annotations(annotations: list[FrameAnnotation], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([a.to_dict() for a in annotations], indent=2))


def read_annotations(path: Path) -> list[FrameAnnotation]:
    return [FrameAnnotation.from_dict(d) for d in json.loads(path.read_text())]
