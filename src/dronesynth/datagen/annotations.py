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
    extract_instances,
    image_size,
    load_image,
    load_mask,
)
from dronesynth.datagen.contrast import mask_alpha, measure_contrast, to_grey
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
    # Islands this object rendered as; >1 means thin structure dropped out or
    # something occluded the airframe. Defaults so annotations written before
    # instance grouping still load.
    components: int = 1
    # Signed difference between the object's own pixels and the background
    # immediately around it, in the normal render. The only field here derived
    # from the normal rather than the mask, and so the only one that can say the
    # label describes something visible. None when it could not be sampled, and
    # on annotations written before it existed -- which is not zero contrast.
    contrast: float | None = None


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


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


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

    mask = load_mask(pair.mask)
    binary = binarize_mask(mask, threshold)
    alpha = mask_alpha(mask)
    grey = to_grey(load_image(pair.normal))
    boxes = tuple(
        AnnotatedBox(
            class_id=class_id,
            x=b.x,
            y=b.y,
            w=b.w,
            h=b.h,
            mask_area=b.mask_area,
            fill_ratio=round(b.fill_ratio, 4),
            components=b.components,
            contrast=_rounded(measure_contrast(grey, alpha, binary, b)),
        )
        for b in extract_instances(binary, min_box_area)
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


def parse_annotations(text: str) -> list[FrameAnnotation]:
    """Annotations straight from JSON text, for readers holding no local file."""
    return [FrameAnnotation.from_dict(d) for d in json.loads(text)]


def read_annotations(path: Path) -> list[FrameAnnotation]:
    return parse_annotations(path.read_text())
