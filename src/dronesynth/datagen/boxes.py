"""Mask render -> bounding boxes.

The mask render shows the drone against black: the renderer paints the drone
actor with a flat material and hides everything else, so every pixel above the
threshold belongs to a drone *by construction*.

That makes connected components the wrong unit to label. Connectivity is a fact
about rasterisation, not about how many objects are in frame — a propeller
blade whose supporting arm is thinner than a pixel renders as its own island,
and so does a drone crossing behind a branch. Boxing each island separately
turns one drone into several labels.

So components are extracted as a primitive and then grouped into *instances*,
which is the unit a detector is being taught. Today the grouping is trivial
because a frame holds one drone, so every island is part of it. When per-drone
mask values arrive the grouping key becomes the value; the shape of the code
does not change. See docs/plans/instance-boxes.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DetectedBox:
    """One object's bounding box, in pixel coordinates.

    ``mask_area`` is the object's pixel count; ``fill_ratio`` is
    mask_area / box area — a drone silhouette fills a fair fraction of its
    box, so a very low ratio flags a suspicious detection.

    ``components`` is how many disconnected islands the object rendered as. It
    is always 1 for a cleanly rendered drone and rises when structure thinner
    than a pixel drops out, or when something occludes the middle of the
    airframe. The box is right either way, but the count is worth keeping: it
    is the signal that caught the propeller-blade fragmentation in the first
    place, and without it a render regression is invisible in the labels.
    """

    x: int
    y: int
    w: int
    h: int
    mask_area: int
    components: int = 1

    @property
    def box_area(self) -> int:
        return self.w * self.h

    @property
    def fill_ratio(self) -> float:
        return self.mask_area / self.box_area


def image_size(path: Path) -> tuple[int, int]:
    """(width, height) of an image, read from metadata without decoding pixels."""
    with Image.open(path) as image:
        return image.size


def load_image(path: Path) -> np.ndarray:
    # IMREAD_COLOR drops the alpha channel EasySynth writes (opaque everywhere,
    # so it would put every pixel above any threshold)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def load_mask(path: Path) -> np.ndarray:
    return load_image(path)


def binarize_mask(mask: np.ndarray, threshold: int) -> np.ndarray:
    """Boolean drone/background map: any color channel above threshold is drone."""
    if mask.ndim == 3:
        return (mask[:, :, :3] > threshold).any(axis=2)
    return mask > threshold


def extract_boxes(binary: np.ndarray, min_box_area: int) -> list[DetectedBox]:
    """Connected components of a boolean mask, as boxes, largest first.

    A primitive. Callers labelling a frame want :func:`extract_instances`,
    because a component is a piece of an object rather than an object.
    """
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    boxes = []
    for label in range(1, count):  # label 0 is background
        x, y, w, h, area = stats[label]
        box = DetectedBox(x=int(x), y=int(y), w=int(w), h=int(h), mask_area=int(area))
        if box.box_area >= min_box_area:
            boxes.append(box)
    return sorted(boxes, key=lambda b: b.mask_area, reverse=True)


def merge_components(boxes: list[DetectedBox]) -> DetectedBox | None:
    """The union of every component, as the one object they are pieces of.

    ``mask_area`` sums the pieces rather than filling the union, so a drone
    split by an occluder keeps an honest fill ratio and stays visible to QC.
    """
    if not boxes:
        return None
    x = min(b.x for b in boxes)
    y = min(b.y for b in boxes)
    return DetectedBox(
        x=x,
        y=y,
        w=max(b.x + b.w for b in boxes) - x,
        h=max(b.y + b.h for b in boxes) - y,
        mask_area=sum(b.mask_area for b in boxes),
        components=sum(b.components for b in boxes),
    )


def extract_instances(binary: np.ndarray, min_box_area: int) -> list[DetectedBox]:
    """One box per object in frame — what a labelled frame actually asserts.

    Every foreground pixel is drone, so all of them belong to the single drone
    the scene holds and ``min_box_area`` applies to that drone rather than to
    its pieces. Filtering pieces first would be the more cautious order but is
    the wrong one: it would discard a detached blade as noise and leave the box
    short of the airframe, which is the failure it is meant to prevent.

    Returns a list so that per-drone mask values can extend it to several
    instances without changing what callers expect.
    """
    merged = merge_components(extract_boxes(binary, min_box_area=1))
    if merged is None or merged.box_area < min_box_area:
        return []
    return [merged]
